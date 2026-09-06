"""
Data Ingestion Module for IBG Knowledge Base.

This module handles parallel data ingestion from PDF files into Elasticsearch,
including document preprocessing, chunking, embedding generation, and indexing.
"""

import re
import os
import json
import math
import shutil
import tempfile
import asyncio
import logging
import multiprocessing
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

import pymupdf4llm
import pikepdf
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .index_mapping import index_mapping
from .utils import (
    MpnetEmbedder,
    APIChatAnthropicChainCreator,
    connect_to_elasticsearch,
    count_logical_cpu,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ParallelDataIngestor:
    """
    A class that handles data ingestion in parallel using multiprocessing.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the ParallelDataIngestor instance.
        
        Args:
            config (dict): Configuration dictionary containing all settings
        """
        self.config: dict = config
        self.summary_dict: dict = {}
        self.file_segments: List[List[dict]] = []
    
    def delete_by_query(self, query: dict) -> None:
        """
        Delete chunks by query.
        
        Args:
            query (dict): Conditions for deletion
        """
        data_ingestor = DataIngestor(config=self.config)
        data_ingestor.delete_by_query(query=query)
    
    def get_file_segment(self, file_path: str, metadata_dict: dict) -> List[List[dict]]:
        """
        Get all the file segments in one list.
        
        Args:
            file_path (str): Path to the PDF file
            metadata_dict (dict): Metadata for the file
        
        Returns:
            List[List[dict]]: List of file segments
        """
        data_ingestor = DataIngestor(config=self.config)
        self.file_segments = data_ingestor.split_file_by_worker_count(
            file_path=file_path,
            metadata_dict=metadata_dict
        )
        self.summary_dict = data_ingestor.summary_dict
        return self.file_segments
    
    def ingest_data(self, file_segment: List[dict], metadata_dict: dict) -> None:
        """
        Ingest each file segment.
        
        Args:
            file_segment (List[dict]): File segment to ingest
            metadata_dict (dict): Metadata for the file
        """
        data_ingestor = DataIngestor(config=self.config)
        data_ingestor.summary_dict = self.summary_dict
        data_ingestor.preprocess_and_ingest_file(
            file_segment=file_segment,
            metadata_dict=metadata_dict
        )
    
    def invoke(self, file_path: str, metadata_dict: dict) -> None:
        """
        Start ingesting the data into ELK.
        
        Args:
            file_path (str): Path to the PDF file
            metadata_dict (dict): Metadata for the file including:
                - gcin: str
                - file_type: str
                - issue_date: int
                - source_type: str
                - file_name: str
                - page_number: int
                - page_count: int
                - ingestion_date: int
                - data_type: str
                - chunk_index: int
        
        Raises:
            Exception: If any error occurs during ingestion
        """
        try:
            # Create multiprocessing pool
            pool = multiprocessing.Pool(
                processes=min(
                    count_logical_cpu(),
                    self.config["data_ingestor"]["worker_num"]
                )
            )
            
            results = []
            
            # Get file segments and process in parallel
            for file_segment in self.get_file_segment(
                file_path=file_path,
                metadata_dict=metadata_dict
            ):
                result = pool.apply_async(
                    self.ingest_data,
                    args=(file_segment, metadata_dict)
                )
                results.append(result)
            
            # Close pool and wait for completion
            pool.close()
            pool.join()
            
            # Check for any exceptions
            for result in results:
                result.get()
                
        except Exception as e:
            raise Exception(f"Exception raised in ParallelDataIngestor.invoke(): {e}")


class DataIngestor:
    """
    A class that handles data ingestion for a single file or file segment.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the DataIngestor instance.
        
        Args:
            config (dict): Configuration dictionary containing all settings
        """
        self.config: dict = config
        self.file_segment: List[List[dict]] = []
        self.summary_dict: Dict[int, str] = {}  # page_number: summary
        
        # Worker configuration
        self.worker_count: int = min(
            count_logical_cpu(),
            self.config["data_ingestor"]["worker_num"]
        )
        
        # Initialize embedder
        self.embedder = MpnetEmbedder(
            endpoint=self.config["text_embedder"]["mpnet_api"]["endpoint"],
            api_key=self.config["text_embedder"]["mpnet_api"]["api_key"],
            app_code=self.config["text_embedder"]["app_code"],
            project_name=self.config["text_embedder"]["project_name"],
        )
        
        # Connect to Elasticsearch
        self.es_instance, self.es_store_instance = connect_to_elasticsearch(
            hosts=self.config["data_ingestor"]["elasticsearch"]["hosts"],
            basic_auth=(
                self.config["data_ingestor"]["elasticsearch"]["basic_auth"]["username"],
                self.config["data_ingestor"]["elasticsearch"]["basic_auth"]["password"],
            ),
            verify_certs=self.config["data_ingestor"]["elasticsearch"]["verify_certs"],
            index_name=self.config["data_ingestor"]["index_name"],
            embedder=self.embedder
        )
        
        # Initialize LLM
        self.llm = APIChatAnthropicChainCreator(config=self.config)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\u200B",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
            ],
            chunk_size=self.config["data_ingestor"]["chunk_size"],
            chunk_overlap=self.config["data_ingestor"]["chunk_overlap"],
            length_function=lambda x: MpnetEmbedder.count_tokens(x),
        )
        
        # Initialize table splitter
        self.table_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"],
            chunk_size=self.config["data_ingestor"]["chunk_size"],
            chunk_overlap=self.config["data_ingestor"]["chunk_overlap"],
            length_function=lambda x: MpnetEmbedder.count_tokens(x),
        )
        
        # Check if index exists and is empty
        if self.is_index_empty():
            logging.warning(
                f"{self.config['data_ingestor']['index_name']} does not exist in ELK!"
            )
    
    def is_index_empty(self) -> bool:
        """
        Check if the index is empty.
        
        Returns:
            bool: True if index is empty or doesn't exist, False otherwise
        """
        try:
            exists = self.es_instance.indices.exists(
                index=self.config["data_ingestor"]["index_name"]
            )
            return not exists
        except Exception as e:
            logging.info(f"Exception raised in DataIngestor.is_index_empty(): {e}")
            return False
    
    def concatenate_page(self, doc_list: List[Document]) -> List[str]:
        """
        Concatenate content of pages into string based on summarizer batch size.
        
        Args:
            doc_list (List[Document]): List of pages
        
        Returns:
            List[str]: Concatenated page content strings
        """
        batch_size = self.config["data_ingestor"]["summarizer_batch_size"]
        concatenated_pages = []
        
        for i in range(0, len(doc_list), batch_size):
            chunk = []
            for doc in doc_list[i:i + batch_size]:
                chunk.append(
                    f"page_number: {int(doc.metadata['page']) + 1}\n"
                    f"page_content:\n{doc.page_content.strip()}"
                )
            concatenated_pages.append("\n\n---\n\n".join(chunk))
        
        return concatenated_pages
    
    def reset_index_mapping(self, index_mapping: dict) -> None:
        """
        Reset the index mapping.
        
        Args:
            index_mapping (dict): Index mapping for the index
        """
        try:
            self.es_instance.indices.create(
                body=index_mapping,
                index=self.config["data_ingestor"]["index_name"]
            )
        except Exception as e:
            logging.info(f"Exception raised in DataIngestor.reset_index_mapping(): {e}")
    
    def delete_by_query(self, query: dict) -> None:
        """
        Delete chunks by query.
        
        Args:
            query (dict): Conditions for deletion
        
        Raises:
            Exception: If deletion fails
        """
        try:
            delete_result = self.es_instance.delete_by_query(
                index=self.config["data_ingestor"]["index_name"],
                body={"query": query}
            )
            logging.info(f"Deletion outcome: {delete_result}")
        except Exception as e:
            raise Exception(f"Exception raised in DataIngestor.delete_by_query(): {e}")
    
    @staticmethod
    def get_text(text: str) -> str:
        """
        Extract non-table text from markdown content.
        
        Args:
            text (str): Markdown text
        
        Returns:
            str: Non-table text content
        """
        pattern = r"^(?!\n).+"
        non_table_lines = re.findall(pattern, text, re.MULTILINE)
        non_table_lines = [line.strip() for line in non_table_lines if line.strip()]
        return "\n\n".join(non_table_lines)
    
    @staticmethod
    def get_table(text: str) -> str:
        """
        Extract table content from markdown text.
        
        Args:
            text (str): Markdown text
        
        Returns:
            str: Table content
        """
        pattern = r"(\n.+\n((\-{3,})+\n(.+\n?)+\n))"
        matches = re.findall(pattern, text)
        table_list = []
        for match in matches:
            table_list.append(match[0].strip())
        return "\n\n".join(table_list)
    
    @staticmethod
    def is_all_non_alnum_or_empty(s: str) -> bool:
        """
        Return True if the string is empty or contains only non-alphanumeric characters.
        
        Args:
            s (str): Input string
        
        Returns:
            bool: True if empty or only non-alphanumeric
        """
        return not s or not any(c.isalnum() for c in s)
    
    @staticmethod
    def get_page_content(file_path: str) -> Tuple[Optional[List[Document]], Optional[str]]:
        """
        Get page level content in string format.
        
        Args:
            file_path (str): Path to the PDF file
        
        Returns:
            Tuple[Optional[List[Document]], Optional[str]]: 
                - List of page documents or None if error
                - Error message or None if successful
        """
        try:
            loader = PyMuPDFLoader(file_path)
            texts = loader.load()
            return texts, None
        except Exception as e:
            if "endstream" in str(e):
                try:
                    # Fix PDF using pikepdf
                    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                        tmp_path = tmp_file.name
                        pdf = pikepdf.open(file_path)
                        pdf.save(tmp_path)
                        shutil.move(tmp_path, file_path)
                        logging.info(f"Fixed document using pikepdf: {file_path}")
                    return DataIngestor.get_page_content(file_path)
                except Exception as fix_error:
                    return None, str(fix_error)
            return None, str(e)
    
    async def aget_page_summary(
        self,
        content: str,
        content_context: str = "part of the company's annual report."
    ) -> dict:
        """
        Asynchronously get page level summary in 2-3 sentences.
        
        Args:
            content (str): Content for summarization
            content_context (str): The context of the content
        
        Returns:
            dict: Summary of the page with page_number as key and summary as value
        """
        chunk_summarizer_chain = self.llm.get_chunk_summarizer_chain()
        
        args_dict = {
            "content_context": content_context,
            "content": content.strip(),
        }
        
        response = await chunk_summarizer_chain.ainvoke(args_dict)
        
        if response.strip():
            response_dict = json.loads(response.strip())
            return {
                entry["page_number"]: entry["summary"]
                for entry in response_dict["summaries"]
            }
        else:
            return {}
    
    async def aget_all_page_summary(
        self,
        concatenated_pages: List[str],
        content_context: str = "part of the company's annual report."
    ) -> List[Dict[int, str]]:
        """
        Asynchronously get all page level summaries.
        
        Args:
            concatenated_pages (List[str]): List of concatenated pages
            content_context (str): The context of the content
        
        Returns:
            List[Dict[int, str]]: List of summaries
        """
        tasks = [
            self.aget_page_summary(
                content=content,
                content_context=content_context
            )
            for content in concatenated_pages
        ]
        
        summary_list = await asyncio.gather(*tasks)
        return summary_list
    
    def is_file_ingested(self, gcin: str, issue_date: int) -> bool:
        """
        Check if the latest file was ingested. (Off due to IOW-18917)
        
        Args:
            gcin (str): GCIN of the company
            issue_date (int): Issue date of annual report in int format
        
        Returns:
            bool: True if file is already ingested, False otherwise
        """
        try:
            query = {
                "bool": {
                    "must": [],
                    "filter": [
                        {"match_phrase": {"metadata.gcin.keyword": gcin}},
                        {"range": {"metadata.issue_date": {"gte": int(issue_date)}}}
                    ],
                    "should": [],
                    "must_not": []
                }
            }
            
            hits = self.es_instance.search(
                index=self.config["data_ingestor"]["index_name"],
                query=query
            )["hits"]["total"]["value"]
            
            return hits > 0
        except Exception as e:
            logging.info(f"Exception raised in DataIngestor.is_file_ingested(): {e}")
            return False
    
    def split_file_by_worker_count(
        self,
        file_path: str,
        metadata_dict: dict
    ) -> List[List[dict]]:
        """
        Split the file by number of workers for the multiprocessing task.
        
        Args:
            file_path (str): Path to the PDF file
            metadata_dict (dict): Metadata for the file
        
        Returns:
            List[List[dict]]: List of file segments
        """
        # Check if file is already ingested
        if self.is_file_ingested(
            gcin=metadata_dict["gcin"],
            issue_date=metadata_dict["issue_date"]
        ):
            logging.info(
                f"The more recent file for GCIN {metadata_dict['gcin']}, "
                f"compared to the current file, has been ingested."
            )
            return []
        
        # Convert PDF to markdown with page chunks
        try:
            page_markdown_list = pymupdf4llm.to_markdown(
                doc=file_path,
                page_chunks=True,  # Convert to list of page-level markdowns
                graphics_limit=100,  # Limit dealing with excess vector graphics
            )
        except Exception as e:
            if "endstream" in str(e):
                # Fix PDF using pikepdf
                with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                    tmp_path = tmp_file.name
                    pdf = pikepdf.open(file_path)
                    pdf.save(tmp_path)
                    shutil.move(tmp_path, file_path)
                    logging.info(f"Fixed document using pikepdf: {file_path}")
                    return self.split_file_by_worker_count(
                        file_path=file_path,
                        metadata_dict=metadata_dict
                    )
            else:
                raise Exception(
                    f"Exception raised in DataIngestor.split_file_by_worker_count: {e}"
                )
        
        # Get page content
        doc_list, _ = DataIngestor.get_page_content(file_path=file_path)
        
        # Concatenate pages for summarization
        concatenated_pages = self.concatenate_page(doc_list=doc_list)
        
        # Get page summaries
        summary_list = asyncio.run(
            self.aget_all_page_summary(concatenated_pages=concatenated_pages)
        )
        
        # Update summary dictionary
        page_summary = {
            key: value 
            for d in summary_list 
            for key, value in d.items()
        }
        self.summary_dict.update(page_summary)
        
        # Split pages into segments for workers
        total_pages = len(page_markdown_list)
        base_size = total_pages // self.worker_count
        remainder = total_pages % self.worker_count
        
        segments = []
        start_index = 0
        
        for i in range(self.worker_count):
            # Calculate end index for current part
            extra = 1 if i < remainder else 0
            end_index = start_index + base_size + extra
            
            segment = page_markdown_list[start_index:end_index]
            if segment:  # Only add non-empty segments
                segments.append(segment)
            
            start_index = end_index
        
        return segments
    
    def preprocess_and_ingest_file(
        self,
        file_segment: List[dict],
        metadata_dict: dict
    ) -> None:
        """
        Preprocess and ingest a file segment into Elasticsearch.
        
        Args:
            file_segment (List[dict]): File segment to process
            metadata_dict (dict): Metadata for the file
        
        Raises:
            Exception: If any error occurs during preprocessing or ingestion
        """
        try:
            document_list = []
            batch_size = self.config["data_ingestor"]["batch_size"]
            
            # Base metadata for all chunks
            base_metadata = {
                "gcin": metadata_dict.get("gcin"),
                "issue_date": metadata_dict.get("issue_date"),
                "file_type": metadata_dict.get("file_type"),
                "source_type": metadata_dict.get("source_type"),
                "file_name": metadata_dict.get("file_name"),
                "page_count": metadata_dict.get("page_count"),
                "ingestion_date": metadata_dict.get("ingestion_date"),
            }
            
            for page_markdown in file_segment:
                page_number = page_markdown.get("page", 1)
                
                # Page level metadata
                page_level_metadata = {
                    "page_number": page_number,
                    "summary": self.summary_dict.get(page_number, ""),
                }
                page_level_metadata.update(base_metadata)
                
                # Extract text and tables from markdown
                text_str = DataIngestor.get_table(page_markdown["text"])
                
                # Process tables
                table_list = []
                table_doc = Document(
                    page_content=text_str,
                    metadata={
                        "data_type": "table",
                        "chunk_index": 1,
                        **page_level_metadata
                    }
                )
                
                if DataIngestor.is_all_non_alnum_or_empty(text_str):
                    # Skip empty table
                    pass
                elif MpnetEmbedder.count_tokens(text_str) > self.config["data_ingestor"]["chunk_size"]:
                    # Split large tables
                    table_splits = self.table_splitter.split_documents([table_doc])
                    for index, table_split in enumerate(table_splits):
                        table_split.metadata["chunk_index"] = index + 1
                        table_list.append(table_split)
                else:
                    table_list.append(table_doc)
                
                # Process text
                text_list = []
                text_str = page_markdown["text"].strip()
                is_text_valid = DataIngestor.is_all_non_alnum_or_empty(text_str)
                
                if is_text_valid:
                    # Empty text, create placeholder
                    doc = Document(
                        page_content="",
                        metadata={
                            "data_type": "text",
                            "chunk_index": 1,
                            **page_level_metadata
                        }
                    )
                    text_list.append(doc)
                else:
                    doc = Document(
                        page_content=text_str,
                        metadata={
                            "data_type": "text",
                            "chunk_index": 1,
                            **page_level_metadata
                        }
                    )
                    
                    if MpnetEmbedder.count_tokens(text_str) > self.config["data_ingestor"]["chunk_size"]:
                        # Split large text
                        splits = self.text_splitter.split_documents([doc])
                        for index, split in enumerate(splits):
                            split.metadata["chunk_index"] = index + 1
                            if not DataIngestor.is_all_non_alnum_or_empty(split.page_content):
                                split.page_content += f"\n\nSummary: {page_level_metadata['summary']}"
                            text_list.append(split)
                    else:
                        if not DataIngestor.is_all_non_alnum_or_empty(doc.page_content):
                            doc.page_content += f"\n\nSummary: {page_level_metadata['summary']}"
                        text_list.append(doc)
                
                # Combine text and table documents
                for doc in itertools.chain(text_list, table_list):
                    if doc.page_content.strip():
                        document_list.append(doc)
            
            # Ingest documents in batches
            for i in tqdm(
                range(0, len(document_list), batch_size),
                total=math.ceil(len(document_list) / batch_size),
                desc=f"[PID-{os.getpid()}-Ingestion]"
            ):
                start = i
                end = min(i + batch_size, len(document_list))
                batch = document_list[start:end]
                self.es_store_instance.add_documents(batch)
                
        except Exception as e:
            raise Exception(
                f"Exception raised in DataIngestor.preprocess_and_ingest_file() "
                f"PID-{os.getpid()}: {e}"
            )