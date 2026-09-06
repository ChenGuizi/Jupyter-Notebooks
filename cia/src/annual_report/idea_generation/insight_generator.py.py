"""
Insight Generator Module for IBG Knowledge Base.

This module handles the generation of insights from company annual reports,
including trigger-based analysis, document retrieval, and fact checking.
"""

import os
import re
import json
import uuid
import asyncio
import logging
import string
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple, Any, Dict, Optional, Union

from langchain.docstore.document import Document
from langchain.callbacks.base import BaseCallbackHandler

from .data_retriever import DataRetriever
from .utils import (
    APIChatAnthropic,
    MpnetEmbedder,
    ChainCreator,
    connect_to_elasticsearch,
    remove_tag_and_content,
    extract_content_by_tag,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class CustomHandler(BaseCallbackHandler):
    """
    Custom callback handler for logging LLM prompts and responses.
    """
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """
        Handle LLM start event by logging prompts to file.
        
        Args:
            serialized (Dict[str, Any]): Serialized LLM configuration
            prompts (List[str]): List of prompts sent to LLM
            **kwargs: Additional keyword arguments
        """
        try:
            formatted_prompts = "\n".join(prompts)
            current_date = datetime.today().strftime("%Y%m%d")
            
            # Create intermediate output directory if it doesn't exist
            output_dir = os.path.join(
                os.getcwd(),
                "intermediate_output"
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Write prompt to file for debugging
            file_path = os.path.join(
                output_dir,
                f"{current_date}_{uuid.uuid4()}_insight_generator_prompt.txt"
            )
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Prompt:\n{formatted_prompts}")
                
        except Exception as e:
            logging.warning(f"Failed to log prompt: {e}")


class InsightGenerator:
    """
    A class that handles the logic of insight generation.
    
    Features:
    - Trigger-based insight generation
    - Document retrieval and relevance checking
    - Fact checking and validation
    - Batch processing for large datasets
    """
    
    def __init__(self, config: dict):
        """
        Initialize the InsightGenerator instance.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.chunk_metadata: Dict[str, Any] = {}  # Store the chunk metadata
        self.llm = APIChatAnthropic(config=self.config)
        
        # Initialize embedder
        embedding_config = self.config.get("text_embedder", {})
        mpnet_config = embedding_config.get("mpnet_api", {})
        
        self.embedder = MpnetEmbedder(
            endpoint=mpnet_config.get("endpoint"),
            api_key=mpnet_config.get("api_key"),
            app_code=embedding_config.get("app_code"),
            project_name=embedding_config.get("project_name"),
        )
        
        logging.info("InsightGenerator initialized successfully")
    
    @staticmethod
    def generate_company_trigger_pairs(
        gcin_list: List[str],
        trigger_prompt_list: List[dict]
    ) -> List[Dict[str, dict]]:
        """
        Generate all possible company-trigger pairs.
        
        Args:
            gcin_list (List[str]): List of company GCINs
            trigger_prompt_list (List[dict]): List of trigger prompt dictionaries
        
        Returns:
            List[Dict[str, dict]]: List of dictionaries with company-trigger pairs
        """
        company_trigger_pairs_list = []
        
        for gcin in gcin_list:
            for trigger_dict in trigger_prompt_list:
                company_trigger_pairs_list.append({gcin: trigger_dict})
        
        return company_trigger_pairs_list
    
    async def async_get_chunk_checked_response(
        self,
        query: str,
        chunk_dict: Dict[int, Document]
    ) -> Dict[str, Dict[int, Document]]:
        """
        Asynchronously get chunk relevance checked response.
        
        Args:
            query (str): Query string
            chunk_dict (Dict[int, Document]): Dictionary of chunk documents
        
        Returns:
            Dict[str, Dict[int, Document]]: Ranked chunks (high, medium, low)
        """
        checked_chunk_dict: Dict[str, Dict[int, Document]] = {
            "high": {},
            "medium": {},
            "low": {}
        }
        
        try:
            # Get the first item from chunk_dict
            key, value = list(chunk_dict.items())[0]
            
            # Get retrieval sharper chain
            retrieval_sharper_chain = ChainCreator.get_retrieval_sharper_chain(
                model_class=self.config["insight_generator"]["retrieval_sharper_chain"]["model_class"],
                llm=self.llm,
            )
            
            # Invoke chain
            args_dict = {
                "question": query,
                "context": value.page_content
            }
            
            retrieval_sharper_response = await retrieval_sharper_chain.ainvoke(args_dict)
            
            # Parse response to determine relevance
            relevance_score = retrieval_sharper_response.strip().split("\n")[0].lower()
            
            if "high" in relevance_score:
                checked_chunk_dict["high"][key] = value
            elif "medium" in relevance_score:
                checked_chunk_dict["medium"][key] = value
            else:
                checked_chunk_dict["low"][key] = value
                
        except Exception as e:
            logging.error(f"Error in async_get_chunk_checked_response: {e}")
            checked_chunk_dict["low"] = chunk_dict
        
        return checked_chunk_dict
    
    async def async_get_all_chunk_checked_response(
        self,
        query: str,
        chunk_list: List[Dict[int, Document]]
    ) -> List[Document]:
        """
        Asynchronously check each chunk against user query and rank relevance.
        
        Args:
            query (str): Query string
            chunk_list (List[Dict[int, Document]]): List of chunk dictionaries
        
        Returns:
            List[Document]: List of filtered and ranked chunks
        """
        if not chunk_list:
            logging.warning("Empty chunk list provided")
            return []
        
        # Process chunks in parallel
        checked_chunk_list = await asyncio.gather(
            *[
                self.async_get_chunk_checked_response(
                    query=query,
                    chunk_dict=chunk_dict
                )
                for chunk_dict in chunk_list
            ]
        )
        
        if not checked_chunk_list:
            return []
        
        # Aggregate results
        checked_chunk_dict = {"high": {}, "medium": {}, "low": {}}
        
        for chunk_dict in checked_chunk_list:
            for key in checked_chunk_dict.keys():
                checked_chunk_dict[key].update(chunk_dict.get(key, {}))
        
        # Sort and return ranked chunks
        high_rank_chunk_list = [
            value for key, value in sorted(checked_chunk_dict["high"].items())
        ]
        medium_rank_chunk_list = [
            value for key, value in sorted(checked_chunk_dict["medium"].items())
        ]
        
        # Check criteria for including medium-ranked chunks
        criteria = self.config["insight_generator"]["retrieval_sharper_chain"].get("criteria", {})
        
        if criteria.get("include_medium", False):
            return high_rank_chunk_list + medium_rank_chunk_list
        else:
            return high_rank_chunk_list
    
    async def async_generate_company_trigger_insight(
        self,
        gcin: str,
        trigger_dict: dict
    ) -> dict:
        """
        Asynchronously create insight for the trigger associated with the company.
        
        Args:
            gcin (str): The GCIN of the company
            trigger_dict (dict): Translated trigger prompt
        
        Returns:
            dict: Company insight dictionary following JSON schema
        
        Raises:
            Exception: If no relevant chunks found or trigger is 0
        """
        try:
            logging.info(f"Generating insight for GCIN: {gcin}")
            
            # Get response with relevant context
            args_dict = {
                "gcin": gcin,
                "query": trigger_dict["prompt"]
            }
            
            relevant_context, response = await self.async_get_response(**args_dict)
            
            logging.info(f"Response received for GCIN {gcin}")
            
            if not relevant_context:
                raise Exception("No relevant chunks found!")
            
            if "0" in trigger_dict.get("trigger_code", ""):
                raise Exception("Trigger is 0!")
            
            # Parse response for answer and thinking
            match_answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            answer = match_answer.group(1).strip() if match_answer else ""
            
            match_thinking = re.search(r"<thinking>(.*?)</thinking>", response, re.DOTALL)
            thinking = match_thinking.group(1).strip() if match_thinking else ""
            
            match_page = re.search(r"<page>(.*?)</page>", response, re.DOTALL)
            page_str = match_page.group(1).strip() if match_page else ""
            page_list = [int(page) for page in page_str.split(",")] if page_str else []
            
            # Build insight dictionary
            company_insight_dict = {
                "genki_id": gcin,
                "industry": self.chunk_metadata.get("industry", ""),
                "source_type": self.config["insight_generator"]["source_type"],
                "file_id": self.chunk_metadata.get("file_id", ""),
                "chunks": relevant_context,
                "page_number": sorted(page_list),
                "ideas_generated": answer,
                "published_date": self.chunk_metadata.get("issue_date", ""),
                "trigger_hit": 1,
                "thinking_process": thinking,
            }
            
            return company_insight_dict
            
        except Exception as e:
            logging.error(f"Error generating insight for GCIN {gcin}: {e}")
            
            # Return fallback insight
            return {
                "genki_id": gcin,
                "industry": self.chunk_metadata.get("industry", ""),
                "source_type": self.config["insight_generator"]["source_type"],
                "file_id": self.chunk_metadata.get("file_id", ""),
                "page_number": [1],
                "not_found_msg": "No relevant chunks found",
                "ideas_generated": self.config["insight_generator"]["ideas_generated_fallback"],
                "published_date": self.chunk_metadata.get("issue_date", ""),
                "trigger_hit": 0,
                "thinking_process": "",
            }
    
    def generate_query_dsl(
        self,
        gcin: str,
        filtered_doc_dict: Dict[str, set]
    ) -> dict:
        """
        Generate Elasticsearch query DSL for document retrieval.
        
        Args:
            gcin (str): GCIN of the company
            filtered_doc_dict (Dict[str, set]): Dictionary of filtered documents
        
        Returns:
            dict: Elasticsearch query DSL
        """
        doc_list = []
        
        for issue_date, page_numbers in filtered_doc_dict.items():
            for page_number in page_numbers:
                doc_list.append({
                    "bool": {
                        "must": [
                            {"match_phrase": {"metadata.gcin.keyword": gcin}},
                            {"match_phrase": {"metadata.issue_date": issue_date}},
                            {"match_phrase": {"metadata.page_number": page_number}}
                        ],
                        "should": [],
                        "must_not": []
                    }
                })
        
        query = {
            "bool": {
                "should": doc_list,
                "minimum_should_match": 1
            }
        }
        
        return query
    
    async def async_get_response(
        self,
        gcin: str,
        query: str
    ) -> Tuple[str, str]:
        """
        Generate answer with context from retrieved documents.
        
        Args:
            gcin (str): GCIN of the company
            query (str): Query for the trigger
        
        Returns:
            Tuple[str, str]: Most relevant context and final response
        """
        try:
            # Initialize data retriever
            data_retriever = DataRetriever(
                query=query,
                config=self.config,
                gcin=gcin
            )
            
            # Retrieve documents with retry
            docs = []
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                docs = data_retriever.retrieve_from_elasticsearch(k=10)
                
                if docs:
                    break
                    
                logging.warning(f"Nothing retrieved from ELK! Retry {retry_count + 1}/{max_retries}")
                retry_count += 1
                await asyncio.sleep(1)
            
            if not docs:
                logging.error(f"No documents retrieved for GCIN {gcin}")
                return "", ""
            
            # Store chunk metadata
            if not self.chunk_metadata:
                self.chunk_metadata = docs[0].metadata
            
            # Organize documents by issue date and page number
            filtered_doc_dict: Dict[str, set] = {}
            
            for doc in docs:
                issue_date = doc.metadata.get("issue_date", "")
                page_number = doc.metadata.get("page_number", "")
                
                if issue_date and page_number:
                    if issue_date in filtered_doc_dict:
                        filtered_doc_dict[issue_date].add(page_number)
                    else:
                        filtered_doc_dict[issue_date] = {page_number}
            
            # Build query DSL and get all relevant chunks
            dsl_query = self.generate_query_dsl(
                gcin=gcin,
                filtered_doc_dict=filtered_doc_dict
            )
            
            hits = data_retriever.search_elk(
                query=dsl_query,
                size=100  # Hardcoded value
            )
            
            # Organize chunks by file, page, and chunk index
            text_dict = defaultdict(lambda: defaultdict(dict))
            table_dict = defaultdict(lambda: defaultdict(dict))
            
            for hit in hits.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                metadata = source.get("metadata", {})
                
                file_name = metadata.get("file_name", "")
                page_number = metadata.get("page_number", "")
                data_type = metadata.get("data_type", "")
                chunk_index = metadata.get("chunk_index", 0)
                text = source.get("text", "")
                
                if data_type == "text":
                    text_dict[file_name][page_number][chunk_index] = text
                elif data_type == "table":
                    table_dict[file_name][page_number][chunk_index] = text
            
            # Convert to regular dictionaries
            text_dict = json.loads(json.dumps(text_dict))
            table_dict = json.loads(json.dumps(table_dict))
            
            # Reorder dictionaries by page number
            text_dict_reorder = {}
            table_dict_reorder = {}
            
            for key in text_dict:
                text_dict_reorder[key] = {
                    int(k): text_dict[key][k]
                    for k in sorted(text_dict[key].keys(), key=int)
                }
            
            for key in table_dict:
                table_dict_reorder[key] = {
                    int(k): table_dict[key][k]
                    for k in sorted(table_dict[key].keys(), key=int)
                }
            
            # Build context and relevant page list
            context_doc_list = []
            relevant_page_list = []
            
            # Process text documents
            for doc in text_dict_reorder:
                for page in text_dict_reorder[doc]:
                    page_number = f"Page Number: {page}\n\n"
                    
                    # Extract summary
                    try:
                        text_str = text_dict_reorder[doc][page]
                        summary = extract_content_by_tag(html_string=text_str, tag='summary')[0]
                    except Exception:
                        text_str = text_dict_reorder[doc][page]
                        cleaned_text = remove_tag_and_content(html_string=text_str, tag="summary")
                        summary = ""
                    
                    # Check for tables on this page
                    table_str = ""
                    if doc in table_dict_reorder and page in table_dict_reorder[doc]:
                        sorted_keys = sorted(table_dict_reorder[doc][page].keys())
                        table_str = "\n".join(
                            table_dict_reorder[doc][page][key] for key in sorted_keys
                        )
                        table_dict_reorder[doc].pop(page)
                    
                    # Add to context
                    context_doc_list.append(page_number + cleaned_text + table_str + summary)
                    
                    relevant_page_list.append(
                        Document(
                            page_content=cleaned_text + table_str,
                            metadata={"page_number": page}
                        )
                    )
            
            # Process remaining table-only documents
            for doc in table_dict_reorder:
                for page in table_dict_reorder[doc]:
                    sorted_keys = sorted(table_dict_reorder[doc][page].keys())
                    table_str = "\n".join(
                        table_dict_reorder[doc][page][key] for key in sorted_keys
                    )
                    
                    page_number = f"Page Number: {page}\n\n"
                    context_doc_list.append(page_number + table_str)
                    
                    relevant_page_list.append(
                        Document(
                            page_content=table_str,
                            metadata={"page_number": page}
                        )
                    )
            
            # Prepare context
            context_text = "\n---\n\n".join(context_doc_list)
            
            # Generate answer
            current_date = datetime.today().strftime('%b %Y')
            
            answer_generator_chain = ChainCreator.get_answer_generator_chain(
                model_class=self.config["insight_generator"]["answer_generator_chain"]["model_class"],
                llm=self.llm,
            )
            
            response = await answer_generator_chain.ainvoke(
                {
                    "context": context_text,
                    "curr_date": current_date,
                    "question": query,
                },
                config={"callbacks": [CustomHandler()]}
            )
            
            response = response.strip()
            
            # Extract relevant chunks for response
            chunks = []
            for doc in relevant_page_list:
                if int(doc.metadata["page_number"]) in filtered_doc_dict:
                    chunks.append(doc.page_content)
            
            relevant_context = "\n\n".join(chunks)
            
            return relevant_context, response
            
        except Exception as e:
            logging.error(f"Error in async_get_response: {e}")
            return "", ""
    
    def check_response(
        self,
        context: str,
        query: str,
        response: str
    ) -> bool:
        """
        Check the facts of the response.
        
        Args:
            context (str): Retrieved context
            query (str): Original query
            response (str): Generated response
        
        Returns:
            bool: True if response passes fact checking, False otherwise
        """
        try:
            # Rule-based checking
            keywords_list = self.config["insight_generator"]["fact_checker_chain"]["rule_base_keyword_list"]
            
            if not response.strip() or any(
                word in response.strip().lower() for word in keywords_list
            ):
                return False
            
            # LLM-based checking
            fact_checker_chain = ChainCreator.get_fact_checker_chain(
                model_class=self.config["insight_generator"]["fact_checker_chain"]["model_class"],
                llm=self.llm,
            )
            
            fact_checker_response = fact_checker_chain.invoke(
                {
                    "context": context,
                    "question": query,
                    "response": response
                }
            ).strip()
            
            logging.info(f"Fact checker response: {fact_checker_response}")
            
            # Parse score from response
            pattern = r"(Score|Response|score|response)\s*(\w+)"
            matches = re.findall(
                pattern,
                "".join(
                    char for char in fact_checker_response.strip()
                    if char not in string.punctuation
                )
            )
            
            score = ""
            for match in matches:
                label, value = match
                score += value
            
            if "low" in score.lower():
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error in check_response: {e}")
            return False
    
    def generate_insight(
        self,
        gcin: str,
        trigger_codes: Dict[str, str]
    ) -> List[dict]:
        """
        Generate insights and output the result.
        
        Args:
            gcin (str): GCIN of the company
            trigger_codes (Dict[str, str]): Trigger codes and product codes
        
        Returns:
            List[dict]: All necessary fields for different triggers
        """
        try:
            # Build trigger prompt list
            trigger_prompt_list = []
            trigger_catalogue = self.config.get("trigger_catalogue", {})
            
            for code in trigger_codes:
                try:
                    trigger_info = trigger_catalogue.get(code, {}).copy()
                    trigger_info["trigger_code"] = code
                    trigger_prompt_list.append(trigger_info)
                except Exception as e:
                    logging.warning(f"Trigger {code} is not in the config! Error: {e}")
            
            # Generate company-trigger pairs
            company_trigger_pairs_list = InsightGenerator.generate_company_trigger_pairs(
                gcin_list=[gcin],
                trigger_prompt_list=trigger_prompt_list
            )
            
            # Process in batches
            batch_size = self.config["insight_generator"].get("batch_size", 5)
            
            # Generate insights asynchronously
            company_insight_pair_list = asyncio.run(
                self.async_generate_company_insights(
                    company_trigger_pairs_list=company_trigger_pairs_list,
                    batch_size=batch_size
                )
            )
            
            return company_insight_pair_list
            
        except Exception as e:
            logging.error(f"Error in generate_insight: {e}")
            return []
    
    async def async_generate_company_insights(
        self,
        company_trigger_pairs_list: List[Dict[str, dict]],
        batch_size: int = 5
    ) -> List[dict]:
        """
        Asynchronously generate insights in batches.
        
        Args:
            company_trigger_pairs_list (List[Dict[str, dict]]): List of company-trigger pairs
            batch_size (int): Batch size for processing
        
        Returns:
            List[dict]: List of generated insights
        """
        all_insights = []
        
        for i in range(0, len(company_trigger_pairs_list), batch_size):
            batch = company_trigger_pairs_list[i:i + batch_size]
            
            # Process batch
            batch_insights = await asyncio.gather(
                *[
                    self.async_generate_company_trigger_insight(
                        gcin=gcin,
                        trigger_dict=trigger_dict
                    )
                    for item in batch
                    for gcin, trigger_dict in item.items()
                ]
            )
            
            all_insights.extend(batch_insights)
            
            # Add delay between batches to avoid rate limiting
            if i + batch_size < len(company_trigger_pairs_list):
                await asyncio.sleep(1)
        
        return all_insights