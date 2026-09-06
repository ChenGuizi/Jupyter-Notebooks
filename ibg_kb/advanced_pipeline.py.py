"""
Advanced Pipeline for IBG Knowledge Base.

This module handles the complete pipeline for processing user queries,
including acronym handling, document retrieval, dynamic few-shot learning,
and response generation with fact checking.
"""

import re
import os
import asyncio
import string
import logging as logger
from typing import List, Tuple, Union, Any, Dict

import pandas as pd
from langchain.docstore.document import Document

from dbskbapi.ibgkb.ibgkb_llm_operations import IBGKBLLMOperations
from dbskbapi.utils.constants import IBGKB_USER_CONFIG, AUTHOR_CHOICES
from dbskbapi.utils.all_utils import (
    preprocess,
    connect_to_elkdb,
    update_temperature,
    read_ibg_acronym,
    get_acronym_dict,
    count_replacement,
    replace_acronyms,
    remove_qa_unnecessary_characters,
    generate_select_acronym_template,
    clean_chat_history,
    load_disclaimer_message,
)
from ada_genai.langchain import VertexAIEmbeddings
from dbskbapi.utils.ss_utils import connect_to_s3


class AdvancedPipeline:
    """
    Advanced Pipeline for IBG Knowledge Base.
    
    Handles the complete query processing pipeline including:
    - Acronym detection and replacement
    - Query refinement
    - Document retrieval from Elasticsearch
    - Dynamic few-shot learning
    - Response generation with fact checking
    """
    
    # Class variables
    _history_length = 6  # Take care of the separation of query and chat history in ibgkb
    _acronym_threshold = 3  # Acronyms are not replaced if the number in query exceeds this
    _max_num_chunks = 10  # Max number of chunks retrieved from ELK
    _qn_examples_num_chunks = 10  # Retrieve top n similar dynamic few-shot Q&A pairs
    _qn_examples_similarity_threshold = 0.9  # Threshold for similarity score
    _systems_application_threshold = 1  # If at least top n documents fall under Systems & Application
    
    _scenarios_require_fact_checker_list = [
        "reply_ibgkb_general",
        "search_upload"
    ]  # QA scenarios which require fact check
    
    _non_exist_msg_dict = {
        "reply_ibgkb_general": "Sorry, the relevant documents are not in the existing IBG knowledge base.",
        "search_upload": "Sorry, the answer cannot be found in the uploaded documents.",
    }  # Messages shown if documents cannot be found
    
    _acronym_selection_msg_list = [
        'Please select the full form of',
        "Unsupported acronym selection format, please ask the same query again!"
    ]  # Messages prompted to user for acronym selection

    def __init__(
        self,
        message: str,
        history: List[Dict[str, str]],
        llm_model: str,
        temperature: Union[None, int, float] = 0
    ):
        """
        Initialize the AdvancedPipeline instance.
        
        Args:
            message (str): Original user query
            history (List[Dict[str, str]]): List of chat history with 'content' and 'author'
            llm_model (str): Model name for question answering
            temperature (Union[None, int, float]): Model temperature
        """
        self.USER_RAG_CONFIG = IBGKB_USER_CONFIG
        self.message = message.strip()
        self.disclaimer_message = load_disclaimer_message()
        
        # Clean chat history
        print(f"History before clean: {history}")
        self.history = clean_chat_history(
            chat_history=history,
            non_exist_msg_dict=AdvancedPipeline._non_exist_msg_dict,
            disclainer_msg=self.disclaimer_message,
            acronym_selection_msg_list=AdvancedPipeline._acronym_selection_msg_list,
            history_length=AdvancedPipeline._history_length,
        )
        print(f"History after clean: {self.history}")
        
        self.messages, self.convo_string = preprocess(history)
        
        # Initialize instance variables
        self.if_use_dfs = False  # Whether dynamic few-shot is used
        self.refined_query = ""  # Store refined query
        self.retrieved_docs = ""  # Store retrieved chunk content
        self.qa_router = ""  # Route to different QA scenarios
        self.checked_chunk_list = []  # Store chunks with high relevance
        
        # Acronym handling
        self.acronym_df = read_ibg_acronym()
        self.acronym_dict = get_acronym_dict(self.acronym_df)
        
        # Question answering setup
        question_answering_dict = IBGKBLLMOperations.get_model_family(llm_model, temperature)
        self.USER_RAG_CONFIG["question_answering"] = question_answering_dict
        
        # S3 connector
        self.s3_connector = connect_to_s3(
            key=os.getenv("AWS_KEY"),
            secret=os.getenv("AWS_SECRET_KEY"),
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        )
        
        logger.info(f"[IBGKBLOGS]Question and Answering Model: {question_answering_dict}")

    def acronym_logic_handler(self) -> Union[None, str]:
        """
        Handle acronym related logics.
        
        1. Check if one-to-many acronym exists in query
        2. Check if user is making selection of the full form
        
        Returns:
            Union[None, str]: None or QA scenario
        """
        try:
            # Check previous bot response for acronym selection
            if_make_acronym_selection = re.findall(
                r"A(.+)\n\nPlease select the full form of (.+) from ([1-9]+) options",
                self.history[-1]["content"].strip(),
            )
        except Exception as e:
            if "list index out of range" in str(e):
                if_make_acronym_selection = None
            else:
                logger.error(f"Exception in AdvancedPipeline.acronym_logic_handler: {e}")
                if_make_acronym_selection = None
        
        # Handle acronym selection from user
        if if_make_acronym_selection:
            query, acronym, total_num_full_form = if_make_acronym_selection[0]
            total_num_full_form = int(total_num_full_form)
            
            try:
                selected_num = int(self.message.strip())
                
                # Check if user input is valid
                if selected_num <= total_num_full_form:
                    logger.info(f"[IBGKBLOGS]Original query: {self.message}")
                    select_dict = {acronym: [self.acronym_dict[acronym][selected_num - 1]]}
                    self.message = replace_acronyms(self.message, select_dict, [acronym])
                    logger.info(f"[IBGKBLOGS]Replaced query: {self.message}")
                else:
                    return "reinput_query"
            except Exception as e:
                logger.error(f"Exception in AdvancedPipeline.acronym_logic_handler: {e}")
                return "reinput_query"
        
        # Count acronyms in query
        num_matches, self.matches_list = count_replacement(self.message, self.acronym_dict)
        self.max_num_full_form = max(
            [len(value) for value in [self.acronym_dict[match] for match in self.matches_list]]
        )
        
        if num_matches > 0 and num_matches <= AdvancedPipeline._acronym_threshold:
            logger.info(f"[IBGKBLOGS]{num_matches} acronyms in user query")
            
            if self.max_num_full_form == 1:
                logger.info(f"[IBGKBLOGS]Original query: {self.message}")
                self.message = replace_acronyms(self.message, self.acronym_dict, self.matches_list)
                logger.info(f"[IBGKBLOGS]Replaced query: {self.message}")
            elif self.max_num_full_form > 1:
                return "select_acronym"
            else:
                logger.error(
                    "Exception in AdvancedPipeline.acronym_logic_handler: "
                    "Full form of acronym is not found in the table!"
                )
                raise Exception("Full form of acronym is not found in the table!")
        
        return None

    def upload_logic_handler(self) -> Union[None, str]:
        """
        Handle file uploading related logics.
        
        1. Check if user is uploading the file
        2. Check if user is asking query related to the uploaded files
        
        Returns:
            Union[None, str]: None or QA scenario
        """
        # Check if user uploads file
        match_if_upload = re.findall(
            r"A.+\.(txt|doc|docx|pdf)",
            self.message.strip().split("\n")[0].lower()
        )
        
        # Search through chat history for uploaded documents
        match_if_search_list = [
            re.findall(r"A.+\.(txt|doc|docx|pdf)", chat_dict["content"].lower())
            for chat_dict in self.history
        ]
        
        if match_if_upload:
            return "do_upload"
        elif any(match for match in match_if_search_list):
            return "search_upload"
        return None

    def dynamic_few_shot_logic_handler(self) -> Tuple[str, list]:
        """
        Handle dynamic few-shot related logics.
        
        1. Retrieve question examples from vector database based on the query
        2. Determine if similarity score is above the preset threshold
        
        Returns:
            Tuple[str, list]: Retrieved Q&A examples and source information
        """
        def semantic_build_query_qn_examples(query_body: dict, query: str) -> dict:
            """
            Custom query to be used in Elasticsearch for dynamic few-shot.
            
            Args:
                query_body (dict): Elasticsearch query body
                query (str): Query string
            
            Returns:
                dict: Elasticsearch query body
            """
            query_template = {
                "knn": {
                    "field": "vector",
                    "query_vector": VertexAIEmbeddings(
                        model_name=self.USER_RAG_CONFIG["db_embedding"]["model_name"]
                    ).embed_query(query),
                    "k": AdvancedPipeline._qn_examples_num_chunks,
                    "num_candidates": 100,
                },
                "sort": [{"_score": {"order": "desc"}}]
            }
            return query_template

        # Read dynamic few-shot examples from S3
        with self.s3_connector.Open(os.getenv("IBG_QN_EXAMPLE_TABLE_PATH")) as f:
            metadata_df = pd.read_csv(f, encoding="ISO-8859-1")
        
        # Connect to vector database
        qn_examples_vectorstore, _ = connect_to_elkdb(
            index_name=os.getenv("IBG_QN_EXAMPLE_COLLECTION_NAME"),
            es_user=os.getenv("ELASTICSEARCH_USER"),
            es_password=os.getenv("ELASTICSEARCH_PASSWORD"),
            embedding_name=self.USER_RAG_CONFIG["db_embedding"]["model_name"],
        )
        
        # Create retriever
        qn_examples_retriever = qn_examples_vectorstore.as_retriever(
            search_kwargs={
                "k": AdvancedPipeline._qn_examples_num_chunks,
                "custom_query": semantic_build_query_qn_examples,
            }
        )
        
        # Retrieve relevant documents
        qn_examples_docs = qn_examples_retriever.get_relevant_documents(
            self.refined_query
        )
        
        # Process retrieved examples
        retrieved_qn_examples = ""
        retrieved_qn_examples_source_list = []
        
        for idx, doc in enumerate(qn_examples_docs):
            # Get similarity score
            qn_examples_score = qn_examples_vectorstore.similarity_search_with_score(
                self.refined_query,
                k=AdvancedPipeline._qn_examples_num_chunks,
            )
            
            logger.info(f"[IBGKBLOGS]{doc.page_content}: {qn_examples_score[idx][1]}")
            
            # Check if similarity exceeds threshold
            if qn_examples_score[idx][1] >= AdvancedPipeline._qn_examples_similarity_threshold:
                # Get metadata
                answer = metadata_df.loc[
                    metadata_df["index"] == doc.metadata["index"], "answer"
                ].tolist()[0]
                
                hyperlink = metadata_df.loc[
                    metadata_df["index"] == doc.metadata["index"], "hyperlink"
                ].tolist()[0]
                
                document_name = metadata_df.loc[
                    metadata_df["index"] == doc.metadata["index"], "document_name"
                ].tolist()[0]
                
                page = metadata_df.loc[
                    metadata_df["index"] == doc.metadata["index"], "page"
                ].tolist()[0]
                
                category = metadata_df.loc[
                    metadata_df["index"] == doc.metadata["index"], "category"
                ].tolist()[0]
                
                if_cell_empty = metadata_df[
                    metadata_df["index"] == doc.metadata["index"]
                ].reset_index(inplace=False, drop=True)
                
                # Add to retrieved examples
                retrieved_qn_examples += (
                    f"{idx+1}. Question: {doc.page_content} \nAnswer: {answer}"
                )
                
                # Add to source list if hyperlink exists
                if not pd.isna(if_cell_empty["hyperlink"])[0]:
                    logger.info(
                        f"[IBGKBLOGS]Question Index without Empty Cell: {doc.metadata['index']}"
                    )
                    retrieved_qn_examples_source_list.append(
                        Document(
                            page_content=answer,
                            metadata={
                                'document_name': document_name,
                                'document_name_match': " ".join(
                                    re.sub(r"[^a-zA-Z0-9\s]", "", document_name)
                                    .lower()
                                    .strip()
                                    .split()
                                ),
                                'hyperlink': f"{hyperlink}#page={int(page)}",
                                'page': int(page),
                                'category': category,
                                'date_of_issue': "000000",
                                'date_of_issue_match': "000000",
                                'date_of_ingestion': "000000",
                                'data_type': "dynamic few-shot",
                                'batch_number': 0,
                            }
                        )
                    )
                else:
                    break
        
        logger.info(f"[IBGKBLOGS]Dynamic Few-shot Examples: \n{retrieved_qn_examples}")
        return retrieved_qn_examples, retrieved_qn_examples_source_list

    # OFF FOR FUTURE RELEASE
    def document_category_logic_handler(
        self, es_client: Any, docs: List[Document]
    ) -> str:
        """
        Handle document category related logics.
        
        1. Check if query related to systems application category
           1.1 If top n docs are from same document under system application
           1.2 If top n docs are multimodal chunks under system application
           1.3 If dynamic few-shot is used under system application
        
        Args:
            es_client (Any): Elasticsearch client instance
            docs (List[Document]): List of retrieved documents
        
        Returns:
            str: QA scenario related to docs category
        """
        is_multimodal_counter = 0
        document_name_counter = {}
        
        for i, document in enumerate(docs):
            if i >= AdvancedPipeline._systems_application_threshold:
                break
            
            category = document.metadata["category"]
            is_multimodal = document.metadata.get("is_multimodal", "false")
            document_name_match = document.metadata["document_name_match"]
            
            if "Systems & Application" in category:
                if i == 0 and self.if_use_dfs:
                    print("System Application - Dynamic Few-shot")
                    return "reply_systems_application"
                
                if is_multimodal == "true":
                    is_multimodal_counter += 1
                
                if len(document_name_counter) == 0:
                    document_name_counter[document_name_match] = 1
                elif document_name_match in document_name_counter:
                    document_name_counter[document_name_match] += 1
                
                if is_multimodal_counter >= AdvancedPipeline._systems_application_threshold:
                    print("System Application - Multimodal")
                    return "reply_systems_application"
                elif max(document_name_counter.values()) >= AdvancedPipeline._systems_application_threshold:
                    print("System Application - Doc Name")
                    return "reply_systems_application"
                
                print("System Application - Definition")
        
        return "reply_ibgkb_general"  # Scenarios which are not category specific

    @classmethod
    def update_max_num_chunks(cls, num_docs: int) -> None:
        """
        Update _max_num_chunks if num_docs in the payload is bigger than _max_num_chunks.
        
        Args:
            num_docs (int): Number of documents requested
        """
        cls._max_num_chunks = num_docs

    def chunk_relevance_checker(self, chunk_list: List[Document]) -> None:
        """
        Check each chunk against user query, only keep those that are relevant.
        
        Args:
            chunk_list (List[Document]): List of document chunks to check
        """
        chunk_relevance_check_chain = IBGKBLLMOperations.chunk_relevance_check(
            self.USER_RAG_CONFIG["chunk_relevance_check"]["model_class"],
            self.USER_RAG_CONFIG["chunk_relevance_check"]["model_config"],
        )
        
        for chunk in chunk_list:
            chunk_relevance_check_response = chunk_relevance_check_chain.run(
                question=self.refined_query,
                context=chunk.page_content,
            )
            
            logger.info(f"[IBGKBLOGS]Chunk content:\n{chunk.page_content}")
            logger.info(f"[IBGKBLOGS]Chunk hyperlink: \n{chunk.metadata['hyperlink']}")
            logger.info(
                f"[IBGKBLOGS]Chunk relevance checked answer: "
                f"\n{chunk_relevance_check_response.strip()}"
            )
            
            if "low" not in chunk_relevance_check_response.lower().split("\n")[0]:
                self.checked_chunk_list.append(chunk)

    async def async_get_chunk_checked_response(self, chunk: Document) -> None:
        """
        Asynchronously get chunk relevance checked response.
        
        Args:
            chunk (Document): Document chunk to check
        """
        chunk_relevance_check_chain = IBGKBLLMOperations.chunk_relevance_check(
            self.USER_RAG_CONFIG["chunk_relevance_check"]["model_class"],
            self.USER_RAG_CONFIG["chunk_relevance_check"]["model_config"],
        )
        
        args_dict = {
            "question": self.refined_query,
            "context": chunk.page_content
        }
        
        chunk_relevance_check_response = await asyncio.to_thread(
            chunk_relevance_check_chain.run,
            **args_dict
        )
        
        logger.info(f"[IBGKBLOGS]Chunk content:\n{chunk.page_content}")
        logger.info(f"[IBGKBLOGS]Chunk hyperlink: \n{chunk.metadata['hyperlink']}")
        logger.info(
            f"[IBGKBLOGS]Chunk relevance checked answer: "
            f"\n{chunk_relevance_check_response.strip()}"
        )
        
        if "low" not in chunk_relevance_check_response.lower().split("\n")[0]:
            self.checked_chunk_list.append(chunk)

    async def async_chunk_relevance_checker(self, chunk_list: List[Document]) -> None:
        """
        Asynchronously check each chunk against user query, only keep those that are relevant.
        
        Args:
            chunk_list (List[Document]): List of document chunks to check
        """
        await asyncio.gather(
            *[self.async_get_chunk_checked_response(chunk) for chunk in chunk_list]
        )

    def query_refinement(self) -> None:
        """
        Query refinement to properly rephrase the user query based on chat history.
        """
        # Handle acronym logic
        qa_router_acronym = self.acronym_logic_handler()
        qa_router_upload = self.upload_logic_handler()
        
        if qa_router_acronym:
            self.refined_query = self.message.strip()
            self.qa_router = qa_router_acronym
            return None  # Bypass query refinement
        
        if qa_router_upload:
            initial_question = self.message.strip()
            self.qa_router = qa_router_upload
        else:
            initial_question = self.message
            # Refine query using LLM
            question_generator_chain = IBGKBLLMOperations.query_refinement(
                self.USER_RAG_CONFIG["query_refinement"]["model_class"],
                self.USER_RAG_CONFIG["query_refinement"]["model_config"],
            )
            
            try:
                agent_response = question_generator_chain.run(
                    question=initial_question,
                    chat_history=self.convo_string,
                )
                
                # Extract answer from agent response
                pattern = r"<answer>\s*(.*?)\s*</answer>"
                refined_query = re.findall(pattern, agent_response)
                
                if len(refined_query) == 1:
                    refined_query = refined_query[0].strip()
                else:
                    refined_query = ""
            except Exception as e:
                logger.error(f"Exception in AdvancedPipeline.query_refinement: {e}")
                refined_query = initial_question
        
        # Rule-based keywords checking
        keywords_list = [
            "cannot answer",
            "not able to answer",
            "unable to answer",
            "does not mention",
        ]
        
        # Check if refined query is empty or contains negative keywords
        if len(refined_query) == 0 or any(
            word in refined_query.lower() for word in keywords_list
        ):
            self.refined_query = initial_question
            logger.info("[IBGKBLOGS]Refined query is empty, no refinement is required")
        else:
            self.refined_query = refined_query
            logger.info(f"[IBGKBLOGS]Refined query: {self.refined_query}")
        
        return None

    def chatbot_response(self, num_docs: int) -> Tuple[Union[List[Document], str], str]:
        """
        Generate chatbot response.
        
        Args:
            num_docs (int): Number of document chunks to extract
        
        Returns:
            Tuple[Union[List[Document], str], str]: Documents retrieved and final response
        """
        # Load QA chains
        answer_upload_doc_chain, combine_docs_chain = IBGKBLLMOperations.question_answering(
            self.USER_RAG_CONFIG["question_answering"]["model_class"],
            self.USER_RAG_CONFIG["question_answering"]["model_config"],
        )
        
        # QA scenarios
        if self.qa_router == "select_acronym":
            # Scenario 1: User selects full form from options
            for index, bool_value in enumerate(
                [len(self.acronym_dict[key]) > 1 for key in self.matches_list]
            ):
                if bool_value:
                    match_str = self.matches_list[index]
                    break
            
            response = generate_select_acronym_template(
                self.refined_query, self.acronym_dict, match_str
            )
            return [], response
        
        elif self.qa_router == "reinput_query":
            # Scenario 2: User inputs wrong format
            return [], "Unsupported acronym selection format, please ask the same query again!"
        
        elif self.qa_router == "do_upload":
            # Scenario 3: User uploads document
            return [], "File contents received."
        
        elif self.qa_router == "search_upload":
            # Scenario 4: User asks questions about uploaded docs
            logger.info(
                f"[IBGKBLOGS]Check Answer Generation using this chain: {answer_upload_doc_chain}"
            )
            
            upload_doc_history_context = ""
            for chat_dict in self.history:
                match_if_upload = re.findall(
                    r"A.+\.(txt|doc|docx|pdf)", chat_dict["content"]
                )
                if match_if_upload and chat_dict["author"] == AUTHOR_CHOICES.USER.value:
                    upload_doc_history_context += f"{chat_dict['content'].strip()}\n"
            
            logger.info(
                f"[IBGKBLOGS]Uploaded document's content: {upload_doc_history_context}"
            )
            
            response = answer_upload_doc_chain.run(
                question=self.refined_query,
                context=upload_doc_history_context,
            )
            response = remove_qa_unnecessary_characters(response)
            logger.info(f"[IBGKBLOGS]Initial answer: \n{response}")
            return [], response
        
        else:
            # Scenario 5: User asks questions about docs in vector DB
            self.qa_router = "reply_ibgkb_general"
            
            # Update max chunks if needed
            if num_docs > AdvancedPipeline._max_num_chunks:
                AdvancedPipeline.update_max_num_chunks(num_docs)
            
            logger.info(f"[IBGKBLOGS]Max num chunks: {AdvancedPipeline._max_num_chunks}")
            
            # Build custom query
            def semantic_build_query(query_body: dict, query: str) -> dict:
                """Custom query for Elasticsearch retrieval."""
                query_template = {
                    "knn": {
                        "field": "vector",
                        "query_vector": VertexAIEmbeddings(
                            model_name=self.USER_RAG_CONFIG["db_embedding"]["model_name"]
                        ).embed_query(query),
                        "k": AdvancedPipeline._max_num_chunks,
                        "num_candidates": 100,
                    },
                    "sort": [{"_score": {"order": "desc"}}]
                }
                return query_template
            
            # Connect to vector database
            ibg_vectorstore, ibg_es_client = connect_to_elkdb(
                index_name=os.getenv("IBG_ELK_COLLECTION_NAME"),
                es_user=os.getenv("ELASTICSEARCH_USER"),
                es_password=os.getenv("ELASTICSEARCH_PASSWORD"),
                embedding_name=self.USER_RAG_CONFIG["db_embedding"]["model_name"],
            )
            
            # Create retriever
            doc_retriever = ibg_vectorstore.as_retriever(
                search_kwargs={
                    "k": AdvancedPipeline._max_num_chunks,
                    "custom_query": semantic_build_query,
                }
            )
            
            # Step 1: Retrieve top chunks from ELK
            docs = doc_retriever.get_relevant_documents(self.refined_query)
            logger.info(f"[IBGKBLOGS]Initial chunks retrieved: {docs}")
            logger.info(f"[IBGKBLOGS]Initial number of chunks retrieved: {len(docs)}")
            
            # Step 2: Check relevance of chunks (OFF FOR FUTURE RELEASE)
            # self.chunk_relevance_checker(docs)
            # asyncio.run(self.async_chunk_relevance_checker(docs))
            # docs = self.checked_chunk_list
            # logger.info(f"[IBGKBLOGS]Number of relevance checked chunks: {len(docs)}")
            
            # Step 3: Retrieve dynamic few-shot examples
            retrieved_qn_examples, retrieved_qn_examples_source_list = (
                self.dynamic_few_shot_logic_handler()
            )
            
            if retrieved_qn_examples_source_list:
                self.if_use_dfs = True
            
            # Step 4: Merge source metadata from dynamic few-shot
            if retrieved_qn_examples_source_list:
                retrieved_qn_examples_source_list.extend(docs)
                docs = retrieved_qn_examples_source_list
            
            # Step 5: Return if no documents
            if len(docs) == 0:
                return [], "No relevant documents found."
            
            # Step 6: Use top num_docs for Q&A
            if len(docs) > num_docs:
                docs = docs[:num_docs]
            
            logger.info(f"[IBGKBLOGS]Number of chunks for QnA: {len(docs)}")
            
            # Step 7: Prepare retrieved docs for context
            self.retrieved_docs = ""
            for idx, doc in enumerate(docs):
                if idx == 0 and retrieved_qn_examples:
                    self.retrieved_docs += f"\n\n{retrieved_qn_examples}"
                
                self.retrieved_docs += (
                    f"\n\nDocument Title: {doc.metadata['document_name']} "
                    f"\n{doc.page_content}"
                )
            
            # Generate response
            response = combine_docs_chain.run(
                input_documents=docs,
                chat_history=self.messages,
                question=self.refined_query,
                question_examples=retrieved_qn_examples,
            )
            response = remove_qa_unnecessary_characters(response)
            logger.info(f"[IBGKBLOGS]Initial answer:\n{response}")
            
            # Step 8: Remove duplicate documents
            duplicate_check = {}
            docs_non_duplicate_list = []
            for document in docs:
                joining_key = str(document.metadata["hyperlink"]).strip()
                if joining_key not in duplicate_check:
                    duplicate_check[joining_key] = True
                    docs_non_duplicate_list.append(document)
            
            # Further zoom into category related solutions (OFF FOR FUTURE RELEASE)
            # self.qa_router = self.document_category_logic_handler(ibg_es_client, docs)
            
            if self.qa_router == "reply_systems_application":
                return docs_non_duplicate_list, "Please refer to the following documents."
            
            return docs_non_duplicate_list, response.strip()

    def fact_checker(
        self, docs: Union[List[Document], str], response: str
    ) -> Tuple[Union[List[Document], str], str]:
        """
        Check for answer relevance and accuracy; ensure proper in-text citation.
        
        Args:
            docs (Union[List[Document], str]): Documents retrieved
            response (str): Answer generated
        
        Returns:
            Tuple[Union[List[Document], str], str]: Documents retrieved and final response
        """
        logger.info(f"[IBGKBLOGS]QA router: {self.qa_router}")
        
        # No chunk retrieved is relevant to the query
        if isinstance(docs, list) and len(docs) == 0:
            return "", AdvancedPipeline._non_exist_msg_dict.get(self.qa_router, "")
        
        # Check if fact checking is required
        if self.qa_router in AdvancedPipeline._scenarios_require_fact_checker_list:
            # Step 1: Rule-based keywords checking
            keywords_list = [
                "not found",
                "sorry",
                "unfortunately",
                "unfortunate",
                "apologize",
                "apology",
                "could not find",
                "cannot find",
                "unable to find",
                "not able to find",
            ]
            
            if (not response.strip()) or any(
                word in response.strip().lower() for word in keywords_list
            ):
                return "", AdvancedPipeline._non_exist_msg_dict.get(self.qa_router, "")
            
            # Step 2: LLM-based fact checking
            fact_check_chain = IBGKBLLMOperations.fact_check(
                self.USER_RAG_CONFIG["fact_check"]["model_class"],
                self.USER_RAG_CONFIG["fact_check"]["model_config"],
            )
            
            fact_check_response = fact_check_chain.run(
                question=self.refined_query,
                response=response,
                context=self.retrieved_docs,
            )
            
            logger.info(f"[IBGKBLOGS]Fact-checked answer:\n{fact_check_response.strip()}")
            
            # Parse fact check score
            pattern = r"(Score|Response|score|response)\s*(\w+)"
            matches = re.findall(
                pattern,
                "".join(
                    char for char in fact_check_response.strip()
                    if char not in string.punctuation
                )
            )
            
            score = ""
            for match in matches:
                label, value = match
                score += value
            
            if "low" in score.lower():
                return "", AdvancedPipeline._non_exist_msg_dict.get(self.qa_router, "")
            
            # Add disclaimer message
            logger.info(f"[IBGKBLOGS]Disclaimer message:\n{self.disclaimer_message}")
            response += self.disclaimer_message
            return docs, response
        
        return [], response