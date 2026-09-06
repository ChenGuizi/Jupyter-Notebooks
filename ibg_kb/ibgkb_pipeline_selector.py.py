"""
Pipeline Selector for IBG KB Query Processing.

This module provides a factory/selector pattern for choosing between
different pipeline implementations for processing user queries.

Pipeline Options:
1. Basic Pipeline: User query > Similarity Search > Answer Generation
2. Advanced Pipeline: User query > Query Refinement > Answer Generation > Fact Check
   (Acronym Replacement, Metadata Filtering, Document Uploading, 
    Dynamic Few-shot, Document Category Handler)
"""

from typing import List, Dict, Union, Any, Optional, Tuple
from dbskbapi.utils.exceptions import KbError
from dbskbapi.ibgkb.advanced_pipeline import AdvancedPipeline


class PipelineSelector:
    """
    Factory/Selector class for choosing the appropriate pipeline implementation.
    
    This class provides class methods to select and execute either the basic
    or advanced pipeline for processing KB queries.
    """
    
    # Available pipeline options
    PIPELINE_OPTIONS = ["basic_pipeline", "advanced_pipeline"]
    
    @classmethod
    def pipeline_selector(
        cls,
        pipeline_name: str,
        message: str,
        messages: List[Dict[str, str]],
        llm_selection: str,
        temperature: Union[int, float] = 0,
        fact_check: bool = True,
        num_docs: int = 4,
    ) -> Tuple[Union[List[Any], str], str]:
        """
        Select and execute the appropriate pipeline.
        
        Args:
            pipeline_name (str): Name of the pipeline to execute
            message (str): Original user query
            messages (List[Dict[str, str]]): Chat history
            llm_selection (str): LLM model selection
            temperature (Union[int, float]): Model temperature
            fact_check (bool): Whether to perform fact checking
            num_docs (int): Number of documents to retrieve
            
        Returns:
            Tuple[Union[List[Any], str], str]: Retrieved documents and final response
            
        Raises:
            KbError: If an invalid pipeline name is provided
        """
        if pipeline_name == "basic_pipeline":
            return cls.basic_pipeline(
                message=message,
                messages=messages,
                llm_selection=llm_selection,
                temperature=temperature,
                fact_check=fact_check,
                num_docs=num_docs,
            )
        elif pipeline_name == "advanced_pipeline":
            return cls.advanced_pipeline(
                message=message,
                messages=messages,
                llm_selection=llm_selection,
                temperature=temperature,
                fact_check=fact_check,
                num_docs=num_docs,
            )
        else:
            raise KbError(
                f"The selected pipeline isn't available. "
                f"Select from {cls.PIPELINE_OPTIONS}"
            )
    
    @classmethod
    def basic_pipeline(
        cls,
        message: str,
        messages: List[Dict[str, str]],
        llm_selection: str,
        temperature: Union[int, float] = 0,
        fact_check: bool = True,
        num_docs: int = 4,
    ) -> Tuple[Union[List[Any], str], str]:
        """
        Execute the basic pipeline.
        
        Flow: User query > Similarity Search > Answer Generation
        
        Args:
            message (str): Original user query
            messages (List[Dict[str, str]]): Chat history
            llm_selection (str): LLM model selection
            temperature (Union[int, float]): Model temperature
            fact_check (bool): Whether to perform fact checking
            num_docs (int): Number of documents to retrieve
            
        Returns:
            Tuple[Union[List[Any], str], str]: Retrieved documents and final response
        """
        # Initialize the advanced pipeline (reuses core functionality)
        llm_model = AdvancedPipeline(
            message=message,
            history=messages,
            llm_model=llm_selection,
            temperature=temperature,
        )
        
        # Basic pipeline: Query refinement is minimal/optional
        # For basic pipeline, we might want to skip query refinement
        # or use a simplified version
        
        # Get response directly with minimal processing
        docs, final_response = llm_model.chatbot_response(num_docs)
        
        # Optionally apply fact checking if requested
        if fact_check:
            docs, final_response = llm_model.fact_checker(docs, final_response)
        
        return docs, final_response
    
    @classmethod
    def advanced_pipeline(
        cls,
        message: str,
        messages: List[Dict[str, str]],
        llm_selection: str,
        temperature: Union[int, float] = 0,
        fact_check: bool = True,
        num_docs: int = 4,
    ) -> Tuple[Union[List[Any], str], str]:
        """
        Execute the advanced pipeline.
        
        Flow: User query > Query Refinement > Answer Generation > Fact Check
        Features: Acronym Replacement, Metadata Filtering, Document Uploading,
                  Dynamic Few-shot, Document Category Handler
        
        Args:
            message (str): Original user query
            messages (List[Dict[str, str]]): Chat history
            llm_selection (str): LLM model selection
            temperature (Union[int, float]): Model temperature
            fact_check (bool): Whether to perform fact checking
            num_docs (int): Number of documents to retrieve
            
        Returns:
            Tuple[Union[List[Any], str], str]: Retrieved documents and final response
        """
        # Initialize the advanced pipeline
        llm_model = AdvancedPipeline(
            message=message,
            history=messages,
            llm_model=llm_selection,
            temperature=temperature,
        )
        
        # Step 1: Query Refinement
        # Handles acronym replacement, metadata filtering, document uploading detection
        llm_model.query_refinement()
        
        # Step 2: Answer Generation
        # Retrieves documents and generates answer
        # Includes dynamic few-shot and document category handling
        docs, final_response = llm_model.chatbot_response(num_docs)
        
        # Step 3: Fact Check
        # Validates answer relevance and accuracy
        if fact_check:
            docs, final_response = llm_model.fact_checker(docs, final_response)
        
        return docs, final_response
    
    @classmethod
    def get_available_pipelines(cls) -> List[str]:
        """
        Get list of available pipeline options.
        
        Returns:
            List[str]: List of available pipeline names
        """
        return cls.PIPELINE_OPTIONS.copy()