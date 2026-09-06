"""
Pipeline Selector for IBG KB Query Processing.

This module provides separate implementations for basic and advanced pipelines.
"""

from typing import List, Dict, Union, Any, Optional, Tuple
from dbskbapi.utils.exceptions import KbError
from dbskbapi.ibgkb.advanced_pipeline import AdvancedPipeline


class BasicPipeline:
    """
    Basic Pipeline for KB Query Processing.
    
    Flow: User query > Similarity Search > Answer Generation
    """
    
    def __init__(
        self,
        message: str,
        messages: List[Dict[str, str]],
        llm_selection: str,
        temperature: Union[int, float] = 0,
    ):
        """
        Initialize the Basic Pipeline.
        
        Args:
            message (str): Original user query
            messages (List[Dict[str, str]]): Chat history
            llm_selection (str): LLM model selection
            temperature (Union[int, float]): Model temperature
        """
        self.message = message
        self.messages = messages
        self.llm_selection = llm_selection
        self.temperature = temperature
        
        # Reuse AdvancedPipeline for core functionality
        self.pipeline = AdvancedPipeline(
            message=message,
            history=messages,
            llm_model=llm_selection,
            temperature=temperature,
        )
    
    def process(
        self,
        fact_check: bool = True,
        num_docs: int = 4,
    ) -> Tuple[Union[List[Any], str], str]:
        """
        Process query through basic pipeline.
        
        Args:
            fact_check (bool): Whether to perform fact checking
            num_docs (int): Number of documents to retrieve
            
        Returns:
            Tuple[Union[List[Any], str], str]: Retrieved documents and final response
        """
        # Skip query refinement for basic pipeline
        # Get response with minimal processing
        docs, final_response = self.pipeline.chatbot_response(num_docs)
        
        # Optionally apply fact checking
        if fact_check:
            docs, final_response = self.pipeline.fact_checker(docs, final_response)
        
        return docs, final_response


class AdvancedPipelineWrapper:
    """
    Advanced Pipeline for KB Query Processing.
    
    Flow: User query > Query Refinement > Answer Generation > Fact Check
    Features: Acronym Replacement, Metadata Filtering, Document Uploading,
              Dynamic Few-shot, Document Category Handler
    """
    
    def __init__(
        self,
        message: str,
        messages: List[Dict[str, str]],
        llm_selection: str,
        temperature: Union[int, float] = 0,
    ):
        """
        Initialize the Advanced Pipeline.
        
        Args:
            message (str): Original user query
            messages (List[Dict[str, str]]): Chat history
            llm_selection (str): LLM model selection
            temperature (Union[int, float]): Model temperature
        """
        self.message = message
        self.messages = messages
        self.llm_selection = llm_selection
        self.temperature = temperature
        
        # Initialize the actual advanced pipeline
        self.pipeline = AdvancedPipeline(
            message=message,
            history=messages,
            llm_model=llm_selection,
            temperature=temperature,
        )
    
    def process(
        self,
        fact_check: bool = True,
        num_docs: int = 4,
    ) -> Tuple[Union[List[Any], str], str]:
        """
        Process query through advanced pipeline.
        
        Args:
            fact_check (bool): Whether to perform fact checking
            num_docs (int): Number of documents to retrieve
            
        Returns:
            Tuple[Union[List[Any], str], str]: Retrieved documents and final response
        """
        # Step 1: Query Refinement
        self.pipeline.query_refinement()
        
        # Step 2: Answer Generation
        docs, final_response = self.pipeline.chatbot_response(num_docs)
        
        # Step 3: Fact Check
        if fact_check:
            docs, final_response = self.pipeline.fact_checker(docs, final_response)
        
        return docs, final_response


class PipelineSelector:
    """
    Factory/Selector class for choosing the appropriate pipeline implementation.
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
            pipeline = BasicPipeline(
                message=message,
                messages=messages,
                llm_selection=llm_selection,
                temperature=temperature,
            )
            return pipeline.process(fact_check=fact_check, num_docs=num_docs)
        
        elif pipeline_name == "advanced_pipeline":
            pipeline = AdvancedPipelineWrapper(
                message=message,
                messages=messages,
                llm_selection=llm_selection,
                temperature=temperature,
            )
            return pipeline.process(fact_check=fact_check, num_docs=num_docs)
        
        else:
            raise KbError(
                f"The selected pipeline isn't available. "
                f"Select from {cls.PIPELINE_OPTIONS}"
            )
    
    @classmethod
    def get_available_pipelines(cls) -> List[str]:
        """
        Get list of available pipeline options.
        
        Returns:
            List[str]: List of available pipeline names
        """
        return cls.PIPELINE_OPTIONS.copy()