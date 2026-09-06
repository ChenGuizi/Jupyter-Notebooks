"""
IBG KB LLM Operations Module.

This module handles all LLM-related operations including:
- Model family selection and configuration
- Query refinement
- Question answering
- Chunk relevance checking
- Fact checking
"""

import re
from typing import Any, Dict, Tuple, Union, Optional

from langchain.chains import LLMChain, StuffDocumentsChain, SequentialChain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from dbskbapi.ibgkb.prompts import (
    answer_generation,
    query_refinement,
    chunk_relevance_check,
    fact_check,
)
from dbskbapi.utils.constants import CHAT_MODEL_AVAILABLE
from ada_genai.vertexai import HarmCategory, HarmBlockThreshold


class IBGKBLLMOperations:
    """
    LLM Operations for IBG Knowledge Base.
    
    Provides static methods for different model settings and task chain initiations.
    Supports both Gemini and Claude model families.
    """
    
    @staticmethod
    def get_model_family(model_name: str, temperature: float = 0) -> Dict[str, Any]:
        """
        Get model family configuration based on model name.
        
        Args:
            model_name (str): Name of the model to use
            temperature (float): Temperature for model generation
        
        Returns:
            Dict[str, Any]: Model configuration dictionary
        """
        if "gemini" in model_name.lower():
            question_answering_dict = {
                "model_class": "gemini",
                "model_config": {
                    "model_name": model_name,
                    "temperature": float(temperature),
                    "max_output_tokens": 2048,
                    "safety_settings": {
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    }
                }
            }
        else:
            # Default to Claude for other models
            question_answering_dict = {
                "model_class": "claude",
                "model_config": {
                    "model_name": "claude-3-5-sonnet@20240620",
                    "temperature": float(temperature),
                    "max_tokens_to_sample": 2048,
                }
            }
        
        return question_answering_dict

    @staticmethod
    def query_refinement(model_family: str, task_config: Dict[str, Any]) -> LLMChain:
        """
        Create query refinement chain.
        
        Args:
            model_family (str): Model family name (e.g., 'gemini', 'claude')
            task_config (Dict[str, Any]): Model configuration
        
        Returns:
            LLMChain: Query refinement chain
        """
        # Get prompt template for the specific model family
        model_key = model_family.split('_')[0]
        template = getattr(query_refinement, f"{model_key}_qrefinement_prompt")
        
        # Create prompt
        prompt = PromptTemplate.from_template(template)
        
        # Initialize LLM
        llm = CHAT_MODEL_AVAILABLE[model_family](**task_config)
        
        # Create chain
        question_generator_chain = LLMChain(llm=llm, prompt=prompt)
        
        return question_generator_chain

    @staticmethod
    def question_answering(
        model_family: str,
        task_config: Dict[str, Any]
    ) -> Tuple[LLMChain, StuffDocumentsChain]:
        """
        Create question answering chains for both uploaded documents and vector DB.
        
        Args:
            model_family (str): Model family name (e.g., 'gemini', 'claude')
            task_config (Dict[str, Any]): Model configuration
        
        Returns:
            Tuple[LLMChain, StuffDocumentsChain]: 
                - Chain for uploaded document Q&A
                - Chain for vector DB Q&A with document stuffing
        """
        # Initialize LLM
        llm = CHAT_MODEL_AVAILABLE[model_family](**task_config)
        
        # Get model key
        model_key = model_family.split('_')[0]
        
        # ---------- Chain for Uploaded Documents ----------
        template_upload_doc = getattr(
            answer_generation,
            f"{model_key}_qna_upload_doc_prompt"
        )
        
        system_prompt_upload_doc = SystemMessagePromptTemplate.from_template(
            template_upload_doc
        )
        
        prompt_upload_doc = ChatPromptTemplate(
            messages=[
                system_prompt_upload_doc,
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )
        
        answer_upload_doc_chain = LLMChain(llm=llm, prompt=prompt_upload_doc)
        
        # ---------- Chain for Vector DB Documents ----------
        template = getattr(answer_generation, f"{model_key}_qna_prompt")
        
        system_prompt = SystemMessagePromptTemplate.from_template(template)
        
        prompt = ChatPromptTemplate(
            messages=[
                system_prompt,
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )
        
        # Create StuffDocumentsChain for combining multiple documents
        combine_docs_chain = StuffDocumentsChain(
            llm_chain=LLMChain(llm=llm, prompt=prompt),
            document_prompt=PromptTemplate(
                input_variables=["page_content"],
                template="content: {page_content}"
            ),
            document_variable_name="context",
            document_separator="\n\n",
        )
        
        return answer_upload_doc_chain, combine_docs_chain

    @staticmethod
    def chunk_relevance_check(
        model_family: str,
        task_config: Dict[str, Any]
    ) -> LLMChain:
        """
        Create chunk relevance check chain.
        
        Args:
            model_family (str): Model family name (e.g., 'gemini', 'claude')
            task_config (Dict[str, Any]): Model configuration
        
        Returns:
            LLMChain: Chunk relevance check chain
        """
        # Get prompt template for the specific model family
        model_key = model_family.split('_')[0]
        template = getattr(
            chunk_relevance_check,
            f"{model_key}_chunk_relevance_check_prompt"
        )
        
        # Create prompt
        prompt = PromptTemplate.from_template(template)
        
        # Initialize LLM
        llm = CHAT_MODEL_AVAILABLE[model_family](**task_config)
        
        # Create chain
        chunk_relevance_check_chain = LLMChain(llm=llm, prompt=prompt)
        
        return chunk_relevance_check_chain

    @staticmethod
    def fact_check(
        model_family: str,
        task_config: Dict[str, Any]
    ) -> LLMChain:
        """
        Create fact checking chain.
        
        Args:
            model_family (str): Model family name (e.g., 'gemini', 'claude')
            task_config (Dict[str, Any]): Model configuration
        
        Returns:
            LLMChain: Fact check chain
        """
        # Get prompt template for the specific model family
        model_key = model_family.split('_')[0]
        template = getattr(fact_check, f"{model_key}_factcheck_prompt")
        
        # Create prompt
        prompt = PromptTemplate.from_template(template)
        
        # Initialize LLM
        llm = CHAT_MODEL_AVAILABLE[model_family](**task_config)
        
        # Create chain
        fact_check_chain = LLMChain(llm=llm, prompt=prompt)
        
        return fact_check_chain

    @staticmethod
    def create_sequential_chain(
        chains: list,
        input_variables: list,
        output_variables: list
    ) -> SequentialChain:
        """
        Create a sequential chain from multiple chains.
        
        Args:
            chains (list): List of chains to sequence
            input_variables (list): Input variables for the first chain
            output_variables (list): Output variables from the last chain
        
        Returns:
            SequentialChain: Sequential chain
        """
        return SequentialChain(
            chains=chains,
            input_variables=input_variables,
            output_variables=output_variables,
            verbose=True,
        )

    @staticmethod
    def get_chat_model(
        model_family: str,
        task_config: Dict[str, Any]
    ) -> Any:
        """
        Get chat model instance.
        
        Args:
            model_family (str): Model family name
            task_config (Dict[str, Any]): Model configuration
        
        Returns:
            Any: Chat model instance
        """
        return CHAT_MODEL_AVAILABLE[model_family](**task_config)


# Additional helper functions for common operations

def get_model_key(model_family: str) -> str:
    """
    Extract model key from model family string.
    
    Args:
        model_family (str): Model family string (e.g., 'gemini_pro', 'claude_sonnet')
    
    Returns:
        str: Model key (e.g., 'gemini', 'claude')
    """
    return model_family.split('_')[0]


def get_prompt_template(
    module: Any,
    model_key: str,
    prompt_type: str
) -> str:
    """
    Get prompt template from module.
    
    Args:
        module: Module containing prompt templates
        model_key (str): Model key (e.g., 'gemini', 'claude')
        prompt_type (str): Type of prompt (e.g., 'qna', 'factcheck')
    
    Returns:
        str: Prompt template
    """
    prompt_name = f"{model_key}_{prompt_type}_prompt"
    return getattr(module, prompt_name)


def create_chain(
    model_family: str,
    task_config: Dict[str, Any],
    prompt_template: str,
    chain_type: str = "llm"
) -> Union[LLMChain, StuffDocumentsChain]:
    """
    Generic chain creation function.
    
    Args:
        model_family (str): Model family name
        task_config (Dict[str, Any]): Model configuration
        prompt_template (str): Prompt template
        chain_type (str): Type of chain ('llm' or 'stuff')
    
    Returns:
        Union[LLMChain, StuffDocumentsChain]: Created chain
    """
    llm = CHAT_MODEL_AVAILABLE[model_family](**task_config)
    prompt = PromptTemplate.from_template(prompt_template)
    
    if chain_type == "llm":
        return LLMChain(llm=llm, prompt=prompt)
    elif chain_type == "stuff":
        return StuffDocumentsChain(
            llm_chain=LLMChain(llm=llm, prompt=prompt),
            document_prompt=PromptTemplate(
                input_variables=["page_content"],
                template="content: {page_content}"
            ),
            document_variable_name="context",
        )
    else:
        raise ValueError(f"Unsupported chain type: {chain_type}")