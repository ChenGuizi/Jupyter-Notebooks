"""
Chain Creator Module for IBG Knowledge Base.

This module provides a factory class for creating various LLM chains
used throughout the IBG Knowledge Base system, including:
- Answer generation
- Chunk reranking
- Retrieval sharpening
- Fact checking
- Summarization
- Report generation
- Card fact checking
"""

from typing import Optional, Type, Any, Dict, Union
from langchain_core.runnables import RunnableSequence
from langchain_core.language_models.chat_models import BaseChatModel

from .all_utils import create_llm_chain
from .prompt import (
    answer_generator,
    chunk_reranker,
    retrieval_sharper,
    fact_checker,
    chunk_summarizer,
    keypoints_summarizer,
    report_generator,
    card_fact_checker,
)


class ChainCreator:
    """
    A factory class that initiates different LLM chains.
    
    Provides static methods for creating various chain types with
    consistent configuration and prompt templates.
    """
    
    @staticmethod
    def get_chunk_reranker_chain(
        model_class: str,
        llm: BaseChatModel
    ) -> RunnableSequence:
        """
        Create chunk reranker chain.
        
        Args:
            model_class (str): Class of the model (e.g., 'gemini', 'claude')
            llm (BaseChatModel): Instance of LLM
        
        Returns:
            RunnableSequence: Chunk reranker chain
        """
        template = getattr(
            chunk_reranker,
            f"{model_class}_chunk_reranker_prompt"
        )
        chunk_reranker_chain = create_llm_chain(llm=llm, template=template)
        return chunk_reranker_chain
    
    @staticmethod
    def get_answer_generator_chain(
        model_class: str,
        llm: BaseChatModel
    ) -> RunnableSequence:
        """
        Create answer generator chain.
        
        Args:
            model_class (str): Class of the model (e.g., 'gemini', 'claude')
            llm (BaseChatModel): Instance of LLM
        
        Returns:
            RunnableSequence: Answer generator chain
        """
        template = getattr(
            answer_generator,
            f"{model_class}_answer_generator_prompt"
        )
        answer_generator_chain = create_llm_chain(llm=llm, template=template)
        return answer_generator_chain
    
    @staticmethod
    def get_retrieval_sharper_chain(
        model_class: str,
        llm: BaseChatModel
    ) -> RunnableSequence:
        """
        Create retrieval sharper chain.
        
        Args:
            model_class (str): Class of the model (e.g., 'gemini', 'claude')
            llm (BaseChatModel): Instance of LLM
        
        Returns:
            RunnableSequence: Retrieval sharper chain
        """
        template = getattr(
            retrieval_sharper,
            f"{model_class}_retrieval_sharper_prompt"
        )
        retrieval_sharper_chain = create_llm_chain(llm=llm, template=template)
        return retrieval_sharper_chain
    
    @staticmethod
    def get_fact_checker_chain(
        model_class: str,
        llm: BaseChatModel
    ) -> RunnableSequence:
        """
        Create fact checker chain.
        
        Args:
            model_class (str): Class of the model (e.g., 'gemini', 'claude')
            llm (BaseChatModel): Instance of LLM
        
        Returns:
            RunnableSequence: Fact checker chain
        """
        template = getattr(
            fact_checker,
            f"{model_class}_fact_checker_prompt"
        )
        fact_checker_chain = create_llm_chain(llm=llm, template=template)
        return fact_checker_chain
    
    @staticmethod
    def get_chunk_summarizer_chain(
        model_class: str,
        llm: BaseChatModel
    ) -> RunnableSequence:
        """
        Create chunk summarizer chain.
        
        Args:
            model_class (str): Class of the model (e.g., 'gemini', 'claude')
            llm (BaseChatModel): Instance of LLM
        
        Returns:
            RunnableSequence: Chunk summarizer chain
        """
        template = getattr(
            chunk_summarizer,
            f"{model_class}_chunk_summarizer_prompt"
        )
        chunk_summarizer_chain = create_llm_chain(llm=llm, template=template)
        return chunk_summarizer_chain
    
    @staticmethod
    def get_keypoints_summarizer_chain(
        model_class: str,
        llm: BaseChatModel
    ) -> RunnableSequence:
        """
        Create keypoints summarizer chain.
        
        Args:
            model_class (str): Class of the model (e.g., 'gemini', 'claude')
            llm (BaseChatModel): Instance of LLM
        
        Returns:
            RunnableSequence: Keypoints summarizer chain
        """
        template = getattr(
            keypoints_summarizer,
            f"{model_class}_keypoints_summarizer_prompt"
        )
        keypoints_summarizer_chain = create_llm_chain(llm=llm, template=template)
        return keypoints_summarizer_chain
    
    @staticmethod
    def get_report_generator_chain(
        model_class: str,
        llm: BaseChatModel
    ) -> RunnableSequence:
        """
        Create report generator chain.
        
        Args:
            model_class (str): Class of the model (e.g., 'gemini', 'claude')
            llm (BaseChatModel): Instance of LLM
        
        Returns:
            RunnableSequence: Report generator chain
        """
        template = getattr(
            report_generator,
            f"{model_class}_report_generator_prompt"
        )
        report_generator_chain = create_llm_chain(llm=llm, template=template)
        return report_generator_chain
    
    @staticmethod
    def get_card_fact_checker_chain(
        model_class: str,
        llm: BaseChatModel
    ) -> RunnableSequence:
        """
        Create card fact checker chain.
        
        Args:
            model_class (str): Class of the model (e.g., 'gemini', 'claude')
            llm (BaseChatModel): Instance of LLM
        
        Returns:
            RunnableSequence: Card fact checker chain
        """
        template = getattr(
            card_fact_checker,
            f"{model_class}_card_fact_checker_prompt"
        )
        card_fact_checker_chain = create_llm_chain(llm=llm, template=template)
        return card_fact_checker_chain
    
    @staticmethod
    def get_chain_by_type(
        chain_type: str,
        model_class: str,
        llm: BaseChatModel,
        **kwargs
    ) -> RunnableSequence:
        """
        Generic method to get a chain by type.
        
        Args:
            chain_type (str): Type of chain to create
            model_class (str): Class of the model (e.g., 'gemini', 'claude')
            llm (BaseChatModel): Instance of LLM
            **kwargs: Additional arguments for specific chains
        
        Returns:
            RunnableSequence: Requested chain
        
        Raises:
            ValueError: If chain_type is not supported
        """
        chain_map = {
            "chunk_reranker": ChainCreator.get_chunk_reranker_chain,
            "answer_generator": ChainCreator.get_answer_generator_chain,
            "retrieval_sharper": ChainCreator.get_retrieval_sharper_chain,
            "fact_checker": ChainCreator.get_fact_checker_chain,
            "chunk_summarizer": ChainCreator.get_chunk_summarizer_chain,
            "keypoints_summarizer": ChainCreator.get_keypoints_summarizer_chain,
            "report_generator": ChainCreator.get_report_generator_chain,
            "card_fact_checker": ChainCreator.get_card_fact_checker_chain,
        }
        
        if chain_type not in chain_map:
            raise ValueError(
                f"Unsupported chain type: {chain_type}. "
                f"Available types: {list(chain_map.keys())}"
            )
        
        return chain_map[chain_type](model_class=model_class, llm=llm, **kwargs)
    
    @staticmethod
    def get_all_chains(
        model_class: str,
        llm: BaseChatModel
    ) -> Dict[str, RunnableSequence]:
        """
        Create all available chains.
        
        Args:
            model_class (str): Class of the model (e.g., 'gemini', 'claude')
            llm (BaseChatModel): Instance of LLM
        
        Returns:
            Dict[str, RunnableSequence]: Dictionary of all chains
        """
        return {
            "chunk_reranker": ChainCreator.get_chunk_reranker_chain(
                model_class=model_class, llm=llm
            ),
            "answer_generator": ChainCreator.get_answer_generator_chain(
                model_class=model_class, llm=llm
            ),
            "retrieval_sharper": ChainCreator.get_retrieval_sharper_chain(
                model_class=model_class, llm=llm
            ),
            "fact_checker": ChainCreator.get_fact_checker_chain(
                model_class=model_class, llm=llm
            ),
            "chunk_summarizer": ChainCreator.get_chunk_summarizer_chain(
                model_class=model_class, llm=llm
            ),
            "keypoints_summarizer": ChainCreator.get_keypoints_summarizer_chain(
                model_class=model_class, llm=llm
            ),
            "report_generator": ChainCreator.get_report_generator_chain(
                model_class=model_class, llm=llm
            ),
            "card_fact_checker": ChainCreator.get_card_fact_checker_chain(
                model_class=model_class, llm=llm
            ),
        }


# Convenience function for creating chains with default settings
def create_chain(
    chain_type: str,
    model_class: str,
    llm: BaseChatModel,
    **kwargs
) -> RunnableSequence:
    """
    Convenience function to create a chain.
    
    Args:
        chain_type (str): Type of chain to create
        model_class (str): Class of the model (e.g., 'gemini', 'claude')
        llm (BaseChatModel): Instance of LLM
        **kwargs: Additional arguments for specific chains
    
    Returns:
        RunnableSequence: Requested chain
    """
    return ChainCreator.get_chain_by_type(
        chain_type=chain_type,
        model_class=model_class,
        llm=llm,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    from langchain_google_vertexai import ChatVertexAI
    from langchain_anthropic import ChatAnthropic
    
    # Example with Gemini
    gemini_llm = ChatVertexAI(
        model="gemini-1.5-pro-001",
        temperature=0.2,
        max_output_tokens=2048,
    )
    
    # Create an answer generator chain
    answer_chain = ChainCreator.get_answer_generator_chain(
        model_class="gemini",
        llm=gemini_llm
    )
    
    print("Created answer generator chain for Gemini")
    
    # Example with Claude
    claude_llm = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=0.2,
        max_tokens=2048,
    )
    
    # Create a fact checker chain
    fact_check_chain = ChainCreator.get_fact_checker_chain(
        model_class="claude",
        llm=claude_llm
    )
    
    print("Created fact checker chain for Claude")
    
    # Get all chains
    all_chains = ChainCreator.get_all_chains(
        model_class="gemini",
        llm=gemini_llm
    )
    
    print(f"Created {len(all_chains)} chains for Gemini")
    
    # Use convenience function
    summary_chain = create_chain(
        chain_type="chunk_summarizer",
        model_class="gemini",
        llm=gemini_llm
    )
    
    print("Created chunk summarizer chain using convenience function")