"""
Chunk Reranker Prompts for IBG Knowledge Base.

This module contains prompt templates for reranking document chunks based on
their relevance to a given query, supporting both Claude and Gemini models.
"""

from typing import Dict, Any, Optional, List


def claude_chunk_reranker_prompt() -> str:
    """
    Chunk reranker prompt template for Claude models.
    
    Returns:
        str: Prompt template for Claude chunk reranking
    
    This prompt instructs the model to:
    1. Review a group of document chunks
    2. Rank them by relevance to the query
    3. Return the most relevant chunks
    """
    return """
Query: {query}

You are an expert at evaluating a group of documents and determining which ones are most relevant to a given query.
Your task is to identify the most relevant chunks from the list below that best answer the query.

<documents>
{chunks}
</documents>

Instructions:
1. Evaluate each document chunk for relevance to the query.
2. Rank the chunks from most relevant to least relevant.
3. Return the top {num_chunks} most relevant chunks.
4. If chunks are equally relevant, maintain their original order.
5. Provide a brief reason for each ranking.

Output Format:
<ranking>
1. [Chunk ID] - Relevance Score: [score] - Reason: [brief explanation]
2. [Chunk ID] - Relevance Score: [score] - Reason: [brief explanation]
...
</ranking>

Return the reranked chunks in order of relevance.
"""


def gemini_chunk_reranker_prompt() -> str:
    """
    Chunk reranker prompt template for Gemini models.
    
    Returns:
        str: Prompt template for Gemini chunk reranking
    """
    return """
You are an expert document evaluator.
Given a query and a list of document chunks, rank the chunks by relevance to the query.

QUERY: {query}

DOCUMENT CHUNKS:
{chunks}

RULES:
1. Rank chunks from most relevant to least relevant.
2. Consider semantic relevance, not just keyword matching.
3. Return the top {num_chunks} most relevant chunks.
4. Provide a relevance score (0-100) for each chunk.
5. Explain briefly why each chunk is ranked where it is.

FORMAT:
<ranking>
Rank 1: [Chunk ID] - Score: [X/100] - Reason: [explanation]
Rank 2: [Chunk ID] - Score: [X/100] - Reason: [explanation]
...
</ranking>

Return only the ranked chunks in the specified format.
"""


def get_chunk_reranker_prompt(model_type: str = "claude") -> str:
    """
    Get the appropriate chunk reranker prompt for the specified model.
    
    Args:
        model_type (str): Type of model ('claude' or 'gemini')
    
    Returns:
        str: Prompt template for chunk reranking
    
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type == "claude":
        return claude_chunk_reranker_prompt()
    elif model_type == "gemini":
        return gemini_chunk_reranker_prompt()
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'claude' or 'gemini'.")


def format_chunk_reranker_prompt(
    query: str,
    chunks: List[str],
    num_chunks: int = 5,
    model_type: str = "claude"
) -> str:
    """
    Format the chunk reranker prompt with provided parameters.
    
    Args:
        query (str): The user query
        chunks (List[str]): List of document chunks
        num_chunks (int): Number of chunks to return
        model_type (str): Type of model ('claude' or 'gemini')
    
    Returns:
        str: Formatted prompt string
    """
    template = get_chunk_reranker_prompt(model_type)
    
    # Format chunks with IDs
    formatted_chunks = ""
    for idx, chunk in enumerate(chunks, 1):
        formatted_chunks += f"Chunk {idx}: {chunk}\n\n"
    
    return template.format(
        query=query,
        chunks=formatted_chunks,
        num_chunks=num_chunks
    )


def parse_reranker_response(response: str) -> List[Dict[str, Any]]:
    """
    Parse the chunk reranker response into structured data.
    
    Args:
        response (str): The model's response containing ranked chunks
    
    Returns:
        List[Dict[str, Any]]: List of ranked chunks with scores and reasons
    
    Example:
        >>> response = "<ranking>\\n1. Chunk 1 - Score: 95 - Reason: Direct match\\n</ranking>"
        >>> result = parse_reranker_response(response)
        >>> print(result)  # [{"rank": 1, "chunk_id": "Chunk 1", "score": 95, "reason": "Direct match"}]
    """
    results = []
    
    try:
        # Extract ranking section
        ranking_match = re.search(r'<ranking>(.*?)</ranking>', response, re.DOTALL)
        if not ranking_match:
            return results
        
        ranking_text = ranking_match.group(1)
        
        # Parse each line
        lines = ranking_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to parse different formats
            # Format 1: "Rank 1: Chunk 1 - Score: 95 - Reason: Direct match"
            # Format 2: "1. Chunk 1 - Relevance Score: 95 - Reason: Direct match"
            # Format 3: "1: Chunk 1 | Score: 95 | Reason: Direct match"
            
            patterns = [
                r'(?:Rank\s*)?(\d+)[:\.]\s*(.+?)\s*[-|]\s*Score:?\s*(\d+)',
                r'(?:Rank\s*)?(\d+)[:\.]\s*(.+?)\s*[-|]\s*Relevance Score:?\s*(\d+)',
                r'(?:Rank\s*)?(\d+)[:\.]\s*(.+?)\s*[-|]\s*Score:?\s*(\d+).*?Reason:?\s*(.+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) >= 3:
                        result = {
                            "rank": int(groups[0]),
                            "chunk_id": groups[1].strip(),
                            "score": int(groups[2]) if groups[2].isdigit() else 0,
                        }
                        if len(groups) >= 4:
                            result["reason"] = groups[3].strip()
                        results.append(result)
                        break
        
    except Exception as e:
        logging.warning(f"Error parsing reranker response: {e}")
    
    return results


# Dictionary mapping model types to prompt functions
CHUNK_RERANKER_PROMPTS = {
    "claude": claude_chunk_reranker_prompt,
    "gemini": gemini_chunk_reranker_prompt,
}


def get_all_prompts() -> Dict[str, str]:
    """
    Get all available chunk reranker prompts.
    
    Returns:
        Dict[str, str]: Dictionary of prompt names and their templates
    """
    return {
        "claude_chunk_reranker_prompt": claude_chunk_reranker_prompt(),
        "gemini_chunk_reranker_prompt": gemini_chunk_reranker_prompt(),
    }


# Example usage
if __name__ == "__main__":
    import re
    import logging
    
    # Get Claude prompt
    claude_prompt = claude_chunk_reranker_prompt()
    print("Claude Chunk Reranker Prompt:")
    print(claude_prompt)
    print("\n" + "="*50 + "\n")
    
    # Get Gemini prompt
    gemini_prompt = gemini_chunk_reranker_prompt()
    print("Gemini Chunk Reranker Prompt:")
    print(gemini_prompt)
    print("\n" + "="*50 + "\n")
    
    # Format a prompt with parameters
    sample_chunks = [
        "This document discusses revenue growth in Q3 2024.",
        "This document covers employee hiring trends.",
        "This document details market expansion strategies.",
        "This document provides financial risk assessment.",
    ]
    
    formatted_prompt = format_chunk_reranker_prompt(
        query="What is the revenue growth outlook?",
        chunks=sample_chunks,
        num_chunks=3,
        model_type="claude"
    )
    print("Formatted Prompt:")
    print(formatted_prompt)
    print("\n" + "="*50 + "\n")
    
    # Test response parsing
    sample_response = """
    <ranking>
    1. Chunk 1 - Score: 95 - Reason: Directly addresses revenue growth
    2. Chunk 4 - Score: 80 - Reason: Related to financial metrics
    3. Chunk 3 - Score: 60 - Reason: Market context
    </ranking>
    """
    
    parsed_results = parse_reranker_response(sample_response)
    print("Parsed Results:")
    for result in parsed_results:
        print(f"Rank {result['rank']}: {result['chunk_id']} (Score: {result['score']}) - {result.get('reason', 'No reason provided')}")