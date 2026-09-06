"""
Retrieval Sharper Prompts for IBG Knowledge Base.

This module contains prompt templates for refining and sharpening retrieval results,
helping to identify the most relevant document chunks for a given query.
Supports both Claude and Gemini models.
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List, Union


def claude_retrieval_sharper_prompt() -> str:
    """
    Retrieval sharper prompt template for Claude models.
    
    Returns:
        str: Prompt template for Claude retrieval sharpening
    
    This prompt instructs the model to:
    1. Evaluate document chunks for relevance
    2. Rank chunks by relevance level (high/medium/low)
    3. Provide reasoning for the ranking
    4. Output in structured format
    """
    return """
You are an expert at evaluating document relevance for information retrieval.
Given a query and a list of document chunks, determine which chunks are most relevant.

<query>
{query}
</query>

<documents>
{chunks}
</documents>

<instructions>
1. For each document chunk, evaluate its relevance to the query.
2. Classify each chunk as:
   - HIGH: Directly answers the query or contains key information
   - MEDIUM: Partially relevant or contains related information
   - LOW: Not relevant or only tangentially related
3. Provide a brief justification for each classification.
4. Consider semantic relevance, not just keyword matching.
5. If multiple chunks are relevant, prioritize those with more specific information.
</instructions>

<output_format>
<relevance>
High relevance chunks:
- [Chunk ID 1]: [Brief justification]
- [Chunk ID 2]: [Brief justification]

Medium relevance chunks:
- [Chunk ID 3]: [Brief justification]

Low relevance chunks:
- [Chunk ID 4]: [Brief justification]
</relevance>

Return only the relevance classification in the specified format.
"""


def gemini_retrieval_sharper_prompt() -> str:
    """
    Retrieval sharper prompt template for Gemini models.
    
    Returns:
        str: Prompt template for Gemini retrieval sharpening
    """
    return """
You are a document relevance evaluator.
Given a query and document chunks, classify the relevance of each chunk.

QUERY: {query}

DOCUMENTS:
{chunks}

CLASSIFICATION RULES:
1. HIGH: Directly answers the query, contains specific facts or figures
2. MEDIUM: Partially relevant, contains related context
3. LOW: Not relevant, off-topic

JUSTIFICATION: For each chunk, explain why it received its classification.

OUTPUT FORMAT:
High relevance:
- [Chunk ID]: [Justification]

Medium relevance:
- [Chunk ID]: [Justification]

Low relevance:
- [Chunk ID]: [Justification]

Return only the classifications in the specified format.
"""


def get_retrieval_sharper_prompt(model_type: str = "claude") -> str:
    """
    Get the appropriate retrieval sharper prompt for the specified model.
    
    Args:
        model_type (str): Type of model ('claude' or 'gemini')
    
    Returns:
        str: Prompt template for retrieval sharpening
    
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type == "claude":
        return claude_retrieval_sharper_prompt()
    elif model_type == "gemini":
        return gemini_retrieval_sharper_prompt()
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'claude' or 'gemini'.")


def format_retrieval_sharper_prompt(
    query: str,
    chunks: List[str],
    model_type: str = "claude"
) -> str:
    """
    Format the retrieval sharper prompt with provided parameters.
    
    Args:
        query (str): The user query
        chunks (List[str]): List of document chunks
        model_type (str): Type of model ('claude' or 'gemini')
    
    Returns:
        str: Formatted prompt string
    """
    template = get_retrieval_sharper_prompt(model_type)
    
    # Format chunks with IDs
    formatted_chunks = ""
    for idx, chunk in enumerate(chunks, 1):
        formatted_chunks += f"Chunk {idx}: {chunk}\n\n"
    
    return template.format(
        query=query,
        chunks=formatted_chunks
    )


def parse_retrieval_sharper_response(response: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse the retrieval sharper response into structured data.
    
    Args:
        response (str): The model's response containing relevance classification
    
    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary with relevance levels
    
    Example:
        >>> response = '''High relevance:
        ... - Chunk 1: Directly answers the query
        ... Medium relevance:
        ... - Chunk 2: Partially relevant'''
        >>> result = parse_retrieval_sharper_response(response)
        >>> print(result['high'][0]['chunk_id'])  # "Chunk 1"
    """
    result = {
        "high": [],
        "medium": [],
        "low": []
    }
    
    try:
        # Extract relevance sections
        sections = re.findall(
            r'(high|medium|low)\s*(?:relevance|rel)?\s*:?\s*(.*?)(?=(?:high|medium|low)\s*(?:relevance|rel)?\s*:|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        
        # If sections not found, try to parse line by line
        if not sections:
            return parse_retrieval_sharper_response_line_by_line(response)
        
        for level, content in sections:
            level = level.lower()
            if level not in result:
                continue
            
            # Parse each chunk in the section
            # Look for patterns like "- Chunk X: ..." or "Chunk X: ..."
            chunk_pattern = r'(?:[-*]\s*)?Chunk\s*(\d+)\s*:?\s*(.*?)(?=(?:[-*]\s*)?Chunk\s*\d+|$)'
            chunks = re.findall(chunk_pattern, content, re.DOTALL | re.IGNORECASE)
            
            for chunk_id, justification in chunks:
                if justification.strip():
                    result[level].append({
                        "chunk_id": int(chunk_id),
                        "justification": justification.strip()
                    })
    
    except Exception as e:
        logging.warning(f"Error parsing retrieval sharper response: {e}")
    
    return result


def parse_retrieval_sharper_response_line_by_line(response: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse retrieval sharper response line by line.
    
    Args:
        response (str): The model's response
    
    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary with relevance levels
    """
    result = {
        "high": [],
        "medium": [],
        "low": []
    }
    
    current_level = None
    
    for line in response.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Check for level indicators
        line_lower = line.lower()
        if 'high' in line_lower and ('relevance' in line_lower or 'rel' in line_lower):
            current_level = 'high'
            continue
        elif 'medium' in line_lower and ('relevance' in line_lower or 'rel' in line_lower):
            current_level = 'medium'
            continue
        elif 'low' in line_lower and ('relevance' in line_lower or 'rel' in line_lower):
            current_level = 'low'
            continue
        
        # Parse chunk lines
        if current_level and ('chunk' in line_lower or '-' in line):
            # Try to extract chunk ID and justification
            chunk_pattern = r'(?:[-*]\s*)?Chunk\s*(\d+)\s*:?\s*(.*)'
            match = re.match(chunk_pattern, line, re.IGNORECASE)
            
            if match:
                chunk_id = int(match.group(1))
                justification = match.group(2).strip()
                if justification:
                    result[current_level].append({
                        "chunk_id": chunk_id,
                        "justification": justification
                    })
    
    return result


def filter_chunks_by_relevance(
    chunks: List[str],
    relevance_response: str,
    min_relevance: str = "medium"
) -> List[str]:
    """
    Filter chunks based on relevance classification.
    
    Args:
        chunks (List[str]): Original list of chunks
        relevance_response (str): Model response with relevance classification
        min_relevance (str): Minimum relevance level to include ('high' or 'medium')
    
    Returns:
        List[str]: Filtered list of chunks
    """
    if not chunks or not relevance_response:
        return chunks
    
    parsed = parse_retrieval_sharper_response(relevance_response)
    
    # Determine which levels to include
    include_levels = ['high']
    if min_relevance.lower() == 'medium':
        include_levels.append('medium')
    
    # Collect chunk IDs to include
    included_ids = []
    for level in include_levels:
        if level in parsed:
            for item in parsed[level]:
                included_ids.append(item['chunk_id'])
    
    # Filter chunks based on IDs (1-indexed)
    filtered_chunks = []
    for idx, chunk in enumerate(chunks, 1):
        if idx in included_ids:
            filtered_chunks.append(chunk)
    
    return filtered_chunks


# Dictionary mapping model types to prompt functions
RETRIEVAL_SHARPER_PROMPTS = {
    "claude": claude_retrieval_sharper_prompt,
    "gemini": gemini_retrieval_sharper_prompt,
}


def get_all_prompts() -> Dict[str, str]:
    """
    Get all available retrieval sharper prompts.
    
    Returns:
        Dict[str, str]: Dictionary of prompt names and their templates
    """
    return {
        "claude_retrieval_sharper_prompt": claude_retrieval_sharper_prompt(),
        "gemini_retrieval_sharper_prompt": gemini_retrieval_sharper_prompt(),
    }


# Example usage
if __name__ == "__main__":
    # Get Claude prompt
    claude_prompt = claude_retrieval_sharper_prompt()
    print("Claude Retrieval Sharper Prompt:")
    print(claude_prompt)
    print("\n" + "="*50 + "\n")
    
    # Get Gemini prompt
    gemini_prompt = gemini_retrieval_sharper_prompt()
    print("Gemini Retrieval Sharper Prompt:")
    print(gemini_prompt)
    print("\n" + "="*50 + "\n")
    
    # Format a prompt with parameters
    sample_chunks = [
        "Revenue increased by 15% in Q3 2024 due to strong market demand.",
        "The company hired 50 new employees in the technology division.",
        "Financial outlook remains positive with projected growth of 10%.",
        "The marketing team launched a new campaign in Southeast Asia.",
    ]
    
    formatted_prompt = format_retrieval_sharper_prompt(
        query="What is the revenue growth outlook?",
        chunks=sample_chunks,
        model_type="claude"
    )
    print("Formatted Prompt:")
    print(formatted_prompt)
    print("\n" + "="*50 + "\n")
    
    # Test response parsing
    sample_response = """
    High relevance:
    - Chunk 1: Directly provides revenue growth data of 15%
    - Chunk 3: Mentions positive financial outlook with 10% growth
    
    Medium relevance:
    - Chunk 4: Related to company activities but not revenue
    
    Low relevance:
    - Chunk 2: About hiring, not revenue growth
    """
    
    parsed_results = parse_retrieval_sharper_response(sample_response)
    
    print("Parsed Results:")
    for level in ['high', 'medium', 'low']:
        print(f"\n{level.upper()} Relevance:")
        for item in parsed_results[level]:
            print(f"  - Chunk {item['chunk_id']}: {item['justification']}")
    
    print("\n" + "="*50 + "\n")
    
    # Test chunk filtering
    filtered_chunks = filter_chunks_by_relevance(
        chunks=sample_chunks,
        relevance_response=sample_response,
        min_relevance="medium"
    )
    print(f"Filtered Chunks ({len(filtered_chunks)} of {len(sample_chunks)}):")
    for idx, chunk in enumerate(filtered_chunks, 1):
        print(f"{idx}. {chunk[:100]}...")