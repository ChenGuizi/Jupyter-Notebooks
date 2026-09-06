"""
Answer Generator Prompts for IBG Knowledge Base.

This module contains prompt templates for answer generation using various LLM models
(Gemini, Claude, etc.) with consistent formatting and instructions.
"""

from typing import Dict, Any, Optional


def claude_answer_generator_prompt() -> str:
    """
    Answer generator prompt template for Claude models.
    
    Returns:
        str: Prompt template for Claude answer generation
    
    This prompt instructs the model to:
    1. Act as a Corporate Banking Relationship Manager
    2. Use provided context to answer questions
    3. Not make up information not in context
    4. Provide reasoning in thinking tags
    5. Output answer in XML-style tags
    """
    return """
You will be acting as a Corporate Banking Relationship Manager. 
Use the following <context> to answer the question based on the provided context.

<context>
{context}
</context>

<instructions>
Instruction 1: Use following rules to answer the question.
1. Do not answer if the information is not available in the context.
2. Do not make up information and only answer the question based on the provided context.
3. Accurately interpret numerical data.
4. Use the date {curr_date} to answer if relevant.
5. If the context contains multiple pieces of information, synthesize them appropriately.

Instruction 2: Use following rules to output the answer.
1. Before answering, explain your reasoning step-by-step in <thinking></thinking> tags.
2. Output your final answer in <answer></answer> tags.
3. If the information is not available, state that clearly.
4. Cite specific parts of the context when possible.
</instructions>

Human: {question}
"""


def gemini_answer_generator_prompt() -> str:
    """
    Answer generator prompt template for Gemini models.
    
    Returns:
        str: Prompt template for Gemini answer generation
    """
    return """
You are a Corporate Banking Relationship Manager assistant.
Answer the following question based on the provided context.

CONTEXT:
{context}

INSTRUCTIONS:
1. Only use information from the context to answer.
2. If the information is not in the context, say "The information is not available."
3. Do not make up or infer information not present in the context.
4. Provide accurate interpretations of numerical data.
5. Use the current date {curr_date} when relevant.

FORMAT:
1. First, provide your reasoning in <thinking> tags.
2. Then, provide your final answer in <answer> tags.

QUESTION: {question}
"""


def claude_answer_generator_upload_doc_prompt() -> str:
    """
    Answer generator prompt template for Claude models when using uploaded documents.
    
    Returns:
        str: Prompt template for Claude answer generation with uploaded docs
    """
    return """
You are a Corporate Banking Relationship Manager assistant.
Answer the following question based on the uploaded document content.

DOCUMENT CONTENT:
{context}

INSTRUCTIONS:
1. Answer questions strictly based on the uploaded document content.
2. If the information is not in the uploaded documents, state that clearly.
3. Do not use external knowledge or make up information.
4. Provide accurate analysis of the document content.
5. Use the current date {curr_date} when relevant.

FORMAT:
1. Provide your reasoning in <thinking> tags.
2. Provide your final answer in <answer> tags.

QUESTION: {question}
"""


def gemini_answer_generator_upload_doc_prompt() -> str:
    """
    Answer generator prompt template for Gemini models when using uploaded documents.
    
    Returns:
        str: Prompt template for Gemini answer generation with uploaded docs
    """
    return """
You are a Corporate Banking Relationship Manager assistant.
Answer the following question based on the uploaded document content.

DOCUMENT:
{context}

RULES:
1. Only use information from the uploaded document.
2. If the answer is not in the document, say "Not found in the uploaded document."
3. Do not add external information.
4. Be precise and accurate in your analysis.
5. Use the current date {curr_date} if applicable.

OUTPUT:
1. <thinking>Your reasoning process here</thinking>
2. <answer>Your final answer here</answer>

QUESTION: {question}
"""


def get_answer_generator_prompt(
    model_type: str,
    use_upload_doc: bool = False
) -> str:
    """
    Get the appropriate answer generator prompt for the specified model.
    
    Args:
        model_type (str): Type of model ('claude' or 'gemini')
        use_upload_doc (bool): Whether to use upload document specific prompt
    
    Returns:
        str: Prompt template for answer generation
    
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type == "claude":
        if use_upload_doc:
            return claude_answer_generator_upload_doc_prompt()
        return claude_answer_generator_prompt()
    
    elif model_type == "gemini":
        if use_upload_doc:
            return gemini_answer_generator_upload_doc_prompt()
        return gemini_answer_generator_prompt()
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'claude' or 'gemini'.")


def format_answer_generator_prompt(
    context: str,
    question: str,
    curr_date: str = "",
    model_type: str = "claude",
    use_upload_doc: bool = False
) -> str:
    """
    Format the answer generator prompt with provided parameters.
    
    Args:
        context (str): The context/document content
        question (str): The user question
        curr_date (str): Current date for contextual answers
        model_type (str): Type of model ('claude' or 'gemini')
        use_upload_doc (bool): Whether to use upload document specific prompt
    
    Returns:
        str: Formatted prompt string
    """
    template = get_answer_generator_prompt(model_type, use_upload_doc)
    
    return template.format(
        context=context,
        question=question,
        curr_date=curr_date
    )


# Dictionary mapping model types to prompt functions
ANSWER_GENERATOR_PROMPTS = {
    "claude": claude_answer_generator_prompt,
    "claude_upload": claude_answer_generator_upload_doc_prompt,
    "gemini": gemini_answer_generator_prompt,
    "gemini_upload": gemini_answer_generator_upload_doc_prompt,
}


def get_all_prompts() -> Dict[str, str]:
    """
    Get all available answer generator prompts.
    
    Returns:
        Dict[str, str]: Dictionary of prompt names and their templates
    """
    return {
        "claude_answer_generator_prompt": claude_answer_generator_prompt(),
        "claude_answer_generator_upload_doc_prompt": claude_answer_generator_upload_doc_prompt(),
        "gemini_answer_generator_prompt": gemini_answer_generator_prompt(),
        "gemini_answer_generator_upload_doc_prompt": gemini_answer_generator_upload_doc_prompt(),
    }


# Example usage
if __name__ == "__main__":
    # Get Claude prompt
    claude_prompt = claude_answer_generator_prompt()
    print("Claude Prompt Template:")
    print(claude_prompt)
    print("\n" + "="*50 + "\n")
    
    # Get Gemini prompt
    gemini_prompt = gemini_answer_generator_prompt()
    print("Gemini Prompt Template:")
    print(gemini_prompt)
    print("\n" + "="*50 + "\n")
    
    # Format a prompt with parameters
    formatted_prompt = format_answer_generator_prompt(
        context="Company XYZ reported revenue of $1.2B in 2024.",
        question="What was the company's revenue?",
        curr_date="August 2025",
        model_type="claude"
    )
    print("Formatted Prompt:")
    print(formatted_prompt)
    print("\n" + "="*50 + "\n")
    
    # Get all prompts
    all_prompts = get_all_prompts()
    for name, prompt in all_prompts.items():
        print(f"{name}:\n{prompt[:200]}...\n")