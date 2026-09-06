"""
Fact Checker Prompts for IBG Knowledge Base.

This module contains prompt templates for fact checking and relevance scoring
of responses against provided context. Supports both Claude and Gemini models.
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple


def claude_fact_checker_prompt() -> str:
    """
    Fact checker prompt template for Claude models.
    
    Returns:
        str: Prompt template for Claude fact checking
    
    This prompt instructs the model to:
    1. Evaluate response relevance to the question
    2. Score the response as low/medium/high
    3. Provide reasoning for the score
    4. Consider factual accuracy and completeness
    """
    return """
You are a relevance response rating agent. You give a score (low/medium/high) for responses based on a provided context and the reason why you gave this score.

You are responsible to ensure that you only respond with a score between one of the following 3 options: "low", "medium", "high".

Your responses should be relevant, concise, and provide enough information to help the model understand the question, without adding unnecessary or irrelevant details.

<criterion>
- low: The answer is not relevant or provides the wrong information. It misses the key point or contains mostly irrelevant content.
- medium: The answer captures the main idea but includes some redundant or unnecessary details that dilute the relevance.
- high: The answer is concise and fully relevant. It captures only the essential information without extra or irrelevant content.
</criterion>

<example_1>
Question: What type of organism is commonly used in preparation of foods such as cheese and yogurt?
Context: Mesophiles grow best in moderate temperature, typically between 25°C and 40°C (77°F and 104°F).
Response: In moderate temperature, typically between 25°C and 40°C (77°F and higher).
Score: low
Reason: Because it is only using irrelevant information, missing the correct answer (mesophiles).
</example_1>

<example_2>
Question: Changes from a less-ordered state to a more-ordered state (such as a liquid to a solid) are always what?
Context: Changes of state are examples of phase changes, or phase transitions. All phase changes are accompanied by changes in the energy of a system. Changes from a more-ordered state to a less-ordered state (such as a liquid to a solid) is called fusion (or melting). The energy required to melt 1 mol of a substance is its enthalpy of fusion. The energy change required to vaporize 1 mol of a substance is the enthalpy of vaporization. The direct conversion of a solid to a gas is sublimation. The direct conversion of a liquid to a gas is condensation.
Response: Sublimation
Score: high
Reason: Because it captures the essence, and there is no irrelevant information in the response.
</example_2>

Now score this response:
Question: {question}
Context: {context}
Response: {response}

Output Format:
Score: [low/medium/high]
Reason: [Your detailed reasoning]
"""


def gemini_fact_checker_prompt() -> str:
    """
    Fact checker prompt template for Gemini models.
    
    Returns:
        str: Prompt template for Gemini fact checking
    """
    return """
You are a response relevance evaluator. Score the response based on the context and question.

QUESTION: {question}

CONTEXT: {context}

RESPONSE: {response}

SCORING CRITERIA:
- HIGH: Fully relevant, captures essential information, no irrelevant content
- MEDIUM: Captures main idea but has some redundant or unnecessary details
- LOW: Not relevant, wrong information, misses key points

OUTPUT FORMAT:
Score: [low/medium/high]
Reason: [Brief explanation for the score]

Provide only the score and reason in the specified format.
"""


def get_fact_checker_prompt(model_type: str = "claude") -> str:
    """
    Get the appropriate fact checker prompt for the specified model.
    
    Args:
        model_type (str): Type of model ('claude' or 'gemini')
    
    Returns:
        str: Prompt template for fact checking
    
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type == "claude":
        return claude_fact_checker_prompt()
    elif model_type == "gemini":
        return gemini_fact_checker_prompt()
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'claude' or 'gemini'.")


def format_fact_checker_prompt(
    question: str,
    context: str,
    response: str,
    model_type: str = "claude"
) -> str:
    """
    Format the fact checker prompt with provided parameters.
    
    Args:
        question (str): The original question
        context (str): The context to check against
        response (str): The response to evaluate
        model_type (str): Type of model ('claude' or 'gemini')
    
    Returns:
        str: Formatted prompt string
    """
    template = get_fact_checker_prompt(model_type)
    
    return template.format(
        question=question,
        context=context,
        response=response
    )


def parse_fact_checker_response(response: str) -> Tuple[str, str]:
    """
    Parse the fact checker response into score and reason.
    
    Args:
        response (str): The model's response containing score and reason
    
    Returns:
        Tuple[str, str]: (score, reason)
    
    Example:
        >>> response = "Score: high\\nReason: Directly answers the question"
        >>> score, reason = parse_fact_checker_response(response)
        >>> print(score)  # "high"
        >>> print(reason)  # "Directly answers the question"
    """
    score = "low"
    reason = ""
    
    try:
        # Try to find Score and Reason patterns
        score_match = re.search(
            r'[Ss]core\s*[:=]\s*(low|medium|high)',
            response,
            re.IGNORECASE
        )
        if score_match:
            score = score_match.group(1).lower()
        
        # Try to find Reason
        reason_match = re.search(
            r'[Rr]eason\s*[:=]\s*(.*?)(?=(?:Score:|$))',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if reason_match:
            reason = reason_match.group(1).strip()
        
        # If reason not found, try alternative pattern
        if not reason:
            # Look for text after "Reason:" or "because"
            alt_match = re.search(
                r'(?:[Rr]eason|[Bb]ecause)\s*[:=]?\s*(.*?)(?:$|\.)',
                response,
                re.DOTALL
            )
            if alt_match:
                reason = alt_match.group(1).strip()
        
        # Clean up reason
        if reason:
            # Remove extra whitespace
            reason = re.sub(r'\s+', ' ', reason)
            # Capitalize first letter
            if reason:
                reason = reason[0].upper() + reason[1:] if len(reason) > 1 else reason
        
    except Exception as e:
        logging.warning(f"Error parsing fact checker response: {e}")
    
    return score, reason


def is_response_factual(
    score: str,
    threshold: str = "medium"
) -> bool:
    """
    Determine if a response is considered factual based on score.
    
    Args:
        score (str): The score from fact checker
        threshold (str): Minimum acceptable score ('high' or 'medium')
    
    Returns:
        bool: True if response passes the threshold
    
    Example:
        >>> is_response_factual("high", "medium")  # True
        >>> is_response_factual("low", "medium")   # False
    """
    score_order = {"high": 3, "medium": 2, "low": 1}
    threshold_order = {"high": 3, "medium": 2, "low": 1}
    
    score_value = score_order.get(score.lower(), 0)
    threshold_value = threshold_order.get(threshold.lower(), 2)
    
    return score_value >= threshold_value


def validate_fact_checker_response(response: str) -> bool:
    """
    Validate that the fact checker response contains required elements.
    
    Args:
        response (str): The fact checker response
    
    Returns:
        bool: True if response is valid
    """
    if not response or not response.strip():
        return False
    
    # Check if response contains a score
    score_match = re.search(
        r'[Ss]core\s*[:=]\s*(low|medium|high)',
        response,
        re.IGNORECASE
    )
    if not score_match:
        return False
    
    # Check if response contains a reason
    reason_match = re.search(
        r'[Rr]eason\s*[:=]|because',
        response,
        re.IGNORECASE
    )
    if not reason_match:
        return False
    
    return True


# Dictionary mapping model types to prompt functions
FACT_CHECKER_PROMPTS = {
    "claude": claude_fact_checker_prompt,
    "gemini": gemini_fact_checker_prompt,
}


def get_all_prompts() -> Dict[str, str]:
    """
    Get all available fact checker prompts.
    
    Returns:
        Dict[str, str]: Dictionary of prompt names and their templates
    """
    return {
        "claude_fact_checker_prompt": claude_fact_checker_prompt(),
        "gemini_fact_checker_prompt": gemini_fact_checker_prompt(),
    }


# Example usage
if __name__ == "__main__":
    # Get Claude prompt
    claude_prompt = claude_fact_checker_prompt()
    print("Claude Fact Checker Prompt:")
    print(claude_prompt)
    print("\n" + "="*50 + "\n")
    
    # Get Gemini prompt
    gemini_prompt = gemini_fact_checker_prompt()
    print("Gemini Fact Checker Prompt:")
    print(gemini_prompt)
    print("\n" + "="*50 + "\n")
    
    # Format a prompt with parameters
    formatted_prompt = format_fact_checker_prompt(
        question="What is the capital of France?",
        context="France is a country in Europe. Its capital is Paris, which is known for the Eiffel Tower.",
        response="Paris is the capital of France and is famous for the Eiffel Tower.",
        model_type="claude"
    )
    print("Formatted Prompt:")
    print(formatted_prompt)
    print("\n" + "="*50 + "\n")
    
    # Test response parsing
    sample_response = """
    Score: high
    Reason: The response correctly identifies Paris as the capital of France and provides accurate additional information about the Eiffel Tower, which is directly supported by the context.
    """
    
    score, reason = parse_fact_checker_response(sample_response)
    print(f"Parsed Score: {score}")
    print(f"Parsed Reason: {reason}")
    print(f"Is Factual: {is_response_factual(score)}")
    
    print("\n" + "="*50 + "\n")
    
    # Test validation
    valid = validate_fact_checker_response(sample_response)
    print(f"Valid Response: {valid}")
    
    # Test with different scores
    for test_score in ["high", "medium", "low"]:
        factual = is_response_factual(test_score, "medium")
        print(f"Score '{test_score}' with threshold 'medium': {factual}")