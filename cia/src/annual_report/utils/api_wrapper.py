"""
API Wrapper Module for IBG Knowledge Base.

This module provides a wrapper for Anthropic/Claude API with:
- Rate limiting with token bucket
- Retry logic with exponential backoff
- Message transformation
- Token counting
- Async and sync support
- Tool binding support
"""

import os
import re
import json
import uuid
import time
import asyncio
import logging
import traceback
from typing import List, Dict, Any, Optional, Union, Sequence, Tuple, Type, Callable
from operator import itemgetter
from pydantic import BaseModel, Field, model_validator

import httpx
import tiktoken
from anthropic.types import Message, TextBlock, Usage

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel as PydanticBaseModel

from .message_utils import (
    format_messages_anthropic,
    convert_to_anthropic_tool,
    extract_tool_calls,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class TokenBucket:
    """
    A token bucket for rate limiting.
    
    Controls the throughput of API calls by limiting the number of tokens
    that can be consumed per second.
    """
    
    def __init__(self, rate: float, capacity: int):
        """
        Initialize the token bucket.
        
        Args:
            rate (float): The rate at which tokens are generated (tokens per second)
            capacity (int): The maximum number of tokens the bucket can hold
        """
        self.rate = rate  # tokens per second
        self.capacity = capacity  # maximum tokens in the bucket
        self.tokens = capacity
        self.last_check = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """
        Acquire a token from the bucket.
        
        Waits if no tokens are available.
        """
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed_time = now - self.last_check
                
                # Refill tokens based on elapsed time
                self.tokens += elapsed_time * self.rate
                
                # Cap tokens to capacity
                if self.tokens > self.capacity:
                    self.tokens = self.capacity
                
                if self.tokens >= 1:
                    self.last_check = now
                    self.tokens -= 1  # Use a token
                    return
                else:
                    # Wait before retrying
                    await asyncio.sleep(0.1)


class APIChatAnthropic(BaseChatModel):
    """
    Wrapper for Anthropic/Claude API.
    
    Features:
    - Rate limiting with token bucket
    - Retry logic with exponential backoff
    - Message transformation
    - Token counting
    - Async and sync support
    - Tool binding support
    """
    
    config: Dict[str, Any] = Field(default_factory=dict)
    model: Optional[str] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    max_retries: int = Field(default=10)
    token_bucket: Optional[TokenBucket] = None
    
    @model_validator(mode="after")
    def set_attributes_from_config(self) -> 'APIChatAnthropic':
        """
        Set attributes from configuration after initialization.
        
        Returns:
            APIChatAnthropic: Self with updated attributes
        """
        api_wrapper = self.config.get("api_wrapper", {})
        claude_api = api_wrapper.get("claude_api", {})
        
        self.model = self.model or claude_api.get("model", "claude-3-5-sonnet-20240620")
        self.max_output_tokens = self.max_output_tokens or claude_api.get("max_tokens_to_sample", 2048)
        self.temperature = self.temperature or api_wrapper.get("temperature", 0.2)
        
        # Initialize token bucket for rate limiting
        self.token_bucket = TokenBucket(
            rate=api_wrapper.get("token_rate", 100),
            capacity=api_wrapper.get("token_capacity", 1000)
        )
        
        return self
    
    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return self.model or "claude"
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get default parameters for API calls."""
        return {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
    
    @staticmethod
    def count_tokens(text: str, encoding_constructor: str = "cl100k_base") -> int:
        """
        Count the number of tokens in the text.
        
        Args:
            text (str): Input text
            encoding_constructor (str): Tokenizer to use (default: cl100k_base)
        
        Returns:
            int: Number of tokens in the text
        """
        if not text:
            return 0
        
        try:
            # Set cache directory for tiktoken
            if "TIKTOKEN_CACHE_DIR" not in os.environ:
                current_directory = os.path.dirname(os.path.abspath(__file__))
                cache_dir = os.path.join(current_directory, "tiktoken_cache")
                os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
                os.makedirs(cache_dir, exist_ok=True)
            
            encoding = tiktoken.get_encoding(encoding_constructor)
            return len(encoding.encode(text))
            
        except Exception as e:
            logging.warning(f"Token counting error: {e}")
            # Fallback to character-based estimation
            return len(text) // 4
    
    @staticmethod
    def transform_messages(
        messages: List[Dict[str, Union[str, dict]]]
    ) -> List[Dict[str, Union[str, dict]]]:
        """
        Transform messages for API use.
        
        Args:
            messages (List[Dict[str, Union[str, dict]]]): Conversation between user and assistant
        
        Returns:
            List[Dict[str, Union[str, dict]]]: Transformed messages
        """
        transformed_messages = []
        
        for message in messages:
            transformed = {
                "role": message.get("role", "user"),
                "content": message.get("content", "")
            }
            transformed_messages.append(transformed)
        
        return transformed_messages
    
    def get_header_and_body(
        self,
        messages: List[Dict[str, Union[str, dict]]]
    ) -> Tuple[str, str, Dict[str, str], Dict[str, Any]]:
        """
        Get the header and body for the POST request.
        
        Args:
            messages (List[Dict[str, Union[str, dict]]]): Conversation between user and assistant
        
        Returns:
            Tuple[str, str, Dict[str, str], Dict[str, Any]]: 
                uniqueId, api_endpoint, headers, body
        """
        api_wrapper = self.config.get("api_wrapper", {})
        claude_api = api_wrapper.get("claude_api", {})
        
        api_endpoint = claude_api.get("endpoint")
        unique_id = f"{api_wrapper.get('unique_id_prefix', 'IDW')}-{uuid.uuid4()}"
        
        headers = {
            "app_code": api_wrapper.get("app_code", ""),
            "api_key": claude_api.get("api_key", ""),
            "Content-Type": "application/json",
        }
        
        # Transform messages
        transformed_messages = self.transform_messages(messages)
        
        body = {
            "utterance": transformed_messages,
            "uniqueId": unique_id,
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        
        return unique_id, api_endpoint, headers, body
    
    def api_invoke(
        self,
        messages: List[Dict[str, Union[str, dict]]]
    ) -> Message:
        """
        Call the API to generate answer (synchronous).
        
        Args:
            messages (List[Dict[str, Union[str, dict]]]): Conversation between user and assistant
        
        Returns:
            Message: Response message from the API
        
        Raises:
            Exception: If API call fails after max retries
        """
        retry = 1
        sleep = 1
        
        while retry <= self.max_retries:
            try:
                unique_id, api_endpoint, headers, body = self.get_header_and_body(
                    messages=messages
                )
                
                response = httpx.post(
                    url=api_endpoint,
                    json=body,
                    headers=headers,
                    verify=self.config.get("api_wrapper", {}).get("verify", False),
                )
                
                if response.status_code == 200:
                    answer = response.json().get("data", {}).get("answer", "{}")
                    
                    # Parse answer
                    try:
                        answer_data = json.loads(answer)
                    except json.JSONDecodeError:
                        answer_data = {"content": answer}
                    
                    return self._create_message(
                        unique_id=unique_id,
                        content=answer_data.get("content", answer),
                        messages=messages,
                        output_tokens=self.count_tokens(answer)
                    )
                else:
                    logging.warning(
                        f"API call failed with status {response.status_code}: {response.text}"
                    )
                    raise httpx.HTTPStatusError(
                        f"Status: {response.status_code}, Response: {response.text}",
                        request=response.request,
                        response=response
                    )
                    
            except Exception as e:
                logging.warning(f"API call attempt {retry} failed: {str(e)}")
                logging.debug(f"Traceback: {traceback.format_exc()}")
                
                if retry == self.max_retries:
                    raise Exception(f"Exception raised in APIChatAnthropic.api_invoke(): {str(e)}")
                
                time.sleep(sleep)
                sleep *= 2
                retry += 1
        
        # Should not reach here
        raise Exception("Maximum retries exceeded in api_invoke")
    
    async def api_ainvoke(
        self,
        messages: List[Dict[str, Union[str, dict]]]
    ) -> Message:
        """
        Call the API to generate answer (asynchronous).
        
        Args:
            messages (List[Dict[str, Union[str, dict]]]): Conversation between user and assistant
        
        Returns:
            Message: Response message from the API
        
        Raises:
            Exception: If API call fails after max retries
        """
        retry = 1
        sleep = 1
        
        while retry <= self.max_retries:
            try:
                # Acquire token from bucket for rate limiting
                await self.token_bucket.acquire()
                
                unique_id, api_endpoint, headers, body = self.get_header_and_body(
                    messages=messages
                )
                
                async with httpx.AsyncClient(
                    verify=self.config.get("api_wrapper", {}).get("verify", False),
                    timeout=None
                ) as client:
                    response = await client.post(
                        url=api_endpoint,
                        json=body,
                        headers=headers,
                    )
                
                if response.status_code == 200:
                    answer = response.json().get("data", {}).get("answer", "{}")
                    
                    # Parse answer
                    try:
                        answer_data = json.loads(answer)
                    except json.JSONDecodeError:
                        answer_data = {"content": answer}
                    
                    return self._create_message(
                        unique_id=unique_id,
                        content=answer_data.get("content", answer),
                        messages=messages,
                        output_tokens=self.count_tokens(answer)
                    )
                else:
                    logging.warning(
                        f"API call failed with status {response.status_code}: {response.text}"
                    )
                    raise httpx.HTTPStatusError(
                        f"Status: {response.status_code}, Response: {response.text}",
                        request=response.request,
                        response=response
                    )
                    
            except Exception as e:
                logging.warning(f"API call attempt {retry} failed: {str(e)}")
                logging.debug(f"Traceback: {traceback.format_exc()}")
                
                if retry == self.max_retries:
                    raise Exception(f"Exception raised in APIChatAnthropic.api_ainvoke(): {str(e)}")
                
                await asyncio.sleep(sleep)
                sleep *= 2
                retry += 1
        
        # Should not reach here
        raise Exception("Maximum retries exceeded in api_ainvoke")
    
    def _create_message(
        self,
        unique_id: str,
        content: str,
        messages: List[Dict[str, Union[str, dict]]],
        output_tokens: int
    ) -> Message:
        """
        Create a Message object from API response.
        
        Args:
            unique_id (str): Unique ID for the message
            content (str): Response content
            messages (List[Dict[str, Union[str, dict]]]): Input messages
            output_tokens (int): Number of output tokens
        
        Returns:
            Message: Formatted message
        """
        # Count input tokens (excluding tool use)
        input_tokens = sum(
            self.count_tokens(message.get("content", [{}])[0].get("text", ""))
            for message in messages
            if message.get("role") != "tool_use"
        )
        
        return Message(
            id=unique_id,
            content=[TextBlock(text=content, type="text")],
            model=self.model,
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        )
    
    def _format_params(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Format parameters for API call.
        
        Args:
            messages (List[BaseMessage]): List of messages
            stop (Optional[List[str]]): Stop sequences
            **kwargs: Additional parameters
        
        Returns:
            Dict[str, Any]: Formatted parameters
        """
        params = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        
        # Format messages for Anthropic
        formatted_messages = format_messages_anthropic(messages)
        params["messages"] = formatted_messages
        
        # Update with additional parameters
        params.update(kwargs)
        
        if stop:
            params["stop_sequences"] = stop
        
        # Remove None values
        return {k: v for k, v in params.items() if v is not None}
    
    def _format_output(self, data: Any, **kwargs: Any) -> ChatResult:
        """
        Format API output to ChatResult.
        
        Args:
            data (Any): API response data
            **kwargs: Additional parameters
        
        Returns:
            ChatResult: Formatted chat result
        """
        # Extract content
        content = ""
        if hasattr(data, "content"):
            content = data.content[0].text if data.content else ""
        
        # Extract tool calls
        tool_calls = extract_tool_calls(data.content if hasattr(data, "content") else [])
        
        # Create message
        if tool_calls:
            msg = AIMessage(content=content, tool_calls=tool_calls)
        else:
            msg = AIMessage(content=content)
        
        # Add usage metadata
        if hasattr(data, "usage"):
            msg.usage_metadata = {
                "input_tokens": data.usage.input_tokens,
                "output_tokens": data.usage.output_tokens,
                "total_tokens": data.usage.input_tokens + data.usage.output_tokens,
            }
        
        # Create generation
        generation = ChatGeneration(message=msg)
        
        # LLM output
        llm_output = {
            "model": self.model,
            "usage": msg.usage_metadata if hasattr(msg, "usage_metadata") else {},
        }
        
        return ChatResult(generations=[generation], llm_output=llm_output)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> ChatResult:
        """
        Override the _generate method to implement the chat model logic.
        
        Args:
            messages (List[BaseMessage]): List of messages
            stop (Optional[List[str]]): Stop sequences
            run_manager (Optional[Any]): Run manager
            **kwargs: Additional parameters
        
        Returns:
            ChatResult: Chat generation result
        """
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        
        # Transform messages for API
        api_messages = self.transform_messages(params["messages"])
        
        # Call API
        api_output = self.api_invoke(messages=api_messages)
        
        return self._format_output(api_output, **kwargs)
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> ChatResult:
        """
        Override the _agenerate method to implement the chat model logic.
        
        Args:
            messages (List[BaseMessage]): List of messages
            stop (Optional[List[str]]): Stop sequences
            run_manager (Optional[Any]): Run manager
            **kwargs: Additional parameters
        
        Returns:
            ChatResult: Chat generation result
        """
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        
        # Transform messages for API
        api_messages = self.transform_messages(params["messages"])
        
        # Call API asynchronously
        api_output = await self.api_ainvoke(messages=api_messages)
        
        return self._format_output(api_output, **kwargs)
    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        tool_choice: Optional[Union[Dict[str, str], str]] = None,
        **kwargs: Any
    ) -> Runnable:
        """
        Bind tool-like objects to this chat model.
        
        Args:
            tools (Sequence): Tools to bind
            tool_choice (Optional): Tool choice configuration
            **kwargs: Additional parameters
        
        Returns:
            Runnable: Bound runnable
        """
        # Convert tools to Anthropic format
        formatted_tools = [convert_to_anthropic_tool(tool) for tool in tools]
        
        # Handle tool choice
        if not tool_choice:
            pass
        elif isinstance(tool_choice, dict):
            pass
        elif isinstance(tool_choice, str):
            if tool_choice in ("any", "auto"):
                pass
            else:
                raise ValueError(f"Unsupported tool choice type: {tool_choice}")
        else:
            raise ValueError(
                f"Unsupported tool choice type: {type(tool_choice)}. "
                f"Expected dict, 'any', or 'auto'."
            )
        
        return self.bind(tools=formatted_tools, **kwargs)
    
    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        include_raw: bool = False,
        **kwargs: Any
    ) -> Runnable:
        """
        Model wrapper that returns outputs formatted to match the given schema.
        
        Args:
            schema (Union[Dict, Type[BaseModel]]): Output schema
            include_raw (bool): Whether to include raw output
            **kwargs: Additional parameters
        
        Returns:
            Runnable: Structured output runnable
        """
        # For now, return the model with basic output parsing
        # This can be enhanced with proper structured output parsing
        return self