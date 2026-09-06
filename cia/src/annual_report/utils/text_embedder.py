"""
MPNet Embedder Module for IBG Knowledge Base.

This module provides a wrapper for the MPNet-base-v2 embedding model,
handling embedding generation, token counting, and retry logic.
"""

import os
import re
import uuid
import time
import json
import ast
import logging
import urllib3
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import requests
from langchain_core.embeddings import Embeddings
from transformers import PreTrainedTokenizerFast

# Disable insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class MpnetEmbedder(Embeddings):
    """
    Wrapper for MPNet-base-v2 embedding models.
    
    This class provides:
    - Embedding generation via API
    - Token counting for text
    - Retry logic with exponential backoff
    - Batch processing support
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        app_code: str,
        project_name: str,
        max_retries: int = 10,
        timeout: int = 30,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the MPNet embedder.
        
        Args:
            endpoint (str): API endpoint for embedding service
            api_key (str): API key for authentication
            app_code (str): Application code for authorization
            project_name (str): Project name for identification
            max_retries (int): Maximum number of retry attempts
            timeout (int): Request timeout in seconds
            model_name (Optional[str]): Name of the embedding model
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.app_code = app_code
        self.project_name = project_name
        self.max_retries = max_retries
        self.timeout = timeout
        self.model_name = model_name or "mpnet-base-v2"
        
        # Headers for API requests
        self.headers = {
            "API_KEY": api_key,
            "APP_CODE": app_code,
            "Content-Type": "application/json",
        }
        
        # Initialize tokenizer
        self._init_tokenizer()
        
        logging.info(f"MpnetEmbedder initialized with model: {self.model_name}")
    
    def _init_tokenizer(self) -> None:
        """
        Initialize the tokenizer for MPNet.
        
        Loads the tokenizer from the tokenizer directory or falls back to
        a default configuration.
        """
        try:
            # Try to load from tokenizer directory
            current_dir = Path(__file__).parent
            tokenizer_path = os.path.join(
                current_dir,
                "tokenizer",
                "mpnet_tokenizer.py"
            )
            
            if os.path.exists(tokenizer_path):
                self.tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=tokenizer_path
                )
            else:
                # Fallback to using the transformers library
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-mpnet-base-v2"
                )
                
            logging.info("Tokenizer initialized successfully")
            
        except Exception as e:
            logging.warning(f"Failed to initialize tokenizer: {e}")
            logging.warning("Using fallback token count method")
            self.tokenizer = None
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Count the number of tokens in the text.
        
        Args:
            text (str): Input text
        
        Returns:
            int: Number of tokens in the text
        """
        if not text:
            return 0
        
        try:
            # Use the instance tokenizer if available
            if hasattr(MpnetEmbedder, '_tokenizer') and MpnetEmbedder._tokenizer:
                tokens = MpnetEmbedder._tokenizer.tokenize(text)
                return len(tokens)
            
            # Fallback to rough estimation
            # Average token count for English: ~4 characters per token
            return len(text) // 4
            
        except Exception as e:
            logging.warning(f"Token counting error: {e}")
            # Fallback to character-based estimation
            return len(text) // 4
    
    def embed(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Embed a list of strings.
        
        Args:
            texts (List[str]): List of strings to embed
            batch_size (Optional[int]): Batch size for processing
            show_progress (bool): Whether to show progress bar
        
        Returns:
            List[List[float]]: List of embeddings, one for each text
        
        Raises:
            Exception: If embedding generation fails after max retries
        """
        if not texts:
            return []
        
        # Set default batch size
        if batch_size is None:
            batch_size = 32
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts with retry logic.
        
        Args:
            texts (List[str]): List of texts to embed
        
        Returns:
            List[List[float]]: List of embeddings
        
        Raises:
            Exception: If embedding generation fails after max retries
        """
        if not texts:
            return []
        
        retry = 1
        sleep_time = 1
        
        while retry <= self.max_retries:
            try:
                # Prepare request data
                data = {
                    "utterance": texts,
                    "uniqueId": f"IDW-{self.project_name}-{uuid.uuid4()}",
                }
                
                # Make API request
                response = requests.post(
                    url=self.endpoint,
                    headers=self.headers,
                    json=data,
                    verify=False,
                    timeout=self.timeout
                )
                
                # Check response status
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Extract embeddings from response
                    answer = response_data.get("data", {}).get("answer", "{}")
                    embedding_data = ast.literal_eval(answer)
                    embeddings = embedding_data.get("embeds", [])
                    
                    if embeddings:
                        logging.debug(f"Successfully embedded {len(embeddings)} texts")
                        return embeddings
                    else:
                        logging.warning("No embeddings returned in response")
                        raise ValueError("No embeddings in response")
                else:
                    logging.warning(
                        f"API request failed with status {response.status_code}: "
                        f"{response.text}"
                    )
                    raise requests.RequestException(
                        f"Status: {response.status_code}, Response: {response.text}"
                    )
                    
            except Exception as e:
                logging.warning(
                    f"Embedding attempt {retry}/{self.max_retries} failed: {str(e)}"
                )
                
                if retry == self.max_retries:
                    logging.error("Maximum retries exceeded for embedding")
                    raise Exception(f"Exception raised in MpnetEmbedder.embed(): {str(e)}")
                
                # Exponential backoff
                time.sleep(sleep_time)
                sleep_time *= 2
                retry += 1
        
        return []
    
    def embed_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (Optional[int]): Batch size for processing
        
        Returns:
            List[List[float]]: List of embeddings, one for each text
        """
        return self.embed(texts, batch_size=batch_size)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text (str): Text to embed
        
        Returns:
            List[float]: Embedding for the text
        """
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else []
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts (alias for embed).
        
        Args:
            texts (List[str]): List of texts to embed
        
        Returns:
            List[List[float]]: List of embeddings
        """
        return self.embed(texts)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            int: Embedding dimension (768 for MPNet-base-v2)
        """
        return 768
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding.
        
        Args:
            text (str): Input text
        
        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters if needed
        # text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text
    
    def batch_embed_with_progress(
        self,
        texts: List[str],
        batch_size: int = 32,
        desc: str = "Embedding"
    ) -> List[List[float]]:
        """
        Embed texts with progress bar.
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size for processing
            desc (str): Description for progress bar
        
        Returns:
            List[List[float]]: List of embeddings
        """
        try:
            from tqdm import tqdm
            
            all_embeddings = []
            
            for i in tqdm(
                range(0, len(texts), batch_size),
                desc=desc,
                unit="batch"
            ):
                batch = texts[i:i + batch_size]
                embeddings = self.embed(batch)
                all_embeddings.extend(embeddings)
            
            return all_embeddings
            
        except ImportError:
            # Fallback if tqdm is not installed
            logging.warning("tqdm not installed, using simple batch processing")
            return self.embed(texts, batch_size=batch_size)


# Convenience functions
def create_embedder(config: Dict[str, Any]) -> MpnetEmbedder:
    """
    Create an MPNet embedder from configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    
    Returns:
        MpnetEmbedder: Configured embedder instance
    """
    embedder_config = config.get("text_embedder", {})
    mpnet_config = embedder_config.get("mpnet_api", {})
    
    return MpnetEmbedder(
        endpoint=mpnet_config.get("endpoint"),
        api_key=mpnet_config.get("api_key"),
        app_code=embedder_config.get("app_code"),
        project_name=embedder_config.get("project_name"),
        max_retries=mpnet_config.get("max_retries", 10),
        timeout=mpnet_config.get("timeout", 30),
        model_name=embedder_config.get("model_name", "mpnet-base-v2"),
    )


def get_embedding_similarity(
    embedding1: List[float],
    embedding2: List[float]
) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1 (List[float]): First embedding
        embedding2 (List[float]): Second embedding
    
    Returns:
        float: Cosine similarity score
    """
    import math
    
    if not embedding1 or not embedding2:
        return 0.0
    
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in embedding1))
    magnitude2 = math.sqrt(sum(b * b for b in embedding2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        "text_embedder": {
            "mpnet_api": {
                "endpoint": "http://localhost:8080/embed",
                "api_key": "your-api-key",
                "max_retries": 5,
                "timeout": 30,
            },
            "app_code": "your-app-code",
            "project_name": "ibg-kb",
            "model_name": "mpnet-base-v2",
        }
    }
    
    # Create embedder
    embedder = create_embedder(config)
    
    # Embed texts
    texts = [
        "This is the first document.",
        "This is the second document.",
        "And this is the third document.",
    ]
    
    embeddings = embedder.embed(texts)
    
    print(f"Number of texts: {len(texts)}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    
    # Test query embedding
    query = "What is this document about?"
    query_embedding = embedder.embed_query(query)
    print(f"Query embedding dimension: {len(query_embedding)}")
    
    # Calculate similarity
    if embeddings:
        similarity = get_embedding_similarity(query_embedding, embeddings[0])
        print(f"Similarity with first document: {similarity:.4f}")