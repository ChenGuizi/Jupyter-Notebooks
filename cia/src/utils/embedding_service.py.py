"""
Embedding Service Module for IBG Knowledge Base.

This module provides functionality for calling embedding services,
generating vector embeddings, and adding them to DataFrames for
use in vector search and retrieval systems.
"""

import json
import uuid
import logging
import numpy as np
import pandas as pd
import requests
from typing import List, Optional, Tuple, Dict, Any, Union
from sklearn.preprocessing import normalize

# Disable SSL warnings for internal services
requests.packages.urllib3.disable_warnings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class EmbeddingService:
    """
    Service class for handling embedding generation and vector operations.
    
    Features:
    - Call embedding service APIs
    - Generate vector embeddings for text
    - Add embeddings to DataFrames
    - Batch processing with configurable step sizes
    - Vector normalization
    """
    
    def __init__(
        self,
        service_id: Optional[str] = None,
        api_key: Optional[str] = None,
        app_code: Optional[str] = None,
        username: Optional[str] = None
    ):
        """
        Initialize the EmbeddingService instance.
        
        Args:
            service_id (Optional[str]): Service identifier in format "url-verb"
            api_key (Optional[str]): API key for authentication
            app_code (Optional[str]): Application code
            username (Optional[str]): Username for service
        """
        self.service_id = service_id
        self.api_key = api_key
        self.app_code = app_code or "ING"
        self.username = username
        self.url = None
        self.verb = None
        
        if service_id:
            self.url, self.verb = service_id.split("-")
        
        self.headers = {
            "APP_CODE": self.app_code,
            "Content-Type": "application/json",
        }
        
        if api_key:
            self.headers["API_KEY"] = api_key
        
        logging.info(f"EmbeddingService initialized with app_code: {self.app_code}")
    
    def call_embedding_service(
        self,
        texts: List[str],
        service_id: Optional[str] = None,
        batch_size: int = 100
    ) -> np.ndarray:
        """
        Call the embedding service to get vector embeddings for texts.
        
        Args:
            texts (List[str]): List of text strings to embed
            service_id (Optional[str]): Service identifier override
            batch_size (int): Batch size for processing
        
        Returns:
            np.ndarray: Array of embeddings with shape (n_texts, embedding_dim)
        
        Raises:
            Exception: If embedding service call fails
        """
        if not texts:
            return np.array([])
        
        # Use service_id from parameter or instance
        if service_id:
            url, verb = service_id.split("-")
        else:
            url = self.url
            verb = self.verb
        
        if not url:
            raise ValueError("Service URL not configured")
        
        # Generate unique ID for this request
        unique_id = f"IDW-ACCOUNT-PLAN-UID-{uuid.uuid4()}"
        
        # Prepare request data
        data = {
            "utterance": texts,
            "uniqueId": unique_id,
        }
        
        if self.username:
            data["username"] = self.username
        
        # Make API request
        try:
            response = requests.post(
                url=url,
                headers=self.headers,
                json=data,
                verify=False,
                timeout=60
            )
            
            if response.status_code != 200:
                logging.error(f"Embedding service error: {response.status_code}")
                logging.error(f"Response: {response.text[:500]}")
                raise Exception(f"Embedding service returned status {response.status_code}")
            
            # Parse response
            response_data = response.json()
            
            # Extract embeddings from response
            # Expected format: {"data": {"answer": "embeds: [0.1, 0.2, ...]"}}
            answer = response_data.get("data", {}).get("answer", "")
            
            if not answer:
                logging.warning("Empty answer from embedding service")
                return np.array([])
            
            # Parse embeddings
            embeddings = self._parse_embeddings(answer)
            
            logging.info(f"Successfully generated {len(embeddings)} embeddings")
            return np.array(embeddings, dtype=np.float64)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error in embedding service: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            raise
    
    def _parse_embeddings(self, answer: str) -> List[List[float]]:
        """
        Parse embeddings from service response.
        
        Args:
            answer (str): Raw answer string from service
        
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            # Try to parse as JSON first
            if answer.startswith('[') or answer.startswith('{'):
                data = json.loads(answer)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "embeds" in data:
                    return data["embeds"]
            
            # Try to extract from string format
            # Format: "embeds: [0.1;0.2;0.3]"
            if ":" in answer:
                parts = answer.split(":", 1)
                if len(parts) > 1:
                    embed_str = parts[1].strip()
                    # Remove brackets if present
                    embed_str = embed_str.strip('[]')
                    # Split by semicolon for list of vectors
                    vector_strings = embed_str.split(';')
                    
                    embeddings = []
                    for vec_str in vector_strings:
                        try:
                            # Try to parse as list of floats
                            vec = json.loads(vec_str.strip())
                            if isinstance(vec, list):
                                embeddings.append([float(x) for x in vec])
                        except json.JSONDecodeError:
                            # Try parsing as comma-separated values
                            try:
                                vec = [float(x.strip()) for x in vec_str.split(',')]
                                embeddings.append(vec)
                            except ValueError:
                                continue
                    
                    if embeddings:
                        return embeddings
            
            logging.warning(f"Could not parse embeddings from: {answer[:200]}")
            return []
            
        except Exception as e:
            logging.error(f"Error parsing embeddings: {e}")
            return []
    
    def add_vector_to_dataframe(
        self,
        data_frame: pd.DataFrame,
        text_column: str = "text",
        embedding_dim: int = 768,
        batch_size: int = 1000,
        normalize_vectors: bool = True
    ) -> pd.DataFrame:
        """
        Add vector embeddings to a DataFrame.
        
        Args:
            data_frame (pd.DataFrame): Input DataFrame with text column
            text_column (str): Name of column containing text
            embedding_dim (int): Dimension of embeddings
            batch_size (int): Batch size for processing
            normalize_vectors (bool): Whether to normalize embeddings
        
        Returns:
            pd.DataFrame: DataFrame with added 'text_embedding' column
        
        Example:
            >>> df = pd.DataFrame({"text": ["Hello world", "Another text"]})
            >>> service = EmbeddingService()
            >>> result = service.add_vector_to_dataframe(df)
            >>> print(result.columns)  # Contains 'text_embedding'
        """
        if data_frame.empty:
            logging.warning("Empty DataFrame provided")
            return data_frame
        
        # Create copy to avoid modifying original
        result_df = data_frame.copy()
        
        # Calculate number of batches
        num_rows = result_df.shape[0]
        num_batches = int(np.ceil(num_rows / batch_size))
        
        logging.info(f"Processing {num_rows} rows in {num_batches} batches")
        
        # Initialize embedding column
        result_df["text_embedding"] = [None] * num_rows
        
        # Process in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_rows)
            
            logging.info(f"Processing batch {batch_idx + 1}/{num_batches}")
            
            # Get texts for this batch
            batch_df = result_df.iloc[start_idx:end_idx]
            texts = batch_df[text_column].tolist()
            
            # Generate embeddings
            try:
                embeddings = self.call_embedding_service(texts)
                
                if len(embeddings) == 0:
                    logging.warning(f"Empty embeddings for batch {batch_idx}")
                    continue
                
                # Reshape if needed
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                
                # Normalize vectors if requested
                if normalize_vectors:
                    embeddings = self.normalize_vectors(embeddings)
                
                # Ensure correct dimension
                if embeddings.shape[1] != embedding_dim:
                    logging.warning(
                        f"Embedding dimension {embeddings.shape[1]} "
                        f"does not match expected {embedding_dim}"
                    )
                
                # Convert to list for storage
                vector_list = [vec.tolist() for vec in embeddings]
                
                # Assign to DataFrame
                result_df.iloc[start_idx:end_idx, 
                              result_df.columns.get_loc("text_embedding")] = vector_list
                
                logging.info(f"Added embeddings for {len(vector_list)} rows")
                
            except Exception as e:
                logging.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        return result_df
    
    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length.
        
        Args:
            vectors (np.ndarray): Array of vectors
        
        Returns:
            np.ndarray: Normalized vectors
        """
        if vectors.size == 0:
            return vectors
        
        # Handle single vector case
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            return vectors / norm if norm > 0 else vectors
        
        # Handle batch of vectors
        return normalize(vectors, norm='l2', axis=1)
    
    def generate_random_vectors(
        self,
        num_vectors: int,
        embedding_dim: int = 768
    ) -> np.ndarray:
        """
        Generate random vectors for testing purposes.
        
        Args:
            num_vectors (int): Number of vectors to generate
            embedding_dim (int): Dimension of each vector
        
        Returns:
            np.ndarray: Array of random vectors
        """
        vectors = np.random.randn(num_vectors, embedding_dim)
        return self.normalize_vectors(vectors)
    
    def cosine_similarity(
        self,
        vector1: Union[List[float], np.ndarray],
        vector2: Union[List[float], np.ndarray]
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector1 (Union[List[float], np.ndarray]): First vector
            vector2 (Union[List[float], np.ndarray]): Second vector
        
        Returns:
            float: Cosine similarity score
        
        Example:
            >>> service = EmbeddingService()
            >>> v1 = [1.0, 0.0]
            >>> v2 = [0.0, 1.0]
            >>> score = service.cosine_similarity(v1, v2)
            >>> print(score)  # 0.0
        """
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        # Normalize vectors
        v1 = self.normalize_vectors(v1) if v1.ndim == 1 else v1
        v2 = self.normalize_vectors(v2) if v2.ndim == 1 else v2
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)


# Convenience functions

def call_embedding_service(
    texts: List[str],
    service_id: str,
    api_key: Optional[str] = None,
    app_code: str = "ING",
    username: Optional[str] = None
) -> np.ndarray:
    """
    Convenience function to call embedding service.
    
    Args:
        texts (List[str]): List of texts to embed
        service_id (str): Service identifier in format "url-verb"
        api_key (Optional[str]): API key
        app_code (str): Application code
        username (Optional[str]): Username
    
    Returns:
        np.ndarray: Array of embeddings
    """
    service = EmbeddingService(
        service_id=service_id,
        api_key=api_key,
        app_code=app_code,
        username=username
    )
    return service.call_embedding_service(texts)


def add_vector_to_df(
    data_frame: pd.DataFrame,
    text_column: str = "text",
    embedding_dim: int = 768,
    batch_size: int = 1000,
    normalize_vectors: bool = True,
    service_id: Optional[str] = None,
    api_key: Optional[str] = None,
    app_code: str = "ING"
) -> pd.DataFrame:
    """
    Convenience function to add vectors to DataFrame.
    
    Args:
        data_frame (pd.DataFrame): Input DataFrame
        text_column (str): Column containing text
        embedding_dim (int): Embedding dimension
        batch_size (int): Batch size for processing
        normalize_vectors (bool): Whether to normalize vectors
        service_id (Optional[str]): Service identifier
        api_key (Optional[str]): API key
        app_code (str): Application code
    
    Returns:
        pd.DataFrame: DataFrame with embeddings
    """
    service = EmbeddingService(
        service_id=service_id,
        api_key=api_key,
        app_code=app_code
    )
    
    return service.add_vector_to_dataframe(
        data_frame=data_frame,
        text_column=text_column,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        normalize_vectors=normalize_vectors
    )


# Example usage
if __name__ == "__main__":
    # Create sample DataFrame
    sample_df = pd.DataFrame({
        "text": [
            "This is the first document about revenue growth.",
            "This document discusses market expansion strategies.",
            "This text covers financial risk assessment.",
            "This is about employee hiring and retention.",
        ]
    })
    
    print("Sample DataFrame:")
    print(sample_df)
    print("\n" + "="*50 + "\n")
    
    # Initialize service
    service = EmbeddingService(
        service_id="http://localhost:8080-POST",
        app_code="ING"
    )
    
    # Generate random embeddings for demonstration
    embeddings = service.generate_random_vectors(
        num_vectors=len(sample_df),
        embedding_dim=768
    )
    
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Add vectors to DataFrame
    result_df = sample_df.copy()
    result_df["text_embedding"] = [emb.tolist() for emb in embeddings]
    
    print("\nResult DataFrame with embeddings:")
    print(f"Shape: {result_df.shape}")
    print(f"Columns: {result_df.columns.tolist()}")
    
    # Test cosine similarity
    if len(embeddings) >= 2:
        similarity = service.cosine_similarity(embeddings[0], embeddings[1])
        print(f"\nCosine similarity between first two vectors: {similarity:.4f}")