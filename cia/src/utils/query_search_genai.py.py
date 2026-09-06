"""
Query Search Module for GenAI.

This module handles semantic search queries using Elasticsearch with vector embeddings,
including cosine similarity scoring and configurable search parameters.
"""

import os
import re
import json
import uuid
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import normalize

from src.utils import embedder


class QuerySearchGenAI:
    """
    Handles semantic search queries using vector embeddings and Elasticsearch.
    
    Features:
    - Vector-based similarity search
    - Cosine similarity scoring
    - Configurable search parameters
    - Retry logic for API calls
    - Result deduplication
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the QuerySearchGenAI instance.
        
        Args:
            config_path (Optional[str]): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.embedder = embedder  # Imported embedder instance
        self._init_api_config()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from properties file.
        
        Args:
            config_path (Optional[str]): Path to configuration file
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        config = {}
        
        # Default config paths
        file_path = "/app/ibgdw/config/env-config.properties"
        file_path_2 = "/app/ibgdw/certs/secrets.properties"
        
        try:
            # Try first config file
            if os.path.exists(file_path) and os.path.isfile(file_path):
                print(f"File '{file_path}' exists.")
                with open(file_path, 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            config[key] = value
            
            # Try second config file
            if os.path.exists(file_path_2) and os.path.isfile(file_path_2):
                print(f"File '{file_path_2}' exists.")
                with open(file_path_2, 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            config[key] = value
            
            print("Using local properties file.")
            
        except Exception as e:
            print(f"Error loading config: {e}")
        
        return config
    
    def _init_api_config(self) -> None:
        """
        Initialize API configuration from config.
        """
        # Get GenAI API configuration
        self.genai_url = self.config.get("genai.url")
        
        # Get LLM API key
        self.llm_apikey = self.config.get("server.claude.apikey")
        
        # Set headers
        self.headers = {
            "app_code": "IBGN",
            "Content-Type": "application/json",
        }
        
        print(f"GenAI URL: {self.genai_url}")
        print(f"LLM API Key: {self.llm_apikey[:10]}...")
    
    def search_elasticsearch(
        self,
        index_name: str,
        query_vector: List[float],
        chunk_size: int = 10,
        source_fields: List[str] = None
    ) -> Dict[str, Any]:
        """
        Search Elasticsearch using vector similarity.
        
        Args:
            index_name (str): Name of the Elasticsearch index
            query_vector (List[float]): Query embedding vector
            chunk_size (int): Number of results to return
            source_fields (List[str]): Fields to include in source
        
        Returns:
            Dict[str, Any]: Search response from Elasticsearch
        """
        if source_fields is None:
            source_fields = ["file_name", "text"]
        
        # Build query using script_score for cosine similarity
        query = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'text_embedding') + 1.0",
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
            },
            "_source": source_fields,
            "size": chunk_size
        }
        
        # Execute search
        response = self._search_elasticsearch(
            index_name=index_name,
            query=query
        )
        
        return response
    
    def _search_elasticsearch(
        self,
        index_name: str,
        query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute Elasticsearch search with retry logic.
        
        Args:
            index_name (str): Name of the Elasticsearch index
            query (Dict[str, Any]): Elasticsearch query
        
        Returns:
            Dict[str, Any]: Search response
        """
        # This would typically use an Elasticsearch client
        # For now, assuming we have an ES instance available
        try:
            # Placeholder for actual ES search
            # es_client = get_es_client()
            # response = es_client.search(index=index_name, body=query)
            # return response
            
            # For demonstration, return empty response
            return {
                "hits": {
                    "hits": []
                }
            }
            
        except Exception as e:
            print(f"Error searching Elasticsearch: {e}")
            return {
                "hits": {
                    "hits": []
                }
            }
    
    def search_and_extract(
        self,
        query_text: str,
        index_name: str,
        chunk_size: int = 10
    ) -> Tuple[List[str], List[str]]:
        """
        Search and extract relevant documents.
        
        Args:
            query_text (str): Search query text
            index_name (str): Name of the Elasticsearch index
            chunk_size (int): Number of results to return
        
        Returns:
            Tuple[List[str], List[str]]: (file_names, document_contents)
        """
        # Generate query vector
        query_vector = self.embedder.embed_query(query_text)
        
        # Search Elasticsearch
        response = self.search_elasticsearch(
            index_name=index_name,
            query_vector=query_vector,
            chunk_size=chunk_size
        )
        
        # Extract results
        file_list = []
        doc_list = []
        file_set = set()
        
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            file_name = source.get("file_name", "")
            doc_content = source.get("text", "")
            
            if file_name and file_name not in file_set:
                file_set.add(file_name)
                file_list.append(file_name)
            
            if doc_content:
                doc_list.append(doc_content)
        
        return file_list, doc_list
    
    def genai_interact(self, prompt: str) -> str:
        """
        Interact with GenAI API with retry logic.
        
        Args:
            prompt (str): Prompt to send to GenAI
        
        Returns:
            str: Response from GenAI
        """
        url = self.genai_url
        headers = {
            "app_code": "IBGN",
            "Content-Type": "application/json",
        }
        
        # Prepare request body
        data = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
        
        retry = 0
        max_retries = 3
        
        while retry < max_retries:
            try:
                print(f"Calling GenAI URL: {url}")
                
                response = requests.post(
                    url=url,
                    json=data,
                    headers=headers,
                    verify=False
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("data", {}).get("answer", "")
                    return answer
                else:
                    print(f"GenAI API error: {response.status_code}")
                    retry += 1
                    
            except Exception as e:
                print(f"Error in GenAI interaction: {e}")
                retry += 1
                import time
                time.sleep(1)
        
        return ""
    
    def semantic_search(
        self,
        query: str,
        index_name: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search with cosine similarity.
        
        Args:
            query (str): Search query
            index_name (str): Elasticsearch index name
            top_k (int): Number of results
        
        Returns:
            List[Dict[str, Any]]: Search results with scores
        """
        # Get query embedding
        query_vector = self.embedder.embed_query(query)
        
        # Normalize vector
        query_vector = normalize([query_vector])[0].tolist()
        
        # Search with cosine similarity
        response = self.search_elasticsearch(
            index_name=index_name,
            query_vector=query_vector,
            chunk_size=top_k
        )
        
        # Process results
        results = []
        for hit in response.get("hits", {}).get("hits", []):
            # Calculate cosine similarity (score - 1.0)
            score = hit.get("_score", 0) - 1.0
            results.append({
                "score": score,
                "source": hit.get("_source", {}),
                "id": hit.get("_id", "")
            })
        
        return results
    
    def deduplicate_results(
        self,
        results: List[Dict[str, Any]],
        key_field: str = "file_name"
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate search results by a key field.
        
        Args:
            results (List[Dict[str, Any]]): Search results
            key_field (str): Field to use for deduplication
        
        Returns:
            List[Dict[str, Any]]: Deduplicated results
        """
        seen = set()
        deduplicated = []
        
        for result in results:
            key = result.get("source", {}).get(key_field, "")
            if key and key not in seen:
                seen.add(key)
                deduplicated.append(result)
        
        return deduplicated


# Global instance for convenience
_query_search_instance = None


def get_query_search_instance(config_path: Optional[str] = None) -> QuerySearchGenAI:
    """
    Get or create a QuerySearchGenAI instance.
    
    Args:
        config_path (Optional[str]): Path to configuration file
    
    Returns:
        QuerySearchGenAI: QuerySearchGenAI instance
    """
    global _query_search_instance
    if _query_search_instance is None:
        _query_search_instance = QuerySearchGenAI(config_path)
    return _query_search_instance


def search_query(
    query_text: str,
    index_name: str,
    chunk_size: int = 10
) -> Tuple[List[str], List[str]]:
    """
    Convenience function for searching queries.
    
    Args:
        query_text (str): Search query text
        index_name (str): Name of the Elasticsearch index
        chunk_size (int): Number of results to return
    
    Returns:
        Tuple[List[str], List[str]]: (file_names, document_contents)
    """
    searcher = get_query_search_instance()
    return searcher.search_and_extract(
        query_text=query_text,
        index_name=index_name,
        chunk_size=chunk_size
    )


# Example usage
if __name__ == "__main__":
    # Initialize searcher
    searcher = QuerySearchGenAI()
    
    # Example query
    query = "What is the revenue growth outlook?"
    index_name = "annual_reports"
    chunk_size = 5
    
    # Perform search
    file_names, doc_contents = searcher.search_and_extract(
        query_text=query,
        index_name=index_name,
        chunk_size=chunk_size
    )
    
    print(f"Found {len(file_names)} files")
    print(f"Found {len(doc_contents)} documents")
    
    # Semantic search
    results = searcher.semantic_search(
        query=query,
        index_name=index_name,
        top_k=10
    )
    
    print(f"Semantic search returned {len(results)} results")
    
    # Deduplicate results
    deduplicated = searcher.deduplicate_results(results)
    print(f"Deduplicated to {len(deduplicated)} results")