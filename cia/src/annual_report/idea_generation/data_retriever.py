"""
Data Retriever Module for IBG Knowledge Base.

This module handles information retrieval from Elasticsearch, including
semantic search using vector embeddings and metadata filtering.
"""

import os
import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union

from langchain.docstore.document import Document
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, ConnectionError

from ..utils import MpnetEmbedder, connect_to_elasticsearch


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataRetriever:
    """
    A class that handles different information retrieval methods.
    
    Supports:
    - Vector similarity search using embeddings
    - Metadata filtering (GCIN, issue date, etc.)
    - Hybrid search combining vector and keyword search
    """
    
    def __init__(
        self,
        query: str,
        config: dict,
        gcin: str = "",
        embedding_model_name: Optional[str] = None,
    ):
        """
        Initialize the DataRetriever instance.
        
        Args:
            query (str): User query for retrieval
            config (dict): Configuration dictionary
            gcin (str): Unique identifier of the company (optional)
            embedding_model_name (Optional[str]): Name of embedding model to use
        """
        self.query = query
        self.config = config
        self.gcin = gcin
        self.max_retries = 10
        
        # Initialize embedder
        embedding_config = self.config.get("text_embedder", {})
        mpnet_config = embedding_config.get("mpnet_api", {})
        
        self.embedder = MpnetEmbedder(
            endpoint=mpnet_config.get("endpoint"),
            api_key=mpnet_config.get("api_key"),
            app_code=embedding_config.get("app_code"),
            project_name=embedding_config.get("project_name"),
            model_name=embedding_model_name,
        )
        
        # Connect to Elasticsearch
        es_config = self.config["data_retriever"]["elasticsearch"]
        
        self.es_instance, self.es_store_instance = connect_to_elasticsearch(
            hosts=es_config["hosts"],
            basic_auth=(
                es_config["basic_auth"]["username"],
                es_config["basic_auth"]["password"],
            ),
            verify_certs=es_config.get("verify_certs", False),
            index_name=self.config["data_retriever"]["index_name"],
            embedder=self.embedder
        )
        
        logging.info(f"DataRetriever initialized with GCIN: {self.gcin}")
    
    def search_elk(
        self,
        query: dict,
        size: int = 10
    ) -> dict:
        """
        Search Elasticsearch with the given query.
        
        Args:
            query (dict): Elasticsearch query body
            size (int): Number of results to return
        
        Returns:
            dict: Search response from Elasticsearch
        
        Raises:
            Exception: If maximum retries are exceeded
        """
        retry = 1
        
        while retry <= self.max_retries:
            try:
                logging.info(f"Searching ELK (attempt {retry}/{self.max_retries})...")
                
                response = self.es_instance.search(
                    index=self.config["data_retriever"]["index_name"],
                    body=query,
                    size=size,
                )
                
                logging.info(f"Search completed successfully.")
                return response
                
            except Exception as e:
                logging.warning(f"Search attempt {retry} failed: {str(e)}")
                logging.debug(f"Traceback: {traceback.format_exc()}")
                
                retry += 1
                if retry > self.max_retries:
                    logging.error("Maximum retries exceeded for search_elk()")
                    raise Exception("Exception raised in DataRetriever.search_elk(): "
                                  f"Reached maximum retries ({self.max_retries})")
        
        return {}
    
    def get_max_field_value(self, field_name: str) -> int:
        """
        Get the maximum value of a field for the given GCIN.
        
        Args:
            field_name (str): The name of the field (e.g., "metadata.issue_date")
        
        Returns:
            int: Maximum value of the field
        
        Raises:
            Exception: If the query fails
        """
        try:
            # Build query to get max value
            query = {
                "query": {
                    "match_phrase": {
                        "metadata.gcin.keyword": self.gcin
                    }
                },
                "aggs": {
                    "max_value": {
                        "max": {
                            "field": field_name
                        }
                    }
                },
                "size": 0  # Only return aggregation results
            }
            
            response = self.es_instance.search(
                index=self.config["data_retriever"]["index_name"],
                body=query
            )
            
            max_value = int(response.get("aggregations", {})
                           .get("max_value", {})
                           .get("value", 0))
            
            logging.info(f"Max {field_name} for GCIN {self.gcin}: {max_value}")
            return max_value
            
        except Exception as e:
            logging.error(f"Failed to get max field value for {field_name}: {e}")
            raise
    
    def build_vector_search_query(
        self,
        query_vector: List[float],
        num_candidates: int = 100,
        knn_k: int = 10
    ) -> dict:
        """
        Build a vector similarity search query.
        
        Args:
            query_vector (List[float]): Query embedding vector
            num_candidates (int): Number of candidates per shard
            knn_k (int): Number of nearest neighbors to return
        
        Returns:
            dict: Elasticsearch query body for vector search
        """
        query_template = {
            "knn": {
                "field": "vector",
                "query_vector": query_vector,
                "k": knn_k,
                "num_candidates": num_candidates,
            },
            "sort": [{"_score": {"order": "desc"}}]
        }
        
        # Add metadata filter if GCIN is provided
        if self.gcin:
            max_issue_date = self.get_max_field_value("metadata.issue_date")
            query_template["query"] = {
                "bool": {
                    "must": [],
                    "filter": [
                        {"match_phrase": {"metadata.gcin.keyword": self.gcin}},
                        {"match_phrase": {"metadata.issue_date": str(max_issue_date)}}
                    ],
                    "should": [],
                    "must_not": []
                }
            }
        else:
            query_template["query"] = {"match_all": {}}
        
        return query_template
    
    def build_hybrid_search_query(
        self,
        query_vector: List[float],
        text_query: str,
        num_candidates: int = 100,
        knn_k: int = 10
    ) -> dict:
        """
        Build a hybrid search query combining vector and text search.
        
        Args:
            query_vector (List[float]): Query embedding vector
            text_query (str): Text query for keyword search
            num_candidates (int): Number of candidates per shard
            knn_k (int): Number of nearest neighbors to return
        
        Returns:
            dict: Elasticsearch query body for hybrid search
        """
        query_template = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "description": {
                                    "query": text_query,
                                    "boost": 1.0
                                }
                            }
                        },
                        {
                            "match": {
                                "metadata.summary": {
                                    "query": text_query,
                                    "boost": 0.5
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "knn": {
                "field": "vector",
                "query_vector": query_vector,
                "k": knn_k,
                "num_candidates": num_candidates,
                "boost": 2.0
            }
        }
        
        # Add metadata filter if GCIN is provided
        if self.gcin:
            max_issue_date = self.get_max_field_value("metadata.issue_date")
            query_template["query"]["bool"]["filter"] = [
                {"match_phrase": {"metadata.gcin.keyword": self.gcin}},
                {"match_phrase": {"metadata.issue_date": str(max_issue_date)}}
            ]
        
        return query_template
    
    def retrieve_from_elasticsearch(
        self,
        k: int = 10,
        num_candidates: int = 100,
        use_hybrid: bool = False
    ) -> List[Document]:
        """
        Retrieve chunks from Elasticsearch using vector similarity search.
        
        Args:
            k (int): Number of chunks to retrieve
            num_candidates (int): Number of candidates for vector search
            use_hybrid (bool): If True, use hybrid search (vector + text)
        
        Returns:
            List[Document]: List of retrieved documents with metadata
        
        Raises:
            Exception: If retrieval fails after max retries
        """
        retry = 1
        
        while retry <= self.max_retries:
            try:
                logging.info(f"Retrieving from Elasticsearch (attempt {retry}/{self.max_retries})...")
                
                # Generate query embedding
                query_vector = self.embedder.embed_query(self.query)
                
                # Build search query
                if use_hybrid:
                    search_query = self.build_hybrid_search_query(
                        query_vector=query_vector,
                        text_query=self.query,
                        num_candidates=num_candidates,
                        knn_k=k
                    )
                else:
                    search_query = self.build_vector_search_query(
                        query_vector=query_vector,
                        num_candidates=num_candidates,
                        knn_k=k
                    )
                
                # Execute search
                response = self.search_elk(query=search_query, size=k)
                
                # Parse results
                documents = []
                hits = response.get("hits", {}).get("hits", [])
                
                for hit in hits:
                    source = hit.get("_source", {})
                    score = hit.get("_score", 0.0)
                    
                    # Create document with metadata
                    doc = Document(
                        page_content=source.get("description", ""),
                        metadata={
                            "gcin": source.get("metadata", {}).get("gcin"),
                            "file_type": source.get("metadata", {}).get("file_type"),
                            "file_name": source.get("metadata", {}).get("file_name"),
                            "data_type": source.get("metadata", {}).get("data_type"),
                            "page_number": source.get("metadata", {}).get("page_number"),
                            "chunk_index": source.get("metadata", {}).get("chunk_index"),
                            "issue_date": source.get("metadata", {}).get("issue_date"),
                            "ingestion_date": source.get("metadata", {}).get("ingestion_date"),
                            "summary": source.get("metadata", {}).get("summary"),
                            "hyperlink": source.get("metadata", {}).get("hyperlink"),
                            "score": score,
                        }
                    )
                    documents.append(doc)
                
                logging.info(f"Retrieved {len(documents)} documents from Elasticsearch")
                return documents
                
            except Exception as e:
                logging.warning(f"Retrieval attempt {retry} failed: {str(e)}")
                logging.debug(f"Traceback: {traceback.format_exc()}")
                
                retry += 1
                if retry > self.max_retries:
                    logging.error("Maximum retries exceeded for retrieve_from_elasticsearch()")
                    raise Exception("Exception raised in DataRetriever.retrieve_from_elasticsearch(): "
                                  f"Reached maximum retries ({self.max_retries})")
        
        return []
    
    def retrieve_by_gcin_and_date(
        self,
        gcin: str,
        issue_date: Optional[int] = None,
        k: int = 10
    ) -> List[Document]:
        """
        Retrieve documents by GCIN and optionally by issue date.
        
        Args:
            gcin (str): GCIN of the company
            issue_date (Optional[int]): Issue date filter
            k (int): Number of documents to retrieve
        
        Returns:
            List[Document]: List of retrieved documents
        """
        try:
            # Build filter query
            filter_conditions = [
                {"match_phrase": {"metadata.gcin.keyword": gcin}}
            ]
            
            if issue_date:
                filter_conditions.append(
                    {"match_phrase": {"metadata.issue_date": str(issue_date)}}
                )
            
            query = {
                "query": {
                    "bool": {
                        "must": [],
                        "filter": filter_conditions,
                        "should": [],
                        "must_not": []
                    }
                },
                "sort": [{"_score": {"order": "desc"}}],
                "size": k
            }
            
            # Execute search
            response = self.search_elk(query=query, size=k)
            
            # Parse results
            documents = []
            hits = response.get("hits", {}).get("hits", [])
            
            for hit in hits:
                source = hit.get("_source", {})
                doc = Document(
                    page_content=source.get("description", ""),
                    metadata=source.get("metadata", {})
                )
                documents.append(doc)
            
            logging.info(f"Retrieved {len(documents)} documents for GCIN {gcin}")
            return documents
            
        except Exception as e:
            logging.error(f"Failed to retrieve documents for GCIN {gcin}: {e}")
            raise
    
    def get_total_document_count(self) -> int:
        """
        Get the total number of documents in the index.
        
        Returns:
            int: Total document count
        """
        try:
            response = self.es_instance.count(
                index=self.config["data_retriever"]["index_name"]
            )
            return response.get("count", 0)
        except Exception as e:
            logging.error(f"Failed to get document count: {e}")
            return 0
    
    def get_document_count_by_gcin(self, gcin: str) -> int:
        """
        Get the number of documents for a specific GCIN.
        
        Args:
            gcin (str): GCIN to filter by
        
        Returns:
            int: Document count for the GCIN
        """
        try:
            query = {
                "query": {
                    "match_phrase": {
                        "metadata.gcin.keyword": gcin
                    }
                }
            }
            
            response = self.es_instance.count(
                index=self.config["data_retriever"]["index_name"],
                body=query
            )
            return response.get("count", 0)
            
        except Exception as e:
            logging.error(f"Failed to get document count for GCIN {gcin}: {e}")
            return 0


# Convenience function for quick retrieval
def retrieve_documents(
    query: str,
    config: dict,
    gcin: str = "",
    k: int = 10,
    use_hybrid: bool = False
) -> List[Document]:
    """
    Quick retrieval function for getting documents from Elasticsearch.
    
    Args:
        query (str): User query
        config (dict): Configuration dictionary
        gcin (str): GCIN filter (optional)
        k (int): Number of documents to retrieve
        use_hybrid (bool): Use hybrid search if True
    
    Returns:
        List[Document]: List of retrieved documents
    """
    retriever = DataRetriever(query=query, config=config, gcin=gcin)
    return retriever.retrieve_from_elasticsearch(k=k, use_hybrid=use_hybrid)


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = {
        "text_embedder": {
            "mpnet_api": {
                "endpoint": "http://localhost:8080",
                "api_key": "your-api-key"
            },
            "app_code": "your-app-code",
            "project_name": "ibg-kb"
        },
        "data_retriever": {
            "elasticsearch": {
                "hosts": ["localhost:9200"],
                "basic_auth": {
                    "username": "elastic",
                    "password": "password"
                },
                "verify_certs": False
            },
            "index_name": "ibg_kb_documents"
        }
    }
    
    # Initialize retriever
    retriever = DataRetriever(
        query="What are the financial highlights?",
        config=config,
        gcin="123456789"
    )
    
    # Retrieve documents
    documents = retriever.retrieve_from_elasticsearch(k=5)
    
    # Display results
    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")