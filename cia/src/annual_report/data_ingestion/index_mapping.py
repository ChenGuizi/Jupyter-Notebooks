"""
Elasticsearch Index Mapping for IBG Knowledge Base.

This module defines the index mapping schema for the Elasticsearch index used
by the IBG Knowledge Base data ingestion pipeline. It includes mappings for
document content, metadata fields, and vector embeddings for similarity search.
"""

import logging
from typing import Dict, Any, Optional

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError


# Index Mapping Definition
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            # Document content fields
            "description": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            "message": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            
            # Vector field for embeddings (similarity search)
            "vector": {
                "type": "dense_vector",
                "index": True,
                "similarity": "cosine",
                "index_options": {
                    "type": "int8_hnsw",
                    "m": 16,
                    "ef_construction": 100
                }
            },
            
            # Metadata nested object
            "metadata": {
                "properties": {
                    # GCIN (Global Company Identification Number)
                    "gcin": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    
                    # File ID
                    "file_id": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    
                    # File type (e.g., "pdf", "docx")
                    "file_type": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    
                    # File name
                    "file_name": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    
                    # Document title
                    "title": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    
                    # Data type (e.g., "text", "table", "image")
                    "data_type": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    
                    # Source type (e.g., "annual_report")
                    "source_type": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    
                    # Page number
                    "page_number": {
                        "type": "long"
                    },
                    
                    # Total page count
                    "page_count": {
                        "type": "long"
                    },
                    
                    # Chunk index within page
                    "chunk_index": {
                        "type": "long"
                    },
                    
                    # Issue date (as epoch timestamp)
                    "issue_date": {
                        "type": "long"
                    },
                    
                    # Ingestion date (as epoch timestamp)
                    "ingestion_date": {
                        "type": "long"
                    },
                    
                    # Page summary (generated by LLM)
                    "summary": {
                        "type": "text"
                    },
                    
                    # Hyperlink to original document
                    "hyperlink": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    }
                }
            }
        }
    }
}


def get_index_mapping() -> Dict[str, Any]:
    """
    Get the index mapping configuration.
    
    Returns:
        Dict[str, Any]: Index mapping dictionary
    """
    return INDEX_MAPPING


def create_index(
    es_client: Elasticsearch,
    index_name: str,
    mapping: Optional[Dict[str, Any]] = None,
    force_recreate: bool = False
) -> bool:
    """
    Create the Elasticsearch index with the specified mapping.
    
    Args:
        es_client (Elasticsearch): Elasticsearch client instance
        index_name (str): Name of the index to create
        mapping (Optional[Dict[str, Any]]): Index mapping (uses default if None)
        force_recreate (bool): If True, delete existing index and recreate
    
    Returns:
        bool: True if index was created successfully, False otherwise
    
    Raises:
        Exception: If index creation fails
    """
    try:
        mapping = mapping or INDEX_MAPPING
        
        # Check if index exists
        if es_client.indices.exists(index=index_name):
            logging.info(f"Index '{index_name}' already exists.")
            
            if force_recreate:
                logging.info(f"Force recreating index '{index_name}'...")
                # Delete existing index
                es_client.indices.delete(index=index_name, ignore=[400, 404])
                logging.info(f"Deleted index '{index_name}'.")
            else:
                logging.info(f"Keeping existing index '{index_name}'.")
                return True
        
        # Create index with mapping
        response = es_client.indices.create(
            index=index_name,
            body=mapping,
            ignore=[400]  # Ignore bad request errors
        )
        
        logging.info(f"Successfully created index '{index_name}': {response}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to create index '{index_name}': {e}")
        raise


def get_existing_mapping(
    es_client: Elasticsearch,
    index_name: str
) -> Optional[Dict[str, Any]]:
    """
    Get the existing mapping for an index.
    
    Args:
        es_client (Elasticsearch): Elasticsearch client instance
        index_name (str): Name of the index
    
    Returns:
        Optional[Dict[str, Any]]: Index mapping if exists, None otherwise
    """
    try:
        if es_client.indices.exists(index=index_name):
            response = es_client.indices.get_mapping(index=index_name)
            return response.get(index_name, {}).get("mappings", {})
        else:
            logging.info(f"Index '{index_name}' does not exist.")
            return None
    except Exception as e:
        logging.error(f"Failed to get mapping for index '{index_name}': {e}")
        return None


def update_mapping(
    es_client: Elasticsearch,
    index_name: str,
    new_properties: Dict[str, Any]
) -> bool:
    """
    Update the mapping with new properties.
    
    Args:
        es_client (Elasticsearch): Elasticsearch client instance
        index_name (str): Name of the index
        new_properties (Dict[str, Any]): New properties to add
    
    Returns:
        bool: True if mapping was updated successfully
    
    Raises:
        Exception: If mapping update fails
    """
    try:
        if not es_client.indices.exists(index=index_name):
            logging.error(f"Index '{index_name}' does not exist.")
            return False
        
        # Update mapping
        response = es_client.indices.put_mapping(
            index=index_name,
            body={"properties": new_properties}
        )
        
        logging.info(f"Successfully updated mapping for index '{index_name}': {response}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to update mapping for index '{index_name}': {e}")
        raise


def delete_index(
    es_client: Elasticsearch,
    index_name: str,
    ignore_missing: bool = True
) -> bool:
    """
    Delete an Elasticsearch index.
    
    Args:
        es_client (Elasticsearch): Elasticsearch client instance
        index_name (str): Name of the index to delete
        ignore_missing (bool): If True, ignore if index doesn't exist
    
    Returns:
        bool: True if index was deleted successfully
    """
    try:
        if not es_client.indices.exists(index=index_name):
            if ignore_missing:
                logging.info(f"Index '{index_name}' does not exist. Skipping deletion.")
                return True
            else:
                raise ValueError(f"Index '{index_name}' does not exist.")
        
        response = es_client.indices.delete(index=index_name)
        logging.info(f"Successfully deleted index '{index_name}': {response}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to delete index '{index_name}': {e}")
        raise


def index_exists(es_client: Elasticsearch, index_name: str) -> bool:
    """
    Check if an index exists.
    
    Args:
        es_client (Elasticsearch): Elasticsearch client instance
        index_name (str): Name of the index to check
    
    Returns:
        bool: True if index exists
    """
    try:
        return es_client.indices.exists(index=index_name)
    except Exception as e:
        logging.error(f"Failed to check if index '{index_name}' exists: {e}")
        return False


def get_index_settings(
    es_client: Elasticsearch,
    index_name: str
) -> Optional[Dict[str, Any]]:
    """
    Get settings for an index.
    
    Args:
        es_client (Elasticsearch): Elasticsearch client instance
        index_name (str): Name of the index
    
    Returns:
        Optional[Dict[str, Any]]: Index settings if exists, None otherwise
    """
    try:
        if not es_client.indices.exists(index=index_name):
            logging.info(f"Index '{index_name}' does not exist.")
            return None
        
        response = es_client.indices.get_settings(index=index_name)
        return response.get(index_name, {}).get("settings", {})
        
    except Exception as e:
        logging.error(f"Failed to get settings for index '{index_name}': {e}")
        return None


def update_index_settings(
    es_client: Elasticsearch,
    index_name: str,
    settings: Dict[str, Any]
) -> bool:
    """
    Update settings for an index.
    
    Args:
        es_client (Elasticsearch): Elasticsearch client instance
        index_name (str): Name of the index
        settings (Dict[str, Any]): New settings
    
    Returns:
        bool: True if settings were updated successfully
    """
    try:
        if not es_client.indices.exists(index=index_name):
            logging.error(f"Index '{index_name}' does not exist.")
            return False
        
        response = es_client.indices.put_settings(
            index=index_name,
            body=settings
        )
        
        logging.info(f"Successfully updated settings for index '{index_name}': {response}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to update settings for index '{index_name}': {e}")
        raise


# Convenience function for common setup
def setup_index(
    es_client: Elasticsearch,
    index_name: str,
    force_recreate: bool = False,
    mapping: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Set up the index with proper mapping and settings.
    
    This is a convenience function that combines index creation with
    mapping setup.
    
    Args:
        es_client (Elasticsearch): Elasticsearch client instance
        index_name (str): Name of the index
        force_recreate (bool): If True, delete and recreate the index
        mapping (Optional[Dict[str, Any]]): Custom mapping (uses default if None)
    
    Returns:
        bool: True if index was set up successfully
    """
    try:
        # Check current mapping
        current_mapping = get_existing_mapping(es_client, index_name)
        
        if current_mapping and not force_recreate:
            logging.info(f"Index '{index_name}' already exists with mapping.")
            
            # Check if vector field is configured correctly
            properties = current_mapping.get("properties", {})
            if "vector" not in properties:
                logging.warning(
                    f"Index '{index_name}' is missing 'vector' field. "
                    "Consider recreating with proper mapping."
                )
            
            return True
        
        # Create or recreate the index
        return create_index(
            es_client=es_client,
            index_name=index_name,
            mapping=mapping,
            force_recreate=force_recreate
        )
        
    except Exception as e:
        logging.error(f"Failed to set up index '{index_name}': {e}")
        return False


# Example usage
if __name__ == "__main__":
    # This is an example of how to use the index mapping functions
    import os
    from elasticsearch import Elasticsearch
    
    # Connect to Elasticsearch
    es = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_HOST", "localhost:9200")],
        basic_auth=(
            os.getenv("ELASTICSEARCH_USER", "elastic"),
            os.getenv("ELASTICSEARCH_PASSWORD", "password")
        ),
        verify_certs=False
    )
    
    # Setup index
    index_name = "ibg_kb_documents"
    
    success = setup_index(
        es_client=es,
        index_name=index_name,
        force_recreate=False
    )
    
    if success:
        print(f"Index '{index_name}' is ready.")
    else:
        print(f"Failed to set up index '{index_name}'.")