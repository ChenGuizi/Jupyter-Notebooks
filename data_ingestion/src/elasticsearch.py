"""
elasticsearch.py

Helpers for connecting to Elasticsearch, ingesting document chunks into a
vector store + Elasticsearch index, checking ingestion completeness, and
managing document versions in the index.

NOTE: This file was reconstructed from a badly OCR-corrupted source. The
overall structure, function names, and logic follow the original docstrings
and code fragments as closely as possible. Some details (exact query bodies,
exact control flow in a few branches) were inferred from context and may
need adjustment to match the original behavior exactly.
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import pytz
from tqdm import tqdm
from langchain_core.documents.base import Document

from ada_genai.experimental.langchain import ElasticsearchStore
from ada_genai.langchain import VertexAIEmbeddings


def connect_to_elk(collection_name, es_url):
    """Connect to an Elasticsearch-backed vector store and raw ES client.

    Args:
        collection_name (str): Name of the ES index / collection.
        es_url (str): URL of the Elasticsearch server.

    Returns:
        tuple: (es_store_instance, es_instance)
            es_store_instance: The LangChain ElasticsearchStore vector store.
            es_instance: The raw Elasticsearch client.
    """
    try:
        es_store_instance = ElasticsearchStore(
            es_url=es_url,  # elk server
            index_name=collection_name,  # collection name
            embedding=VertexAIEmbeddings(model_name="textembedding-gecko@003"),  # embedder
            es_user=os.environ["ELASTICSEARCH_USER"],
            es_password=os.environ["ELASTICSEARCH_PASSWORD"],
        )
        es_instance = es_store_instance.connect_to_elasticsearch(
            es_url=es_url,
            username=os.environ["ELASTICSEARCH_USER"],
            password=os.environ["ELASTICSEARCH_PASSWORD"],
        )
        print("Connect Successfully!")
    except Exception as e:
        print(f"Connect Unsuccessfully: {e}")
        return None, None

    return es_store_instance, es_instance


def delete_collection(es_store_instance, collection_name):
    """Delete an entire collection/index from the Elasticsearch instance.

    Args:
        es_store_instance (object): The ElasticsearchStore vector store instance.
        collection_name (str): The name of the index/collection to delete.

    Returns:
        None

    This calls the delete_index helper method to fully remove the collection
    from Elasticsearch.
    """
    es_store_instance.delete_index(collection_name)
    return None


def get_num_chunks_in_collection(es_instance, collection_name):
    """Get the total number of documents/chunks in an Elasticsearch collection.

    Args:
        es_instance (object): The Elasticsearch client object.
        collection_name (str): The name of the index/collection.

    Returns:
        int: The total number of documents in the Elasticsearch collection.

    It performs a count API call on the index to return the number of
    documents.
    """
    num_objects = es_instance.count(index=collection_name)["count"]
    return num_objects


def get_all_chunks_in_collection(es_instance, collection_name):
    """Get all chunks/documents from the Elasticsearch collection.

    Args:
        es_instance (object): The Elasticsearch client object.
        collection_name (str): The name of the index/collection.

    Returns:
        list: The list of hits/documents in the collection.

    It gets the total number of documents, then does a search API call to
    retrieve all the documents in batches. The hits are returned.
    """
    num_objects = get_num_chunks_in_collection(
        es_instance=es_instance, collection_name=collection_name
    )
    res = es_instance.search(
        index=collection_name,
        body={"size": num_objects, "query": {"match_all": {}}},
    )
    ingested_chunks = res["hits"]["hits"]
    return ingested_chunks


def get_docs_to_reingest(es_instance, documents_to_ingest, collection_name, delete=False):
    """Identify and return documents that need to be reingested.

    Args:
        es_instance (object): Elasticsearch client instance.
        documents_to_ingest (list): List of original documents to ingest.
        collection_name (str): The Elasticsearch index name.
        delete (bool): Whether to delete existing chunks after checking.

    Returns:
        list: Documents (grouped by chunk list) that need to be reingested.

    It compares existing ingested chunks in ES to the original list of
    documents to find differences in chunk counts, indicating a failed or
    partial ingestion. For any documents with missing chunks, it optionally
    deletes existing chunks then returns the list of documents to reingest.
    """
    # GET METADATA AND EMBEDDINGS OF ALL CHUNKS INGESTED IN THE COLLECTION #
    ingested_chunks = get_all_chunks_in_collection(
        es_instance=es_instance, collection_name=collection_name
    )

    # COUNT THE NUMBER OF CHUNKS INGESTED PER DOCUMENT #
    ingested_chunk_dict = {}
    for chunk in ingested_chunks:
        tmp = chunk["_source"]["metadata"]
        key = tmp["document_name_match"] + "_" + str(tmp["date_of_issue_match"])
        chunk_count = ingested_chunk_dict.get(key, 0)
        chunk_count = chunk_count + 1
        ingested_chunk_dict.update({key: chunk_count})

    print(f"Total Number of Chunks Ingested: {sum(ingested_chunk_dict.values())}")

    # COUNT THE NUMBER OF CHUNKS THAT SHOULD HAVE BEEN INGESTED PER DOCUMENT #
    supposed_chunk_dict = {}
    for doc in documents_to_ingest:
        for chunk in doc:
            tmp = chunk.metadata
            key = tmp["document_name_match"] + "_" + str(tmp["date_of_issue_match"])
            chunk_count = supposed_chunk_dict.get(key, 0)
            chunk_count = chunk_count + 1
            supposed_chunk_dict.update({key: chunk_count})

    print(f"Total Number of Chunks That Should Be Ingested: {sum(supposed_chunk_dict.values())}")

    # COMPARE INGESTED VS SUPPOSED TO IDENTIFY DOCUMENTS NOT INGESTED PROPERLY #
    to_reingest = []
    for key in supposed_chunk_dict.keys():
        supposed_num_chunks = supposed_chunk_dict.get(key, 0)
        ingested_num_chunks = ingested_chunk_dict.get(key, 0)
        if supposed_num_chunks != ingested_num_chunks:
            to_reingest.append(key)

    print(f"Documents Not Ingested Properly: {', '.join(to_reingest)}")

    indexes_to_delete = []
    for _key in to_reingest:
        for chunk in ingested_chunks:
            tmp = chunk["_source"]["metadata"]
            key = tmp["document_name_match"] + "_" + str(tmp["date_of_issue_match"])
            if key == _key:
                indexes_to_delete.append(chunk["_id"])

    docs_to_reingest = []
    for key in to_reingest:
        tmp = key.split("_")
        metadata_date_of_issue_match = tmp[-1]
        metadata_doc_name_match = "_".join(tmp[:-1])

        chunk_lst = []
        for doc in documents_to_ingest:
            for chunk in doc:
                if metadata_date_of_issue_match == str(chunk.metadata["date_of_issue_match"]):
                    if metadata_doc_name_match == chunk.metadata["document_name_match"]:
                        chunk_lst.append(chunk)

        if len(chunk_lst) > 0:
            docs_to_reingest.append(chunk_lst)

    if delete is True:
        resp = delete_all_chunks_of_document_in_collection(
            es_instance=es_instance,
            collection_name=collection_name,
            ids=indexes_to_delete,
        )

    return docs_to_reingest


def ingest(
    all_document_chunk_list,
    es_store_instance,
    es_instance,
    collection_name,
    batch_size=100,
    timer_per_batch=60,
):
    """Main entry point for ingesting documents into ES and vector db.

    Args:
        all_document_chunk_list (list): List of all document chunks to ingest.
        es_store_instance (object): Vector storage client object (e.g. Milvus).
        es_instance (object): Elasticsearch client object.
        collection_name (str): Name of ES index to ingest into.
        batch_size (int): Batch size for ingestion.
        timer_per_batch (int): Timer between batches.

    Returns:
        list: Documents that failed to be ingested.

    This orchestrates the main ingestion process:
        1. Calls batch_process to index the chunks.
        2. Checks for failed docs with get_docs_to_reingest.
        3. Returns any failed documents to retry ingesting.

    The batch_process method handles ingesting batches of chunks into both
    the vector db for similarity search and Elasticsearch for metadata
    storage. A timer is used between batches to pause ingestion and not
    overload APIs. Failed documents are identified and returned to be
    reingested.
    """
    all_failed_docs = []

    if all_document_chunk_list:
        for i in range(0, len(all_document_chunk_list), batch_size):
            end = min(i + batch_size, len(all_document_chunk_list))
            batch = all_document_chunk_list[i:end]

            failed_docs = batch_process(
                batch,
                es_store_instance,
                es_instance,
                collection_name,
                batch_size=batch_size,
            )

            all_failed_docs.extend(failed_docs)
            time.sleep(timer_per_batch)

    return all_failed_docs


def batch_process(
    all_document_chunk_list: List[List[Document]],
    vector_db: Any,
    es_instance: Any,
    collection_name: str,
    batch_size: int,
) -> List[Document]:
    """Batch process and index a collection of documents into a vector database.

    Args:
        all_document_chunk_list (List[List[Document]]): A list of documents
            to be indexed, organized in chunks.
        vector_db (Any): A connection to the vector database (e.g., Milvus).
        es_instance (Any): An Elasticsearch client instance.
        collection_name (str): The name of the ELK index.
        batch_size (int): Size of batches for indexing.

    Returns:
        List[Document]: A list of documents that failed to be indexed.
    """
    failed_docs = []

    try:
        current_hit = es_instance.search(
            index=collection_name, query={"match_all": {}}
        )["hits"]["total"]["value"]
    except Exception as e:
        if "NotFoundError" in str(e):
            current_hit = 0

    print("Processing Documents into Vector DB:")
    print(f"Total Number of Documents to Ingest: {len(all_document_chunk_list)}")

    for index, docs in tqdm(
        enumerate(all_document_chunk_list),
        total=len(all_document_chunk_list),
        desc="Ingestion Progress",
    ):
        for start in range(0, len(docs), batch_size):
            end = min(start + batch_size, len(docs))
            batch = docs[start:end]

            try:
                vector_db.add_documents(batch)
            except Exception as e:
                print(
                    f"API call failed for all_docs index: {index}, batch start: "
                    f"{start + 1}, end: {end}, error: {str(e)}"
                )
                failed_docs.extend(batch)

            time.sleep(1)

    after_hit = es_instance.search(
        index=collection_name, query={"match_all": {}}
    )["hits"]["total"]["value"]

    print(f"Total Number of Chunks Successfully Ingested: {after_hit - current_hit}")
    print(
        f"Date of Ingestion: "
        f"{datetime.now(pytz.timezone('Asia/Singapore')).strftime('%y-%m-%d')}"
    )

    return failed_docs


def get_ids_of_chunks_of_doc_in_collection(
    es_instance, collection_name, document_name_match, date_of_issue_match
):
    """Get IDs of chunks for a document from the Elasticsearch collection.

    Args:
        es_instance (object): Elasticsearch client instance.
        collection_name (str): Name of the Elasticsearch index.
        document_name_match (str): Name of the document to match on.
        date_of_issue_match (int): Date of issue to match on.

    Returns:
        list: List of chunk IDs matching the document name and date.

    Retrieves all chunks from the collection. Constructs a query key from
    document_name_match and date_of_issue_match. Checks each chunk's
    metadata for matching name and date. Returns a list of IDs of all
    matching chunks.
    """
    a_key = document_name_match + "_" + str(date_of_issue_match)
    ingested_chunks = get_all_chunks_in_collection(
        es_instance=es_instance, collection_name=collection_name
    )

    ids = []
    for chunk in ingested_chunks:
        tmp = chunk["_source"]["metadata"]
        id_ = chunk["_id"]
        key = tmp["document_name_match"] + "_" + str(tmp["date_of_issue_match"])
        if key == a_key:
            ids.append(id_)

    return ids


def delete_all_chunks_of_document_in_collection(es_instance, collection_name, ids):
    """Delete all chunks for a document based on a list of chunk IDs.

    Args:
        es_instance (object): Elasticsearch client instance.
        collection_name (str): Name of the Elasticsearch index.
        ids (list): List of chunk IDs to delete.

    Returns:
        None

    Iterates through the list of IDs and uses the delete API to remove each
    chunk from the Elasticsearch collection.
    """
    for id_ in ids:
        resp = es_instance.delete(index=collection_name, id=id_)
    return None


def create_query(metadata_doc_name_match: str, range_parameter: dict) -> dict:
    """Create an Elasticsearch bool query to filter documents.

    Args:
        metadata_doc_name_match (str): Name of the document to match.
        range_parameter (dict): Range query conditions, e.g. {"lt": 123}.

    Returns:
        dict: The Elasticsearch bool query.

    This creates a bool query with two must clauses:
        1. Term query to match the exact document name.
        2. Range query to filter by a date or other range.

    The term query matches on the .keyword analyzed form of the document
    name field for an exact match.

    Example Usage:
        q = create_query("Document Title", {"lt": 20150101})
    """
    query = {
        "bool": {
            "must": [
                {
                    "term": {
                        "metadata.document_name_match.keyword": metadata_doc_name_match
                    }
                },
                {"range": {"metadata.date_of_issue_match": range_parameter}},
            ]
        }
    }
    return query


def check_document_version_in_vector_db(
    metadata_doc_name_match: str,
    metadata_date_of_issue_match: int,
    es_instance: Any,
    collection_name: str,
) -> bool:
    """Check if there are other versions of the current document already ingested.

    If the version of the document in the vector database is lower, delete
    those documents. If the version of the document in the vector database
    is higher, the current document will not be ingested.

    Args:
        metadata_doc_name_match (str): Document name.
        metadata_date_of_issue_match (int): Date of issue.
        es_instance (Any): An Elasticsearch client instance.
        collection_name (str): The name of the ELK index.

    Returns:
        bool: Whether the current document needs to be ingested.
    """
    lower_version_hits = es_instance.search(
        index=collection_name,
        query=create_query(
            metadata_doc_name_match,
            range_parameter={"lt": metadata_date_of_issue_match},
        ),
    )["hits"]["total"]["value"]  # number of lower version docs

    higher_version_hits = es_instance.search(
        index=collection_name,
        query=create_query(
            metadata_doc_name_match,
            range_parameter={"gte": metadata_date_of_issue_match},
        ),
    )["hits"]["total"]["value"]  # number of higher (or equal) version docs

    if lower_version_hits > 0:
        start_time = time.time()
        resp = es_instance.delete_by_query(
            index=collection_name,
            body={
                "query": create_query(
                    metadata_doc_name_match,
                    range_parameter={"lt": metadata_date_of_issue_match},
                )
            },
            refresh=True,  # wait for the deletion
        )
        end_time = time.time()
        print(
            f"Old Version of the Document Found and Replaced with the New Document\n"
            f"Document Name: {metadata_doc_name_match}\n"
            f"Deletion Time: {end_time - start_time} seconds"
        )

    if higher_version_hits > 0:
        return False
    else:
        return True


def get_total_chunks(es_instance, collection_name):
    """Get total number of chunks in a given ELK index.

    Args:
        es_instance (object): Elasticsearch client instance.
        collection_name (str): Name of ELK index.

    Returns:
        int: Total number of chunks in the index.
    """
    num_chunks = es_instance.search(
        index=collection_name,
        query={"bool": {"must": [{"term": {"_index": collection_name}}]}},
    )["hits"]["total"]["value"]

    return num_chunks
