import json
import pandas as pd
from elasticsearch import Elasticsearch
from typing import Union, List
from utils.logger import logger
from config import Setting

if __name__ == "__main__":
    pass

class Configurator(ConfigParser):
    def __init__(self):
        self.file_path = "/app/edge/config/config.properties"
        self.parser = ConfigParser()
        self.parser.read(self.file_path, encoding='utf-8')

    def get_config(self, section, key):
        try:
            value = self.parser.get(section, key)
            return value
        except Exception as e:
            print(f"Error getting config: {e}")
            return None

def es_connect():
    es_url = config.get('ES', 'server_http')
    es = Elasticsearch([es_url])
    return es

# ... (There is a large block of seemingly unrelated code or comments here)
# It looks like an argparse section for the "scripts" that might have been scrolled past or is illegible due to screen glare.

def create_index(index_name):
    [blurred]

# ... (Repeated code blocks)
# There are several blocks that look like this:
# [blurred] ... open a file like "info.txt"... 
# but the actual logic inside is hidden.

def get_data(index_name, size):
    es = es_connect()
    res = es.search(index=index_name, size=size, body={"query": {"match_all": {}}})
    return res

def get_data_es(index_name, search, size):
    es = es_connect()
    res = es.search(index=index_name, body={"query": {"match": {"name": search}}}, size=size)
    return res

def get_data_es_year(index_name, search, year, size):
    es = es_connect()
    res = es.search(index=index_name, body={"query": {"bool": {"must": [{"match": {"name": search}}, {"term": {"year": year}}]}}}, size=size)
    return res

def get_data_es_range(index_name, start_date, end_date, size):
    es = es_connect()
    res = es.search(index=index_name, body={"query": {"range": {"date": {"gte": start_date, "lte": end_date}}}}, size=size)
    return res

def get_data_es_publish(index_name, search, year, size):
    es = es_connect()
    res = es.search(index=index_name, body={"query": {"bool": {"must": [{"match": {"name": search}}, {"term": {"year": year}}, {"term": {"status": "published"}}]}}}, size=size)
    return res

def get_json(index_name, search, size):
    es = es_connect()
    res = es.search(index=index_name, body={"query": {"bool": {"must": [{"match": {"name": search}}]}}}, size=size)
    return res

def get_json_data(index_name, search, size):
    es = es_connect()
    res = es.search(index=index_name, body={"query": {"bool": {"must": [{"match": {"name": search}}]}}}, size=size)
    # ... (logic to process 'res' probably)
    return res

def delete_index(index_name):
    es = es_connect()
    res = es.indices.delete(index=index_name, ignore=[400, 404])
    return res

def delete_data(index_name, id):
    es = es_connect()
    res = es.delete(index=index_name, id=id)
    return res

def list_all_indexes():
    es = es_connect()
    res = es.indices.get_alias("*")
    return res

def get_status(index_name):
    es = es_connect()
    res = es.indices.exists(index=index_name)
    return res

def insert_data(index_name, doc):
    es = es_connect()
    res = es.index(index=index_name, body=doc)
    return res

def insert_data_bulk(index_name, docs):
    es = es_connect()
    res = es.bulk(index=index_name, body=docs)
    return res

def update_data(index_name, id, doc):
    es = es_connect()
    res = es.update(index=index_name, id=id, body={"doc": doc})
    return res

def update_data_query(index_name, query, doc):
    es = es_connect()
    res = es.update_by_query(index=index_name, body={"query": query, "script": {"source": doc}})
    return res

def delete_data_query(index_name, query):
    es = es_connect()
    res = es.delete_by_query(index=index_name, body={"query": query})
    return res

def query_data(index_name, query, size):
    es = es_connect()
    res = es.search(index=index_name, body={"query": query, "size": size})
    return res

def get_total_count(index_name):
    es = es_connect()
    res = es.count(index=index_name)
    return res['count']

def get_all_docs(index_name, size):
    es = es_connect()
    res = es.search(index=index_name, body={"query": {"match_all": {}}, "size": size})
    return res

def get_all_docs_scroll(index_name, size, scroll_time):
    es = es_connect()
    res = es.search(index=index_name, body={"query": {"match_all": {}}, "size": size}, scroll=scroll_time)
    return res