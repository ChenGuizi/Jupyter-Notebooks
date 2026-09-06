# %% [markdown]
# # Non-Multimodal Data Ingestion Pipeline
# 
# This notebook handles the complete data ingestion pipeline for KB documents including:
# 1. Semi-automating KB file download (run locally)
# 2. Data ingestion to Elasticsearch (run in CDSW)
# 3. Technical metadata update (run locally)

# %% [markdown]
# ## 1. Semi-Automating KB File Download - Run in Local
# 
# **Manual steps needed:**
# - Ensure category of documents are chosen strictly from the dropdown list provided in business metadata
# - Bulk download Sharepoint folder to local machine first (needed because Sharepoint requires user authentication)
# - For documents not downloaded successfully, visit hyperlinks and download manually
# - Ensure all documents are in PDF format (convert Word and PPT, reject all other formats)

# %%
import os
import re
import json
import shutil
import requests
import urllib
import pandas as pd
from datetime import datetime
from urllib.parse import urlparse

# %% [markdown]
# ### 1.1 Utility Functions

# %%
def extract_domain(url):
    """
    Extract the domain name from a given URL.
    
    Parameters:
        url (str): URL string to extract domain from
    
    Returns:
        str: Domain name
    """
    parsed_url = urlparse(url)
    return parsed_url.netloc


def list_all_files(file_path):
    """
    List all files in a folder and its subfolders.
    
    Parameters:
        file_path (str): Folder path
    
    Returns:
        list: List of file paths
    """
    if not os.path.isdir(file_path):
        raise ValueError("Provided path is not a directory or does not exist.")
    
    file_list = []
    for root, dirs, files in os.walk(file_path):
        for filename in files:
            file_list.append(os.path.join(root, filename))
    return file_list


def split_filepath_into_subfolders(file_path):
    """
    Get subfolder names from file path.
    
    Parameters:
        file_path (str): File path
    
    Returns:
        list: Subfolder names
    """
    subfolders = []
    while True:
        file_path, folder = os.path.split(file_path)
        if folder:
            subfolders.insert(0, folder)
        if not file_path:
            subfolders.insert(0, file_path)
            break
    return subfolders


def get_metadata(source_file):
    """
    Get metadata of a file.
    
    Parameters:
        source_file (str): File path
    
    Returns:
        dict: File metadata including name, modified date, size, extension, and path info
    """
    try:
        # Get file metadata
        file_name = os.path.basename(source_file)
        modified_time = os.path.getmtime(source_file)
        file_size = os.path.getsize(source_file)
        file_extension = os.path.splitext(source_file)[1]
        
        # Get subfolder hierarchy
        subfolders = split_filepath_into_subfolders(source_file)
        filepath_N1 = subfolders[-2] if len(subfolders) >= 2 else ""
        filepath_N2 = subfolders[-3] if len(subfolders) >= 3 else ""
        
        # Convert modified time to datetime
        modified_date = datetime.fromtimestamp(modified_time)
        
        return {
            'file_name': file_name,
            'modified_date': modified_date,
            'file_size': file_size,
            'file_extension': file_extension,
            'filepath': source_file,
            'filepath_N1': filepath_N1,
            'filepath_N2': filepath_N2
        }
    except Exception as e:
        return str(e)


def extract_id_and_decode(url):
    """
    Decode the URL and extract the file ID/name.
    
    Parameters:
        url (str): URL to decode
    
    Returns:
        str: Decoded file name with extension
    """
    # Clean URL
    url = url.replace("%25u2013", "-").replace("%2E", ".")
    parsed_url = urllib.parse.urlparse(url)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    
    # Try to get 'id' or 'file' parameter
    decoded_id = None
    if 'id' in query_params:
        id_value = query_params['id'][0]
        decoded_id = urllib.parse.unquote(id_value)
    elif 'file' in query_params:
        file_value = query_params['file'][0]
        decoded_id = urllib.parse.unquote(file_value)
    
    # If not found, try decoding the full URL
    if not decoded_id:
        decoded_url = urllib.parse.unquote(url)
        
        # Extract based on file extension
        if ".pdf" in decoded_url:
            parts = decoded_url.split(".pdf")[0].split('/')[-1]
            decoded_id = parts + ".pdf"
        elif ".pptx" in decoded_url:
            parts = decoded_url.split(".pptx")[0].split('/')[-1]
            decoded_id = parts + ".pptx"
        elif ".ppt" in decoded_url:
            parts = decoded_url.split(".ppt")[0].split('/')[-1]
            decoded_id = parts + ".ppt"
        elif ".docx" in decoded_url:
            parts = decoded_url.split(".docx")[0].split('/')[-1]
            decoded_id = parts + ".docx"
        elif ".doc" in decoded_url:
            parts = decoded_url.split(".doc")[0].split('/')[-1]
            decoded_id = parts + ".doc"
        elif ".xlsx" in decoded_url:
            parts = decoded_url.split(".xlsx")[0].split('/')[-1]
            decoded_id = parts + ".xlsx"
        elif ".xls" in decoded_url:
            parts = decoded_url.split(".xls")[0].split('/')[-1]
            decoded_id = parts + ".xls"
    
    return decoded_id


def download_file(url, save_path):
    """
    Download file from Intranet directly.
    
    Parameters:
        url (str): URL of the file
        save_path (str): Path to save the file
    """
    if "go.mydbs.net" in url:
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"File downloaded successfully to {save_path}")
        else:
            print(f"Failed to download file from {url}")


def prepare_sharepoint_url(x_list):
    """
    Prepare Sharepoint URL for download.
    
    Parameters:
        x_list (list): List of URL parts
    
    Returns:
        str: Joined URL
    """
    return '/'.join(x_list)


# %% [markdown]
# ### 1.2 Read and Prepare Metadata for Downloading
# 
# Reads KBdocs_metadata_template_MASTER.xlsx (Excel file containing list of requested files)

# %%
# Specify business metadata file path and batch number
metadata_local_base = 'C://Users/mengxuantan/OneDrive - DBS Bank Ltd/IBG GEN AI/BG Query Bot/Document metadata/'
business_metadata_filepath = os.path.join(
    metadata_local_base, 
    'Business metadata/KBdocs metadata template_MASTER_foruploadingtoGrecesharepoint.xlsx'
)

batch_num = 'Batch 6'
ibgk_sharepoint_root = 'https://dbs1bank.s'

# Load URL encoding mapping
with open('../urlencode.json') as f:
    char_to_urlencode = json.load(f)

# %%
# Read business metadata
df_info = pd.read_excel(business_metadata_filepath, sheet_name='Sheet1')

# Clean and prepare data
df_info = df_info.loc[df_info.notnull().all(axis=1), :].reset_index(drop=True)
df_info = df_info.loc[2::].reset_index(drop=True)
df_info = df_info.loc[:, [i for i in df_info.columns if pd.notnull(i)]]
df_info = df_info.loc[pd.notnull(df_info["Document name"]), :].reset_index(drop=True)
df_info = df_info[df_info['INGESTION BATCH (SL TO FILL IN )'] == batch_num].copy()

# Set column names
df_info.columns = df_info.loc[0, :]

# %% [markdown]
# ### 1.3 URL Encoding and Processing

# %%
def url_encode(x):
    """
    URL encode the hyperlink.
    
    Parameters:
        x (str): Hyperlink to encode
    
    Returns:
        str: URL encoded hyperlink
    """
    if ibgk_sharepoint_root in x:
        split_link = x.split('/')
        filename = split_link[-1]
        filename_split = list(filename)
        
        # Encode special characters
        for idx in range(len(filename_split) - 3):
            char = filename_split[idx]
            char_join = ''.join(filename_split[idx:idx+3])
            if char in char_to_urlencode and char_join not in char_to_urlencode:
                print(f"Processing: {filename}")
                filename_split[idx] = char_to_urlencode[char]
        
        split_link[-1] = ''.join(filename_split)
        return '/'.join(split_link)
    return x

# %%
# Apply URL encoding
df_info["Hyperlink of the document (in intranet/sharepoint/mega etc)"] = df_info[
    "Hyperlink of the document (in intranet/sharepoint/mega etc)"
].apply(lambda x: url_encode(x))

# %%
# Prepare Sharepoint URLs
df_info['proc url type'] = 'dbs1bank.sharepoint.com'
df_info['proc url manualbulkdownload'] = df_info.loc[
    df_info['proc url type'] == 'dbs1bank.sharepoint.com', 
    'proc url'
].apply(lambda x: extract_id_and_decode(x))

# %%
# Paths for local storage
base_destination_folder = "C://Users/mengxuantan/OneDrive - DBS Bank Ltd/shared docs/"

# Check for duplicate indexes (separate delete and add before merging)
df_info.columns = ['proc sharepoint local_' + col for col in df_info.columns]

# Merge with existing file list
# Note: Merge logic would be implemented here based on actual data structure

# %%
# Prepare base description
cols = [
    'joining_key', 
    'document.name based on.filename',
    'document name based on first page', 
    'document name by user', 
    'hyperlink', 
    'category',
    'date of issue based on filename', 
    'date of issue based on first page', 
    'date of issue by user'
]

# %% [markdown]
# ## 2. Data Ingestion - Run in CDSW

# %%
# Change to working directory
os.chdir('/home/cdsw/ibgkb_data_ingestion')

# %%
# Import required modules
from src.metadata import create_metadata, convert_metadata_df_to_utf8_compatible
from src.non_multi_model_processing import create_chunks_from_documents
from src.elasticsearch import connect_to_elk, ingest, get_total_chunks
from ada_genai.experimental.langchain import ElasticsearchStore
from ada_genai.langchain import VertexAIEmbeddings
from ada_genai.auth import sso_auth
from urllib.request import pathname2url
import pandas as pd

# %%
# Authentication
sso_auth.login()

# %%
# Specify main CDSW path configurations
batch_num = 6

os.environ["CDSW_SHAREPOINT_PATH"] = f"/home/cdsw/ibgkb_data_ingestion/data/local_batch_{batch_num}/"
os.environ["SHAREPOINT_URL"] = "https://dbs1bank.sharepoint.com/sites/satibg.kb/Shared%20Documents/"

file_directory = f"/home/cdsw/ibgkb_data_ingestion/data/local_batch_{batch_num}/"

# %%
# Create metadata from file directory
metadata_df = create_metadata(file_directory)
metadata_df = metadata_df[metadata_df['joining_key'].notnull()]
metadata_df['joining_key'] = metadata_df['joining_key'].astype(str)

# %% [markdown]
# ### 2.1 Merge with Business Metadata

# %%
def merge_with_business_metadata(business_metadata_filepath, tech_metadata):
    """
    Merge technical metadata with business metadata.
    
    Parameters:
        business_metadata_filepath (str): Path to business metadata Excel
        tech_metadata (DataFrame): Technical metadata
    
    Returns:
        tuple: (tech_metadata_add, df_info_delete)
    """
    # Read business metadata
    df_info = pd.read_excel(business_metadata_filepath, sheet_name='Sheet1')
    
    # Filter ADD operations
    df_info_add = df_info[df_info['ACTION (ADD/UPDATE/DELETE)'] == 'ADD'].copy()
    df_info_delete = df_info[df_info['ACTION (ADD/UPDATE/DELETE)'] == 'DELETE'].copy()
    
    # Clean date columns
    df_info_add['Date that file was last updated'] = df_info_add['Date that file was last updated'].astype(str).apply(
        lambda x: re.sub(r"[^a-zA-Z0-9]", "", x.lower())
    )
    
    # Merge with technical metadata
    cols = ['joining_key', 'date of issue by user', 'document.name by user']
    tech_metadata = pd.merge(df_info_add[cols], tech_metadata, on='joining_key', how='left')
    
    cols = ['joining_key', 'document name', 'document.name.based.on.filename', 'date of issue based on filename']
    
    return tech_metadata, df_info_delete

# %%
# Merge metadata
tech_metadata, df_info_delete = merge_with_business_metadata(
    business_metadata_filepath, 
    metadata_df
)

# %% [markdown]
# ### 2.2 Set up Connection to Elasticsearch

# %%
def create_query(metadata_hyperlink):
    """
    Create an Elasticsearch bool query to filter documents based on hyperlinks.
    
    Parameters:
        metadata_hyperlink (str): Hyperlink of document to match
    
    Returns:
        dict: The Elasticsearch bool query
    """
    query = {
        "bool": {
            "must": [{
                "match_phrase": {
                    "metadata.hyperlink": metadata_hyperlink
                }
            }]
        }
    }
    return query


def delete_elk_docs_from_hyperlinks(metadata_delete, es_instance, collection_name):
    """
    Delete documents from Elasticsearch based on hyperlinks.
    
    Parameters:
        metadata_delete (DataFrame): Metadata for documents to delete
        es_instance: Elasticsearch instance
        collection_name (str): Name of the collection/index
    
    Returns:
        int: Total number of deleted chunks
    """
    hyperlinks_delete = metadata_delete['Hyperlink of the document (in intranet/sharepoint/mega etc)'].values
    total_del_chunks = 0
    
    for metadata_hyperlink in hyperlinks_delete:
        # Query for documents
        num_hits = es_instance.search(
            index=collection_name,
            query=create_query(metadata_hyperlink)
        )["hits"]["total"]["value"]
        
        total_del_chunks += num_hits
        print(f"Deleting: {metadata_hyperlink}")
        print(f"Number of hits: {num_hits}")
        print(f"Total deletion chunks so far: {total_del_chunks}")
        
        # Uncomment below for actual deletion
        # resp = es_instance.delete_by_query(
        #     index=collection_name,
        #     body={"query": create_query(metadata_hyperlink)},
        #     refresh=True
        # )
    
    return total_del_chunks

# %%
# UAT Settings
# es_url = "https://amlp-es-co01.uat.dbs.com:9200"
# collection_name = 'datasets-kb.kb-ibg.sg.uat'
# collection_name = 'datasets-kb.policy-ibg.sg.uat.test_v3'

# PROD Settings
es_url = "https://amlp-es-co01.sgp.dbs.com:9200"
collection_name = 'datasets-kb.kb-bg.sg.prod'

# %%
# Connect to Elasticsearch
es_store_instance, es_instance = connect_to_elk(collection_name, es_url)

# Get total chunks
total_chunks = get_total_chunks(es_instance, collection_name)
print(f"Total chunks in Elasticsearch: {total_chunks}")

# %% [markdown]
# ### 2.3 Data Deletion from ELK

# %%
# Delete documents marked for deletion
total_deleted = delete_elk_docs_from_hyperlinks(
    df_info_delete, 
    es_instance, 
    collection_name
)
print(f"Total deleted chunks: {total_deleted}")

# %% [markdown]
# ### 2.4 Data Ingestion to ELK

# %%
# Prepare columns for ingestion
cols = ['joining_key', 'hyperlink', 'category', 'document.name', 'date of Issue']
metadata_df_add = tech_metadata[cols].copy()

# %%
# Convert to UTF-8 compatible format
metadata_fill_in_utf8_path = '/home/cdsw/ibgkb_data_ingestion/data/metadata_fill_in_utf8.csv'
metadata_df_add.to_csv(metadata_fill_in_utf8_path, index=False, header=True)

# Convert metadata to UTF-8 compatible
convert_metadata_df_to_utf8_compatible(metadata_fill_in_utf8_path)

# %%
# Load converted metadata
metadata_fill_in_df = pd.read_csv(metadata_fill_in_utf8_path)

# %%
# Create document chunks
all_document_chunk_list = create_chunks_from_documents(
    metadata_fill_in_df,
    os.environ["CDSW_SHAREPOINT_PATH"]
)

# Check number of chunks to be ingested
print(f"Total chunks to ingest: {len(all_document_chunk_list)}")

# %%
# Ingest documents to Elasticsearch
batch_size = 150
timer_per_batch = 60

failed_docs = ingest(
    all_document_chunk_list=all_document_chunk_list,
    es_store_instance=es_store_instance,
    es_instance=es_instance,
    collection_name=collection_name,
    batch_size=batch_size,
    timer_per_batch=timer_per_batch
)

print(f"Failed documents: {len(failed_docs) if failed_docs else 0}")

# %%
# Verify final counts
final_total = get_total_chunks(es_instance, collection_name)
print(f"Final total chunks: {final_total}")

# %% [markdown]
# ## 3. Technical Metadata Update - Run in Local
# 
# Update metadata accordingly

# %%
def get_prev_metadata(metadata_local_base, current_batch_num):
    """
    Get previous batch metadata and combine.
    
    Parameters:
        metadata_local_base (str): Base path for metadata
        current_batch_num (int): Current batch number
    
    Returns:
        DataFrame: Previous metadata batches combined
    """
    metadata_local_base_path = os.path.join(metadata_local_base, 'tech_metadata/')
    metadata_batches = []
    
    for subdir, dirs, files in os.walk(metadata_local_base_path):
        for file in files:
            if re.match(r"batch_\d+_metadata", file):
                batch_num = int(re.findall(r"batch_(\d+)_metadata", file)[0])
                if batch_num < current_batch_num:
                    file_path = os.path.join(metadata_local_base_path, file)
                    metadata_batch = pd.read_csv(file_path)
                    metadata_batches.append(metadata_batch)
    
    metadata_prev_batches = pd.concat(metadata_batches, axis=0)
    return metadata_prev_batches


def save_metadata(metadata_local_base_path, cols, current_batch_num, current_batch_add, 
                  current_batch_delete, full_catalog):
    """
    Save metadata to local storage.
    
    Parameters:
        metadata_local_base_path (str): Base path for metadata
        cols (list): Columns to save
        current_batch_num (int): Current batch number
        current_batch_add (DataFrame): Added metadata
        current_batch_delete (DataFrame): Deleted metadata
        full_catalog (DataFrame): Full catalog
    """
    # Save add metadata
    add_path = os.path.join(metadata_local_base_path, f'tech_metadata/batch_{current_batch_num}_add.csv')
    current_batch_add[cols].to_csv(add_path, header=True, index=False)
    
    # Save delete metadata
    del_path = os.path.join(metadata_local_base_path, f'tech_metadata/batch_{current_batch_num}_delete.csv')
    current_batch_delete[cols].to_csv(del_path, header=True, index=False)
    
    # Save full catalog
    full_catalog_path = os.path.join(metadata_local_base_path, f'Catalog/batch_{current_batch_num}_catalog.csv')
    full_catalog[cols].to_csv(full_catalog_path, header=True, index=False)
    
    # Save master catalog
    master_catalog_path = os.path.join(metadata_local_base_path, 'Catalog/master_catalog.csv')
    full_catalog[cols].to_csv(master_catalog_path, header=True, index=False)

# %%
# Configuration for local update
metadata_local_base_path = 'C://Users/mengxuantan/OneDrive - DBS Bank Ltd/IBG GEN AI/BG Query Bot/Document metadata/'
current_batch_num = 6

# %%
# Get previous metadata
metadata_prev_batches = get_prev_metadata(metadata_local_base_path, current_batch_num)

# %%
# Load current batch metadata
metadata_add_path = os.path.join(metadata_local_base_path, f'tech_metadata/batch_{current_batch_num}_add.csv')
metadata_del_path = os.path.join(metadata_local_base_path, f'tech_metadata/batch_{current_batch_num}_delete.csv')

current_batch_add = pd.read_csv(metadata_add_path)
current_batch_delete = pd.read_csv(metadata_del_path)

# %%
# Process current batch
current_batch_add['hyperlink_lower'] = current_batch_add['hyperlink'].str.lower()
current_batch_add['batch'] = current_batch_num

delete_links = current_batch_delete['Hyperlink of the document (in intranet/sharepoint/mega etc)'].values.tolist()
delete_links_lower = [x.lower() for x in delete_links]

metadata_prev_batches['hyperlink_lower'] = metadata_prev_batches['hyperlink'].str.lower()

# Get delete metadata from previous batches
current_batch_delete_metadata = metadata_prev_batches[
    metadata_prev_batches['hyperlink_lower'].isin(delete_links_lower)
].copy()
current_batch_delete_metadata['batch'] = current_batch_num

# %%
# Build full catalog
cols_to_save = ['joining_key', 'document.name', 'hyperlink', 'category', 'date_of_issue']

# Use previous batch catalog as base
prev_catalog_path = os.path.join(
    metadata_local_base_path, 
    f'Catalog/batch_{current_batch_num - 1}_catalog.csv'
)
prev_catalog = pd.read_csv(prev_catalog_path)
cols_to_save = prev_catalog.columns.tolist()

# Remove deleted chunks
full_catalog = prev_catalog[~prev_catalog['hyperlink_lower'].isin(delete_links_lower)]

# Add current batch
full_catalog = pd.concat([full_catalog, current_batch_add], axis=0)

# %%
# Save updated metadata
save_metadata(
    metadata_local_base_path,
    cols_to_save,
    current_batch_num,
    current_batch_add,
    current_batch_delete_metadata,
    full_catalog
)

print("Metadata update completed successfully!")

# %% [markdown]
# ## 4. Summary

# %%
print("=" * 50)
print("NON-MULTIMODAL INGESTION SUMMARY")
print("=" * 50)
print(f"Batch Number: {batch_num}")
print(f"Collection: {collection_name}")
print(f"Total chunks ingested: {len(all_document_chunk_list)}")
print(f"Final chunks in Elasticsearch: {final_total}")
if failed_docs:
    print(f"⚠️ Failed documents: {len(failed_docs)}")
else:
    print("✅ All documents ingested successfully!")
print("=" * 50)