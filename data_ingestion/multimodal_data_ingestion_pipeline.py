# %% [markdown]
# # Multimodal Data Ingestion Pipeline
# This notebook processes documents with images, extracts image data, and ingests them into Elasticsearch.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import os
import csv
import json
import time
import asyncio
import datetime
import urllib
import nest_asyncio

import pandas as pd
import numpy as np
import pikepdf

from tqdm import tqdm
from vertexai.preview.generative.models import GenerativeModel
from ada_genai.auth import sso_auth
from ada_genai.vertexai import patch_vertexai_use_ada

# Custom imports
from src.multi_modal_processing import (
    convert_single_pdf_to_images,
    get_gemini_response,
    create_chunks_from_multimodal_documents
)
from src.elasticsearch import connect_to_elk, ingest, get_total_chunks

# %%
# Authentication
sso_auth.login()

# Apply VertexAI patch
patch_vertexai_use_ada()

# Apply nest_asyncio for nested event loops
nest_asyncio.apply()

# %% [markdown]
# ## 2. Configuration

# %%
# Batch number for this run
BATCH_NUM = 6

# Paths configuration
os.environ["CDSW_SHAREPOINT_PATH"] = f"/home/cdsw/ibgkb_data_ingestion/data/local_batch_{BATCH_NUM}/"
os.environ["CDSW_MULTIMODAL_PATH"] = "/home/cdsw/ibgkb_data_ingestion/intermediate_data/multimodal/"
os.environ["SHAREPOINT_URL"] = "https://dbs1bank.sharepoint.com/sites/sgibg.kb/Shared%20Documents"

# Metadata file path
metadata_fill_in_path = '/home/cdsw/ibgkb_data_ingestion/data/metadata_fill_in_utf8.csv'
multimodal_intermediate_path = f'/home/cdsw/ibgkb_data_ingestion/metadata_multimodal_batch_{BATCH_NUM}.csv'

# Load metadata
metadata_fill_in_df = pd.read_csv(metadata_fill_in_path)

# Gemini model configuration
# model = GenerativeModel("gemini-1.5-pro-001")
model = GenerativeModel("gemini-1.5-flash-001")
SYSTEM_PROMPT = 'Explain the image'

# Batch processing configuration
BATCH_SIZE = 5  # For image explanation
INGESTION_BATCH_SIZE = 150
TIMER_PER_BATCH = 60

# %% [markdown]
# ## 3. Extract Images from Documents

# %%
def extract_images_from_documents(metadata_df, batch_num):
    """
    Identify pages in documents that contain images and save them as images.
    
    Args:
        metadata_df: DataFrame containing document metadata
        batch_num: Batch number for processing
    
    Returns:
        dict: Mapping of (document_key, page) to image path
    """
    total_mapping_dict = {}
    
    for index, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Parsing Progress"):
        print(f"Processing {index+1}: {row['document.name']}")
        
        # Get local document path
        hyperlink_parts = row['hyperlink'].split(os.environ["SHAREPOINT_URL"])
        local_doc_path = os.path.join(
            os.environ["CDSW_SHAREPOINT_PATH"],
            urllib.request.url2pathname(hyperlink_parts[-1])
        )
        
        # Convert PDF to images
        mapping_dict = convert_single_pdf_to_images(
            local_doc_path,
            os.environ["CDSW_MULTIMODAL_PATH"],
            row["joining_key"]
        )
        
        if mapping_dict:
            total_mapping_dict.update(mapping_dict)
    
    return total_mapping_dict

# %%
# Extract images from all documents
total_mapping_dict = extract_images_from_documents(metadata_fill_in_df, BATCH_NUM)
print(f"Total images extracted: {len(total_mapping_dict)}")

# %%
# Count total mappings
counter = 0
for key in total_mapping_dict:
    for page in total_mapping_dict[key]:
        counter += 1
print(f"Total page-image mappings: {counter}")

# %% [markdown]
# ## 4. Asynchronous Image Processing with Gemini

# %%
# Initialize CSV file for multimodal intermediate data
def initialize_intermediate_csv(file_path):
    """Create CSV file with headers on first run."""
    try:
        with open(file_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["joining_key", "page", "description"])
        print(f"Created new file: {file_path}")
    except Exception as e:
        print(f"Error creating file: {e}")

# %%
# Read existing processed data to avoid reprocessing
def get_processed_pages(file_path):
    """Get dictionary of already processed pages."""
    try:
        df = pd.read_csv(file_path)
        df_grouped = df.groupby('joining_key')['page'].apply(list).reset_index()
        return dict(zip(df_grouped['joining_key'], df_grouped['page']))
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}

# %%
# Initialize or load existing data
initialize_intermediate_csv(multimodal_intermediate_path)
check_dict = get_processed_pages(multimodal_intermediate_path)

# %%
# Filter out already processed items
flatten_list = []
for key, pages in total_mapping_dict.items():
    for page, img_path in pages.items():
        # Skip if already processed
        if key in check_dict and int(page) in check_dict[key]:
            continue
        flatten_list.append((key, page, img_path))

print(f"Remaining images to process: {len(flatten_list)}")

# %%
# Define async functions for image processing
async def explain_image(row):
    """Process a single image with Gemini."""
    start_time = time.time()
    joining_key, page, img_path = row
    
    print(f"{img_path} is requesting")
    
    try:
        # Get Gemini response
        response = await asyncio.to_thread(
            get_gemini_response,
            model,
            SYSTEM_PROMPT,
            img_path
        )
        
        print(f"{img_path} received the response")
        
        if response == "ERROR":
            print(f"{img_path} has ERROR.")
        else:
            # Write result to CSV
            async with aiofiles.open(multimodal_intermediate_path, 'a') as f:
                writer = csv.writer(f)
                await writer.writerow([joining_key, page, response])
            print(f"{img_path} finished writing")
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
    
    end_time = time.time()
    print(f"{img_path} completed. Execution time: {end_time - start_time:.2f} seconds")

async def process_batch(batch_list):
    """Process a batch of images concurrently."""
    tasks = [explain_image(row) for row in batch_list]
    await asyncio.gather(*tasks)

async def main():
    """Main async function to process all images in batches."""
    total_items = len(flatten_list)
    
    for i in range(0, total_items, BATCH_SIZE):
        start_idx = i
        end_idx = min(i + BATCH_SIZE, total_items)
        batch = flatten_list[start_idx:end_idx]
        
        start_time = time.time()
        await process_batch(batch)
        end_time = time.time()
        
        print(f"Batch {i//BATCH_SIZE + 1} completed. Time: {end_time - start_time:.2f} seconds")
        
        # Rate limiting
        if end_idx < total_items:
            print("Waiting 50 seconds before next batch...")
            time.sleep(50)

# %%
# Set up exception handler and run
def exception_handler(loop, context):
    print('Exception handler called')
    print(context)

loop = asyncio.get_event_loop()
loop.set_exception_handler(exception_handler)
loop.run_until_complete(main())

# %% [markdown]
# ## 5. Multimodal Ingestion to Elasticsearch

# %%
# Configuration for UAT or PROD
ENVIRONMENT = 'UAT'  # Change to 'PROD' for production

if ENVIRONMENT == 'UAT':
    es_url = "https://amlp-es-co01.uat.dbs.com:9200"
    collection_name = 'datasets-kb.policy-ibg.sg.uat.test_v3'
else:  # PROD
    es_url = "https://amlp-es-co01.sgp.dbs.com:9200"
    collection_name = 'datasets-kb.kb-bg.sg.prod'

print(f"Environment: {ENVIRONMENT}")
print(f"Collection: {collection_name}")

# %%
# Connect to Elasticsearch
es_store_instance, es_instance = connect_to_elk(collection_name, es_url)
print("Connected successfully!")

# %%
# Check existing chunks
total_chunks = get_total_chunks(es_instance, collection_name)
print(f"Total existing chunks: {total_chunks}")

# %%
# Load metadata and multimodal data
metadata_fill_in_df = pd.read_csv(metadata_fill_in_path)
metadata_multimodal_df = pd.read_csv(multimodal_intermediate_path)

print(f"Metadata records: {len(metadata_fill_in_df)}")
print(f"Multimodal records: {len(metadata_multimodal_df)}")

# %%
# Create document chunks
data_type = 'image'
all_document_chunk_list = create_chunks_from_multimodal_documents(
    metadata_fill_in_df,
    metadata_multimodal_df,
    data_type,
    BATCH_NUM
)

print(f"Total chunks created: {len(all_document_chunk_list)}")

# %%
# Ingest into Elasticsearch
failed_docs = ingest(
    all_document_chunk_list=all_document_chunk_list,
    es_store_instance=es_store_instance,
    es_instance=es_instance,
    collection_name=collection_name,
    batch_size=INGESTION_BATCH_SIZE,
    timer_per_batch=TIMER_PER_BATCH
)

print(f"Failed documents: {len(failed_docs) if failed_docs else 0}")

# %%
# Verify final count
final_total = get_total_chunks(es_instance, collection_name)
print(f"Final total chunks: {final_total}")
print("Ingestion completed successfully!")

# %% [markdown]
# ## 6. Summary

# %%
print("=" * 50)
print("MULTIMODAL INGESTION SUMMARY")
print("=" * 50)
print(f"Environment: {ENVIRONMENT}")
print(f"Collection: {collection_name}")
print(f"Total images extracted: {len(total_mapping_dict)}")
print(f"Total processed: {len(metadata_multimodal_df)}")
print(f"Total chunks ingested: {len(all_document_chunk_list)}")
print(f"Final chunks in Elasticsearch: {final_total}")
if failed_docs:
    print(f"⚠️ Failed documents: {len(failed_docs)}")
else:
    print("✅ All documents ingested successfully!")
print("=" * 50)