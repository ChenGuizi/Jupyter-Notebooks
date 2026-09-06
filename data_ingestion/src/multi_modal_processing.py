"""
multi_modal_processing.py

Utilities for multimodal PDF processing: identifying pages containing
images/tables, converting PDF pages to PNG images, querying a Gemini
generative model for image descriptions, and building LangChain Document
chunks from multimodal metadata.

NOTE: This file was reconstructed from a badly OCR-corrupted source. The
overall structure, function names, and logic follow the original docstrings
and code fragments as closely as possible. Some details (exact thresholds,
exact metadata field wiring) were inferred from context and may need
adjustment to match the original behavior exactly.
"""

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import fitz
import pytz
from PIL import Image as PIL_Image
from tqdm import tqdm
from langchain_core.documents.base import Document
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Image,
)


def get_page_number_with_images(doc: Any, image_count_threshold: int = 3) -> List[int]:
    """Identify page numbers in a PDF document that contain images or tables.

    Args:
        doc (Any): An opened fitz (PyMuPDF) PDF document object.
        image_count_threshold (int): Maximum number of times an image can
            repeat across pages before it is considered a recurring
            logo/watermark rather than meaningful content. Defaults to 3.

    Returns:
        List[int]: A sorted, de-duplicated list of 1-indexed page numbers
        that contain images (excluding images that repeat on more pages
        than image_count_threshold) or tables.

    Iterates through each page of the document, collecting pages that
    contain tables, and building up a dictionary mapping each image object
    to the list of pages it appears on. Images that appear on more pages
    than image_count_threshold are treated as recurring boilerplate (e.g.
    logos) and excluded from the result.
    """
    img_page_list = []
    tab_page_list = []
    img_obj_dict = {}

    for page_number, page in enumerate(doc):
        image_list = page.get_images()
        tabs = page.find_tables()

        if tabs.tables:
            tab_page_list.append(page_number + 1)

        if image_list:
            # print(f"Found {len(image_list)} images on page {page_number+1}")
            for img in image_list:
                if img not in img_obj_dict:
                    img_obj_dict[img] = [page_number + 1]
                else:
                    img_obj_dict[img].append(page_number + 1)

    for key, value in img_obj_dict.items():
        if len(value) <= image_count_threshold:
            img_page_list.extend(value)
            # print(f"{key}: {value}")

    img_page_list.extend(tab_page_list)  # merge with pages that have tables

    print(f"List of pages: {list(dict.fromkeys(img_page_list))}\n")

    return list(dict.fromkeys(img_page_list))


def convert_single_pdf_to_pngs(
    path_to_pdf_file: str,
    path_to_save: str,
    joining_key: str,
    zoom_x: int = 3,
    zoom_y: int = 3,
) -> Dict[str, Dict[int, str]]:
    """Convert a PDF file to a series of PNG images.

    Args:
        path_to_pdf_file (str): The path to the PDF file.
        path_to_save (str): The directory to save the PNG images.
        joining_key (str): The joining key used to index the returned mapping.
        zoom_x (int): Horizontal zoom factor.
        zoom_y (int): Vertical zoom factor.

    Returns:
        Dict[str, Dict[int, str]]: A dictionary containing the joining key
        and a nested dictionary of page number to saved page-level image
        path.

    Only pages identified by get_page_number_with_images as containing
    tables/images are skipped (since they may need separate handling);
    all other pages are rendered to PNG and saved.
    """
    pdf_title = path_to_pdf_file.split("/")[-1].split(".pdf")[0]
    doc = fitz.open(path_to_pdf_file)
    mapping_dict = {joining_key: {}}

    mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor in each dimension
    page_list = get_page_number_with_images(doc)

    if not page_list:
        return None

    for page_number, page in enumerate(doc):
        if page_number + 1 not in page_list:
            continue

        pix = page.get_pixmap(matrix=mat)
        save_path = os.path.join(
            path_to_save, pdf_title, f"page_{str(page_number + 1)}.png"
        )

        mapping_dict[joining_key][page_number + 1] = save_path

        try:
            pix.save(save_path)
        except Exception as e:
            if "No such file or directory" in str(e):
                os.mkdir(os.path.join(path_to_save, pdf_title))
                pix.save(save_path)
            else:
                print(f"Save {pdf_title} Page {page_number + 1} Unsuccessfully: {e}")
                raise Exception(e)

    return mapping_dict


def get_gemini_response(model, system_prompt, img_path):
    """Get a Gemini generative model's text response for an image.

    Args:
        model: The GenerativeModel instance to query.
        system_prompt (str): The prompt/instructions to send along with the image.
        img_path (str): Path to the image file to send to the model.

    Returns:
        str: The stripped text response from the model, or "ERROR" if the
        call fails.

    Sets safety settings to block none of the harm categories, and uses a
    deterministic generation config (temperature 0) for consistent output.
    """
    safety_setting = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }

    generation_config = GenerationConfig(
        temperature=0,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        max_output_tokens=1024,
    )

    try:
        response = model.generate_content(
            [system_prompt, Image.load_from_file(img_path)],
            safety_settings=safety_setting,
            generation_config=generation_config,
        )
        return response.text.strip()
    except Exception as e:
        print(e)
        return "ERROR"


def create_chunks_from_multimodal_documents(
    metadata_fill_in_df,
    metadata_multimodal_df,
    data_type,
    batch_number,
):
    """Join multimodal metadata with document metadata and create chunked
    Document objects.

    Args:
        metadata_fill_in_df (DataFrame): DataFrame with document metadata.
        metadata_multimodal_df (DataFrame): DataFrame with multimodal
            (per-image/page) metadata, including a "description" column
            produced from the Gemini model.
        data_type (str): The original type of the data which the texts are
            being converted from.
        batch_number (int): Batch number of document ingestion.

    Returns:
        list: List of Document chunks for all files.

    For each row in metadata_multimodal_df, looks up the matching document
    in metadata_fill_in_df via "joining_key", builds a normalized
    "document_name_match" and "date_of_issue_match" for deduplication /
    versioning, and creates a LangChain Document with the image/page
    description as page_content. Rows with an empty description are
    skipped.
    """
    all_document_chunk_list = []

    for index, row in tqdm(
        metadata_multimodal_df.iterrows(),
        total=len(metadata_multimodal_df),
        desc="Parsing Progress",
    ):
        metadata_doc_name = metadata_fill_in_df.loc[
            metadata_fill_in_df["joining_key"] == row["joining_key"], "document_name"
        ].tolist()[0]

        metadata_doc_name_match = "".join(
            re.sub(r"[^a-zA-Z0-9\s]", "", metadata_doc_name)
            .lower()
            .strip()
            .split()
        )

        metadata_hyperlink = metadata_fill_in_df.loc[
            metadata_fill_in_df["joining_key"] == row["joining_key"], "hyperlink"
        ].tolist()[0]

        metadata_category = metadata_fill_in_df.loc[
            metadata_fill_in_df["joining_key"] == row["joining_key"], "category"
        ].tolist()[0]

        date_of_issue = metadata_fill_in_df.loc[
            metadata_fill_in_df["joining_key"] == row["joining_key"], "date_of_issue"
        ].tolist()[0]

        metadata_date_of_issue = (
            datetime(int(str(date_of_issue)[:4]), int(str(date_of_issue)[4:6]), int(str(date_of_issue)[6:]))
            .astimezone(pytz.utc)
            .strftime("%Y/%m/%d %H:%M:%S")
        )

        metadata_date_of_issue_match = int(date_of_issue)  # make sure the type is int

        # conversion of the timezone since server is utc
        metadata_date_of_ingestion = (
            datetime.now(pytz.timezone("Asia/Singapore"))
            .astimezone(pytz.utc)
            .strftime("%Y/%m/%d %H:%M:%S")
        )

        if row["description"].strip() == "":
            continue

        document_chunk = Document(
            page_content=row["description"].strip(),
            metadata={
                "document_name": metadata_doc_name,
                "document_name_match": metadata_doc_name_match,
                "hyperlink": f"{metadata_hyperlink}#page={row['page']}",
                "category": metadata_category,
                "page": row["page"],
                "date_of_issue_match": metadata_date_of_issue_match,
                "date_of_ingestion": metadata_date_of_ingestion,
                "date_of_issue": metadata_date_of_issue,
                "data_type": data_type,
                "batch_number": batch_number,
            },
        )

        all_document_chunk_list.append(document_chunk)

    return all_document_chunk_list
