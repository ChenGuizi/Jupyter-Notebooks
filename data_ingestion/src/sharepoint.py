"""
sharepoint.py

Utilities for working with locally-synced SharePoint (iBG KB) folders:
finding newly updated files to ingest, inventorying file types, cleaning
up duplicate/undesired files, and reorganizing an old zip archive into a
new folder/zip structure based on an instruction CSV.

NOTE: This file was reconstructed from a badly OCR-corrupted source. The
overall structure, function names, and logic follow the original docstrings
and code fragments as closely as possible. Some details (exact default
paths, exact control flow in a few branches) were inferred from context and
may need adjustment to match the original behavior exactly.
"""

import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import pandas as pd


def identify_newly_updated_files_in_a_folder_and_copy_into_another_folder(
    date_to_check_against, ibankid=None, file_dir=None, new_file_dir=None
):
    """Identify files that have been recently updated in a folder and copy
    them to another folder.

    The purpose of this function is so that we can easily retrieve the new
    files that are required for ingestion rather than having to comb
    through manually or refer to past metadata. To use this function, go
    to the SharePoint folder online and click the sync button so that the
    folder is synced locally to your laptop.

    This function is meant to be used in a local desktop environment if you
    would like to use it on SharePoint folders that have been synced
    locally to your laptop. It can also be used in CDSW for files that are
    located in the CDSW project directory.

    Args:
        date_to_check_against (str): Date string in DD-MM-YYYY format. Files
            modified after this date will be copied. Suggested to use the
            date when the last file was added into the SharePoint, i.e. the
            last ingestion date. For instance, if the last ingestion took
            place on 25-04-2024, then use 25-04-2024.
        ibankid (str, optional): iBankID. Used to construct the default
            file_dir. The SharePoint folder would have been synced locally
            to your laptop and hence the file directory would contain your
            iBankID.
        file_dir (str, optional): Input directory path containing files.
        new_file_dir (str, optional): Output directory path to copy new
            files into.

    Defaults:
        If ibankid is provided, file_dir defaults to a SharePoint folder
        mapped locally. new_file_dir defaults to a "new" subfolder within
        file_dir.

    For each file in file_dir, checks the modification time. If greater
    than date_to_check_against (or less than a hard-coded cutoff date),
    copies the file to new_file_dir, preserving relative folder structure.

    Returns:
        None
    """
    if ibankid is not None:
        ibgkb_sharepoint_dir_locally_synced = (
            f"C://Users/{ibankid}/DBS Bank Ltd/IBG_KB - Documents"
        )

        if file_dir is None:
            file_dir = os.path.join(ibgkb_sharepoint_dir_locally_synced, "SG")

        if new_file_dir is None:
            new_file_dir = os.path.join(ibgkb_sharepoint_dir_locally_synced, "new/SG")

    for category in os.listdir(file_dir):
        category_file_dir = os.path.join(file_dir, category)
        for sub_category in os.listdir(category_file_dir):
            sub_category_file_dir = os.path.join(category_file_dir, sub_category)
            for file in os.listdir(sub_category_file_dir):
                filepath = os.path.join(sub_category_file_dir, file)

                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))

                if (
                    mtime > datetime.strptime(date_to_check_against, "%d-%m-%Y")
                ) or (mtime < datetime.strptime("2024-04-11", "%Y-%m-%d")):
                    new_filepath = new_file_dir + filepath.split(file_dir)[1]
                    new_file_parent_dir = os.path.dirname(new_filepath)

                    if not os.path.exists(new_file_parent_dir):
                        os.makedirs(new_file_parent_dir)

                    shutil.copyfile(filepath, new_filepath)

    return None


def identify_type_of_files_in_a_folder(ibankid=None, file_dir=None):
    """Scan a folder and identify the file types based on extensions.

    This function may be used to understand what types of files are in the
    folder.

    Args:
        ibankid (str, optional): User iBankID. Used to construct default
            file_dir if not provided.
        file_dir (str, optional): Root folder path to scan.

    Defaults:
        If ibankid provided but file_dir is None, file_dir defaults to a
        SharePoint folder mapped locally.

    Recursively scans file_dir and constructs a list of unique file
    extensions found.

    Returns:
        ext_lst (list of str): List of unique file extensions.
    """
    if ibankid is not None:
        ibgkb_sharepoint_dir_locally_synced = (
            f"C://Users/{ibankid}/DBS Bank Ltd/IBG_KB - Documents"
        )

        if file_dir is None:
            file_dir = os.path.join(ibgkb_sharepoint_dir_locally_synced, "SG")

    ext_lst = []
    for category in os.listdir(file_dir):
        category_file_dir = os.path.join(file_dir, category)
        for sub_category in os.listdir(category_file_dir):
            sub_category_file_dir = os.path.join(category_file_dir, sub_category)
            for file in os.listdir(sub_category_file_dir):
                filepath = os.path.join(sub_category_file_dir, file)
                file_name, file_extension = os.path.splitext(file)

                if file_extension not in ext_lst:
                    ext_lst.append(file_extension)

    return ext_lst


def delete_files_of_certain_file_types_or_already_have_another_pdf_copy_in_the_folder(
    ibankid=None, file_dir=None, undesired_filetypes=[".mp4", ".msg"]
):
    """Delete files of certain file types or duplicates based on criteria.

    Args:
        ibankid (str, optional): The iBankID username used to construct a
            default synced SharePoint directory path if provided.
        file_dir (str, optional): The root directory path to search within.
            If None, defaults to a hard-coded SharePoint directory
            constructed from ibankid.
        undesired_filetypes (list, optional): A list of file extensions to
            delete if found. Defaults to ['.mp4', '.msg'].

    The function searches recursively within file_dir for files to delete
    based on:
        - File type is in undesired_filetypes list.
        - Filename without extension already exists as a .pdf version.

    Prints information about zip files and non-PDF files not yet converted.

    Returns:
        None
    """
    if ibankid is not None:
        ibgkb_sharepoint_dir_locally_synced = (
            f"C://Users/{ibankid}/DBS Bank Ltd/IBG_KB - Documents"
        )

        if file_dir is None:
            file_dir = os.path.join(ibgkb_sharepoint_dir_locally_synced, "SG")

    for category in os.listdir(file_dir):
        category_file_dir = os.path.join(file_dir, category)
        for sub_category in os.listdir(category_file_dir):
            sub_category_file_dir = os.path.join(category_file_dir, sub_category)
            for file in os.listdir(sub_category_file_dir):
                filepath = os.path.join(sub_category_file_dir, file)
                file_name, file_extension = os.path.splitext(file)

                if file_extension != ".pdf":
                    if file_extension in undesired_filetypes:
                        os.remove(filepath)
                    elif file_extension == ".zip":
                        print(f"Zip file: {filepath}")
                    else:
                        all_files = os.listdir(sub_category_file_dir)
                        all_files = [f for f in all_files if f != file]
                        all_files = [os.path.splitext(f)[0] for f in all_files]

                        if file_name in all_files:
                            os.remove(filepath)
                            print(f"Removed: {filepath}")
                        else:
                            print(f"Yet to be converted: {filepath}")

    return None


def reorganise_files(old_zip_filename, new_zip_filename, instruction_filename):
    """Reorganize files from an old zip according to instructions into a new zip.

    Note that this is an old function and may not work anymore.

    Args:
        old_zip_filename (str): Path to the old zip file to extract files from.
        new_zip_filename (str): Path to save the new zip file to.
        instruction_filename (str): Path to a CSV file with instructions for
            reorganizing the files. The instruction CSV should have columns:
            Category, Sub Category, document_name, hyperlink.

    The function:
        1. Reads in the instruction CSV.
        2. Unzips the old zip to a temporary folder.
        3. Flattens the folder structure.
        4. Creates a new folder structure based on the instructions CSV.
        5. Moves files into the new folder structure.
        6. Zips the new folder to the new_zip_filename.
        7. Deletes the temporary folders.

    Returns:
        None
    """
    # INSTRUCTION FILE PROCESSING #
    df = pd.read_csv(instruction_filename)
    df["document_name"] = df["document_name"] + ".pdf"
    df["document_name_from_hyperlink"] = df["hyperlink"].apply(
        lambda x: unquote(x.split("/")[-1])
    )
    df["document_name_to_use"] = [
        row["document_name"]
        if row["document_name"] == row["document_name_from_hyperlink"]
        else row["document_name_from_hyperlink"]
        for index, row in df.iterrows()
    ]

    # UNZIP OLD ZIP #
    new_folder_name = "data/tmp/new"
    old_folder_name = "data/tmp/old"

    with zipfile.ZipFile(old_zip_filename, "r") as zip_ref:
        zip_ref.extractall(old_folder_name)

    # FLATTEN OLD FILES #
    all_folders_in_old_ibg_kb = [x[0] for x in os.walk(old_folder_name)][1:]
    for folder in all_folders_in_old_ibg_kb:
        for file in os.listdir(folder):
            filepath = os.path.join(folder, file)
            if os.path.isfile(filepath):
                new_filepath = os.path.join(old_folder_name, file)
                os.rename(filepath, new_filepath)

    # MAKE NEW DIRECTORY FOR NEW FILES #
    if not os.path.exists(new_folder_name):
        os.mkdir(new_folder_name)

    for category in df["Category"].unique():
        new_category_folder = os.path.join(new_folder_name, category)
        if not os.path.exists(new_category_folder):
            os.mkdir(new_category_folder)

        for sub_category in df[df["Category"] == category]["Sub Category"].unique():
            new_sub_category_folder = os.path.join(new_category_folder, sub_category)
            if not os.path.exists(new_sub_category_folder):
                os.mkdir(new_sub_category_folder)

            filenames = list(
                df[
                    (df["Category"] == category)
                    & (df["Sub Category"] == sub_category)
                ]["document_name_to_use"].unique()
            )

            for filename in filenames:
                old_filepath = os.path.join(old_folder_name, filename)
                new_filepath = os.path.join(new_sub_category_folder, filename)
                os.rename(old_filepath, new_filepath)

    # ZIP NEW FILES #
    new_zip_filename = new_zip_filename.split(".")[0]
    shutil.make_archive(new_zip_filename, "zip", new_folder_name)
    shutil.rmtree("data/tmp")

    print("Successfully reorganised!")

    return None
