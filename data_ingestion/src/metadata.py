"""
metadata.py

Functions to detect and parse date strings, extract document titles/dates
from PDF text, and build/export a metadata table for a directory of files.

NOTE: This file was reconstructed from a badly OCR-corrupted source. The
overall structure, function names, and logic follow the original docstrings
and code fragments as closely as possible. A few details (exact regex
patterns, exact column names) were inferred from context and may need
adjustment to match the original behavior exactly.
"""

from src.utils import convert_pdf_to_str
from urllib.parse import unquote
from urllib.request import pathname2url
from datetime import date
import re
import csv
import os
import pandas as pd


# ---------------------------------------------------------------------------
# Date detection / parsing
# ---------------------------------------------------------------------------
#
# Functions to detect and parse date strings.
# This module defines regex patterns and a function to check if a string
# matches expected date formats.
#
# The date regex patterns match formats like:
#   - 02Jan2020
#   - January 2020
#   - 02Jan20
#   - 2020.11.14
#   - 09Apr24
#   - Aug 2023
#
# TODAY constant stores today's date in YYMMDD format for comparison.

date_format_1 = r"^\d{2}[A-Za-z]{3}\d{4}$"      # e.g. 02Jan2020
date_format_2 = r"^[A-Za-z]+\s\d{4}$"           # e.g. January 2020
date_format_3 = r"^\d{2}[A-Za-z]{3}\d{2}$"      # e.g. 02Jan20
date_format_4 = r"^\d{4}\.\d{2}\.\d{2}$"        # e.g. 2020.11.14
date_format_5 = r"^\d{4}$"                      # e.g. 2020
date_format_6 = r"^\d{4}\.\d{2}\.\d{2}$"        # e.g. 2020.11.14

TODAY = date.today().strftime("%y%m%d")

month_dict = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
    "January": "01", "February": "02", "March": "03", "April": "04",
    "May": "05", "June": "06", "July": "07", "August": "08",
    "September": "09", "October": "10", "November": "11", "December": "12",
}


def check_date_format(date_string):
    """Check if a string matches any of the defined date regex formats.

    Args:
        date_string (str): The date string to check.

    Returns:
        bool: True if the date format matches, False otherwise.
    """
    if re.match(date_format_1, date_string):
        return True
    elif re.match(date_format_2, date_string):
        return True
    elif re.match(date_format_3, date_string):
        return True
    elif re.match(date_format_4, date_string):
        return True
    elif re.match(date_format_5, date_string):
        return True
    elif re.match(date_format_6, date_string):
        return True
    else:
        return False


def extract_date(date_string):
    """Extract and format date strings into YYMMDD format.

    This parses date strings using different regex patterns and outputs
    a standardized YYMMDD format. It handles formats like:
        - "02Jan2020"
        - "January 2020"
        - "02Jan20"
        - "2020.11.14"
        - "09Apr24"
        - "Aug 2023"

    A dictionary is used to map month name abbreviations to numbers.

    Args:
        date_string (str): The input date string.

    Returns:
        str: The formatted YYMMDD date, or None if no match.
    """
    formatted_date = None

    # Match DDMonYYYY, e.g. "02Jan2020"
    if re.search(r"(\d{2})([A-Za-z]{3,})(\d{4})", date_string):
        match = re.search(r"(\d{2})([A-Za-z]{3,})(\d{4})", date_string)
        day = match.group(1)
        month_str = match.group(2).capitalize()
        month = month_dict.get(month_str[:3])
        year = match.group(3)[2:]
        if month:
            formatted_date = year + month + day

    # Match DDMonYY, e.g. "02Jan20" / "09Apr24"
    elif re.search(r"(\d{2})([A-Za-z]{3,})(\d{2})$", date_string):
        match = re.search(r"(\d{2})([A-Za-z]{3,})(\d{2})$", date_string)
        day = match.group(1)
        month_str = match.group(2).capitalize()
        month = month_dict.get(month_str[:3])
        year = match.group(3)
        if month:
            formatted_date = year + month + day

    # Match Month YYYY / Mon YYYY, e.g. "January 2020", "Aug 2023"
    elif re.search(r"([A-Za-z]+)\s(\d{4})", date_string):
        match = re.search(r"([A-Za-z]+)\s(\d{4})", date_string)
        month_str = match.group(1)
        month = month_dict.get(month_str)
        year = match.group(2)[2:]
        if month:
            formatted_date = year + month + "01"

    # Match YYYY.MM.DD, e.g. "2020.11.14"
    elif re.search(r"(\d{4})\.(\d{2})\.(\d{2})", date_string):
        match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", date_string)
        formatted_date = match.group(1)[2:] + match.group(2) + match.group(3)

    # Match bare YYYY
    elif re.search(r"^(\d{4})$", date_string):
        match = re.search(r"^(\d{4})$", date_string)
        formatted_date = match.group(1)[2:] + "01" + "01"

    else:
        formatted_date = None

    return formatted_date


def remove_dates_and_versions(text):
    """Remove version numbers and dates from text.

    Args:
        text (str): The input text.

    Returns:
        str: Text with dates and versions stripped out.

    This removes, using pattern matching:
        - Versions like "v1.2", "Version: 1.5"
        - Dates in various formats: 150816, 20230115
        - Dates preceded by key phrases like "Date:", "Last Reviewed:"
        - Dates with month names, e.g. "Jan 2023", "January 2023"

    It uses regex substitutions to strip the date and version patterns,
    removes extra whitespace, and returns the stripped string.
    """
    # Remove versions in the format "v1.2" or "Version: 1.2"
    text = re.sub(r"[vV]ersion\s*:?\s*\d+\.\d+", "", text)
    text = re.sub(r"\bv\d+\.\d+\b", "", text)

    months = (
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
        r"|January|February|March|April|May|June|July|August"
        r"|September|October|November|December)"
    )

    # Numeric dates, e.g. 150816, 20230115
    text = re.sub(r"\b\d{6}\b", "", text)
    text = re.sub(r"\b\d{8}\b", "", text)
    text = re.sub(r"\b\d{4}\.\d{2}\.\d{2}\b", "", text)

    # Jan 2023 / January 2023
    text = re.sub(months + r"\s\d{4}", "", text)

    # Date: Jan 2023 / Date: January 2023
    text = re.sub(r"Date:?\s*" + months + r"\s\d{4}", "", text, flags=re.IGNORECASE)

    # Last Reviewed: Jan 2023 / Last Reviewed January 2023
    text = re.sub(
        r"Last Reviewed:?\s*" + months + r"\s\d{4}", "", text, flags=re.IGNORECASE
    )

    # DDMonYYYY / DDMonYY, e.g. 02Jan2020, 02Jan20
    text = re.sub(r"\d{1,2}" + months + r"\d{2,4}", "", text)

    # Bare 4-digit year
    text = re.sub(r"\b(19|20)\d{2}\b", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Document title / date extraction
# ---------------------------------------------------------------------------

def get_document_name_date(st):
    """Extract document title and date from extracted PDF text.

    Args:
        st (str): The raw text extracted from a PDF document.

    Returns:
        tuple:
            title (str): The extracted document title.
            date (str): The extracted document date in YYMMDD format.

    This function tries to extract the most likely title paragraph and date
    from a block of PDF text. Various heuristics are used to filter
    paragraphs that look like document titles vs header/footer text.
    """
    st_lst = st.split("\n\n")
    new_st_lst = []
    for item in st_lst:
        item_lst = [line.strip() for line in item.split("\n")]
        item_lst = [line for line in item_lst if len(line) > 3]
        if len(item_lst) > 1:
            new_st_lst.append(" ".join(item_lst))
        else:
            item_lst = item.split("\n")
            if len(item_lst) >= 1:
                new_st_lst.append(item)

    new_st_lst = [item.strip() for item in new_st_lst]
    new_st_lst = [item for item in new_st_lst if len(item) > 3]

    dates = [extract_date(re.sub(r"-", "", text)) for text in new_st_lst]
    dates = [d for d in dates if d is not None]
    dates = [d for d in dates if d <= TODAY]

    date = max(dates) if len(dates) > 0 else None

    try:
        i = 0
        title = new_st_lst[i]
        while (
            title.lower().startswith("for")
            or title.lower().startswith("to be")
            or title.lower().startswith("(")
            or title.lower().startswith("dear")
            or title.lower().startswith("for internal reference and circulation only")
            or title.lower().startswith("private & confidential")
            or "monetary authority of singapore" in title.lower()
            or "institutional banking group" in title.lower()
            or "ibg" in title.lower()
            or "moody's analytics knowledge services" in title.lower()
            or title.split(" ")[-1] in ["on", "from", "of", "and", "&", "with", "or"]
            or check_date_format(title)
        ):
            i += 1
            title = new_st_lst[i]

        title = " ".join(new_st_lst[: i + 1])
        title = title.split("Note:")[0]
        title = title.split("Appendix")[0]
        title = clean_text(title)

        return title, date
    except IndexError:
        return None, None


def clean_text(title):
    """Clean text in a document title by replacing non-standard quotes.

    Args:
        title (str): The document title text to clean.

    Returns:
        str: The cleaned title text with standardized single and double quotes.

    For example:
        Input:  The document's title with "fancy" quotes
        Output: The document's title with "fancy" quotes
    """
    title = title.replace("\u2018", "'")
    title = title.replace("\u2019", "'")
    title = title.replace("\u201c", '"')
    title = title.replace("\u201d", '"')
    return title


# ---------------------------------------------------------------------------
# Metadata table construction
# ---------------------------------------------------------------------------

def create_metadata(file_directory):
    """Create a DataFrame containing metadata extracted from files.

    Args:
        file_directory (str): The path to the directory containing files to process.

    Returns:
        pd.DataFrame: A DataFrame with extracted file metadata.

    Walks the given directory and builds a DataFrame row by row with metadata
    extracted from each file:
        - Joining key based on filename
        - Document name based on filename
        - Document name based on parsing first page
        - URL/hyperlink
        - Category (folder name)
        - Date based on filename
        - Date based on first page
    """
    metadata_df = pd.DataFrame(
        columns=[
            "joining_key",
            "document_name_based_on_filename",
            "document_name_based_on_first_page",
            "hyperlink",
            "category",
            "date_based_on_filename",
            "date_based_on_first_page",
        ]
    )

    duplicate_check_list = []  # capture duplicated files if any

    for root, dirs, files in os.walk(file_directory):
        for file in files:
            file_full_path = os.path.join(root, file)
            identifier_str = file + str(os.path.getsize(file_full_path))

            if identifier_str in duplicate_check_list:
                # drop duplicated files with same file name and file size
                print(f"Duplicate Files Found: {file_full_path}")
                continue
            else:
                duplicate_check_list.append(identifier_str)

            joining_key = re.sub(r"[^a-zA-Z0-9]", "", file.lower())
            category = root.split(os.environ["CDSW_SHAREPOINT_PATH"])[-1]
            if category.startswith("/"):
                category = category[1:]  # remove the first character if it is /

            hyperlink = os.path.join(
                os.environ["SHAREPOINT_URL"], pathname2url(category + "/" + file)
            )

            filename = ".".join(file.split(".")[0:-1])
            date2 = extract_date(filename.replace("-", ""))

            try:
                st = convert_pdf_to_str(path=file_full_path)
                if len(st) > 0:
                    title, doc_date = get_document_name_date(st=st)
                    row_list = [
                        joining_key,
                        filename,  # fill in manually
                        clean_text(title) if title else remove_dates_and_versions(filename),
                        hyperlink,
                        category,
                        date2,
                        doc_date,  # fill in manually
                    ]
                else:
                    row_list = [
                        joining_key,
                        filename,  # fill in manually
                        None,
                        hyperlink,
                        category,
                        date2,
                        None,  # fill in manually
                    ]
                    print(f"Empty/picture document: {file_full_path}")

                metadata_df.loc[len(metadata_df)] = row_list
            except Exception:
                print(f"Failed parsing PDF: {file_full_path}")
                row_list = [
                    joining_key,
                    filename,  # fill in manually
                    None,
                    hyperlink,
                    category,
                    None,  # fill in manually
                    None,
                ]
                metadata_df.loc[len(metadata_df)] = row_list

    return metadata_df


def clean_and_export_metadata(metadata_file_in_path, metadata_file_in_utf8_path):
    """Clean up metadata CSV, dedupe documents, and export to UTF-8 CSV.

    Args:
        metadata_file_in_path (str): Path to the raw input metadata CSV.
        metadata_file_in_utf8_path (str): Path to write the cleaned UTF-8 CSV.

    Returns:
        None

    Reads the input CSV, removes duplicate document names, removes
    non-UTF-8 characters from names, and writes the cleaned metadata to the
    output CSV in UTF-8 encoding. Keeps only the latest version of
    duplicates based on date.
    """
    doc_dict = {}
    # remove duplicates with same document name and same date of issue but
    # different file names, only keep the latest version
    # remove non-utf-8 characters in document name

    with open(metadata_file_in_path, "r", encoding="UTF-8", errors="ignore") as infile:
        reader = list(csv.reader(infile))

    header = reader[0]
    doc_name_index = header.index("document_name")
    issue_of_date_index = header.index("date_of_issue")

    for index, row in enumerate(reader):
        if index == 0:
            continue

        row[doc_name_index] = re.sub(r"\s\s+", " ", row[doc_name_index])
        # remove extra space caused by non-utf-8 characters in document name

        if row[doc_name_index] in doc_dict:
            print(f"Document Name: {row[doc_name_index]}")
            print(f"Existing Version: {int(doc_dict[row[doc_name_index]][issue_of_date_index])}")
            print(f"New Version: {int(row[issue_of_date_index])}")
            if int(row[issue_of_date_index]) <= int(
                doc_dict[row[doc_name_index]][issue_of_date_index]
            ):
                continue

        doc_dict[row[doc_name_index]] = row

    with open(metadata_file_in_utf8_path, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        for doc_name in doc_dict:
            writer.writerow(doc_dict[doc_name])

    return None


# ---------------------------------------------------------------------------
# Historical metadata batches
# ---------------------------------------------------------------------------

def check_historical_metadata_exists(metadata_dir):
    """Check if metadata files exist in a directory.

    Args:
        metadata_dir (str): Path to the directory to check for metadata files.

    Returns:
        bool: True if any metadata files are found, False otherwise.

    Looks for subdirectory names starting with 'batch', then checks those
    directories for any files containing 'metadata' in the name. This
    indicates historical/previous metadata files exist in the structure:
        metadata/batch1
        metadata/batch2
        ...
    """
    for pos_dir in os.listdir(metadata_dir):
        if pos_dir.startswith("batch"):
            pos_dir_full_path = os.path.join(metadata_dir, pos_dir)
            if os.path.isdir(pos_dir_full_path):
                for file in os.listdir(pos_dir_full_path):
                    if "metadata" in file:
                        return True

    return False


def get_metadata_batches(metadata_dir):
    """Get list of metadata batch names from subdirectory names.

    Args:
        metadata_dir (str): Path to the base metadata directory.

    Returns:
        list: A list of batch name strings.

    Looks for subdirectories under metadata_dir starting with 'batch' and
    returns a list of those batch names.
    """
    batches = []
    for pos_dir in os.listdir(metadata_dir):
        if pos_dir.startswith("batch"):
            batches.append(pos_dir)
    return batches


def get_historical_metadata(metadata_dir):
    """Load historical metadata CSVs into a combined DataFrame.

    Args:
        metadata_dir (str): Base path containing metadata CSVs in batch subdirs.

    Returns:
        pd.DataFrame: DataFrame with concatenated historical metadata.

    Checks for existence of batch metadata files. Iterates batch subdirs and
    reads CSVs into DataFrames. Concatenates into a single DataFrame with a
    'batch' column.
    """
    metadata_df = pd.DataFrame(
        columns=["joining_key", "document_name", "hyperlink", "category", "date_of_issue", "batch"]
    )

    if check_historical_metadata_exists(metadata_dir=metadata_dir):
        batches = get_metadata_batches(metadata_dir=metadata_dir)

        for batch in batches:
            batch_metadata_df = pd.read_csv(
                os.path.join(metadata_dir, batch, "metadata_file_in_utf8.csv")
            )
            batch_metadata_df["batch"] = batch
            metadata_df = pd.concat([metadata_df, batch_metadata_df], axis=0)

    return metadata_df
