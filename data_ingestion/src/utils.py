"""
utils.py

Utility functions for converting PDF files to text or HTML strings using
pdfminer.

NOTE: This file was reconstructed from a badly OCR-corrupted source. The
overall structure and logic follow the original docstring and code
fragments as closely as possible. Some details (exact page-selection
control flow) were inferred from context and may need adjustment to match
the original behavior exactly.
"""

from io import BytesIO, StringIO

from pdfminer.converter import HTMLConverter, TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage


def convert_pdf_to_str(path, output_html=False, select_pages=None, skip_pages=None):
    """Convert a PDF file to text or HTML string.

    Args:
        path (str): Path to the PDF file to convert.
        output_html (bool, optional): Whether to output HTML instead of
            simple text. Defaults to False.
        select_pages (list, optional): List of page numbers (0-indexed) to
            select. Defaults to None, which selects all pages.
        skip_pages (list, optional): List of page numbers (0-indexed) to
            skip. Defaults to None, which skips no pages.

    Returns:
        str: The text or HTML contents of the PDF as a string.

    The function leverages pdfminer to parse the PDF and convert it to a
    string. select_pages and skip_pages can be used to control which pages
    are processed.
    """
    rsrcmgr = PDFResourceManager(codec="utf-8")
    laparams = LAParams()

    if output_html:
        retstr = BytesIO()
        device = HTMLConverter(rsrcmgr, retstr, codec="utf-8", laparams=laparams)
    else:
        retstr = StringIO()
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    fp = open(path, "rb")
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    password = ""
    maxpages = 0
    caching = True
    pagenos = set()

    for idx, page in enumerate(
        PDFPage.get_pages(
            fp,
            pagenos,
            maxpages=maxpages,
            password=password,
            caching=caching,
            check_extractable=True,
        )
    ):
        if select_pages:
            if idx not in select_pages:
                continue

        if skip_pages and idx in skip_pages:
            continue

        interpreter.process_page(page)

    output = retstr.getvalue()

    if output_html:
        output = output.decode()

    fp.close()
    device.close()
    retstr.close()

    return output
