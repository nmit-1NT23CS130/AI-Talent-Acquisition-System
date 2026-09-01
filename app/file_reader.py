"""
file_reader.py
==============
Extract text from PDF, Word and Text files
"""

import pdfplumber
from docx import Document
import io

def extract_text(file) -> str:
    """
    Extract text from uploaded file.
    Supports: .pdf, .docx, .txt
    Returns: extracted text as string
    """
    filename = file.name.lower()

    # PDF
    if filename.endswith('.pdf'):
        return extract_from_pdf(file)

    # Word
    elif filename.endswith('.docx'):
        return extract_from_word(file)

    # Text
    elif filename.endswith('.txt'):
        return extract_from_txt(file)

    else:
        return ""


def extract_from_pdf(file) -> str:
    """Extract text from PDF file."""
    try:
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"PDF error: {e}")
        return ""


def extract_from_word(file) -> str:
    """Extract text from Word (.docx) file."""
    try:
        doc = Document(io.BytesIO(file.read()))
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Word error: {e}")
        return ""


def extract_from_txt(file) -> str:
    """Extract text from .txt file."""
    try:
        return file.read().decode(
            "latin-1", errors="ignore").strip()
    except Exception as e:
        print(f"Text error: {e}")
        return ""