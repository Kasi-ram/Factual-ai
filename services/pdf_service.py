"""
PDF Service for Factual-ai
===========================
This service extracts text pages from PDF documents.
It uses pypdf (pure-python PDF library) to parse bytes, extract text from individual pages,
and compile them into (page_number, text) tuples.
"""

import io
from pypdf import PdfReader

class PDFService:
    """
    Parses PDF bytes and extracts text by page.
    """

    def extract_pages(self, pdf_bytes: bytes) -> list:
        """
        Extracts pages from raw PDF bytes.
        
        Args:
            pdf_bytes: Raw bytes of the uploaded PDF file.
            
        Returns:
            A list of tuples (page_number, page_text_content).
        """
        pages = []
        # Create a file-like stream from the raw bytes
        stream = io.BytesIO(pdf_bytes)
        reader = PdfReader(stream)

        # Enumerate and extract text from each page
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                if text and text.strip():
                    pages.append(
                        (
                            page_number,
                            text.strip()
                        )
                    )
            except Exception:
                # Silently skip pages that cannot be parsed
                continue

        return pages

    def extract_text(self, pdf_bytes: bytes) -> str:
        """
        Compiles all page text from the PDF into a single string.
        """
        pages = self.extract_pages(pdf_bytes)
        return "\n".join(
            page_text
            for _, page_text in pages
        )
