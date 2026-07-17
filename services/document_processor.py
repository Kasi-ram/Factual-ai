"""
Document Processor for Factual-ai
=================================
This class handles page extraction routing for different document file formats.
It extracts text pages from PDF files using PDFService, and handles plain text-based formats
(TXT, MD, CSV, JSON, HTML, HTM, XML) via TextService.
"""

from services.pdf_service import PDFService
from services.text_service import TextService

class DocumentProcessor:
    """
    Routes document byte processing to format-specific services.
    """

    def __init__(self):
        self.pdf_service = PDFService()
        self.text_service = TextService()

    def process(self, uploaded_file) -> list:
        """
        Parses document bytes based on file extension.
        
        Args:
            uploaded_file: Streamlit UploadedFile object.
            
        Returns:
            A list of tuples (page_number, page_text_content).
        """
        extension = (
            uploaded_file.name
            .split(".")[-1]
            .lower()
        )

        # Route PDF processing
        if extension == "pdf":
             return self.pdf_service.extract_pages(
                 uploaded_file.getvalue()
             )
        
        # Route plain text, markdown, CSV, JSON, HTML, etc.
        if extension in {"txt", "md", "csv", "json", "html", "htm", "xml"}:
            return self.text_service.extract_pages(
                uploaded_file.getvalue()
            )

        raise Exception(f"Unsupported file type: {extension}")
