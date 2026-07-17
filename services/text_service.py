"""
Text Service for Factual-ai
===========================
This service handles plain-text-based documents (TXT, MD, CSV, JSON, HTML, etc.).
It decodes text bytes into a single page with page_number = 1.
"""

class TextService:
    """
    Decodes text bytes into a single-page list structure.
    """

    def extract_pages(self, text_bytes: bytes) -> list:
        """
        Decodes raw bytes to a UTF-8 string and packs it into a single-page tuple list.
        
        Args:
            text_bytes: Raw bytes of the uploaded text file.
            
        Returns:
            A list containing a single tuple: [(1, decoded_text)].
        """
        try:
            # Attempt to decode as standard UTF-8
            text = text_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback to ISO-8859-1 if UTF-8 fails
            text = text_bytes.decode("latin-1")

        return [
            (
                1,
                text
            )
        ]
