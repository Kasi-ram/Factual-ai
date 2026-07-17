"""
Chunk Service for Factual-ai
=============================
This service splits document text into smaller, overlapping chunks (size: 500 chars, overlap: 100).
It implements a lightweight, pure-python recursive text splitter that behaves like 
LangChain's text splitter. This eliminates the heavy langchain dependency, improving 
startup time and throughput.
"""

class ChunkService:
    """
    Lightweight recursive character text splitter service.
    """

    def __init__(self):
        self.chunk_size = 500
        self.chunk_overlap = 100
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list:
        """
        Splits a string into overlapping chunks.
        
        Args:
            text: Input string.
            
        Returns:
            A list of string chunks.
        """
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text]

        # Find the first separator present in the text
        separator = ""
        for sep in self.separators:
            if sep in text:
                separator = sep
                break

        # Split text by separator
        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)

        chunks = []
        current_chunk = []
        current_len = 0

        for split in splits:
            split_len = len(split)
            if separator != "":
                split_len += len(separator)
                
            # If adding this split exceeds size, finalize the current chunk
            if current_len + split_len > self.chunk_size:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                
                # Backtrack to apply chunk overlap
                overlap_chars = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    item_len = len(item)
                    if separator != "":
                        item_len += len(separator)
                    if overlap_len + item_len <= self.chunk_overlap:
                        overlap_chars.insert(0, item)
                        overlap_len += item_len
                    else:
                        break
                current_chunk = overlap_chars
                current_len = overlap_len

            current_chunk.append(split)
            current_len += split_len

        # Append last chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def split_pages(self, pages: list) -> list:
        """
        Helper method to split multiple page objects.
        
        Args:
            pages: List of dicts containing 'text' and 'page'.
            
        Returns:
            List of chunk dicts containing 'text' and 'page'.
        """
        chunks = []
        for page in pages:
            page_chunks = self.split_text(page["text"])
            for chunk in page_chunks:
                chunks.append({
                    "text": chunk,
                    "page": page["page"]
                })
        return chunks
