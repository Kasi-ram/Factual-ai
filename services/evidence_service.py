"""
Evidence Service for Factual-ai
================================
This service selects the most relevant evidence chunks from the retrieved ChromaDB results.
To optimize throughput and reduce latency (by ~500ms to 1s per query), this service avoids
making a separate LLM call and instead returns the top 3 semantically closest chunks.
Grounding and factuality checking are performed in the downstream RAGService prompt itself.
"""

from typing import List, Dict, Any

class EvidenceService:
    """
    Selects top evidence chunks from similarity search results.
    """

    def select(self, question: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters and selects the best evidence chunks to answer the question.
        
        Args:
            question: The user's input query.
            results: A list of dicts containing document text, distance, and metadata.
            
        Returns:
            A list containing up to the top 3 most similar document chunks.
        """
        if not results:
            return []

        # Return the top 3 semantically closest chunks based on Chroma similarity distance.
        # This bypasses the overhead of an extra LLM call, greatly improving throughput.
        return results[:3]
