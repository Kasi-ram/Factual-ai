"""
Embedding Service for Factual-ai
=================================
This service interacts with the Google Gemini API to generate semantic vector embeddings
for text chunks. It uses the configured Gemini API key and model (default: gemini-embedding-001).
"""

import os
from google import genai

class EmbeddingService:
    """
    Service wrapper for Google's Gemini text embedding API.
    """

    def __init__(self):
        # Initialize Google GenAI client using Gemini API key
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        # Configure embedding model, default to gemini-embedding-001
        self.model = os.getenv(
            "EMBEDDING_MODEL",
            "gemini-embedding-001"
        )

    def embed(self, text: str) -> list:
        """
        Generates a vector embedding (list of floats) for the input text.
        
        Args:
            text: A non-empty string.
            
        Returns:
            A list of floats representing the semantic embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Embedding text cannot be empty.")

        # Request vector embedding
        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )

        # Return the raw list of floats
        return response.embeddings[0].values
