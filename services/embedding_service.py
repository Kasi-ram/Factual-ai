"""
Embedding Service for Factual-ai
=================================
This service interacts with an OpenAI-compatible text embedding API (such as GenAILab, 
OpenAI, or Groq) to generate semantic vector embeddings for text chunks.
"""

import os
import httpx
from openai import OpenAI

class EmbeddingService:
    """
    Service wrapper for OpenAI-compatible text embedding APIs.
    """

    def __init__(self):
        # Configure client to bypass SSL verify similar to LLMService
        self.http_client = httpx.Client(verify=False)
        
        # Read API key, base URL, and model from environment variables, falling back to LLM values or defaults
        self.base_url = os.getenv(
            "EMBEDDING_BASE_URL",
            os.getenv("LLM_BASE_URL", "https://genailab.tcs.in")
        )
        self.api_key = os.getenv(
            "EMBEDDING_API_KEY",
            os.getenv("LLM_API_KEY", "sk-h4SzToxOqOneSAXq191PXA")
        )
        self.model = os.getenv(
            "EMBEDDING_MODEL",
            "azure_ai/genailab-maas-text-embedding-3-small"
        )

        # Initialize the OpenAI client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=self.http_client
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

        # Request vector embedding using standard OpenAI embeddings interface
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )

        # Return the raw list of floats
        return response.data[0].embedding

