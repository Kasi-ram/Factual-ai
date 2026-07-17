"""
Chroma Service for Factual-ai
=============================
This module interfaces with the local ChromaDB persistent vector database.
It handles indexing document chunks, query semantic searches (using cosine distance),
counting chunks per session, and clearing document chunks on reset.
"""

import chromadb
import hashlib
from pathlib import Path

class ChromaService:
    """
    Service to manage local persistent collections in ChromaDB.
    """

    # Local project root path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    
    # Store database in the project's root folder under chroma_db
    DB_PATH = str(PROJECT_ROOT / "chroma_db")
    
    # Collection name changed to be general-genre
    COLLECTION_NAME = "factual_documents"

    def __init__(self):
        # Initialize persistent chromadb client
        self.client = chromadb.PersistentClient(
            path=self.DB_PATH
        )
        self._load_collection()

    def _load_collection(self):
        """
        Retrieves or creates a named collection in Chroma with Cosine distance metric.
        """
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        chunks,
        embeddings,
        source,
        document_id,
        knowledge_base_id
    ):
        """
        Upserts document chunks, metadatas, and vector embeddings into the collection.
        
        Args:
            chunks: List of chunk dicts containing text and page number.
            embeddings: List of matching numeric vector embeddings.
            source: Filename of the uploaded document.
            document_id: Hash value of the document.
            knowledge_base_id: Session-specific database namespace.
        """
        # Generate unique hash ids for each chunk
        ids = [
            hashlib.sha256(
                f"{document_id}:{index}".encode("utf-8")
            ).hexdigest()
            for index, _ in enumerate(chunks)
        ]

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        metadatas = [
            {
                "source": source,
                "page": chunk["page"],
                "document_id": document_id,
                "knowledge_base_id": knowledge_base_id
            }
            for chunk in chunks
        ]

        # Upsert the vectors and data into Chroma
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(
        self,
        query_embedding,
        knowledge_base_id,
        limit=10
    ):
        """
        Performs similarity search against the current namespace.
        
        Args:
            query_embedding: The embedded user question vector.
            knowledge_base_id: Session-specific database namespace.
            limit: Maximum matching results to return.
            
        Returns:
            A list of dicts with document text, distance score, and metadata.
        """
        where = {
            "knowledge_base_id": knowledge_base_id
        }

        total_count = self.collection.count()
        if total_count == 0:
            return []

        # Query using the query embedding vector
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, total_count),
            where=where
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "document": document,
                "metadata": metadata,
                "distance": distance
            }
            for document, metadata, distance
            in zip(documents, metadatas, distances)
        ]

    def count(self, knowledge_base_id):
        """
        Counts the total active chunks in the current session namespace.
        """
        results = self.collection.get(
            where={"knowledge_base_id": knowledge_base_id},
            include=[]
        )
        return len(results["ids"])

    def distance_metric(self):
        """
        Returns the configured distance space space metric (default: l2).
        """
        metadata = self.collection.metadata or {}
        return metadata.get("hnsw:space", "l2")

    def delete_document(self, document_id, knowledge_base_id):
        """
        Deletes all chunks belonging to a specific document under a namespace.
        """
        self.collection.delete(
            where={
                "$and": [
                    {"document_id": document_id},
                    {"knowledge_base_id": knowledge_base_id}
                ]
            }
        )

    def clear(self, knowledge_base_id):
        """
        Deletes all chunks belonging to a specific session namespace.
        """
        self.collection.delete(
            where={"knowledge_base_id": knowledge_base_id}
        )
