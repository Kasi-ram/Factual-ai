"""
Document Ingestion Service for Factual-ai
=========================================
This service manages the workflow of uploading, processing, chunking, embedding, 
and indexing document files into ChromaDB. It also registers files in the
local DocumentRegistry metadata cache.
"""

import logging
import os
from services.chunk_service import ChunkService
from services.embedding_service import EmbeddingService
from services.chroma_service import ChromaService
from services.document_processor import DocumentProcessor
from services.document_registry import DocumentRegistry

class DocumentIngestionService:
    """
    Coordinates ingestion, chunking, embedding, and vector database indexing.
    """

    # Maximum file upload size limit (default: 20MB)
    MAX_UPLOAD_BYTES = int(
        os.getenv("MAX_UPLOAD_BYTES", 20 * 1024 * 1024)
    )

    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.chunk_service = ChunkService()
        self.embedding_service = EmbeddingService()
        self.chroma_service = ChromaService()
        self.registry = DocumentRegistry()
        self.logger = logging.getLogger(__name__)

    def ingest_document(self, uploaded_file, knowledge_base_id="default") -> dict:
        """
        Processes and indexes an uploaded file.
        
        Args:
            uploaded_file: Streamlit UploadedFile object.
            knowledge_base_id: Session-specific database namespace.
            
        Returns:
            A status dict indicating success, duplicate, empty, or failed.
        """
        file_bytes = uploaded_file.getvalue()

        # Reject files exceeding the maximum size limit
        if len(file_bytes) > self.MAX_UPLOAD_BYTES:
            return {
                "status": "rejected",
                "source": uploaded_file.name,
                "message": "The uploaded file is too large."
            }

        # Calculate SHA-256 hash to identify duplicates
        file_hash = self.registry.calculate_hash(file_bytes)

        if self.registry.exists(file_hash, knowledge_base_id):
            return {
                "status": "duplicate",
                "source": uploaded_file.name
            }

        indexed = False

        try:
            # Parse text/pages using DocumentProcessor
            pages = self.document_processor.process(uploaded_file)

            chunks = []
            for page_number, text in pages:
                page_chunks = self.chunk_service.split_text(text)
                for chunk_text in page_chunks:
                    if not chunk_text.strip():
                        continue
                    chunks.append(
                        {
                            "text": chunk_text,
                            "page": page_number
                        }
                    )

            if not chunks:
                return {
                    "status": "empty",
                    "source": uploaded_file.name,
                    "pages": len(pages),
                    "chunks": 0
                }

            # Embed document chunks sequentially
            embeddings = [
                self.embedding_service.embed(chunk["text"])
                for chunk in chunks
            ]

            # Index chunks and embeddings in ChromaDB
            self.chroma_service.add_chunks(
                chunks=chunks,
                embeddings=embeddings,
                source=uploaded_file.name,
                document_id=file_hash,
                knowledge_base_id=knowledge_base_id
            )

            indexed = True

            # Register document in metadata registry
            self.registry.register(
                file_hash=file_hash,
                filename=uploaded_file.name,
                pages=len(pages),
                chunks=len(chunks),
                knowledge_base_id=knowledge_base_id
            )

            return {
                "status": "success",
                "source": uploaded_file.name,
                "pages": len(pages),
                "chunks": len(chunks)
            }

        except Exception as error:
            self.logger.exception(
                "Document ingestion failed for %s",
                uploaded_file.name
            )

            # Rollback ChromaDB writes if registry fails midway
            if indexed:
                try:
                    self.chroma_service.delete_document(
                        file_hash,
                        knowledge_base_id
                    )
                except Exception:
                    self.logger.exception(
                        "Failed to roll back document %s",
                        uploaded_file.name
                    )

            return {
                "status": "failed",
                "source": uploaded_file.name,
                "message": "Document indexing failed. Please try again."
            }

    def reset(self, knowledge_base_id="default"):
        """
        Clears all documents in ChromaDB and DocumentRegistry for the active session namespace.
        """
        self.chroma_service.clear(knowledge_base_id)
        self.registry.clear(knowledge_base_id)

    def stats(self, knowledge_base_id="default") -> dict:
        """
        Retrieves active statistics (number of docs, number of chunks) for the namespace.
        """
        return {
            "documents": self.registry.count(knowledge_base_id),
            "chunks": self.chroma_service.count(knowledge_base_id)
        }

    def documents(self, knowledge_base_id="default") -> list:
        """
        Lists details of all registered files under the namespace.
        """
        return self.registry.list_documents(knowledge_base_id)
