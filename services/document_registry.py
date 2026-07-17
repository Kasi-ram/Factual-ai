"""
Document Registry for Factual-ai
================================
This class manages the list of uploaded documents and metadata in a local
JSON database file. The registry isolates files by session-specific knowledge_base_id.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime

class DocumentRegistry:
    """
    Metadata storage registry for uploaded documents, backed by document_registry.json.
    """

    # Database file is placed in the project root directory
    FILE = Path(__file__).resolve().parents[1] / "document_registry.json"

    def __init__(self):
        self._ensure_file()

    def _ensure_file(self):
        """
        Creates document_registry.json with an empty object if it doesn't exist.
        """
        if not self.FILE.exists():
            self.FILE.write_text("{}", encoding="utf-8")

    def _load(self) -> dict:
        """
        Reads registry JSON data. Supports backward compatibility migration.
        """
        self._ensure_file()
        try:
            data = json.loads(
                self.FILE.read_text(encoding="utf-8")
            )
            # Migrate legacy flat registry formats
            if data and all(
                "filename" in value
                for value in data.values()
                if isinstance(value, dict)
            ):
                return {"default": data}
            return data
        except json.JSONDecodeError:
            return {}

    def _save(self, data):
        """
        Writes data to the registry JSON file.
        """
        self.FILE.write_text(
            json.dumps(data, indent=4),
            encoding="utf-8"
        )

    def calculate_hash(self, file_bytes) -> str:
        """
        Computes SHA-256 hash string for document contents.
        """
        return hashlib.sha256(file_bytes).hexdigest()

    def exists(self, file_hash: str, knowledge_base_id="default") -> bool:
        """
        Checks if a document hash is already registered in the active namespace.
        """
        data = self._load()
        return file_hash in data.get(knowledge_base_id, {})

    def register(
        self,
        file_hash: str,
        filename: str,
        pages: int,
        chunks: int,
        knowledge_base_id="default"
    ):
        """
        Saves document details into the registry.
        
        Args:
            file_hash: SHA-256 string.
            filename: Original file name.
            pages: Document page count.
            chunks: Document chunk count.
            knowledge_base_id: Session namespace.
        """
        data = self._load()
        documents = data.setdefault(knowledge_base_id, {})
        documents[file_hash] = {
            "filename": filename,
            "pages": pages,
            "chunks": chunks,
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save(data)

    def list_documents(self, knowledge_base_id="default") -> list:
        """
        Lists files registered under the namespace.
        """
        return list(
            self._load().get(knowledge_base_id, {}).values()
        )

    def count(self, knowledge_base_id="default") -> int:
        """
        Counts the total files under the namespace.
        """
        return len(
            self._load().get(knowledge_base_id, {})
        )

    def clear(self, knowledge_base_id="default"):
        """
        Deletes the namespace storage block from the JSON database.
        """
        data = self._load()
        data.pop(knowledge_base_id, None)
        self._save(data)
