import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from services.langgraph_agent import LangGraphAgent
from services.document_ingestion_service import DocumentIngestionService

app = FastAPI(title="Factual-ai Backend API")

# Configure CORS so local web frontend / CCWeb add-on can communicate with it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
agent = LangGraphAgent()
ingestion_service = DocumentIngestionService()

# In-memory helper to mock uploaded file structure for ingestion service
class MockUploadedFile:
    def __init__(self, filename: str, content: bytes):
        self.name = filename
        self.content = content
    def getvalue(self) -> bytes:
        return self.content

class QueryRequest(BaseModel):
    question: str
    thread_id: str
    knowledge_base_id: str

@app.post("/query")
async def run_query(payload: QueryRequest):
    try:
        result = agent.ask(
            question=payload.question,
            thread_id=payload.thread_id,
            knowledge_base_id=payload.knowledge_base_id
        )
        return {
            "status": "success",
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result.get("confidence", "Low"),
            "confidence_score": result.get("confidence_score", 0.0),
            "evidence": result.get("evidence", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents(knowledge_base_id: str):
    try:
        docs = ingestion_service.documents(knowledge_base_id)
        stats = ingestion_service.stats(knowledge_base_id)
        return {
            "status": "success",
            "documents": docs,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(
    knowledge_base_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # Read file contents
        content = await file.read()
        mock_file = MockUploadedFile(file.filename, content)
        
        # Ingest document
        result = ingestion_service.ingest_document(
            uploaded_file=mock_file,
            knowledge_base_id=knowledge_base_id
        )
        
        if result["status"] == "success":
            return {
                "status": "success",
                "message": f"Indexed {result['source']}",
                "pages": result["pages"],
                "chunks": result["chunks"]
            }
        elif result["status"] == "duplicate":
            return {
                "status": "duplicate",
                "message": "Document already exists in this session."
            }
        elif result["status"] in {"empty", "rejected"}:
            raise HTTPException(status_code=400, detail="Could not read content or document is empty.")
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "Ingestion failed."))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_kb(knowledge_base_id: str = Form(...)):
    try:
        ingestion_service.reset(knowledge_base_id)
        return {
            "status": "success",
            "message": "Knowledge base cleared."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
