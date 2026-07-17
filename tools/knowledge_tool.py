"""
Knowledge Tool for Factual-ai
=============================
This tool wraps the RAGService to search uploaded documents and answer questions
grounded in the retrieved evidence. It is registered as a node in the LangGraph agent.
"""

from services.rag_service import RAGService

class KnowledgeTool:
    """
    RAG Search tool execution wrapper.
    """

    def __init__(self):
        self.rag = RAGService()

    def execute(self, question: str, knowledge_base_id: str = "default") -> dict:
        """
        Executes a similarity search and answers based on matched chunks.
        
        Args:
            question: The user query.
            knowledge_base_id: Session-specific database namespace.
            
        Returns:
            A dict containing 'answer', 'sources', and 'found'.
        """
        return self.rag.ask(
            question,
            knowledge_base_id
        )
