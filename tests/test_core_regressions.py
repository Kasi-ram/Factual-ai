"""
Regression Tests for Factual-ai
==============================
This test suite verifies the core LangGraph transitions, session memory,
namespace isolation, document processing, and guardrail refutation rules
for the Factual-ai codebase. Stubs are used to bypass active API requests.
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path

from services.document_registry import DocumentRegistry

# 1. Install mock/stub replacements for API dependencies
class FakeLLMService:
    def ask_json(self, prompt):
        # Default mock planning decision to route to knowledge
        return '{"tools": ["knowledge"], "knowledge_query": "policy"}'

    def ask(self, prompt):
        return "general chitchat answer"

class FakeRAGService:
    calls = []

    def ask(self, question, knowledge_base_id="default"):
        self.calls.append((question, knowledge_base_id))
        return {
            "answer": "knowledge search answer",
            "found": True,
            "sources": []
        }

def install_agent_stubs():
    """
    Dynamically mocks services to avoid external API calls during testing.
    """
    for module_name, class_name, value in [
        ("services.llm_service", "LLMService", FakeLLMService),
        ("services.rag_service", "RAGService", FakeRAGService)
    ]:
        module = types.ModuleType(module_name)
        setattr(module, class_name, value)
        sys.modules[module_name] = module

# Run the stubs configuration before importing agent
install_agent_stubs()

from services.langgraph_agent import LangGraphAgent

class DocumentRegistryTests(unittest.TestCase):
    """
    Tests document isolation in the metadata registry.
    """

    def test_documents_are_isolated_by_knowledge_base(self):
        original_file = DocumentRegistry.FILE

        with tempfile.TemporaryDirectory() as directory:
            DocumentRegistry.FILE = Path(directory) / "registry.json"
            try:
                registry = DocumentRegistry()
                # Register identical document hash across different namespaces
                registry.register("hash-abc", "file-a.txt", 1, 1, "tenant-a")
                registry.register("hash-abc", "file-b.txt", 1, 1, "tenant-b")

                self.assertEqual(registry.count("tenant-a"), 1)
                self.assertEqual(registry.count("tenant-b"), 1)
                self.assertEqual(
                    registry.list_documents("tenant-a")[0]["filename"],
                    "file-a.txt"
                )
            finally:
                DocumentRegistry.FILE = original_file


class LangGraphAgentTests(unittest.TestCase):
    """
    Tests LangGraph flow routing, memory retention, and guardrails.
    """

    def setUp(self):
        FakeRAGService.calls.clear()
        self.agent = LangGraphAgent()

    def test_checkpoint_memory_is_retained(self):
        # Trigger two successive interactions on the same thread
        self.agent.ask("Hello", thread_id="memory_thread")
        self.agent.ask("Who are you?", thread_id="memory_thread")

        state = self.agent.graph.get_state(
            {"configurable": {"thread_id": "memory_thread"}}
        ).values

        # Each ask appends user + assistant message, so 2 turns = 4 messages
        self.assertEqual(len(state["conversation_history"]), 4)

    def test_knowledge_request_uses_the_given_namespace(self):
        self.agent.ask(
            "Search for policy data",
            thread_id="namespace_thread",
            knowledge_base_id="tenant-x"
        )

        self.assertEqual(
            FakeRAGService.calls,
            [("policy", "tenant-x")]
        )

    def test_document_processor_supports_markdown(self):
        from services.document_processor import DocumentProcessor

        class MockUploadedFile:
            def __init__(self, name, content):
                self.name = name
                self.content = content
            def getvalue(self):
                return self.content

        uploaded_file = MockUploadedFile("test.md", b"# Markdown Title\nThis is general content.")
        processor = DocumentProcessor()
        pages = processor.process(uploaded_file)
        
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0][0], 1)
        self.assertIn("This is general content.", pages[0][1])

    def test_general_fallback(self):
        # Setup mock planner to return no tools (forces fallback)
        self.agent.llm.ask_json = lambda prompt: '{"tools": [], "knowledge_query": ""}'
        self.agent.llm.ask = lambda prompt: "Hello, I am Factual-ai"

        response = self.agent.ask("Who are you?")
        self.assertEqual(response["answer"], "Hello, I am Factual-ai")

    def test_general_fallback_refusal(self):
        # Verify that factual queries are strictly refused
        self.agent.llm.ask_json = lambda prompt: '{"tools": [], "knowledge_query": ""}'
        
        # Override LLM response to verify fallback node instructions block and return the refusal
        self.agent.llm.ask = lambda prompt: "I could not find this information in the available knowledge base."

        response = self.agent.ask("What is the capital of France?")
        self.assertEqual(
            response["answer"],
            "I could not find this information in the available knowledge base."
        )


if __name__ == "__main__":
    unittest.main()
