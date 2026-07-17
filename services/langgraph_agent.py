"""
LangGraph Agent for Factual-ai
==============================
This module defines the agent architecture using a state-graph compiled with langgraph.
The agent comprises a planner node, a knowledge lookup node (RAG flow), a general conversational
fallback node (handling chitchat and strict refusals), and a final answer compilation node.
Memory checkpointer is enabled for thread/session context.
"""

import json
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from services.llm_service import LLMService
from tools.knowledge_tool import KnowledgeTool

class AgentState(TypedDict):
    """
    Dictionary schema definition for the Factual-ai graph state.
    """
    question: str                  # Current user input question
    conversation_history: list     # List of past conversation turns
    knowledge_base_id: str         # Session-specific directory namespace
    selected_tools: list           # Tools selected by the planner
    knowledge_query: str           # The rewritten search query for RAG
    knowledge_answer: str          # Raw answer generated from evidence
    knowledge_found: bool          # Flag indicating if evidence matched
    answer: str                    # Current active answer text
    sources: list                  # Matched document sources and page indices

class LangGraphAgent:
    """
    Agent implementation establishing a RAG + Chitchat state graph pipeline.
    """

    def __init__(self):
        self.llm = LLMService()
        self.knowledge_tool = KnowledgeTool()

        # Initialize the state graph
        graph = StateGraph(AgentState)

        # Register execution nodes
        graph.add_node("planner", self.planner_node)
        graph.add_node("knowledge", self.knowledge_node)
        graph.add_node("general", self.general_node)
        graph.add_node("final_answer", self.final_answer_node)

        # Start edge transitions
        graph.add_edge(START, "planner")

        # Planner conditionally routes to knowledge retrieval or general node
        graph.add_conditional_edges(
            "planner",
            self.route_after_planner,
            {
                "knowledge": "knowledge",
                "general": "general"
            }
        )

        # Knowledge search conditionally routes to final compiler or general chitchat fallback
        graph.add_conditional_edges(
            "knowledge",
            self.route_after_knowledge,
            {
                "general": "general",
                "final_answer": "final_answer"
            }
        )

        # Connect fallback node to final compiler
        graph.add_edge("general", "final_answer")
        
        # End step
        graph.add_edge("final_answer", END)

        # Establish memory checkpointer
        self.memory = InMemorySaver()
        self.graph = graph.compile(
            checkpointer=self.memory
        )

    def planner_node(self, state) -> dict:
        """
        Analyzes query context to decide if RAG tool or casual chitchat node is required.
        """
        question = state["question"]
        conversation_history = state.get("conversation_history") or []

        # Build short conversation log
        history_text = "\n".join(
            [
                f'{message["role"]}: {message["content"]}'
                for message in conversation_history[-6:]
            ]
        )

        # Formulate instructions for DeepSeek-V3
        prompt = f"""
You are the Planner for Factual-ai.
Your job is to determine whether the user's question requires searching uploaded documents.

Do NOT answer the question. Only output structured JSON.

Rules:
1. If the user asks about factual information, document contents, specific data, or policies, set tools to ["knowledge"] and rewrite the question into a complete, standalone knowledge_query.
2. If the user question depends on previous conversation history, rewrite it to be a standalone question.
3. If the user is just saying hello, greeting you, or asking about your name/identity/capabilities, set tools to [] and knowledge_query to "".

Return ONLY valid JSON using this structure:
{{
    "tools": [],
    "knowledge_query": ""
}}

CONVERSATION HISTORY:
{history_text}

CURRENT QUESTION:
{question}
"""
        # Call model using JSON mode
        response = self.llm.ask_json(prompt)
        decision = json.loads(response)

        selected_tools = [
            tool
            for tool in decision.get("tools", [])
            if tool in {"knowledge"}
        ]

        knowledge_query = decision.get("knowledge_query", "").strip()

        # Post-processing: force knowledge if query exists
        if knowledge_query and "knowledge" not in selected_tools:
            selected_tools.append("knowledge")

        return {
            "selected_tools": selected_tools,
            "knowledge_query": knowledge_query,
            "knowledge_answer": "",
            "knowledge_found": False,
            "answer": "",
            "sources": []
        }

    def route_after_planner(self, state) -> str:
        """
        Routes execution path to 'knowledge' or 'general'.
        """
        if "knowledge" in state["selected_tools"]:
            return "knowledge"
        return "general"

    def knowledge_node(self, state) -> dict:
        """
        Invokes similarity retrieval and compiles answers from document evidence.
        """
        result = self.knowledge_tool.execute(
            state["knowledge_query"],
            state.get("knowledge_base_id", "default")
        )

        return {
            "knowledge_answer": result["answer"],
            "knowledge_found": result["found"],
            "sources": result["sources"]
        }

    def route_after_knowledge(self, state) -> str:
        """
        Directs state transitions after document search step.
        """
        if state.get("knowledge_found"):
            return "final_answer"
        return "general"

    def general_node(self, state) -> dict:
        """
        Serves casual greetings, chitchat, and enforces strict general factual refusals.
        """
        question = state["question"]
        conversation_history = state.get("conversation_history") or []

        history_text = "\n".join(
            [
                f'{message["role"]}: {message["content"]}'
                for message in conversation_history[-6:]
            ]
        )

        prompt = f"""
You are Factual-ai, a professional knowledge assistant.
The uploaded knowledge base was searched first and did not contain an answer.

You must ONLY answer simple casual greetings (e.g., "hello", "hi", "how are you") or questions about your own identity/capabilities (e.g., "who are you?", "what is your name?", "what can you do?").

For ANY other question—including general knowledge, sports, history, science, geography, news, math, or any factual informational request—you MUST respond exactly with:

"I could not find this information in the available knowledge base."

Do NOT attempt to answer or provide information on these general topics.

CONVERSATION HISTORY:
{history_text}

USER QUESTION:
{question}
"""
        answer = self.llm.ask(prompt)
        return {
            "answer": answer
        }

    def final_answer_node(self, state) -> dict:
        """
        Compiles the final answer content and updates the conversational memory log.
        """
        selected_tools = state["selected_tools"]
        knowledge_answer = state.get("knowledge_answer", "")
        current_answer = state.get("answer", "")

        # Select compiled RAG answer if found, otherwise fall back to casual general node answer
        if "knowledge" in selected_tools and state.get("knowledge_found"):
            answer = knowledge_answer
        else:
            answer = current_answer

        history = state.get("conversation_history") or []
        history = history + [
            {
                "role": "user",
                "content": state["question"]
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]

        return {
            "answer": answer,
            "conversation_history": history
        }

    def ask(self, question, thread_id="default", knowledge_base_id="default") -> dict:
        """
        Synchronously triggers a query through the LangGraph.
        """
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        result = self.graph.invoke(
            {
                "question": question,
                "knowledge_base_id": knowledge_base_id
            },
            config=config
        )
        return {
            "answer": result["answer"],
            "sources": result["sources"]
        }

    def stream(self, question, thread_id="default", knowledge_base_id="default"):
        """
        Streams node state transitions and prints progress.
        """
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        initial_state = {
            "question": question,
            "knowledge_base_id": knowledge_base_id
        }

        result = initial_state.copy()
        received_update = False

        for update in self.graph.stream(
            initial_state,
            config=config,
            stream_mode="updates"
        ):
            node = next(iter(update.keys()))
            state = update[node]
            result.update(state)
            received_update = True
            yield {
                "type": "node",
                "node": node,
                "state": state
            }

        if not received_update:
            result = self.graph.invoke(
                initial_state,
                config=config
            )

        yield {
            "type": "result",
            "answer": result.get("answer", ""),
            "sources": result.get("sources", [])
        }
