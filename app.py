"""
Factual-ai Streamlit Application
================================
This is the main entry point for the Factual-ai interface.
It presents a clean, professional dark-themed chatbot UI powered by Streamlit,
allowing users to upload text, markdown, HTML, CSV, JSON, and PDF documents,
manage active knowledge bases, and ask questions grounded in the uploaded files.
"""

import dotenv
dotenv.load_dotenv()

import uuid
import streamlit as st
from services.langgraph_agent import LangGraphAgent
from services.document_ingestion_service import DocumentIngestionService

# 1. Page Configuration
st.set_page_config(
    page_title="Factual-ai",
    page_icon="📚",
    layout="wide"
)

# 2. Minimal Premium CSS for Dark Theme Enhancements
st.markdown("""
<style>
    /* Styling for the main header */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #f8fafc;
    }
    
    /* Premium border-radius and outline for Streamlit chat inputs */
    .stChatInputContainer {
        border-radius: 12px !important;
        border: 1px solid #334155 !important;
        background-color: #1e293b !important;
    }
    
    /* File uploader container custom styling */
    .stFileUploader {
        border: 1px dashed #475569 !important;
        background-color: #0f172a !important;
        border-radius: 12px;
        padding: 10px;
    }

    /* Style the document list blocks in the side panel */
    .doc-card {
        padding: 12px;
        border-radius: 8px;
        background-color: #1e293b;
        border: 1px solid #334155;
        margin-bottom: 10px;
    }

    /* Style the error cards */
    .error-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #2a1215;
        border: 1px solid #7f1d1d;
        color: #fca5a5;
        margin-bottom: 15px;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Agent and Ingestion Services
@st.cache_resource
def load_agent():
    """Instantiates the Factual-ai LangGraphAgent once and caches it."""
    return LangGraphAgent()

@st.cache_resource
def load_ingestion_service():
    """Instantiates the DocumentIngestionService once and caches it."""
    return DocumentIngestionService()

agent = load_agent()
ingestion_service = load_ingestion_service()

# 4. Initialize Session States
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "knowledge_base_id" not in st.session_state:
    st.session_state.knowledge_base_id = str(uuid.uuid4())

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Main Layout Title
st.title("📚 Factual-ai")
st.caption("Professional General-Genre Knowledge Assistant")

# 6. Render Chat Messages History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("is_html"):
            st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"])
        
        # Display citation sources if present
        if message.get("sources"):
            st.markdown("**Sources:**")
            for source in message["sources"]:
                st.caption(f'📄 {source["source"]} - Page {source["page"]}')

# 7. Chat Input Query
question = st.chat_input("Ask a question grounded in the uploaded documents...")

if question:
    # Append user question to local state
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Display modern status spinner during processing
        with st.status("Factual-ai is working...", expanded=True) as status:
            st.write("Understanding your request...")
            
            response = {"answer": "", "sources": []}
            
            # Map LangGraph transitions to friendly statuses
            node_messages = {
                "planner": "Planning request...",
                "knowledge": "Searching knowledge base documents...",
                "general": "Answering casual chitchat...",
                "final_answer": "Preparing final answer..."
            }

            try:
                # Run stream execution
                for event in agent.stream(
                    question,
                    thread_id=st.session_state.thread_id,
                    knowledge_base_id=st.session_state.knowledge_base_id
                ):
                    if event["type"] == "node":
                        node_name = event["node"]
                        msg = node_messages.get(node_name)
                        if msg:
                            st.write(msg)
                    elif event["type"] == "result":
                        response = {
                            "answer": event["answer"],
                            "sources": event["sources"]
                        }

                status.update(
                    label="Completed",
                    state="complete",
                    expanded=False
                )
            except Exception as e:
                status.update(
                    label="Execution Failed",
                    state="error",
                    expanded=False
                )
                
                error_title = type(e).__name__
                error_msg = str(e)
                
                # Check for common error types to display friendly hints
                hint = "Please check your network connection and retry. If the problem persists, contact your systems administrator."
                if "api key" in error_msg.lower() or "api_key" in error_msg.lower():
                    hint = "A missing or invalid API key was detected. Please verify the `LLM_API_KEY` and `GEMINI_API_KEY` configurations in your local `.env` file."
                elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                    hint = "The service rate limits or API quotas have been exceeded. Please wait a short while before trying again."
                
                # Generate styled HTML error card
                error_html = f"""
                <div class="error-card">
                    <h4 style="margin: 0 0 10px 0; color: #fecaca; font-family: 'Inter', sans-serif;">⚠️ Factual-ai Execution Error</h4>
                    <p style="margin: 0 0 8px 0; font-size: 14px;">An unexpected error occurred while compiling your request.</p>
                    <ul style="margin: 0 0 12px 0; padding-left: 20px; font-size: 13px;">
                        <li><b>Error Category:</b> <code>{error_title}</code></li>
                        <li><b>System Message:</b> <i>{error_msg}</i></li>
                    </ul>
                    <hr style="border: 0; border-top: 1px solid #7f1d1d; margin: 10px 0;" />
                    <span style="font-size: 13px; color: #fca5a5;"><b>Troubleshooting Advice:</b> {hint}</span>
                </div>
                """
                response = {
                    "answer": error_html,
                    "sources": [],
                    "is_html": True
                }

        # Output final compiled answer
        if response.get("is_html"):
            st.markdown(response["answer"], unsafe_allow_html=True)
        else:
            st.markdown(response["answer"])

        # Display citations
        if response["sources"]:
            st.markdown("**Sources:**")
            for source in response["sources"]:
                st.caption(f'📄 {source["source"]} - Page {source["page"]}')

    # Append assistant response to local state
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
            "is_html": response.get("is_html", False)
        }
    )

# 8. Sidebar Configuration Panel
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write("Factual-ai is running with the TCS GenAILab DeepSeek-V3 model.")

    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()

    st.subheader("📁 Upload Documents")
    
    # File uploader supporting multiple general formats
    uploaded_file = st.file_uploader(
        "Upload document file",
        type=["pdf", "txt", "md", "csv", "json", "html"],
        key=f"doc_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        if st.button("➕ Add to Knowledge Base"):
            with st.spinner("Processing and indexing document..."):
                result = ingestion_service.ingest_document(
                    uploaded_file,
                    st.session_state.knowledge_base_id
                )

            if result["status"] == "success":
                st.success(f'Indexed {result["source"]}')
                st.write(f'Pages: {result["pages"]} | Chunks: {result["chunks"]}')
                st.session_state.uploader_key += 1
                st.rerun()
            elif result["status"] == "duplicate":
                st.warning("Document already exists in this session.")
            elif result["status"] in {"empty", "rejected"}:
                st.warning("Could not read content or document is empty.")
            else:
                st.error(result.get("message", "Ingestion failed."))

    st.divider()

    # Reset active collection button
    if st.button("⚠️ Reset Knowledge Base"):
        ingestion_service.reset(st.session_state.knowledge_base_id)
        st.session_state.uploader_key += 1
        st.session_state.messages = []
        st.success("Knowledge base cleared.")
        st.rerun()

    st.divider()

    # 9. Sidebar Stats and File Listings Panel
    st.subheader("📊 Session Documents")
    
    docs = ingestion_service.documents(st.session_state.knowledge_base_id)
    stats = ingestion_service.stats(st.session_state.knowledge_base_id)

    if docs:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Docs", value=stats["documents"])
        with col2:
            st.metric(label="Total Chunks", value=stats["chunks"])

        st.write("**Registered Files:**")
        for doc in docs:
            uploaded_at = doc.get("uploaded_at", "Unknown")
            st.markdown(
                f"""
                <div class="doc-card">
                    <b>📄 {doc['filename']}</b><br/>
                    <small>Chunks: {doc['chunks']} | Pages: {doc['pages']}</small><br/>
                    <small style="color: #64748b;">Uploaded: {uploaded_at}</small>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No documents uploaded yet.")
