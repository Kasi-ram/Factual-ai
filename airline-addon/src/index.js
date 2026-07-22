import addOnUISdk from "https://new.express.adobe.com/static/add-on-sdk/sdk.js";

const API_BASE_URL = "http://localhost:8000";

// State variables
let threadId = localStorage.getItem("aeroqa_thread_id") || generateUUID();
let knowledgeBaseId = localStorage.getItem("aeroqa_kb_id") || "default_kb";

localStorage.setItem("aeroqa_thread_id", threadId);
localStorage.setItem("aeroqa_kb_id", knowledgeBaseId);

// Helper function to generate unique session IDs
function generateUUID() {
    return 'aeroqa-' + Math.random().toString(36).substr(2, 9) + '-' + Math.random().toString(36).substr(2, 9);
}

// UI Elements
let clickMeButton;
let queryInput;
let btnSend;
let chatMessages;
let indicator;
let fileInput;
let uploadedDocList;
let statDocs;
let statChunks;
let sidebarErrorContainer;
let btnNewChat;
let btnResetDb;

async function initializeApp() {
    // Retrieve DOM Elements
    queryInput = document.getElementById("query-input");
    btnSend = document.getElementById("btn-send");
    chatMessages = document.getElementById("chat-messages");
    indicator = document.getElementById("indicator");
    fileInput = document.getElementById("file-input");
    uploadedDocList = document.getElementById("uploaded-doc-list");
    statDocs = document.getElementById("stat-docs");
    statChunks = document.getElementById("stat-chunks");
    sidebarErrorContainer = document.getElementById("sidebar-error-container");
    btnNewChat = document.getElementById("btn-new-chat");
    btnResetDb = document.getElementById("btn-reset-db");

    // Check backend connection and initial status
    await checkBackendConnection();

    // Register Event Listeners
    btnSend.addEventListener("click", handleSendQuery);
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") handleSendQuery();
    });

    fileInput.addEventListener("change", handleFileUpload);

    btnNewChat.addEventListener("click", () => {
        threadId = generateUUID();
        localStorage.setItem("aeroqa_thread_id", threadId);
        
        // Reset message list in UI
        chatMessages.innerHTML = `
            <div class="message-row assistant">
                <div class="bubble">
                    Welcome to <strong>AeroQA Airline Grounded Assistant</strong>. 🛫  
                    Please upload your airline manuals, ticket policies, or schedules on the sidebar panel, and ask questions grounded strictly in the reference data!
                </div>
            </div>
        `;
    });

    btnResetDb.addEventListener("click", async () => {
        if (confirm("Are you sure you want to clear all uploaded airline documents?")) {
            try {
                const formData = new FormData();
                formData.append("knowledge_base_id", knowledgeBaseId);

                const response = await fetch(`${API_BASE_URL}/reset`, {
                    method: "POST",
                    body: formData
                });
                
                if (response.ok) {
                    await refreshDocuments();
                    chatMessages.innerHTML += `
                        <div class="message-row assistant">
                            <div class="bubble" style="border-color: rgba(239, 68, 68, 0.2);">
                                ⚠️ <strong>Database Cleared</strong>: All cached airline manuals have been removed from the session.
                            </div>
                        </div>
                    `;
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
            } catch (error) {
                console.error("Error clearing DB:", error);
            }
        }
    });
}

// Check if running inside Adobe Express iframe or standalone browser tab
const isInsideIframe = window.self !== window.top;
if (isInsideIframe) {
    addOnUISdk.ready.then(async () => {
        console.log("addOnUISdk is ready for use.");
        await initializeApp();
    });
} else {
    console.log("Running in standalone local web browser mode.");
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initializeApp);
    } else {
        initializeApp();
    }
}

async function checkBackendConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/documents?knowledge_base_id=${knowledgeBaseId}`);
        if (response.ok) {
            // Enable UI controls
            queryInput.disabled = false;
            btnSend.disabled = false;
            sidebarErrorContainer.innerHTML = "";
            await refreshDocuments();
        } else {
            showConnectionError();
        }
    } catch (error) {
        showConnectionError();
    }
}

function showConnectionError() {
    queryInput.disabled = true;
    btnSend.disabled = true;
    sidebarErrorContainer.innerHTML = `
        <div class="error-card">
            <strong>Backend Connection Failed</strong><br>
            Please start the Python API server by running:<br>
            <code style="background: rgba(0,0,0,0.3); padding: 2px 4px; border-radius: 4px; display: inline-block; margin-top: 5px;">python server.py</code>
        </div>
    `;
}

async function refreshDocuments() {
    try {
        const response = await fetch(`${API_BASE_URL}/documents?knowledge_base_id=${knowledgeBaseId}`);
        if (response.ok) {
            const data = await response.json();
            
            // Update stats
            statDocs.innerText = data.documents.length;
            statChunks.innerText = data.stats.chunks || 0;

            // Render list
            if (data.documents.length === 0) {
                uploadedDocList.innerHTML = `<p style="font-size: 11px; color: var(--text-secondary); text-align: center; margin-top: 10px;">No files uploaded yet.</p>`;
            } else {
                uploadedDocList.innerHTML = data.documents.map(doc => `
                    <div class="doc-card">
                        <div class="doc-name" title="${doc.source}">${doc.source}</div>
                        <div class="doc-meta">${doc.chunks} chunks</div>
                    </div>
                `).join("");
            }
        }
    } catch (error) {
        console.error("Error fetching docs:", error);
    }
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Show loading indicator in sidebar
    sidebarErrorContainer.innerHTML = `
        <div style="font-size: 11px; color: var(--accent-secondary); text-align: center; margin-top: 10px;">
            ⌛ Indexing "${file.name}"...
        </div>
    `;

    try {
        const formData = new FormData();
        formData.append("knowledge_base_id", knowledgeBaseId);
        formData.append("file", file);

        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: "POST",
            body: formData
        });

        if (response.ok) {
            sidebarErrorContainer.innerHTML = "";
            await refreshDocuments();
            
            // Add notification to chat
            chatMessages.innerHTML += `
                <div class="message-row assistant">
                    <div class="bubble" style="border-color: rgba(34, 197, 94, 0.2);">
                        ✅ Successfully uploaded and indexed <strong>${file.name}</strong>. The knowledge base is updated!
                    </div>
                </div>
            `;
            chatMessages.scrollTop = chatMessages.scrollHeight;
        } else {
            const error = await response.json();
            sidebarErrorContainer.innerHTML = `
                <div class="error-card">
                    <strong>Upload failed</strong>: ${error.detail || "Unknown error"}
                </div>
            `;
        }
    } catch (error) {
        console.error("Upload error:", error);
        sidebarErrorContainer.innerHTML = `
            <div class="error-card">
                <strong>Upload error</strong>: Could not connect to API server.
            </div>
        `;
    }
    // Clear input
    fileInput.value = "";
}

async function handleSendQuery() {
    const question = queryInput.value.trim();
    if (!question) return;

    // 1. Render User Question Bubble
    chatMessages.innerHTML += `
        <div class="message-row user">
            <div class="bubble">${escapeHTML(question)}</div>
        </div>
    `;
    queryInput.value = "";
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // 2. Disable controls and show typing indicator
    queryInput.disabled = true;
    btnSend.disabled = true;
    indicator.style.display = "flex";

    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: question,
                thread_id: threadId,
                knowledge_base_id: knowledgeBaseId
            })
        });

        if (response.ok) {
            const data = await response.json();
            
            // Render Assistant Response Bubble with Badges and Evidence
            renderAssistantBubble(data);
        } else {
            chatMessages.innerHTML += `
                <div class="message-row assistant">
                    <div class="bubble" style="border-color: rgba(239, 68, 68, 0.2); color: #fca5a5;">
                        ❌ Error getting answer from backend. Please verify your Python console.
                    </div>
                </div>
            `;
        }
    } catch (error) {
        console.error("Query error:", error);
        chatMessages.innerHTML += `
            <div class="message-row assistant">
                <div class="bubble" style="border-color: rgba(239, 68, 68, 0.2); color: #fca5a5;">
                    ❌ Connection error: Could not reach the API backend.
                </div>
            </div>
        `;
    }

    // 3. Enable controls and hide typing indicator
    queryInput.disabled = false;
    btnSend.disabled = false;
    indicator.style.display = "none";
    queryInput.focus();
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function renderAssistantBubble(data) {
    const confidenceLower = (data.confidence || "low").toLowerCase();
    
    // Create unique ID for the evidence panel accordion
    const evidencePanelId = "ev-panel-" + Math.random().toString(36).substr(2, 9);
    
    let confidenceBadgeHTML = `<div class="confidence-badge ${confidenceLower}">🔴 Low Confidence</div>`;
    if (confidenceLower === "high") {
        confidenceBadgeHTML = `<div class="confidence-badge ${confidenceLower}">🟢 High Confidence (${Math.round(data.confidence_score * 100)}%)</div>`;
    } else if (confidenceLower === "medium") {
        confidenceBadgeHTML = `<div class="confidence-badge ${confidenceLower}">🟡 Medium Confidence (${Math.round(data.confidence_score * 100)}%)</div>`;
    }

    // Build evidence list if available
    let evidenceHTML = "";
    if (data.evidence && data.evidence.length > 0) {
        const evidenceItemsHTML = data.evidence.map((ev, index) => `
            <div class="evidence-item">
                <div class="evidence-meta">
                    <span>📄 Chunk #${index + 1} (${ev.source})</span>
                    <span>Page ${ev.page || 1}</span>
                </div>
                <div class="evidence-text">${escapeHTML(ev.text)}</div>
            </div>
        `).join("");

        evidenceHTML = `
            <div class="evidence-expander">
                <div class="evidence-header" onclick="document.getElementById('${evidencePanelId}').classList.toggle('expanded')">
                    <span>🔍 Grounding Evidence (${data.evidence.length} sources)</span>
                    <span>▼</span>
                </div>
                <div class="evidence-body" id="${evidencePanelId}">
                    ${evidenceItemsHTML}
                </div>
            </div>
        `;
    }

    chatMessages.innerHTML += `
        <div class="message-row assistant">
            <div class="bubble">
                ${confidenceBadgeHTML}
                <div>${formatMarkdown(data.answer)}</div>
                ${evidenceHTML}
            </div>
        </div>
    `;
}

function escapeHTML(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatMarkdown(text) {
    // Simple bold markdown conversion (**bold**)
    let formatted = escapeHTML(text)
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>")
        .replace(/\n/g, "<br>");
    return formatted;
}
