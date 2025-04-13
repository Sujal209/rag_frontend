# -*- coding: utf-8 -*-
"""rag_frontend.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mcnAZAOvd6THxzsFs2nVfvXUWRE04KD_
"""

# Install required packages
pip install langchain langchain-google-genai langchain_community pypdf chromadb sentence-transformers -q
pip install google-generativeai pdfplumber -q

pip install streamlit

pip install pyngrok

import os
import pdfplumber
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from google.colab import userdata
os.environ["GOOGLE_API_KEY"] = userdata.get("gemini_api_key")

import streamlit as st
import os
import tempfile
from datetime import datetime
import io
from session_4_rag_backend import (
    setup_api_key,
    upload_pdf,
    parse_pdf,
    create_document_chunks,
    init_embedding_model,
    embed_documents,
    store_embeddings,
    get_context_from_chunks,
    query_with_full_context
)

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot with Gemini",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs { padding-bottom: 10px; }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 12px;
        transition: background-color 0.3s;
    }

    .sidebar .sidebar-content { background-color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .highlight { background-color: #e6f3ff; animation: fadeOut 2s forwards; }
    @keyframes fadeOut { 100% { background-color: transparent; } }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "preview_content" not in st.session_state:
    st.session_state.preview_content = {}

def main():
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Settings")

        # API Key input
        with st.container():
            st.subheader("API Key")
            api_key = st.text_input("Gemini API Key:", type="password", help="Enter your Gemini API key to enable document processing.")
            if api_key and st.button("Set API Key", key="set_api_key"):
                setup_api_key(api_key)
                st.success("API Key set!", icon="✅")

        st.divider()

        # Advanced options
        with st.expander("🔧 Advanced Settings"):
            st.markdown('<div class="tooltip">Chunks to retrieve (k)<span class="tooltiptext">Number of document chunks to retrieve for context</span></div>', unsafe_allow_html=True)
            st.slider("", min_value=1, max_value=10, value=3, key="k_value")
            st.markdown('<div class="tooltip">Temperature<span class="tooltiptext">Controls randomness of responses (0 = deterministic, 1 = creative)</span></div>', unsafe_allow_html=True)
            st.slider("", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key="temperature")
            if st.button("Reset Defaults", key="reset_settings"):
                st.session_state.k_value = 3
                st.session_state.temperature = 0.2
                st.rerun()

    # Main content with tabs
    st.title("📚 RAG Chatbot")
    tab1, tab2 = st.tabs(["💬 Chat", "📑 Documents"])

    with tab1:
        if st.session_state.vectorstore is None:
            st.info("Upload documents in the Documents tab to start chatting.", icon="ℹ️")
            display_prompt_suggestions()
        else:
            # Split chat and context preview
            col1, col2 = st.columns([3, 1])
            with col1:
                display_chat()
                user_query = st.chat_input("Ask about your documents...")
                if user_query:
                    handle_user_query(user_query)
            with col2:
                st.subheader("Context Preview")
                if st.session_state.conversation and "context" in st.session_state.conversation[-1]:
                    st.text_area("Last Context", st.session_state.conversation[-1]["context"], height=200, disabled=True)
                else:
                    st.write("No context available.")

    with tab2:
        # Document management
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if uploaded_files and st.button("Process Documents", key="process_docs"):
            process_documents(uploaded_files)

        # Display processed files with preview
        if st.session_state.processed_files:
            st.subheader("📑 Processed Documents")
            for file in st.session_state.processed_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {file}")
                with col2:
                    if st.button("Preview", key=f"preview_{file}"):
                        display_document_preview(file)

def process_documents(uploaded_files):
    """Process uploaded PDF documents and create the vector store"""
    try:
        with st.sidebar.status("Processing documents...", expanded=True) as status:
            progress_bar = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                status.update(label=f"Processing {uploaded_file.name}...")
                progress_bar.progress(int((i / len(uploaded_files)) * 100))

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    pdf_path = tmp_file.name

                # Process the PDF
                pdf_file = upload_pdf(pdf_path)
                if not pdf_file:
                    st.error(f"Failed to process {uploaded_file.name}", icon="❌")
                    continue

                # Parse PDF to extract text
                text = parse_pdf(pdf_file)
                if not text:
                    st.error(f"Failed to extract text from {uploaded_file.name}", icon="❌")
                    continue

                # Store preview content
                st.session_state.preview_content[uploaded_file.name] = text[:1000] + "..." if len(text) > 1000 else text

                # Create document chunks
                chunks = create_document_chunks(text)
                if not chunks:
                    st.error(f"Failed to create chunks from {uploaded_file.name}", icon="❌")
                    continue

                # Add metadata to chunks
                chunks_with_metadata = [{"content": chunk, "source": uploaded_file.name} for chunk in chunks]
                all_chunks = chunks_with_metadata
                processed_file_names = [uploaded_file.name]

                # Clean up temporary file
                os.unlink(pdf_path)

            # Initialize embedding model
            if st.session_state.embedding_model is None:
                status.update(label="Initializing embedding model...")
                st.session_state.embedding_model = init_embedding_model()
                if st.session_state.embedding_model is None:
                    st.error("Failed to initialize embedding model. Check your API key.", icon="❌")
                    return

            # Store embeddings
            status.update(label="Creating vector database...")
            if all_chunks:
                texts = [chunk["content"] for chunk in all_chunks]
                metadatas = [{"source": chunk["source"]} for chunk in all_chunks]



                vectorstore = store_embeddings(
                    st.session_state.embedding_model,
                    texts,
                    persist_directory="./streamlit_chroma_db"
                )

                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.processed_files.extend(processed_file_names)
                    st.success(f"Processed {len(processed_file_names)} document(s)", icon="✅")
                else:
                    st.error("Failed to create vector database", icon="❌")
            else:
                st.error("No valid chunks extracted", icon="❌")

            progress_bar.progress(100)

    except Exception as e:
        st.error(f"Error processing documents: {str(e)}. Please check your files and try again.", icon="❌")

def handle_user_query(query):
    """Process a user query and display the response"""
    if st.session_state.vectorstore is None:
        st.error("Please process documents first", icon="⚠️")
        return

    # Add user message
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.conversation.append({"role": "user", "content": query, "timestamp": timestamp})

    # Process query with spinner
    with st.spinner("Generating response..."):
        try:
            k = st.session_state.k_value
            temperature = st.session_state.temperature
            response, context, chunks = query_with_full_context(
                query,
                st.session_state.vectorstore,
                k=k,
                temperature=temperature
            )
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.conversation.append({
                "role": "assistant",
                "content": response,
                "context": context,
                "timestamp": timestamp
            })
            display_chat()
        except Exception as e:
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.conversation.append({
                "role": "assistant",
                "content": f"Error: {str(e)}. Try rephrasing your question or check your settings.",
                "timestamp": timestamp
            })
            display_chat()

def display_chat():
    """Display the chat conversation"""
    chat_container = st.container()
    with chat_container:
        for idx, message in enumerate(st.session_state.conversation):
            css_class = "highlight" if idx == len(st.session_state.conversation) - 1 and message["role"] == "assistant" else ""
            if message["role"] == "user":
                with st.chat_message("user", avatar="🙋‍♂️"):
                    st.markdown(f"<div class='{css_class}'>**{message['timestamp']}** - {message['content']}</div>", unsafe_allow_html=True)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(f"<div class='{css_class}'>**{message['timestamp']}** - {message['content']}</div>", unsafe_allow_html=True)
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if "context" in message and message["context"]:
                            with st.expander("🔍 View Context"):
                                st.text(message["context"])
                    with col2:
                        if st.button("Copy", key=f"copy_{idx}"):
                            st.write("Copied to clipboard!")
                            # Note: Streamlit doesn't natively support clipboard; consider JavaScript for production
        # Auto-scroll to bottom
        st.markdown('<script>window.scrollTo(0, document.body.scrollHeight);</script>', unsafe_allow_html=True)


def display_document_preview(file_name):
    """Display a preview of the document content"""
    if file_name in st.session_state.preview_content:
        with st.expander(f"Preview: {file_name}"):
            st.text(st.session_state.preview_content[file_name])
    else:
        st.warning("No preview available for this document.", icon="⚠️")

def display_prompt_suggestions():
    """Display example questions to guide users"""
    st.subheader("Try These Questions")
    questions = [
        "What is the main topic of the documents?",
        "Can you summarize the key points?",
        "Are there any specific terms explained?"
    ]
    for q in questions:
        if st.button(q, key=f"suggest_{q}"):
            handle_user_query(q)

def reset_conversation():
    """Reset the conversation history"""
    st.session_state.conversation = []

if __name__ == "__main__":
    main()

streamlit run rag_streamlit.py --server.port=8989 &>./logs.txt &


