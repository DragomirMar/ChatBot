import streamlit as st
from extract_text import *
from llm import OllamaModel
from vector_database import VectorDatabase
import logging
from logger_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

db = VectorDatabase()

def ask(prompt):
    llm = OllamaModel()
    return llm.inference(prompt)    

def clear_database():
    st.write("Database cleared.")
    db.clear_database()

def show_sources():
    sources = db.get_sources()
    st.write("Sources in the database:")
    for source in sources:
        st.write(f"- {source}")

def process_url(url):
    st.write(f"Processing URL: {url}")
    try:
        chunks = extract_chunks_from_url(url)
        
        if not chunks:
            st.error("Failed to extract content from URL. Please check if the URL is valid and accessible.")
            return
        
        # Add chunks to database (no need to pass source again as it's already in metadata)
        db.add_document_chunks(chunks)
        
        st.success(f"Successfully added {len(chunks)} chunks from URL to database")
        logger.info(f"URL content added successfully: {url}")
        
    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")
        logger.error(f"Error processing URL {url}: {str(e)}")

def process_pdf(uploaded_files):
    st.write("Saving files to DB...")
    
    for file in uploaded_files:
        chunks = extract_chunks_from_pdf(file)
        db.add_document_chunks(chunks, file.name)
    logger.info("Documents added successfully")

# Title
st.set_page_config(
    page_title="RAG ChatBot",
    layout="wide")

st.title('Upload Documents to RAG ChatBot')

col1, col2 = st.columns(2)

# PDF Upload
with col1:
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        label_visibility="collapsed", 
        accept_multiple_files=True
    )    
    
    if st.button("Confirm Files"):
        if uploaded_files:
            process_pdf(uploaded_files)
        else:
            st.warning("Please select at least one PDF file.")

# URL Upload
with col2:
    st.subheader("Add URLs")
    input_url = st.text_input(
        "Enter URL", 
        placeholder="https://example.com/article"
    )
    
    if st.button("Confirm URL"):
        if input_url.strip():
            process_url(input_url.strip())
        else:
            st.warning("Please enter a valid URL.")

st.markdown("---")
# Buttons for database operations
col1, col2 = st.columns([1,1]) 
with col1:
    if st.button("Show Sources"):
        show_sources()

with col2:
    if st.button("Clear Database"):
        clear_database()
            
        
# Chat Bot
st.title('Chat Bot 🤖')

if st.button("🧹", help="Clear chat"):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input field (pinned to bottom automatically)
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user's message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Response
    with st.spinner("KG Bot is thinking..."):
        # Search vector DB
        relevant_chunks = db.similarity_search(user_input, k=5)
        print(f"Relevant chunks: {relevant_chunks}")
        logger.info(f"Found {len(relevant_chunks)} relevant chunks")
        
        # Extract page_content from each chunk and join them
        if relevant_chunks:
            context = "\n".join([chunk.page_content for chunk in relevant_chunks])
        else:
            context = "No relevant context found."

        # Add context to user query
        enriched_prompt = f"""
            You are a helpful assistant. Use the following context to help answer the user's question.
            Only use the context if it is relevant.
            Summarize your answer in no more than 3 sentences.

            Context:
            {context}

            User Question:
            {user_input}

            Answer:"""

        response_text = ask(enriched_prompt)

    response = f"{response_text}"
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})