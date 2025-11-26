import streamlit as st
from extract_text import *
from llm import OllamaModel
from vector_database import VectorDatabase
from knowledge_graph_retriever import KnowledgeGraphRetriever
# from deleete_later import KnowledgeGraphRetriever
import logging
from logger_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

db = VectorDatabase()
kg_retriever = KnowledgeGraphRetriever()

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
    
    # Also show KG stats
    kg_stats = kg_retriever.get_kg_stats()
    st.write("\nKnowledge Graph Statistics:")
    st.write(f"- Entities: {kg_stats.get('total_entities', 0)}")
    st.write(f"- Relationships: {kg_stats.get('total_relationships', 0)}")

def process_url(url):
    st.write(f"Processing URL: {url}")
    try:
        chunks = extract_chunks_from_url(url)
        
        if not chunks:
            st.error("Failed to extract content from URL. Please check if the URL is valid and accessible.")
            return
        
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
    page_title="RAG ChatBot with Knowledge Graph",
    layout="wide")

st.title('Upload Documents to RAG ChatBot with Knowledge Graph 🧠')

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

# Knowledge Graph Settings (collapsible)
with st.expander("⚙️ Knowledge Graph Settings"):
    use_kg = st.checkbox("Enable Knowledge Graph Enhancement", value=True, 
                         help="Use knowledge graph to enhance answers with entity information")
    max_hops = st.slider("Relationship Hops", min_value=1, max_value=2, value=1,
                         help="Number of relationship connections to traverse (1=direct, 2=friends-of-friends)")
    max_entities = st.slider("Max Entities", min_value=1, max_value=10, value=5,
                             help="Maximum number of entities to retrieve from query")
            
        
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
        # 1. Search vector DB for document chunks
        relevant_chunks = db.similarity_search(user_input, k=5)
        logger.info(f"Found {len(relevant_chunks)} relevant chunks from vector DB")
        
        # Extract page_content from chunks
        if relevant_chunks:
            doc_context = "\n".join([chunk.page_content for chunk in relevant_chunks])
        else:
            doc_context = ""
        
        # 2. Retrieve knowledge graph context (if enabled)
        kg_context = ""
        if use_kg:
            kg_context = kg_retriever.retrieve_kg_context(
                user_input, 
                max_entities=max_entities, 
                max_hops=max_hops
            )
        
        # 3. Combine contexts
        combined_context = ""
        
        if doc_context:
            combined_context += "**Document Context:**\n" + doc_context + "\n\n"
        
        if kg_context:
            combined_context += "**Knowledge Graph Context:**\n" + kg_context + "\n\n"
        
        if not combined_context:
            combined_context = "No relevant context found."
        
        # 4. Create enriched prompt
        enriched_prompt = f"""You are a helpful assistant. Use the following context to answer the user's question accurately.

            The context includes:
            1. Document Context: Relevant excerpts from uploaded documents
            2. Knowledge Graph Context: Structured information about entities and their relationships

            Use both sources to provide a comprehensive answer. Prioritize factual information from the knowledge graph when available.
            Keep your answer concise (no more than 3-4 sentences) but informative.

            Context:
            {combined_context}

            User Question:
            {user_input}

            Answer:"""

        response_text = ask(enriched_prompt)

    # Display response with indicator if KG was used
    kg_indicator = " 🧠" if kg_context else ""
    response = f"{response_text}{kg_indicator}"
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Optional: Show what was retrieved (for debugging/transparency)
    if kg_context:
        with st.expander("🔍 View Retrieved Knowledge"):
            st.markdown("**From Documents:**")
            st.text(doc_context[:500] + "..." if len(doc_context) > 500 else doc_context)
            st.markdown("**From Knowledge Graph:**")
            st.markdown(kg_context)