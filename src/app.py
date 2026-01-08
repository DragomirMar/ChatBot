import streamlit as st
from configuration.logger_config import setup_logging
import rag_service

setup_logging()


st.set_page_config(page_title="RAG ChatBot with Knowledge Graph", layout="wide")
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
            st.write("Processing document(s)...")
            success, message = rag_service.process_pdfs(uploaded_files)
            if success:
                st.success(message)
            else:
                st.error(message)
        else:
            st.warning("Please select at least one PDF file.")

# URL Upload
with col2:
    st.subheader("Add URLs")
    input_url = st.text_input("Enter URL", placeholder="https://example.com/article")
    
    if st.button("Confirm URL"):
        if input_url.strip():
            st.write("Processing url(s)...")
            success, message = rag_service.process_url(input_url.strip())
            if success:
                st.success(message)
            else:
                st.error(message)
        else:
            st.warning("Please enter a valid URL.")

st.markdown("---")

# Database Operations
col1, col2 = st.columns([1,1]) 
with col1:
    if st.button("Show Sources"):
        # Sources
        sources = rag_service.get_sources()
        st.write("Sources in the database:")
        for source in sources:
            st.write(f"- {source}")
        
        # KG stats
        kg_stats = rag_service.get_knowledge_graph_stats()
        st.write("\nKnowledge Graph Statistics:")
        st.write(f"- Entities: {kg_stats.get('total_entities', 0)}")
        st.write(f"- Relationships: {kg_stats.get('total_relationships', 0)}")

with col2:
    if st.button("Clear Database"):
        rag_service.clear_database()
        st.write("Database cleared.")

# Knowledge Graph Settings (collapsible)
with st.expander("⚙️ Knowledge Graph Settings"):
    use_kg = st.checkbox("Enable Knowledge Graph Enhancement", value=True, 
                         help="Use knowledge graph to enhance answers with entity information")           
        
# Chatbot
st.title('Chat Bot')
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if st.button("Clear Chat", help="Clear chat"):
    st.session_state.messages = []
    st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input field, pinned to bottom
user_input = st.chat_input("Type your message...")
if user_input:
    # Show user's message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Response 
    with st.spinner("KG Bot is thinking..."):
        response, kg_context, doc_context = rag_service.generate_response(user_input, use_kg)
        
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        if kg_context:
            with st.expander("🔍 View Retrieved Knowledge"):
                st.markdown("**From Documents:**")
                st.text(doc_context[:500] + "..." if len(doc_context) > 500 else doc_context)
                st.markdown("**From Knowledge Graph:**")
                st.markdown(kg_context)