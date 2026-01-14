import logging
from typing import List, Tuple
from text_extractor import *
from llm import OllamaModel
from database.vector_database import VectorDatabase
from knowledge_graph.knowledge_graph_retriever import KnowledgeGraphRetriever

logger = logging.getLogger(__name__)

db = VectorDatabase()
kg_retriever = KnowledgeGraphRetriever()
llm = OllamaModel()

def get_sources() -> List[str]:
    return db.get_sources()

def get_knowledge_graph_stats() -> dict:
    return kg_retriever.get_kg_stats()

def clear_database():
    db.clear_database()

def process_url(url) -> Tuple[bool, str]:
    """Process URL and add it to the vector database."""
    try:
        chunks = extract_chunks_from_url(url)
        
        if not chunks:
            return False, "Failed to extract content!"
        
        db.add_document_chunks(chunks)
        
        logger.info(f"URL content added successfully: {url}")
        return True, "Successfully added content to the database"
        
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        return False, f"Error: {str(e)}"

def process_pdfs(uploaded_files) -> Tuple[bool, str]:
    """Process multiple PDF files."""
    try:
        for file in uploaded_files:
            chunks = extract_chunks_from_pdf(file)
            db.add_document_chunks(chunks, file.name)
        
        logger.info("Documents added successfully!")
        return True, f"Successfully processed {len(uploaded_files)} file(s)"
        
    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}")
        return False, f"Error: {str(e)}"

def _create_enriched_prompt(user_input: str, use_kg: bool) -> str:
    """Combines document and KG context to create enriched prompt."""
    # 1. Search vector DB for document chunks
    relevant_chunks = db.similarity_search(user_input, k=5)
    logger.info(f"Found {len(relevant_chunks)} relevant chunks from vector DB")
        
    # 1.1 Extract content and combine chunks
    doc_context = "\n".join([c.page_content for c in relevant_chunks]) if relevant_chunks else ""

    # 2. Retrieve knowledge graph context (if enabled)
    kg_context = kg_retriever.retrieve_kg_context(user_input) if use_kg else ""

    # 3. Combine contexts
    combined_context = ""
    if doc_context:
        combined_context += f"**Document Context:**\n{doc_context}\n\n"
    if kg_context:
        combined_context += f"**Knowledge Graph Context:**\n{kg_context}\n\n"
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
    return enriched_prompt, kg_context, doc_context

def generate_response(user_input: str, use_kg: bool) -> Tuple[str, str, str]:
    prompt, kg_context, doc_context = _create_enriched_prompt(user_input, use_kg)
    response = llm.inference(prompt)
    
    return response, kg_context, doc_context