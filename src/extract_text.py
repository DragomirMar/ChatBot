import requests
from bs4 import BeautifulSoup
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import logging

logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, 
    chunk_overlap=150,  
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
    add_start_index=True
)

def extract_chunks_from_pdf(files):
    logger.info(f"extract_chunks_from_pdf")
    
    if not isinstance(files, list):
        files = [files]
    
    all_chunks = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.getbuffer())
            temp_file.flush()

            docs = PyPDFLoader(temp_file.name).load()
            
            # Process each document to filter and clean text
            processed_docs = []
            for doc in docs:
                # Filter and clean the text content
                cleaned_text = doc.page_content
                if cleaned_text: 
                    # Update the metadata to use the original filename instead of temp path
                    doc.metadata['source'] = file.name
                    doc.page_content = cleaned_text
                    processed_docs.append(doc)
            
            # Split the cleaned documents into chunks
            if processed_docs:
                chunks = text_splitter.split_documents(processed_docs)
                # Ensure all chunks have the correct source
                for chunk in chunks:
                    chunk.metadata['source'] = file.name
                all_chunks.extend(chunks)

    return all_chunks

def extract_chunks_from_url(url):
    logger.info(f"extract_chunks_from_url: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 1. Try to find main article content
        main = soup.find('main')
        if not main:
            logger.warning("No <main> tag found, falling back to <article>")
            main = soup.find('article')
        
        if not main:
            logger.warning("No <main> or <article> tag found, using <body>")
            main = soup.find('body')
        
        # 2. Remove unwanted elements 
        unwanted_tags = ['figure', 'figcaption', 'script', 'style', 'aside', 'noscript', 'footer', 'nav', 'header']
        for tag in main.find_all(unwanted_tags):
            tag.decompose()
        
        # 3. Extract text from tags 
        tags_to_include = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'li', 'blockquote', 'pre','code', 'strong', 'em', 'b', 'i', 'u',
        'mark', 'span', 'small', 'time','summary', 'address']
        structured_parts = []
        
        for tag in main.find_all(tags_to_include):
            text = tag.get_text(separator=" ", strip=True)
            if text:
                # Ensure tags end with punctuation so the text is coherent
                if tag.name != 'p':
                    if not text.endswith(('.', '?', '!', ':', ';')):
                        text += '.'
                structured_parts.append(text)

        text = " ".join(structured_parts)

        # 4. Clean whitespace
        cleaned_text = ' '.join(text.split())
        
        # 5. Clean and Split into chunks
        cleaned_text = clean_text_for_chunking(text)
        
        # Create chunks with proper URL as source
        chunks = text_splitter.create_documents([cleaned_text], metadatas=[{"source": url}])
        
        # Ensure all chunks have the URL as source
        for chunk in chunks:
            chunk.metadata['source'] = url
            
        logger.info(f"Successfully extracted {len(chunks)} chunks from URL: {url}")
        return chunks
        
    except Exception as e:
        logger.error(f"Error extracting chunks from URL {url}: {str(e)}")
        return []

def clean_text_for_chunking(text):
    # Remove duplicate sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in seen and len(sentence) > 15:
            unique_sentences.append(sentence)
            seen.add(sentence)
    
    text = ' '.join(unique_sentences)
    
    # Remove author/date patterns
    text = re.sub(r'By\s+[A-Za-z\s-]+\|\s*[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}\.?', '', text)
    text = re.sub(r'\|\s*[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}\.?', '', text)

    # Fix duplicate words
    text = re.sub(r'\b([A-Z][a-z]+)\s*\.\s*\1\b', r'\1', text)
    
    # Clean spacing
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()