import os
import re
import logging
import tempfile
from typing import List, Union

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, 
    chunk_overlap=150,  
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
    add_start_index=True
)

def extract_chunks_from_pdf(
    files: Union[List, object],
    # splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """Extract and chunk text from one or more uploaded PDF files."""

    logger.info("extract_chunks_from_pdf")

    if not isinstance(files, list):
        files = [files]

    all_chunks: List[Document] = []

    for file in files:
        docs = _load_pdf_documents(file)
        if not docs:
            continue

        docs = _normalize_pdf_metadata(docs, source_name=file.name)
        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata["source"] = file.name

        all_chunks.extend(chunks)

    return all_chunks


def extract_chunks_from_url(
    url: str,
    # splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """Fetch a URL, extract clean text, and split it into chunks."""
    
    logger.info(f"extract_chunks_from_url: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        raw_text = _extract_text_from_html(response.content)
        cleaned_text = normalize_text(raw_text)
        
        # Create chunks with proper URL as source
        chunks = splitter.create_documents([cleaned_text], metadatas=[{"source": url}])
        
        # Ensure all chunks have the URL as source
        for chunk in chunks:
            chunk.metadata['source'] = url
            
        logger.info(f"Successfully extracted {len(chunks)} chunks from URL: {url}")
        return chunks
        
    except Exception as e:
        logger.error(f"Error extracting chunks from URL {url}: {str(e)}")
        return []


def normalize_text(text):
    """Normalize, deduplicate, and clean extracted text before chunking."""
    
    # Remove duplicate sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    seen = set()
    unique_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue
        if sentence in seen:
            continue

        seen.add(sentence)
        unique_sentences.append(sentence)

    text = " ".join(unique_sentences)
    
    # Remove author/date patterns
    text = re.sub(r'By\s+[A-Za-z\s-]+\|\s*[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}\.?', '', text)
    text = re.sub(r'\|\s*[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}\.?', '', text)

    # Fix duplicate words
    text = re.sub(r'\b([A-Z][a-z]+)\s*\.\s*\1\b', r'\1', text)
    
    # Clean spacing
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# ---------------------------------------------------------------------------
# PDF helper methods
# ---------------------------------------------------------------------------

def _load_pdf_documents(file) -> List[Document]:
    """Load PDF documents using a temporary file."""

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.getbuffer())
            tmp.flush()
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        return [doc for doc in docs if doc.page_content]

    except Exception as exc:
        logger.error(f"Failed to load PDF {getattr(file, 'name', '')}: {exc}")
        return []

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _normalize_pdf_metadata(
    docs: List[Document],
    source_name: str,
) -> List[Document]:

    normalized_docs: List[Document] = []

    for doc in docs:
        doc.metadata["source"] = source_name
        doc.page_content = doc.page_content.strip()
        normalized_docs.append(doc)

    return normalized_docs


# ---------------------------------------------------------------------------
# HTML helper methods
# ---------------------------------------------------------------------------

def _extract_text_from_html(html: bytes) -> str:
    """Extract structured text from HTML content."""

    soup = BeautifulSoup(html, "html.parser")

    # 1. Try to find main article content
    main = soup.find('main')
    if not main:
        logger.warning("No <main> tag found, falling back to <article>")
        main = soup.find('article')
    
    if not main:
        logger.warning("No <main> or <article> tag found, using <body>")
        main = soup.find('body')

    # 2. Remove unwanted elements from main
    _remove_unwanted_tags(main)

    # 3. Extract text from tags
    tags_to_include = [
        "p", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "blockquote", "pre", "code",
        "strong", "em", "b", "i", "u",
        "mark", "span", "small", "time",
        "summary", "address",
    ]

    parts: List[str] = []

    for tag in main.find_all(tags_to_include):
        text = tag.get_text(separator=" ", strip=True)
        if text:
                # Ensure tags end with punctuation so the text is coherent
                if tag.name != 'p':
                    if not text.endswith(('.', '?', '!', ':', ';')):
                        text += '.'
                parts.append(text)

    return " ".join(parts)


def _remove_unwanted_tags(container) -> None:

    unwanted = [
        "figure", "figcaption", "script", "style",
        "aside", "noscript", "footer", "nav", "header",
    ]

    for tag in container.find_all(unwanted):
        tag.decompose()