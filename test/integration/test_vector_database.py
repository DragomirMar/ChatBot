import sys
import os
import pytest
from langchain_core.documents import Document

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from database.vector_database import VectorDatabase


@pytest.fixture
def db(tmp_path):
    """Create an isolated vector database for each test."""
    return VectorDatabase(persist_directory=str(tmp_path))


def test_add_and_get_chunks_by_source(db):
    docs = [
        Document(page_content="Berlin is the capital of Germany", metadata={"source": "doc1"}),
        Document(page_content="Paris is the capital of France", metadata={"source": "doc1"}),
    ]

    db.add_document_chunks(docs)

    results = db.get_chunks_by_source("doc1")

    assert len(results) == 2
    assert any("Germany" in d.page_content for d in results)
    assert any("France" in d.page_content for d in results)


def test_similarity_search(db):
    docs = [
        Document(page_content="Python is a programming language", metadata={"source": "tech"}),
        Document(page_content="Football is a popular sport", metadata={"source": "sports"}),
    ]

    db.add_document_chunks(docs)

    results = db.similarity_search("programming", k=1)

    assert len(results) == 1
    assert "Python" in results[0].page_content


def test_delete_by_source(db):
    docs = [
        Document(page_content="Temporary content", metadata={"source": "temp"}),
    ]

    db.add_document_chunks(docs)

    deleted = db.delete_by_source("temp")
    remaining = db.get_chunks_by_source("temp")

    assert deleted is True
    assert remaining == []