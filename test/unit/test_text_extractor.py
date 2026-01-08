import pytest
import sys
import os
from langchain.schema import Document

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from text_extractor import (
    extract_chunks_from_pdf,
    extract_chunks_from_url,
    normalize_text,
    _extract_text_from_html,
)


def test_normalize_text_removes_duplicates():
    text = (
        "Hello good world. Hello good world. "
        "This is a unique sentence. This is a unique sentence."
    )

    cleaned = normalize_text(text)

    assert cleaned.count("Hello good world") == 1
    assert cleaned.count("This is a unique sentence") == 1


def test_normalize_text_removes_short_sentences():
    text = "Hi. This sentence is long enough to stay. Ok."

    cleaned = normalize_text(text)

    assert "Hi." not in cleaned
    assert "Ok." not in cleaned
    assert "long enough" in cleaned


def test_extract_text_from_html_basic():
    html = b"""
    <html>
        <body>
            <h1>Title</h1>
            <p>This is a paragraph.</p>
            <script>alert('x')</script>
        </body>
    </html>
    """

    text = _extract_text_from_html(html)

    assert "Title." in text
    assert "This is a paragraph." in text
    assert "alert" not in text


def test_extract_chunks_from_pdf(monkeypatch):
    dummy_doc = Document(page_content="Some PDF text", metadata={})

    def mock_load_pdf_documents(file):
        return [dummy_doc]

    monkeypatch.setattr(
        'text_extractor._load_pdf_documents',
        mock_load_pdf_documents,
    )

    class DummyFile:
        name = "test.pdf"

    chunks = extract_chunks_from_pdf(DummyFile())

    assert len(chunks) > 0
    assert chunks[0].metadata["source"] == "test.pdf"


def test_extract_chunks_from_url(monkeypatch):
    html = b"<html><body><p>Some web content.</p></body></html>"

    class MockResponse:
        content = html

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        'text_extractor.requests.get',
        lambda url: MockResponse(),
    )

    chunks = extract_chunks_from_url("https://example.com")

    assert len(chunks) > 0
    assert chunks[0].metadata["source"] == "https://example.com"