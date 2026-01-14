import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from entity_extractor import (
    extract_entities,
    _extract_capitalized_fallback,
    clean_tokens,
    is_valid_token,
    is_valid_string
)
import spacy
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL = "en_core_web_sm"
nlp = spacy.load(MODEL)


def test_extract_entities_basic():
    text = "Barack Obama was the 44th President of the United States."
    entities = extract_entities(text)
    logger.info(f"Extracted entities: {entities}")
    assert "Barack Obama" in entities
    assert "United States" in entities


def test_extract_entities_capitalized_fallback():
    text = "This text mentions Python and OpenAI models."
    entities = extract_entities(text)
    logger.info(f"Extracted entities: {entities}")
    assert "Python" in entities
    assert "OpenAI" in entities
    
    
def test_extract_entities_next_phrase():
    text = "IBM are working on developing AI technologies in New York City for the new iPhone."
    entities = extract_entities(text)
    logger.info(f"Extracted entities: {entities}")
    assert "IBM" in entities
    assert "New York City" in entities
    assert "AI" in entities
    assert "iPhone" in entities or "Phone" in entities


def test_is_valid_string():
    assert is_valid_string("Hello")
    assert not is_valid_string("a")
    assert not is_valid_string("the")


def test_is_valid_token():
    doc = nlp("Hello world!")
    token = doc[0]  # "Hello"
    assert is_valid_token(token)
    token = doc[1]  # "world"
    assert is_valid_token(token)
    token = doc[2]  # "!"
    assert not is_valid_token(token)


def test_clean_tokens():
    doc = nlp("Hello world!")
    tokens = clean_tokens(doc)
    assert "Hello" in tokens
    assert "world" in tokens
    assert "!" not in tokens


def test_capitalized_fallback():
    text = "This text mentions OpenAI and ChatGPT models."
    entities = _extract_capitalized_fallback(text)
    assert "OpenAI" in entities
    assert "ChatGPT" in entities
