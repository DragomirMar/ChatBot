import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from typing import List
import logging

logger = logging.getLogger(__name__)

MIN_TOKEN_LEN = 2
MODEL = "en_core_web_sm"

# Load spaCy model
try:
    nlp = spacy.load(MODEL)
    logger.info(f"Loaded spaCy model: {MODEL}")
except OSError:
    logger.error(f"spaCy model '{MODEL}' not found. Run: python -m spacy download {MODEL}")
    raise


def is_valid_token(token) -> bool:
    """Check if a spaCy token is valid for entity extraction."""
    if token.is_punct or token.is_space:
        return False
    lemma = token.lemma_.lower().strip()
    if len(lemma) < MIN_TOKEN_LEN or lemma in STOP_WORDS:
        return False
    if re.fullmatch(r"[^\w]+", lemma):
        return False
    return True


def is_valid_string(text_str: str) -> bool:
    """Check if a string is valid (not stopword, meets min length)."""
    cleaned = text_str.lower().strip()
    return len(cleaned) >= MIN_TOKEN_LEN and cleaned not in STOP_WORDS and not re.fullmatch(r"[^\w]+", cleaned)


def clean_tokens(tokens) -> List[str]:
    """Return list of cleaned tokens from a spaCy token iterable."""
    return [tok.text.strip() for tok in tokens if is_valid_token(tok) and tok.pos_ not in ("VERB", "AUX")]


def _extract_noun_phrases(doc) -> set:
    phrases = set()
    for chunk in doc.noun_chunks:
        cleaned = clean_tokens(chunk)
        if len(cleaned) >= MIN_TOKEN_LEN:
            phrases.add(" ".join(cleaned))
    return phrases


def _extract_named_entities(doc) -> set:
    entities = set()
    for ent in doc.ents:
        cleaned = clean_tokens(ent)
        if cleaned:
            entities.add(" ".join(cleaned))
    return entities


def _extract_single_entities(doc) -> set:
    single_entities = set()
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and is_valid_token(token):
            single_entities.add(token.text.strip())
    return single_entities


def _extract_capitalized_fallback(text: str) -> set:
    """Fallback: extract capitalized words/phrases if NLP misses them."""
    entities = set()
    pattern = r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\b'

    for match in re.finditer(pattern, text):
        phrase = match.group()
        
        # Remove punctuation from the phrase
        cleaned_phrase = phrase.strip('.,!?;:')
        
        # Validate each word in the phrase
        words = cleaned_phrase.split()
        valid_words = [w for w in words if is_valid_string(w)]

        # Add multi-word phrases or single valid words
        if len(valid_words) >= MIN_TOKEN_LEN:  # Multi-word
            entities.add(" ".join(valid_words))
        elif len(valid_words) == 1:
            entities.add(valid_words[0])

    return entities


def extract_entities(text: str) -> List[str]:

    if not text or not text.strip():
        logger.warning("Empty text provided for entity extraction")
        return []

    doc = nlp(text)
    entities = set()

    # Extract multi-word noun phrases
    entities.update(_extract_noun_phrases(doc))
    # Extract named entities (like: PERSON, ORG, GPE, DATE, EVENT, etc.)
    entities.update(_extract_named_entities(doc))
    # Extract single entities (proper nouns, etc.)
    entities.update(_extract_single_entities(doc))
    # Extract capitalized words and  phrases
    entities.update(_extract_capitalized_fallback(text))

    logger.info(f"Extracted {len(entities)} entities from text")
    logger.debug(f"Entities: {entities}")

    return sorted(entities)