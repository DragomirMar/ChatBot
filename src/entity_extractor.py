import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from typing import List
import logging

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Fast entity extraction using spaCy NLP and pattern matching.
    Designed for real-time query processing in RAG + KG systems.
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize the entity extractor.
        
        Args:
            model: spaCy model to use (default: en_core_web_sm)
        """
        try:
            self.nlp = spacy.load(model)
            logger.info(f"Loaded spaCy model: {model}")
        except OSError:
            logger.error(f"spaCy model '{model}' not found. Run: python -m spacy download {model}")
            raise
    
    def is_valid_token(self, token, min_token_len=2) -> bool:
        """Check if token should be included."""
        if token.is_punct or token.is_space:
            return False
        lemma = token.lemma_.lower().strip()
        if len(lemma) < min_token_len or lemma in STOP_WORDS:
            return False
        if re.fullmatch(r"[^\w]+", lemma):
            return False
        return True

    def is_valid_string(self, text_str, min_token_len=2) -> bool:
        """Check if a string is valid (not stopword, meets min length)."""
        cleaned = text_str.lower().strip()
        return (
            len(cleaned) >= min_token_len 
            and cleaned not in STOP_WORDS 
            and not re.fullmatch(r"[^\w]+", cleaned)
        )
        
    def clean_tokens(self, tokens, preserve_case=True)-> List[str]:
        """Return list of cleaned tokens, optionally preserving original case."""
        return [
            tok.text.strip() if preserve_case else tok.lemma_.lower().strip()
            for tok in tokens 
            if self.is_valid_token(tok) and tok.pos_ not in ("VERB", "AUX")
        ]

    def extract_entities(self, text, min_token_len=2, include_singles=False, use_simple_fallback=True)-> List[str]:
        """
        Extract meaningful entities from text using spaCy NLP and capitalized words and phrases.
        
        Args:
            text: Input text to extract entities from
            min_token_len: Minimum character length for tokens (default: 2)
            include_singles: Include single-word nouns/proper nouns (default: False)
            use_simple_fallback: Also use simple capitalization-based extraction (default: True)
        
        Returns:
            Sorted list of unique entity strings (preserving original casing)
        """
        
        if not text or not text.strip():
            logger.warning("Empty text provided for entity extraction")
            return []
        
        doc = self.nlp(text)
        entities = set()
        
        
        # Extract multi-word noun phrases (preserve original casing)
        for chunk in doc.noun_chunks:
            cleaned = self.clean_tokens(chunk, preserve_case=True)
            if len(cleaned) >= 2:
                entities.add(" ".join(cleaned))
        
        # Extract named entities - these are the most important (PERSON, ORG, GPE, DATE, EVENT, etc.)
        for ent in doc.ents:
            cleaned = self.clean_tokens(ent, preserve_case=True)
            if len(cleaned) >= 2 or (len(cleaned) == 1 and include_singles):
                entities.add(" ".join(cleaned))
        
        # Optional: single important tokens (proper nouns, etc.)
        if include_singles:
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN") and self.is_valid_token(token):
                    entities.add(token.text.strip())
        
        # Simple fallback: capitalized words (useful for catching missed entities)
        if use_simple_fallback:
            # Find sequences of capitalized words
            cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            for phrase in cap_phrases:
                # Check each word in the phrase
                words = phrase.split()
                valid_words = [w for w in words if self.is_valid_string(w)]
                if len(valid_words) >= 2:  # Keep multi-word phrases
                    entities.add(" ".join(valid_words))
                elif len(valid_words) == 1 and include_singles:  # Keep singles if requested
                    entities.add(valid_words[0])
            
            # Individual capitalized words
            words = text.split()
            for word in words:
                if word and word[0].isupper():
                    cleaned_word = word.strip('.,!?;:')
                    if self.is_valid_string(cleaned_word):
                        if include_singles or len(cleaned_word.split()) > 1:
                            entities.add(cleaned_word)
        
        
        # Title case all entities to match MongoDB format
        entities_title_cased = [entity.title() for entity in entities]
        
        logger.info(f"Extracted {len(entities_title_cased)} entities from query")
        logger.debug(f"Entities: {entities_title_cased}")
        
        return sorted(entities_title_cased)