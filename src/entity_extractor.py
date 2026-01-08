import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from typing import List
import logging

logger = logging.getLogger(__name__)

class EntityExtractor:
    MIN_TOKEN_LEN = 2
    MODEL = "en_core_web_sm"
    
    def __init__(self, model: str = MODEL):
        try:
            self.nlp = spacy.load(model)
            logger.info(f"Loaded spaCy model: {model}")
        except OSError:
            logger.error(f"spaCy model '{model}' not found. Run: python -m spacy download {model}")
            raise
    
    def is_valid_token(self, token) -> bool:
        """Check if token is valid."""
        if token.is_punct or token.is_space:
            return False
        lemma = token.lemma_.lower().strip()
        if len(lemma) < self.MIN_TOKEN_LEN or lemma in STOP_WORDS:
            return False
        if re.fullmatch(r"[^\w]+", lemma):
            return False
        return True

    def is_valid_string(self, text_str) -> bool:
        """Check if a string is valid (not stopword, meets min length)."""
        cleaned = text_str.lower().strip()
        return (
            len(cleaned) >= self.MIN_TOKEN_LEN 
            and cleaned not in STOP_WORDS 
            and not re.fullmatch(r"[^\w]+", cleaned)
        )
        
    def clean_tokens(self, tokens)-> List[str]:
        """Return list of cleaned tokens."""
        return [
            tok.text.strip() for tok in tokens 
                if self.is_valid_token(tok) and tok.pos_ not in ("VERB", "AUX")
        ]
        
    def _extract_noun_phrases(self, doc) -> set:
        """Extract multi-word noun phrases."""
        phrases = set()
        for chunk in doc.noun_chunks:
            cleaned = self.clean_tokens(chunk)
            if len(cleaned) >= self.MIN_TOKEN_LEN:
                phrases.add(" ".join(cleaned))
        return phrases
                
    def _extract_named_entities(self, doc) -> set:
        """Extract spaCy named entities."""
        entities = set()
        for ent in doc.ents:
            cleaned = self.clean_tokens(ent)
            if cleaned:
                entities.add(" ".join(cleaned))
        return entities
    
    def _extract_single_entities(self, doc) -> set:
        """Extract single-word entities (may divide phrases in single words, but it is useful for comparison with knowledge graph entities later).""" 
        single_entities = set()
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and self.is_valid_token(token):
                 single_entities.add(token.text.strip())
        return single_entities
    
    def _extract_capitalized_fallback(self, text: str) -> set:
        """
        A fallback/safety method which extracts capitalized words and phrases
        using pattern matching, in case NLP methods miss some entities.
        """
        entities = set()
        pattern = r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\b'
        
        for match in re.finditer(pattern, text):
            phrase = match.group()
            # Remove punctuation from the phrase
            cleaned_phrase = phrase.strip('.,!?;:')
            
            # Validate each word in the phrase
            words = cleaned_phrase.split()
            valid_words = [w for w in words if self.is_valid_string(w)]
            
            # Add multi-word phrases OR single valid words
            if len(valid_words) >= self.MIN_TOKEN_LEN:  # Multi-word
                entities.add(" ".join(valid_words))
            elif len(valid_words) == 1:  # Single word
                entities.add(valid_words[0])
        
        return entities

    def extract_entities(self, text: str)-> List[str]:
        """ Extract entities from text using spaCy NLP and 
        capitalized words and phrases (as fallback). """
        
        if not text or not text.strip():
            logger.warning("Empty text provided for entity extraction")
            return []
        
        doc = self.nlp(text)
        entities = set()
        
        # Extract multi-word noun phrases
        entities.update(self._extract_noun_phrases(doc))
        # Extract named entities (like: PERSON, ORG, GPE, DATE, EVENT, etc.)
        entities.update(self._extract_named_entities(doc))
        # Extract single entities (proper nouns, etc.)
        entities.update(self._extract_single_entities(doc))
        # Extract capitalized words and  phrases
        entities.update(self._extract_capitalized_fallback(text))

        logger.info(f"Extracted {len(entities)} entities from query")
        logger.debug(f"Entities: {entities}")
        
        return sorted(entities)