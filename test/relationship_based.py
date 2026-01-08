"""
RELATIONSHIP-FOCUSED APPROACH
Key insight: Relationships between entities are more valuable than entity descriptions alone.
"""
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pymongo import MongoClient
from typing import List, Dict, Tuple, Set
import logging
from entity_extractor import EntityExtractor
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

class RelationshipFocusedKGRetriever:
    """
    Focus on relationships between query entities rather than all relationships.
    This provides the graph structure value that RAG cannot provide.
    """
    
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client["knowledge_graph"]
        self.entities_collection = self.db["entities"]
        self.relationships_collection = self.db["relationships"]
        
        self._create_indexes()
        self._all_entity_names = self._load_all_entity_names()
    
    def _load_all_entity_names(self) -> List[str]:
        try:
            all_entities = self.entities_collection.find({}, {"name": 1})
            return [e["name"] for e in all_entities]
        except Exception as e:
            logger.error(f"Error loading entity names: {e}")
            return []
    
    def _create_indexes(self):
        try:
            self.entities_collection.create_index("name")
            self.relationships_collection.create_index([("subject", 1), ("object", 1)])
            self.relationships_collection.create_index([("subject", 1), ("predicate", 1)])
            self.relationships_collection.create_index([("object", 1), ("predicate", 1)])
            logger.info("Indexes created")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def extract_potential_entities(self, text: str) -> List[str]:
        extractor = EntityExtractor()
        return extractor.extract_entities(
            text, min_token_len=2, include_singles=True, use_simple_fallback=True
        )
    
    def fuzzy_match_entities(
        self, 
        entity_name: str, 
        names_list: List[str], 
        threshold: int = 70, 
        limit: int = 3
    ) -> List[Tuple[str, float]]:
        matches = process.extract(
            entity_name, names_list, scorer=fuzz.WRatio, 
            score_cutoff=threshold, limit=limit
        )
        return [(match[0], match[1]) for match in matches]
    
    def link_entities(
        self,
        extracted_entities: List[str],
        threshold: int = 70,
        max_matches_per_entity: int = 3
    ) -> Dict[str, List[Tuple[str, float]]]:
        linking_results = {}
        
        for entity in extracted_entities:
            matches = self.fuzzy_match_entities(
                entity, self._all_entity_names, threshold, max_matches_per_entity
            )
            linking_results[entity] = matches
        
        return linking_results
    
    def get_entity_info(self, entity_name: str) -> Dict:
        try:
            return self.entities_collection.find_one(
                {"name": entity_name.strip().title()}
            )
        except Exception as e:
            logger.error(f"Error retrieving entity: {e}")
            return None
    
    # ========================================================================
    # NEW: RELATIONSHIP-CENTRIC METHODS
    # ========================================================================
    
    def get_direct_relationships_between_entities(
        self, 
        entity_set: Set[str]
    ) -> List[Dict]:
        """
        Get relationships that CONNECT entities in the set.
        This is the MOST valuable information - shows how query entities relate.
        
        Example:
            entity_set = {"Germany", "Lithuania", "Basketball"}
            
            Returns:
            - Germany defeated Lithuania (connects Germany + Lithuania) ✓
            - Germany plays_in Eurobasket (connects Germany + Basketball) ✓
            - Lithuania plays_in Eurobasket (connects Lithuania + Basketball) ✓
            
            Does NOT return:
            - Germany located_in Europe (only involves Germany) ✗
            - Random_Country plays_in Eurobasket (doesn't involve query entities) ✗
        """
        entity_list = [e.strip().title() for e in entity_set]
        
        try:
            # Find relationships where BOTH subject and object are in our entity set
            connecting_relationships = list(self.relationships_collection.find({
                "subject": {"$in": entity_list},
                "object": {"$in": entity_list}
            }))
            
            logger.info(
                f"Found {len(connecting_relationships)} relationships connecting "
                f"{len(entity_set)} entities"
            )
            
            return connecting_relationships
            
        except Exception as e:
            logger.error(f"Error getting connecting relationships: {e}")
            return []
    
    def get_contextual_relationships(
        self,
        entity_set: Set[str],
        max_per_entity: int = 5
    ) -> List[Dict]:
        """
        Get a LIMITED set of additional relationships for context.
        These provide background but aren't the main graph structure.
        
        Strategy: Only get relationships that involve query entities.
        """
        entity_list = [e.strip().title() for e in entity_set]
        contextual_rels = []
        
        try:
            for entity in entity_list:
                # Get limited relationships for THIS entity
                entity_rels = list(self.relationships_collection.find({
                    "$or": [
                        {"subject": entity},
                        {"object": entity}
                    ]
                }).limit(max_per_entity))
                
                contextual_rels.extend(entity_rels)
            
            logger.info(f"Found {len(contextual_rels)} contextual relationships")
            return contextual_rels
            
        except Exception as e:
            logger.error(f"Error getting contextual relationships: {e}")
            return []
    
    def deduplicate_relationships(
        self, 
        relationships: List[Dict]
    ) -> List[Dict]:
        """
        Remove duplicate relationships.
        (subject, predicate, object) should appear only once.
        """
        seen = set()
        unique_rels = []
        
        for rel in relationships:
            rel_tuple = (
                rel.get('subject', ''),
                rel.get('predicate', ''),
                rel.get('object', '')
            )
            
            if rel_tuple not in seen:
                seen.add(rel_tuple)
                unique_rels.append(rel)
        
        logger.info(f"Deduplicated: {len(relationships)} → {len(unique_rels)} relationships")
        return unique_rels
    
    def prioritize_relationships(
        self,
        relationships: List[Dict],
        entity_set: Set[str],
        query: str
    ) -> List[Dict]:
        """
        Prioritize relationships by relevance.
        
        Priority order:
        1. Relationships connecting query entities (highest value)
        2. Relationships matching query keywords
        3. Other relationships (background context)
        """
        entity_list = [e.lower() for e in entity_set]
        query_lower = query.lower()
        
        # Score each relationship
        scored_rels = []
        
        for rel in relationships:
            score = 0
            subject = rel.get('subject', '').lower()
            predicate = rel.get('predicate', '').lower()
            obj = rel.get('object', '').lower()
            
            # Priority 1: Connects two query entities (HIGHEST VALUE)
            if subject in entity_list and obj in entity_list:
                score += 100
            
            # Priority 2: Predicate matches query keywords
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3 and word in predicate:
                    score += 50
            
            # Priority 3: Involves at least one query entity
            if subject in entity_list or obj in entity_list:
                score += 10
            
            scored_rels.append((score, rel))
        
        # Sort by score (highest first)
        scored_rels.sort(key=lambda x: x[0], reverse=True)
        
        prioritized = [rel for score, rel in scored_rels]
        
        logger.info(
            f"Prioritized relationships - "
            f"Top priority: {sum(1 for s, _ in scored_rels if s >= 100)}, "
            f"Medium: {sum(1 for s, _ in scored_rels if 10 < s < 100)}, "
            f"Low: {sum(1 for s, _ in scored_rels if s <= 10)}"
        )
        
        return prioritized
    
    # ========================================================================
    # MAIN RETRIEVAL WITH RELATIONSHIP FOCUS
    # ========================================================================
    
    def retrieve_kg_context(
        self,
        query: str,
        fuzzy_threshold: int = 70,
        fuzzy_matches_per_entity: int = 3,
        max_total_entities: int = 12,
        max_connecting_relationships: int = 20,
        max_contextual_relationships_per_entity: int = 3,
        include_entity_descriptions: bool = True
    ) -> Tuple[str, List[str], Dict, Dict]:
        """
        Relationship-focused retrieval.
        
        Strategy:
        1. Extract and link entities (your balanced approach)
        2. Find relationships BETWEEN linked entities (THE KEY VALUE!)
        3. Add limited contextual relationships for background
        4. Optionally include entity descriptions
        
        Args:
            query: User's question
            fuzzy_threshold: Entity matching threshold
            fuzzy_matches_per_entity: Max KG entities per extracted entity
            max_total_entities: Limit on total entities (with sorting if needed)
            max_connecting_relationships: Max relationships between query entities
            max_contextual_relationships_per_entity: Background relationships per entity
            include_entity_descriptions: Whether to include entity descriptions
        
        Returns:
            (context, linked_entities, linking_details, relationship_stats)
        """
        # Step 1: Extract entities
        potential_entities = self.extract_potential_entities(query)
        
        if not potential_entities:
            return "", [], {}, {}
        
        logger.info(f"Extracted: {potential_entities}")
        
        # Step 2: Link entities with per-entity balance
        linking_results = self.link_entities(
            potential_entities, fuzzy_threshold, fuzzy_matches_per_entity
        )
        
        # Step 3: Collect all linked entities
        all_linked_entities = set()
        for matches in linking_results.values():
            for kg_entity, score in matches:
                all_linked_entities.add(kg_entity)
        
        if not all_linked_entities:
            return "", [], linking_results, {}
        
        # Step 4: Apply limit if too many entities
        linked_entities_list = list(all_linked_entities)
        if len(linked_entities_list) > max_total_entities:
            logger.info(
                f"Limiting entities: {len(linked_entities_list)} → {max_total_entities}"
            )
            
            # Flatten with scores for sorting
            entities_with_scores = []
            for entity, matches in linking_results.items():
                for kg_entity, score in matches:
                    entities_with_scores.append((kg_entity, score))
            
            # Sort by confidence and limit
            entities_with_scores.sort(key=lambda x: x[1], reverse=True)
            linked_entities_list = [
                e for e, s in entities_with_scores[:max_total_entities]
            ]
            all_linked_entities = set(linked_entities_list)
        
        logger.info(f"Using {len(all_linked_entities)} entities")
        
        # ====================================================================
        # RELATIONSHIP RETRIEVAL (THE MAIN VALUE)
        # ====================================================================
        
        # Step 5: Get connecting relationships (HIGHEST PRIORITY)
        connecting_rels = self.get_direct_relationships_between_entities(
            all_linked_entities
        )
        
        # Step 6: Get contextual relationships (BACKGROUND)
        contextual_rels = self.get_contextual_relationships(
            all_linked_entities,
            max_per_entity=max_contextual_relationships_per_entity
        )
        
        # Step 7: Combine and deduplicate
        all_relationships = connecting_rels + contextual_rels
        all_relationships = self.deduplicate_relationships(all_relationships)
        
        # Step 8: Prioritize and limit
        prioritized_rels = self.prioritize_relationships(
            all_relationships, all_linked_entities, query
        )
        
        # Limit total relationships
        max_total_rels = max_connecting_relationships + (
            max_contextual_relationships_per_entity * len(all_linked_entities)
        )
        prioritized_rels = prioritized_rels[:max_total_rels]
        
        relationship_stats = {
            "connecting_relationships": len(connecting_rels),
            "contextual_relationships": len(contextual_rels),
            "total_after_dedup": len(all_relationships),
            "final_count": len(prioritized_rels)
        }
        
        # ====================================================================
        # BUILD CONTEXT
        # ====================================================================
        
        kg_context_parts = []
        
        # Part 1: Relationships (THE MAIN VALUE)
        if prioritized_rels:
            kg_context_parts.append("\n**Knowledge Graph Relationships:**")
            
            for rel in prioritized_rels:
                kg_context_parts.append(
                    f"  - {rel['subject']} {rel['predicate']} {rel['object']}"
                )
        
        # Part 2: Entity descriptions (optional, supplementary)
        if include_entity_descriptions:
            kg_context_parts.append("\n**Entity Descriptions:**")
            
            for entity_name in linked_entities_list:
                entity_info = self.get_entity_info(entity_name)
                if entity_info:
                    kg_context_parts.append(
                        f"\n{entity_info['name']}: {entity_info['description']}"
                    )
        
        context = "\n".join(kg_context_parts)
        
        logger.info(
            f"Built context: {len(all_linked_entities)} entities, "
            f"{len(prioritized_rels)} relationships"
        )
        
        return context, linked_entities_list, linking_results, relationship_stats
    
    def get_kg_stats(self) -> Dict:
        try:
            return {
                "total_entities": self.entities_collection.count_documents({}),
                "total_relationships": self.relationships_collection.count_documents({})
            }
        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("RELATIONSHIP-FOCUSED KG RETRIEVAL TEST")
    print("="*80)
    
    kg = RelationshipFocusedKGRetriever()
    
    test_query = "What was the score between Germany and Lithuania in basketball?"
    
    print(f"\nQuery: {test_query}\n")
    
    context, entities, linking, rel_stats = kg.retrieve_kg_context(
        test_query,
        fuzzy_threshold=70,
        fuzzy_matches_per_entity=3,
        max_total_entities=12,
        max_connecting_relationships=20,
        max_contextual_relationships_per_entity=3,
        include_entity_descriptions=True
    )
    
    print(f"Linked Entities ({len(entities)}):")
    for e in entities:
        print(f"  • {e}")
    
    print(f"\nRelationship Statistics:")
    for key, value in rel_stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nContext Preview:")
    print(context[:800])
    if len(context) > 800:
        print("...")
        
        
        
        
        
        
        














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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_chunks_from_url(
    url: str,
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """Fetch a URL, extract clean text, and split it into chunks."""

    logger.info(f"extract_chunks_from_url: {url}")

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        raw_text = _extract_text_from_html(response.content)
        cleaned_text = preprocess_extracted_text(raw_text)

        if not cleaned_text:
            return []

        chunks = splitter.create_documents(
            [cleaned_text],
            metadatas=[{"source": url}],
        )

        logger.info(f"Successfully extracted {len(chunks)} chunks from URL: {url}")
        return chunks

    except Exception as exc:
        logger.error(f"Error extracting chunks from URL {url}: {exc}")
        return []


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# HTML helpers
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


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_extracted_text(text: str) -> str:
    """Normalize, deduplicate, and clean extracted text before chunking."""

    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)

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

    # Remove author / date patterns
    text = re.sub(r"By\s+[A-Za-z\s-]+\|\s*[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}\.?","",text)
    text = re.sub(r"\|\s*[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}\.?","", text)

    # Fix duplicated capitalized words (e.g., "Title. Title")
    text = re.sub(r"\b([A-Z][a-z]+)\s*\.\s*\1\b", r"\1", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

