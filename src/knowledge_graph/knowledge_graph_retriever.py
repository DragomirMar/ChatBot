from pymongo import MongoClient
from typing import List, Dict, Tuple, Optional
import logging
from entity_extractor import extract_entities
from rapidfuzz import process, fuzz
from knowledge_graph.relationship_strategy import RelationshipStrategy

logger = logging.getLogger(__name__)

class KnowledgeGraphRetriever:
    """ Retrieves relevant entities and relationships from MongoDB knowledge graph. """

    FUZZY_THRESHOLD = 75
    MAX_FUZZY_MATCHES_PER_ENTITY = 2    
    MAX_MATCHES_PER_ENTITY = 3

    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client["knowledge_graph"]
        self.entities_collection = self.db["entities"]
        self.relationships_collection = self.db["relationships"]
        
        self.relationship_strategy = RelationshipStrategy(relationships_collection=self.relationships_collection)

        self._create_indexes()
        self._all_entity_names = self._load_all_entity_names()

    def _create_indexes(self):
        """Creates indexes on frequently queried fields for faster queries"""
        try:
            self.entities_collection.create_index("name")
            self.relationships_collection.create_index("subject")
            self.relationships_collection.create_index("object")
            self.relationships_collection.create_index("predicate")
            logger.info("Knowledge graph indexes created successfully")
        except Exception as e:
            logger.warning(f"Index creation error: {e}")

    def _load_all_entity_names(self) -> List[str]:
        """ Loads all entity names from MongoDB into memory, for fuzzy matching. """
        try:
            all_entities = self.entities_collection.find({}, {"name": 1})
            entity_names = [e["name"] for e in all_entities]
            logger.info(f"Loaded {len(entity_names)} entity names from database")
            return entity_names
        except Exception as e:
            logger.error(f"Failed loading entity names: {e}")
            return []

    def extract_potential_entities(self, text: str) -> List[str]:
        return extract_entities(text)

    def fuzzy_match_entities(self, entity_name: str) -> List[Tuple[str, float]]:
        """ Fuzzy match extract entity name against list of knowledge graph entity names. """
        matches = process.extract(
            entity_name,
            self._all_entity_names,
            scorer=fuzz.WRatio,
            score_cutoff=self.FUZZY_THRESHOLD,
            limit=self.MAX_FUZZY_MATCHES_PER_ENTITY
        )
        return [(match[0], match[1]) for match in matches]   

    def link_entities(self, extracted_entities: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """Maps extracted entities to real Knowledge Graph entities and shows their confidence scores."""
        linked = {}
        for entity in extracted_entities:
            linked[entity] = self.fuzzy_match_entities(entity)
        return linked

    def _limit_matches(self, linking_results: Dict[str, List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
        seen = set()
        results = []
        for matches in linking_results.values():
            for name, score in matches[:self.MAX_MATCHES_PER_ENTITY]:
                if name not in seen:
                    seen.add(name)
                    results.append((name, score))
        return results

    def get_entity_info(self, entity_name: str) -> Optional[Dict]:
        """Retrieves entity information from MongoDB and returns description if entity found."""
        try:
            return self.entities_collection.find_one({"name": entity_name.strip().title()})
        except Exception as e:
            logger.error(f"Error retrieving entity {entity_name}: {e}")
            return None
        
    def get_kg_stats(self) -> Dict:
        return {
            "entities": self.entities_collection.count_documents({}),
            "relationships": self.relationships_collection.count_documents({})
        }

    def _build_kg_context(self, matched_entities: List[Tuple[str, float]]) -> str:
        """ Builds formatted knowledge graph context from matched entities. """
        context_parts = []

        # 1. Retrieve relationships using strategic approach
        logger.debug("=" * 50)
        logger.info("Retrieving relationships using strategic approach...")
        logger.debug("=" * 50)
        
        relationships, strategy = self.relationship_strategy.retrieve_relationships(matched_entities)
        
        logger.info(f"Strategy used: {strategy}")
        logger.info(f"Total relationships retrieved: {len(relationships)}")

        # 2. Format relationships
        if relationships:
            context_parts.append("\n=== RELEVANT RELATIONSHIPS ===")
            for r in relationships:
                context_parts.append(f"- {r['subject']} {r['predicate']} {r['object']}")

        # 3. Format entity descriptions
        context_parts.append("\n=== ENTITY DESCRIPTIONS ===")
        
        processed_entities = set()
        for name, score in matched_entities:
            if name in processed_entities:
                continue
            processed_entities.add(name)
            entity_info = self.get_entity_info(name)
            if entity_info:
                context_parts.append(f"\n**{entity_info['name']}** ({score:.1f}%)")
                context_parts.append(entity_info["description"])
        
        logger.info("=" * 50)
        logger.info(f"Built context with {len(relationships)} relationships and {len(processed_entities)} entity descriptions")
        logger.info("=" * 50)

        return "\n".join(context_parts)

    def retrieve_kg_context(self, query: str) -> str:
        """Main method to retrieve knowledge graph context for a query."""
        
        # 1. Extract potential entities from user's query
        potential_entities = self.extract_potential_entities(query)
        
        # Log results
        if potential_entities:
            logger.info(f"Found potential entities in the user's query: {potential_entities}")
        else:
            logger.info("No potential entities found in query")
            return ""    

        # 2. Link extracted entities to Knowledge Graph entities
        linked_entities = self.link_entities(potential_entities)
        
        # Log linked entities results
        total_matches = sum(len(matches) for matches in linked_entities.values())
        logger.info(f"Found {total_matches} total entity matches across {len(linked_entities)} extracted entities")
        
        for entity_name, matches in linked_entities.items():
            if matches:
                logger.info(
                    f"Matches for '{entity_name}': "
                    f"{[(name, f'{score:.1f}') for name, score in matches]}"
                )
            else:
                logger.info(f"No matches found for '{entity_name}'") 
                
        # 3. Limit total matches 
        matches = self._limit_matches(linked_entities)
        
        # Log limited matches
        logger.info(f"Limited to {len(matches)} matched entities: {[name for name, _ in matches]}")
        
        # 4. Retrieve Knowledge Graph context
        kg_context = self._build_kg_context(matches)
        
        return kg_context if kg_context else ""