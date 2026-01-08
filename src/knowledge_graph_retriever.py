from pymongo import MongoClient
from typing import List, Dict, Tuple
import logging
from entity_extractor import EntityExtractor
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

class KnowledgeGraphRetriever:
    """ Retrieves relevant entities and relationships from MongoDB knowledge graph. """
    
    DEFAULT_FUZZY_THRESHOLD = 70
    DEFAULT_MAX_FUZZY_MATCHES_PER_ENTITY = 5    
    DEFAULT_MAX_MATCHES_PER_ENTITY = 3
    
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client["knowledge_graph"]
        self.entities_collection = self.db["entities"]
        self.relationships_collection = self.db["relationships"]
        
        self.extractor = EntityExtractor()

        self._create_indexes()
        self._all_entity_names = self._load_all_entity_names()
    
    def _load_all_entity_names(self) -> List[str]:
        """ Load all entity names from MongoDB into memory, for fuzzy matching. """
        try:
            all_entities = self.entities_collection.find({}, {"name": 1})
            entity_names = [e["name"] for e in all_entities]
            logger.info(f"Loaded {len(entity_names)} entity names from database")
            return entity_names
        except Exception as e:
            logger.error(f"Error loading entity names: {e}")
            return []
    
    def _create_indexes(self):
        """Create indexes on frequently queried fields for faster queries"""
        try:
            self.entities_collection.create_index("name")
            self.relationships_collection.create_index("subject")
            self.relationships_collection.create_index("object")
            self.relationships_collection.create_index("predicate")
            logger.info("Knowledge graph indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def extract_potential_entities(self, text: str) -> List[str]:
        """Extract potential entities from text using EntityExtractor."""
        return self.extractor.extract_entities(text)
    
    def fuzzy_match_entities(self, entity_name: str, names_list: List[str]) -> List[Tuple[str, float]]:
        """ Fuzzy match extracted entity name against list of knowledge graph entity names. """
        matches = process.extract(
            entity_name,
            names_list,
            scorer=fuzz.WRatio,
            score_cutoff=self.DEFAULT_FUZZY_THRESHOLD,
            limit=self.DEFAULT_MAX_FUZZY_MATCHES_PER_ENTITY
        )
        # Extract only name and score (ignore index from rapidfuzz)
        return [(match[0], match[1]) for match in matches]   
    
    def link_entities(self, extracted_entities: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Entity Linking: Map extracted entities to KG entities.
        
        Returns:
            {
                "Germany": [("Germany", 100.0), ("Germany's Team", 85.0)],
                "Basketball": [("Basketball", 100.0)]
            }
        """
        linked = {}
        
        for entity_name in extracted_entities:
            matches = self.fuzzy_match_entities(
                entity_name, 
                self._all_entity_names
            )
            linked[entity_name] = matches
        
        return linked
            
    def get_entity_info(self, entity_name: str) -> Dict:
        """Retrieves entity information from MongoDB and returns description if entity found."""
        try:
            entity = self.entities_collection.find_one(
                {"name": entity_name.strip().title()}
            )
            return entity if entity else None
        except Exception as e:
            logger.error(f"Error retrieving entity {entity_name}: {e}")
            return None
        
    def _limit_matches(self, linking_results: Dict[str, List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
        """Limit matches for each entity to a certain number. Then remove duplicate entities that may come from other matches."""
        unique_entities = set()
        results = list()

        for entity, matches in linking_results.items():
            for kg_entity in matches[:self.DEFAULT_MAX_MATCHES_PER_ENTITY]:
                if kg_entity[0] not in unique_entities:
                    results.append(kg_entity)
                    unique_entities.add(kg_entity[0])
        
        return results
    
    def get_relationships_for_entity(self, entity_name: str) -> List[Dict]:
        """Get all relationships where the entity is subject or object."""
        max_hops: int = 2
        
        try:
            entity_title = entity_name.strip().title()

            relationships = list(self.relationships_collection.find({
                "$or": [
                    {"subject": entity_title},
                    {"object": entity_title}
                ]
            }))
            
            # # For 2-hop, get relationships of connected entities
            # if max_hops > 1 and relationships:
            #     connected_entities = set()
            #     for rel in relationships:
            #         connected_entities.add(rel["subject"])
            #         connected_entities.add(rel["object"])
                
            #     # Remove the original entity
            #     connected_entities.discard(entity_title)
                
            #     # Get second-hop relationships
            #     if connected_entities:
            #         second_hop = list(self.relationships_collection.find({
            #             "$or": [
            #                 {"subject": {"$in": list(connected_entities)}},
            #                 {"object": {"$in": list(connected_entities)}}
            #             ]
            #         }).limit(20))  # Limit to avoid too much data
                    
            #         relationships.extend(second_hop)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error retrieving relationships for {entity_name}: {e}")
            return []
    
    def _build_kg_context(self, matched_entities: List[Tuple[str, float]]) -> str:
        """ Build formatted knowledge graph context from matched entities. """
        kg_context_parts = []
        processed_entities = set()
        
        for entity_name, confidence_score in matched_entities:
            # Skip if already processed 
            if entity_name in processed_entities:
                continue
            
            processed_entities.add(entity_name)
            
            # Get entity description
            entity_info = self.get_entity_info(entity_name)
            
            # Log entity's info
            if not entity_info:
                logger.debug(f"No entity info found for '{entity_name}'")
                continue
            
            kg_context_parts.append(
                f"\n**Entity: {entity_info['name']}** (confidence: {confidence_score:.1f}%)"
                f"\nDescription: {entity_info['description']}"
            )
            
            # Get and format relationships
            relationships = self.get_relationships_for_entity(entity_info['name'])
            
            if relationships:
                kg_context_parts.append("\nRelationships:")
                
                # Deduplicate relationships
                seen = set()
                for rel in relationships:
                    rel_tuple = (rel['subject'], rel['predicate'], rel['object'])
                    if rel_tuple not in seen:
                        seen.add(rel_tuple)
                        kg_context_parts.append(
                            f"  - {rel['subject']} {rel['predicate']} {rel['object']}"
                        )
        
        logger.info(f"Retrieved context for {len(kg_context_parts)} entities from knowledge graph")
        return "\n".join(kg_context_parts)
    
    def retrieve_kg_context(self, query: str) -> str:
        """Main method to retrieve knowledge graph context for a query."""
        
        # 1. Extract potential entities from user's query
        potential_entities = self.extract_potential_entities(query)
        
        # Log extracted entities
        if potential_entities:
            logger.info(f"Found potential entities in the users query: {potential_entities}")
        else:
            logger.info("No potential entities found in query")
            return ""     
        
        # 2. Link entities to Knowledge Graph
        logger.info("Linking entities to knowledge graph...")
        linking_results = self.link_entities(potential_entities)
        
        # Log linking results
        total_matches = sum(len(matches) for matches in linking_results.values())
        logger.info(f"Found {total_matches} total entity matches across {len(linking_results)} extracted entities")
        
        # Log details for each entity
        for entity_name, matches in linking_results.items():
            if matches:
                logger.info(
                    f"Matches for '{entity_name}': "
                    f"{[(name, f'{score:.1f}') for name, score in matches]}"
                )
            else:
                logger.info(f"No matches found for '{entity_name}'") 
        
        # 3. Limit total matches
        all_matches = self._limit_matches(linking_results)
        
        # Log limited matches
        logger.info(f"All Matches: {all_matches}")
        
        # 4. Retrieve entity's info
        kg_context = self._build_kg_context(all_matches)
    
        if kg_context:
            return kg_context
        else:
            return ""
        
    def get_kg_stats(self) -> Dict:
        try:
            return {
                "total_entities": self.entities_collection.count_documents({}),
                "total_relationships": self.relationships_collection.count_documents({})
            }
        except Exception as e:
            logger.error(f"Error getting KG stats: {e}")
            return {"error": str(e)}