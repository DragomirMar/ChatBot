from pymongo import MongoClient
from typing import List, Dict, Tuple
import logging
from entity_extractor import EntityExtractor
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

class KnowledgeGraphRetriever:
    """
    Retrieves relevant entities and relationships from MongoDB knowledge graph
    to enhance RAG-based question answering.
    """
    
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client["knowledge_graph"]
        self.entities_collection = self.db["entities"]
        self.relationships_collection = self.db["relationships"]
        
        # Create indexes for faster queries
        self._create_indexes()
        
         # Load all entity names for fuzzy matching
        self._all_entity_names = self._load_all_entity_names()
    
    def _load_all_entity_names(self) -> List[str]:
        """
        Load all entity names from MongoDB into memory.
        This allows to compare query entities against all KG entities.
        """
        try:
            all_entities = self.entities_collection.find({}, {"name": 1})
            entity_names = [e["name"] for e in all_entities]
            logger.info(f"Loaded {len(entity_names)} entity names from database")
            return entity_names
        except Exception as e:
            logger.error(f"Error loading entity names: {e}")
            return []
    
    def _create_indexes(self):
        """Create indexes on frequently queried fields"""
        try:
            self.entities_collection.create_index("name")
            self.relationships_collection.create_index("subject")
            self.relationships_collection.create_index("object")
            self.relationships_collection.create_index("predicate")
            logger.info("Knowledge graph indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def extract_potential_entities(self, text: str) -> List[str]:
        """
        Extract potential entity names from users query. These will be used to find relevant entities
        in the knowledge graph.
        """
        extractor = EntityExtractor()
        potential_entities = extractor.extract_entities(
            text, min_token_len=2, include_singles=True, use_simple_fallback=True
        )
        return potential_entities
    
    def fuzzy_match_entities(self, entity_name: str, names_list: List[str], threshold: int = 60, limit: int = 5) -> List[Tuple[str, float]]:
        matches = process.extract(
            entity_name,
            names_list,
            scorer=fuzz.WRatio,
            score_cutoff=threshold,
            limit=limit
        )
        return [(match[0], match[1]) for match in matches]   
    
    def get_entity_info(self, entity_name: str) -> Dict:
        """
        Retrieve entity information from MongoDB.
        Returns entity description if found.
        """
        try:
            entity = self.entities_collection.find_one(
                {"name": entity_name.strip().title()}
            )
            return entity if entity else None
        except Exception as e:
            logger.error(f"Error retrieving entity {entity_name}: {e}")
            return None
    
    def get_relationships_for_entity(self, entity_name: str, max_hops: int = 1) -> List[Dict]:
        """
        Get all relationships where the entity is subject or object.
        
        Args:
            entity_name: Name of the entity
            max_hops: Number of relationship hops to traverse 
        
        Returns:
            List of relationship dictionaries
        """
        try:
            entity_title = entity_name.strip().title()
            
            # Find relationships where entity is subject or object
            relationships = list(self.relationships_collection.find({
                "$or": [
                    {"subject": entity_title},
                    {"object": entity_title}
                ]
            }))
            
            # For 2-hop, get relationships of connected entities
            if max_hops > 1 and relationships:
                connected_entities = set()
                for rel in relationships:
                    connected_entities.add(rel["subject"])
                    connected_entities.add(rel["object"])
                
                # Remove the original entity
                connected_entities.discard(entity_title)
                
                # Get second-hop relationships
                if connected_entities:
                    second_hop = list(self.relationships_collection.find({
                        "$or": [
                            {"subject": {"$in": list(connected_entities)}},
                            {"object": {"$in": list(connected_entities)}}
                        ]
                    }).limit(20))  # Limit to avoid too much data
                    
                    relationships.extend(second_hop)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error retrieving relationships for {entity_name}: {e}")
            return []
    
    def retrieve_kg_context(self,
                            query: str,
                            max_entities: int = 5,
                            max_hops: int = 1,
                            fuzzy_threshold: int = 70,       
                            fuzzy_matches_per_entity: int = 5  
                            ) -> str:
        """
        Main method to retrieve knowledge graph context for a query.
        
        Args:
            query: User's question
            max_entities: Maximum number of entities to retrieve
            max_hops: Number of relationship hops (1-2 recommended)
        
        Returns:
            Formatted string with entity descriptions and relationships
        """
        
        # 1. Extract potential entities from user's query
        potential_entities = self.extract_potential_entities(query)
        
        # Log extracted entities
        if potential_entities:
            logger.info(f"Found potential entities: {potential_entities}")
        else:
            logger.info("No potential entities found in query")
            return ""
        
        # Context helpers
        kg_context_parts = []
        entities_found = 0
        processed_entities = set()  
    
        # 2. Retrieve entities, description and relationships
        for entity_name in potential_entities[:max_entities]:
            
            # 2.1 Fuzzy match potential entity's name against KG entities
            fuzzy_matches = self.fuzzy_match_entities(
                entity_name, 
                self._all_entity_names, 
                threshold=fuzzy_threshold,
                limit=fuzzy_matches_per_entity 
            )
            
            if fuzzy_matches:
                logger.info(
                    f"Fuzzy matches for '{entity_name}': "
                    f"{[(name, f'{score:.1f}') for name, score in fuzzy_matches]}"
                )
            else:
                logger.info(f"No fuzzy matches found for '{entity_name}'")
            
            for matched_entity, confidence_score in fuzzy_matches:    
                 # Skip if already processed 
                if matched_entity in processed_entities:
                    continue
                
                processed_entities.add(matched_entity)  
            
                # 2.2 Get entity description
                entity_info = self.get_entity_info(matched_entity)
                
                if entity_info:
                    entities_found += 1
                    kg_context_parts.append(f"\n**Entity: {entity_info['name']}**"
                                            f"(matched '{entity_name}' with {confidence_score:.1f}% confidence)"
                                            )
                    kg_context_parts.append(f"Description: {entity_info['description']}")
                    
                    # 2.3 Get relationships
                    relationships = self.get_relationships_for_entity(entity_info['name'], max_hops)
                    
                    if relationships:
                        kg_context_parts.append("Relationships:")
                        # Deduplicate relationships
                        seen = set()
                        for rel in relationships:
                            rel_tuple = (rel['subject'], rel['predicate'], rel['object'])
                            if rel_tuple not in seen:
                                seen.add(rel_tuple)
                                kg_context_parts.append(
                                    f"  - {rel['subject']} {rel['predicate']} {rel['object']}"
                                )
        
        if entities_found == 0:
            logger.info("No matching entities found in knowledge graph")
            return ""
        
        logger.info(f"Retrieved context for {entities_found} entities from knowledge graph")
        return "\n".join(kg_context_parts)
    
    
    def get_kg_stats(self) -> Dict:
        """Get statistics about the knowledge graph"""
        try:
            return {
                "total_entities": self.entities_collection.count_documents({}),
                "total_relationships": self.relationships_collection.count_documents({})
            }
        except Exception as e:
            logger.error(f"Error getting KG stats: {e}")
            return {"error": str(e)}