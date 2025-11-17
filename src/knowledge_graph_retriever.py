from pymongo import MongoClient
from typing import List, Dict, Set, Tuple
import logging
from entity_extractor import EntityExtractor

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
        Extract potential entity names from text.
        Simple approach: capitalized words and phrases.
        You can enhance this with NER later.
        """
        extractor = EntityExtractor()
        potential_entities = extractor.extract_entities(
            text, min_token_len=2, include_singles=True, use_simple_fallback=True
        )
        return potential_entities
        
    
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
            max_hops: Number of relationship hops to traverse (1 or 2 recommended)
        
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
    
    def retrieve_kg_context(self, query: str, max_entities: int = 5, max_hops: int = 1) -> str:
        """
        Main method to retrieve knowledge graph context for a query.
        
        Args:
            query: User's question
            max_entities: Maximum number of entities to retrieve
            max_hops: Number of relationship hops (1-2 recommended)
        
        Returns:
            Formatted string with entity descriptions and relationships
        """
        # Extract potential entities from query
        potential_entities = self.extract_potential_entities(query)
        
        if not potential_entities:
            logger.info("No potential entities found in query")
            return ""
        
        logger.info(f"Found potential entities: {potential_entities}")
        
        # Retrieve entity information and relationships
        kg_context_parts = []
        entities_found = 0
        
        for entity_name in potential_entities[:max_entities]:
            # Get entity description
            entity_info = self.get_entity_info(entity_name)
            
            if entity_info:
                entities_found += 1
                kg_context_parts.append(f"\n**Entity: {entity_info['name']}**")
                kg_context_parts.append(f"Description: {entity_info['description']}")
                
                # Get relationships
                relationships = self.get_relationships_for_entity(entity_name, max_hops)
                
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