from typing import List, Dict, Tuple, Set
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class RelationshipStrategy:
    """Handles strategic relationship retrieval from knowledge graph."""
 
    MAX_HOPS = 2
    MAX_TOTAL_RELATIONSHIPS = 30
    MAX_RELATIONSHIPS_PER_ENTITY = 7
    
    def __init__(self, relationships_collection):
        self.relationships_collection = relationships_collection
    
    def retrieve_relationships(self, matched_entities: List[Tuple[str, float]]) -> Tuple[List[Dict], str]:
        """ Retrieves relationships using tiered strategy."""
        entities = [e for e, _ in matched_entities]
        relationships = []

        # 1. If Single entity → minimal edges
        if len(entities) == 1:
            entity_titled = entities[0].title()
            rels = list(self.relationships_collection.find({
                "$or": [{"subject": entity_titled}, {"object": entity_titled}]
            }).limit(self.MAX_RELATIONSHIPS_PER_ENTITY))
            return rels, "single_entity"

        # 2. If Multiple entities: connecting paths
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                for path in self.find_path_between_entities(entities[i], entities[j]):
                    relationships.extend(path)

        relationships = self._deduplicate_relationships(relationships)
        if relationships:
            return relationships[:self.MAX_TOTAL_RELATIONSHIPS], "path_based"

        # 3. Fallback: shared contexts
        shared = self.find_shared_contexts(entities)
        for ctx, connected in shared.items():
            for e in connected:
                relationships.extend(self.find_direct_relationships(e, ctx))

        relationships = self._deduplicate_relationships(relationships)
        if relationships:
            return relationships[:self.MAX_TOTAL_RELATIONSHIPS], "shared_context"

        # 4.Second Fallback (if no connections found): get relationships for each entity
        for e in entities:
            e_titled = e.title()
            relationships.extend(
                list(self.relationships_collection.find({
                    "$or": [{"subject": e_titled}, {"object": e_titled}]
                }).limit(3)) # Top 3 relationships per entity
            )

        return self._deduplicate_relationships(relationships), "minimal_fallback"
    
    def find_direct_relationships(self, entity1: str, entity2: str) -> List[Dict]:
        """Find direct relationships between two entities."""
        entity1 = entity1.title()
        entity2 = entity2.title()
        return list(self.relationships_collection.find({
            "$or": [
                {"subject": entity1, "object": entity2},
                {"subject": entity2, "object": entity1}
            ]
        }))

    def find_path_between_entities(self, entity1: str, entity2: str) -> List[List[Dict]]:
        """ Find paths between two entities using BFS."""
        entity1 = entity1.title()
        entity2 = entity2.title()
        
        if entity1 == entity2:
            return []  
        
        # BFS to find paths
        queue = deque([(entity1, [], {entity1})])  # (current_entity, path_so_far)
        paths = []

        while queue:
            current, path, visited = queue.popleft()
            if len(path) >= self.MAX_HOPS:
                continue

            rels = list(self.relationships_collection.find({
                "$or": [{"subject": current}, {"object": current}]
            }).limit(15))

            for rel in rels:
                nxt = rel["object"] if rel["subject"] == current else rel["subject"]
                if nxt in visited:
                    continue

                new_path = path + [rel]

                if nxt == entity2:
                    paths.append(new_path)
                else:
                    queue.append((nxt, new_path, visited | {nxt}))

        paths.sort(key=lambda p: (len(p), len(set(r["predicate"] for r in p))))
        return paths[:3]

    def find_shared_contexts(self, entities: List[str]) -> Dict[str, List[str]]:
        """ Find entities that connect to multiple query entities contextually."""
        entity_titles = [e.strip().title() for e in entities]
            
        # Track which entities each node connects to
        connections = defaultdict(set)
        
        for entity in entity_titles:
            # Get neighbors within max_hops
            neighbors = self._get_neighborhood(entity)
            
            for neighbor in neighbors:
                connections[neighbor].add(entity)
        
        # Filter to only entities that connect to 2+ query entities
        shared_contexts = {
            context: list(connected_entities) 
            for context, connected_entities in connections.items()
            if len(connected_entities) >= 2 and context not in entity_titles
        }
        
        if shared_contexts:
            logger.info(f"Found {len(shared_contexts)} shared context entities")
        
        return shared_contexts

    def _get_neighborhood(self, entity: str) -> Set[str]:
        """Get all entities within N hops of given entity."""
        visited = {entity}
        frontier = {entity}

        for _ in range(self.MAX_HOPS):
            nxt = set()
            for node in frontier:
                rels = list(self.relationships_collection.find({
                    "$or": [{"subject": node}, {"object": node}]
                }).limit(15))
                for r in rels:
                    neighbor = r["object"] if r["subject"] == node else r["subject"]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        nxt.add(neighbor)
            frontier = nxt
        
        return visited
    
    @staticmethod
    def _deduplicate_relationships(relationships: List[Dict]) -> List[Dict]:
        """Removes duplicate relationships."""
        seen = set()
        unique = []
        for r in relationships:
            key = (r["subject"], r["predicate"], r["object"])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique