import pytest
from knowledge_graph.relationship_strategy import RelationshipStrategy

@pytest.mark.unit
class TestRelationshipStrategyUnit:
    
    def test_find_direct_relationships_exists(self, relationship_strategy):
        relationships = relationship_strategy.find_direct_relationships(
            "Germany", 
            "Lithuania"
        )
        
        assert len(relationships) >= 1
        assert any(
            r["subject"] == "Germany" and r["object"] == "Lithuania"
            for r in relationships
        )
    
    def test_find_direct_relationships_case_insensitive(self, relationship_strategy):
        """Test that entity names are case-insensitive."""
        relationships = relationship_strategy.find_direct_relationships(
            "germany", 
            "LITHUANIA"  
        )
        
        assert len(relationships) >= 1
    
    def test_find_path_direct_connection(self, relationship_strategy):
        """Test finding path when direct connection exists."""
        paths = relationship_strategy.find_path_between_entities(
            "Germany", 
            "Lithuania"
        )
        
        assert len(paths) >= 1
        # Shortest path should be direct (1 hop)
        assert len(paths[0]) == 1
    
    def test_find_path_through_intermediate(self, relationship_strategy):
        paths = relationship_strategy.find_path_between_entities(
            "Germany", 
            "Finland"
        )

        assert isinstance(paths, list)
    
    def test_find_shared_contexts(self, relationship_strategy):
        entities = ["Germany", "Lithuania"]
        shared = relationship_strategy.find_shared_contexts(entities)
        
        # Eurobasket should be a shared context
        assert "Eurobasket" in shared
        assert set(shared["Eurobasket"]) == {"Germany", "Lithuania"}
    
    def test_retrieve_relationships_single_entity(self, relationship_strategy):
        """Test strategy selection for single entity."""
        matched_entities = [("Germany", 90.0)]
        
        relationships, strategy_name = relationship_strategy.retrieve_relationships(
            matched_entities
        )
        
        assert strategy_name == "single_entity"
        assert isinstance(relationships, list)
        assert len(relationships) <= relationship_strategy.MAX_RELATIONSHIPS_PER_ENTITY
    
    def test_retrieve_relationships_multiple_entities(self, relationship_strategy):
        """Test strategy selection for multiple entities."""
        matched_entities = [("Germany", 90.0), ("Lithuania", 85.0)]
        
        relationships, strategy_name = relationship_strategy.retrieve_relationships(
            matched_entities
        )
        
        # Should use one of the multi-entity strategies
        assert strategy_name in ["path_based", "shared_context", "minimal_fallback"]
        assert isinstance(relationships, list)
    
    def test_deduplicate_relationships(self, relationship_strategy):
        """Test that duplicate relationships are removed."""
        relationships = [
            {"subject": "Germany", "predicate": "beat", "object": "Lithuania"},
            {"subject": "Germany", "predicate": "beat", "object": "Lithuania"},  # Duplicate
            {"subject": "Germany", "predicate": "plays_in", "object": "Eurobasket"},
            {"subject": "Germany", "predicate": "plays_in", "object": "Eurobasket"},  # Duplicate
        ]
        
        unique = relationship_strategy._deduplicate_relationships(relationships)
        
        assert len(unique) == 2
        # Verify exact deduplication
        seen_tuples = set()
        for r in unique:
            tuple_key = (r["subject"], r["predicate"], r["object"])
            assert tuple_key not in seen_tuples
            seen_tuples.add(tuple_key)
    