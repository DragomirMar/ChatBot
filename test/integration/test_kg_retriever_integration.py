import pytest

@pytest.mark.integration
class TestKGRetrieverIntegration:
    
    def test_load_entity_names_from_db(self, kg_retriever_integration):
        entity_names = kg_retriever_integration._all_entity_names
        
        assert len(entity_names) > 0
        assert "Germany" in entity_names
        assert "Lithuania" in entity_names
    
    def test_fuzzy_matching_against_db(self, kg_retriever_integration):
        matches = kg_retriever_integration.fuzzy_match_entities("Germany")
        
        assert len(matches) > 0
        assert matches[0][0] == "Germany"
        assert matches[0][1] == 100.0
    
    def test_entity_info_retrieval_from_db(self, kg_retriever_integration):
        entity_info = kg_retriever_integration.get_entity_info("Germany")
        
        assert entity_info is not None
        assert entity_info["name"] == "Germany"
        assert "description" in entity_info
        assert len(entity_info["description"]) > 0
    
    def test_relationship_strategy_with_db(self, kg_retriever_integration):
        matched_entities = [("Germany", 90.0), ("Lithuania", 85.0)]
        
        relationships, strategy = kg_retriever_integration.relationship_strategy.retrieve_relationships(
            matched_entities
        )
        
        assert isinstance(relationships, list)
        assert isinstance(strategy, str)

        assert len(relationships) > 0
    
    def test_build_context_with_db_data(self, kg_retriever_integration):
        matched_entities = [("Germany", 90.0), ("Lithuania", 85.0)]
        
        context = kg_retriever_integration._build_kg_context(matched_entities)
        
        assert isinstance(context, str)
        assert len(context) > 0
        assert "Germany" in context
        assert "Lithuania" in context
        assert "ENTITY DESCRIPTIONS" in context
    
    def test_full_retrieval_with_db(self, kg_retriever_integration):
        query = "Who won Germany vs Lithuania?"
        
        context = kg_retriever_integration.retrieve_kg_context(query)
        
        assert isinstance(context, str)
        assert len(context) > 0
        # Should contain entity information
        assert "Germany" in context or "Lithuania" in context
    
    def test_stats_from_db(self, kg_retriever_integration):
        stats = kg_retriever_integration.get_kg_stats()
        
        assert "entities" in stats
        assert "relationships" in stats
        assert stats["entities"] > 0
        assert stats["relationships"] > 0