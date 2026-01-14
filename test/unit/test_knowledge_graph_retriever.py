import pytest
from knowledge_graph.knowledge_graph_retriever import KnowledgeGraphRetriever

@pytest.mark.unit
class TestKnowledgeGraphRetrieverUnit:
    
    def test_extract_potential_entities(self, kg_retriever_with_mocks, sample_entities):
        entities = kg_retriever_with_mocks.extract_potential_entities("test query")
        
        assert entities == sample_entities
    
    def test_fuzzy_match_exact(self, kg_retriever_with_mocks):
        """Test fuzzy matching with exact match."""
        matches = kg_retriever_with_mocks.fuzzy_match_entities("Germany")
        
        assert len(matches) > 0
        assert matches[0][0] == "Germany"
        assert matches[0][1] == 100.0
    
    def test_fuzzy_match_partial(self, kg_retriever_with_mocks):
        """Test fuzzy matching with typo."""
        matches = kg_retriever_with_mocks.fuzzy_match_entities("Germny")
        
        assert len(matches) > 0
        assert any("Germany" in match[0] for match in matches)
    
    def test_link_entities(self, kg_retriever_with_mocks):
        extracted = ["Germany", "Lithuania"]
        linked = kg_retriever_with_mocks.link_entities(extracted)
        
        assert isinstance(linked, dict)
        assert "Germany" in linked
        assert "Lithuania" in linked
        assert all(isinstance(matches, list) for matches in linked.values())
    
    def test_limit_matches_deduplicates(self, kg_retriever_with_mocks):
        """Test that limit_matches removes duplicates."""
        linking_results = {
            "germany": [("Germany", 95.0), ("Germany Team", 80.0)],
            "deutschland": [("Germany", 90.0), ("Germany Team", 75.0)],  # Duplicates
        }
        
        limited = kg_retriever_with_mocks._limit_matches(linking_results)
        
        entity_names = [name for name, _ in limited]
        assert len(entity_names) == len(set(entity_names))
        assert entity_names.count("Germany") == 1
        assert entity_names.count("Germany Team") == 1
    
    def test_get_entity_info(self, kg_retriever_with_mocks):
        """Test retrieving existing entity."""
        entity_info = kg_retriever_with_mocks.get_entity_info("Germany")
        
        assert entity_info is not None
        assert entity_info["name"] == "Germany"
        assert "description" in entity_info

    def test_get_entity_info_case_insensitive(self, kg_retriever_with_mocks):
        """Test that entity lookup is case-insensitive."""
        entity_info = kg_retriever_with_mocks.get_entity_info("germany")
        
        assert entity_info is not None
        assert entity_info["name"] == "Germany"
    
    def test_build_kg_context(self, kg_retriever_with_mocks):
        matched_entities = [("Germany", 90.0), ("Lithuania", 85.0)]
        
        context = kg_retriever_with_mocks._build_kg_context(matched_entities)
        
        assert isinstance(context, str)
        assert len(context) > 0
        # Should contain section headers
        assert "ENTITY DESCRIPTIONS" in context