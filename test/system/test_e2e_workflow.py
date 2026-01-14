import pytest
from knowledge_graph.knowledge_graph_retriever import KnowledgeGraphRetriever
from mongomock import MongoClient

@pytest.mark.system
class TestE2EWorkflow:
    
    @pytest.fixture(autouse=True)
    def setup_system(self, sample_entity_data, sample_relationships, monkeypatch):
        mock_client = MongoClient()
        
        monkeypatch.setattr(
            "knowledge_graph.knowledge_graph_retriever.MongoClient",
            lambda uri: mock_client
        )
        
        # Create retriever
        self.retriever = KnowledgeGraphRetriever()
        
        # Populate database
        self.retriever.entities_collection.insert_many(sample_entity_data)
        self.retriever.relationships_collection.insert_many(sample_relationships)
        self.retriever._all_entity_names = self.retriever._load_all_entity_names()
        
        # Recreate strategy with populated collection
        from knowledge_graph.relationship_strategy import RelationshipStrategy
        self.retriever.relationship_strategy = RelationshipStrategy(
            self.retriever.relationships_collection
        )
    
    def test_complete_query_workflow(self, monkeypatch):
        monkeypatch.setattr(
            "knowledge_graph.knowledge_graph_retriever.extract_entities",
            lambda text: ["Germany", "Lithuania"]
        )
        
        query = "Who won Germany vs Lithuania?"
        
        # Execute full workflow
        context = self.retriever.retrieve_kg_context(query)
        
        assert isinstance(context, str)
        assert len(context) > 0
        
        # Should contain entity information
        assert "Germany" in context
        assert "Lithuania" in context
        
        # Should have structured sections
        assert "ENTITY DESCRIPTIONS" in context
    
    def test_multiple_queries_sequential(self, monkeypatch, e2e_test_queries):
        def mock_extract(text):
            for test_case in e2e_test_queries:
                if test_case["query"] == text:
                    return test_case["expected_entities"]
            return []
        
        monkeypatch.setattr(
            "knowledge_graph.knowledge_graph_retriever.extract_entities",
            mock_extract
        )
        
        results = []
        for test_case in e2e_test_queries:
            context = self.retriever.retrieve_kg_context(test_case["query"])
            results.append({
                "query": test_case["query"],
                "context": context,
                "success": len(context) > 0
            })
        
        # All queries should produce results
        assert all(r["success"] for r in results)
    
    def test_performance_metrics_e2e(self, monkeypatch):
        """Test that E2E workflow completes in reasonable time."""
        import time
        
        monkeypatch.setattr(
            "knowledge_graph.knowledge_graph_retriever.extract_entities",
            lambda text: ["Germany", "Lithuania"]
        )
        
        start_time = time.time()
        context = self.retriever.retrieve_kg_context("Who won Germany vs Lithuania?")
        elapsed_time = time.time() - start_time
        
        # Should complete in less than 5 seconds
        assert elapsed_time < 5.0
        assert len(context) > 0