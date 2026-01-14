import pytest

@pytest.mark.integration
class TestMongoDBOperations:
    
    def test_database_connection(self, test_database):
        assert test_database is not None
        assert "entities" in test_database.list_collection_names()
        assert "relationships" in test_database.list_collection_names()
    
    def test_entity_insertion_and_retrieval(self, test_database):
        # Insert a test entity
        test_entity = {"name": "Test Entity", "description": "Test description"}
        result = test_database["entities"].insert_one(test_entity)
        
        assert result.inserted_id is not None
        
        # Retrieve it
        retrieved = test_database["entities"].find_one({"name": "Test Entity"})
        assert retrieved is not None
        assert retrieved["name"] == "Test Entity"
        assert retrieved["description"] == "Test description"
    
    def test_relationship_queries(self, test_database):
        rels = list(test_database["relationships"].find({"subject": "Germany"}))
        
        assert len(rels) > 0
        assert all(r["subject"] == "Germany" for r in rels)
    
    def test_indexes_exist(self, test_database):
        entity_indexes = test_database["entities"].index_information()
        rel_indexes = test_database["relationships"].index_information()
        
        # Check entity indexes
        assert any("name" in str(idx) for idx in entity_indexes.values())
        
        # Check relationship indexes
        assert any("subject" in str(idx) for idx in rel_indexes.values())
        assert any("object" in str(idx) for idx in rel_indexes.values())
        assert any("predicate" in str(idx) for idx in rel_indexes.values())
    
    def test_count_operations(self, test_database):
        entity_count = test_database["entities"].count_documents({})
        rel_count = test_database["relationships"].count_documents({})
        
        assert entity_count > 0
        assert rel_count > 0