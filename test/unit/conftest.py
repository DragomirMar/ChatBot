import pytest
from mongomock import MongoClient
import sys
import os
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from knowledge_graph.relationship_strategy import RelationshipStrategy

@pytest.fixture
def mock_mongo_client():
    return MongoClient()

@pytest.fixture
def mock_relationships_collection(mock_mongo_client, sample_relationships):
    db = mock_mongo_client["knowledge_graph"]
    collection = db["relationships"]
    collection.insert_many(sample_relationships)
    return collection

@pytest.fixture
def mock_entities_collection(mock_mongo_client, sample_entity_data):
    db = mock_mongo_client["knowledge_graph"]
    collection = db["entities"]
    collection.insert_many(sample_entity_data)
    return collection

@pytest.fixture
def relationship_strategy(mock_relationships_collection):
    return RelationshipStrategy(mock_relationships_collection)

@pytest.fixture
def mock_entity_extractor(monkeypatch, sample_entities):
    def mock_extract(text):
        return sample_entities

    monkeypatch.setattr(
        "knowledge_graph.knowledge_graph_retriever.extract_entities",
        mock_extract
    )
    return mock_extract

@pytest.fixture
def kg_retriever_with_mocks(mock_mongo_client, mock_entity_extractor, monkeypatch):
    monkeypatch.setattr(
        "knowledge_graph.knowledge_graph_retriever.MongoClient",
        lambda uri: mock_mongo_client
    )

    from knowledge_graph.knowledge_graph_retriever import KnowledgeGraphRetriever
    
    retriever = KnowledgeGraphRetriever()

    retriever.entities_collection.insert_many([
        {"name": "Germany", "description": "National basketball team"},
        {"name": "Lithuania", "description": "National basketball team"},
        {"name": "Eurobasket", "description": "Basketball championship"},
    ])
    
    retriever.relationships_collection.insert_many([
        {"subject": "Germany", "predicate": "beat", "object": "Lithuania"},
        {"subject": "Germany", "predicate": "plays_in", "object": "Eurobasket"},
        {"subject": "Lithuania", "predicate": "plays_in", "object": "Eurobasket"},
    ])

    retriever._all_entity_names = retriever._load_all_entity_names()
    
    return retriever