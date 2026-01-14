import pytest
import os
from mongomock import MongoClient as MockMongoClient
from pymongo import MongoClient as RealMongoClient
import sys
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

@pytest.fixture(scope="session")
def use_real_mongodb():
    """Determine which database to use: real MongoDB or mock."""
    return os.getenv("USE_REAL_MONGODB", "false").lower() == "true"


@pytest.fixture(scope="session")
def mongodb_uri():
    """Get MongoDB URI from environment or use default."""
    return os.getenv("MONGODB_URI", "mongodb://localhost:27017/")


@pytest.fixture(scope="function")
def mongo_client(use_real_mongodb, mongodb_uri):
    """Create MongoDB client (real or mock based on configuration)."""
    if use_real_mongodb:
        client = RealMongoClient(mongodb_uri)

        yield client

        client.drop_database("test_knowledge_graph")
    else:
        yield MockMongoClient()


@pytest.fixture
def test_database(mongo_client, sample_entity_data, sample_relationships):
    """Create test database with sample data."""
    db = mongo_client["test_knowledge_graph"]
    
    db["entities"].delete_many({})
    db["relationships"].delete_many({})

    db["entities"].insert_many(sample_entity_data)
    db["relationships"].insert_many(sample_relationships)

    db["entities"].create_index("name")
    db["relationships"].create_index("subject")
    db["relationships"].create_index("object")
    db["relationships"].create_index("predicate")
    
    yield db
    
    # Cleanup
    db["entities"].delete_many({})
    db["relationships"].delete_many({})


@pytest.fixture
def kg_retriever_integration(test_database, monkeypatch):
    """Create KG retriever with test database."""
    client = test_database.client
    
    monkeypatch.setattr(
        "knowledge_graph.knowledge_graph_retriever.MongoClient",
        lambda uri: client
    )

    monkeypatch.setattr(
        "knowledge_graph.knowledge_graph_retriever.extract_entities",
        lambda text: ["Germany", "Lithuania", "Eurobasket"]
    )
    
    from knowledge_graph.knowledge_graph_retriever import KnowledgeGraphRetriever
    
    retriever = KnowledgeGraphRetriever()
    retriever.db = test_database
    retriever.entities_collection = test_database["entities"]
    retriever.relationships_collection = test_database["relationships"]
    retriever._all_entity_names = retriever._load_all_entity_names()
    
    from knowledge_graph.relationship_strategy import RelationshipStrategy
    retriever.relationship_strategy = RelationshipStrategy(
        retriever.relationships_collection
    )
    
    return retriever