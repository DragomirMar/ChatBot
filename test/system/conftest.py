import pytest
import os
import sys
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

@pytest.fixture(scope="session")
def env():
    # Save current environment
    current_env = os.environ.copy()
    
    # Set test environment
    os.environ["MONGODB_URI"] = os.getenv(
        "TEST_MONGODB_URI", 
        "mongodb://localhost:27017/"
    )
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(current_env)

@pytest.fixture
def e2e_test_queries():
    return [
        {
            "query": "Who won the game between Germany and Lithuania?",
            "expected_entities": ["Germany", "Lithuania"],
            "expected_keywords": ["beat", "won", "game"]
        },
        {
            "query": "Which teams participated in EuroBasket 2025?",
            "expected_entities": ["Eurobasket"],
            "expected_keywords": ["teams", "participated"]
        },
        {
            "query": "Where was the tournament hosted?",
            "expected_entities": ["Finland", "Eurobasket"],
            "expected_keywords": ["hosted", "tournament"]
        }
    ]