import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_graph.knowledge_graph_retriever import KnowledgeGraphRetriever
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)



# Test variable
test_users_query = "How did the game between Germany and Lithuania ended?"

kg = KnowledgeGraphRetriever()

if __name__ == "__main__":
    print("="*80)
    print("KG RETRIEVAL TEST")
    print("="*80)

    print(f"\n 1. Query: {test_users_query}\n")
    
    results = kg.retrieve_kg_context(
        test_users_query
    )
    