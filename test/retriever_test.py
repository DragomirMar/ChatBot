import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_graph_retriever import KnowledgeGraphRetriever

# Initialize
kg = KnowledgeGraphRetriever()

# Test 1: Extract entities
query = "How did the game between Germany and Lithuania ended?"
entities = kg.extract_potential_entities(query)
print(f"Extracted: {entities}")

db_entities = kg._load_all_entity_names()

# Test 2: Fuzzy matching
match = kg.fuzzy_match_entities("Lithuania", db_entities)
print(f"Fuzzy match new: {match}")

# Test 3: Complete pipeline
context = kg.retrieve_kg_context(query)
# print(f"Linked entities: {linked}")
print(f"Context length: {len(context)} chars")

# Test 4: Check stats
stats = kg.get_kg_stats()
print(f"Stats: {stats}")