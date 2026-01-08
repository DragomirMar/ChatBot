import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from entity_extractor import EntityExtractor
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

ee = EntityExtractor()

text_phrase = "Barack Obama was the 44th President of the United States."
next_phrase = "IBM are working on developing AI technologies in New York City for the new iPhone."
text = "Check out example.com and api.github.com"


if __name__ == "__main__":
    print("="*80)
    print("ENTITY EXTRACTION TEST")
    print("="*80)

    entities = ee.extract_entities(text_phrase)
    entities1 = ee.extract_entities(next_phrase)
    next_entities = ee.extract_entities(text)

    print(f"\n Extracted Entities from phrase (included singles):\n'{text_phrase}'\n")
    for ent in entities:
        print(f"- {ent}")
        
    print(f"\n Extracted Entities from phrase (not included singles):\n'{next_phrase}'\n")
    for ent in entities1:
        print(f"- {ent}")
        
    print(f"\n Extracted Entities from phrase (included singles):\n'{text}'\n")
    for ent in next_entities:
        print(f"- {ent}")
        