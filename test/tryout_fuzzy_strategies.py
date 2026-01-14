from difflib import SequenceMatcher, get_close_matches
from rapidfuzz import fuzz, process
from typing import List, Tuple

NAMES: List[str] = [  
    "John Smith", "Jon Smyth", "Jane Smith", "John Smithson",
    "Bob Johnson", "Robert Johnson", "Alice Williams", "Alicia Williams",
    "Michael Brown", "Mike Brown", "Christopher Davis", "Lithuania",
    "Germany", "Against Lithiuania", "Germany vs Lithiuania",
    "Lithuania National Team", "Lithuania Basketball Team","Lithuanians Giants"]

# Calculate similarity
def simple_ratio(s1: str, s2: str) -> float:
    """Calculate similarity between two strings using SequenceMatcher."""
    score = SequenceMatcher(None, s1, s2).ratio() 
    return score

def fuzzy_match(query: str, names_list: List[str], threshold: float = 0.8) -> List[str]:
    """ Fuzzy match a query string against a list of names. """
    matches = []
    for name in names_list:
        score = simple_ratio(query, name)
        if score >= threshold:
            matches.append((name, score))
    
    return matches

def fuzzy_match_with_get_close_matches(query: str, names_list: List[str], n: int = 5, cutoff: float = 0.6) -> List[str]:
    """ Using difflib's get_close_matches - simpler but less control """
    return get_close_matches(query, names_list, n=n, cutoff=cutoff)

def fuzzy_match_with_partial(query: str, names_list: List[str], threshold: float = 0.6) -> List[Tuple[str, float]]:
    """ Fuzzy matching that also considers partial matches. """
    matches = []
    query_lower = query.lower()
    
    for name in names_list:
        name_lower = name.lower()
        
        # Full string similarity
        full_score = simple_ratio(query, name)
        
        # Check if query is contained in name (substring match)
        substring_bonus = 0.2 if query_lower in name_lower else 0
        
        # Check if any word in query matches any word in name
        query_words = query_lower.split()
        name_words = name_lower.split()
        word_matches = sum(1 for qw in query_words for nw in name_words if simple_ratio(qw, nw) > 0.8)
        word_bonus = 0.1 * word_matches
        
        # Combined score
        final_score = min(full_score + substring_bonus + word_bonus, 1.0)
        
        if final_score >= threshold:
            matches.append((name, final_score))
    
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

def fuzzy_match_simple_ratio(query: str, names_list: List[str], threshold: int = 60, limit: int = 10) -> List[Tuple[str, float]]:
    """ Simple ratio: Basic similarity comparison. """
    matches = process.extract(
        query,
        names_list,
        scorer=fuzz.ratio,
        score_cutoff=threshold,
        limit=limit
    )
    return [(match[0], match[1]) for match in matches]

def fuzzy_match_partial_ratio(query: str, names_list: List[str], threshold: int = 60, limit: int = 10) -> List[Tuple[str, float]]:
    """ Partial ratio: Matches substrings."""
    matches = process.extract(
        query,
        names_list,
        scorer=fuzz.partial_ratio,
        score_cutoff=threshold,
        limit=limit
    )
    return [(match[0], match[1]) for match in matches]

def fuzzy_match_weighted(query: str, names_list: List[str], threshold: int = 60, limit: int = 10) -> List[Tuple[str, float]]:
    """ Weighted ratio: Automatically chooses best algorithm. """
    matches = process.extract(
        query,
        names_list,
        scorer=fuzz.WRatio,
        score_cutoff=threshold,
        limit=limit
    )
    return [(match[0], match[1]) for match in matches]


if __name__ == "__main__":
    
    query = "lithuania"
    threshold = 0.7
    
    print("=" * 80)
    print("FUZZY MATCHING DEMONSTRATION")
    print("=" * 80)
    
    print("\n1. Basic Fuzzy Matching:")
    matched_names = fuzzy_match(query, NAMES, threshold)
    print(f"Matches for '{query}' (threshold={threshold}):")
    for name, score in matched_names:
        print(f" - {name} (score: {score:.2f})")
        
    print("\n2. Fuzzy Matching with get_close_matches:")
    matched_names_gc = fuzzy_match_with_get_close_matches(query, NAMES, n=5, cutoff=threshold)
    print(f"Matches for '{query}' (cutoff={threshold}):")
    for name in matched_names_gc:
        print(f" - {name}")
        
    print("\n3. Partial Matching with Word Bonuses (threshold=0.6):")
    matches = fuzzy_match_with_partial(query, NAMES, threshold=0.6)
    if matches:
        for name, score in matches:
            print(f"   {name:30} | Score: {score:.3f}")
            
    print("\n4. Rapiduzz Simple Ratio (direct character comparison):")
    matches = fuzzy_match_simple_ratio(query, NAMES, threshold=60)
    if matches:
        for name, score in matches[:5]:
            print(f"   {name:30} | Score: {score:.1f}")
            
    print("\n2. Partial Ratio (substring matching):")
    matches = fuzzy_match_partial_ratio(query, NAMES, threshold=60)
    if matches:
        for name, score in matches[:5]:
            print(f"   {name:30} | Score: {score:.1f}")
            
    print("\n5. Weighted Ratio (smart auto-selection) *** RECOMMENDED ***:")
    matches = fuzzy_match_weighted(query, NAMES, threshold=60)
    if matches:
        for name, score in matches[:5]:
            print(f"   {name:30} | Score: {score:.1f}")