import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (with database)"
    )
    config.addinivalue_line(
        "markers", "system: System/E2E tests (full workflow)"
    )

@pytest.fixture
def sample_query():
    return "Who won the game between Germany and Lithuania in EuroBasket 2025?"

@pytest.fixture
def sample_entities():
    return ["Germany", "Lithuania", "Eurobasket"]

@pytest.fixture
def sample_relationships():
    return [
        {"subject": "Germany", "predicate": "beat", "object": "Lithuania"},
        {"subject": "Germany", "predicate": "scored", "object": "107 Points"},
        {"subject": "Lithuania", "predicate": "scored", "object": "88 Points"},
        {"subject": "Germany", "predicate": "plays_in", "object": "Eurobasket"},
        {"subject": "Lithuania", "predicate": "plays_in", "object": "Eurobasket"},
        {"subject": "Eurobasket", "predicate": "hosted_in", "object": "Finland"},
    ]

@pytest.fixture
def sample_entity_data():
    return [
        {
            "name": "Germany",
            "description": "National basketball team of Germany that competed in EuroBasket 2025"
        },
        {
            "name": "Lithuania",
            "description": "National basketball team of Lithuania that competed in EuroBasket 2025"
        },
        {
            "name": "Eurobasket",
            "description": "European basketball championship tournament"
        },
        {
            "name": "Finland",
            "description": "Country in Northern Europe that hosted EuroBasket 2025"
        },
    ]