from fastapi.testclient import TestClient

from app.main import create_app

DOCUMENT = {"text": "Asana costs more than Trello.", "source": "manual-1", "brand": "Asana"}
QUESTION = {"question": "How does Asana pricing compare to Trello?"}


def test_tenant_cannot_see_another_tenants_data(store, mock_llm):
    client = TestClient(create_app(store))
    assert client.post("/tenants/acme/ingest", json={"documents": [DOCUMENT]}).status_code == 200

    acme = client.post("/tenants/acme/ask", json=QUESTION).json()
    globex = client.post("/tenants/globex/ask", json=QUESTION).json()

    assert acme["found_in_corpus"] is True
    assert acme["citations"][0]["source"] == "manual-1"

    assert globex["found_in_corpus"] is False
    assert globex["citations"] == []
    assert "Asana" not in globex["answer"]
