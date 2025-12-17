# tests\test_api.py

def test_reset_endpoint(client):
    resp = client.get("/reset")

    assert resp.status_code == 200

    data = resp.json()

    assert "state" in data
    assert "info" in data
    assert isinstance(data["state"], list)
    assert isinstance(data["info"], dict)


def test_step_endpoint(client):
    client.get("/reset")

    resp = client.post("/step", json={"action": "take"})

    assert resp.status_code == 200

    data = resp.json()

    assert "state" in data
    assert "reward" in data
    assert "done" in data
    assert "truncated" in data
    assert "info" in data


def test_legal_actions_endpoint(client):
    client.get("/reset")
    resp = client.get("/legal_actions")

    assert resp.status_code == 200
    data = resp.json()

    assert "legal_actions" in data
    assert isinstance(data["legal_actions"], list)
