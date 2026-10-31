"""Smoke tests for JagX AI multi-agent core (no live LLM calls)."""
from jagx_api.agent_router import select_agents
from jagx_api.agents import list_capabilities


def test_capabilities():
    caps = list_capabilities()
    assert caps["version"].startswith("10.")
    assert caps["created_by"] == "JagX & JRILICENSE"
    ids = {a["id"] for a in caps["agents"]}
    assert {"engineer", "researcher", "architect", "creator", "africa", "media"} <= ids


def test_router_coding():
    agents = select_agents("Write a FastAPI endpoint that returns health status")
    assert "engineer" in agents


def test_router_research():
    agents = select_agents("What is the history of mobile money in Kenya?")
    assert "researcher" in agents or "africa" in agents


def test_router_africa():
    agents = select_agents("How do I accept Naira payments in Lagos with low fees?")
    assert "africa" in agents


def test_router_media():
    agents = select_agents("Generate an image of a Lagos skyline at sunset")
    assert "media" in agents


def test_router_default():
    agents = select_agents("Hello, how are you?")
    assert len(agents) >= 1
