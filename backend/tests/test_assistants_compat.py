"""Contract tests for the LangGraph-compatible assistants endpoints."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.gateway.routers import assistants_compat


def _assistant(name: str) -> assistants_compat.AssistantResponse:
    return assistants_compat.AssistantResponse(
        assistant_id=name,
        graph_id="lead_agent",
        name=name,
    )


def _assistant_with_meta(
    name: str,
    created_by: str,
    *,
    graph_id: str = "lead_agent",
    extra: dict[str, object] | None = None,
) -> assistants_compat.AssistantResponse:
    metadata: dict[str, object] = {"created_by": created_by}
    if extra:
        metadata.update(extra)
    return assistants_compat.AssistantResponse(
        assistant_id=name,
        graph_id=graph_id,
        name=name,
        metadata=metadata,
    )


def _make_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    app = FastAPI()
    app.include_router(assistants_compat.router)
    monkeypatch.setattr(
        assistants_compat,
        "_list_assistants",
        lambda: [_assistant("lead_agent"), _assistant("researcher"), _assistant("writer")],
    )
    return app


@pytest.mark.parametrize(
    "body",
    [
        {"limit": 0},
        {"limit": -1},
        {"limit": 1001},
        {"offset": -1},
    ],
)
def test_search_rejects_invalid_pagination(monkeypatch: pytest.MonkeyPatch, body: dict[str, int]) -> None:
    with TestClient(_make_app(monkeypatch)) as client:
        response = client.post("/api/assistants/search", json=body)

    assert response.status_code == 422


def test_search_accepts_contract_maximum_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(_make_app(monkeypatch)) as client:
        response = client.post("/api/assistants/search", json={"limit": 1000})

    assert response.status_code == 200
    assert [assistant["assistant_id"] for assistant in response.json()] == ["lead_agent", "researcher", "writer"]


def test_search_preserves_default_and_positive_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(_make_app(monkeypatch)) as client:
        default_response = client.post("/api/assistants/search")
        offset_response = client.post("/api/assistants/search", json={"offset": 1, "limit": 1})

    assert default_response.status_code == 200
    assert [assistant["assistant_id"] for assistant in default_response.json()] == ["lead_agent", "researcher", "writer"]
    assert offset_response.status_code == 200
    assert [assistant["assistant_id"] for assistant in offset_response.json()] == ["researcher"]


def _metadata_catalog() -> list[assistants_compat.AssistantResponse]:
    return [
        _assistant_with_meta("lead_agent", "system"),
        _assistant_with_meta("researcher", "user", extra={"role": "research"}),
        # Same graph and name substring as researcher, different metadata, so a
        # graph_id/name filter alone cannot explain a metadata-only result.
        _assistant_with_meta("research_bot", "system"),
        _assistant_with_meta("writer", "user", graph_id="writer_graph", extra={"role": "draft"}),
    ]


def _metadata_search_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    app = _make_app(monkeypatch)
    monkeypatch.setattr(assistants_compat, "_list_assistants", _metadata_catalog)
    return TestClient(app)


def test_search_filters_by_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    with _metadata_search_client(monkeypatch) as client:
        response = client.post("/api/assistants/search", json={"metadata": {"created_by": "user"}})

    assert response.status_code == 200
    assert [assistant["assistant_id"] for assistant in response.json()] == ["researcher", "writer"]


def test_search_metadata_match_nothing_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    with _metadata_search_client(monkeypatch) as client:
        response = client.post("/api/assistants/search", json={"metadata": {"created_by": "nobody"}})

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.parametrize("body", [None, {}, {"metadata": None}, {"metadata": {}}])
def test_search_omitted_or_empty_metadata_returns_all(monkeypatch: pytest.MonkeyPatch, body: dict[str, object] | None) -> None:
    with _metadata_search_client(monkeypatch) as client:
        response = client.post("/api/assistants/search", json=body)

    assert response.status_code == 200
    assert [assistant["assistant_id"] for assistant in response.json()] == ["lead_agent", "researcher", "research_bot", "writer"]


def test_search_metadata_requires_every_key(monkeypatch: pytest.MonkeyPatch) -> None:
    with _metadata_search_client(monkeypatch) as client:
        response = client.post(
            "/api/assistants/search",
            json={"metadata": {"created_by": "user", "role": "research"}},
        )

    assert response.status_code == 200
    assert [assistant["assistant_id"] for assistant in response.json()] == ["researcher"]


def test_search_metadata_combines_with_graph_id_and_name(monkeypatch: pytest.MonkeyPatch) -> None:
    with _metadata_search_client(monkeypatch) as client:
        response = client.post(
            "/api/assistants/search",
            json={"metadata": {"created_by": "user"}, "graph_id": "lead_agent", "name": "research"},
        )

    assert response.status_code == 200
    assert [assistant["assistant_id"] for assistant in response.json()] == ["researcher"]
