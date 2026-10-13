import math
from datetime import datetime, timezone

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from engagement_intelligence.service.app import create_app
from engagement_intelligence.service.settings import ServiceSettings

AS_OF = "2025-12-01T00:00:00Z"
LATER = "2025-12-08T00:00:00Z"
AS_OF_DT = datetime(2025, 12, 1, tzinfo=timezone.utc)


@pytest.fixture
def service(tmp_path, small_bundle, small_pipeline):
    bundle, _ = small_bundle
    events = tmp_path / "events.parquet"
    pq.write_table(small_pipeline.observable, events)
    settings = ServiceSettings(bundle_dir=bundle.save(tmp_path / "bundle"), events_path=events,
                               store_path=tmp_path / "store.jsonl", trace_path=tmp_path / "trace.jsonl",
                               default_as_of=AS_OF_DT)
    app = create_app(settings)
    return TestClient(app), app


def _first_scored(client) -> dict:
    body = client.post("/assess", json={"as_of": AS_OF, "persist": False}).json()
    return next(a for a in body["assessments"] if a["status"] == "scored")


def test_health_and_model_info(service, small_bundle):
    client, _ = service
    bundle = small_bundle[0]
    assert client.get("/health").json() == {"status": "ok", "model_version": bundle.version}
    info = client.get("/model-info").json()
    assert [f["name"] for f in info["features"]] == bundle.feature_names
    assert info["policy_dependent_features_used"] == []


def test_batch_assessment(service, small_bundle):
    client, _ = service
    response = client.post("/assess", json={"as_of": AS_OF})
    assert response.status_code == 200
    body = response.json()
    scored = [a for a in body["assessments"] if a["status"] == "scored"]
    summary = body["summary"]
    assert summary["scored"] == len(scored) > 0
    capacity = small_bundle[0].operating_point.review_capacity
    assert summary["within_capacity"] == math.ceil(capacity * len(scored))
    assert sum(summary["band_shares"].values()) == pytest.approx(1.0)
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_single_assessment_matches_batch_and_does_not_persist(service):
    client, _ = service
    first = _first_scored(client)
    single = client.get(f"/accounts/{first['account_id']}/assessment", params={"as_of": AS_OF}).json()
    assert single["risk_score"] == pytest.approx(first["risk_score"])
    assert single["within_review_capacity"] is None
    assert client.get(f"/accounts/{first['account_id']}/risk-history").json() == []


def test_risk_history_accumulates_and_tracks_transitions(service):
    client, _ = service
    client.post("/assess", json={"as_of": AS_OF})
    second = client.post("/assess", json={"as_of": LATER}).json()
    account = next(a["account_id"] for a in second["assessments"]
                   if a["status"] == "scored" and a["risk_change"]["transition"] != "new")
    history = client.get(f"/accounts/{account}/risk-history").json()
    assert len(history) == 2
    assert history[1]["risk_change"]["previous_assessed_at"] is not None


def test_recent_events_endpoint(service):
    client, _ = service
    account = _first_scored(client)["account_id"]
    events = client.get(f"/accounts/{account}/events", params={"as_of": AS_OF, "limit": 5}).json()
    assert 0 < len(events) <= 5
    assert all(datetime.fromisoformat(e["recorded_at"]) <= AS_OF_DT for e in events)


def test_unknown_account_is_404(service):
    client, _ = service
    assert client.get("/accounts/acc_999999/assessment", params={"as_of": AS_OF}).status_code == 404


def test_naive_as_of_rejected(service):
    client, _ = service
    response = client.get("/accounts/acc_000001/assessment", params={"as_of": "2025-12-01T00:00:00"})
    assert response.status_code == 422


def test_every_request_is_traced(service):
    client, app = service
    response = client.post("/assess", json={"as_of": AS_OF})
    client.get("/accounts/acc_999999/assessment", params={"as_of": AS_OF})
    records = app.state.services.tracer.read_all()
    assess_record = next(r for r in records if r["request_id"] == response.headers["X-Request-ID"])
    assert assess_record["route"] == "/assess" and assess_record["status_code"] == 200
    assert assess_record["scored"] > 0 and "band_shares" in assess_record and assess_record["latency_ms"] > 0
    assert any(r["status_code"] == 404 and r["route"] == "/accounts/{account_id}/assessment"
               and r["path_params"] == {"account_id": "acc_999999"} for r in records)