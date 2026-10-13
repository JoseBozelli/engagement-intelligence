"""FastAPI service exposing the scoring engine (D-045).

Run on the reference data:
  EI_BUNDLE_DIR=artifacts/models/<version> \
  EI_EVENTS_PATH=data/reference/observable/events.parquet \
  EI_STORE_PATH=data/reference/scoring/assessments.jsonl \
  EI_TRACE_PATH=data/reference/scoring/traces.jsonl \
  EI_DEFAULT_AS_OF=2026-07-01T00:00:00+00:00 \
  uv run uvicorn engagement_intelligence.service.app:create_app_from_env --factory
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from engagement_intelligence.core.envelope import EventEnvelope
from engagement_intelligence.core.tracing import JSONLTracer, new_run_id
from engagement_intelligence.models.bundle import ModelBundle
from engagement_intelligence.scoring.engine import Scorer, UnknownAccountError
from engagement_intelligence.scoring.schemas import EngagementAssessment
from engagement_intelligence.scoring.sources import TableEventSource
from engagement_intelligence.scoring.store import JsonlAssessmentStore
from engagement_intelligence.service.observability import LockedTracer, batch_summary
from engagement_intelligence.service.schemas import (
    AssessRequest,
    AssessResponse,
    BatchSummary,
    FeatureInfo,
    ModelInfo,
)
from engagement_intelligence.service.settings import ServiceSettings

_TARGET_FIELDS = ("name", "engagement_role", "horizon_days", "eligibility_window_days",
                  "label_settlement_days", "assessment_weekday")


@dataclass
class Services:
    bundle: ModelBundle
    scorer: Scorer
    tracer: LockedTracer
    settings: ServiceSettings
    scoring_lock: threading.Lock


def model_info_from(bundle: ModelBundle) -> ModelInfo:
    m = bundle.manifest
    descriptions = {f["name"]: f["description"] for f in m["feature_spec"]["features"]}
    return ModelInfo(
        model_version=bundle.version,
        model=m["model"],
        features=[FeatureInfo(name=n, description=descriptions[n]) for n in bundle.feature_names],
        policy_dependent_features_used=m["policy_dependent_features_used"],
        target={k: m["target"][k] for k in _TARGET_FIELDS},
        calibration=m["calibration"],
        operating_point=m["operating_point"],
        training=m["training"],
        provenance=m["provenance"],
        validation_summary=m["validation_summary"],
    )


def create_app(settings: ServiceSettings) -> FastAPI:
    bundle = ModelBundle.load(settings.bundle_dir)
    scorer = Scorer(bundle, TableEventSource.from_parquet(settings.events_path),
                    JsonlAssessmentStore(settings.store_path))
    services = Services(bundle, scorer, LockedTracer(JSONLTracer(settings.trace_path)),
                        settings, threading.Lock())
    app = FastAPI(title="Engagement Intelligence", version=bundle.version,
                  description="Engagement-risk assessments from longitudinal event data.")
    app.state.services = services

    def resolve_as_of(value: datetime | None) -> datetime:
        as_of = value or settings.default_as_of or datetime.now(timezone.utc)
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise HTTPException(status_code=422, detail="as_of must be timezone-aware, e.g. 2026-07-01T00:00:00Z")
        return as_of.astimezone(timezone.utc)

    @app.middleware("http")
    async def trace_requests(request: Request, call_next):
        request_id = new_run_id()
        request.state.request_id = request_id
        request.state.trace = {}
        start = time.perf_counter()
        status, error = 500, None
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            route = request.scope.get("route")
            services.tracer.log({
                "request_id": request_id,
                "method": request.method,
                "route": getattr(route, "path", request.url.path),
                "path_params": dict(request.scope.get("path_params", {})),
                "query": dict(request.query_params),
                "model_version": bundle.version,
                "status_code": status,
                "latency_ms": round(1000 * (time.perf_counter() - start), 3),
                "error": error,
                **request.state.trace,
            })

    @app.exception_handler(UnknownAccountError)
    async def unknown_account(request: Request, exc: UnknownAccountError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_version": bundle.version}

    @app.get("/model-info", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        return model_info_from(bundle)

    @app.post("/assess", response_model=AssessResponse)
    def assess(body: AssessRequest, request: Request) -> AssessResponse:
        as_of = resolve_as_of(body.as_of)
        with services.scoring_lock:
            results = scorer.assess(as_of, body.account_ids, rank_capacity=True, persist=body.persist)
        summary = batch_summary(results)
        request.state.trace.update({
            "as_of": as_of.isoformat(),
            "requested_accounts": None if body.account_ids is None else len(body.account_ids),
            "persist": body.persist,
            **summary,
        })
        return AssessResponse(request_id=request.state.request_id, model_version=bundle.version,
                              as_of=as_of, summary=BatchSummary(**summary), assessments=results)

    @app.get("/accounts/{account_id}/assessment", response_model=EngagementAssessment)
    def account_assessment(account_id: str, request: Request, as_of: datetime | None = None) -> EngagementAssessment:
        as_of = resolve_as_of(as_of)
        with services.scoring_lock:
            result = scorer.assess(as_of, [account_id], rank_capacity=False, persist=False)[0]
        request.state.trace.update({"as_of": as_of.isoformat(), "status": result.status,
                                    "risk_band": result.risk_band})
        return result

    @app.get("/accounts/{account_id}/events", response_model=list[EventEnvelope])
    def account_events(account_id: str, request: Request, as_of: datetime | None = None,
                       limit: int = Query(default=20, ge=1, le=500)) -> list[EventEnvelope]:
        as_of = resolve_as_of(as_of)
        events = scorer.recent_events(account_id, as_of, limit)
        request.state.trace.update({"as_of": as_of.isoformat(), "returned": len(events)})
        return events

    @app.get("/accounts/{account_id}/risk-history", response_model=list[EngagementAssessment])
    def risk_history(account_id: str, request: Request) -> list[EngagementAssessment]:
        history = scorer.risk_history(account_id)
        request.state.trace.update({"returned": len(history)})
        return history

    return app


def create_app_from_env() -> FastAPI:
    return create_app(ServiceSettings.from_env())