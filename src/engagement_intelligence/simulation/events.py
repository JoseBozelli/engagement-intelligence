"""Clean event generation driven by daily latent state (simulator truth).

EventGenerator observes one account's daily state, emits reference-vocabulary
events, and returns experience shocks. The operator policy (reminders, nudges,
follow-ups, tasks) reads only the account's own events whose time has passed,
never latent state. Observation noise is applied in a separate step.
"""

from __future__ import annotations

import heapq
from datetime import date, datetime, timedelta

import pyarrow as pa

from engagement_intelligence.core.arrow_io import EVENT_ARROW_SCHEMA, canonical_json, check_schema
from engagement_intelligence.core.ids import make_event_id, make_ref_id
from engagement_intelligence.core.seeding import rng_for
from engagement_intelligence.schemas.events import REFERENCE_VOCABULARY, Actor, EventType
from engagement_intelligence.simulation.config import SimConfig
from engagement_intelligence.simulation.dynamics import (
    BoundPredictor,
    DayState,
    bind_predictor,
    evaluate_bound,
    sigmoid,
    simulate,
)
from engagement_intelligence.simulation.population import LATENT_PREFIX, utc_midnight


class EventGenerator:
    def __init__(self, cfg: SimConfig, account: dict) -> None:
        self._ev = cfg.events
        self._seed = cfg.seed
        self.account_id: str = account["account_id"]
        self._tenant_id: str = account["tenant_id"]
        self._created_at: datetime = account["created_at"]
        self._horizon = utc_midnight(cfg.calendar.end)
        self._rng = rng_for(cfg.seed, "events", account["account_index"])

        latents = {n: account[LATENT_PREFIX + n] for n in cfg.latents}
        s, f, t, m = self._ev.sessions, self._ev.followups, self._ev.tasks, self._ev.messages
        self._booking = bind_predictor(s.booking, latents)
        self._account_cancel = bind_predictor(s.account_cancel, latents)
        self._no_show = bind_predictor(s.no_show, latents)
        self._followup_completion = bind_predictor(f.completion, latents)
        self._task_completion = bind_predictor(t.completion, latents)
        self._open = bind_predictor(m.open, latents)
        self._reply = bind_predictor(m.reply, latents)
        self._resets_nudge = frozenset(m.nudge_reset_by)

        self.events: list[dict] = []
        self._event_seq = 0
        self._ref_seq = 0
        self._session: dict | None = None
        self._next_task_day = t.interval_days
        self._last_engagement = self._created_at
        self._pending_engagements: list[datetime] = []
        self._last_nudge: datetime | None = None

        self._emit(EventType.ACCOUNT_CREATED, Actor.SYSTEM, self._created_at)

    # --- daily hook ------------------------------------------------------------------

    def on_day(self, day: int, today: date, state: DayState) -> float:
        day_start = utc_midnight(today)
        values = {
            "satisfaction": state.satisfaction,
            "disrupted": float(state.disrupted),
            "dip": float(state.dip),
            "disengaged": float(state.disengaged),
        }
        shock = 0.0
        if self._session is not None and self._session["at"] < day_start + timedelta(days=1):
            shock += self._resolve_session(values)
        if self._session is None and self._chance(self._booking, values):
            self._book(day_start, values)
        if day >= self._next_task_day:
            self._next_task_day += self._ev.tasks.interval_days
            self._task(day_start, values)
        self._maybe_nudge(day_start, values)
        return shock

    # --- behaviors -------------------------------------------------------------------

    def _book(self, day_start: datetime, values: dict) -> None:
        s = self._ev.sessions
        actor = Actor.OPERATOR if self._rng.random() < s.operator_booking_share else Actor.ACCOUNT
        booked_at = self._time_on(day_start)
        lead_days = 1 + int(self._rng.poisson(s.lead_days_mean - 1))
        session_at = self._time_on(day_start + timedelta(days=lead_days))
        ref = self._new_ref()
        if not self._emit(EventType.SESSION_BOOKED, actor, booked_at, ref, lead_days=lead_days):
            return
        self._session = {"ref": ref, "at": session_at, "booked_at": booked_at}
        reminder_day = utc_midnight(session_at.date()) - timedelta(days=self._ev.messages.reminder_days_before)
        reminder_at = self._time_on(reminder_day)
        if reminder_at > booked_at:
            self._message(reminder_at, "reminder", values)

    def _resolve_session(self, values: dict) -> float:
        session, self._session = self._session, None
        s = self._ev.sessions
        ref, at = session["ref"], session["at"]
        if self._rng.random() < s.operator_cancel_prob:
            self._emit(EventType.SESSION_CANCELLED, Actor.OPERATOR, self._before(at, session["booked_at"]), ref)
            return s.operator_cancel_shock
        if self._chance(self._account_cancel, values):
            self._emit(EventType.SESSION_CANCELLED, Actor.ACCOUNT, self._before(at, session["booked_at"]), ref)
            return 0.0
        if self._chance(self._no_show, values):
            self._emit(EventType.SESSION_NO_SHOW, Actor.ACCOUNT, at, ref)
            return 0.0
        self._emit(EventType.SESSION_COMPLETED, Actor.ACCOUNT, at, ref)
        self._maybe_followup(at, ref, values)
        return s.completed_shock

    def _maybe_followup(self, session_at: datetime, session_ref: str, values: dict) -> None:
        f = self._ev.followups
        if self._rng.random() >= f.required_prob:
            return
        ref = self._new_ref()
        requested = session_at + timedelta(hours=self._rng.exponential(f.request_delay_hours_mean))
        if self._emit(EventType.FOLLOWUP_REQUIRED, Actor.OPERATOR, requested, ref,
                      related_ref_id=session_ref) and self._chance(self._followup_completion, values):
            done = requested + timedelta(days=self._rng.exponential(f.delay_days_mean))
            self._emit(EventType.FOLLOWUP_COMPLETED, Actor.ACCOUNT, done, ref)

    def _task(self, day_start: datetime, values: dict) -> None:
        t = self._ev.tasks
        ref = self._new_ref()
        requested = self._time_on(day_start)
        if self._emit(EventType.TASK_REQUIRED, Actor.OPERATOR, requested, ref) and self._chance(
            self._task_completion, values
        ):
            done = requested + timedelta(days=self._rng.exponential(t.delay_days_mean))
            self._emit(EventType.TASK_COMPLETED, Actor.ACCOUNT, done, ref)

    def _message(self, sent_at: datetime, kind: str, values: dict) -> None:
        m = self._ev.messages
        ref = self._new_ref()
        if not self._emit(EventType.MESSAGE_SENT, Actor.OPERATOR, sent_at, ref, kind=kind):
            return
        if not self._chance(self._open, values):
            return
        opened = sent_at + timedelta(hours=self._rng.exponential(m.open_delay_hours_mean))
        if self._emit(EventType.MESSAGE_OPENED, Actor.ACCOUNT, opened, ref) and self._chance(self._reply, values):
            replied = opened + timedelta(hours=self._rng.exponential(m.reply_delay_hours_mean))
            self._emit(EventType.MESSAGE_REPLIED, Actor.ACCOUNT, replied, ref)

    def _maybe_nudge(self, day_start: datetime, values: dict) -> None:
        m = self._ev.messages
        while self._pending_engagements and self._pending_engagements[0] <= day_start:
            self._last_engagement = max(self._last_engagement, heapq.heappop(self._pending_engagements))
        silent = day_start - self._last_engagement >= timedelta(days=m.nudge_after_days)
        cooled = self._last_nudge is None or day_start - self._last_nudge >= timedelta(days=m.nudge_cooldown_days)
        if silent and cooled:
            self._last_nudge = self._time_on(day_start)
            self._message(self._last_nudge, "nudge", values)

    # --- helpers ---------------------------------------------------------------------

    def _chance(self, bound: BoundPredictor, values: dict) -> bool:
        return self._rng.random() < sigmoid(evaluate_bound(bound, values))

    def _time_on(self, day_start: datetime) -> datetime:
        h = self._ev.activity_hours
        t = day_start + timedelta(hours=self._rng.uniform(h.start, h.end))
        return max(t, self._created_at + timedelta(minutes=1))

    def _before(self, at: datetime, not_before: datetime) -> datetime:
        notice = timedelta(hours=self._rng.uniform(0.0, self._ev.sessions.cancel_notice_hours_max))
        return max(at - notice, not_before + timedelta(minutes=1))

    def _new_ref(self) -> str:
        ref = make_ref_id(self._seed, self.account_id, self._ref_seq)
        self._ref_seq += 1
        return ref

    def _emit(self, event_type: EventType, actor: Actor, timestamp: datetime,
              ref_id: str | None = None, related_ref_id: str | None = None, **attributes) -> bool:
        if timestamp >= self._horizon:
            return False
        self.events.append(
            {
                "event_id": make_event_id(self._seed, self.account_id, self._event_seq),
                "tenant_id": self._tenant_id,
                "account_id": self.account_id,
                "event_type": str(event_type),
                "actor": str(actor),
                "timestamp": timestamp,
                "recorded_at": timestamp,
                "ref_id": ref_id,
                "related_ref_id": related_ref_id,
                "attributes_json": canonical_json(attributes),
                "schema_version": 1,
            }
        )
        self._event_seq += 1
        if actor is Actor.ACCOUNT and str(event_type) in self._resets_nudge:
            heapq.heappush(self._pending_engagements, timestamp)
        return True


def simulate_world(cfg: SimConfig, accounts: pa.Table) -> tuple[pa.Table, pa.Table, pa.Table]:
    """States (with experience feedback), disengagement episodes, and clean events."""
    states, episodes, generators = simulate(cfg, accounts, lambda row: EventGenerator(cfg, row))
    events = pa.Table.from_pylist([e for g in generators for e in g.events], schema=EVENT_ARROW_SCHEMA)
    events = events.sort_by(
        [("account_id", "ascending"), ("timestamp", "ascending"), ("event_id", "ascending")]
    )
    check_schema(events)
    REFERENCE_VOCABULARY.check_table(events)
    return states, episodes, events