"""End-to-end tests for the single-device shift sampling QC API.

Acceptance times are controlled by updating the ``created_at`` column of the
temp SQLite DB directly (test-only clock manipulation): ``ON_TIME`` is before
the collection deadline, ``LATE`` is after it.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

GRID_START = "2026-11-10T00:00:00Z"
GRID_END = "2026-11-10T08:00:00Z"
DEADLINE = "2026-11-10T08:15:00Z"
ON_TIME = "2026-11-10T08:10:00.000Z"   # accepted before the deadline
LATE = "2026-11-10T08:20:00.000Z"      # accepted after the deadline


@pytest.fixture()
def ctx():
    tmp = tempfile.TemporaryDirectory()
    db_path = os.path.join(tmp.name, "test.db")
    settings = Settings(db_path=db_path, fail_device=None)
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, db_path, tmp
    tmp.cleanup()


def reading(source, rev, device, ts, meter, kwh):
    return {
        "source_id": source, "revision": rev, "device_id": device,
        "ts": ts, "meter_no": meter, "cumulative_kwh": kwh,
    }


def handover(source, rev, device, ts, old_m, old_v, new_m, new_v):
    return {
        "source_id": source, "revision": rev, "device_id": device, "ts": ts,
        "old_meter_no": old_m, "old_final_kwh": old_v,
        "new_meter_no": new_m, "new_initial_kwh": new_v,
    }


def submit_grid(client, device="D1", prefix="S", hours=8, base=1000.0,
                step=1.0, skip=(), overrides=None, meter="M1"):
    """Post one reading per 15-minute sample moment, one source per moment."""
    overrides = overrides or {}
    t0 = datetime(2026, 11, 10, tzinfo=timezone.utc)
    sources = {}
    for i in range(hours * 4 + 1):
        if i in skip:
            continue
        ts = (t0 + timedelta(minutes=15 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        src = f"{prefix}{i + 1}"
        sources[i] = src
        kwh = overrides.get(i, base + step * i)
        r = client.post("/readings", json=reading(src, 1, device, ts, meter, kwh))
        assert r.status_code == 201
    return sources


def set_accepted(db_path, accepted_at, table="readings", source_id=None, revision=None):
    """Test-only clock control: set the acceptance time of stored rows."""
    conn = sqlite3.connect(db_path)
    try:
        sql = f"UPDATE {table} SET created_at = ?"
        params = [accepted_at]
        if source_id is not None:
            sql += " WHERE source_id = ?"
            params.append(source_id)
            if revision is not None:
                sql += " AND revision = ?"
                params.append(revision)
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def assess(client, device="D1", start=GRID_START, end=GRID_END, deadline=DEADLINE):
    return client.post("/shift-qc", json={
        "device_id": device, "shift_start": start, "shift_end": end,
        "collect_deadline": deadline,
    })


def sample_at(body, ts):
    return next(s for s in body["samples"] if s["ts"] == ts)


# ---------------------------------------------------------------------------
# 1. Happy path: every sample on time and unique, full interval coverage
# ---------------------------------------------------------------------------

def test_shift_qc_ready_when_all_on_time(ctx):
    client, db_path, _ = ctx
    submit_grid(client)
    set_accepted(db_path, ON_TIME)

    r = assess(client)
    assert r.status_code == 201
    body = r.json()
    assert body["analysis_ready"] is True
    assert body["expected_samples"] == 33
    assert body["on_time_count"] == 33
    assert body["missing_count"] == 0
    assert body["overdue_count"] == 0
    assert body["multi_source_count"] == 0
    assert body["shift_energy_kwh"] == pytest.approx(32.0)  # 32 intervals x 1.0
    assert body["reasons"] == []
    assert body["coverage"]["full_coverage"] is True
    assert body["status"] == "current"
    assert all(
        s["status"] == "on_time" and len(s["sources"]) == 1
        for s in body["samples"]
    )

    # Persisted and queryable as the current conclusion.
    got = client.get("/shift-qc", params={"device_id": "D1"}).json()
    assert len(got) == 1
    assert got[0]["id"] == body["id"]
    assert got[0]["analysis_ready"] is True
    assert got[0]["samples"][0]["sources"][0]["source_id"] == "S1"


# ---------------------------------------------------------------------------
# 2. Sample defects: missing / overdue / multi-source
# ---------------------------------------------------------------------------

def test_shift_qc_missing_sample(ctx):
    client, db_path, _ = ctx
    submit_grid(client, skip={5})  # no reading at 01:15
    set_accepted(db_path, ON_TIME)

    body = assess(client).json()
    assert body["analysis_ready"] is False
    assert body["missing_count"] == 1
    s = sample_at(body, "2026-11-10T01:15:00.000Z")
    assert s["status"] == "missing" and s["sources"] == []
    assert any("missing_samples" in r for r in body["reasons"])
    # Intervals still tile the shift (01:00->01:30 spans the gap), so the
    # energy stays computable; only the sampling verdict blocks analysis.
    assert body["shift_energy_kwh"] == pytest.approx(32.0)


def test_shift_qc_overdue_sample(ctx):
    client, db_path, _ = ctx
    submit_grid(client)
    set_accepted(db_path, ON_TIME)
    set_accepted(db_path, LATE, source_id="S10", revision=1)  # 02:15 reading

    body = assess(client).json()
    assert body["analysis_ready"] is False
    assert body["overdue_count"] == 1
    s = sample_at(body, "2026-11-10T02:15:00.000Z")
    assert s["status"] == "overdue"
    assert s["sources"][0]["source_id"] == "S10"
    assert s["sources"][0]["accepted_at"] == LATE
    assert any("overdue_samples" in r and "S10" in r for r in body["reasons"])


def test_shift_qc_late_revision_loses_punctuality(ctx):
    """A late revision must not inherit the old revision's on-time status."""
    client, db_path, _ = ctx
    submit_grid(client)
    set_accepted(db_path, ON_TIME)  # revision 1 of S7 was punctual...

    # ...but the correcting revision 2 (current) arrives after the deadline.
    r = client.post("/readings", json=reading(
        "S7", 2, "D1", "2026-11-10T01:30:00Z", "M1", 1006.4))
    assert r.status_code == 201
    set_accepted(db_path, LATE, source_id="S7", revision=2)

    body = assess(client).json()
    s = sample_at(body, "2026-11-10T01:30:00.000Z")
    assert s["status"] == "overdue"           # judged by the current revision
    assert s["sources"][0]["revision"] == 2
    assert body["overdue_count"] == 1
    assert body["analysis_ready"] is False


def test_shift_qc_multi_source_same_moment(ctx):
    client, db_path, _ = ctx
    submit_grid(client)
    # A second source reports the same 00:30 moment with the same value.
    r = client.post("/readings", json=reading(
        "X1", 1, "D1", "2026-11-10T00:30:00Z", "M1", 1002.0))
    assert r.status_code == 201
    set_accepted(db_path, ON_TIME)

    body = assess(client).json()
    assert body["analysis_ready"] is False
    assert body["multi_source_count"] == 1
    s = sample_at(body, "2026-11-10T00:30:00.000Z")
    assert s["status"] == "multi_source"
    assert {x["source_id"] for x in s["sources"]} == {"S3", "X1"}
    assert any("multi_source_samples" in r for r in body["reasons"])


# ---------------------------------------------------------------------------
# 3. Interval coverage: undeterminable / pending rebuild are never zero
# ---------------------------------------------------------------------------

def test_shift_qc_undeterminable_interval_not_zero(ctx):
    client, db_path, _ = ctx
    # Reading at index 10 (02:30) goes backward -> interval undeterminable.
    submit_grid(client, overrides={10: 1005.0})
    set_accepted(db_path, ON_TIME)

    body = assess(client).json()
    assert body["analysis_ready"] is False
    assert body["on_time_count"] == 33          # sampling itself is fine
    assert body["shift_energy_kwh"] is None     # never reported as zero
    undet = body["coverage"]["undeterminable_intervals"]
    assert len(undet) == 1
    assert undet[0]["reason"].startswith("negative_difference")
    assert any("intervals_undeterminable" in r for r in body["reasons"])


def test_shift_qc_pending_rebuild_after_failed_rebuild(ctx):
    client, db_path, _ = ctx
    submit_grid(client)
    set_accepted(db_path, ON_TIME)

    # Late off-grid reading; the automatic rebuild is forced to fail, so the
    # device is left with stale intervals only.
    r = client.post(
        "/readings", headers={"X-Fail-Rebuild": "true"},
        json=reading("L1", 1, "D1", "2026-11-10T04:07:00Z", "M1", 1016.7),
    )
    assert r.json()["rebuild"]["success"] is False

    body = assess(client).json()
    assert body["analysis_ready"] is False
    assert body["on_time_count"] == 33          # grid samples still punctual
    assert body["coverage"]["full_coverage"] is False
    assert body["coverage"]["pending_rebuild"] is True
    assert body["shift_energy_kwh"] is None     # pending rebuild, not zero
    assert any("pending_rebuild" in r for r in body["reasons"])


def test_shift_qc_no_data_device(ctx):
    client, _, _ = ctx
    body = assess(client).json()
    assert body["analysis_ready"] is False
    assert body["missing_count"] == 33
    assert body["shift_energy_kwh"] is None
    assert body["coverage"]["pending_rebuild"] is False
    assert any("interval_coverage_gap" in r for r in body["reasons"])


# ---------------------------------------------------------------------------
# 4. Conclusion lifecycle: invalidation, supersede, restart
# ---------------------------------------------------------------------------

def test_new_reading_invalidates_conclusion_unrelated_device_unaffected(ctx):
    client, db_path, _ = ctx
    submit_grid(client, "D1", "S")
    submit_grid(client, "D2", "K")
    set_accepted(db_path, ON_TIME)
    a1 = assess(client, "D1").json()
    a2 = assess(client, "D2").json()
    assert a1["analysis_ready"] and a2["analysis_ready"]

    # A newly accepted reading on D1 revokes D1's conclusion immediately.
    client.post("/readings", json=reading(
        "S99", 1, "D1", "2026-11-10T09:00:00Z", "M1", 1040.0))

    assert client.get("/shift-qc", params={"device_id": "D1"}).json() == []
    stale = client.get("/shift-qc", params={"device_id": "D1", "status": "stale"}).json()
    assert len(stale) == 1 and stale[0]["id"] == a1["id"]
    # Unrelated device's conclusion keeps serving as current.
    cur2 = client.get("/shift-qc", params={"device_id": "D2"}).json()
    assert len(cur2) == 1 and cur2[0]["id"] == a2["id"]


def test_handover_acceptance_invalidates_conclusion(ctx):
    client, db_path, _ = ctx
    submit_grid(client)
    set_accepted(db_path, ON_TIME)
    assert assess(client).json()["analysis_ready"] is True

    r = client.post("/handovers", json=handover(
        "H1", 1, "D1", "2026-11-10T10:00:00Z", "M1", 1100.0, "M2", 0.0))
    assert r.status_code == 201
    assert client.get("/shift-qc", params={"device_id": "D1"}).json() == []
    assert len(client.get(
        "/shift-qc", params={"device_id": "D1", "status": "stale"}).json()) == 1


def test_idempotent_resubmission_keeps_conclusion(ctx):
    client, db_path, _ = ctx
    submit_grid(client)
    set_accepted(db_path, ON_TIME)
    assess(client)

    r = client.post("/readings", json=reading(
        "S1", 1, "D1", "2026-11-10T00:00:00Z", "M1", 1000.0))
    assert r.json()["idempotent"] is True
    assert len(client.get("/shift-qc", params={"device_id": "D1"}).json()) == 1


def test_failed_rebuild_and_restart_do_not_restore_conclusion(ctx):
    client, db_path, _ = ctx
    submit_grid(client)
    set_accepted(db_path, ON_TIME)
    assess(client)

    # Accepted reading + failing rebuild: conclusion is revoked and the
    # failed rebuild must not bring it back.
    client.post(
        "/readings", headers={"X-Fail-Rebuild": "true"},
        json=reading("L1", 1, "D1", "2026-11-10T09:00:00Z", "M1", 1040.0),
    )
    assert client.get("/shift-qc", params={"device_id": "D1"}).json() == []

    # Simulated process restart against the same DB: still not restored.
    app2 = create_app(Settings(db_path=db_path, fail_device=None))
    with TestClient(app2) as client2:
        assert client2.get("/shift-qc", params={"device_id": "D1"}).json() == []
        stale = client2.get(
            "/shift-qc", params={"device_id": "D1", "status": "stale"}).json()
        assert len(stale) == 1
        # Recompute after restart reads consistent current data.
        r = client2.post("/rebuild", params={"device_id": "D1"})
        assert r.json()["success"] is True
        set_accepted(db_path, ON_TIME)
        body = assess(client2).json()
        assert body["analysis_ready"] is True
        # The 09:00 reading lies outside the shift window; in-shift energy
        # is exactly the 32 one-kWh intervals.
        assert body["shift_energy_kwh"] == pytest.approx(32.0)


def test_reassessment_supersedes_same_window_only(ctx):
    client, db_path, _ = ctx
    submit_grid(client)
    set_accepted(db_path, ON_TIME)

    first = assess(client).json()
    second = assess(client).json()  # same window again
    assert second["id"] != first["id"]

    current = client.get("/shift-qc", params={"device_id": "D1"}).json()
    assert len(current) == 1 and current[0]["id"] == second["id"]
    all_rows = client.get("/shift-qc", params={"device_id": "D1", "status": "all"}).json()
    assert len(all_rows) == 2

    # A different shift window gets its own current conclusion.
    third = assess(
        client, start="2026-11-10T08:00:00Z", end="2026-11-10T16:00:00Z",
        deadline="2026-11-10T16:15:00Z",
    ).json()
    assert third["analysis_ready"] is False  # afternoon grid never submitted
    current = client.get("/shift-qc", params={"device_id": "D1"}).json()
    assert len(current) == 2


# ---------------------------------------------------------------------------
# 5. Input validation
# ---------------------------------------------------------------------------

def test_shift_qc_validation_errors(ctx):
    client, _, _ = ctx
    base = {
        "device_id": "D1", "shift_start": GRID_START, "shift_end": GRID_END,
        "collect_deadline": DEADLINE,
    }
    # end not after start
    assert client.post("/shift-qc", json={
        **base, "shift_end": GRID_START}).status_code == 422
    # window not an integer multiple of 15 minutes
    assert client.post("/shift-qc", json={
        **base, "shift_end": "2026-11-10T08:10:00Z"}).status_code == 422
    # deadline must be later than shift end
    assert client.post("/shift-qc", json={
        **base, "collect_deadline": GRID_END}).status_code == 422
    # timestamps must carry a timezone
    assert client.post("/shift-qc", json={
        **base, "shift_start": "2026-11-10T00:00:00"}).status_code == 422


def test_shift_qc_requires_deadline_passed(ctx):
    client, _, _ = ctx
    r = client.post("/shift-qc", json={
        "device_id": "D1", "shift_start": GRID_START, "shift_end": GRID_END,
        "collect_deadline": "2999-01-01T00:00:00Z",
    })
    assert r.status_code == 422
    assert "has not passed" in r.json()["detail"]


def test_shift_qc_openapi_and_docs(ctx):
    client, _, _ = ctx
    spec = client.get("/openapi.json").json()
    assert "/shift-qc" in spec["paths"]
    assert "post" in spec["paths"]["/shift-qc"]
    assert "get" in spec["paths"]["/shift-qc"]
    assert client.get("/docs").status_code == 200
