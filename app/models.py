"""Pydantic request/response models and shared helpers."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def to_utc_iso(dt: datetime) -> str:
    """Normalize any aware datetime to a UTC ISO-8601 string."""
    if dt.tzinfo is None:
        raise ValueError("timestamp must include a timezone (use UTC, e.g. 2026-01-01T00:00:00Z)")
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class ReadingIn(BaseModel):
    source_id: str = Field(min_length=1, description="来源编号，同一读数来源的唯一标识")
    revision: int = Field(ge=1, description="递增修订号，首次为 1，再次提交必须 +1")
    device_id: str = Field(min_length=1)
    ts: datetime = Field(description="读数时刻（UTC，需带时区）")
    meter_no: str = Field(min_length=1)
    cumulative_kwh: float = Field(ge=0, description="累计电表读数 kWh（非负）")

    @field_validator("source_id", "device_id", "meter_no")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class HandoverIn(BaseModel):
    source_id: str = Field(min_length=1, description="来源编号")
    revision: int = Field(ge=1, description="递增修订号")
    device_id: str = Field(min_length=1)
    ts: datetime = Field(description="换表时刻（UTC，需带时区）")
    old_meter_no: str = Field(min_length=1)
    old_final_kwh: float = Field(ge=0, description="旧表末值（累计 kWh）")
    new_meter_no: str = Field(min_length=1)
    new_initial_kwh: float = Field(ge=0, description="新表初值（累计 kWh）")

    @field_validator("source_id", "device_id", "old_meter_no", "new_meter_no")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("new_meter_no")
    @classmethod
    def _distinct(cls, v: str, info):
        old = info.data.get("old_meter_no")
        if old is not None and v == old:
            raise ValueError("new_meter_no must differ from old_meter_no")
        return v


class AcceptOut(BaseModel):
    accepted: bool
    kind: str  # reading | handover
    source_id: str
    revision: int
    idempotent: bool = False
    affected_devices: list[str]
    rebuild: dict


class IntervalOut(BaseModel):
    id: int
    device_id: str
    start_ts: str
    end_ts: str
    start_meter_no: str
    end_meter_no: str
    energy_kwh: float | None
    undeterminable: bool
    reason: str | None
    status: str
    start_source_id: str
    start_revision: int
    end_source_id: str
    end_revision: int
    handover_source_id: str | None
    handover_revision: int | None


class RebuildOut(BaseModel):
    run_id: int
    success: bool
    rebuilt_devices: list[str]
    failed_devices: dict[str, str]
    detail: str | None = None


# ---------------------------------------------------------------------------
# Shift sampling quality control
# ---------------------------------------------------------------------------

class ShiftQcIn(BaseModel):
    device_id: str = Field(min_length=1, description="设备编号")
    shift_start: datetime = Field(description="班次开始时刻（UTC，需带时区）")
    shift_end: datetime = Field(
        description="班次结束时刻（UTC，需带时区）；与开始时刻相隔 15 分钟整数倍"
    )
    collect_deadline: datetime = Field(
        description="收数截止时刻（UTC，需带时区），必须晚于班次结束；截止后才可核对"
    )

    @field_validator("device_id")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("shift_start", "shift_end", "collect_deadline")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError(
                "timestamp must include a timezone (use UTC, e.g. 2026-01-01T00:00:00Z)"
            )
        return v

    @model_validator(mode="after")
    def _check_window(self) -> "ShiftQcIn":
        start = self.shift_start.astimezone(timezone.utc)
        end = self.shift_end.astimezone(timezone.utc)
        deadline = self.collect_deadline.astimezone(timezone.utc)
        if end <= start:
            raise ValueError("shift_end must be later than shift_start")
        seconds = (end - start).total_seconds()
        if seconds % 900 != 0:
            raise ValueError(
                "shift_start and shift_end must be an integer multiple of "
                "15 minutes apart"
            )
        if seconds / 900 + 1 > 10001:
            raise ValueError("shift window too long: more than 10000 sample moments")
        if deadline <= end:
            raise ValueError("collect_deadline must be later than shift_end")
        return self


class ShiftSampleSourceOut(BaseModel):
    source_id: str
    revision: int
    accepted_at: str


class ShiftSampleOut(BaseModel):
    ts: str
    status: str  # on_time | overdue | missing | multi_source
    sources: list[ShiftSampleSourceOut]


class ShiftSpanOut(BaseModel):
    start_ts: str
    end_ts: str


class ShiftUndeterminableOut(BaseModel):
    start_ts: str
    end_ts: str
    reason: str | None


class ShiftCoverageOut(BaseModel):
    full_coverage: bool
    gaps: list[ShiftSpanOut]
    undeterminable_intervals: list[ShiftUndeterminableOut]
    boundary_crossing_intervals: list[ShiftSpanOut]
    pending_rebuild: bool


class ShiftQcOut(BaseModel):
    id: int
    device_id: str
    shift_start: str
    shift_end: str
    collect_deadline: str
    assessed_at: str
    status: str  # current | stale
    analysis_ready: bool
    shift_energy_kwh: float | None  # NULL unless exactly determinable; never a fake zero
    expected_samples: int
    on_time_count: int
    overdue_count: int
    missing_count: int
    multi_source_count: int
    samples: list[ShiftSampleOut]
    coverage: ShiftCoverageOut
    reasons: list[str]
