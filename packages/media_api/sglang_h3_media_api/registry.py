"""In-memory job registry mapping media job ids to engine job ids."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class MediaJob:
    job_id: str
    model: str
    background: bool
    created_at: float = field(default_factory=time.time)
    engine_id: str | None = None
    cancelled: bool = False


_JOBS: dict[str, MediaJob] = {}


def create(job_id: str, model: str, background: bool) -> MediaJob:
    job = MediaJob(job_id=job_id, model=model, background=background)
    _JOBS[job_id] = job
    return job


def exists(job_id: str) -> bool:
    return job_id in _JOBS


def get(job_id: str) -> MediaJob | None:
    return _JOBS.get(job_id)


def remove(job_id: str) -> None:
    _JOBS.pop(job_id, None)


def bind_engine(job_id: str, engine_id: str) -> None:
    job = _JOBS.get(job_id)
    if job is not None:
        job.engine_id = engine_id


def mark_cancelled(job_id: str) -> None:
    job = _JOBS.get(job_id)
    if job is not None:
        job.cancelled = True


def all_jobs() -> list[MediaJob]:
    return list(_JOBS.values())
