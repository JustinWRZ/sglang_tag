"""media v1 HTTP routes installed onto the engine app via a create_app hook."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from typing import Any

from fastapi import APIRouter
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse, Response

from . import capabilities as caps
from . import registry
from .mapping import to_engine_payload
from .validation import validate_request

media_router = APIRouter()

_ERROR_TYPE_BY_ENGINE_MARKERS = (
    "must be",
    "required",
    "not supported",
    "not served",
    "invalid",
)


def _error_body(type_: str, message: str, **extra: Any) -> dict[str, Any]:
    error: dict[str, Any] = {"type": type_, "message": message}
    error.update({k: v for k, v in extra.items() if v is not None})
    return {"error": error}


def _validation_body(message: str, errors: list[dict]) -> dict[str, Any]:
    return {"error": "validation_error", "message": message, "errors": errors}


def _video_api():
    from sglang.multimodal_gen.runtime.entrypoints.openai import video_api

    return video_api


async def _engine_raw(engine_id: str | None) -> dict[str, Any] | None:
    if not engine_id:
        return None
    return await _video_api().VIDEO_STORE.get(engine_id)


def _engine_status(raw: dict[str, Any] | None) -> str:
    if not raw:
        return "queued"
    status = raw.get("status")
    if status in {"completed", "failed", "cancelled"}:
        return str(status)
    return "running"


def _classify_error(message: str) -> str:
    lowered = message.lower()
    if any(marker in lowered for marker in _ERROR_TYPE_BY_ENGINE_MARKERS):
        return "user_error"
    return "internal_error"


def _stage_for(status: str) -> str:
    if status == "completed":
        return "done"
    return "generating"


def _translate_status(job: registry.MediaJob, raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    engine_status = _engine_status(raw)
    if job.cancelled:
        status = "cancelled"
    else:
        status = engine_status
    created_at = float(raw.get("created_at") or job.created_at)
    completed_at = raw.get("completed_at")
    completed_at = float(completed_at) if completed_at else None
    elapsed = (completed_at or time.time()) - created_at
    body: dict[str, Any] = {
        "job_id": job.job_id,
        "status": status,
        "model": job.model,
        "background": job.background,
        "created_at": created_at,
        "started_at": None,
        "completed_at": completed_at,
        "elapsed_s": round(elapsed, 3),
        "stage": _stage_for(status),
        "progress": 1.0 if status == "completed" else None,
    }
    if status == "failed":
        message = "generation failed"
        error_field = (raw or {}).get("error")
        if isinstance(error_field, dict):
            message = str(error_field.get("message") or message)
        elif error_field:
            message = str(error_field)
        body["error"] = {
            "type": _classify_error(message),
            "message": message,
            "stage": "generating",
            "upstream_task_id": job.engine_id,
        }
    return body


async def _wait_terminal(
    job: registry.MediaJob, wait_s: float
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        raw = await _engine_raw(job.engine_id)
        body = _translate_status(job, raw)
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body, raw
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, raw
        await asyncio.sleep(min(0.5, remaining))


def _result_envelope(
    job: registry.MediaJob, raw: dict[str, Any], base_url: str
) -> dict[str, Any]:
    raw = raw or {}
    created_at = float(raw.get("created_at") or job.created_at)
    completed_at = float(raw.get("completed_at") or time.time())
    artifact: dict[str, Any] = {
        "type": "video",
        "url": f"{base_url}media/{job.job_id}/artifact.mp4",
        "mime_type": "video/mp4",
    }
    seed = raw.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        artifact["seed"] = seed
    file_paths = raw.get("file_paths") or []
    if file_paths and file_paths[0] and os.path.exists(file_paths[0]):
        artifact["size_bytes"] = os.path.getsize(file_paths[0])
    artifact["expires_at"] = completed_at + caps.ARTIFACT_TTL_S
    return {
        "job_id": job.job_id,
        "status": "completed",
        "model": job.model,
        "batch": 1,
        "artifacts": [artifact],
        "created_at": created_at,
        "completed_at": completed_at,
        "elapsed_s": round(completed_at - created_at, 3),
    }


@media_router.post("/generate")
async def media_generate(request: Request):
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" not in content_type:
        return JSONResponse(
            status_code=422,
            content=_validation_body(
                "content-type must be application/json",
                [{"loc": ["header", "content-type"], "msg": "expected application/json", "type": "value_error"}],
            ),
        )
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(
            status_code=422,
            content=_validation_body(
                "request body is not valid JSON",
                [{"loc": ["body"], "msg": "invalid JSON", "type": "json_invalid"}],
            ),
        )
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=422,
            content=_validation_body(
                "request body must be a JSON object",
                [{"loc": ["body"], "msg": "expected object", "type": "model_type"}],
            ),
        )

    normalized, errors = validate_request(body)
    if errors:
        return JSONResponse(
            status_code=422, content=_validation_body("request validation failed", errors)
        )

    job_id = normalized.get("job_id") or secrets.token_hex(16)
    if registry.exists(job_id):
        return JSONResponse(
            status_code=409,
            content=_error_body("user_error", f"job_id {job_id} already exists", status_code=409),
        )

    model = normalized.get("model") or caps.service_models()[0]
    background = normalized.get("background", caps.DEFAULT_BACKGROUND)
    registry.create(job_id, model=model, background=background)

    engine_payload = to_engine_payload(normalized)
    request._body = json.dumps(engine_payload).encode()

    from fastapi import HTTPException

    try:
        result = await asyncio.wait_for(
            _video_api().create_video(request), timeout=caps.SUBMIT_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        registry.remove(job_id)
        return JSONResponse(
            status_code=422,
            content=_validation_body(
                "reference asset admission timed out",
                [
                    {
                        "loc": ["assets"],
                        "msg": (
                            "reference asset download or admission did not finish "
                            f"within {caps.SUBMIT_TIMEOUT_S:.0f}s"
                        ),
                        "type": "value_error",
                    }
                ],
            ),
        )
    except HTTPException as exc:
        registry.remove(job_id)
        detail = str(exc.detail)
        if exc.status_code == 429:
            return JSONResponse(
                status_code=429,
                content=_error_body("upstream_exhausted", detail, status_code=429),
            )
        if 400 <= exc.status_code < 500:
            return JSONResponse(
                status_code=400, content=_error_body("user_error", detail)
            )
        return JSONResponse(
            status_code=502, content=_error_body("upstream_api_error", detail)
        )
    except Exception as exc:  # noqa: BLE001
        registry.remove(job_id)
        return JSONResponse(
            status_code=502, content=_error_body("internal_error", str(exc))
        )

    engine_id = getattr(result, "id", None) or (result or {}).get("id")
    registry.bind_engine(job_id, engine_id)
    job = registry.get(job_id)
    assert job is not None

    headers = {
        "X-Job-Id": job_id,
        "Location": f"/{job_id}/status",
        "Retry-After": "3",
    }
    if not background:
        body_status, raw = await _wait_terminal(job, caps.SYNC_WAIT_S)
        if body_status is None:
            return JSONResponse(
                status_code=202, content=_translate_status(job, raw), headers=headers
            )
        if body_status["status"] == "completed":
            envelope = _result_envelope(job, raw or {}, str(request.base_url))
            return JSONResponse(
                status_code=200,
                content=envelope,
                headers={"X-Job-Id": job_id, "X-Elapsed-S": str(envelope["elapsed_s"])},
            )
        if body_status["status"] == "failed":
            return JSONResponse(
                status_code=502,
                content={"error": body_status["error"]},
                headers=headers,
            )
        return JSONResponse(status_code=202, content=body_status, headers=headers)

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "pending", "model": model},
        headers=headers,
    )


@media_router.get("/jobs")
async def media_list_jobs(
    status: str | None = None, limit: int = 50, offset: int = 0
):
    jobs = registry.all_jobs()
    bodies = []
    for job in jobs:
        raw = await _engine_raw(job.engine_id)
        body = _translate_status(job, raw)
        if status and body["status"] != status:
            continue
        bodies.append(body)
    count = len(bodies)
    safe_limit = max(1, min(limit, 500))
    safe_offset = max(0, offset)
    return {
        "jobs": bodies[safe_offset : safe_offset + safe_limit],
        "count": count,
        "limit": safe_limit,
        "offset": safe_offset,
    }


@media_router.get("/health")
async def media_health():
    return {
        "api": caps.API_NAME,
        "api_version": caps.API_VERSION,
        "service": caps.SERVICE_NAME,
        "modalities": list(caps.MODALITIES),
        "models": caps.service_models(),
        "mode": caps.MODE,
    }


@media_router.get("/media/{job_id}/artifact.mp4")
async def media_artifact(job_id: str):
    job = registry.get(job_id)
    if job is None:
        return JSONResponse(
            status_code=404, content=_error_body("user_error", "unknown job_id")
        )
    raw = await _engine_raw(job.engine_id)
    body = _translate_status(job, raw)
    if body["status"] in {"failed", "cancelled"}:
        return JSONResponse(
            status_code=410,
            content=_error_body(
                "user_error", f"job {body['status']}; no artifact available"
            ),
        )
    if body["status"] != "completed":
        return JSONResponse(status_code=202, content=body)
    completed_at = body.get("completed_at") or time.time()
    if time.time() > completed_at + caps.ARTIFACT_TTL_S:
        return JSONResponse(
            status_code=410,
            content=_error_body(
                "user_error",
                f"artifact expired (TTL {caps.ARTIFACT_TTL_S:.0f}s)",
            ),
        )
    try:
        return await _video_api().download_video_content(
            video_id=job.engine_id, variant=None
        )
    except Exception:  # noqa: BLE001
        return JSONResponse(
            status_code=410,
            content=_error_body("user_error", "artifact no longer available"),
        )


@media_router.get("/{job_id}/status")
async def media_job_status(job_id: str, request: Request):
    job = registry.get(job_id)
    if job is None:
        return JSONResponse(
            status_code=404, content=_error_body("user_error", "unknown job_id")
        )
    wait_raw = request.query_params.get("wait", "0")
    try:
        wait_s = max(0.0, min(float(wait_raw), 60.0))
    except ValueError:
        return JSONResponse(
            status_code=422,
            content=_validation_body(
                "invalid wait parameter",
                [{"loc": ["query", "wait"], "msg": "wait must be a number 0..60", "type": "value_error"}],
            ),
        )
    body, raw = await _wait_terminal(job, wait_s)
    if body is None:
        body = _translate_status(job, raw)
    return JSONResponse(content=body, headers={"X-Job-Id": job_id})


@media_router.get("/{job_id}")
async def media_job_result(job_id: str, request: Request):
    job = registry.get(job_id)
    if job is None:
        return JSONResponse(
            status_code=404, content=_error_body("user_error", "unknown job_id")
        )
    raw = await _engine_raw(job.engine_id)
    body = _translate_status(job, raw)
    headers = {"X-Job-Id": job_id}
    if body["status"] == "completed":
        envelope = _result_envelope(job, raw or {}, str(request.base_url))
        headers["X-Elapsed-S"] = str(envelope["elapsed_s"])
        return JSONResponse(status_code=200, content=envelope, headers=headers)
    if body["status"] in {"failed", "cancelled"}:
        error = body.get("error") or _error_body("user_error", f"job {body['status']}")["error"]
        return JSONResponse(status_code=410, content={"error": error}, headers=headers)
    return JSONResponse(status_code=202, content=body, headers=headers)


@media_router.delete("/{job_id}")
async def media_job_cancel(job_id: str):
    job = registry.get(job_id)
    if job is None:
        return JSONResponse(
            status_code=404, content=_error_body("user_error", "unknown job_id")
        )
    raw = await _engine_raw(job.engine_id)
    engine_status = _engine_status(raw)
    if job.cancelled or engine_status in {"completed", "failed", "cancelled"}:
        return JSONResponse(
            status_code=409,
            content=_error_body(
                "user_error",
                f"job already in terminal state ({engine_status})",
                status_code=409,
            ),
        )
    try:
        await _video_api().delete_video(video_id=job.engine_id)
    except Exception:  # noqa: BLE001
        pass
    registry.mark_cancelled(job_id)
    return {"job_id": job_id, "status": "cancelled"}


_STRIP_ROUTES = frozenset(
    {
        ("POST", "/generate"),
        ("GET", "/health"),
        ("GET", "/{job_id}"),
        ("GET", "/{job_id}/status"),
    }
)


_LEGACY_COMPAT_PATHS = frozenset({"/generate", "/{job_id}", "/{job_id}/status"})


def _strip_child_routes(router) -> int:
    removed = 0
    kept = []
    for route in router.routes:
        methods = getattr(route, "methods", None) or set()
        path = getattr(route, "path", None)
        if path and any((method, path) in _STRIP_ROUTES for method in methods):
            removed += 1
            continue
        kept.append(route)
    if removed:
        router.routes[:] = kept
    return removed


def install_media_routes(app) -> None:
    import logging

    logger = logging.getLogger("sglang_h3_media_api")
    removed_containers = []
    removed_children = 0
    kept_top = []
    for route in list(app.router.routes):
        inner = getattr(route, "original_router", None)
        if inner is not None and hasattr(inner, "routes"):
            paths = {getattr(r, "path", None) for r in inner.routes}
            if paths and _LEGACY_COMPAT_PATHS.issubset(paths):
                removed_containers.append(
                    sorted(p for p in paths if p in _LEGACY_COMPAT_PATHS)
                )
                continue
            removed_children += _strip_child_routes(inner)
            kept_top.append(route)
            continue
        kept_top.append(route)
    app.router.routes[:] = kept_top
    app.include_router(media_router)
    logger.warning(
        "h3_media_api installed: removed legacy containers=%s stripped child routes=%d "
        "media routes appended",
        removed_containers,
        removed_children,
    )


def wrap_create_app(original_fn, server_args):
    app = original_fn(server_args)
    install_media_routes(app)
    return app
