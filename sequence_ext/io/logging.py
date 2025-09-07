# sequence_ext/io/logging.py
"""
Centralized, production-grade logging for QUASAR.

- Human or JSON logs (env: QUASAR_LOG_JSON=1)
- File sink with rotation/retention (env: QUASAR_LOG_FILE, QUASAR_LOG_RETENTION)
- Log level via env (QUASAR_LOG_LEVEL=DEBUG|INFO|...)
- Context via contextvars (run_id, component) with Loguru .patch()
- Stdlib logging routed into Loguru

Compatible with Loguru >=0.6 (no 'patcher' kw in .add()).
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Optional, Any, Dict
from contextvars import ContextVar

from loguru import logger as _base_logger

# -----------------------------------------------------------------------------
# Context fields
# -----------------------------------------------------------------------------
_run_id_ctx: ContextVar[Optional[str]] = ContextVar("_run_id_ctx", default=None)
_component_ctx: ContextVar[Optional[str]] = ContextVar("_component_ctx", default=None)


def current_context() -> Dict[str, Any]:
    return {"run_id": _run_id_ctx.get(), "component": _component_ctx.get()}


class _Contextualize:
    """Inject contextvars into each record via Loguru .patch()."""

    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        extra = record.setdefault("extra", {})
        extra.setdefault("run_id", _run_id_ctx.get())
        extra.setdefault("component", _component_ctx.get())
        return record


# We keep a module-level handle to the *current* logger (possibly patched).
_log = _base_logger


def _get_logger():
    return _log


# -----------------------------------------------------------------------------
# Stdlib logging interception into Loguru
# -----------------------------------------------------------------------------
class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        log = _get_logger()
        try:
            level = log.level(record.levelname).name
        except (ValueError, AttributeError):
            level = record.levelno

        # Find caller frame depth so Loguru reports correct source
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        log.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _setup_stdlib_bridge() -> None:
    logging.root.handlers = []
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(handlers=[_InterceptHandler()], level=logging.NOTSET)
    for name in list(logging.root.manager.loggerDict.keys()):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True


# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------
_HUMAN_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "run_id={extra[run_id]} component={extra[component]} - <level>{message}</level>"
)

_JSON_FMT = (
    "{{"
    "\"ts\":\"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}\","
    "\"level\":\"{level}\","
    "\"name\":\"{name}\","
    "\"func\":\"{function}\","
    "\"line\":{line},"
    "\"run_id\":\"{extra[run_id]}\","
    "\"component\":\"{extra[component]}\","
    "\"message\":{message!r}"
    "}}"
)


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v not in ("0", "false", "False", "no", "No", "")


def _env_value(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
def reconfigure_logging(
    *,
    level: Optional[str] = None,
    json: Optional[bool] = None,
    file_path: Optional[str] = None,
    retention: Optional[str] = None,
) -> None:
    """
    Reconfigure the global logger.

    - level: env QUASAR_LOG_LEVEL or "INFO"
    - json: env QUASAR_LOG_JSON (bool)
    - file_path: env QUASAR_LOG_FILE
    - retention: env QUASAR_LOG_RETENTION or "7 days"
    """
    global _log

    lvl = (level or _env_value("QUASAR_LOG_LEVEL", "INFO")).upper()
    as_json = (json if json is not None else _env_flag("QUASAR_LOG_JSON", False))
    file_path = file_path or _env_value("QUASAR_LOG_FILE", None)
    retention = retention or _env_value("QUASAR_LOG_RETENTION", "7 days")

    # Start from the base logger, apply a patcher that injects contextvars
    patched = _base_logger.patch(_Contextualize())
    patched.remove()

    fmt = _JSON_FMT if as_json else _HUMAN_FMT
    patched.add(
        sys.stderr,
        format=fmt,
        level=lvl,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )

    if file_path:
        patched.add(
            file_path,
            format=fmt,
            level=lvl,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            rotation="50 MB",
            retention=retention,
            compression="zip",
        )

    _log = patched
    _setup_stdlib_bridge()


# Apply defaults at import
reconfigure_logging()

# Public logger bound with current context (can be re-bound by callers)
logger = _get_logger().bind(run_id=_run_id_ctx.get(), component=_component_ctx.get())


# -----------------------------------------------------------------------------
# Context helpers
# -----------------------------------------------------------------------------
class set_run_id:
    """Context manager to set a run_id for all logs within the block."""

    def __init__(self, run_id: Optional[str]) -> None:
        self._run_id = run_id
        self._token = None

    def __enter__(self) -> "set_run_id":
        self._token = _run_id_ctx.set(self._run_id)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _run_id_ctx.reset(self._token)


class bind_component:
    """Context manager to tag logs with a component name."""

    def __init__(self, component: Optional[str]) -> None:
        self._component = component
        self._token = None

    def __enter__(self) -> "bind_component":
        self._token = _component_ctx.set(self._component)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _component_ctx.reset(self._token)


def with_context(**extras: Any):
    """Return a logger bound with additional fields."""
    return _get_logger().bind(**extras)


__all__ = [
    "logger",
    "reconfigure_logging",
    "set_run_id",
    "bind_component",
    "with_context",
    "current_context",
]
