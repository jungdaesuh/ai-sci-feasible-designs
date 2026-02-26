"""Deterministic runtime failure taxonomy for evaluation telemetry."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FailureClassification:
    failure_label: str
    failure_source: str
    failure_signature: str | None
    normalized_error: str | None


_SPACE_RE = re.compile(r"\s+")

_ERROR_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "vmec_jacobian_bad",
        "vmec",
        re.compile(r"jacobian.*bad|bad.*jacobian", re.IGNORECASE),
    ),
    (
        "vmec_linear_solve_failure",
        "vmec",
        re.compile(r"singular|factorization|linear solve", re.IGNORECASE),
    ),
    (
        "qi_not_enough_crossings",
        "qi",
        re.compile(r"not enough crossings", re.IGNORECASE),
    ),
    (
        "qi_computation_failure",
        "qi",
        re.compile(r"\bqi\b|boozer|quasi[-\s]?isodynamic", re.IGNORECASE),
    ),
    (
        "vmec_not_converged",
        "vmec",
        re.compile(
            r"(did not|failed to|unable to)\s+converg|non[-\s]?converg", re.IGNORECASE
        ),
    ),
    (
        "invalid_boundary",
        "geometry",
        re.compile(r"invalid boundary|boundary parameters", re.IGNORECASE),
    ),
    (
        "evaluation_timeout",
        "runtime",
        re.compile(r"timeout|timed out|deadline", re.IGNORECASE),
    ),
    (
        "resource_exhausted",
        "runtime",
        re.compile(r"oom|out of memory|resource exhausted", re.IGNORECASE),
    ),
    (
        "physics_runtime_error",
        "runtime",
        re.compile(r"physics evaluation failed", re.IGNORECASE),
    ),
)


def _normalize_error(error_message: str) -> str:
    return _SPACE_RE.sub(" ", error_message.strip().lower())


def _coerce_error_message(error_message: Any) -> str:
    return error_message if isinstance(error_message, str) else str(error_message)


def classify_failure(error_message: Any | None) -> FailureClassification:
    """Return deterministic labels/signatures for runtime failures."""
    if error_message is None:
        return FailureClassification(
            failure_label="ok",
            failure_source="none",
            failure_signature=None,
            normalized_error=None,
        )

    normalized = _normalize_error(_coerce_error_message(error_message))
    label = "unknown_runtime_error"
    source = "runtime"

    for candidate_label, candidate_source, pattern in _ERROR_PATTERNS:
        if pattern.search(normalized):
            label = candidate_label
            source = candidate_source
            break

    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    signature = f"{label}:{digest}"
    return FailureClassification(
        failure_label=label,
        failure_source=source,
        failure_signature=signature,
        normalized_error=normalized,
    )
