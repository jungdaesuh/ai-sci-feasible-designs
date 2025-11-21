"""Phase 0 bootstrap helper cited by docs/MASTER_PLAN_AI_SCIENTIST.md:85-110.

Per the bootstrap checklist, this script verifies that the Python/NetCDF/VMEC
toolchain can execute a tiny forward-model demo and dumps the captured metrics
plus provenance metadata into `reports/bootstrap/`."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np

from ai_scientist import tools


def _sample_boundary(
    repo_root: Path,
) -> dict[str, list[list[float] | None] | int | bool | None]:
    boundary_path = (
        repo_root
        / "constellaration"
        / "hugging_face_competition"
        / "inputs"
        / "boundary.json"
    )
    with boundary_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {key: value for key, value in raw.items() if value is not None}


def _json_default(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Type {type(obj).__name__} not JSON serializable")


def _capture_git_sha(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tools.clear_evaluation_cache()
    evaluation = tools.evaluate_p1(
        _sample_boundary(repo_root), stage="screen", use_cache=False
    )
    report_dir = repo_root / "reports" / "bootstrap"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "phase0_bootstrap_metrics.json"

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _capture_git_sha(repo_root),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "netcdf_version": netCDF4.__version__,
        "evaluation": evaluation,
        "cache_stats": tools.get_cache_stats("screen"),
    }

    report_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(f"Wrote bootstrap metrics to {report_path}")


if __name__ == "__main__":
    main()
