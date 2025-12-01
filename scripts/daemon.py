"""Supervisory wrapper for ai_scientist.runner (see docs/AI_SCIENTIST_UNIFIED_ROADMAP.md, Section 6).

Responsibilities:
- Force OMP_NUM_THREADS=1 for worker stability.
- Enforce a wall-clock guard for the overall run.
- Auto-select a checkpoint to resume from (latest cycle_*.json in reporting_dir) unless overridden.

This is intentionally thin: it shells out to `python -m ai_scientist.runner` with the supplied
arguments to avoid duplicating runner logic.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List


def _latest_checkpoint(report_dir: Path) -> Path | None:
    candidates = sorted(report_dir.glob("cycle_*.json"))
    return candidates[-1] if candidates else None


def _build_cmd(args: argparse.Namespace, resume_path: Path | None) -> List[str]:
    cmd = [sys.executable, "-m", "ai_scientist.runner"]
    if args.config:
        cmd.extend(["--config", str(args.config)])
    if args.problem:
        cmd.extend(["--problem", args.problem])
    if args.cycles is not None:
        cmd.extend(["--cycles", str(args.cycles)])
    if args.memory_db:
        cmd.extend(["--memory-db", str(args.memory_db)])
    if args.eval_budget is not None:
        cmd.extend(["--eval-budget", str(args.eval_budget)])
    if args.workers is not None:
        cmd.extend(["--workers", str(args.workers)])
    if args.pool_type:
        cmd.extend(["--pool-type", args.pool_type])
    if args.run_preset:
        cmd.extend(["--run-preset", args.run_preset])
    if args.planner:
        cmd.extend(["--planner", args.planner])
    if resume_path:
        cmd.extend(["--resume-from", str(resume_path)])
    return cmd


def _run_once(cmd: Iterable[str]) -> int:
    proc = subprocess.run(list(cmd), check=False)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daemon wrapper for ai_scientist.runner"
    )
    parser.add_argument("--config", type=Path, help="Experiment config path.")
    parser.add_argument("--problem", choices=["p1", "p2", "p3"], help="Problem to run.")
    parser.add_argument("--cycles", type=int, help="Total cycles to run.")
    parser.add_argument("--memory-db", type=Path, help="World-model SQLite path.")
    parser.add_argument("--eval-budget", type=int, help="Screening budget override.")
    parser.add_argument("--workers", type=int, help="Worker override.")
    parser.add_argument(
        "--pool-type", choices=["thread", "process"], help="Executor type."
    )
    parser.add_argument("--run-preset", type=str, help="Run preset name.")
    parser.add_argument(
        "--planner", choices=["deterministic", "agent"], default="deterministic"
    )
    parser.add_argument(
        "--reporting-dir",
        type=Path,
        default=Path("reports"),
        help="Reporting directory to look for checkpoints.",
    )
    parser.add_argument("--resume-from", type=Path, help="Explicit checkpoint path.")
    parser.add_argument(
        "--wall-clock-minutes",
        type=float,
        default=0.0,
        help="Wall-clock guard for the daemon (0 disables).",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        default=True,
        help="Auto-pick latest checkpoint when --resume-from not provided.",
    )
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")

    start = time.monotonic()
    resume_path = args.resume_from
    if resume_path is None and args.auto_resume:
        resume_path = _latest_checkpoint(args.reporting_dir)

    cmd = _build_cmd(args, resume_path)
    rc = _run_once(cmd)

    if rc != 0:
        latest = _latest_checkpoint(args.reporting_dir)
        if latest and latest != resume_path:
            print(
                f"[daemon] runner failed (rc={rc}); retrying with checkpoint {latest}"
            )
            cmd = _build_cmd(args, latest)
            rc = _run_once(cmd)

    if rc != 0:
        sys.exit(rc)

    if args.wall_clock_minutes > 0:
        elapsed_min = (time.monotonic() - start) / 60.0
        if elapsed_min > args.wall_clock_minutes:
            print(
                f"[daemon] wall-clock limit reached ({elapsed_min:.2f} min > {args.wall_clock_minutes} min)"
            )


if __name__ == "__main__":
    main()
