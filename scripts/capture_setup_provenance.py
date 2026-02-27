#!/usr/bin/env python
# ruff: noqa: E402
"""Capture runtime/setup provenance under artifacts/setup for reproducibility."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MODEL_PROVIDER",
    "AI_SCIENTIST_INSTRUCT_MODEL",
    "AI_SCIENTIST_THINKING_MODEL",
)

_UNAVAILABLE = "unavailable"
_UNKNOWN = "unknown"


@dataclass(frozen=True)
class SetupProvenance:
    captured_at_utc: str
    hostname: str
    platform: str
    machine: str
    python_executable: str
    python_version: str
    pip_version: str
    uv_version: str
    docker_version: str
    git_sha: str
    git_branch: str
    constellaration_sha: str
    vmecpp_sha: str
    vmec_mode: str
    vmec_mode_reason: str
    env: dict[str, str]
    capture_command: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_command(command: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
    cwd_str = str(cwd) if cwd is not None else None
    try:
        completed = subprocess.run(
            command,
            cwd=cwd_str,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return (127, "", "")
    return (completed.returncode, completed.stdout, completed.stderr)


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _command_version(command: list[str], *, cwd: Path | None = None) -> str:
    code, stdout, stderr = _run_command(command, cwd=cwd)
    if code != 0:
        return _UNAVAILABLE
    line = _first_non_empty_line(stdout) or _first_non_empty_line(stderr)
    return line if line else _UNAVAILABLE


def _git_output(repo: Path, args: list[str]) -> str:
    code, stdout, _stderr = _run_command(["git", "-C", str(repo), *args])
    if code != 0:
        return _UNKNOWN
    line = _first_non_empty_line(stdout)
    return line if line else _UNKNOWN


def _select_vmec_mode(requested_mode: str, *, docker_available: bool) -> tuple[str, str]:
    if requested_mode == "docker":
        return ("docker", "selected by --vmec-mode=docker")
    if requested_mode == "native":
        return ("native", "selected by --vmec-mode=native")
    if docker_available:
        return ("docker", "auto-selected because docker CLI is available")
    return ("native", "auto-fallback because docker CLI is unavailable")


def _capture_command(repo_root: Path) -> str:
    script_path = Path(__file__).resolve()
    try:
        script_display = str(script_path.relative_to(repo_root))
    except ValueError:
        script_display = str(script_path)
    command = [sys.executable, script_display, *sys.argv[1:]]
    return " ".join(shlex.quote(part) for part in command)


def _env_snapshot() -> dict[str, str]:
    return {key: os.environ.get(key, "") for key in _ENV_KEYS}


def _collect_provenance(repo_root: Path, vmec_mode_request: str) -> SetupProvenance:
    docker_version = _command_version(["docker", "--version"])
    docker_available = docker_version != _UNAVAILABLE
    vmec_mode, vmec_mode_reason = _select_vmec_mode(
        vmec_mode_request,
        docker_available=docker_available,
    )
    return SetupProvenance(
        captured_at_utc=_utc_now_iso(),
        hostname=platform.node(),
        platform=platform.platform(),
        machine=platform.machine(),
        python_executable=sys.executable,
        python_version=_command_version([sys.executable, "--version"]),
        pip_version=_command_version([sys.executable, "-m", "pip", "--version"]),
        uv_version=_command_version(["uv", "--version"]),
        docker_version=docker_version,
        git_sha=_git_output(repo_root, ["rev-parse", "HEAD"]),
        git_branch=_git_output(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        constellaration_sha=_git_output(
            repo_root / "constellaration",
            ["rev-parse", "HEAD"],
        ),
        vmecpp_sha=_git_output(repo_root / "vmecpp", ["rev-parse", "HEAD"]),
        vmec_mode=vmec_mode,
        vmec_mode_reason=vmec_mode_reason,
        env=_env_snapshot(),
        capture_command=_capture_command(repo_root),
    )


def _versions_lines(provenance: SetupProvenance) -> list[str]:
    base_lines = [
        f"captured_at_utc={provenance.captured_at_utc}",
        f"hostname={provenance.hostname}",
        f"platform={provenance.platform}",
        f"machine={provenance.machine}",
        f"python_executable={provenance.python_executable}",
        f"python_version={provenance.python_version}",
        f"pip_version={provenance.pip_version}",
        f"uv_version={provenance.uv_version}",
        f"docker_version={provenance.docker_version}",
        f"git_sha={provenance.git_sha}",
        f"git_branch={provenance.git_branch}",
        f"constellaration_sha={provenance.constellaration_sha}",
        f"vmecpp_sha={provenance.vmecpp_sha}",
        f"vmec_mode={provenance.vmec_mode}",
        f"vmec_mode_reason={provenance.vmec_mode_reason}",
    ]
    env_lines = [f"env_{key}={provenance.env.get(key, '')}" for key in _ENV_KEYS]
    return base_lines + env_lines


def _render_commands_md(provenance: SetupProvenance) -> str:
    lines = [
        "# Setup Commands",
        "",
        "## Captured command",
        "",
        "```bash",
        provenance.capture_command,
        "```",
        "",
        "## Version probe commands",
        "",
        "```bash",
        f"{shlex.quote(sys.executable)} --version",
        f"{shlex.quote(sys.executable)} -m pip --version",
        "uv --version",
        "docker --version",
        "git rev-parse HEAD",
        "git rev-parse --abbrev-ref HEAD",
        "git -C constellaration rev-parse HEAD",
        "git -C vmecpp rev-parse HEAD",
        "```",
        "",
        "## Setup bootstrap templates",
        "",
        "```bash",
        "python -m venv .venv && source .venv/bin/activate",
        "pip install -e \".[test,experiments]\"",
        "cp .env.example .env",
        "set -a; source .env; set +a",
        "```",
        "",
        "```bash",
        "uv sync --extra test --extra experiments",
        "cp .env.example .env",
        "set -a; source .env; set +a",
        "```",
        "",
        "```bash",
        "docker build -t ai-scientist .",
        "docker run --rm -v \"$PWD\":/app ai-scientist",
        "```",
        "",
    ]
    return "\n".join(lines)


def _render_vmec_path_md(provenance: SetupProvenance) -> str:
    lines = [
        "# VMEC Path Selection",
        "",
        f"- `selected_mode`: `{provenance.vmec_mode}`",
        f"- `reason`: {provenance.vmec_mode_reason}",
        "",
        "## Recommended execution",
    ]
    if provenance.vmec_mode == "docker":
        lines.extend(
            [
                "",
                "```bash",
                "docker build -t ai-scientist .",
                "docker run --rm -v \"$PWD\":/app ai-scientist",
                "```",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "```bash",
                "python -m ai_scientist.runner --config configs/experiment.yaml --problem p1",
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Raw probes",
            "",
            f"- `docker_version`: {provenance.docker_version}",
            f"- `constellaration_sha`: {provenance.constellaration_sha}",
            f"- `vmecpp_sha`: {provenance.vmecpp_sha}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifacts(output_dir: Path, provenance: SetupProvenance) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    versions_path = output_dir / "versions.txt"
    commands_path = output_dir / "commands.md"
    vmec_path = output_dir / "vmec_path.md"
    json_path = output_dir / "provenance.json"

    versions_path.write_text(
        "\n".join(_versions_lines(provenance)) + "\n",
        encoding="utf-8",
    )
    commands_path.write_text(_render_commands_md(provenance), encoding="utf-8")
    vmec_path.write_text(_render_vmec_path_md(provenance), encoding="utf-8")
    json_path.write_text(
        json.dumps(asdict(provenance), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return [versions_path, commands_path, vmec_path, json_path]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture runtime setup provenance under artifacts/setup.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/setup"),
        help="Output directory for setup provenance files.",
    )
    parser.add_argument(
        "--vmec-mode",
        choices=("auto", "native", "docker"),
        default="auto",
        help="VMEC path policy to record.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    provenance = _collect_provenance(repo_root, args.vmec_mode)
    created = _write_artifacts(args.output_dir, provenance)
    for path in created:
        print(path.as_posix())


if __name__ == "__main__":
    main()
