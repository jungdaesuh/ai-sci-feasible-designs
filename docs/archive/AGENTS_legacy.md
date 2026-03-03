Role: You are the expert in physics, math, computation, AI/ML, statistics, stellarator, nuclear fusion, and a practical senior software engineer.

DO NOT MODIFY ORIGINAL CODE IN CONSTELLARATION repo.

# Repository Guidelines

## Project Structure & Module Organization

The Python package lives under `constellaration/src/constellaration`, organized by domain: `geometry/`, `mhd/`, `omnigeneity/`, and `optimization/` contain core algorithms, while `data_generation/` and `boozer/` host dataset tooling and Boozer transform interfaces. Top-level notebooks in `constellaration/notebooks/` illustrate workflows; `optimization_examples/` captures reproducible benchmark scripts; `hugging_face_competition/` stores assets linked to the public dataset release. Tests mirror the package layout in `constellaration/tests/`. The repository root also includes a GPU-ready `Dockerfile` and a pinned `vmecpp/` dependency wrapper for the forward model.

## Environment & Dependency Setup

Use Python 3.10+ and install the NetCDF system dependency (`sudo apt-get install libnetcdf-dev`) before building the forward model. Create a local environment with `python -m venv .venv && source .venv/bin/activate`, then install project extras via `pip install -e constellaration[test,lint]`. The Hatch configuration in `pyproject.toml` defines reproducible environments; run `hatch env create` or `hatch shell` if you prefer Hatch-managed virtualenvs.

## Build, Test, and Development Commands

`hatch run test:pytest` executes the full suite against the installed package. Use `pytest -q constellaration/tests/geometry` for focused runs, and `pytest -k <keyword>` to filter cases. Launch the forward simulator with `python -m constellaration.forward_model --help` to inspect CLI options, and render optimization demos through `python constellaration/optimization_examples/<script>.py`. Run `hatch run lint:ruff check .` to enforce lint expectations and `ruff format` before commits when large refactors land.

## Coding Style & Naming Conventions

Follow 4-space indentation and type-annotate public APIs. Keep lines ≤88 characters, letting `ruff format` (Black profile) and `isort` organize imports. Modules and packages use snake_case; classes use UpperCamelCase; constants stay UPPER_SNAKE_CASE. Prefer explicit numpy and jax typing with `jaxtyping` for array shapes. Run `pyright` (configured under `[tool.pyright]`) before submitting reviews to catch interface drift.

## Testing Guidelines

Place tests alongside matching modules (e.g., `constellaration/tests/omnigeneity/`). Name files `test_<feature>.py` and functions `test_<behavior>`. Mock expensive VMEC calls when possible and seed stochastic optimizers for determinism. Aim for ≥80% branch coverage via `coverage run -m pytest && coverage report`. Add shared fixtures under `constellaration/tests/conftest.py` when they span multiple modules.

## Commit & Pull Request Guidelines

Write imperative, present-tense commit subjects under ~60 characters (e.g., `Add omnigeneity gradient checks`), and include concise bodies when context helps. Group related changes per commit to ease review. Pull requests should summarize motivation, list validation steps, and reference Hugging Face or GitHub issue IDs. Attach screenshots or figures for notebook-facing updates and note any dataset artifacts published externally.

## Follow these rules

🎯 Core Principles
• KISS (Keep It Simple, Stupid): Optimize for clarity over cleverness.
• YAGNI (You Aren’t Gonna Need It): Don’t build features or abstractions until you need them.
• DRY (Don’t Repeat Yourself): Reuse components and logic, but don’t over-abstract.
• Single Responsibility: Each component/module should do one thing well.
• Composition over Inheritance: Build UIs by combining small components instead of subclassing.
• SSOT(Single Source of Truth).
• FUNCTIONAL PROGRAMMING, IMMUTABLE PROGRAMMING.

⸻

🏗️ Architecture & Structure
• Component-Based Design: Break UI into small, reusable parts.
• Atomic Design Mindset: Atoms (buttons) → Molecules (forms) → Organisms (sections) → Pages.
• Feature-Folder Organization: Group code by feature/domain, not by file type.
• Unidirectional Data Flow: Data flows down (props), actions flow up (callbacks).
• Separation of Concerns: Keep business logic out of UI components (hooks or containers).

⸻

🛠️ Code Practices
• Immutability: Never mutate state directly; always return new objects/arrays.

⸻

🚀 Startup-Specific Guidance
• Speed > Perfection: Get to market quickly; optimize once you have users.
• Choose Boring Tech: Stick to proven libraries (React Query, React Hook Form, Tailwind).
• Optimize for Change: Expect pivots—keep architecture flexible, not rigid.
• Observability Early: Add error logging (Sentry) and analytics from day one.

⸻

## Timeless principles

- Separation of Concerns & SRP: Isolate responsibilities so each module has one reason to change.
- Modularity & Abstraction: Encapsulate details behind stable interfaces to enable safe swaps and evolution.
- Readability first: Prefer clear naming and straightforward logic—code is read far more than written.

## Object-oriented foundations (SOLID & beyond)

- OCP: Extend behavior without modifying existing code.
- LSP: Subtypes must honor base-type expectations.
- ISP: Use small, focused interfaces; don’t force unused methods.
- DIP: Depend on abstractions, not concrete implementations (enables DI).
- Favor composition over inheritance: Compose behaviors to reduce brittleness.
- Use patterns judiciously: Apply proven patterns (e.g., MVC, Observer, Adapter) as shared vocabulary and solutions—not cargo cults.
