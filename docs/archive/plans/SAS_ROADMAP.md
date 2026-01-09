# Stellarator AI Scientist (SAS): Research-Grade Autonomous System Roadmap

## Context & References (Self-Contained)

**ConStellaration Challenge** ([`ConStellaration Fusion Challenge_ Benchmarks and Solution Strategies.pdf`](ConStellaration Fusion Challenge\_ Benchmarks and Solution Strategies.pdf)):

- **P1 Geometric**: min max elongation E; eq const A=4.0, avg δ=-0.6, ι_edge/Nfp=0.3. Baseline: ALM+NGOpt E=1.27, score=0.969.
- **P2 Simple QI**: min Δ_QI / L_grad / Nfp; ineq A≤A_max, E≤E_max, M≤M_max. Baseline: ALM+NGOpt obj=8.61, score=0.431.
- **P3 MHD-stable QI**: Pareto min A, max L*grad; constr Δ_QI≤ε, W≥0, ⟨χ*∇r⟩≤C_max. Baseline: ALM+NGOpt sweep fixed-A, HV=130 (4pts).
- Dataset: ~160k QI-like boundaries+VMEC metrics (HF: `proxima-fusion/constellaration`).

**Codebase** ([`constellaration/src`](src/constellaration/)):

- Forward: [`forward_model.forward_model()`](src/constellaration/forward_model.py) → metrics (A,E,δ,ι/Nfp,M,Δ_QI,L_grad,W,C) via VMEC++/Boozer.
- Pipeline: [`p1_fresh.py`](src/constellaration/pipeline/p1_fresh.py): init ellipse + δ-proj (FD line search r_cos modes (2,0),(2,±1),(1,±1)) + feas eval (low/from/high fid).
- Opt: [`augmented_lagrangian_runner.py`](src/constellaration/optimization/augmented_lagrangian_runner.py) (NGOpt+ALM pen); [`scipy_minimize_runner.py`](src/constellaration/optimization/scipy_minimize_runner.py).
- Data gen: notebooks/generative_model (PCA+GMM+MCMC sampler).

**AI Scientist Inspirations**:

- **Kosmos** ([`2511.02824v2.pdf`](2511.02824v2.pdf)): WorldModel coords parallel analysis/lit agents → discoveries (traces/cites).
- **Jr. AI Scientist** ([`2511.04583v1.pdf`](2511.04583v1.pdf)): Extend baseline code: idea gen/impl/experiment (3 stages: impl/improve/ablate)/write/reflect.
- **SciAgent** ([`2511.08151v1.pdf`](2511.08151v1.pdf)): Hierarchical coord/worker/sub-agents adaptive pipelines.
- Others: MADD multi-agent drug disc.

**Targets**: P1 E<1.20; P2 obj<8.0; P3 HV>150 (10-pt Pareto). Compute: baseline days(96CPU) → hours(GPU surrogate+parallel VMEC).

## High-Level Architecture

```
CoordinatorAgent (LLM classify P1/P2/P3 → route BenchmarkWorker)
├── BenchmarkWorker (P1/P2/P3: template p1_fresh.py)
│   └── SubAgents (shared):
│       ├── IdeaGenerator: LLM+surrogate (dataset PCA/GMM/MCMC perturb/cond-gen/physics mods)
│       ├── Experimenter: Nevergrad (NGOpt/NSGA-II low-fid surrogate → topK VMEC refine δ-proj+trust-constr)
│       ├── Evaluator: feas (|constr-rel|<tol), obj/HV vs baselines
│       └── Reflector: LLM analyze fails → select/refine
└── WorldModel (YAML/JSON repo: shapes/metrics/plots/lit/traces/provenance)
Tools: forward_model, HF dataset, Nevergrad/scipy, Torch/Sklearn surrogates, MLflow log, Ray parallel.
CLI: sas_orchestrate.py --benchmark P1 --cycles 20 --target E<1.20
```

## Development Phases

1. **Bootstrap (One-Time, ~hours)**: Train surrogates/dataset tools.
2. **Discovery Cycles (20 cycles, ~hours/day)**: Hybrid ML-Physics loop.
3. **Aggregate/Report**: Pareto, plots, paper (Jr.AI style).

## Per-Benchmark Plans

### P1 Geometric

| Step       | Agent/Tool                         | Details                             | Code                          |
| ---------- | ---------------------------------- | ----------------------------------- | ----------------------------- |
| Pop        | IdeaGen+Sampler                    | Dataset feas subset + perturb modes | `datasets/sampler.py`         |
| Screen     | Exp (NGOpt low-fid NN/GP E/constr) | Trust-constr budget=50              | `surrogates/nn_surrogate.py`  |
| Refine     | δ-proj+scipy high-fid              | 20 its                              | `p1_sas.py` (extend p1_fresh) |
| **Target** | E<1.20                             |                                     |                               |

### P2 Simple QI

| Step       | Agent/Tool                         | Details                        | Code                    |
| ---------- | ---------------------------------- | ------------------------------ | ----------------------- |
| Pop        | IdeaGen+Sampler                    | QI subset low Δ_QI high L_grad | `p2_sas.py`             |
| Screen     | ALM+NGOpt surrogate obj/constr pen |                                | `opt/aug_lag_runner.py` |
| Refine     | High-fid VMEC                      |                                |                         |
| **Target** | obj<8.0                            |                                |                         |

### P3 MHD-Stable QI

| Step       | Agent/Tool                         | Details          | Code                   |
| ---------- | ---------------------------------- | ---------------- | ---------------------- |
| Pop        | IdeaGen+Sampler                    | QI-stable subset | `p3_sas.py`            |
| MultiObj   | NSGA-II surrogate multi-obj/constr | pop=100 gen=50   | `opt/nsga_runner.py`   |
| Refine     | High-fid cull Pareto               | HV calc          | `utils/hypervolume.py` |
| **Target** | HV>150 (10pts)                     |                  |                        |

## Codebase Gaps & Fixes

| Gap            | Priority | Fix                        | File                                            |
| -------------- | -------- | -------------------------- | ----------------------------------------------- |
| No HF dataset  | High     | Loader/filter/PCA/GMM/MCMC | `datasets/sampler.py` (from notebooks)          |
| No surrogates  | High     | Torch NN/GP reg obj/constr | `surrogates/nn_surrogate.py`, `gp_surrogate.py` |
| No NSGA        | High     | Nevergrad NSGA-II          | `opt/nsga_runner.py`                            |
| No P2/P3       | High     | Template p1_fresh          | `p2_sas.py`, `p3_sas.py`                        |
| No HV          | Med      | Hypervolume impl           | `utils/hypervolume.py`                          |
| No orchestrate | High     | LLM agents + CLI cycles    | `ai_scientist/agents/*`, `sas_orchestrate.py`   |
| No Docker      | Low      | Prod image                 | `Dockerfile.sas`                                |

## Validation & Production

- **Soundness**: Surrogates R²>0.9/0.8; hybrid ML(approx global)+physics(exact local); ALM feas-first; repro baselines.
- **Tests**: Unit (pipelines/feas); E2E (baselines); CV surrogates.
- **Deploy**: Docker (constell+nevergrad+torch+hf+jax); CLI cycles; MLflow log.
