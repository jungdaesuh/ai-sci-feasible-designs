# Stellarator AI Scientist (SAS) Detailed Implementation TODO Checklist

**Self-Contained Context** (All refs/code links herein; only needs [`SAS_ROADMAP.md`](SAS_ROADMAP.md)):

- **Challenge**: P1 min E (eq A=4,δ=-0.6,ι/Nfp=0.3); P2 min Δ_QI/L_grad/Nfp (ineq A≤max,E≤max,M≤max); P3 Pareto min A max L_grad (constr Δ_QI≤ε,W≥0,C≤C_max). Baselines: P1 E=1.27 score=0.969; P2 obj=8.61 score=0.431; P3 HV=130 (4pts). **Targets**: P1 E<1.20; P2<8.0; P3>150 (10pts).
- **Codebase**: [`forward_model.forward_model()`](constellaration/src/constellaration/forward_model.py) metrics; [`p1_fresh.py`](constellaration/src/constellaration/pipeline/p1_fresh.py) init+δ-proj+feas; opt/ ALM/NGOpt/scipy.
- **Dataset**: HF `proxima-fusion/constellaration` ~160k boundaries/metrics.
- **Arch**: Coord→BenchmarkWorker→SubAgents (IdeaGen/Exp/Eval/Refl); WorldModel YAML/JSON.
- **Tools**: VMEC++/forward, HF datasets, Nevergrad/scipy, Torch/Sklearn, Ray parallel, MLflow log.
- **CLI**: `sas_orchestrate.py --benchmark P1 --cycles 20`.

## Phase 1: Bootstrap (One-Time ~hours)

- [ ] **Dataset Tools** `datasets/sampler.py`
  - [ ] `from datasets import load_dataset; ds = load_dataset("proxima-fusion/constellaration", split="train")`
  - [ ] Filter feas subsets: P1 `ds.filter(lambda ex: abs(ex['aspect_ratio']-4)<0.1 & abs(ex['average_triangularity']+0.6)<0.1 & abs(ex['edge_rotational_transform_over_nfp']-0.3)<0.1)`
  - [ ] P2/P3 analogous constr.
  - [ ] PCA: `from sklearn.decomposition import PCA; pca = PCA(n_components=10).fit(coeffs)` (80d→10d).
  - [ ] GMM: `from sklearn.mixture import GaussianMixture; gmm = GaussianMixture(n_components=50).fit(pca_feats[feas_mask])`
  - [ ] MCMC sample posterior (notebook/generative_model → func): sample low-obj/high-HV biased.
- [ ] **Surrogates** `surrogates/train.py`
  - [ ] NN: Torch `nn.Sequential(Linear(80,128),ReLU,128,ReLU,Linear(out=objs+constrs))` MSE+feas BCE; train/val split; save `p1_surrogate.pt`.
  - [ ] GP: `from sklearn.gaussian_process import GaussianProcessRegressor; gp.fit(X=coeffs,y=objs)` uncert.
  - [ ] Per benchmark/cond target subsets.
- [ ] **Test**: Unit `test_sampler.py`, `test_surrogates.py` (R²>0.9 train/0.8 OOB).

## Phase 2: Core Agents & Orchestrator (~days)

- [ ] **Agents Dir** `ai_scientist/agents/`
  - [ ] `coord_agent.py`: LLMChain prompt "Classify problem: [desc] → P1/P2/P3" → route BenchmarkWorker.
  - [ ] `idea_gen.py`: Prompt "Propose 10 Fourier perturbations for [P?]: target low obj/high feas; inspire dataset top/physics (wells m=3 smooth)" + sampler(100 near best/target).
  - [ ] `experimenter.py`: Nevergrad NGOpt/NSGA-II (budget=50/100 low-fid surrogate obj/constr ALM pen) → topK VMEC refine (`pX_sas.py` δ-proj+trust-constr high-fid).
  - [ ] `evaluator.py`: `compute_feasibility(metrics,tol)`; P3 HV `from emoa import hypervolume`.
  - [ ] `reflector.py`: Prompt "Analyze top5 fails: non-conv→retry init; infeas→subspace; sub-obj→perturb" → select 1-3.
- [ ] **WorldModel** `worldmodel.py`: YAML CRUD (shapes: list[dict coeffs]; metrics/plots paths; feas/obj/HV; traces[list traj_id]).
- [ ] **Benchmark Pipelines** `pipeline/`
  - [ ] `base_pipeline.py`: Modularize p1_fresh (init/projs/feas).
  - [ ] `p1_sas.py`: Extend base + NGOpt low → refine high.
  - [ ] `p2_sas.py`: Template P1 + QI obj/constr ALM.
  - [ ] `p3_sas.py`: NSGA + surrogate callback + HV.
- [ ] **CLI** `sas_orchestrate.py`
  ```
  for cycle in range(N):
    coord → ideas = idea_gen(world)
    exps = parallel_experimenter(ideas, low_fid_surrogate)
    top = evaluator(exps); refine_high = experimenter(topK)
    best = reflector(refine); world.update(best)
  aggregate_pareto(world); generate_report(world)
  ```

## Phase 3: Integration & Production (~days)

- [ ] **Opt Extensions** `optimization/`
  - [ ] `nsga_runner.py`: `ng.opt.NGOpt` NSGA-II multi-obj.
- [ ] **Utils** `utils/hypervolume.py`: EMOA HV calc.
- [ ] **Tests** `tests/`
  - [ ] Unit pipelines/feas/surrogates.
  - [ ] E2E repro baselines P1/P2/P3.
- [ ] **Docker** `Dockerfile.sas`
  ```
  FROM python:3.11; pip install constellaration[all] nevergrad torch scikit-learn datasets jax jaxlib ray mlflow
  COPY . /sas; WORKDIR /sas; CMD ["python", "sas_orchestrate.py"]
  ```
- [ ] **Logging**: MLflow track shapes/metrics/HV/traj provenance.

## Phase 4: Validation & Iterate

- [ ] Repro baselines scores/time.
- [ ] Run P1 20 cycles → E<1.20 feas.
- [ ] P2/P3 analogous.
- [ ] Ablate: no-surrogate (physics-only slow), no-ML (dataset random poor).

**Done**: ✅ All checkboxes → Prod SAS beats baselines autonomously.
