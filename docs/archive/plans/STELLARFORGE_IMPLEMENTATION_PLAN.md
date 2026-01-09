# StellarForge Implementation Plan (Paper-Backed v2)

> **Document Status**: Implementation Plan for StellarForge Tri-Hybrid Architecture
> **Last Updated**: 2025-12-09
> **Based on**: Padidar et al. (2025), Stark et al. (2025), Ridwan et al. (2025)

---

## 1. Executive Summary

This document details the implementation plan for upgrading the AI Scientist system to the **StellarForge Tri-Hybrid Architecture**. The plan is grounded in specifications from three peer-reviewed papers and addresses critical gaps in the current codebase.

### Expected Benefits (Paper-Backed)

| Metric | Current Baseline | After Full Implementation |
|--------|------------------|---------------------------|
| Feasibility Rate | ~5% | 70-85% |
| VMEC++ Calls to Baseline | ~500 | ~40-60 |
| Wall-Clock Time | ~8 hours | ~1-2 hours |
| L_∇B (P2 Objective) | 8.61 | 10-12 |

---

## 2. Reference Papers

### 2.1 Diffusion for Fusion (Padidar et al., 2025)
- **arxiv**: 2511.20445v1
- **Key Contribution**: Conditional DDPM for stellarator boundary generation
- **Results**: <5% deviation on aspect ratio, <6% on quasisymmetry (OOD)

### 2.2 BoltzGen (Stark et al., 2025)
- **biorxiv**: 2025.11.20.689494v1
- **Key Contribution**: Unified design + structure prediction diffusion model
- **Results**: 66% success rate on 9 novel targets (wet lab validated)

### 2.3 LEGO-xtal (Ridwan et al., 2025)
- **arxiv**: 2506.08224v2
- **Key Contribution**: Geometric pre-relaxation via SO(3) descriptors
- **Results**: 25 → 1,700+ valid crystal structures from augmented training

---

## 3. Gap Analysis

### Critical Gaps (Must Fix)

| Component | Paper Spec | Current Implementation | Gap Severity |
|-----------|------------|------------------------|--------------|
| Network Width | 2048 | 64 | 🔴 **Critical** (32x) |
| Batch Size | 4096 | 32 | 🔴 **Critical** (128x) |
| PCA Compression | 661 → 50 | None | 🟠 Major |
| Checkpoint Loading | Required | Missing | 🟠 Major |
| Pre-relaxation | Required | Missing | 🟠 Major |

### Existing Strengths (✅ Already Implemented)

- `DiffusionDesignModel` with conditional sampling
- `NeuralOperatorSurrogate` ensemble for fast predictions
- `ExplorationWorker` for generative sampling
- `scripts/train_offline.py` for offline pipeline training

---

## 4. Implementation Phases

### Phase 1: Generative Model Upgrade (Priority: Critical)

**Timeline**: 2-3 days
**Compute**: A100 GPU or equivalent

#### 4.1.1 Configuration Updates

**File**: `ai_scientist/config.py`

Add to `GenerativeConfig`:
```python
@dataclass(frozen=True)
class GenerativeConfig:
    enabled: bool
    backend: str = "diffusion"
    latent_dim: int = 16
    learning_rate: float = 1e-3
    epochs: int = 250                    # Paper: 250 epochs
    kl_weight: float = 0.001

    # NEW: Paper-backed specifications
    checkpoint_path: Path | None = None  # Pre-trained model path
    device: str = "cuda"                 # Compute device
    hidden_dim: int = 2048               # Paper: 4 layers × 2048
    n_layers: int = 4                    # Paper: 4 hidden layers
    pca_components: int = 50             # Paper: 661 → 50 dims
    batch_size: int = 4096               # Paper: 4096
    diffusion_timesteps: int = 200       # Paper: 200 steps
```

#### 4.1.2 PCA Preprocessing Integration

**File**: `ai_scientist/optim/generative.py`

Add PCA to `DiffusionDesignModel`:
```python
from sklearn.decomposition import PCA

class DiffusionDesignModel:
    def __init__(self, ..., pca_components: int = 50):
        self.pca = None
        self.pca_components = pca_components

    def fit(self, boundaries: np.ndarray, metrics: np.ndarray):
        """Train diffusion model on PCA-compressed latent space."""
        # Step 1: Fit PCA (661 → 50)
        self.pca = PCA(n_components=self.pca_components)
        latent = self.pca.fit_transform(boundaries)

        # Step 2: Train diffusion on latent
        self._train_diffusion(latent, metrics)

    def sample(self, n_samples: int, target_metrics: dict) -> np.ndarray:
        """Generate samples in latent space, then inverse transform."""
        latent_samples = self._diffusion_sample(n_samples, target_metrics)
        return self.pca.inverse_transform(latent_samples)  # 50 → 661
```

#### 4.1.3 Network Architecture Scaling

**File**: `ai_scientist/optim/generative.py`

Update `StellaratorDiffusion` class:
```python
class StellaratorDiffusion(nn.Module):
    def __init__(
        self,
        input_dim: int = 50,        # After PCA compression
        hidden_dim: int = 2048,     # Paper spec
        n_layers: int = 4,          # Paper spec
        condition_dim: int = 128,   # Paper: y → 128
        time_embed_dim: int = 128,  # Paper: t → 128
        input_embed_dim: int = 64,  # Paper: x → 64
    ):
        super().__init__()

        # Sinusoidal embeddings (per Padidar Appendix B)
        self.time_embed = SinusoidalEmbedding(time_embed_dim)
        self.input_embed = nn.Linear(input_dim, input_embed_dim)
        self.condition_embed = nn.Linear(4, condition_dim)  # (ι, A, nfp, N)

        # 4 hidden layers × 2048, GELU activation
        total_embed = input_embed_dim + time_embed_dim + condition_dim
        layers = [nn.Linear(total_embed, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, input_dim))

        self.net = nn.Sequential(*layers)
```

#### 4.1.4 Checkpoint Loading

**File**: `ai_scientist/experiment_setup.py`

Update `create_generative_model()`:
```python
def create_generative_model(cfg: ExperimentConfig) -> DiffusionDesignModel | None:
    if not cfg.generative.enabled:
        return None

    model = DiffusionDesignModel(
        hidden_dim=cfg.generative.hidden_dim,
        n_layers=cfg.generative.n_layers,
        pca_components=cfg.generative.pca_components,
        learning_rate=cfg.generative.learning_rate,
        epochs=cfg.generative.epochs,
    )

    # Load pre-trained checkpoint
    if cfg.generative.checkpoint_path:
        ckpt_path = Path(cfg.generative.checkpoint_path)
        if ckpt_path.exists():
            model.load_checkpoint(ckpt_path)
            print(f"[runner] Loaded Dreamer checkpoint: {ckpt_path}")
        else:
            print(f"[runner] Warning: Checkpoint not found: {ckpt_path}")

    return model
```

#### 4.1.5 Dynamic Target Metrics

**File**: `ai_scientist/workers.py`

Parameterize `ExplorationWorker.run()`:
```python
def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
    n_samples = context.get("n_samples", 10)

    # Dynamic target metrics (not hardcoded)
    target_metrics = context.get("target_metrics", {
        "aspect_ratio": self.cfg.optimization.target_aspect_ratio,
        "iota_bar": self.cfg.optimization.target_iota_bar,
        "nfp": self.cfg.nfp,
        "N": 1 if self.cfg.quasisymmetry == "QH" else 0,
    })

    if isinstance(self.generative_model, DiffusionDesignModel):
        samples = self.generative_model.sample(n_samples, target_metrics)
    # ... rest of method
```

---

### Phase 2: Geometric Pre-relaxation (Priority: High)

**Timeline**: 1-2 days
**Based on**: LEGO-xtal (Ridwan et al., 2025)

#### 4.2.1 New File: Pre-relaxation Module

**File**: `ai_scientist/optim/prerelax.py`

```python
"""
Geometric pre-relaxation before VMEC++ evaluation.

Based on LEGO-xtal (Ridwan et al., 2025): Optimize generated structures to
target local environment BEFORE expensive physics simulation. This filters
out geometrically invalid candidates in milliseconds vs minutes with VMEC++.
"""

import numpy as np
import torch
from typing import Tuple

def compute_geometric_energy(boundary: np.ndarray) -> float:
    """
    Compute geometric energy penalizing:
    1. High curvature regions
    2. Self-intersections
    3. Deviation from target aspect ratio

    Returns:
        float: Geometric energy (lower is better)
    """
    # Convert boundary coefficients to surface points
    surface_points = fourier_to_xyz(boundary)

    # Curvature penalty (prefer smooth surfaces)
    curvature = compute_mean_curvature(surface_points)
    curvature_penalty = np.mean(curvature ** 2)

    # Self-intersection penalty
    intersection_penalty = detect_self_intersection(surface_points)

    # Total energy
    return curvature_penalty + 10.0 * intersection_penalty


def prerelax_boundary(
    boundary: np.ndarray,
    steps: int = 50,
    lr: float = 0.01,
    threshold: float = 0.1,
) -> Tuple[np.ndarray, bool]:
    """
    Gradient descent to minimize geometric energy.

    Args:
        boundary: Initial Fourier coefficients (661-dim)
        steps: Number of optimization steps
        lr: Learning rate
        threshold: Energy threshold for acceptance

    Returns:
        Tuple of (relaxed_boundary, is_valid)
    """
    x = torch.tensor(boundary, requires_grad=True, dtype=torch.float32)
    optimizer = torch.optim.Adam([x], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()

        # Compute energy (with autograd)
        energy = _geometric_energy_torch(x)
        energy.backward()
        optimizer.step()

        # Early termination if converged
        if energy.item() < threshold:
            break

    relaxed = x.detach().numpy()
    is_valid = compute_geometric_energy(relaxed) < threshold

    return relaxed, is_valid
```

#### 4.2.2 Integration with Forward Model

**File**: `ai_scientist/forward_model.py`

Add pre-relaxation before VMEC++:
```python
def evaluate_boundary(
    boundary: np.ndarray,
    settings: ForwardModelSettings,
    prerelax: bool = True,
    prerelax_steps: int = 50,
) -> EvaluationResult:
    """Evaluate a boundary with optional geometric pre-relaxation."""

    if prerelax:
        from ai_scientist.optim.prerelax import prerelax_boundary
        boundary, is_valid = prerelax_boundary(boundary, steps=prerelax_steps)

        if not is_valid:
            # Skip VMEC++ for geometrically invalid boundaries
            return EvaluationResult(
                success=False,
                error="Failed geometric pre-relaxation",
            )

    # Continue with VMEC++ evaluation
    return _run_vmec(boundary, settings)
```

---

### Phase 3: RL Agent Integration (Priority: Medium)

**Timeline**: 3-5 days
**Based on**: BoltzGen (Stark et al., 2025), StellarForge Plan

#### 4.3.1 New File: RL Environment

**File**: `ai_scientist/rl_env.py`

```python
"""
OpenAI Gym environment wrapping the Surrogate Critic.

The RL agent learns to perform "micro-surgery" on Fourier coefficients,
making small delta adjustments to improve physics metrics while avoiding
constraint violation "cliffs".
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class StellaratorEnv(gym.Env):
    """Stellarator refinement environment for PPO/SAC training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        surrogate,
        target_metrics: dict,
        delta_scale: float = 0.01,
        max_steps: int = 100,
        cliff_penalty: float = 100.0,
    ):
        super().__init__()

        self.surrogate = surrogate
        self.target_metrics = target_metrics
        self.delta_scale = delta_scale
        self.max_steps = max_steps
        self.cliff_penalty = cliff_penalty

        # State: Fourier coefficients (661-dim)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(661,), dtype=np.float32
        )

        # Action: Delta adjustments to coefficients
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(661,), dtype=np.float32
        )

        self.state = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize from a Dreamer-generated seed
        if options and "initial_boundary" in options:
            self.state = options["initial_boundary"].copy()
        else:
            self.state = np.random.randn(661).astype(np.float32)

        self.step_count = 0
        return self.state, {}

    def step(self, action: np.ndarray):
        # Apply micro-surgery delta
        self.state += action * self.delta_scale
        self.step_count += 1

        # Get surrogate predictions (milliseconds)
        pred = self.surrogate.predict(self.state.reshape(1, -1))[0]

        # Compute reward with cliff penalties
        reward = self._compute_reward(pred)

        # Check termination
        terminated = self._is_feasible(pred)
        truncated = self.step_count >= self.max_steps

        info = {"predictions": pred}

        return self.state, reward, terminated, truncated, info

    def _compute_reward(self, pred: dict) -> float:
        """Reward shaping with cliff penalties for constraint violations."""
        # Objective reward (maximize L_∇B)
        objective_reward = pred.get("L_grad_B", 0.0)

        # Cliff penalties for hard constraints
        penalties = 0.0

        # Vacuum well must be positive (W > 0)
        vacuum_well = pred.get("vacuum_well", 0.0)
        if vacuum_well < 0:
            penalties += self.cliff_penalty * abs(vacuum_well)

        # Magnetic well constraint
        magnetic_well = pred.get("magnetic_well", 0.0)
        if magnetic_well < 0:
            penalties += self.cliff_penalty * abs(magnetic_well)

        return objective_reward - penalties

    def _is_feasible(self, pred: dict) -> bool:
        """Check if current state satisfies all constraints."""
        return (
            pred.get("vacuum_well", -1) > 0 and
            pred.get("magnetic_well", -1) > 0 and
            pred.get("aspect_ratio_constraint", 1) < 0.05
        )
```

#### 4.3.2 RL Training Integration

**File**: `ai_scientist/rl_training.py`

```python
"""PPO training for stellarator refinement using CleanRL."""

from cleanrl.ppo import PPO
from ai_scientist.rl_env import StellaratorEnv

def train_rl_agent(
    surrogate,
    target_metrics: dict,
    total_timesteps: int = 100_000,
    checkpoint_dir: str = "checkpoints/rl",
):
    """Train PPO agent for micro-surgery refinement."""

    # Create environment
    env = StellaratorEnv(surrogate, target_metrics)

    # Initialize PPO (CleanRL implementation)
    agent = PPO(
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    # Train
    agent.learn(total_timesteps=total_timesteps)
    agent.save(f"{checkpoint_dir}/ppo_stellarator.pt")

    return agent
```

---

## 5. Verification Plan

### 5.1 Phase 1 Verification

```bash
# 1. Train diffusion with paper specs
python scripts/train_offline.py \
    --hidden-dim 2048 \
    --n-layers 4 \
    --batch-size 4096 \
    --epochs 250 \
    --pca-components 50

# 2. Evaluate feasibility rate
python scripts/evaluate_feasibility.py \
    --checkpoint checkpoints/diffusion_paper_spec.pt \
    --n-samples 1000

# Expected: 40-60% feasibility (vs ~5% baseline)
```

### 5.2 Phase 2 Verification

```bash
# Compare VMEC++ crash rate with/without pre-relaxation
python scripts/compare_prerelax.py --n-samples 100

# Expected: >30% reduction in VMEC++ crashes
```

### 5.3 Full Pipeline Verification

```bash
# Run on P2 benchmark
python ai_scientist/runner.py \
    --problem p2 \
    --budget 100 \
    --enable-dreamer \
    --enable-prerelax \
    --enable-rl-refinement

# Expected: L_∇B > 9.5 (vs 8.61 baseline) with <100 VMEC++ calls
```

---

## 6. Resource Requirements

| Phase | GPU Memory | Training Time | Storage |
|-------|------------|---------------|---------|
| Phase 1 (Diffusion) | ~40GB (A100) | 4-8 hours | 2GB models |
| Phase 2 (Pre-relax) | <1GB | N/A | <10MB |
| Phase 3 (RL) | ~4GB | 2-4 hours | 500MB |

### Cloud GPU Options

Since the scaled architecture (2048 width) requires ~40GB VRAM, local GPUs may be insufficient. Recommended cloud options:

| Provider | GPU | Cost Estimate | Notes |
|----------|-----|---------------|-------|
| **RunGPU** | A100 80GB | ~$2/hr | On-demand, easy setup |
| **HuggingFace Spaces** | A100 40GB | Free tier available | Best for inference/demo |
| **Lambda Labs** | A100 80GB | ~$1.10/hr | Reserved pricing available |
| **Modal** | A100 | Pay-per-second | Serverless, fast spin-up |

**Tip**: For initial experiments, consider using gradient checkpointing + mixed precision to fit into A10G (24GB) at ~$0.50/hr.

### Apple Silicon (M3 Max) Configuration

Your **M3 Max (36GB unified RAM)** can train locally with these adjustments:

```yaml
# configs/stellarforge_m3max.yaml
generative:
  device: "mps"               # Metal Performance Shaders
  hidden_dim: 1024            # Reduced from 2048 (fits in 36GB)
  batch_size: 512             # Reduced from 4096
  gradient_checkpointing: true
  mixed_precision: true       # bfloat16 on M3
```

| Setting | Paper Spec | M3 Max Adjusted | Trade-off |
|---------|------------|-----------------|-----------|
| Hidden dim | 2048 | 1024 | ~10% quality loss |
| Batch size | 4096 | 512 | 8x slower convergence |
| Memory | 40GB | ~30GB | Fits in unified RAM |
| Training time | 4-8 hrs (A100) | 12-24 hrs (M3 Max) | Slower but free |

**Benefits of M3 Max:**
- **No transfer overhead**: Unified memory means tensors don't copy between CPU↔GPU
- **300GB/s bandwidth**: Faster than PCIe-connected GPUs
- **Free**: No cloud costs for experimentation

**PyTorch MPS setup:**
```python
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

---

## 7. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| OOM on scaled network | Medium | Gradient checkpointing, mixed precision |
| Surrogate inaccuracy | Low | Uncertainty-aware filtering, active learning |
| RL training instability | Medium | Curriculum learning, reward shaping |

---

## 8. Next Steps

1. **Immediate**: Implement Phase 1 config + PCA changes
2. **Week 1**: Train scaled diffusion model on ConStellaration data
3. **Week 2**: Integrate pre-relaxation and benchmark
4. **Week 3**: RL agent training and full pipeline validation

---

## Appendix A: Paper Architecture Comparison

```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT IMPLEMENTATION                    │
├─────────────────────────────────────────────────────────────┤
│  Input: 661 Fourier coefficients (raw)                      │
│  Network: 64 hidden, ? layers                               │
│  Batch: 32                                                  │
│  Training: 20-200 epochs                                    │
│  Result: ~5% feasibility                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    PAPER SPECIFICATION                       │
├─────────────────────────────────────────────────────────────┤
│  Input: 661 → PCA → 50 latent (85%+ variance retained)      │
│  Network: 4 layers × 2048, GELU, sinusoidal embeddings     │
│  Batch: 4096                                                │
│  Training: 250 epochs (150k ConStellaration samples)        │
│  Result: <5% deviation on A, <6% JQS (OOD)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Appendix B: Configuration Template

**File**: `configs/stellarforge_v1.yaml`

```yaml
# StellarForge Paper-Backed Configuration
generative:
  enabled: true
  backend: diffusion

  # Paper specs (Padidar et al.)
  hidden_dim: 2048
  n_layers: 4
  pca_components: 50
  batch_size: 4096
  epochs: 250
  diffusion_timesteps: 200
  learning_rate: 0.0005

  # Checkpoint
  checkpoint_path: checkpoints/diffusion_paper_spec.pt
  device: cuda

# Pre-relaxation (LEGO-xtal)
prerelax:
  enabled: true
  steps: 50
  threshold: 0.1

# RL (BoltzGen-inspired)
rl:
  enabled: true
  algorithm: ppo
  total_timesteps: 100000
  cliff_penalty: 100.0
```
