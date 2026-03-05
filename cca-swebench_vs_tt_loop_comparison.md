# In-Depth Loop Implementation Comparison: `cca-swebench` vs `tt/discover`

## Scope and framing

This report compares how each codebase implements "run until done" behavior, with emphasis on backend control-flow semantics:

- What is one loop iteration?
- Who decides to continue?
- What condition means "target reached"?
- What guardrails prevent runaway execution?
- How state/progress is persisted across iterations?

Important scope note:

- For `tt`, this analysis is on `tt/discover` (the path you pointed to), which contains multiple execution styles.
- `tt/discover` includes both target-driven loops and fixed-budget loops.

---

## 1) High-level architecture of the loop

### `cca-swebench`: recursive orchestrator loop (event/exception-driven)

Runtime chain:

`scripts/run_sbp.sh` -> `app.pex --prompt ...` -> `scripts/run_swebench.py` -> `scripts/utils.run_agent_with_prompt()` -> `Confucius.invoke_analect(Entry, entry_name="Code")` -> `CodeAssistEntry` -> `AnthropicLLMOrchestrator` -> `BaseOrchestrator._process_messages()`

Core files:

- `scripts/run_sbp.sh`
- `scripts/run_swebench.py`
- `scripts/utils.py`
- `confucius/analects/code/entry.py`
- `confucius/orchestrator/base.py`
- `confucius/orchestrator/anthropic.py`
- `confucius/orchestrator/extensions/solo/base.py`

Core shape:

- One orchestrator "turn" processes model output and extensions.
- Extensions can raise `OrchestratorInterruption` to request another turn.
- The orchestrator recursively calls itself until no interruption remains.

### `tt/discover`: explicit imperative loops (`while True` / `for range`)

In the HF app training path, the loop is direct and imperative:

- Global training state tracks `best_cycles`, `should_stop`, and step counters.
- `while True` runs chunk-by-chunk training.
- Break conditions include user stop, target reached (`best_cycles <= TARGET_CYCLES`), max steps, and max wall-clock.

In RL cookbook paths, rollout loops are explicit per-episode/per-batch:

- `do_single_rollout`: `while True` until `episode_done`.
- Trainer: fixed total batches from `num_epochs * len(dataset)`.

Core files:

- `discover/hf_space/app.py`
- `discover/tinker_cookbook/rl/rollouts.py`
- `discover/tinker_cookbook/recipes/ttt/env_ttt.py`
- `discover/tinker_cookbook/rl/train.py`
- `discover/ttt_lite_perf_takehome.py`

---

## 2) Side-by-side loop semantics

| Dimension | `cca-swebench` | `tt/discover` |
|---|---|---|
| Loop primitive | Recursive function (`_process_messages`) | Imperative loops (`while True`, `for`) |
| Continue signal | Exception/event (`OrchestratorInterruption`) | Boolean/bound checks in loop body |
| Main unit of work | LLM turn + extension dispatch + tool-use processing | Training chunk / rollout step / generation step |
| Primary stop in core engine | No interruption raised, explicit termination exception, or max-iterations | Target threshold, manual stop flag, step budget, time budget, episode_done |
| Target-first behavior | Mostly extension-defined (e.g., solo progress 100) | Native in HF app loop (`best_cycles <= TARGET_CYCLES`) |
| Guardrail style | Hard recursion guard via `max_iterations` | Multiple explicit loop breaks and cycle/time/step caps |
| Persistence | Session memory + trajectory dump + save in finally | Mutable state dict + periodic adapter save/upload (HF path), DB/checkpoint by mode |

---

## 3) `cca-swebench` deep dive (recursive orchestration with guardrails)

### 3.1 Core control loop

`BaseOrchestrator._process_messages`:

- Checks iteration cap (`max_iterations`, default 1000).
- Gets root tag for current task and increments iteration counter.
- Processes root output and completion hooks.
- If `OrchestratorInterruption` is raised, processes interruption payload and recurses.

This is effectively:

```python
def process_messages(task):
    assert num_iterations < max_iterations
    root = get_root_tag(task)
    num_iterations += 1
    try:
        process_root(root)
        on_process_messages_complete()
    except OrchestratorInterruption as exc:
        process_interruption(exc)
        process_messages(task)  # recursive continue
```

### 3.2 Anthropic tool-use sub-loop

`AnthropicLLMOrchestrator._process_messages` wraps base behavior with a second recursive cycle:

- Collect tool calls into `_tool_use_queue`.
- Drain tool queue and emit tool results.
- Re-enter `_process_messages(task, context)` after queue handling.
- If queue is empty, runs completion hooks that may still raise interruption.

So the runtime has two nested continuation layers:

- Message-processing recursion.
- Tool-queue recursion.

### 3.3 "Target reached" is extension policy, not core metric

In `SoloModeExtension`:

- Progress tool updates `_current_progress`.
- If progress `< 100`, `on_process_messages_complete` raises interruption to keep going.
- If progress is `100`, it stops requesting continuation.
- If status is `ERROR`, it stops auto-looping and waits for user action.

This means core orchestrator is generic; "goal reached" semantics are delegated to extension logic.

### 3.4 Guardrails in `cca-swebench`

- Iteration hard stop via `MaxIterationsReachedError`.
- Explicit termination channel `OrchestratorTermination` handled at top-level `impl`.
- Tool names sanitized before execution in Anthropic path.
- Full trajectory dump and session state save in `finally` block at runner level.

### 3.5 Operational properties

Strengths:

- Highly composable: new extensions can inject new continue/stop policies without rewriting loop engine.
- Natural fit for tool-calling and multi-stage LLM interactions.
- Good observability and replay via memory + trajectory artifacts.

Tradeoffs:

- Harder to reason about than plain loops because continuation is exception-driven and distributed across extensions.
- Potentially deep recursive chains if interruption policy is too permissive.
- Correctness depends on extension discipline (e.g., emitting proper completion/termination signals).

---

## 4) `tt/discover` deep dive (explicit loops with concrete break conditions)

### 4.1 HF app path: true target-driven closed loop

In `hf_space/app.py`:

- Global state tracks `best_cycles` and `should_stop`.
- Reward function updates `best_cycles` when a better correct solution appears.
- Trainer callback can request stop when target is reached.
- Outer `while True` loop exits on:
  - manual stop flag
  - target reached (`best_cycles <= TARGET_CYCLES`)
  - max total steps reached
  - max minutes exceeded

This is a direct "loop until target/budget/stop" implementation.

### 4.2 RL rollout path: episode-driven loop

In `rollouts.py`:

- `do_single_rollout` loops until env returns `episode_done`.

In `recipes/ttt/env_ttt.py`:

- `step()` currently returns `episode_done=True` per step for this env, so rollouts are single-step episodes in this configuration.

### 4.3 Fixed-budget trainer path

In `rl/train.py`:

- Total batches are computed as `num_epochs * len(dataset)`.
- Training runs from `start_batch` to `end_batch=num_batches_total`.
- Completion is budget-based, not target-threshold based.

In `ttt_lite_perf_takehome.py`:

- Outer loop is `for step in range(args.steps)` with archive updates per step.

### 4.4 Guardrails in `tt/discover`

- Explicit numeric break conditions are easy to audit.
- Simulator path has cycle caps and validation checks to avoid runaway or invalid programs.
- Manual stop flag is available in HF app mode.
- Adapter saves/checkpoints happen periodically (mode-dependent).

### 4.5 Operational properties

Strengths:

- Simple and predictable control flow.
- Easy to test and reason about: each break condition is local and visible.
- Target metrics (e.g., cycles threshold) can be directly embedded into stop logic.

Tradeoffs:

- Less composable than orchestration middleware if you need many heterogeneous continuation policies.
- Different submodules can diverge in loop semantics (`while True` target-driven vs `for` fixed-budget), which requires mode-specific understanding.

---

## 5) ASCII control-flow comparison

### `cca-swebench` (recursive/event-driven)

```text
[Entry: CodeAssistEntry]
        |
        v
[AnthropicLLMOrchestrator._process_messages]
        |
        v
[BaseOrchestrator._process_messages]
   | check max_iterations
   | get_root_tag + process output
   |
   +--> interruption? --yes--> [process interruption] --> recurse _process_messages(task)
   |                                ^
   |                                |
   +--> no -------------------------+
        |
        v
[tool_use_queue exists?]
   | yes -> process queue -> clear -> recurse _process_messages(task)
   | no  -> extension completion hook may interrupt -> recurse
   v
[return]
```

### `tt/discover` HF app path (explicit target loop)

```text
[init training_state: best_cycles, should_stop, step]
        |
        v
      while True
        |
        +--> if should_stop: break
        +--> if best_cycles <= TARGET_CYCLES: break
        +--> if step >= max_total_steps: break
        +--> if elapsed >= max_minutes: break
        |
        +--> train one chunk
        +--> evaluate generations / reward updates best_cycles
        +--> save adapter/checkpoint
        |
        +--> if not auto_continue: break
        v
      [done]
```

---

## 6) Backend-engineering interpretation

If you think in service architecture terms:

- `cca-swebench` loop is like a middleware/state-machine runtime where "continue" is emitted as control events (exceptions), and each extension can participate in policy.
- `tt/discover` loop is like a batch worker with explicit guard conditions around a main processing loop.

Which is better depends on objective:

- Need pluggable agent behaviors, tool orchestration, and policy injection:
  - `cca-swebench` style is stronger.
- Need deterministic optimization loops around a scalar objective with transparent stop criteria:
  - `tt` style is stronger.

---

## 7) Direct answer to "which reaches a target in a closed loop?"

Within the analyzed paths:

- `tt/discover` HF app has the most explicit closed loop until target (`best_cycles <= TARGET_CYCLES`).
- `cca-swebench` core loop is not inherently metric-targeted; it is continuation-driven. A target concept appears through extensions (for example solo progress reaching 100).

So if "closed loop until numeric objective is reached" is the criterion, `tt/discover` is the clearer match.

---

## 8) Evidence map (line-level)

`cca-swebench`:

- Entrypoint shell wrapper and artifact export: `scripts/run_sbp.sh:16`, `scripts/run_sbp.sh:35`
- Prompt runner: `scripts/run_swebench.py:103`
- Agent invoke + finally persistence: `scripts/utils.py:24`, `scripts/utils.py:34`, `scripts/utils.py:37`
- Entry dispatch to `Code`: `confucius/core/entry/entry.py:48`
- `CodeAssistEntry` wiring `AnthropicLLMOrchestrator` + extensions: `confucius/analects/code/entry.py:64`, `confucius/analects/code/entry.py:83`, `confucius/analects/code/entry.py:92`
- Recursive message loop + max-iteration guard: `confucius/orchestrator/base.py:28`, `confucius/orchestrator/base.py:212`, `confucius/orchestrator/base.py:226`, `confucius/orchestrator/base.py:229`
- Top-level termination catch: `confucius/orchestrator/base.py:237`
- Anthropic recursive tool-use handling: `confucius/orchestrator/anthropic.py:193`, `confucius/orchestrator/anthropic.py:196`, `confucius/orchestrator/anthropic.py:204`, `confucius/orchestrator/anthropic.py:210`, `confucius/orchestrator/anthropic.py:213`
- Tool queue drain loop: `confucius/orchestrator/anthropic.py:260`
- Solo progress-driven continuation: `confucius/orchestrator/extensions/solo/base.py:102`, `confucius/orchestrator/extensions/solo/base.py:166`, `confucius/orchestrator/extensions/solo/base.py:185`, `confucius/orchestrator/extensions/solo/base.py:194`
- Interruption/termination exception types: `confucius/orchestrator/exceptions.py:9`, `confucius/orchestrator/exceptions.py:27`, `confucius/orchestrator/exceptions.py:31`

`tt/discover`:

- Target constant and mutable training state: `discover/hf_space/app.py:81`, `discover/hf_space/app.py:94`
- Best-cycle update in reward function: `discover/hf_space/app.py:493`
- Callback early stop on target/manual stop: `discover/hf_space/app.py:697`, `discover/hf_space/app.py:702`, `discover/hf_space/app.py:704`
- Outer `while True` break conditions: `discover/hf_space/app.py:724`, `discover/hf_space/app.py:726`, `discover/hf_space/app.py:728`, `discover/hf_space/app.py:731`, `discover/hf_space/app.py:733`
- Manual stop switch: `discover/hf_space/app.py:854`, `discover/hf_space/app.py:859`
- Episode loop in rollout: `discover/tinker_cookbook/rl/rollouts.py:23`, `discover/tinker_cookbook/rl/rollouts.py:45`
- Environment marks episode done: `discover/tinker_cookbook/recipes/ttt/env_ttt.py:311`, `discover/tinker_cookbook/recipes/ttt/env_ttt.py:313`
- Fixed-budget total steps by epochs: `discover/tinker_cookbook/rl/train.py:1536`, `discover/tinker_cookbook/rl/train.py:1550`
- Fixed-step outer loop in lite variant: `discover/ttt_lite_perf_takehome.py:101`
