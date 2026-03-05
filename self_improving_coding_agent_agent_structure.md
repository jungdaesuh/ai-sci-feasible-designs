# self_improving_coding_agent Agent Structure

## 1) High-level architecture
- The system is a two-loop design:
  - Outer loop: experiment/evaluation/self-improvement in `runner.py`
  - Inner loop: event-driven agent runtime in `base_agent/agent.py` and `base_agent/src/agents/base_agent.py`
- Outer loop benchmarks each agent iteration, then runs meta-improvement to produce the next agent version.
- Inner loop uses an event bus + callgraph to coordinate tool calls, agent calls, and completion state.

## 2) Core structure (where agents live)
- Outer orchestration and benchmarking:
  - `runner.py`
- Runtime entry and lifecycle:
  - `base_agent/__main__.py`
  - `base_agent/agent.py`
- Role implementations:
  - `base_agent/src/agents/implementations/main_orchestrator.py`
  - `base_agent/src/agents/implementations/coder.py`
  - `base_agent/src/agents/implementations/problem_solver.py`
  - `base_agent/src/agents/implementations/reasoner.py`
  - `base_agent/src/agents/implementations/archive_explorer.py`
- Agent execution core:
  - `base_agent/src/agents/base_agent.py`
  - `base_agent/src/agents/agent_calling.py`
- Memory/state/callgraph:
  - `base_agent/src/events/event_bus.py`
  - `base_agent/src/events/event_bus_utils.py`
  - `base_agent/src/callgraph/manager.py`
- Tool framework:
  - `base_agent/src/tools/base_tool.py`
  - `base_agent/src/tools/__init__.py`
  - `base_agent/src/tools/execute_command.py`
  - `base_agent/src/tools/file_tools.py`
  - `base_agent/src/tools/edit_tools/overwrite_file.py`
- Benchmark abstraction and analysis:
  - `base_agent/src/benchmarks/base.py`
  - `base_agent/src/benchmarks/__init__.py`
  - `base_agent/src/utils/archive_analysis.py`

## 3) Execution flow
1. `python runner.py` initializes `results/run_*` and seeds `agent_0` from `base_agent`.
2. For each iteration, benchmark jobs are run (containerized), then meta-improvement is executed.
3. Container executes `python -m <agent_module> benchmark ...`.
4. `base_agent/__main__.py` parses command, creates `Agent`, and calls `exec`.
5. `Agent.exec` initializes overseer/server, creates `MainOrchestratorAgent`, publishes problem statement, and runs with timeout/cost monitors.
6. `BaseAgent.execute` builds context from event state, calls LLM, dispatches tool/agent calls, and ends on completion signal.
7. Results (`answer.txt`, traces, summaries) are scored and aggregated into run outputs (`results.jsonl`, performance summary files).

## 4) Extension/configuration points
- New agents/tools: subclass base classes with registry-based auto-registration (`__init_subclass__`), then expose in role allowlists.
- Benchmark extension: add benchmark class and register in benchmark registry.
- Model/provider surface: env-driven model config + provider routing/failover in LLM API layer.
- Reasoning policies: reasoning-structure tools and meta-improvement logic can be swapped/extended.

## 5) Strengths and risks
- Strength:
  - Strong modularity (roles, registries, typed interfaces, event-bus state, callgraph, oversight) supports iterative self-improvement and debuggability.
- Risk:
  - Potential maintainability drift from duplicated base-agent variants and output-contract drift between docs and runtime artifacts.
