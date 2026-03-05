# cca-swebench Agent Structure

## 1) High-level architecture
- This repo is a SWE-bench harness around the `confucius` agent framework.
- Main runtime stack:
  - `Confucius` session context
  - `Entry` dispatch to a concrete role (`Code` or `NoteTaker`)
  - `AnthropicLLMOrchestrator` + extension pipeline
- Orchestration is recursive: generate -> parse text/tags/tool-calls -> execute extensions -> continue/interrupt until completion or limits.

## 2) Core structure (where agents live)
- Harness/scripts:
  - `scripts/run_swebench.py`
  - `scripts/utils.py`
  - `scripts/run_sbp.sh`
  - `scripts/run_note_taker.py`
- Entry-role system:
  - `confucius/core/entry/entry.py`
  - `confucius/core/entry/manager.py`
  - `confucius/core/entry/decorators.py`
  - `confucius/analects/code/entry.py`
  - `confucius/analects/note_taker/entry.py`
- Orchestrator core:
  - `confucius/orchestrator/base.py`
  - `confucius/orchestrator/llm.py`
  - `confucius/orchestrator/anthropic.py`
- Memory/state/artifacts:
  - `confucius/core/memory.py`
  - `confucius/core/storage.py`
  - `confucius/core/artifact.py`
  - `confucius/orchestrator/extensions/memory/hierarchical/extension.py`
- Tool-use:
  - `confucius/orchestrator/extensions/tool_use.py`
  - `confucius/orchestrator/extensions/command_line/base.py`
  - `confucius/orchestrator/extensions/file/edit.py`
  - `confucius/orchestrator/extensions/function/base.py`
  - `confucius/orchestrator/extensions/solo/base.py`

## 3) Execution flow
1. `scripts/run_swebench.py` builds SWE-bench prompt from task inputs.
2. `scripts/utils.py::run_agent_with_prompt` creates a `Confucius` session and calls `Entry(..., entry_name="Code")`.
3. `Entry` resolves and runs the registered entry class.
4. `CodeAssistEntry` builds extension stack and creates `AnthropicLLMOrchestrator`.
5. Orchestrator loads memory, formats prompt, calls provider via LLM manager, parses tool-use outputs.
6. Tool calls are dispatched to enabled extensions; outputs are written back into memory.
7. Recursive loop continues until completion/termination/max-iteration.
8. Session and trajectory are persisted (`/tmp/confucius/traj_<session>.json`).

## 4) ASCII visual flow
```text
[SWE-bench entrypoint]
scripts/run_sbp.sh (TASK_ID -> app.pex --prompt ... )
        |
        v
[PROMPT BUILD]
run_swebench.py reads problem .txt + builds prompt template
        |
        v
[SESSION INIT]
scripts/utils.py -> Confucius(...)
        |
        v
[ENTRY DISPATCH]
Entry(entry_name="Code") -> CodeAssistEntry
        |
        v
[ORCHESTRATOR CALL]
CodeAssistEntry -> AnthropicLLMOrchestrator
        |
        v
+---------------------- TOOL EXECUTION LOOP ----------------------+
| LLM call -> parse response -> enqueue tool_use -> execute tools |
| -> write tool results/messages to memory -> recurse             |
+-----------------------------------------------------------------+
        |
        v
[STOP CONDITIONS]
Solo progress reaches 100 (no interruption) OR max-iterations/termination
        |
        v
[TRAJECTORY SAVE + COMPLETION]
finally: dump_trajectory + save; run_sbp.sh copies traj file out
```

Node-to-file map:
- `SWE-bench entrypoint` + `PROMPT BUILD`: `scripts/run_sbp.sh`, `scripts/run_swebench.py`, `confucius/analects/code/tasks.py`
- `SESSION INIT`: `scripts/utils.py`, `confucius/lib/confucius.py`
- `ENTRY DISPATCH` + `ORCHESTRATOR CALL`: `confucius/core/entry/entry.py`, `confucius/analects/code/entry.py`
- `TOOL EXECUTION LOOP` + memory updates: `confucius/orchestrator/llm.py`, `confucius/orchestrator/anthropic.py`, `confucius/orchestrator/extensions/tool_use.py`, `confucius/core/memory.py`
- `STOP CONDITIONS` + save path: `confucius/orchestrator/extensions/solo/base.py`, `confucius/orchestrator/base.py`, `scripts/utils.py`, `confucius/lib/confucius.py`, `scripts/run_sbp.sh`

## 5) Extension/configuration points
- New roles: implement entry mixin + register with `@public` entry decorators.
- New behaviors: implement orchestrator `Extension`/`ToolUseExtension` callbacks and add to entry extension list.
- Model/provider surface: `LLMParams` and `AutoLLMManager` route across Bedrock/OpenAI/Azure/Google providers.
- Tool policy/safety surface: command allowlists/validators and file access policy injection.
- Runtime configuration relies heavily on env vars and entry defaults rather than a single central external config file.

## 6) Strengths and risks
- Strength:
  - Clean extension-oriented architecture with explicit memory/storage/artifact channels and provider-agnostic tool-call handling.
- Risk:
  - Operational policies can be permissive by default (file access + command surfaces), and recursive loop control depends on robust progress/termination signaling.
