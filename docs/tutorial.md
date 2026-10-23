# hifi_code — Usage Tutorial

## What it is

`hifi_code` is an **autonomous coding loop**. An LLM drives a small set of
primitives through the `ast-bro` code server (a structural AST tool), edits files
surgically, and runs tests + lint after every change. It has two layers:

- **Inner loop** — the coding loop. Given a goal, it navigates, reads, edits, and
  verifies until the goal is met and verify passes.
- **Outer loop** (`--outer`) — drafts a machine-checkable **spec** with you
  interactively, then hands it to the inner loop. If the inner loop finds the spec
  wrong, it can emit a structured `revise` that flows back to the outer loop to
  re-draft one field and re-seed.

## Prerequisites

1. **`ast-bro`** must be on your `PATH` (defaults to `ast-bro mcp`). It builds the
   search index automatically on `seed`.
2. **An OpenAI-compatible LLM server** at `http://localhost:8080/v1` (llama.cpp).
   You must pin the model — the default `gpt-4o-mini` will 400.

Recommended models (reasoning models that match the prompt's design):

- `code-Ornith-1.5-35B` ← recommended
- `code-Qwen3.6-35B`, `agent-Mellum2.1-12B`, `general-Mistral-Small-4-119B`, `reason-*`

## Install

```bash
pip install -e .        # provides the `hifi-code` entry point
```

## Configure (do this first)

The only thing you *must* override is the model. Set it as an env var:

```bash
export HIFI_CODE_LLM_MODEL=code-Ornith-1.5-35B
```

Other env vars (all optional, defaults shown):

| Env var | Default | Purpose |
|---|---|---|
| `HIFI_CODE_LLM_BASE_URL` | `http://localhost:8080/v1` | LLM endpoint |
| `HIFI_CODE_LLM_MODEL` | `gpt-4o-mini` ⚠️ pin this | model name |
| `HIFI_CODE_SERVER` | `ast-bro mcp` | code-server command |

Verify config (tests + lint) is tunable via `VerifyConfig`; defaults run
`pytest -q` and `ruff check .`.

## Run it

### 1. Quick mode — goal straight to the inner loop

```bash
hifi-code "Add a `greet(name)` function to greeting.py that returns 'Hello, <name>'"
```

Emits newline-delimited JSON, one object per loop step (navigate/read/edit/verify/finish).

### 2. Spec-first mode — draft with you, then code

```bash
hifi-code --outer
```

The outer loop renders a draft spec and asks you to approve or veto assumptions for
each field: `events`, `state`, `trigger`, `data_model`, `transitions`, `modules`,
`seams`, `side_effects`, `inputs`, `outputs`, `constraints`, `edge_cases`,
`acceptance`, `assumptions`.

Type `resolve` (accept all assumptions), `edit` (change a field), or `abort`. On a
valid spec it serializes it into a goal string and runs the inner loop. If the inner
loop hits a structural gap, it emits `revise`, the outer loop re-drafts that one
field, and the inner loop re-seeds — up to 5 revision rounds.

### 3. MCP server mode — embed the loop as an MCP tool

```bash
hifi-code --transport mcp
```

Exposes a single `code_loop` tool (`goal`, `root`, `max_steps`) over stdio.

### Common flags
- `--root DIR` — codebase root to seed from (default `.`)
- `--max-steps N` — override the 40-step budget

## The action vocabulary

Every tick the model emits one JSON action:

```json
{"action": "navigate", "tool": "map", "path": "src/foo.py"}   // map|digest|search|callers|callees|impact|context|surface|find_related|implements
{"action": "read", "path": "greet"}                            // show a symbol's exact source
{"action": "edit", "edits": [{"path": "greeting.py", "old": "...", "new": "..."}]}
{"action": "verify"}                                           // run tests + lint
{"action": "revise", "field": "data_model", "op": "add", "value": "Order", "reason": "..."}
{"action": "plan", "subgoals": [{"goal": "...", "path": "a.py"}, {"goal": "...", "path": "b.py"}]}
{"action": "finish"}
```

Rules that trip people up:

- For `navigate`, the field is literally `"tool"` with the tool name as its **value**
  — not `{ "map": {...} }`.
- `edit` prefers symbol-scoped `{path, symbol, from, to}` (harness fetches exact
  source), else exact-block `{path, old, new}`. `from` must match the source
  **verbatim and exactly once**. To create a file, use empty `old`.
- `plan` subgoal paths **must be disjoint** (no shared file) so they run in parallel.

## The revise policy (Move 3 — the "when to revise" gate)

`revise` is not for ordinary bugs — it's for when **the spec is wrong** (a structural
contradiction or a gap the spec couldn't model). The loop uses a hybrid heuristic:

- A counter `verify_failures` grows **only** on a failing `verify` with no applied
  edit or passing verify in between.
- It resets to 0 on a **passing verify** or an **applied edit**.
- Once it reaches `revise_threshold` (default **3**), the next prompt carries a
  concrete nudge: *"verify has failed 3 times… if this points to a wrong/unmodeled
  spec assumption, emit `revise`."*

It's always a **suggestion to the model, never an auto-revise** — ordinary code
fixes keep the counter at zero, so the gate fires on structural stuckness, not every
fix. Adjust with `Settings.revise_threshold` (no CLI flag yet).

## Guardrails

- Verify **before** finishing.
- Never edit denylisted files (sandbox denylist, default empty).
- `propose_only=True` records edits but never writes to disk (useful to watch the
  loop without touching files).
- After `max_stalls` (12) consecutive ticks with no progress, the loop exits with a
  `stalled` finish.
- A rejected edit is feedback: re-read, adjust, retry — don't re-submit the same text.

## Example: watch it work

```bash
export HIFI_CODE_LLM_MODEL=code-Ornith-1.5-35B
hifi-code "Add a `square(x)` function to sweep_scratch.py" --max-steps 20
```

You'll see JSON lines for each step: `navigate map` → `read square` → `edit` →
`verify` (ok) → `finish`.
