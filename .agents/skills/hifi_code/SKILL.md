---
name: hifi_code
description: Autonomous code-editing loop driven by ast-bro. Use when asked to implement, refactor, or fix code in this repo. Keeps edits surgical, context-aware, and verified. Write new code or change existing code.
---

# hifi_code

hifi_code is an autonomous coding loop. An LLM drives a small set of
primitives through the `ast-bro` code server, then edits files surgically and
verifies after every change. This skill is the playbook the loop follows — use
it to keep edits compact, safe, and correct.

## The loop

Every tick the model emits one JSON action: `navigate | read | edit | verify |
revise | finish`. The loop enforces a step budget and a stall detector, so finish
as soon as the goal is met and verify passes.

Before the first tick the loop automatically builds the ast-bro search index, so
`search` and `find_related` work without any action from you. Do not emit an
`init` action — it is handled for you.

## Required JSON shape

Copy these field names **exactly** — the harness rejects any action whose fields
don't match. The most common mistake is using the tool name as the key instead
of the literal field `tool`.

- `navigate`: `{ "action": "navigate", "tool": "<map|digest|search|callers|callees|impact|context|surface|find_related|implements>", "path": "<file or symbol>" }`
- `read`: `{ "action": "read", "path": "<file or symbol>" }`
- `edit`: `{ "action": "edit", "edits": [ { "path": "<file>", "old": "<exact source>", "new": "<replacement>" } ] }`
- `revise`: `{ "action": "revise", "field": "<spec field>", "op": "<add|remove|replace>", "value": "<new value>" }`
- `verify`: `{ "action": "verify" }`
- `finish`: `{ "action": "finish" }`

For `navigate`, the field is literally the word `tool` and its **value** is the
tool name (for example `"tool": "map"`). Do **not** write `{ "map": {...} }`.

## The workflow

1. **Map first.** `navigate` with `map` (or `digest`) to get signatures and
   the module structure. Identify the target symbol and its neighbors.
2. **Read the exact bytes.** `read` to pull a symbol's full source body.
   `read` is the `show` primitive — navigate only for analysis, never to read.
3. **Check the ripple.** If the edit could affect callers or dependents, run
   `callers`, `context`, or `impact` for the symbol before editing. This is the
   step most often skipped and most worth doing for a generation task.
4. **Edit surgically.** Submit the smallest possible edit (see below).
5. **Verify.** Run `verify` (tests + lint) after any change.
6. **Revise** the spec with `revise` when a failure means the *spec* is wrong, not
   just the code -- a structural contradiction or a gap the spec couldn't model
   (e.g. `trigger`, `data_model`, `modules`, `seams` were guessed and now break
   the build). `revise` hands control back to the outer loop to re-draft that
   one field and re-seeds you with the updated goal; it is not success. Do not
   `revise` for ordinary code fixes -- fix the code.
7. **Finish** only when the goal is met *and* verify passes.

## Editing: prefer symbol-scoped edits

An edit is a list of `{edits: [...]}`. Two shapes:

- **Symbol-scoped (prefer):** `{path, symbol, from, to}` — changes one substring
  inside a symbol. The harness fetches the exact source via `show`, so you never
  reproduce a file body from memory.
- **Exact block:** `{path, old, new}` — replaces one exact block in a file.
  To CREATE a brand-new file, use an empty `old` and put the entire new
  file content in `new`.

Rules for both:

- **`from`/`to` are exact source characters** — a slice of the body you read,
  NOT a quoted description. If the source shows `"hello"` with a trailing space,
  `from` is those exact characters including quotes and the space.
- **`from` must occur exactly once** in the symbol (symbol-scoped) or file
  (exact block). If it appears more than once, narrow it with surrounding lines
  until it is unique. A common substring like `return` or `self.` will be
  replaced everywhere and silently corrupt the symbol.
- **Copy `from` verbatim** from the source you read. It must match exactly or the
  harness rejects it.
- **Touch only what the goal requires** and match existing style.

## Handling rejections

A rejected edit is feedback, not a dead end. The result carries a `reason`:

- `"invalid decision"` → the action or its fields didn't match the schema; the
  `hint` lists the valid actions. Fix the decision and retry.

- `"old text not found"` → re-read the file/symbol, then retry with exact text.
- `"not unique"` → include more surrounding lines to disambiguate.
- `"invalid Python"` → your `new` broke syntax; fix and retry.
- `"denylist"` → the file is protected; do not edit it.

Do not guess or re-submit the same text. Re-read, adjust, retry. After
`max_stalls` (12) consecutive ticks with no progress (no applied edit, no
passing verify) the loop exits with a `stalled` finish.

## Parallel goals

For a multi-file goal, use `plan` with one subgoal per file. The paths MUST be
disjoint (no two subgoals share a file) so the harness can run them safely in
parallel and concatenate the results.

## Tools

- `plan`: `{subgoals:[{goal, path}]}` — decompose a multi-file goal into
  parallel subtasks; each targets a DIFFERENT file path (disjoint).
- `navigate`: ast-bro analysis tools — `map`, `digest`, `search`, `callers`,
  `callees`, `impact`, `context`, `surface`, `find_related`, `implements`.
- `read`: pull a symbol's full source body (`show`).
- `edit`: apply surgical edits.
- `verify`: run tests and lint, then report overall ok.
- `finish`: stop when the goal is met.

## Guardrails

- After an edit, verify before finishing.
- Never modify files on the denylist.
- Respond with a single JSON object only.
