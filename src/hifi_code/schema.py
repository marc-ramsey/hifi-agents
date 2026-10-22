"""Machine-checkable schema for the loop driver's decisions.

The driver is a stateless per-tick model: each tick it emits one JSON decision
and is torn down. This module is the single source of truth for the action
vocabulary and the required fields of each action, so the contract is
*enforced* (invalid decisions are rejected before dispatch) rather than merely
*advised* in the prompt.

``validate_decision`` runs on every parsed decision. It returns the decision
unchanged when valid and raises :class:`SchemaError` (with a specific,
actionable message) when the action is unknown or a required field is missing
or the wrong type. ``Planner.tick`` catches that and rejects the decision, so
the model gets structured feedback it can self-correct from.
"""

from typing import Any


class Action:
    """The loop's action vocabulary. Kept as a plain string set so the schema
    keys line up with the raw strings a model emits (``navigate`` etc.)."""

    NAVIGATE = "navigate"
    READ = "read"
    EDIT = "edit"
    PLAN = "plan"
    VERIFY = "verify"
    REVISE = "revise"
    FINISH = "finish"

    ALL = (NAVIGATE, READ, EDIT, PLAN, VERIFY, REVISE, FINISH)


class SchemaError(ValueError):
    """A parsed decision violated the action schema."""


# Required fields per action: field name -> expected type. Params the model
# sometimes nests under ``args`` are merged into the top level before checking,
# so a decision may declare its fields either way.
_REQUIRED: dict[str, dict[str, type]] = {
    Action.NAVIGATE: {"tool": str},
    Action.READ: {"path": str},
    Action.EDIT: {"edits": list},
    Action.PLAN: {"subgoals": list},
    Action.VERIFY: {},
    Action.REVISE: {"field": str, "op": str},
    Action.FINISH: {},
}


def _merge_params(decision: dict[str, Any]) -> dict[str, Any]:
    """Merge params that may live at the top level or nested under ``args``."""
    params = dict(decision.get("args") or {})
    params.update(
        {k: v for k, v in decision.items() if k not in ("args", "action")}
    )
    return params


def validate_decision(decision: dict[str, Any]) -> dict[str, Any]:
    """Validate a parsed decision against the action schema.

    Returns ``decision`` unchanged when valid; raises :class:`SchemaError` with
    a specific message when the action is unknown or a required field is missing
    or the wrong type.
    """
    action = decision.get("action")
    if action not in _REQUIRED:
        raise SchemaError(
            f"unknown action {action!r}; valid actions: "
            + " | ".join(Action.ALL)
        )
    params = _merge_params(decision)
    for field, expected in _REQUIRED[action].items():
        if field not in params:
            raise SchemaError(
                f"{action!r} decision missing required field {field!r}"
            )
        if not isinstance(params[field], expected):
            raise SchemaError(
                f"{action!r} field {field!r} must be {expected.__name__}"
            )
    return decision
