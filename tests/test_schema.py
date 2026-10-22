"""Tests for the decision schema -- the single source of truth for the action
vocabulary and the required fields of each action."""

import pytest

from hifi_code.schema import Action, SchemaError, validate_decision

VALID = {
    "navigate": {"action": "navigate", "tool": "map"},
    "read": {"action": "read", "path": "x.py"},
    "edit": {"action": "edit", "edits": [{"path": "x.py", "old": "a", "new": "b"}]},
    "plan": {"action": "plan", "subgoals": [{"goal": "g", "path": "x.py"}]},
    "verify": {"action": "verify"},
    "revise": {"action": "revise", "field": "modules", "op": "add"},
    "finish": {"action": "finish", "reason": "done"},
}


@pytest.mark.parametrize("decision", list(VALID.values()))
def test_valid_decisions_pass_through(decision):
    # Returns the same object unchanged so callers can use it in place.
    assert validate_decision(decision) is decision


def test_unknown_action_is_rejected():
    with pytest.raises(SchemaError, match="unknown action"):
        validate_decision({"action": "run"})


def test_missing_required_field_is_rejected():
    with pytest.raises(SchemaError, match="missing required field 'tool'"):
        validate_decision({"action": "navigate"})


def test_wrong_type_is_rejected():
    with pytest.raises(SchemaError, match="must be str"):
        validate_decision({"action": "read", "path": 5})


def test_edit_requires_edits_list():
    # A single-object edit ({path, old, new}) is not the schema; edits is a list.
    with pytest.raises(SchemaError, match="missing required field 'edits'"):
        validate_decision({"action": "edit", "path": "x.py", "old": "a", "new": "b"})


def test_revise_requires_field_and_op():
    # revise needs a field to re-draft and an op (add|remove|replace).
    with pytest.raises(SchemaError, match="missing required field 'field'"):
        validate_decision({"action": "revise", "op": "add"})
    with pytest.raises(SchemaError, match="missing required field 'op'"):
        validate_decision({"action": "revise", "field": "modules"})


def test_nested_args_are_merged():
    # The model may nest params under `args`; they validate the same as top-level.
    # validate_decision returns the decision unchanged (merge is validation-only).
    assert validate_decision({"action": "navigate", "args": {"tool": "map"}}) == {
        "action": "navigate", "args": {"tool": "map"}
    }


def test_error_message_lists_valid_actions():
    with pytest.raises(SchemaError) as excinfo:
        validate_decision({"action": "bogus"})
    for action in Action.ALL:
        assert action in str(excinfo.value)
