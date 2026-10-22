"""Tests for the spec schema -- the single source of truth for the spec
vocabulary and the required fields an outer loop builds with the user.

Mirrors test_schema.py / test_prompt.py: the schema is the enforcement layer and
the prompt (``render_prompt``) is derived from it, so the drift guard here is
"the renderer emits every field name in the right group" -- if the schema changes,
the generated prompt changes with it.
"""

from hifi_code.spec_data import DATA_EVENT_SPEC
from hifi_code.spec_schema import Group, State


def _full_instance() -> dict:
    """A fully-resolved instance: every field present and well-typed."""
    return {
        "events": ["OrderPlaced"],
        "state": ["order: pending -> paid"],
        "trigger": "HTTP request arrives",
        "data_model": ["order: id: int, total: decimal, status: str"],
        "transitions": ["OrderPlaced -> order: paid"],
        "modules": ["domain/", "adapters/", "main.py"],
        "seams": ["Store protocol"],
        "side_effects": ["send confirmation email"],
        "inputs": ["order payload"],
        "outputs": ["201 created"],
        "constraints": ["no PII in logs"],
        "edge_cases": ["payment failed"],
        "acceptance": ["order persisted and email sent"],
        "assumptions": ["in-process queue"],
    }


def test_render_prompt_lists_every_field_in_its_group():
    # Drift guard: the generated prompt carries every field, grouped correctly.
    prompt = DATA_EVENT_SPEC.render_prompt()
    for f in DATA_EVENT_SPEC.fields:
        assert f.name in prompt
    for group in Group:
        assert f"## {group.value.title()}" in prompt


def test_render_prompt_marks_required_and_default():
    prompt = DATA_EVENT_SPEC.render_prompt()
    assert "(required)" in prompt
    assert "(default)" in prompt
    assert "- `events`:" in prompt
    assert "- `assumptions`:" in prompt


def test_render_draft_flags_gaps_as_assumptions():
    # A required field with a default is shown as a proposed assumption, not a gap.
    draft = DATA_EVENT_SPEC.render_draft({"events": ["OrderPlaced"]})
    assert "side_effects" in draft
    assert "assumed" in draft
    # A required field with no default is a genuine blocker.
    assert "NEEDS INPUT" in draft


def test_render_draft_marks_unblockable_gap():
    draft = DATA_EVENT_SPEC.render_draft({})
    assert "- `events`: **NEEDS INPUT**" in draft


def test_render_draft_flags_structural_gaps_as_needs_input():
    # Structural preconditions (trigger/data_model/modules/seams) are required with
    # no default, so an empty instance flags them as NEEDS INPUT rather than
    # silently assuming them -- the draft front-loads the skeleton.
    draft = DATA_EVENT_SPEC.render_draft({})
    for name in ("trigger", "data_model", "modules", "seams"):
        assert f"- `{name}`: **NEEDS INPUT**" in draft


def test_render_prompt_lists_structural_fields():
    # STRUCTURE group renders its fields grouped correctly.
    prompt = DATA_EVENT_SPEC.render_prompt()
    for name in ("modules", "seams"):
        assert f"- `{name}`:" in prompt
    assert "## Structure" in prompt


def test_validate_instance_requires_structural_fields():
    for name in ("trigger", "data_model", "modules", "seams"):
        bad = _full_instance()
        del bad[name]
        assert any(name in e for e in DATA_EVENT_SPEC.validate_instance(bad))


def test_render_draft_echoes_filled_fields():
    draft = DATA_EVENT_SPEC.render_draft({"events": ["OrderPlaced"]})
    assert "- `events`: ['OrderPlaced']" in draft


def test_validate_instance_passes_for_full_resolved():
    assert DATA_EVENT_SPEC.validate_instance(_full_instance()) == []


def test_validate_instance_rejects_missing_required():
    inst = _full_instance()
    del inst["transitions"]
    errs = DATA_EVENT_SPEC.validate_instance(inst)
    assert any("transitions" in e for e in errs)


def test_validate_instance_allows_gaps_in_draft():
    # Draft allows gaps; render_draft surfaces them instead of rejecting.
    assert DATA_EVENT_SPEC.validate_instance({}, state=State.DRAFT) == []


def test_validate_instance_defaulted_fields_are_optional_in_resolved():
    # constraints / edge_cases / assumptions are not required; a resolved spec
    # may omit them.
    inst = _full_instance()
    for opt in ("constraints", "edge_cases", "assumptions"):
        del inst[opt]
    assert DATA_EVENT_SPEC.validate_instance(inst) == []


def test_validate_instance_required_defaulted_fields_are_optional_in_resolved():
    # side_effects / outputs are required *with* a default ([]); leaving them
    # empty is a defaultable gap, not a blocker.
    inst = _full_instance()
    for opt in ("side_effects", "outputs"):
        del inst[opt]
    assert DATA_EVENT_SPEC.validate_instance(inst) == []


def test_render_goal_serializes_resolved_spec():
    goal = DATA_EVENT_SPEC.to_goal(_full_instance())
    assert "data/event-driven application" in goal
    assert "events" in goal
    assert "transitions" in goal
