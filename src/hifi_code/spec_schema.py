"""Machine-checkable schema for the *spec* an outer loop builds with the user.

The spec is the contract between two loops of the same shape: an outer loop that
drafts it interactively (feedback = the user) and the inner coding loop
(``hifi_code``'s ``Planner``) that executes it (feedback = tests/LSP). This module
is the single source of truth for the spec vocabulary and the required fields of
each section, so the contract is *enforced* (a resolved spec is validated before
it is handed to the coding loop) rather than merely *advised* in a prompt.

Two states share one schema:

- ``draft`` -- gaps are allowed; every blocking gap is shown as a proposed
  default so the user can approve or veto.
- ``resolved`` -- every required field is committed. This is what the coding loop
  consumes.

``validate_instance`` enforces the resolved contract and returns the list of
missing/invalid fields (empty == valid). ``render_prompt`` produces the
declarative vocabulary/format/constraints injected into every subagent template
correct-by-construction, so it can't drift from the schema. ``render_draft``
drives the interactive loop by turning unknowns into "assumed X unless you
object".
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Kind(str, Enum):
    """The shape a field carries; used by ``validate_instance`` to type-check."""

    SCALAR = "scalar"
    LIST = "list"
    DICT = "dict"
    CHOICE = "choice"
    FLAG = "flag"


class State(str, Enum):
    """Which mode a spec instance is in. Draft allows gaps; resolved does not."""

    DRAFT = "draft"
    RESOLVED = "resolved"


class Group(str, Enum):
    """Which prompt section a field renders into -- the renderer's key."""

    VOCABULARY = "vocabulary"
    OUTPUT = "output"
    TRANSITIONS = "transitions"
    CONSTRAINTS = "constraints"


class FieldSpec(BaseModel):
    """One field of a spec: what it is, where it renders, and how it behaves.

    ``required`` + ``default`` drive draft-vs-resolved behavior: a required field
    with a default is a *defaultable* gap (shown as a proposed assumption in draft,
    optional in resolved); a required field with no default is a genuine blocker
    (shown as NEEDS INPUT in draft, rejected in resolved).
    """

    name: str
    kind: Kind
    group: Group
    required: bool = True
    default: Any = None
    default_note: str | None = None
    description: str = ""
    choices: list[str] | None = None

    def is_empty(self, value: Any) -> bool:
        """True when a value counts as unset for rendering/validation.

        An empty list is treated as unset: a spec section with no entries is the
        same as an absent one for the purposes of "did the user fill this in?".
        """
        return value is None or (isinstance(value, list) and len(value) == 0)


class SpecSchema(BaseModel):
    """The full spec: a name and an ordered list of fields.

    A schema is data, not behavior -- the concrete default (a data/event-driven
    application) lives in ``spec_data``. This class is the engine that renders it
    into a prompt, into a draft, and validates an instance against it.
    """

    name: str
    fields: list[FieldSpec] = Field(default_factory=list)

    def render_prompt(self) -> str:
        """Declarative vocabulary/format/constraints, grouped.

        This is what gets spliced into every subagent template so the template's
        declarative parts can't drift from the schema -- the renderer is the only
        writer of these sections.
        """
        out = [f"# Spec schema: {self.name}\n"]
        for group in Group:
            fields = [f for f in self.fields if f.group == group]
            if not fields:
                continue
            out.append(f"## {group.value.title()}\n")
            for f in fields:
                marker = " (required)" if f.required else " (default)"
                line = f"- `{f.name}`: {f.description}{marker}"
                if f.choices:
                    line += f" -- one of {', '.join(f.choices)}"
                out.append(line)
            out.append("")
        return "\n".join(out).rstrip()

    def render_draft(self, instance: dict[str, Any]) -> str:
        """Turn an instance into the draft shown to the user for verification.

        A filled field is echoed back. An absent field that carries a default becomes
        a proposed assumption ("assumed X unless you object"); an absent required
        field with no default becomes NEEDS INPUT. An absent optional field with no
        default is fine and echoed as-is. This is the whole termination mechanism of
        the interactive loop: unknowns surface as approve-or-veto values instead of
        open questions.

        The default check is ``f.default is not None`` rather than ``not is_empty``:
        an empty-list default (``[]``) is a real proposed assumption, not "no default".
        """
        lines = [f"## Draft spec: {self.name}\n"]
        for f in self.fields:
            val = instance.get(f.name)
            if not f.is_empty(val):
                lines.append(f"- `{f.name}`: {val!r}")
            elif f.default is not None:
                note = f" {f.default_note}" if f.default_note else ""
                lines.append(f"- `{f.name}`: {f.default!r}  -- assumed{note}")
            elif f.required:
                lines.append(f"- `{f.name}`: **NEEDS INPUT**")
            else:
                lines.append(f"- `{f.name}`: {val!r}")
        return "\n".join(lines)

    def validate_instance(self, instance: dict[str, Any],
                          state: State = State.RESOLVED) -> list[str]:
        """Return the list of problems with ``instance``; empty == valid.

        In ``draft`` state nothing is an error -- gaps are expected and surfaced by
        ``render_draft`` instead. In ``resolved`` state every required field must
        be present, non-empty, and the right kind/choice. This is the gate the
        coding loop checks before it consumes a spec, mirroring
        ``validate_decision`` for the action schema.
        """
        errors: list[str] = []
        if state == State.DRAFT:
            return errors
        for f in self.fields:
            if not f.required:
                continue
            val = instance.get(f.name)
            if f.is_empty(val):
                # A required field with a default is a *defaultable* gap: leaving
                # it empty is fine (the default applies), so only a required field
                # with no default is a genuine blocker. An empty list counts as
                # empty here, so a ``[]`` default is not rejected.
                if f.default is None:
                    errors.append(f"missing required field {f.name!r}")
                continue
            if f.kind == Kind.LIST and not isinstance(val, list):
                errors.append(f"{f.name!r} must be list")
            elif f.kind == Kind.SCALAR and not isinstance(val, str):
                errors.append(f"{f.name!r} must be str")
            elif f.kind == Kind.DICT and not isinstance(val, dict):
                errors.append(f"{f.name!r} must be dict")
            elif f.kind == Kind.CHOICE and f.choices and val not in f.choices:
                errors.append(f"{f.name!r} must be one of {f.choices}")
        return errors

    def to_goal(self, instance: dict[str, Any]) -> str:
        """Serialize a resolved spec into the goal string the inner loop consumes.

        This is the seam between the two loops: the spec becomes the coding loop's
        goal, so a resolved spec can be handed straight to ``Planner.seed`` (or a
        subgoal's ``goal``) without the model re-deriving intent from scratch.
        """
        lines = [f"Build a {self.name}:\n"]
        for f in self.fields:
            val = instance.get(f.name)
            if f.is_empty(val):
                continue
            lines.append(f"{f.group.value} {f.name}: {val}")
        return "\n".join(lines)
