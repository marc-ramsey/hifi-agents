"""The outer loop: draft a spec with the user, then hand it to the inner loop.

The spec is the contract between two loops of the same shape (see spec_schema):
an outer loop that drafts it interactively (feedback = the user) and the inner
coding loop (hifi_code's ``Planner``) that executes it (feedback = tests/LSP).
This module owns the outer loop.

The draft loop renders the current instance with ``SpecSchema.render_draft``,
turning unknowns into approve-or-veto assumptions. When the user resolves,
required-with-default gaps are materialized from their defaults and the instance
is validated against the resolved contract (``SpecSchema.validate_instance``);
if it fails, the errors are surfaced and the loop continues so the user can fill
them. On a valid spec, ``SpecSchema.to_goal`` serializes it into the goal string
the inner loop consumes -- the seam between the two loops.

Loop logic is separated from I/O behind :class:`SpecDriver` so it can be driven
by a scripted driver in tests and an ``input()``-based driver in the CLI.
"""

from dataclasses import dataclass
from typing import Any

from .spec_schema import Kind, RevisionAction, RevisionRequest, SpecSchema, State


@dataclass
class DraftOutcome:
    """The user's decision after seeing a draft.

    ``instance`` is the user's current view of the spec (the loop updates it on
    an edit). ``resolve`` asks the loop to validate and hand off; ``abort``
    cancels the whole run.
    """

    instance: dict[str, Any]
    resolve: bool = False
    abort: bool = False


class SpecDriver:
    """Interface between the outer loop and its user. Subclass for the CLI or tests."""

    def present_prompt(self, text: str) -> None:
        raise NotImplementedError

    def present_draft(self, draft: str) -> DraftOutcome:
        raise NotImplementedError

    def present_errors(self, errors: list[str]) -> None:
        raise NotImplementedError

    def present_goal(self, goal: str) -> bool:
        """Confirm the goal before handing off; default is to proceed."""
        return True


class SpecLoop:
    """Draft a spec with a user, resolve it, and return ``(instance, goal)``.

    The loop renders the current instance, shows gaps as proposed assumptions,
    and loops until the user resolves or aborts. On resolve it materializes
    required-with-default gaps from their defaults, validates the instance
    against the resolved contract, and -- if valid -- serializes it into the
    goal the inner loop consumes.
    """

    def __init__(self, spec: SpecSchema, driver: SpecDriver, max_rounds: int = 25):
        self.spec = spec
        self.driver = driver
        self.max_rounds = max_rounds
        # The last resolved instance, held so a revision can mutate it in place.
        self.resolved_instance: dict[str, Any] = {}

    async def run(self) -> tuple[dict[str, Any] | None, str]:
        """Draft + resolve. Return ``(instance, goal)`` or ``(None, "")`` if aborted."""
        self.driver.present_prompt(self.spec.render_prompt())
        instance: dict[str, Any] = {}
        for _ in range(self.max_rounds):
            outcome = self.driver.present_draft(self.spec.render_draft(instance))
            if outcome.abort:
                return None, ""
            if not outcome.resolve:
                instance = dict(outcome.instance)
                continue
            resolved = self._resolve(instance)
            errors = self.spec.validate_instance(resolved, State.RESOLVED)
            if errors:
                self.driver.present_errors(errors)
                continue
            goal = self.spec.to_goal(resolved)
            if not self.driver.present_goal(goal):
                return None, ""
            self.resolved_instance = resolved
            return resolved, goal
        return None, ""

    def _resolve(self, instance: dict[str, Any]) -> dict[str, Any]:
        """Fill required-with-default gaps from their defaults so validate passes.

        ``validate_instance`` only looks at required fields, so a required field
        with a default (e.g. ``side_effects=[]``) that the user left empty would
        read as missing. Materializing the default here is what makes such a gap
        "optional in resolved", matching the draft's "assumed X" presentation.
        """
        resolved = dict(instance)
        for f in self.spec.fields:
            if f.is_empty(resolved.get(f.name)) and f.default is not None:
                resolved[f.name] = f.default
        return resolved

    def revise(self, request: RevisionRequest):
        """Re-draft one committed field from a revision request and re-validate.

        The reverse of :meth:`run`: ``run`` moves a spec draft-to-resolved, this
        mutates a committed field when the inner loop finds the spec wrong. The
        field must exist in the schema -- an unknown field is a mechanism error,
        not a validation error. The requested op is applied, required-with-default
        gaps are re-materialized, and the instance is re-validated against the
        resolved contract. Returns ``(instance, goal)`` when valid, or the
        validation errors when it isn't, so the caller can surface them and stop
        rather than hand a broken spec to the coding loop.
        """
        field = next((f for f in self.spec.fields if f.name == request.field), None)
        if field is None:
            return [f"unknown field {request.field!r}"]
        resolved = dict(self.resolved_instance)
        current = resolved.get(field.name)
        if request.op == RevisionAction.ADD:
            resolved[field.name] = (current or []) + (
                [request.value] if request.value is not None else []
            )
        elif request.op == RevisionAction.REMOVE:
            resolved[field.name] = [x for x in (current or []) if x != request.value]
        else:  # REPLACE
            resolved[field.name] = request.value
        resolved = self._resolve(resolved)
        errors = self.spec.validate_instance(resolved, State.RESOLVED)
        if errors:
            return errors
        self.resolved_instance = resolved
        return resolved, self.spec.to_goal(resolved)


async def run_outer(settings, spec: SpecSchema, driver: SpecDriver, root: str = ".",
                    max_revise_rounds: int = 5):
    """Drive the outer loop, then hand the resolved goal to the inner loop.

    The spec is a living document: when the inner loop emits a structured
    ``revise`` request (a structural contradiction or an unmodeled gap), the
    outer loop applies it, re-validates, and re-seeds the inner loop with the
    updated goal. Revisions are bounded by ``max_revise_rounds`` so the loop
    terminates. Emits the inner loop's results as newline-delimited JSON
    (matching the stdio transport). Returns the final goal string on success, or
    ``None`` if the user aborted before resolving or a revision failed validation.
    """
    import json
    import logging

    from .transports.stdio import _drive

    logger = logging.getLogger(__name__)
    loop = SpecLoop(spec, driver)
    resolved, goal = await loop.run()
    if resolved is None:
        return None
    for _ in range(max_revise_rounds):
        logger.info("Handing resolved spec to the inner coding loop...")
        gen = _drive(settings, goal, root)
        revised = False
        try:
            async for result in gen:
                print(json.dumps(result, ensure_ascii=False))
                if result.get("action") == "revise" and result.get("ok"):
                    outcome = loop.revise(RevisionRequest.model_validate(result))
                    if isinstance(outcome, list):
                        driver.present_errors(outcome)
                        return None
                    revised = True
                    resolved, goal = outcome
                    goal = f"{goal}\n\n[spec revision] {result.get('reason', '')}"
                    break
        finally:
            # Close the generator (and its ast-bro session) whether we broke out
            # on a revise or it exhausted on a finish.
            await gen.aclose()
        if not revised:
            return goal
    logger.warning("spec revision budget (%s) exhausted; returning last goal",
                   max_revise_rounds)
    return goal


class CliDriver(SpecDriver):
    """``input()``-based driver for the real CLI.

    Presents the prompt once, then per draft asks the user to edit a field,
    resolve (accept all assumptions), or abort. List fields accept comma- or
    newline-separated values; other kinds are stored as-is.
    """

    def __init__(self, spec: SpecSchema, in_: Any = input):
        self.spec = spec
        self._in = in_
        self._instance: dict[str, Any] = {}

    def present_prompt(self, text: str) -> None:
        print(text)

    def present_draft(self, draft: str) -> DraftOutcome:
        print(draft)
        ans = self._in(
            "\n[resolve] accept all assumptions  "
            "[edit] change a field  "
            "[abort] cancel: "
        ).strip().lower()
        if ans in ("resolve", "r", "done", "yes", "y", ""):
            return DraftOutcome(instance=dict(self._instance), resolve=True)
        if ans in ("abort", "a", "quit", "q", "n", "no"):
            return DraftOutcome(instance={}, abort=True)
        return self._edit()

    def _edit(self) -> DraftOutcome:
        field = self._in("field name (or 'back'): ").strip()
        if field.lower() in ("back", "cancel", "abort"):
            return DraftOutcome(instance=dict(self._instance), abort=True)
        raw = self._in(f"value for {field}: ")
        self._apply(field, raw)
        return DraftOutcome(instance=dict(self._instance))

    def _apply(self, field: str, raw: str) -> None:
        f = next((f for f in self.spec.fields if f.name == field), None)
        if f is None:
            print(f"unknown field: {field}")
            return
        if f.kind == Kind.LIST:
            self._instance[field] = [
                s.strip() for s in raw.replace(",", "\n").splitlines() if s.strip()
            ]
        else:
            self._instance[field] = raw.strip()

    def present_errors(self, errors: list[str]) -> None:
        print("\n" + "\n".join("  - " + e for e in errors) + "\n")

    def present_goal(self, goal: str) -> bool:
        print("\n--- goal for the inner loop ---\n" + goal + "\n-----------------------------")
        return self._in("proceed? [Y/n]: ").strip().lower() not in ("n", "no", "abort")
