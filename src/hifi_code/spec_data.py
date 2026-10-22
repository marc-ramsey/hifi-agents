"""The default spec: a data/event-driven application.

This is the concrete schema an outer loop fills in with the user. It encodes the
methodology -- infer events, state, transitions, and side effects, and only those
-- as a set of fields with required/default annotations the renderer and
validator key off. Fields are ordered so the draft reads in real-time order:
events -> state -> transitions -> side effects -> inputs/outputs -> constraints.
"""

from .spec_schema import FieldSpec, Group, Kind, SpecSchema

DATA_EVENT_SPEC = SpecSchema(
    name="data/event-driven application",
    fields=[
        FieldSpec(
            name="events", kind=Kind.LIST, group=Group.VOCABULARY, required=True,
            description="past-tense facts that trigger a state change, e.g. OrderPlaced, EmailReceived",
        ),
        FieldSpec(
            name="state", kind=Kind.LIST, group=Group.VOCABULARY, required=True,
            description="entities and the states each can be in",
        ),
        FieldSpec(
            name="transitions", kind=Kind.LIST, group=Group.TRANSITIONS, required=True,
            description="for each event: what state it changes and to what",
        ),
        FieldSpec(
            name="side_effects", kind=Kind.LIST, group=Group.OUTPUT, required=True,
            default=[],
            default_note="assume none unless the user specifies one",
            description="observable consequences: email sent, stream written, HTTP response returned",
        ),
        FieldSpec(
            name="inputs", kind=Kind.LIST, group=Group.VOCABULARY, required=True,
            description="data that enters the system",
        ),
        FieldSpec(
            name="outputs", kind=Kind.LIST, group=Group.OUTPUT, required=True,
            default=[],
            description="what leaves: API responses, notifications, downstream writes",
        ),
        FieldSpec(
            name="constraints", kind=Kind.LIST, group=Group.CONSTRAINTS, required=False,
            default=[],
            description="non-functional: perf, security, retention, do-not-touch",
        ),
        FieldSpec(
            name="edge_cases", kind=Kind.LIST, group=Group.CONSTRAINTS, required=False,
            default=[],
            default_note="assume happy-path only unless the user specifies",
            description="failures, empty state, concurrency",
        ),
        FieldSpec(
            name="acceptance", kind=Kind.LIST, group=Group.OUTPUT, required=True,
            description="how each path is verified",
        ),
        FieldSpec(
            name="assumptions", kind=Kind.LIST, group=Group.CONSTRAINTS, required=False,
            default=[],
            description="explicitly-listed defaults the agent chose",
        ),
    ],
)
