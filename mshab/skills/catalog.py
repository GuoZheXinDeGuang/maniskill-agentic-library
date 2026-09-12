"""Canonical, machine-independent skill-graph catalog documents.

The checked-in catalog is a portable description of the four-layer library,
not an inventory of one workstation.  Checkpoint readiness is deliberately
excluded: it belongs to :class:`SkillLibrary` runtime discovery.  Loading a
catalog reconstructs and validates Layer 1/2 rather than trusting duplicated
derived views such as flattened nodes, edges, or execution order.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

from mshab.skills import schema
from mshab.skills.graph import FunctionalGoalGraph, SkillCompositionGraph
from mshab.skills.plan import SkillPlan


CATALOG_SCHEMA_VERSION = "mshab.skill-catalog.v1"
_LAYER_KEYS = (
    "1_functional_goal_graph",
    "2_skill_composition_graph",
    "3_bound_skill_contracts",
    "4_atomic_skills_and_backends",
)


class SkillCatalog:
    """Validated four-layer artifact suitable for source control and VLM input."""

    def __init__(
        self,
        *,
        task: str,
        construction: Mapping[str, Any],
        execution_plan_note: str,
        execution_plans: Mapping[str, SkillPlan],
        goal_graph: FunctionalGoalGraph,
        skill_graph: SkillCompositionGraph,
        bound_contracts: Mapping[str, Mapping[str, Any]],
        atomic_skills: Sequence[Mapping[str, Any]],
    ) -> None:
        schema.require_identifier({"task": task}, "task", where="skill_catalog")
        if skill_graph.task != task:
            raise ValueError("catalog task must match its skill composition graph")
        if skill_graph.goal_graph is not goal_graph:
            raise ValueError("catalog graphs must share the same goal graph object")
        skill_graph.validate()
        if not isinstance(construction, Mapping):
            raise TypeError("catalog construction metadata must be a mapping")
        if not isinstance(execution_plan_note, str) or not execution_plan_note:
            raise ValueError("catalog execution-plan note must be a non-empty string")
        if not isinstance(execution_plans, Mapping):
            raise TypeError("catalog execution_plans must be a mapping")
        if not isinstance(bound_contracts, Mapping):
            raise TypeError("catalog bound_contracts must be a mapping")

        self.task = task
        self.construction = _json_copy(construction, "construction")
        self.execution_plan_note = execution_plan_note
        self.execution_plans = dict(execution_plans)
        self.goal_graph = goal_graph
        self.skill_graph = skill_graph
        self.bound_contracts = _json_copy(bound_contracts, "bound_contracts")
        self.atomic_skills = tuple(_json_copy(atomic_skills, "atomic_skills"))
        self._validate_plans()
        self._validate_contracts()
        self._validate_atomic_skills()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillCatalog":
        where = "skill_catalog"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=(
                "schema_version",
                "task",
                "construction",
                "execution_plans",
                "summary",
                "layers",
            ),
        )
        version = schema.require_str(payload, "schema_version", where=where)
        if version != CATALOG_SCHEMA_VERSION:
            raise schema.SchemaError(
                "{} declares unsupported schema_version {!r}; expected {!r}".format(
                    where, version, CATALOG_SCHEMA_VERSION
                )
            )
        task = schema.require_identifier(payload, "task", where=where)
        layers = schema.require_mapping(payload["layers"], where="skill_catalog.layers")
        schema.require_keys(layers, where="skill_catalog.layers", required=_LAYER_KEYS)

        raw_layer_1 = schema.require_mapping(
            layers[_LAYER_KEYS[0]], where="skill_catalog.layer_1"
        )
        schema.require_keys(
            raw_layer_1,
            where="skill_catalog.layer_1",
            required=("instruction", "goals", "dependencies", "execution_order"),
        )
        goal_graph = FunctionalGoalGraph.from_dict(raw_layer_1)
        declared_goal_order = schema.require_str_tuple(
            raw_layer_1, "execution_order", where="skill_catalog.layer_1"
        )
        if declared_goal_order != goal_graph.execution_order():
            raise schema.SchemaError("catalog Layer-1 execution_order is stale")

        raw_layer_2 = schema.require_mapping(
            layers[_LAYER_KEYS[1]], where="skill_catalog.layer_2"
        )
        schema.require_keys(
            raw_layer_2,
            where="skill_catalog.layer_2",
            required=(
                "task",
                "subgraphs",
                "subgraph_relations",
                "_derived",
                "nodes",
                "edges",
                "candidate_partial_order",
            ),
        )
        skill_graph = SkillCompositionGraph.from_dict(
            raw_layer_2, goal_graph=goal_graph
        )
        declared_candidate_order = schema.require_str_tuple(
            raw_layer_2,
            "candidate_partial_order",
            where="skill_catalog.layer_2",
        )
        if declared_candidate_order != skill_graph.execution_order():
            raise schema.SchemaError("catalog candidate_partial_order is stale")
        regenerated_layer_2 = skill_graph.as_dict()
        regenerated_layer_2.pop("goal_graph", None)
        for key in ("nodes", "edges"):
            if raw_layer_2[key] != regenerated_layer_2[key]:
                raise schema.SchemaError(
                    "catalog Layer-2 derived {} view is stale".format(key)
                )
        skill_graph.validate()

        plan_section = schema.require_mapping(
            payload["execution_plans"], where="skill_catalog.execution_plans"
        )
        schema.require_keys(
            plan_section,
            where="skill_catalog.execution_plans",
            required=("note",),
            optional=tuple(key for key in plan_section if key != "note"),
        )
        note = schema.require_str(plan_section, "note", where="skill_catalog.execution_plans")
        plans = {
            name: SkillPlan.from_dict(value)
            for name, value in plan_section.items()
            if name != "note"
        }
        if not plans:
            raise schema.SchemaError("catalog must contain at least one execution plan")

        catalog = cls(
            task=task,
            construction=schema.require_mapping(
                payload["construction"], where="skill_catalog.construction"
            ),
            execution_plan_note=note,
            execution_plans=plans,
            goal_graph=goal_graph,
            skill_graph=skill_graph,
            bound_contracts=schema.require_mapping(
                layers[_LAYER_KEYS[2]], where="skill_catalog.layer_3"
            ),
            atomic_skills=schema.require_sequence(
                layers, _LAYER_KEYS[3], where="skill_catalog.layers"
            ),
        )
        if _json_copy(payload["summary"], "summary") != catalog.summary:
            raise schema.SchemaError("catalog summary is stale")
        return catalog

    @property
    def summary(self) -> Dict[str, int]:
        nominal = self.execution_plans.get("nominal")
        if nominal is None:
            nominal = next(iter(self.execution_plans.values()))
        return {
            "functional_goals": len(self.goal_graph.goals),
            "goal_skill_subgraphs": len(self.skill_graph.subgraphs),
            "subgraph_relations": len(self.skill_graph.subgraph_relations),
            "skill_nodes": len(self.skill_graph.nodes),
            "skill_edges": len(self.skill_graph.edges),
            "planned_steps": len(nominal.order),
            "bound_contracts": len(self.bound_contracts),
            "registered_atomic_skills": len(self.atomic_skills),
        }

    def as_dict(self) -> Dict[str, Any]:
        layer_1 = self.goal_graph.as_dict()
        layer_1["execution_order"] = list(self.goal_graph.execution_order())
        layer_2 = self.skill_graph.as_dict()
        layer_2.pop("goal_graph", None)
        layer_2["candidate_partial_order"] = list(
            self.skill_graph.execution_order()
        )
        plans = {"note": self.execution_plan_note}
        plans.update(
            (name, plan.as_dict()) for name, plan in self.execution_plans.items()
        )
        return {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "task": self.task,
            "construction": _json_copy(self.construction, "construction"),
            "execution_plans": plans,
            "summary": self.summary,
            "layers": {
                _LAYER_KEYS[0]: layer_1,
                _LAYER_KEYS[1]: layer_2,
                _LAYER_KEYS[2]: _json_copy(
                    self.bound_contracts, "bound_contracts"
                ),
                _LAYER_KEYS[3]: list(
                    _json_copy(self.atomic_skills, "atomic_skills")
                ),
            },
        }

    def _validate_plans(self) -> None:
        goals = self.goal_graph.goals
        nodes = self.skill_graph.nodes
        for name, plan in self.execution_plans.items():
            if not isinstance(name, str) or not name:
                raise TypeError("execution plan names must be non-empty strings")
            if not isinstance(plan, SkillPlan):
                raise TypeError("execution_plans must contain SkillPlan values")
            if set(plan.selections) != set(goals):
                raise ValueError(
                    "execution plan {!r} must select one achiever for every goal".format(
                        name
                    )
                )
            unknown = sorted(set(plan.order) - set(nodes))
            if unknown:
                raise ValueError(
                    "execution plan {!r} contains unknown nodes {}".format(name, unknown)
                )
            for goal_id, node_id in plan.selections.items():
                if goal_id not in nodes[node_id].achieves:
                    raise ValueError(
                        "execution plan {!r} selects node {!r} for unrelated goal {!r}".format(
                            name, node_id, goal_id
                        )
                    )
                executed_achievers = [
                    candidate
                    for candidate in plan.order
                    if goal_id in nodes[candidate].achieves
                ]
                if executed_achievers != [node_id]:
                    raise ValueError(
                        "execution plan {!r} must execute exactly its selected "
                        "achiever for goal {!r}".format(name, goal_id)
                    )
            completed = set()
            for node_id in plan.order:
                unsatisfied = [
                    sorted(group)
                    for group in self.skill_graph.prerequisite_groups(node_id)
                    if not group & completed
                ]
                if unsatisfied:
                    raise ValueError(
                        "execution plan {!r} schedules {!r} before prerequisite "
                        "groups {}".format(name, node_id, unsatisfied)
                    )
                completed.add(node_id)

    def _validate_contracts(self) -> None:
        if set(self.bound_contracts) != set(self.skill_graph.nodes):
            raise ValueError("catalog must contain one bound contract per skill node")
        for node_id, raw in self.bound_contracts.items():
            where = "bound_contracts.{}".format(node_id)
            record = schema.require_mapping(raw, where=where)
            schema.require_keys(
                record,
                where=where,
                required=(
                    "skill_id",
                    "arguments",
                    "preconditions",
                    "effects",
                    "invariants",
                    "verification",
                    "failure_modes",
                    "deletes",
                ),
            )
            node = self.skill_graph.nodes[node_id]
            if schema.require_skill_id(record, "skill_id", where=where) != node.skill_id:
                raise ValueError("catalog contract skill_id does not match its node")
            if schema.require_arguments(record, "arguments", where=where) != dict(
                node.arguments
            ):
                raise ValueError("catalog contract arguments do not match its node")
            for key in (
                "preconditions",
                "effects",
                "invariants",
                "verification",
                "failure_modes",
                "deletes",
            ):
                schema.require_str_tuple(record, key, where=where)

    def _validate_atomic_skills(self) -> None:
        ids = set()
        for index, raw in enumerate(self.atomic_skills):
            where = "atomic_skills[{}]".format(index)
            record = schema.require_mapping(raw, where=where)
            schema.require_keys(
                record,
                where=where,
                required=(
                    "id",
                    "skill_type",
                    "target",
                    "env_id",
                    "max_episode_steps",
                    "backends",
                ),
            )
            skill_id = schema.require_skill_id(record, "id", where=where)
            if skill_id in ids:
                raise ValueError("duplicate atomic skill id {!r}".format(skill_id))
            ids.add(skill_id)
            schema.require_str(record, "skill_type", where=where)
            schema.require_str(record, "target", where=where)
            schema.require_str(record, "env_id", where=where)
            steps = record["max_episode_steps"]
            if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
                raise schema.SchemaError("{}.max_episode_steps must be positive".format(where))
            backends = schema.require_mapping(
                record["backends"], where="{}.backends".format(where)
            )
            if not backends:
                raise schema.SchemaError("{}.backends cannot be empty".format(where))
            for backend_key, raw_backend in backends.items():
                schema.require_identifier(
                    {"key": backend_key}, "key", where="{}.backends".format(where)
                )
                backend_where = "{}.backends.{}".format(where, backend_key)
                backend = schema.require_mapping(raw_backend, where=backend_where)
                schema.require_keys(
                    backend,
                    where=backend_where,
                    required=("executor_type",),
                    optional=(
                        "family",
                        "policy_type",
                        "checkpoint",
                        "config",
                        "checkpoint_sha256",
                    ),
                )
                schema.require_str(backend, "executor_type", where=backend_where)
                for key in (
                    "family",
                    "policy_type",
                    "checkpoint",
                    "config",
                    "checkpoint_sha256",
                ):
                    if key in backend and backend[key] is not None:
                        schema.require_str(backend, key, where=backend_where)
        referenced = {node.skill_id for node in self.skill_graph.nodes.values()}
        missing = sorted(referenced - ids)
        if missing:
            raise ValueError("catalog has no atomic skill records for {}".format(missing))


def _json_copy(value: Any, where: str) -> Any:
    """Return a detached JSON value and reject non-serialisable metadata."""

    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise TypeError("{} must be JSON-serialisable".format(where)) from exc
