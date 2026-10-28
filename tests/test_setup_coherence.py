"""Coherence checks across the experiment setup classes.

Each class has a reference setup (baseline, strict_flow) and three variants:
`_flag` observes only, `_soft` enforces through LLM judges, `_hard` enforces through
deterministic predicates. Variants may differ from their reference only in how a
rule is evaluated, never in what the agents are told or which rules exist.

For hard rules the gateway uses the predicate's own verdict effect and ignores the
YAML `effect`, so the effective effect is computed from `predicate_args.effect`, the
factory default, or the dual-use direct-call behaviour.
"""

import inspect
import unittest
from pathlib import Path

import yaml

from src.control_plane.agent_repo import AgentRepo
from src.control_plane.catalog import Catalog
from src.control_plane.predicates import PREDICATE_REGISTRY

SETUPS_DIR = Path("config/setups")
CLASSES = {
    "baseline": ["baseline_flag", "baseline_soft", "baseline_hard"],
    "strict_flow": ["strict_flow_flag", "strict_flow_soft", "strict_flow_hard"],
}
# Dual-use predicates called with a bare context (no predicate_args).
_DIRECT_CALL_EFFECT = {
    "allowed_handover_targets": "deny",
    "transfer_includes_order_id": "flag",
}
_STRENGTH = {"allow": 0, "flag": 1, "deny": 2}


_ALL_SETUPS = [
    s for reference, variants in CLASSES.items() for s in (reference, *variants)
]
# Judge template variable -> the state_dependencies key that declares it.
_CONTEXT_VARIABLES = {
    "tool_results": "tool_results",
    "recent_dialogue": "recent_dialogue",
    "recent_order": "recent_order",
    "order_status": "order",
    "order_total": "order",
}


def _entries(setup: str) -> list[dict]:
    path = SETUPS_DIR / setup / "guardrails" / "coffee_shop.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["guardrails"]


def _effective_effect(entry: dict) -> str:
    if entry["type"] == "soft":
        return entry.get("effect", "flag")
    args = entry.get("predicate_args") or {}
    if "effect" in args:
        return args["effect"]
    predicate = entry["predicate"]
    if not args and predicate in _DIRECT_CALL_EFFECT:
        return _DIRECT_CALL_EFFECT[predicate]
    return inspect.signature(PREDICATE_REGISTRY[predicate]).parameters["effect"].default


def _concept(guardrail_id: str) -> str:
    return guardrail_id.removeprefix("soft_")


def _concept_effects(setup: str) -> dict[str, str]:
    """Strongest effective effect per rule concept (a soft judge and its hard
    shadow are one concept)."""
    effects: dict[str, str] = {}
    for entry in _entries(setup):
        concept = _concept(entry["id"])
        effect = _effective_effect(entry)
        if _STRENGTH[effect] > _STRENGTH.get(effects.get(concept, "allow"), 0):
            effects[concept] = effect
    return effects


def _agent_concepts(setup: str) -> dict[str, set[str]]:
    return {
        agent.id: {_concept(g) for g in agent.guardrail_ids}
        for agent in AgentRepo(SETUPS_DIR / setup).all().values()
    }


class TestSetupCoherence(unittest.TestCase):
    def test_every_guardrail_defined_and_attached(self):
        for setups in CLASSES.items():
            for setup in (setups[0], *setups[1]):
                with self.subTest(setup=setup):
                    catalog = Catalog(SETUPS_DIR / setup)
                    attached = set()
                    for agent in AgentRepo(SETUPS_DIR / setup).all().values():
                        catalog.guardrails(list(agent.guardrail_ids))
                        attached.update(agent.guardrail_ids)
                    defined = {e["id"] for e in _entries(setup)}
                    self.assertEqual(
                        defined - attached, set(), "defined but never attached"
                    )

    def test_agents_identical_within_class(self):
        for reference, variants in CLASSES.items():
            ref_agents = AgentRepo(SETUPS_DIR / reference).all()
            for setup in variants:
                agents = AgentRepo(SETUPS_DIR / setup).all()
                self.assertEqual(set(agents), set(ref_agents))
                for agent_id, ref in ref_agents.items():
                    with self.subTest(setup=setup, agent=agent_id):
                        agent = agents[agent_id]
                        self.assertEqual(agent.base_prompt, ref.base_prompt)
                        self.assertEqual(agent.tools, ref.tools)
                        self.assertEqual(agent.allowed_handovers, ref.allowed_handovers)
                        self.assertEqual(agent.guideline_ids, ref.guideline_ids)

    def test_guidelines_identical_within_class(self):
        for reference, variants in CLASSES.items():
            ref = (
                SETUPS_DIR / reference / "guidelines" / "coffee_shop.yaml"
            ).read_text()
            for setup in variants:
                with self.subTest(setup=setup):
                    path = SETUPS_DIR / setup / "guidelines" / "coffee_shop.yaml"
                    self.assertEqual(path.read_text(), ref)

    def test_same_rules_on_same_agents_within_class(self):
        for reference, variants in CLASSES.items():
            ref = _agent_concepts(reference)
            for setup in variants:
                with self.subTest(setup=setup):
                    self.assertEqual(_agent_concepts(setup), ref)

    def test_flag_setups_never_deny(self):
        for reference, variants in CLASSES.items():
            setup = f"{reference}_flag"
            for entry in _entries(setup):
                with self.subTest(setup=setup, guardrail=entry["id"]):
                    self.assertEqual(_effective_effect(entry), "flag")

    def test_enforcing_variants_match_reference_strength(self):
        for reference, _ in CLASSES.items():
            ref = _concept_effects(reference)
            for setup in (f"{reference}_soft", f"{reference}_hard"):
                with self.subTest(setup=setup):
                    self.assertEqual(_concept_effects(setup), ref)

    def test_hard_setups_are_all_hard(self):
        for reference in CLASSES:
            for entry in _entries(f"{reference}_hard"):
                with self.subTest(setup=f"{reference}_hard", guardrail=entry["id"]):
                    self.assertEqual(entry["type"], "hard")

    def test_soft_setups_enforce_through_judges(self):
        # A hard rule may only deny in a soft setup when the concept has no judge
        # (the deterministic max_calls counters).
        for reference in CLASSES:
            setup = f"{reference}_soft"
            judged = {_concept(e["id"]) for e in _entries(setup) if e["type"] == "soft"}
            for entry in _entries(setup):
                if entry["type"] == "hard" and _concept(entry["id"]) in judged:
                    with self.subTest(setup=setup, guardrail=entry["id"]):
                        self.assertEqual(_effective_effect(entry), "flag")

    def test_hard_rule_limits_match_reference(self):
        def limits(setup: str) -> dict[str, dict]:
            return {
                e["id"]: {
                    k: v
                    for k, v in (e.get("predicate_args") or {}).items()
                    if k != "effect"
                }
                for e in _entries(setup)
                if e["type"] == "hard"
            }

        for reference, variants in CLASSES.items():
            ref = limits(reference)
            for setup in variants:
                for guardrail_id, args in limits(setup).items():
                    if guardrail_id in ref:
                        with self.subTest(setup=setup, guardrail=guardrail_id):
                            self.assertEqual(args, ref[guardrail_id])

    def test_judge_copies_identical_across_setups(self):
        # A judge id means one judge: the same prompt, template, context and
        # version wherever it runs (the native judges run in six setups).
        fields = ("judge_prompt", "user_template", "state_dependencies", "version")
        seen: dict[str, tuple[str, dict]] = {}
        for setup in _ALL_SETUPS:
            for entry in _entries(setup):
                if entry["type"] != "soft":
                    continue
                judge = {f: entry.get(f) for f in fields}
                first = seen.setdefault(entry["id"], (setup, judge))
                with self.subTest(guardrail=entry["id"], setup=setup, first=first[0]):
                    self.assertEqual(judge, first[1])

    def test_correction_texts_identical_across_setups(self):
        # A rule gives the agent the same correction hint and fallback reply
        # whether a judge or a hard predicate enforces it, in every setup.
        fields = ("correction_hint", "fallback_message")
        seen: dict[str, tuple[str, dict]] = {}
        for setup in _ALL_SETUPS:
            for entry in _entries(setup):
                texts = {f: entry.get(f, "") for f in fields}
                first = seen.setdefault(_concept(entry["id"]), (setup, texts))
                with self.subTest(guardrail=entry["id"], setup=setup, first=first[0]):
                    self.assertEqual(texts, first[1])

    def test_judge_context_declared(self):
        # Every context variable a judge prompt uses is declared in
        # state_dependencies, and every declared dependency is used.
        for setup in _ALL_SETUPS:
            for entry in _entries(setup):
                if entry["type"] != "soft":
                    continue
                text = entry.get("judge_prompt", "") + entry.get("user_template", "")
                used = {
                    dep
                    for variable, dep in _CONTEXT_VARIABLES.items()
                    if "{" + variable + "}" in text
                }
                with self.subTest(setup=setup, guardrail=entry["id"]):
                    self.assertEqual(set(entry.get("state_dependencies") or []), used)


if __name__ == "__main__":
    unittest.main()
