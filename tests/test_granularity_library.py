import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mshab.experiments.granularity import (
    EXPERIMENT_TASK,
    POLICY_SPECS,
    PolicySpec,
    build_granularity_library,
    contract_id,
    library_document,
    spec_for,
)
from mshab.experiments.granularity.render import (
    DEFAULT_ARTIFACT_DIR,
    lower_layer_svg,
)
from mshab.skills.library import ContractLibrary
from mshab.skills.model import ArtifactStatus, ContractType


def _make_ready(checkpoint_root, spec):
    leaf = checkpoint_root / spec.relative_leaf
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "config.yml").write_text("name: ppo\n")
    (leaf / "policy.pt").write_bytes(b"weights")


class GranularityLibraryTests(TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.checkpoint_root = Path(self.temporary_directory.name)
        self.library = build_granularity_library(self.checkpoint_root)

    def test_manifest_enumerates_the_official_download(self):
        ids = [spec.id for spec in POLICY_SPECS]
        self.assertEqual(len(ids), 53)
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(
            Counter(spec.contract_type.value for spec in POLICY_SPECS),
            {"navigate": 3, "pick": 23, "place": 23, "open": 2, "close": 2},
        )
        self.assertEqual(
            spec_for("rl.set_table.pick.013_apple").relative_leaf,
            "rl/set_table/pick/013_apple",
        )
        with self.assertRaisesRegex(ValueError, "open/close"):
            PolicySpec("tidy_house", ContractType.OPEN, "fridge")
        with self.assertRaisesRegex(ValueError, "unsupported pick target"):
            PolicySpec("set_table", ContractType.PICK, "003_cracker_box")
        with self.assertRaisesRegex(KeyError, "not in the granularity manifest"):
            spec_for("vla.manipulation")

    def test_library_binds_every_policy_to_the_contract_of_its_type(self):
        library = self.library
        self.assertIsInstance(library, ContractLibrary)

        contracts = library.find(task=EXPERIMENT_TASK)
        self.assertEqual(
            [contract.id for contract in contracts],
            sorted(contract_id(contract_type) for contract_type in ContractType),
        )
        self.assertTrue(all(contract.target == "all" for contract in contracts))
        self.assertEqual(len(library.policies), 53)

        bindings = [
            (contract.id, policy.id)
            for contract in contracts
            for policy in library.policies_for(contract.id)
        ]
        self.assertEqual(len(bindings), 53)
        for contract in contracts:
            bound = [policy.id for policy in library.policies_for(contract.id)]
            self.assertEqual(bound, sorted(bound))
        for policy in library.policies:
            spec = spec_for(policy.id)
            self.assertEqual(policy.target, spec.target)
            self.assertEqual(
                [contract.id for contract in library.contracts_for(policy.id)],
                [contract_id(spec.contract_type)],
            )
        self.assertEqual(len(library.policies_for(contract_id("pick"))), 23)
        self.assertEqual(
            library.policy("rl.set_table.navigate.all").status,
            ArtifactStatus.MISSING,
        )

    def test_generic_contracts_ground_the_symbolic_interface(self):
        pick = self.library.get(contract_id("pick")).bind({"object": "024_bowl"})
        self.assertEqual(
            pick.preconditions, ("reachable(024_bowl)", "gripper_empty()")
        )
        self.assertEqual(pick.effects, ("holding(024_bowl)",))
        self.assertEqual(pick.deletes, ("gripper_empty()",))

        place = self.library.get(contract_id("place")).bind(
            {"object": "013_apple", "destination": "dining_table"}
        )
        self.assertEqual(
            place.preconditions,
            ("holding(013_apple)", "reachable(dining_table)"),
        )
        self.assertEqual(
            place.effects, ("at(013_apple,dining_table)", "gripper_empty()")
        )
        self.assertEqual(place.deletes, ("holding(013_apple)",))

    def test_policy_selection_respects_the_grounded_target(self):
        library = self.library
        pick = contract_id("pick")
        bowl = {"object": "024_bowl"}

        applicable = [
            policy.id for policy in library.applicable_policies(pick, bowl)
        ]
        self.assertEqual(
            applicable,
            [
                "rl.prepare_groceries.pick.024_bowl",
                "rl.set_table.pick.024_bowl",
                "rl.tidy_house.pick.024_bowl",
                "rl.prepare_groceries.pick.all",
                "rl.set_table.pick.all",
                "rl.tidy_house.pick.all",
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "no ready policy"):
            library.select_policy(pick, arguments=bowl)

        _make_ready(self.checkpoint_root, spec_for("rl.set_table.pick.013_apple"))
        # Without arguments binding order decides, and the apple checkpoint
        # is the first ready one.  With the bowl grounding it is never a
        # candidate, ready or not.
        self.assertEqual(
            library.select_policy(pick).id, "rl.set_table.pick.013_apple"
        )
        with self.assertRaisesRegex(RuntimeError, "no ready policy"):
            library.select_policy(pick, arguments=bowl)
        with self.assertRaisesRegex(ValueError, "trained for target"):
            library.select_policy(pick, "rl.set_table.pick.013_apple", arguments=bowl)

        _make_ready(self.checkpoint_root, spec_for("rl.tidy_house.pick.all"))
        self.assertEqual(
            library.select_policy(pick, arguments=bowl).id,
            "rl.tidy_house.pick.all",
        )
        _make_ready(self.checkpoint_root, spec_for("rl.set_table.pick.024_bowl"))
        self.assertEqual(
            library.select_policy(pick, arguments=bowl).id,
            "rl.set_table.pick.024_bowl",
        )
        self.assertEqual(
            library.select_policy(pick, "rl.tidy_house.pick.all", arguments=bowl).id,
            "rl.tidy_house.pick.all",
        )

    def test_document_is_portable_and_matches_the_committed_artifacts(self):
        document = library_document(self.library)
        self.assertEqual(
            document["summary"],
            {
                "contracts": 5,
                "policies": 53,
                "bindings": 53,
                "policies_by_contract": {
                    "close": 2,
                    "navigate": 3,
                    "open": 2,
                    "pick": 23,
                    "place": 23,
                },
            },
        )
        by_id = {record["id"]: record for record in document["policies"]}
        self.assertEqual(
            by_id["rl.set_table.pick.013_apple"]["contracts"],
            ["mshab.granularity.pick.all"],
        )
        serialized = json.dumps(document, sort_keys=True)
        self.assertNotIn(self.temporary_directory.name, serialized)
        self.assertNotIn('"status"', serialized)
        self.assertNotIn("fallback", serialized.lower())
        self.assertNotIn("alternative", serialized.lower())

        svg = lower_layer_svg(document)
        ET.fromstring(svg)
        expected_json = json.loads((DEFAULT_ARTIFACT_DIR / "library.json").read_text())
        expected_svg = (DEFAULT_ARTIFACT_DIR / "contract_policy_layers.svg").read_text()
        self.assertEqual(document, expected_json)
        self.assertEqual(svg, expected_svg)
