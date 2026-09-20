"""Generate every committed artifact: the Layer-3/4 library document and its
diagram, and the gold graphs of the upper layers with theirs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from mshab.experiments.granularity.higher_layers.render import write_gold_graphs
from mshab.experiments.granularity.lower_layers.library import (
    build_granularity_library,
    library_document,
)
from mshab.experiments.granularity.paths import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_GRAPH_DIR,
)
from mshab.experiments.granularity.svg import text
from mshab.skills.library import ContractLibrary
from mshab.skills.model import CONTRACT_CLASSES, ContractType


def _predicate_list(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "—"


def _contract_lines(contract: Mapping[str, Any]) -> List[tuple]:
    parameters = ", ".join(item["name"] for item in contract["parameters"])
    return [
        ("Inputs", parameters),
        ("Preconditions", _predicate_list(contract["preconditions"])),
        ("Effects", _predicate_list(contract["effects"])),
        ("Deletes", _predicate_list(contract["deletes"])),
    ]


def lower_layer_svg(document: Mapping[str, Any]) -> str:
    """Render only the controlled Contract and Policy layers as deterministic SVG."""

    contracts = list(document["contracts"])
    policies = list(document["policies"])
    if len(contracts) != 5 or len(policies) != 53:
        raise ValueError(
            "the lower-layer diagram requires five contracts and 53 policies"
        )

    card_width = 400
    card_xs = [45 + index * 430 for index in range(5)]
    by_type: Dict[str, List[Mapping[str, Any]]] = {}
    for policy in policies:
        by_type.setdefault(policy["contract_type"], []).append(policy)

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="2200" height="1050" '
        'viewBox="0 0 2200 1050" role="img" '
        'aria-labelledby="title description">',
        (
            '<title id="title">MS-HAB granularity experiment: '
            "Contract and Policy layers</title>"
        ),
        (
            '<desc id="description">Five generic contracts bound to '
            "53 concrete RL checkpoint policies.</desc>"
        ),
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#7552d6"/>',
        "</marker>",
        "</defs>",
        '<rect width="2200" height="1050" fill="#ffffff"/>',
        '<rect x="25" y="20" width="2150" height="405" rx="20" fill="#fbf5f1"/>',
        '<rect x="25" y="455" width="2150" height="550" rx="20" fill="#f6f3fc"/>',
        '<rect x="25" y="20" width="8" height="405" rx="4" fill="#c6531a"/>',
        '<rect x="25" y="455" width="8" height="550" rx="4" fill="#7552d6"/>',
        text(55, 62, "Contract layer", size=27, weight=700, fill="#b94a17"),
        text(
            260,
            62,
            (
                "5 reusable symbolic interfaces · inputs / preconditions / "
                "effects / deletes"
            ),
            size=16,
            fill="#71809b",
        ),
        text(55, 499, "Policy layer", size=27, weight=700, fill="#6842cb"),
        text(
            225,
            499,
            (
                "53 stored RL checkpoint policies · each bound to the "
                "contract of its type"
            ),
            size=16,
            fill="#71809b",
        ),
    ]

    for index, contract in enumerate(contracts):
        x = card_xs[index]
        contract_type = contract["contract_type"]
        out.extend(
            [
                '<rect x="{x}" y="82" width="{width}" height="315" rx="16" '
                'fill="#fffaf2" stroke="#c6531a" stroke-width="3"/>'.format(
                    x=x, width=card_width
                ),
                '<path d="M {x} 98 Q {x} 82 {x2} 82 H {x3} Q {x4} 82 {x4} 98 '
                'V 135 H {x} Z" fill="#c6531a"/>'.format(
                    x=x,
                    x2=x + 16,
                    x3=x + card_width - 16,
                    x4=x + card_width,
                ),
                text(
                    x + card_width / 2,
                    119,
                    CONTRACT_CLASSES[ContractType(contract_type)].__name__,
                    size=20,
                    weight=700,
                    fill="#ffffff",
                    anchor="middle",
                ),
            ]
        )
        line_y = 168
        for label, value in _contract_lines(contract):
            out.append(
                text(
                    x + 16,
                    line_y,
                    label,
                    size=14,
                    weight=700,
                    fill="#c6531a",
                )
            )
            # Long symbolic expressions remain readable on a second line.
            out.append(text(x + 16, line_y + 22, value, size=13, fill="#39465a"))
            line_y += 57
        out.append(
            text(
                x + card_width / 2,
                382,
                "{} policies".format(len(contract["policies"])),
                size=13,
                weight=700,
                fill="#8b5b46",
                anchor="middle",
            )
        )

        center_x = x + card_width / 2
        out.append(
            '<path d="M {x} 538 V 408" fill="none" stroke="#7552d6" '
            'stroke-width="2.5" marker-end="url(#arrow)"/>'.format(x=center_x)
        )
        out.append(
            text(
                center_x + 10,
                448,
                "executes",
                size=12,
                fill="#7552d6",
            )
        )

        group = by_type[contract_type]
        out.extend(
            [
                '<rect x="{x}" y="535" width="{width}" height="400" rx="15" '
                'fill="#fbfaff" stroke="#7552d6" stroke-width="2" '
                'stroke-dasharray="8 6"/>'.format(x=x, width=card_width),
                text(
                    center_x,
                    566,
                    "{} policies".format(len(group)),
                    size=16,
                    weight=700,
                    fill="#6842cb",
                    anchor="middle",
                ),
            ]
        )
        columns = 2 if len(group) > 12 else 1
        rows = (len(group) + columns - 1) // columns
        box_gap = 8
        box_width = (card_width - 28 - box_gap * (columns - 1)) / columns
        for policy_index, policy in enumerate(group):
            column = policy_index // rows
            row = policy_index % rows
            box_x = x + 14 + column * (box_width + box_gap)
            box_y = 584 + row * 27
            out.extend(
                [
                    '<rect x="{x}" y="{y}" width="{width}" height="21" rx="10" '
                    'fill="#ffffff" stroke="#8d6ee0" stroke-width="1.4"/>'.format(
                        x=box_x, y=box_y, width=box_width
                    ),
                    text(
                        box_x + box_width / 2,
                        box_y + 15,
                        policy["short_id"],
                        size=10.5,
                        fill="#6245b1",
                        anchor="middle",
                    ),
                ]
            )

    out.extend(
        [
            text(
                55,
                972,
                "PG = prepare_groceries · ST = set_table · TH = tidy_house",
                size=14,
                fill="#71809b",
            ),
            text(
                2145,
                972,
                (
                    "Bindings: contract → policies in preference order · "
                    "no policy-to-policy edges"
                ),
                size=14,
                fill="#6842cb",
                anchor="end",
            ),
            "</svg>",
        ]
    )
    return "\n".join(out) + "\n"


def write_artifacts(
    library: ContractLibrary,
    json_path: Path,
    svg_path: Path,
) -> Dict[str, Any]:
    """Write the library's deterministic, machine-independent JSON and SVG."""

    document = library_document(library)
    json_path = Path(json_path)
    svg_path = Path(svg_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    svg_path.write_text(lower_layer_svg(document))
    return document


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render the controlled Contract/Policy layers for the "
            "granularity experiment."
        )
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="mshab_checkpoints root; used to construct Policy objects only",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "library.json",
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "contract_policy_layers.svg",
    )
    parser.add_argument("--graphs", type=Path, default=DEFAULT_GRAPH_DIR)
    args = parser.parse_args()
    library = build_granularity_library(Path(args.checkpoint_root))
    document = write_artifacts(library, args.json, args.svg)
    print(
        "wrote {} contracts and {} policies to {} and {}".format(
            document["summary"]["contracts"],
            document["summary"]["policies"],
            args.json,
            args.svg,
        )
    )
    names = write_gold_graphs(library, args.graphs)
    print("wrote gold graphs {} to {}".format(", ".join(names), args.graphs))


if __name__ == "__main__":
    main()
