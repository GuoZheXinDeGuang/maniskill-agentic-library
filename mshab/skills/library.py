"""Registry and local checkpoint discovery for :mod:`mshab.skills`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from mshab.skills.model import (
    ATOMIC_SKILL_CLASSES,
    AtomicSkill,
    CheckpointBackend,
    Skill,
    SkillType,
)


class SkillLibrary:
    """In-memory owner of semantic skills and their execution alternatives."""

    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._skills: Dict[str, Skill] = {}
        for skill in skills:
            self.register(skill)

    def register(self, skill: Skill) -> None:
        if skill.id in self._skills:
            raise ValueError("duplicate skill id {!r}".format(skill.id))
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> Skill:
        try:
            return self._skills[skill_id]
        except KeyError as exc:
            raise KeyError(
                "unknown skill {!r}; available={}".format(
                    skill_id, sorted(self._skills)
                )
            ) from exc

    def find(
        self,
        task: Optional[str] = None,
        skill_type: Optional[SkillType] = None,
        target: Optional[str] = None,
        ready: Optional[bool] = None,
    ) -> List[Skill]:
        result = []
        for skill in self._skills.values():
            if task is not None and skill.task != task:
                continue
            if ready is not None and skill.ready != ready:
                continue
            if skill_type is not None and (
                not isinstance(skill, AtomicSkill) or skill.skill_type != skill_type
            ):
                continue
            if target is not None and (
                not isinstance(skill, AtomicSkill) or skill.target != target
            ):
                continue
            result.append(skill)
        return sorted(result, key=lambda item: item.id)

    def to_dict(self) -> Dict[str, object]:
        skills = [skill.as_dict() for skill in self.find()]
        return {
            "schema_version": "mshab.skill-library.v1",
            "count": len(skills),
            "skills": skills,
        }

    def save_index(self, path: Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_checkpoint_root(cls, root: Path) -> "SkillLibrary":
        """Discover ``family/task/type/target/{config.yml,policy.pt}`` leaves.

        A leaf is registered even when just one artifact is present, allowing
        callers to report ``partial`` downloads instead of silently hiding them.
        Multiple policy families are folded into one semantic AtomicSkill as
        interchangeable backends.
        """

        checkpoint_root = Path(root)
        if not checkpoint_root.is_dir():
            raise FileNotFoundError(
                "checkpoint root does not exist: {}".format(checkpoint_root)
            )

        library = cls()
        leaves = sorted(
            {
                path.parent
                for artifact_name in ("policy.pt", "config.yml")
                for path in checkpoint_root.glob("*/*/*/*/{}".format(artifact_name))
            }
        )
        for leaf in leaves:
            family, task, raw_type, target = leaf.relative_to(checkpoint_root).parts
            try:
                skill_type = SkillType(raw_type)
            except ValueError:
                continue
            skill_id = "mshab.{}.{}.{}".format(task, skill_type.value, target)
            existing = library._skills.get(skill_id)
            if existing is None:
                skill_class = ATOMIC_SKILL_CLASSES[skill_type]
                skill = skill_class(task=task, target=target)
                library.register(skill)
            elif isinstance(existing, AtomicSkill):
                skill = existing
            else:
                raise TypeError("{} is not atomic".format(skill_id))

            policy_type = _policy_type(family, target)
            skill.add_backend(
                CheckpointBackend(
                    key=family,
                    family=family,
                    checkpoint_path=leaf / "policy.pt",
                    config_path=leaf / "config.yml",
                    policy_type=policy_type,
                )
            )
        return library


def _policy_type(family: str, target: str) -> str:
    if family == "rl":
        return "rl_all_obj" if target == "all" else "rl_per_obj"
    return family
