"""Deterministic memory-system router for typed Engram facts."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from engram.models import LifecycleState, MemorySystem, PromotionState

_PREFERENCE_RE = re.compile(
    r"\b(prefer|prefers|preferred|preference|favorite|like|dislike|love|hate|"
    r"package manager|use pnpm|uses pnpm|use npm|uses npm|always use|use .* only|"
    r"communication style|tone|concise|verbose|risk tolerance)\b",
    re.IGNORECASE,
)
_PROCEDURAL_RE = re.compile(
    r"\b(workflow|checklist|procedure|playbook|steps?|how to|runbook|release process|"
    r"successful .*workflow|validation recipe)\b",
    re.IGNORECASE,
)
_PROSPECTIVE_RE = re.compile(
    r"\b(after|when|next|later|remind|follow up|todo|to-do|once .*pass|if .*then)\b",
    re.IGNORECASE,
)
_EPISODIC_RE = re.compile(
    r"\b(session event|tool result|command|failed|failure|fixed|changed|ran|"
    r"because|diagnose|debug|error|disproven|real issue|rejected)\b",
    re.IGNORECASE,
)
_EPISODIC_EVENT_RE = re.compile(r"\b(session event|tool result|command)\b", re.IGNORECASE)
_EPISODIC_RESOLUTION_RE = re.compile(r"\b(disproven|real issue)\b", re.IGNORECASE)
_SEMANTIC_RE = re.compile(
    r"\b(decision|architecture|config|configuration|fact|current|latest|"
    r"uses port|listens on|moved from port|default|canonical|policy|format|"
    r"because|caused by|lives in|path|directory|file)\b",
    re.IGNORECASE,
)
_WORKING_RE = re.compile(
    r"\b(active|temporary|hypothesis|blocker|current plan|in progress|right now)\b",
    re.IGNORECASE,
)

_CATEGORY_ROUTES: dict[str, tuple[MemorySystem, LifecycleState, PromotionState]] = {
    "preference": (MemorySystem.PREFERENCE, LifecycleState.DURABLE, PromotionState.CANDIDATE),
    "pattern": (MemorySystem.PROCEDURAL, LifecycleState.DURABLE, PromotionState.CANDIDATE),
    "workflow": (MemorySystem.PROCEDURAL, LifecycleState.DURABLE, PromotionState.CANDIDATE),
    "decision": (MemorySystem.SEMANTIC, LifecycleState.CANONICAL, PromotionState.CANDIDATE),
    "architecture": (MemorySystem.SEMANTIC, LifecycleState.DURABLE, PromotionState.CANDIDATE),
    "config": (MemorySystem.SEMANTIC, LifecycleState.DURABLE, PromotionState.CANDIDATE),
    "discovery": (MemorySystem.SEMANTIC, LifecycleState.DURABLE, PromotionState.CANDIDATE),
    "bugfix": (MemorySystem.EPISODIC, LifecycleState.DURABLE, PromotionState.RAW),
    "session_summary": (MemorySystem.EPISODIC, LifecycleState.DURABLE, PromotionState.RAW),
    "event": (MemorySystem.EPISODIC, LifecycleState.DURABLE, PromotionState.RAW),
}


class MemoryRoute(BaseModel):
    """Typed routing decision for a memory observation."""

    memory_system: MemorySystem
    memory_subtype: str | None = None
    secondary_systems: list[MemorySystem] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    lifecycle_state: LifecycleState
    promotion_state: PromotionState
    reason: str


class RuleBasedMemoryRouter:
    """Deterministic router for assigning memory-system metadata."""

    def route(
        self,
        text: str,
        category: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRoute:
        md = metadata or {}
        explicit = md.get("memory_system")
        if explicit is not None:
            system = MemorySystem(str(explicit))
            return MemoryRoute(
                memory_system=system,
                memory_subtype=_subtype_from_metadata(md),
                tags=_tags_from_metadata(md),
                lifecycle_state=_lifecycle_from_metadata(md, LifecycleState.DURABLE),
                promotion_state=_promotion_from_metadata(md, PromotionState.CANDIDATE),
                reason="explicit metadata memory_system",
            )

        if category is not None:
            route = _CATEGORY_ROUTES.get(category.lower())
            if route is not None:
                system, lifecycle, promotion = route
                return MemoryRoute(
                    memory_system=system,
                    memory_subtype=_subtype_for_category(category),
                    secondary_systems=_secondary_for(system, category=category),
                    tags=[category.lower()],
                    lifecycle_state=lifecycle,
                    promotion_state=promotion,
                    reason=f"category route: {category}",
                )

        if _PREFERENCE_RE.search(text):
            return MemoryRoute(
                memory_system=MemorySystem.PREFERENCE,
                memory_subtype=_preference_subtype(text),
                secondary_systems=[MemorySystem.SEMANTIC],
                tags=_tags_for(text),
                lifecycle_state=LifecycleState.DURABLE,
                promotion_state=PromotionState.CANDIDATE,
                reason="preference signal",
            )
        if _PROCEDURAL_RE.search(text):
            return MemoryRoute(
                memory_system=MemorySystem.PROCEDURAL,
                memory_subtype=_procedural_subtype(text),
                tags=_tags_for(text),
                lifecycle_state=LifecycleState.DURABLE,
                promotion_state=PromotionState.CANDIDATE,
                reason="procedural workflow signal",
            )
        if _EPISODIC_EVENT_RE.search(text):
            return MemoryRoute(
                memory_system=MemorySystem.EPISODIC,
                memory_subtype="session_event",
                secondary_systems=[MemorySystem.SEMANTIC],
                tags=_tags_for(text),
                lifecycle_state=LifecycleState.DURABLE,
                promotion_state=PromotionState.RAW,
                reason="episodic event signal",
            )
        if _PROSPECTIVE_RE.search(text):
            return MemoryRoute(
                memory_system=MemorySystem.PROSPECTIVE,
                memory_subtype="trigger",
                tags=_tags_for(text),
                lifecycle_state=LifecycleState.TEMPORARY,
                promotion_state=PromotionState.CANDIDATE,
                reason="prospective trigger signal",
            )
        if _EPISODIC_RESOLUTION_RE.search(text):
            return MemoryRoute(
                memory_system=MemorySystem.EPISODIC,
                memory_subtype="error_repair",
                secondary_systems=[MemorySystem.SEMANTIC],
                tags=_tags_for(text),
                lifecycle_state=LifecycleState.DURABLE,
                promotion_state=PromotionState.RAW,
                reason="episodic resolution signal",
            )
        if _WORKING_RE.search(text):
            return MemoryRoute(
                memory_system=MemorySystem.WORKING,
                memory_subtype=_working_subtype(text),
                tags=_tags_for(text),
                lifecycle_state=LifecycleState.TEMPORARY,
                promotion_state=PromotionState.RAW,
                reason="working-memory signal",
            )
        if _SEMANTIC_RE.search(text):
            return MemoryRoute(
                memory_system=MemorySystem.SEMANTIC,
                memory_subtype=_semantic_subtype(text),
                tags=_tags_for(text),
                lifecycle_state=LifecycleState.DURABLE,
                promotion_state=PromotionState.CANDIDATE,
                reason="semantic fact signal",
            )
        if _EPISODIC_RE.search(text):
            return MemoryRoute(
                memory_system=MemorySystem.EPISODIC,
                memory_subtype="error_repair" if "error" in text.lower() else "session_event",
                secondary_systems=[MemorySystem.SEMANTIC],
                tags=_tags_for(text),
                lifecycle_state=LifecycleState.DURABLE,
                promotion_state=PromotionState.RAW,
                reason="episodic event signal",
            )
        return MemoryRoute(
            memory_system=MemorySystem.SEMANTIC,
            memory_subtype="project_fact",
            tags=_tags_for(text),
            lifecycle_state=LifecycleState.DURABLE,
            promotion_state=PromotionState.CANDIDATE,
            reason="default semantic route",
        )


def _secondary_for(system: MemorySystem, category: str) -> list[MemorySystem]:
    if category.lower() == "bugfix":
        return [MemorySystem.SEMANTIC]
    if system == MemorySystem.PREFERENCE:
        return [MemorySystem.SEMANTIC]
    return []


def _subtype_for_category(category: str) -> str | None:
    return {
        "architecture": "architecture",
        "bugfix": "error_repair",
        "config": "config",
        "decision": "project_fact",
        "discovery": "project_fact",
        "event": "session_event",
        "pattern": "workflow",
        "preference": "personal_preference",
        "session_summary": "audit_trail",
        "workflow": "workflow",
    }.get(category.lower())


def _preference_subtype(text: str) -> str:
    low = text.lower()
    if any(token in low for token in ("pnpm", "npm", "yarn", "package manager")):
        return "tool_preference"
    if any(token in low for token in ("communication", "tone", "concise", "verbose")):
        return "communication_style"
    return "personal_preference"


def _procedural_subtype(text: str) -> str:
    low = text.lower()
    if any(token in low for token in ("release", "changelog", "version", "deploy")):
        return "release_process"
    if any(token in low for token in ("test", "validation", "validate")):
        return "validation_recipe"
    return "workflow"


def _working_subtype(text: str) -> str:
    low = text.lower()
    if "blocker" in low:
        return "blocker"
    if "plan" in low:
        return "active_plan"
    return "scratchpad"


def _semantic_subtype(text: str) -> str:
    low = text.lower()
    if any(token in low for token in ("architecture", "schema")):
        return "architecture"
    if any(token in low for token in ("config", "configuration", "port")):
        return "config"
    return "project_fact"


def _tags_for(text: str) -> list[str]:
    low = text.lower()
    tags = []
    for token in ("pnpm", "npm", "yarn", "python", "pytest", "ruff", "mypy", "release"):
        if token in low:
            tags.append(token)
    return tags


def _subtype_from_metadata(metadata: dict[str, Any]) -> str | None:
    raw = metadata.get("memory_subtype")
    return str(raw) if raw is not None else None


def _tags_from_metadata(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("tags", [])
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def _lifecycle_from_metadata(metadata: dict[str, Any], default: LifecycleState) -> LifecycleState:
    raw = metadata.get("lifecycle_state")
    return LifecycleState(str(raw)) if raw is not None else default


def _promotion_from_metadata(metadata: dict[str, Any], default: PromotionState) -> PromotionState:
    raw = metadata.get("promotion_state")
    return PromotionState(str(raw)) if raw is not None else default
