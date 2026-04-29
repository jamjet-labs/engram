"""Scope — multi-tenancy primitive (org_id x user_id) used by every store operation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Scope(BaseModel):
    """Identifies the tenant slice a fact / message / session belongs to.

    Every storage operation requires a Scope; there is no "global" memory.
    """

    model_config = ConfigDict(frozen=True)

    org_id: str = Field(default="default", min_length=1, max_length=128)
    user_id: str = Field(default="default", min_length=1, max_length=128)

    def __hash__(self) -> int:
        return hash((self.org_id, self.user_id))
