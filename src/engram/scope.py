"""Scope — multi-tenancy primitive (org_id x user_id) used by every store operation."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Scope(BaseModel):
    """Identifies the tenant slice a fact / message / session belongs to.

    Every storage operation requires a Scope; there is no "global" memory.

    Each id is 1 to 128 characters. Refused with ValueError: a blank id, one
    with leading or trailing whitespace, and one holding a control character
    (Unicode category Cc, such as NUL, DEL or U+0085) or a lone surrogate.
    Anything else is allowed: quotes, backslashes, ``%``, ``_``, ``/``,
    inner spaces, any letter or emoji. Ids are not normalized, so ``"\\u00fc"`` and
    ``"u\\u0308"`` are two scopes.
    """

    model_config = ConfigDict(frozen=True)

    org_id: str = Field(default="default", min_length=1, max_length=128)
    user_id: str = Field(default="default", min_length=1, max_length=128)

    @field_validator("org_id", "user_id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        # Pydantic has already refused a lone surrogate: it is not valid text.
        if not value.strip():
            raise ValueError("must not be blank")
        if value != value.strip():
            raise ValueError("must not start or end with whitespace")
        if any(unicodedata.category(ch) == "Cc" for ch in value):
            raise ValueError("must not contain a control character")
        return value

    def __hash__(self) -> int:
        return hash((self.org_id, self.user_id))


@dataclass(frozen=True, slots=True)
class ScopeWriteGuard:
    """The erase epoch a write saw when it started, for one scope.

    Each scope has an erase epoch: 0 until its first `ScopedEngram.erase_all`,
    and one more after each, moved in the same transaction as the erase. Take
    a guard when a write starts (`ScopedEngram.guard()`) and pass it to the
    writes that follow. A store that is given one compares it inside the
    write's transaction: a guard for another scope raises ScopeError, and an
    epoch that has moved since raises ScopeErasedError, writing nothing.

    No clock is read, so processes whose clocks differ agree on whether a
    write started before an erase. Erasing one fact does not move the epoch.
    """

    scope: Scope
    erase_epoch: int  # read from the store; 0 when the scope was never erased

    def __post_init__(self) -> None:
        # Checked at run time too: a guard built by hand must still name one scope.
        scope: object = self.scope
        epoch: object = self.erase_epoch
        if not isinstance(scope, Scope):
            raise TypeError("scope must be a Scope")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError("erase_epoch must be an int")
        if epoch < 0:
            raise ValueError("erase_epoch must not be negative")
