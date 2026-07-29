"""Serializable models for a pristine Pytest oracle inventory."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class OracleInventoryFile(BaseModel):
    """One immutable path/content binding from the pristine worktree."""

    model_config = ConfigDict(frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PytestOracleInventory(BaseModel):
    """Files withheld from the model and protected at the patch boundary."""

    model_config = ConfigDict(frozen=True)

    files: tuple[OracleInventoryFile, ...]
    nodeids: tuple[str, ...]
    configured_testpaths: tuple[str, ...]
    support_roots: tuple[str, ...] = ()
    collection_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def protected_paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.files)

    @property
    def protected_roots(self) -> tuple[str, ...]:
        """Test trees where both existing files and new paths are protected."""
        return tuple(
            dict.fromkeys((*self.configured_testpaths, *self.support_roots))
        )
