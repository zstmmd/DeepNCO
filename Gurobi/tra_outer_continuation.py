from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OuterContinuationState:
    """Track whether a TIME_LIMIT outer MIP can resume without rebuilding."""

    shell_sha256: str = ""

    def plan(self, shell_sha256: str, *, resume_requested: bool) -> bool:
        shell_hash = str(shell_sha256)
        can_resume = bool(
            resume_requested
            and self.shell_sha256
            and self.shell_sha256 == shell_hash
        )
        if can_resume:
            return True
        self.clear()
        return False

    def remember(self, shell_sha256: str) -> None:
        self.shell_sha256 = str(shell_sha256)

    def clear(self) -> None:
        self.shell_sha256 = ""
