"""Central configuration (env vars + defaults)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

Difficulty = Literal["easy", "medium", "hard", "expert"]


@dataclass(frozen=True)
class Settings:
    """Runtime settings; override via environment variables."""

    default_difficulty: Difficulty = "hard"
    default_episodes: int = 2
    default_partial_obs: bool = True
    use_browser: bool = True
    normalize_reward: bool = False
    trajectory_dir: str = "trajectories"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        def _bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name, "").strip().lower()
            if raw in ("1", "true", "yes", "on"):
                return True
            if raw in ("0", "false", "no", "off"):
                return False
            return default

        diff = os.environ.get("TRIAGE_DIFFICULTY", "hard").strip().lower()
        if diff not in ("easy", "medium", "hard", "expert"):
            diff = "hard"

        return cls(
            default_difficulty=diff,  # type: ignore[arg-type]
            default_episodes=int(os.environ.get("TRIAGE_EPISODES", "2")),
            default_partial_obs=_bool("TRIAGE_PARTIAL_OBS", True),
            use_browser=_bool("TRIAGE_USE_BROWSER", True),
            normalize_reward=_bool("TRIAGE_NORMALIZE_REWARD", False),
            trajectory_dir=os.environ.get("TRIAGE_TRAJECTORY_DIR", "trajectories"),
            log_level=os.environ.get("TRIAGE_LOG_LEVEL", "INFO").upper(),
        )


def project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))
