# trajectory_logger.py — append episode trajectories as JSONL
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from email_data import ACTION_NAMES


class TrajectoryLogger:
    """One JSON object per line, one episode per object."""

    def __init__(self, path: str, run_meta: dict | None = None):
        self.path     = path
        self.run_meta = run_meta or {}
        self._episode: dict | None = None

    def begin_episode(self, episode: int, seed, rule: dict, emails: list, mode: str, agent_meta: dict | None = None):
        self._episode = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episode":   episode,
            "seed":      seed,
            "mode":      mode,
            "rule":      dict(rule),
            "emails":    list(emails),
            "agent":     agent_meta or {},
            "steps":     [],
            "run":       self.run_meta,
        }

    def log_step(
        self,
        step_idx: int,
        email_idx: int,
        action_idx: int,
        reward: float,
        terminated: bool,
        truncated: bool,
    ):
        if self._episode is None:
            return
        self._episode["steps"].append({
            "step":       step_idx,
            "email_idx":  email_idx,
            "action_idx": int(action_idx),
            "action":     ACTION_NAMES[action_idx],
            "reward":     float(reward),
            "terminated": bool(terminated),
            "truncated":  bool(truncated),
        })

    def end_episode(
        self,
        actions_taken: dict,
        total_reward: float,
        verdict,
        opened_ids=None,
    ):
        if self._episode is None:
            return
        self._episode["actions_taken"] = {str(k): v for k, v in actions_taken.items()}
        if opened_ids is not None:
            self._episode["opened_ids"] = [str(i) for i in opened_ids]
        self._episode["total_reward"] = float(total_reward)
        self._episode["metrics"] = {
            "total_score":     verdict.total_score,
            "max_possible":    verdict.max_possible,
            "triage_accuracy": verdict.triage_accuracy,
            "reward_efficiency": verdict.reward_efficiency,
            "correct_actions": verdict.correct_actions,
            "false_positives": verdict.false_positives,
            "false_negatives": verdict.false_negatives,
            "total_emails":    verdict.total_emails,
        }
        self._episode["email_verdicts"] = [asdict(v) for v in verdict.verdicts]

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._episode, ensure_ascii=False) + "\n")

        self._episode = None
