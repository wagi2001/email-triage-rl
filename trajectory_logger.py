# trajectory_logger.py — append episode trajectories as JSONL
import json
from dataclasses import asdict
from datetime import datetime, timezone

from email_data import ACTION_NAMES


class TrajectoryLogger:
    """One JSON object per line, one episode per line."""

    def __init__(self, path, run_meta=None):
        self.path = path
        self.run_meta = run_meta or {}
        self._episode = None

    def begin_episode(self, episode, seed, rule, emails, mode, agent_meta=None):
        self._episode = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episode": episode,
            "seed": seed,
            "mode": mode,
            "rule": dict(rule),
            "emails": list(emails),
            "agent": agent_meta or {},
            "steps": [],
            "run": self.run_meta,
        }

    def log_step(
        self,
        step_idx,
        email_idx,
        action_idx,
        reward,
        terminated,
        truncated,
        llm_raw_response=None,
    ):
        if self._episode is None:
            return
        action_name = ACTION_NAMES[action_idx]
        step = {
            "step": step_idx,
            "email_idx": email_idx,
            "action_idx": int(action_idx),
            "action": action_name,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        if llm_raw_response is not None:
            step["llm_raw_response"] = llm_raw_response
        self._episode["steps"].append(step)

    def end_episode(
        self,
        actions_taken,
        total_reward,
        verdict,
        reward_consistency=None,
        opened_ids=None,
    ):
        if self._episode is None:
            return
        # JSON keys must be strings
        self._episode["actions_taken"] = {
            str(k): v for k, v in actions_taken.items()
        }
        if opened_ids is not None:
            self._episode["opened_ids"] = [str(i) for i in opened_ids]
        self._episode["total_reward"] = float(total_reward)
        self._episode["reward_consistency"] = reward_consistency
        self._episode["metrics"] = {
            "total_score": verdict.total_score,
            "max_possible": verdict.max_possible,
            "triage_accuracy": verdict.triage_accuracy,
            "reward_efficiency": verdict.reward_efficiency,
            "accuracy": verdict.reward_efficiency,
            "correct_actions": verdict.correct_actions,
            "false_positives": verdict.false_positives,
            "false_negatives": verdict.false_negatives,
            "total_emails": verdict.total_emails,
        }
        self._episode["email_verdicts"] = [asdict(v) for v in verdict.verdicts]

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._episode, ensure_ascii=False) + "\n")

        self._episode = None
