# agent.py
"""Reference agents that act from obs + action_mask (no env peeking)."""
from __future__ import annotations

import numpy as np

from email_data import ACTION_NAMES, EXPAND_ACTION, OPEN_ACTION, PAGE_ACTION, TRIAGE_ACTIONS


def sample_masked_action(action_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random valid (email_idx, action_idx) from the mask."""
    valid = np.argwhere(action_mask)
    if len(valid) == 0:
        return np.array([0, ACTION_NAMES.index("skip")], dtype=np.int64)
    return valid[rng.integers(0, len(valid))].astype(np.int64)


class RandomAgent:
    """Uniform random over legal (email, action) pairs."""

    def __init__(self, action_space=None, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, info):
        return sample_masked_action(info["action_mask"], self.rng)


class RuleBasedAgent:
    """
    Oracle agent for evaluation baselines.

    Reads only from obs (rule_action, primary_match, secondary_match, opened,
    expanded, triaged, visible) and the action_mask in info — no ground-truth
    env peeking.

    Strategy (per page):
      1. Open every unprocessed visible email.
      1.5. Expand every opened visible thread (when use_threads is active).
      2. Apply rule_action when primary_match AND secondary_match are both 1;
         skip otherwise.
      3. Navigate to the next page when the current page is fully triaged.
    """

    def __init__(self, seed=None):
        self._open_idx   = ACTION_NAMES.index(OPEN_ACTION)
        self._expand_idx = ACTION_NAMES.index(EXPAND_ACTION)
        self._page_idx   = ACTION_NAMES.index(PAGE_ACTION)
        self._skip_idx   = ACTION_NAMES.index("skip")

    def act(self, obs, info):
        mask    = info["action_mask"]
        triaged = obs["triaged"]
        visible = obs.get("visible", np.ones(len(triaged)))
        n       = len(triaged)

        # Pass 1 — open any visible un-triaged email
        for idx in range(n):
            if triaged[idx] > 0.5 or visible[idx] < 0.5:
                continue
            if mask[idx, self._open_idx]:
                return np.array([idx, self._open_idx], dtype=np.int64)

        # Pass 1.5 — expand any opened visible thread
        for idx in range(n):
            if triaged[idx] > 0.5 or visible[idx] < 0.5:
                continue
            if mask[idx, self._expand_idx]:
                return np.array([idx, self._expand_idx], dtype=np.int64)

        # Pass 2 — triage based on match features
        primary   = obs["primary_match"]
        secondary = obs["secondary_match"]
        rule_act  = obs["rule_action"]
        act_idx   = int(np.argmax(rule_act))   # index in TRIAGE_ACTIONS
        # TRIAGE_ACTIONS and ACTION_NAMES share the same prefix, so index is identical
        assert TRIAGE_ACTIONS[act_idx] == ACTION_NAMES[act_idx]

        for idx in range(n):
            if triaged[idx] > 0.5 or visible[idx] < 0.5:
                continue
            full_match = primary[idx] > 0.5 and secondary[idx] > 0.5
            chosen = act_idx if full_match else self._skip_idx
            if mask[idx, chosen]:
                return np.array([idx, chosen], dtype=np.int64)

        # Pass 3 — navigate to the next page if untriaged emails remain
        if mask[0, self._page_idx]:
            has_untriaged = any(triaged[i] < 0.5 for i in range(n))
            if has_untriaged:
                return np.array([0, self._page_idx], dtype=np.int64)

        return sample_masked_action(mask, np.random.default_rng(0))
