"""Structured observation encoding for RL training.

Obs layout (all float32):
  rule_action      (4,)        one-hot: which triage action the rule demands
  rule_compound    (4,)        one-hot: none / AND / EXCEPT / EXCEPT_ANY
  filter_on_sender (1,)        1 if primary filter is on sender_email, 0 if subject
  step_fraction    (1,)        steps_taken / max_steps  (time-pressure signal)
  primary_match    (n_emails,) 1 if primary filter matches (hidden until opened in partial-obs)
  secondary_match  (n_emails,) 1 if secondary condition passes (subject-based, always visible)
  opened           (n_emails,) 1 if email has been opened this episode
  triaged          (n_emails,) 1 if email has received a triage action

Design intent
-------------
primary_match AND secondary_match == 1  →  apply rule_action
otherwise                               →  skip

For simple rules secondary_match is always 1 (no constraint).
For AND rules secondary_match is 1 when the secondary condition is satisfied.
For EXCEPT / EXCEPT_ANY rules secondary_match is 1 when the exception does NOT fire.
This unifies all rule types into the same boolean logic for the policy.

filter_on_sender tells the policy whether opening emails is necessary:
  1.0 → sender rule: primary_match is hidden until opened (open first)
  0.0 → subject rule: primary_match is always visible (can triage immediately)
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from email_data import TRIAGE_ACTIONS, _check_condition

# Rule-action index matches TRIAGE_ACTIONS order exactly
_ACTION_SIZE = len(TRIAGE_ACTIONS)          # 4

# Compound-type one-hot indices
_COMPOUND_NONE      = 0
_COMPOUND_AND       = 1
_COMPOUND_EXCEPT    = 2
_COMPOUND_EXCEPT_ANY = 3
_COMPOUND_SIZE      = 4


def _rule_action_onehot(rule: dict) -> np.ndarray:
    arr = np.zeros(_ACTION_SIZE, dtype=np.float32)
    action = rule.get("action", "skip")
    if action in TRIAGE_ACTIONS:
        arr[TRIAGE_ACTIONS.index(action)] = 1.0
    return arr


def _rule_compound_onehot(rule: dict) -> np.ndarray:
    arr = np.zeros(_COMPOUND_SIZE, dtype=np.float32)
    if "also_requires" in rule:
        arr[_COMPOUND_AND] = 1.0
    elif "except_if_any" in rule:
        arr[_COMPOUND_EXCEPT_ANY] = 1.0
    elif "except_if" in rule:
        arr[_COMPOUND_EXCEPT] = 1.0
    else:
        arr[_COMPOUND_NONE] = 1.0
    return arr


def _primary_match(email: dict, rule: dict, is_opened: bool, partial_obs: bool) -> float:
    """1 if the primary condition matches.

    For sender-based rules in partial-obs mode the match is 0 until the email
    is opened (sender revealed).  Subject-based rules are always visible.
    """
    if partial_obs and not is_opened and rule.get("filter_key") == "sender_email":
        return 0.0
    return float(_check_condition(email, rule))


def _secondary_match(email: dict, rule: dict) -> float:
    """1 if the secondary condition allows the action, 0 if it blocks it.

    Unified encoding:
      AND rule        → 1 iff also_requires is satisfied
      EXCEPT rule     → 1 iff except_if is NOT triggered
      EXCEPT_ANY rule → 1 iff none of except_if_any are triggered
      simple rule     → always 1 (no secondary constraint)
    """
    if "also_requires" in rule:
        return float(_check_condition(email, rule["also_requires"]))
    if "except_if" in rule:
        return float(not _check_condition(email, rule["except_if"]))
    if "except_if_any" in rule:
        return float(not any(_check_condition(email, c) for c in rule["except_if_any"]))
    return 1.0


def build_observation(
    rule: dict,
    emails: list[dict],
    n_slots: int,
    opened_ids: set,
    actions_taken: dict,
    partial_obs: bool,
    step_fraction: float = 0.0,
    expanded_ids: set | None = None,
    current_page: int | None = None,
    page_size: int | None = None,
) -> dict:
    rule_action      = _rule_action_onehot(rule)
    rule_compound    = _rule_compound_onehot(rule)
    filter_on_sender = np.array(
        [1.0 if rule.get("filter_key") == "sender_email" else 0.0],
        dtype=np.float32,
    )
    step_frac = np.array([float(np.clip(step_fraction, 0.0, 1.0))], dtype=np.float32)

    expanded_ids = set(expanded_ids or [])

    # Pagination: compute which email slots are on the current page
    if page_size is not None and current_page is not None:
        page_start = current_page * page_size
        page_end   = page_start + page_size
    else:
        page_start, page_end = 0, n_slots

    primary_match   = np.zeros(n_slots, dtype=np.float32)
    secondary_match = np.zeros(n_slots, dtype=np.float32)
    opened          = np.zeros(n_slots, dtype=np.float32)
    triaged         = np.zeros(n_slots, dtype=np.float32)
    expanded        = np.zeros(n_slots, dtype=np.float32)
    visible         = np.zeros(n_slots, dtype=np.float32)

    for idx, email in enumerate(emails[:n_slots]):
        eid       = email["id"]
        is_opened = eid in opened_ids
        on_page   = page_start <= idx < page_end

        primary_match[idx]   = _primary_match(email, rule, is_opened, partial_obs)
        secondary_match[idx] = _secondary_match(email, rule)

        if is_opened:
            opened[idx]   = 1.0
        if eid in actions_taken:
            triaged[idx]  = 1.0
        if eid in expanded_ids:
            expanded[idx] = 1.0
        if on_page:
            visible[idx]  = 1.0

    return {
        "rule_action":      rule_action,
        "rule_compound":    rule_compound,
        "filter_on_sender": filter_on_sender,
        "step_fraction":    step_frac,
        "primary_match":    primary_match,
        "secondary_match":  secondary_match,
        "opened":           opened,
        "triaged":          triaged,
        "expanded":         expanded,
        "visible":          visible,
    }


def observation_space(n_slots: int) -> spaces.Dict:
    return spaces.Dict({
        "rule_action":      spaces.Box(0.0, 1.0, shape=(_ACTION_SIZE,),   dtype=np.float32),
        "rule_compound":    spaces.Box(0.0, 1.0, shape=(_COMPOUND_SIZE,), dtype=np.float32),
        "filter_on_sender": spaces.Box(0.0, 1.0, shape=(1,),              dtype=np.float32),
        "step_fraction":    spaces.Box(0.0, 1.0, shape=(1,),              dtype=np.float32),
        "primary_match":    spaces.Box(0.0, 1.0, shape=(n_slots,),        dtype=np.float32),
        "secondary_match":  spaces.Box(0.0, 1.0, shape=(n_slots,),        dtype=np.float32),
        "opened":           spaces.Box(0.0, 1.0, shape=(n_slots,),        dtype=np.float32),
        "triaged":          spaces.Box(0.0, 1.0, shape=(n_slots,),        dtype=np.float32),
        "expanded":         spaces.Box(0.0, 1.0, shape=(n_slots,),        dtype=np.float32),
        "visible":          spaces.Box(0.0, 1.0, shape=(n_slots,),        dtype=np.float32),
    })
