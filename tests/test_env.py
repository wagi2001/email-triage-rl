import numpy as np
import pytest

from email_data import ACTION_NAMES, EXPAND_ACTION, OPEN_ACTION, PAGE_ACTION, email_matches_rule
from env import EmailTriageEnv
from verifier import Verifier
from agent import RuleBasedAgent


def _run_oracle_episode(env, seed=10):
    agent = RuleBasedAgent()
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env


# ------------------------------------------------------------------
# Full episode
# ------------------------------------------------------------------

def test_oracle_achieves_perfect_triage():
    with EmailTriageEnv(n_emails=6, difficulty="medium", partial_obs=True) as env:
        _run_oracle_episode(env, seed=10)
        v = Verifier()
        verdict = v.log_verify(
            env.rule, env.emails, env.actions_taken,
            partial_obs=env.partial_obs, opened_ids=env.opened_ids,
        )
        assert verdict.triage_accuracy == pytest.approx(1.0)


def test_episode_return_matches_verifier():
    """env.episode_return (triage steps only) matches log_verify total_score."""
    with EmailTriageEnv(n_emails=6, difficulty="medium", partial_obs=True) as env:
        _run_oracle_episode(env, seed=4)
        v = Verifier()
        verdict = v.log_verify(
            env.rule, env.emails, env.actions_taken,
            partial_obs=env.partial_obs, opened_ids=env.opened_ids,
        )
        # episode_return may include invalid-action penalties not in log_verify,
        # but for the oracle agent (zero invalid actions) they must match exactly.
        assert env.episode_return == pytest.approx(verdict.total_score, abs=1e-5)


# ------------------------------------------------------------------
# Per-step reward behaviour
# ------------------------------------------------------------------

def test_open_step_gives_zero_reward():
    with EmailTriageEnv(n_emails=4, difficulty="medium", partial_obs=True) as env:
        env.reset(seed=0)
        open_idx = ACTION_NAMES.index(OPEN_ACTION)
        _, r_open, _, _, _ = env.step([0, open_idx])
        assert r_open == pytest.approx(0.0)


def test_triage_step_gives_nonzero_reward():
    with EmailTriageEnv(n_emails=4, difficulty="medium", partial_obs=True) as env:
        env.reset(seed=0)
        skip_idx = ACTION_NAMES.index("skip")
        _, r_skip, _, _, _ = env.step([0, skip_idx])
        assert r_skip != 0.0


def test_invalid_action_penalized():
    with EmailTriageEnv(n_emails=4, partial_obs=True) as env:
        env.reset(seed=1)
        open_idx = ACTION_NAMES.index(OPEN_ACTION)
        env.step([0, open_idx])
        # Second open on the same email is illegal
        _, r, _, _, info = env.step([0, open_idx])
        assert not info["last_action_valid"]
        assert r == pytest.approx(Verifier.INVALID_ACTION_PENALTY)


# ------------------------------------------------------------------
# Action mask
# ------------------------------------------------------------------

def test_action_mask_allows_triage_before_open():
    with EmailTriageEnv(n_emails=4, partial_obs=True) as env:
        _, info = env.reset(seed=1)
        mask     = info["action_mask"]
        open_idx = ACTION_NAMES.index(OPEN_ACTION)
        skip_idx = ACTION_NAMES.index("skip")
        assert mask[0, open_idx]
        assert mask[0, skip_idx]   # triage allowed without opening first


def test_action_mask_after_open_allows_triage():
    with EmailTriageEnv(n_emails=4, partial_obs=True) as env:
        obs, info = env.reset(seed=1)
        open_idx  = ACTION_NAMES.index(OPEN_ACTION)
        skip_idx  = ACTION_NAMES.index("skip")
        _, _, _, _, info = env.step([0, open_idx])
        mask = info["action_mask"]
        assert not mask[0, open_idx]   # can't open again
        assert mask[0, skip_idx]       # triage now allowed


# ------------------------------------------------------------------
# Observation shape
# ------------------------------------------------------------------

def test_obs_keys_and_shapes():
    with EmailTriageEnv(n_emails=8) as env:
        obs, _ = env.reset(seed=0)
        assert obs["rule_action"].shape      == (4,)
        assert obs["rule_compound"].shape    == (4,)
        assert obs["filter_on_sender"].shape == (1,)
        assert obs["step_fraction"].shape    == (1,)
        assert obs["primary_match"].shape    == (8,)
        assert obs["secondary_match"].shape  == (8,)
        assert obs["opened"].shape           == (8,)
        assert obs["triaged"].shape          == (8,)
        # filter_on_sender is binary
        assert obs["filter_on_sender"][0] in (0.0, 1.0)
        # step_fraction starts at 0 before any steps
        assert obs["step_fraction"][0] == 0.0


# ------------------------------------------------------------------
# Context manager
# ------------------------------------------------------------------

def test_context_manager_closes_cleanly():
    env = EmailTriageEnv(n_emails=4)
    with env:
        env.reset(seed=0)
    assert env._page is None


# ------------------------------------------------------------------
# Email threads
# ------------------------------------------------------------------

def test_thread_messages_attached_when_use_threads():
    with EmailTriageEnv(n_emails=4, use_threads=True) as env:
        env.reset(seed=0)
        for email in env.emails:
            assert "messages" in email
            assert len(email["messages"]) >= 2  # opener + at least one reply


def test_no_thread_messages_by_default():
    with EmailTriageEnv(n_emails=4) as env:
        env.reset(seed=0)
        for email in env.emails:
            assert "messages" not in email


def test_expand_available_only_after_open():
    with EmailTriageEnv(n_emails=4, use_threads=True, partial_obs=True) as env:
        _, info = env.reset(seed=1)
        expand_idx = ACTION_NAMES.index(EXPAND_ACTION)
        open_idx   = ACTION_NAMES.index(OPEN_ACTION)
        mask = info["action_mask"]
        # Before open: expand must be blocked
        assert not mask[0, expand_idx]
        assert mask[0, open_idx]
        # After open: expand becomes available
        env.step([0, open_idx])
        mask2 = env._action_mask()
        assert mask2[0, expand_idx]


def test_expand_blocked_after_already_expanded():
    with EmailTriageEnv(n_emails=4, use_threads=True, partial_obs=True) as env:
        env.reset(seed=1)
        open_idx   = ACTION_NAMES.index(OPEN_ACTION)
        expand_idx = ACTION_NAMES.index(EXPAND_ACTION)
        env.step([0, open_idx])
        env.step([0, expand_idx])
        mask = env._action_mask()
        assert not mask[0, expand_idx]  # can't expand twice


def test_expand_step_gives_zero_reward():
    with EmailTriageEnv(n_emails=4, use_threads=True, partial_obs=True) as env:
        env.reset(seed=1)
        open_idx   = ACTION_NAMES.index(OPEN_ACTION)
        expand_idx = ACTION_NAMES.index(EXPAND_ACTION)
        env.step([0, open_idx])
        _, r, _, _, _ = env.step([0, expand_idx])
        assert r == pytest.approx(0.0)


def test_expanded_flag_appears_in_obs():
    with EmailTriageEnv(n_emails=4, use_threads=True, partial_obs=True) as env:
        obs, _ = env.reset(seed=1)
        assert obs["expanded"].shape == (4,)
        assert obs["expanded"].sum() == pytest.approx(0.0)  # nothing expanded yet
        open_idx   = ACTION_NAMES.index(OPEN_ACTION)
        expand_idx = ACTION_NAMES.index(EXPAND_ACTION)
        env.step([0, open_idx])
        env.step([0, expand_idx])
        assert env.emails[0]["id"] in env.expanded_ids
        obs2 = env._get_obs()
        assert obs2["expanded"][0] == pytest.approx(1.0)
        assert obs2["expanded"][1] == pytest.approx(0.0)


def test_oracle_with_threads_achieves_perfect_triage():
    with EmailTriageEnv(n_emails=6, difficulty="medium", partial_obs=True,
                        use_threads=True) as env:
        _run_oracle_episode(env, seed=10)
        v = Verifier()
        verdict = v.log_verify(
            env.rule, env.emails, env.actions_taken,
            partial_obs=env.partial_obs, opened_ids=env.opened_ids,
        )
        assert verdict.triage_accuracy == pytest.approx(1.0)


# ------------------------------------------------------------------
# Pagination
# ------------------------------------------------------------------

def test_pagination_masks_off_page_emails():
    with EmailTriageEnv(n_emails=8, page_size=4, partial_obs=True) as env:
        _, info = env.reset(seed=0)
        mask     = info["action_mask"]
        skip_idx = ACTION_NAMES.index("skip")
        # Page 0 emails (0-3) are actionable
        assert mask[0, skip_idx]
        assert mask[3, skip_idx]
        # Page 1 emails (4-7) are blocked
        assert not mask[4, skip_idx]
        assert not mask[7, skip_idx]


def test_next_page_navigates_inbox():
    with EmailTriageEnv(n_emails=8, page_size=4, partial_obs=True) as env:
        _, info = env.reset(seed=0)
        page_idx = ACTION_NAMES.index(PAGE_ACTION)
        skip_idx = ACTION_NAMES.index("skip")
        # Navigate to page 1
        env.step([0, page_idx])
        mask = env._action_mask()
        # Now page 1 emails (4-7) are actionable
        assert mask[4, skip_idx]
        assert not mask[0, skip_idx]


def test_next_page_gives_zero_reward():
    with EmailTriageEnv(n_emails=8, page_size=4, partial_obs=True) as env:
        env.reset(seed=0)
        page_idx = ACTION_NAMES.index(PAGE_ACTION)
        _, r, _, _, _ = env.step([0, page_idx])
        assert r == pytest.approx(0.0)


def test_next_page_wraps_to_first():
    with EmailTriageEnv(n_emails=8, page_size=4, partial_obs=True) as env:
        env.reset(seed=0)
        page_idx = ACTION_NAMES.index(PAGE_ACTION)
        env.step([0, page_idx])  # → page 1
        env.step([0, page_idx])  # → page 0 (wraps)
        assert env.current_page == 0


def test_visible_obs_reflects_current_page():
    with EmailTriageEnv(n_emails=8, page_size=4, partial_obs=True) as env:
        obs, _ = env.reset(seed=0)
        assert obs["visible"].shape == (8,)
        # Page 0: slots 0-3 visible, 4-7 hidden
        assert obs["visible"][:4].sum() == pytest.approx(4.0)
        assert obs["visible"][4:].sum() == pytest.approx(0.0)
        # Navigate to page 1
        page_idx = ACTION_NAMES.index(PAGE_ACTION)
        _, _, _, _, info = env.step([0, page_idx])
        obs2 = env._get_obs()
        assert obs2["visible"][:4].sum() == pytest.approx(0.0)
        assert obs2["visible"][4:].sum() == pytest.approx(4.0)


def test_next_page_blocked_when_single_page():
    with EmailTriageEnv(n_emails=4, page_size=None, partial_obs=True) as env:
        _, info = env.reset(seed=0)
        page_idx = ACTION_NAMES.index(PAGE_ACTION)
        assert not info["action_mask"][0, page_idx]


def test_oracle_with_pagination_achieves_perfect_triage():
    with EmailTriageEnv(n_emails=8, difficulty="medium", partial_obs=True,
                        page_size=4) as env:
        _run_oracle_episode(env, seed=10)
        v = Verifier()
        verdict = v.log_verify(
            env.rule, env.emails, env.actions_taken,
            partial_obs=env.partial_obs, opened_ids=env.opened_ids,
        )
        assert verdict.triage_accuracy == pytest.approx(1.0)


def test_oracle_with_threads_and_pagination():
    with EmailTriageEnv(n_emails=8, difficulty="medium", partial_obs=True,
                        use_threads=True, page_size=4) as env:
        _run_oracle_episode(env, seed=7)
        v = Verifier()
        verdict = v.log_verify(
            env.rule, env.emails, env.actions_taken,
            partial_obs=env.partial_obs, opened_ids=env.opened_ids,
        )
        assert verdict.triage_accuracy == pytest.approx(1.0)
