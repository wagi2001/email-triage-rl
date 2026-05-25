import pytest

from email_data import email_matches_rule, generate_episode
from verifier import Verifier


def _perfect_actions(emails, rule, partial_obs):
    v = Verifier()
    actions = {}
    opened = set()
    for e in emails:
        opened.add(e["id"])
        if email_matches_rule(e, rule):
            actions[e["id"]] = rule["action"]
        else:
            actions[e["id"]] = "skip"
    return v.log_verify(
        rule, emails, actions,
        partial_obs=partial_obs, opened_ids=opened,
    )


def test_perfect_episode_triage_accuracy_is_one():
    emails, rule = generate_episode(n_emails=8, seed=1, difficulty="hard")
    verdict = _perfect_actions(emails, rule, partial_obs=True)
    assert verdict.triage_accuracy == pytest.approx(1.0)
    assert verdict.false_positives == 0
    assert verdict.false_negatives == 0


def test_open_step_reward_is_zero():
    """Open steps carry no reward — they are required infrastructure, not goals."""
    emails, rule = generate_episode(n_emails=4, seed=2, difficulty="medium")
    v = Verifier()
    email = emails[0]
    r = v.step_reward(email, rule, "open", partial_obs=True, opened_ids=set())
    assert r == pytest.approx(0.0)


def test_blind_triage_penalized():
    """A non-skip action applied without opening is penalised as a wrong action,
    even when it matches the rule — the sender was hidden, so the agent could not
    have known.  Opening first restores full credit."""
    emails, rule = generate_episode(n_emails=4, seed=0, difficulty="medium")
    match = next(e for e in emails if email_matches_rule(e, rule))
    v = Verifier()

    blind = v.log_verify(
        rule, emails, {match["id"]: rule["action"]},
        partial_obs=True, opened_ids=set(),
    )
    blind_row = next(x for x in blind.verdicts if x.email_id == match["id"])
    assert not blind_row.is_correct
    assert blind_row.points == Verifier.WRONG_ACTION_PENALTY

    opened = v.log_verify(
        rule, emails, {match["id"]: rule["action"]},
        partial_obs=True, opened_ids={match["id"]},
    )
    opened_row = next(x for x in opened.verdicts if x.email_id == match["id"])
    assert opened_row.is_correct
    assert opened_row.points == Verifier.CORRECT_ACTION_REWARD


def test_step_reward_matches_log_verify_points():
    emails, rule = generate_episode(n_emails=4, seed=3, difficulty="medium")
    email = emails[0]
    v = Verifier()
    opened = {email["id"]}
    r = v.step_reward(email, rule, "skip", partial_obs=True, opened_ids=opened)
    points, _, _, _, _ = v._score_triage_action(
        email, rule, "skip", partial_obs=True, opened_ids=opened
    )
    assert r == pytest.approx(points)


def test_invalid_action_reward():
    emails, rule = generate_episode(n_emails=4, seed=3, difficulty="medium")
    v = Verifier()
    r = v.step_reward(emails[0], rule, "skip", partial_obs=True,
                      opened_ids=set(), invalid=True)
    assert r == pytest.approx(Verifier.INVALID_ACTION_PENALTY)


def test_perfect_episode_max_score():
    """A perfect agent achieves reward_efficiency == 1.0."""
    emails, rule = generate_episode(n_emails=8, seed=5, difficulty="medium")
    verdict = _perfect_actions(emails, rule, partial_obs=True)
    assert verdict.reward_efficiency == pytest.approx(1.0)
    assert verdict.total_score == pytest.approx(verdict.max_possible)
