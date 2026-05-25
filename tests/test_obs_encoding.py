import numpy as np

from obs_encoding import build_observation, observation_space

RULE_SIMPLE = {
    "description": "delete all emails from promo@sale.com",
    "action": "delete",
    "filter_key": "sender_email",
    "filter_value": "promo@sale.com",
}

RULE_AND = {
    "description": "archive emails from noreply@github.com only if subject contains 'PR'",
    "action": "archive",
    "filter_key": "sender_email",
    "filter_value": "noreply@github.com",
    "also_requires": {"filter_key": "subject", "filter_value": "PR", "filter_op": "contains"},
}

RULE_EXCEPT = {
    "description": "delete emails from promo@sale.com except if subject contains 'won'",
    "action": "delete",
    "filter_key": "sender_email",
    "filter_value": "promo@sale.com",
    "except_if": {"filter_key": "subject", "filter_value": "won", "filter_op": "contains"},
}

EMAIL_MATCH    = {"id": 0, "sender_email": "promo@sale.com",    "subject": "50% OFF"}
EMAIL_NO_MATCH = {"id": 1, "sender_email": "alice@work.com",    "subject": "Meeting notes"}
EMAIL_GITHUB   = {"id": 2, "sender_email": "noreply@github.com","subject": "PR merged"}
EMAIL_GITHUB_X = {"id": 3, "sender_email": "noreply@github.com","subject": "CI failed"}


def _obs(rule, emails, opened_ids=None, actions_taken=None, partial_obs=True, n=4):
    return build_observation(
        rule, emails, n,
        opened_ids    = set(opened_ids or []),
        actions_taken = dict(actions_taken or {}),
        partial_obs   = partial_obs,
    )


# ------------------------------------------------------------------
# observation_space
# ------------------------------------------------------------------

def test_observation_space_shapes():
    space = observation_space(8)
    assert space["rule_action"].shape      == (4,)
    assert space["rule_compound"].shape    == (4,)
    assert space["filter_on_sender"].shape == (1,)
    assert space["step_fraction"].shape    == (1,)
    assert space["primary_match"].shape    == (8,)
    assert space["secondary_match"].shape  == (8,)
    assert space["opened"].shape           == (8,)
    assert space["triaged"].shape          == (8,)


def test_filter_on_sender_sender_rule():
    obs = _obs(RULE_SIMPLE, [EMAIL_MATCH])
    assert obs["filter_on_sender"][0] == 1.0


def test_filter_on_sender_subject_rule():
    subject_rule = {
        "description": "delete emails whose subject contains 'OFF'",
        "action": "delete",
        "filter_key": "subject",
        "filter_value": "OFF",
        "filter_op": "contains",
    }
    obs = _obs(subject_rule, [EMAIL_MATCH])
    assert obs["filter_on_sender"][0] == 0.0


def test_step_fraction_default_zero():
    obs = _obs(RULE_SIMPLE, [EMAIL_MATCH])
    assert obs["step_fraction"][0] == 0.0


# ------------------------------------------------------------------
# rule_action one-hot
# ------------------------------------------------------------------

def test_rule_action_delete():
    obs = _obs(RULE_SIMPLE, [EMAIL_MATCH])
    assert obs["rule_action"][0] == 1.0   # delete is index 0
    assert obs["rule_action"].sum() == 1.0


def test_rule_action_archive():
    obs = _obs(RULE_AND, [EMAIL_GITHUB])
    assert obs["rule_action"][1] == 1.0   # archive is index 1


# ------------------------------------------------------------------
# rule_compound one-hot
# ------------------------------------------------------------------

def test_compound_none_for_simple_rule():
    obs = _obs(RULE_SIMPLE, [EMAIL_MATCH])
    assert obs["rule_compound"][0] == 1.0  # none

def test_compound_and():
    obs = _obs(RULE_AND, [EMAIL_GITHUB])
    assert obs["rule_compound"][1] == 1.0  # AND

def test_compound_except():
    obs = _obs(RULE_EXCEPT, [EMAIL_MATCH])
    assert obs["rule_compound"][2] == 1.0  # EXCEPT


# ------------------------------------------------------------------
# primary_match — hidden until opened in partial-obs
# ------------------------------------------------------------------

def test_primary_match_hidden_before_open():
    obs = _obs(RULE_SIMPLE, [EMAIL_MATCH], partial_obs=True)
    assert obs["primary_match"][0] == 0.0  # sender hidden

def test_primary_match_revealed_after_open():
    obs = _obs(RULE_SIMPLE, [EMAIL_MATCH], opened_ids=[0], partial_obs=True)
    assert obs["primary_match"][0] == 1.0

def test_primary_match_no_match_after_open():
    obs = _obs(RULE_SIMPLE, [EMAIL_NO_MATCH], opened_ids=[1], partial_obs=True)
    assert obs["primary_match"][0] == 0.0

def test_primary_match_full_obs_always_visible():
    obs = _obs(RULE_SIMPLE, [EMAIL_MATCH], partial_obs=False)
    assert obs["primary_match"][0] == 1.0


# ------------------------------------------------------------------
# secondary_match
# ------------------------------------------------------------------

def test_secondary_match_always_one_for_simple_rule():
    obs = _obs(RULE_SIMPLE, [EMAIL_MATCH, EMAIL_NO_MATCH])
    assert obs["secondary_match"][0] == 1.0
    assert obs["secondary_match"][1] == 1.0

def test_secondary_match_and_rule_subject_present():
    # "PR merged" contains 'PR' → secondary satisfied
    obs = _obs(RULE_AND, [EMAIL_GITHUB])
    assert obs["secondary_match"][0] == 1.0

def test_secondary_match_and_rule_subject_absent():
    # "CI failed" does not contain 'PR' → secondary not satisfied
    obs = _obs(RULE_AND, [EMAIL_GITHUB_X])
    assert obs["secondary_match"][0] == 0.0

def test_secondary_match_except_rule_no_exception():
    # "50% OFF" does not contain 'won' → exception not triggered → secondary = 1
    obs = _obs(RULE_EXCEPT, [EMAIL_MATCH])
    assert obs["secondary_match"][0] == 1.0

def test_secondary_match_except_rule_exception_fires():
    won_email = {"id": 0, "sender_email": "promo@sale.com", "subject": "You won!"}
    obs = _obs(RULE_EXCEPT, [won_email])
    assert obs["secondary_match"][0] == 0.0  # exception fired → skip


# ------------------------------------------------------------------
# opened / triaged flags
# ------------------------------------------------------------------

def test_opened_flag():
    obs = _obs(RULE_SIMPLE, [EMAIL_MATCH, EMAIL_NO_MATCH], opened_ids=[0])
    assert obs["opened"][0] == 1.0
    assert obs["opened"][1] == 0.0

def test_triaged_flag():
    obs = _obs(RULE_SIMPLE, [EMAIL_MATCH, EMAIL_NO_MATCH], actions_taken={0: "delete"})
    assert obs["triaged"][0] == 1.0
    assert obs["triaged"][1] == 0.0

def test_slots_beyond_email_count_are_zero():
    obs = _obs(RULE_SIMPLE, [EMAIL_MATCH], n=4)
    assert obs["primary_match"][1] == 0.0
    assert obs["secondary_match"][1] == 0.0
    assert obs["opened"][1] == 0.0
    assert obs["triaged"][1] == 0.0
