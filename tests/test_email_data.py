import random

import pytest

from email_data import COMPOUND_RULES, LOOKALIKES, email_matches_rule, generate_episode, get_random_rule, _check_condition


def test_sender_rule_exact_match():
    rule = {
        "filter_key": "sender_email",
        "filter_value": "promo@sale.com",
        "action": "delete",
    }
    assert email_matches_rule(
        {"sender_email": "promo@sale.com", "subject": "x"}, rule
    )
    assert not email_matches_rule(
        {"sender_email": "promo@sale.net", "subject": "x"}, rule
    )


def test_subject_contains_case_insensitive():
    rule = {
        "filter_key": "subject",
        "filter_value": "digest",
        "filter_op": "contains",
        "action": "label:newsletter",
    }
    assert email_matches_rule(
        {"subject": "Dev digest #42", "sender_email": "a@b.com"}, rule
    )
    assert not email_matches_rule(
        {"subject": "Team update", "sender_email": "a@b.com"}, rule
    )


def test_hard_episode_has_guaranteed_matches():
    emails, rule = generate_episode(n_emails=12, seed=7, difficulty="hard")
    matches = sum(1 for e in emails if email_matches_rule(e, rule))
    # Compound rules guarantee 2 full matches; simple rules guarantee 3+.
    # Hard mode draws from both pools, so the floor is 2.
    assert matches >= 2


def test_medium_episode_has_some_matches():
    emails, rule = generate_episode(n_emails=8, seed=3, difficulty="medium")
    matches = sum(1 for e in emails if email_matches_rule(e, rule))
    assert matches >= 2


# --- Compound rule tests ---

AND_RULE = {
    "description": "archive emails from noreply@github.com only if subject contains 'PR'",
    "action": "archive",
    "filter_key": "sender_email",
    "filter_value": "noreply@github.com",
    "also_requires": {"filter_key": "subject", "filter_value": "PR", "filter_op": "contains"},
}

EXCEPT_RULE = {
    "description": "delete emails from promo@sale.com except if subject contains 'won'",
    "action": "delete",
    "filter_key": "sender_email",
    "filter_value": "promo@sale.com",
    "except_if": {"filter_key": "subject", "filter_value": "won", "filter_op": "contains"},
}


def test_and_rule_both_conditions_match():
    email = {"sender_email": "noreply@github.com", "subject": "PR merged"}
    assert email_matches_rule(email, AND_RULE)


def test_and_rule_primary_only_no_match():
    # Right sender, wrong subject — should NOT match the AND rule
    email = {"sender_email": "noreply@github.com", "subject": "CI failed"}
    assert not email_matches_rule(email, AND_RULE)


def test_and_rule_secondary_only_no_match():
    # Subject has 'PR' but wrong sender
    email = {"sender_email": "bob@work.com", "subject": "PR review request"}
    assert not email_matches_rule(email, AND_RULE)


def test_except_rule_no_exception_matches():
    email = {"sender_email": "promo@sale.com", "subject": "50% OFF today only!"}
    assert email_matches_rule(email, EXCEPT_RULE)


def test_except_rule_exception_no_match():
    # Primary matches but exception is true — should NOT be actioned
    email = {"sender_email": "promo@sale.com", "subject": "You won!"}
    assert not email_matches_rule(email, EXCEPT_RULE)


def test_except_rule_primary_fails():
    # Wrong sender entirely
    email = {"sender_email": "promo@sale.net", "subject": "50% OFF today only!"}
    assert not email_matches_rule(email, EXCEPT_RULE)


def test_expert_episode_has_full_and_trap_emails():
    emails, rule = generate_episode(n_emails=12, seed=0, difficulty="expert")
    assert "also_requires" in rule or "except_if" in rule or "except_if_any" in rule

    full_matches = sum(1 for e in emails if email_matches_rule(e, rule))
    primary_matches = sum(1 for e in emails if e["sender_email"] == rule["filter_value"])

    assert full_matches >= 2, "need at least 2 actioned emails"
    assert primary_matches > full_matches, "trap emails must outnumber full matches"


def test_expert_rule_pool_is_compound_only():
    for seed in range(20):
        rule = get_random_rule(random.Random(seed), difficulty="expert")
        is_compound = (
            "also_requires" in rule or "except_if" in rule or "except_if_any" in rule
        )
        assert is_compound, f"non-compound rule at seed {seed}"


# --- Negated condition tests ---

NEGATED_AND_RULE = {
    "description": "archive emails from noreply@github.com only if subject does NOT contain 'merged'",
    "action": "archive",
    "filter_key": "sender_email",
    "filter_value": "noreply@github.com",
    "also_requires": {
        "filter_key": "subject",
        "filter_value": "merged",
        "filter_op": "contains",
        "negate": True,
    },
}


def test_negated_and_rule_non_matching_subject_matches():
    # "CI failed" does NOT contain 'merged' → full rule fires
    email = {"sender_email": "noreply@github.com", "subject": "CI failed"}
    assert email_matches_rule(email, NEGATED_AND_RULE)


def test_negated_and_rule_matching_subject_skipped():
    # "PR merged" DOES contain 'merged' → negation makes AND fail → skip
    email = {"sender_email": "noreply@github.com", "subject": "PR merged"}
    assert not email_matches_rule(email, NEGATED_AND_RULE)


def test_negated_and_rule_wrong_sender():
    email = {"sender_email": "bob@work.com", "subject": "CI failed"}
    assert not email_matches_rule(email, NEGATED_AND_RULE)


def test_negate_flag_in_check_condition():
    cond = {"filter_key": "subject", "filter_value": "merged", "filter_op": "contains", "negate": True}
    assert _check_condition({"subject": "CI failed"}, cond)
    assert not _check_condition({"subject": "PR merged"}, cond)


# --- Except-if-any tests ---

EXCEPT_ANY_RULE = {
    "description": "archive emails from noreply@github.com except if subject contains 'failed' or 'issue'",
    "action": "archive",
    "filter_key": "sender_email",
    "filter_value": "noreply@github.com",
    "except_if_any": [
        {"filter_key": "subject", "filter_value": "failed", "filter_op": "contains"},
        {"filter_key": "subject", "filter_value": "issue", "filter_op": "contains"},
    ],
}


def test_except_any_rule_no_exception_matches():
    # "PR merged" triggers no exception → archived
    email = {"sender_email": "noreply@github.com", "subject": "PR merged"}
    assert email_matches_rule(email, EXCEPT_ANY_RULE)


def test_except_any_rule_first_exception():
    # "CI failed" triggers exception 1 → skip
    email = {"sender_email": "noreply@github.com", "subject": "CI failed"}
    assert not email_matches_rule(email, EXCEPT_ANY_RULE)


def test_except_any_rule_second_exception():
    # "New issue opened" triggers exception 2 → skip
    email = {"sender_email": "noreply@github.com", "subject": "New issue opened"}
    assert not email_matches_rule(email, EXCEPT_ANY_RULE)


def test_except_any_rule_wrong_sender():
    email = {"sender_email": "bob@work.com", "subject": "PR merged"}
    assert not email_matches_rule(email, EXCEPT_ANY_RULE)


# --- Ensure new rules are in COMPOUND_RULES ---

def test_compound_rules_include_negated_rules():
    negated = [r for r in COMPOUND_RULES if r.get("also_requires", {}).get("negate")]
    assert len(negated) >= 2, "expected at least 2 negated AND rules"


def test_compound_rules_include_except_any_rules():
    except_any = [r for r in COMPOUND_RULES if "except_if_any" in r]
    assert len(except_any) >= 2, "expected at least 2 except_if_any rules"


def test_expert_inbox_flooded_with_lookalikes():
    """Expert episodes with sender rules must include near-miss lookalike addresses."""
    hits = 0
    for seed in range(20):
        emails, rule = generate_episode(n_emails=12, seed=seed, difficulty="expert")
        if rule.get("filter_key") != "sender_email":
            continue
        target = rule["filter_value"]
        lookalike_addrs = {addr for _, addr in LOOKALIKES.get(target, [])}
        n_lookalikes = sum(1 for e in emails if e["sender_email"] in lookalike_addrs)
        assert n_lookalikes >= 3, (
            f"seed {seed}: only {n_lookalikes} lookalike emails for target {target!r}"
        )
        hits += 1
    assert hits >= 5, "not enough sender-rule episodes sampled"


def test_expert_episode_traps_for_all_rule_types():
    """Across 30 seeds every expert episode must have trap emails."""
    for seed in range(30):
        emails, rule = generate_episode(n_emails=12, seed=seed, difficulty="expert")
        full = sum(1 for e in emails if email_matches_rule(e, rule))
        primary = sum(1 for e in emails if e["sender_email"] == rule["filter_value"])
        assert full >= 2, f"seed {seed}: only {full} full matches"
        assert primary > full, f"seed {seed}: no trap emails (primary={primary}, full={full})"
