# verifier.py
from __future__ import annotations

"""
Verifier for the Email Triage RL environment.

Scores a completed episode (or a single step) given the rule, the email list,
and the agent's action log.  Intentionally stateless — pass everything in.

Reward scale
------------
Only triage decisions (delete / archive / label / skip) carry a reward signal.
Open steps are neutral infrastructure — rewarding or penalising them would
conflict with the policy needing to open emails before it can triage them.

Correct triage action   +0.04
Correct skip            +0.01
Missed required action  -0.03
Wrong action applied    -0.06
False positive          -0.04
Invalid action          -0.01   (wasted move — not a triage decision)
Open step                0.00   (neutral — necessary prerequisite, not a goal)

No per-step penalty, no episode-level clamping.
"""

from dataclasses import dataclass, field
from typing import Optional

from email_data import EXPAND_ACTION, PAGE_ACTION, email_matches_rule


@dataclass
class EmailVerdict:
    email_id: int
    sender_email: str
    subject: str
    should_be_actioned: bool
    correct_action: Optional[str]
    taken_action: str
    is_correct: bool
    points: float
    reason: str


@dataclass
class EpisodeVerdict:
    rule_description: str
    total_score: float
    max_possible: float
    triage_accuracy: float
    reward_efficiency: float
    verdicts: list[EmailVerdict] = field(default_factory=list)
    false_positives: int = 0
    false_negatives: int = 0
    correct_actions: int = 0
    total_emails: int = 0

    def summary(self) -> str:
        lines = [
            f"Rule:              {self.rule_description}",
            f"Score:             {self.total_score:.3f} / {self.max_possible:.3f}",
            f"Triage accuracy:   {self.triage_accuracy * 100:.1f}%",
            f"Reward efficiency: {self.reward_efficiency * 100:.1f}%",
            f"Correct:           {self.correct_actions}/{self.total_emails}",
            f"False pos:         {self.false_positives}  (actioned emails that shouldn't be)",
            f"False neg:         {self.false_negatives}  (missed emails that should be actioned)",
            "",
            f"{'ID':<4} {'Sender':<30} {'Expected':<18} {'Taken':<18} {'Pts':>5}  {'OK'}",
            "-" * 85,
        ]
        for v in self.verdicts:
            ok       = "✓" if v.is_correct else "✗"
            expected = v.correct_action if v.correct_action else "skip"
            lines.append(
                f"{v.email_id:<4} {v.sender_email:<30} {expected:<18} "
                f"{v.taken_action:<18} {v.points:>+6.3f}  {ok}"
            )
        return "\n".join(lines)


class Verifier:
    """Stateless verifier.  Pass rule + emails + actions → EpisodeVerdict."""

    CORRECT_ACTION_REWARD  = +0.04
    CORRECT_SKIP_REWARD    = +0.01
    MISSED_ACTION_PENALTY  = -0.03
    WRONG_ACTION_PENALTY   = -0.06
    FALSE_POSITIVE_PENALTY = -0.04
    INVALID_ACTION_PENALTY = -0.01

    def _score_triage_action(
        self,
        email: dict,
        rule: dict,
        action_name: str,
        partial_obs: bool,
        opened_ids: set,
    ) -> tuple[float, bool, str, bool, bool]:
        """Score a triage action.  Returns (points, is_correct, reason, fp, fn)."""
        eid = email["id"]

        # Blind triage: agent applied a non-skip action without opening the email.
        # Sender was hidden so the agent guessed — penalise even if the guess was right.
        if partial_obs and eid not in opened_ids and action_name != "skip":
            return (self.WRONG_ACTION_PENALTY, False,
                    f"Blind triage: '{action_name}' applied without opening email first",
                    False, False)

        should_be_actioned = email_matches_rule(email, rule)
        correct_action     = rule["action"] if should_be_actioned else None

        if should_be_actioned:
            if action_name == correct_action:
                return (self.CORRECT_ACTION_REWARD, True,
                        f"Correctly applied '{correct_action}'", False, False)
            if action_name == "skip":
                return (self.MISSED_ACTION_PENALTY, False,
                        f"Should have applied '{correct_action}' but skipped", False, True)
            return (self.WRONG_ACTION_PENALTY, False,
                    f"Wrong action '{action_name}', expected '{correct_action}'", False, False)

        if action_name == "skip":
            return (self.CORRECT_SKIP_REWARD, True,
                    "Correctly skipped (not in rule)", False, False)
        return (self.FALSE_POSITIVE_PENALTY, False,
                f"Should have skipped but applied '{action_name}'", True, False)

    def step_reward(
        self,
        email: dict,
        rule: dict,
        action_name: str,
        partial_obs: bool,
        opened_ids: set,
        *,
        invalid: bool = False,
    ) -> float:
        """Reward for one env step.

        Only triage decisions produce a meaningful signal.
        Open steps return 0.0 — they are required infrastructure, not goals.
        Invalid moves are penalised to discourage wasted actions.
        """
        if invalid:
            return self.INVALID_ACTION_PENALTY
        if action_name in ("open", EXPAND_ACTION, PAGE_ACTION):
            return 0.0
        points, _, _, _, _ = self._score_triage_action(
            email, rule, action_name, partial_obs, opened_ids
        )
        return points

    def log_verify(
        self,
        rule: dict,
        emails: list[dict],
        actions_taken: dict[int, str],
        partial_obs: bool = False,
        opened_ids=None,
    ) -> EpisodeVerdict:
        """Score a completed episode from the in-memory action log."""
        opened_ids = set(opened_ids or [])
        verdicts        = []
        total_score     = 0.0
        max_possible    = 0.0
        false_positives = 0
        false_negatives = 0
        correct_actions = 0

        for email in emails:
            eid              = email["id"]
            taken            = actions_taken.get(eid, "skip")
            if taken == "open":
                taken = "skip"

            should_be_actioned = email_matches_rule(email, rule)
            correct_action     = rule["action"] if should_be_actioned else None

            max_possible += (
                self.CORRECT_ACTION_REWARD if should_be_actioned
                else self.CORRECT_SKIP_REWARD
            )

            points, is_correct, reason, fp, fn = self._score_triage_action(
                email, rule, taken, partial_obs, opened_ids
            )
            if fp:
                false_positives += 1
            if fn:
                false_negatives += 1
            if is_correct:
                correct_actions += 1

            total_score += points
            verdicts.append(EmailVerdict(
                email_id=eid,
                sender_email=email["sender_email"],
                subject=email["subject"],
                should_be_actioned=should_be_actioned,
                correct_action=correct_action,
                taken_action=taken,
                is_correct=is_correct,
                points=points,
                reason=reason,
            ))

        triage_accuracy = correct_actions / len(emails) if emails else 0.0

        # Partial-obs penalty: scale triage_accuracy down proportionally to how
        # many emails the agent skipped opening.  sqrt keeps the penalty gradual
        # (opening 75% of emails → 0.87× accuracy, opening 50% → 0.71×).
        if partial_obs and emails:
            opened_ratio = len(opened_ids) / len(emails)
            triage_accuracy *= opened_ratio ** 0.5

        reward_efficiency = (total_score / max_possible) if max_possible > 0 else 0.0

        return EpisodeVerdict(
            rule_description=rule["description"],
            total_score=total_score,
            max_possible=max_possible,
            triage_accuracy=triage_accuracy,
            reward_efficiency=reward_efficiency,
            verdicts=verdicts,
            false_positives=false_positives,
            false_negatives=false_negatives,
            correct_actions=correct_actions,
            total_emails=len(emails),
        )

