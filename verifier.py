# verifier.py
"""
Verifier for the Email Triage RL Environment.

Responsibility: given the rule, the emails, and either
  (a) the agent's action log, or
  (b) the live browser DOM state via Playwright,
determine whether the agent correctly completed the task.

Two verification modes:
  - log_verify()  : fast, checks the in-memory action dict
  - dom_verify()  : ground truth, reads actual DOM state from the browser
                    use this when evaluating LLM agents you don't fully trust

The reward function in env.py uses log_verify() during training for speed.
dom_verify() is used in eval runs and for catching reward hacking.
"""

from dataclasses import dataclass, field
from typing import Optional

from email_data import email_matches_rule


@dataclass
class EmailVerdict:
    email_id: int
    sender_email: str
    subject: str
    should_be_actioned: bool      # does this email match the rule?
    correct_action: Optional[str] # what action the rule requires (None if should be skipped)
    taken_action: str             # what the agent actually did
    is_correct: bool              # verdict
    points: float                 # reward contribution
    reason: str                   # human-readable explanation


@dataclass
class EpisodeVerdict:
    rule_description: str
    total_score: float
    max_possible: float
    triage_accuracy: float        # correct_actions / total_emails
    reward_efficiency: float      # total_score / max_possible (after step penalty)
    verdicts: list[EmailVerdict] = field(default_factory=list)
    false_positives: int = 0      # emails wrongly actioned
    false_negatives: int = 0      # emails that should have been actioned but weren't
    correct_actions: int = 0
    total_emails: int = 0

    @property
    def accuracy(self) -> float:
        """Alias for reward_efficiency (backward compatibility)."""
        return self.reward_efficiency

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
            ok = "✓" if v.is_correct else "✗"
            expected = v.correct_action if v.correct_action else "skip"
            lines.append(
                f"{v.email_id:<4} {v.sender_email:<30} {expected:<18} "
                f"{v.taken_action:<18} {v.points:>+6.3f}  {ok}"
            )
        return "\n".join(lines)


class Verifier:
    """
    Stateless verifier. Pass in the rule, emails, and actions — get back
    a full EpisodeVerdict with per-email breakdowns.

    Strict scoring targets mean episode reward ~0.1–0.4 for strong agents
    (old scale topped out around 2.0+ with 100% accuracy).
    """

    # Strict reward scale — env.py uses this via log_verify()
    CORRECT_ACTION_REWARD = +0.04
    CORRECT_SKIP_REWARD = +0.01
    MISSED_ACTION_PENALTY = -0.03
    WRONG_ACTION_PENALTY = -0.06
    FALSE_POSITIVE_PENALTY = -0.04
    STEP_PENALTY = 0.002          # per env step (open + triage add up)
    BLIND_TRIAGE_PENALTY = -0.05  # acted without opening sender (partial obs)
    EPISODE_MAX_SCORE = 0.40
    EPISODE_MIN_SCORE = -0.20

    def log_verify(
        self,
        rule: dict,
        emails: list[dict],
        actions_taken: dict[int, str],   # {email_id: action_name}
        n_steps: int = 0,
        partial_obs: bool = False,
        opened_ids=None,
    ) -> EpisodeVerdict:
        """
        Verify using the in-memory action log.
        Fast — use during training rollouts.
        """
        opened_ids = set(opened_ids or [])
        verdicts = []
        total_score = 0.0
        max_possible = 0.0
        false_positives = 0
        false_negatives = 0
        correct_actions = 0

        for email in emails:
            eid = email["id"]
            taken = actions_taken.get(eid, "skip")
            if taken == "open":
                taken = "skip"  # open is not a triage decision

            should_be_actioned = email_matches_rule(email, rule)
            correct_action = rule["action"] if should_be_actioned else None

            max_possible += (
                self.CORRECT_ACTION_REWARD if should_be_actioned else self.CORRECT_SKIP_REWARD
            )

            blind = (
                partial_obs
                and taken != "skip"
                and eid not in opened_ids
            )

            if blind:
                points = self.BLIND_TRIAGE_PENALTY
                is_correct = False
                if should_be_actioned:
                    false_negatives += 1
                else:
                    false_positives += 1
                reason = "Acted without opening email (sender was hidden)"
            elif should_be_actioned:
                if taken == correct_action:
                    points = self.CORRECT_ACTION_REWARD
                    is_correct = True
                    correct_actions += 1
                    reason = f"Correctly applied '{correct_action}'"
                elif taken == "skip":
                    points = self.MISSED_ACTION_PENALTY
                    is_correct = False
                    false_negatives += 1
                    reason = f"Should have applied '{correct_action}' but skipped"
                else:
                    points = self.WRONG_ACTION_PENALTY
                    is_correct = False
                    reason = f"Wrong action '{taken}', expected '{correct_action}'"
            else:
                if taken == "skip":
                    points = self.CORRECT_SKIP_REWARD
                    is_correct = True
                    correct_actions += 1
                    reason = "Correctly skipped (not in rule)"
                else:
                    points = self.FALSE_POSITIVE_PENALTY
                    is_correct = False
                    false_positives += 1
                    reason = f"Should have skipped but applied '{taken}'"

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

        if n_steps > 0:
            total_score -= n_steps * self.STEP_PENALTY

        total_score = max(
            self.EPISODE_MIN_SCORE,
            min(self.EPISODE_MAX_SCORE, total_score),
        )
        max_possible = min(self.EPISODE_MAX_SCORE, max_possible)

        triage_accuracy = (
            correct_actions / len(emails) if emails else 0.0
        )
        reward_efficiency = (
            (total_score / max_possible) if max_possible > 0 else 0.0
        )
        reward_efficiency = max(0.0, min(1.0, reward_efficiency))

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

    def dom_verify(
        self,
        rule: dict,
        emails: list[dict],
        page,                            # Playwright page object
        n_steps: int = 0,
        partial_obs: bool = False,
        opened_ids=None,
    ) -> EpisodeVerdict:
        """
        Verify by reading DOM state directly from the browser.
        Slower — use during eval runs or when you don't trust the agent's
        action log (e.g. LLM agents that might hallucinate actions).

        Reads the 'data-action' attribute set on each email row in inbox.html.
        inbox.html must set this attribute when act() is called:
            row.setAttribute('data-action', actionName);
        """
        # Read DOM state
        dom_actions = page.evaluate("""
            () => {
                const rows = document.querySelectorAll('.email-row[data-id]');
                const result = {};
                rows.forEach(row => {
                    const id = parseInt(row.getAttribute('data-id'));
                    const action = row.getAttribute('data-action') || 'skip';
                    result[id] = action;
                });
                return result;
            }
        """)

        # dom_actions is {email_id_str: action_name} from JS
        # normalise keys to int
        actions_taken = {int(k): v for k, v in dom_actions.items()}

        return self.log_verify(
            rule,
            emails,
            actions_taken,
            n_steps=n_steps,
            partial_obs=partial_obs,
            opened_ids=opened_ids,
        )

    def assert_reward_consistency(
        self,
        env_reward: float,
        rule: dict,
        emails: list[dict],
        actions_taken: dict[int, str],
        tolerance: float = 1e-6,
        n_steps: int = 0,
        partial_obs: bool = False,
        opened_ids=None,
    ) -> bool:
        """
        Sanity check: does the reward env.py computed match what the
        verifier computes independently? Catches reward function drift.
        Call this in tests.
        """
        verdict = self.log_verify(
            rule,
            emails,
            actions_taken,
            n_steps=n_steps,
            partial_obs=partial_obs,
            opened_ids=opened_ids,
        )
        delta = abs(verdict.total_score - env_reward)
        if delta > tolerance:
            raise AssertionError(
                f"Reward mismatch: env computed {env_reward:.4f}, "
                f"verifier computed {verdict.total_score:.4f} (delta={delta:.6f})"
            )
        return True