# agent.py
from email_data import ACTION_NAMES, email_matches_rule


class RandomAgent:
    """Takes random actions. Used to verify the env works."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()


class RuleBasedAgent:
    """
    A hand-coded agent that actually reads the rule and acts correctly.
    Use this to verify your reward function gives high scores.
    """

    def __init__(self, env):
        self.env = env

    def act(self, obs, email_idx):
        """Returns the correct action for email at email_idx."""
        rule = self.env.rule
        email = self.env.emails[email_idx]

        should_action = email_matches_rule(email, rule)

        if should_action:
            correct = rule["action"]
            return ACTION_NAMES.index(correct) if correct in ACTION_NAMES else 3
        else:
            return 3  # skip