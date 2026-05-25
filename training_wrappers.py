# training_wrappers.py
"""Gymnasium wrappers for RL training."""
from __future__ import annotations

import numpy as np
import gymnasium as gym


class RunningMeanStd:
    """Welford's online algorithm for running mean and variance (scalar)."""

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)

    @property
    def var(self) -> float:
        # Return 1.0 until we have enough samples so early steps pass through unchanged.
        return self._m2 / self.n if self.n >= 2 else 1.0


class NormalizeReward(gym.Wrapper):
    """
    Normalize per-step rewards by the running std of discounted returns.

    Follows the standard PPO/A2C approach: maintains a running estimate of
    the discounted return variance and divides each raw reward by its std.
    The mean is NOT subtracted — mean-centering removes the sign that tells
    the policy whether an action was good or bad.

    Raw reward is preserved in info["raw_reward"] so verifier consistency
    checks and trajectory logs remain interpretable.

    Args:
        gamma:   Discount factor used to accumulate the running return.
        epsilon: Added to std to prevent division by zero.
        clip:    Symmetric clip on the normalized reward to bound outliers.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ) -> None:
        super().__init__(env)
        self._return_rms = RunningMeanStd()
        self._discounted_return: float = 0.0
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._discounted_return = self._discounted_return * self.gamma + float(reward)
        self._return_rms.update(self._discounted_return)

        std = float(np.sqrt(self._return_rms.var)) + self.epsilon
        normalized = float(np.clip(float(reward) / std, -self.clip, self.clip))

        info["raw_reward"] = float(reward)
        info["reward_rms_std"] = std
        return obs, normalized, terminated, truncated, info

    def reset(self, **kwargs):
        self._discounted_return = 0.0
        return self.env.reset(**kwargs)

    def __getattr__(self, name: str):
        # Gymnasium 1.x removed Wrapper.__getattr__, so we add it back to keep
        # inner-env attributes (episode_return, rule, emails, …) accessible.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)
