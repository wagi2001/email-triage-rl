# env.py
from __future__ import annotations

import logging
import os
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import project_root
from email_data import (
    ACTION_NAMES, EXPAND_ACTION, OPEN_ACTION, PAGE_ACTION,
    SENDERS, THREAD_OPENER_SNIPPETS, THREAD_REPLY_SNIPPETS,
    generate_episode,
)
from obs_encoding import build_observation, observation_space
from verifier import Verifier

logger = logging.getLogger(__name__)

HTML_PATH = "file://" + os.path.join(project_root(), "inbox.html")


class EmailTriageEnv(gym.Env):
    """
    Gymnasium environment for email triage (RL training).

    Observation (structured, policy-ready):
        rule_action, rule_compound,
        primary_match, secondary_match, opened, triaged, expanded, visible

    Action: MultiDiscrete([n_emails, n_actions])
        policy chooses which email and what to do.

    Reward: per-step via Verifier.step_reward (triage only; open/expand/next_page = 0).

    New features (opt-in):
        use_threads=True  — each email is a multi-message thread; agent must open
                            then expand before triaging (adds steps + ambiguity).
        page_size=N       — inbox is paginated; agent must use next_page to reveal
                            later pages (adds navigation steps).

    Browser (render_mode="human" only):
        The Playwright browser is started lazily and synced in render().
        Pure Python training: instantiate with render_mode=None (default).
    """

    metadata = {"render_modes": ["human", None]}

    def __init__(
        self,
        n_emails: int = 8,
        render_mode=None,
        difficulty: str = "easy",
        partial_obs: bool = True,
        use_threads: bool = False,
        page_size: int | None = None,
    ):
        super().__init__()
        self.n_emails    = n_emails
        self.render_mode = render_mode
        self.difficulty  = difficulty
        self.partial_obs = partial_obs
        self.use_threads = use_threads
        self.page_size   = page_size

        self.action_space      = spaces.MultiDiscrete([n_emails, len(ACTION_NAMES)])
        self.observation_space = observation_space(n_emails)

        # Browser state (only used when render_mode="human")
        self._playwright = None
        self._browser    = None
        self._page       = None

        # Episode state
        self.emails         : list[dict]       = []
        self.rule           : dict             = {}
        self.actions_taken  : dict[int, str]   = {}
        self.opened_ids     : set[int]         = set()
        self.expanded_ids   : set[int]         = set()
        self.current_page   : int              = 0
        self.step_count     : int              = 0
        self.episode_return : float            = 0.0

        # Steps per email: open + triage = 2 base; +1 for expand; +N_pages nav
        steps_per_email = 2 + (1 if use_threads else 0)
        nav_steps       = (n_emails // (page_size or n_emails)) * 2 if page_size else 0
        self.max_steps  = n_emails * steps_per_email + nav_steps + n_emails

        self._verifier        = Verifier()
        self._open_action_idx = ACTION_NAMES.index(OPEN_ACTION)
        self._expand_idx      = ACTION_NAMES.index(EXPAND_ACTION)
        self._page_idx        = ACTION_NAMES.index(PAGE_ACTION)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        difficulty = (options or {}).get("difficulty", self.difficulty)
        self.emails, self.rule = generate_episode(
            n_emails=self.n_emails, seed=seed, difficulty=difficulty
        )
        if self.use_threads:
            self._add_thread_messages(self.emails)

        self.actions_taken  = {}
        self.opened_ids     = set()
        self.expanded_ids   = set()
        self.current_page   = 0
        self.step_count     = 0
        self.episode_return = 0.0

        if self.render_mode == "human":
            if self._page is None:
                self._start_browser()
            if self._page is not None:
                self._page.evaluate(
                    "([emails, rule, partial]) => loadInbox(emails, rule, partial)",
                    [self.emails, self.rule["description"], self.partial_obs],
                )

        return self._get_obs(), self._build_info()

    def step(self, action):
        email_idx  = int(np.clip(action[0], 0, len(self.emails) - 1))
        action_idx = int(np.clip(action[1], 0, len(ACTION_NAMES) - 1))
        action_name = ACTION_NAMES[action_idx]
        mask = self._action_mask()

        # --- Page navigation: email_idx is ignored, handled before email targeting ---
        if action_name == PAGE_ACTION:
            valid = bool(mask[0, action_idx])
            if valid and self.page_size is not None:
                n_pages = (len(self.emails) + self.page_size - 1) // self.page_size
                self.current_page = (self.current_page + 1) % n_pages
            reward = 0.0 if valid else self._verifier.INVALID_ACTION_PENALTY
            self.step_count     += 1
            self.episode_return += reward
            obs  = self._get_obs()
            info = self._build_info()
            info["last_action_valid"] = valid
            terminated = len(self.actions_taken) >= len(self.emails)
            truncated  = self.step_count >= self.max_steps
            if self.render_mode == "human":
                self.render()
            return obs, reward, terminated, truncated, info

        # --- Auto-advance: if chosen email already triaged, redirect to first
        #     untriaged email on the current page ---
        if self.emails[email_idx]["id"] in self.actions_taken:
            page_start = (self.current_page * self.page_size) if self.page_size else 0
            page_end   = (page_start + self.page_size) if self.page_size else len(self.emails)
            for i, e in enumerate(self.emails):
                if page_start <= i < page_end and e["id"] not in self.actions_taken:
                    email_idx = i
                    break

        email       = self.emails[email_idx]
        eid         = email["id"]
        valid       = bool(mask[email_idx, action_idx])

        if valid:
            if action_name == OPEN_ACTION:
                self.opened_ids.add(eid)
            elif action_name == EXPAND_ACTION:
                self.expanded_ids.add(eid)
            else:
                self.actions_taken[eid] = action_name

        reward = self._verifier.step_reward(
            email,
            self.rule,
            action_name,
            self.partial_obs,
            self.opened_ids,
            invalid=not valid,
        )
        self.step_count     += 1
        self.episode_return += reward

        terminated = len(self.actions_taken) >= len(self.emails)
        truncated  = self.step_count >= self.max_steps

        obs  = self._get_obs()
        info = self._build_info()
        info["last_action_valid"] = valid

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Sync browser to current episode state (no-op when render_mode is None)."""
        if self._page is None:
            return
        if self.render_mode == "human":
            self._page.evaluate(
                "([emails, rule, partial, openedIds, actionsTaken]) => "
                "loadInbox(emails, rule, partial, openedIds, actionsTaken)",
                [
                    self.emails,
                    self.rule.get("description", ""),
                    self.partial_obs,
                    list(self.opened_ids),
                    {str(k): v for k, v in self.actions_taken.items()},
                ],
            )

    def close(self):
        if self._browser:
            try:
                self._browser.close()
            except Exception as exc:
                logger.warning("Browser close failed: %s", exc)
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception as exc:
                logger.warning("Playwright stop failed: %s", exc)
        self._playwright = None
        self._browser    = None
        self._page       = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_email_opened(self, email_idx: int) -> bool:
        return self.emails[email_idx]["id"] in self.opened_ids

    def is_email_expanded(self, email_idx: int) -> bool:
        return self.emails[email_idx]["id"] in self.expanded_ids

    def _action_mask(self) -> np.ndarray:
        """(n_emails, n_actions) boolean mask — True where the action is legal."""
        mask = np.zeros((self.n_emails, len(ACTION_NAMES)), dtype=bool)

        # Pagination: visible index range for the current page
        if self.page_size is not None:
            page_start = self.current_page * self.page_size
            page_end   = page_start + self.page_size
            n_pages    = (len(self.emails) + self.page_size - 1) // self.page_size
            # next_page is valid at slot 0 whenever there are multiple pages
            if n_pages > 1:
                mask[0, self._page_idx] = True
        else:
            page_start, page_end = 0, len(self.emails)

        for idx, email in enumerate(self.emails):
            eid = email["id"]
            if eid in self.actions_taken:
                continue  # fully triaged: all actions blocked

            # Actions for emails outside the current page are blocked
            if not (page_start <= idx < page_end):
                continue

            opened   = eid in self.opened_ids
            expanded = eid in self.expanded_ids

            for a_idx, a_name in enumerate(ACTION_NAMES):
                if a_name == PAGE_ACTION:
                    continue  # handled above
                if a_name == OPEN_ACTION:
                    if not opened:
                        mask[idx, a_idx] = True
                elif a_name == EXPAND_ACTION:
                    # expand only available after open, and only once
                    if self.use_threads and opened and not expanded:
                        mask[idx, a_idx] = True
                else:
                    # Triage actions always available (verifier penalises blind triage)
                    mask[idx, a_idx] = True

        return mask

    def _get_obs(self) -> dict:
        return build_observation(
            self.rule,
            self.emails,
            self.n_emails,
            self.opened_ids,
            self.actions_taken,
            self.partial_obs,
            step_fraction=self.step_count / self.max_steps,
            expanded_ids=self.expanded_ids,
            current_page=self.current_page if self.page_size is not None else None,
            page_size=self.page_size,
        )

    def _build_info(self) -> dict:
        return {
            "rule":           self.rule.get("description", ""),
            "action_mask":    self._action_mask(),
            "opened_ids":     list(self.opened_ids),
            "expanded_ids":   list(self.expanded_ids),
            "current_page":   self.current_page,
            "episode_return": self.episode_return,
            "step_count":     self.step_count,
        }

    def print_state(self):
        print(f"\nRule: {self.rule['description']}")
        if self.page_size is not None:
            n_pages = (len(self.emails) + self.page_size - 1) // self.page_size
            print(f"Page: {self.current_page + 1}/{n_pages}")
        print(f"{'ID':<4} {'Sender':<30} {'Subject':<35} {'Action'}")
        print("-" * 85)
        for email in self.emails:
            eid   = email["id"]
            taken = self.actions_taken.get(eid, "—")
            print(f"{eid:<4} {email['sender_email']:<30} {email['subject']:<35} {taken}")

    def _add_thread_messages(self, emails: list[dict]) -> None:
        """Attach a realistic reply chain to each email."""
        rng = random.Random(sum((i + 1) * e["id"] for i, e in enumerate(emails)))
        others = SENDERS  # potential repliers
        for email in emails:
            n_replies = rng.randint(1, 3)
            messages = [{
                "sender_name":  email["sender_name"],
                "sender_email": email["sender_email"],
                "snippet":      rng.choice(THREAD_OPENER_SNIPPETS),
            }]
            candidates = [(n, a) for n, a in others if a != email["sender_email"]]
            for _ in range(n_replies):
                reply_name, reply_addr = rng.choice(candidates)
                messages.append({
                    "sender_name":  reply_name,
                    "sender_email": reply_addr,
                    "snippet":      rng.choice(THREAD_REPLY_SNIPPETS),
                })
            email["messages"] = messages

    def _start_browser(self):
        from playwright.sync_api import sync_playwright

        try:
            self._playwright = sync_playwright().start()
            self._browser    = self._playwright.chromium.launch(headless=False)
            logger.debug("Playwright started")
            self._page = self._browser.new_page()
            self._page.goto(HTML_PATH)
            self._page.wait_for_load_state("domcontentloaded")
        except Exception as e:
            if "Executable doesn't exist" in str(e):
                raise RuntimeError(
                    "Playwright Chromium not installed.  Run:\n"
                    "  playwright install chromium"
                ) from e
            raise


try:
    gym.register(
        id="EmailTriage-v0",
        entry_point="env:EmailTriageEnv",
        max_episode_steps=128,
    )
except gym.error.Error:
    pass
