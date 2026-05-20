# env.py
import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from playwright.sync_api import sync_playwright
from email_data import ACTION_NAMES, OPEN_ACTION, TRIAGE_ACTIONS, generate_episode
from verifier import Verifier

# Absolute path to inbox.html
HTML_PATH = "file://" + os.path.abspath("inbox.html")


class EmailTriageEnv(gym.Env):
    """
    Gymnasium environment for email triage.

    Observation: dict with
        - 'emails': list of email dicts (sender_email, subject, id)
        - 'rule': string instruction
        - 'actions_taken': dict of {email_id: action_index}

    Action: Tuple (email_index, action_index)
        email_index  — which email in the inbox (0 to n_emails-1)
        action_index — delete, archive, label, skip, or open (reveal sender)

    Reward: Given at end of episode (strict scale via Verifier.log_verify).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, n_emails=8, render_mode=None, difficulty="easy", partial_obs=True):
        super().__init__()
        self.n_emails = n_emails
        self.render_mode = render_mode
        self.difficulty = difficulty
        self.partial_obs = partial_obs

        # Action space: (which email, which action)
        self.action_space = spaces.MultiDiscrete([n_emails, len(ACTION_NAMES)])

        # Observation space: flat array
        # For now: n_emails × 4 one-hot slots (action taken per email) + rule embedding placeholder
        self.observation_space = spaces.Dict({
            "actions_taken": spaces.Box(
                low=0, high=1,
                shape=(n_emails, len(ACTION_NAMES)),
                dtype=np.float32,
            ),
            "opened": spaces.Box(
                low=0, high=1, shape=(n_emails,), dtype=np.float32
            ),
        })

        # Playwright state
        self._playwright = None
        self._browser = None
        self._page = None

        # Episode state
        self.emails = []
        self.rule = {}
        self.actions_taken = {}  # {email_id: triage action name}
        self.opened_ids = set()  # email ids whose sender has been revealed
        self.step_count = 0
        self.max_steps = n_emails * (3 if partial_obs else 2)
        self._verifier = Verifier()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Launch browser if needed
        if self._page is None:
            self._start_browser()

        difficulty = (options or {}).get("difficulty", self.difficulty)
        self.emails, self.rule = generate_episode(
            n_emails=self.n_emails, seed=seed, difficulty=difficulty
        )
        self.actions_taken = {}
        self.opened_ids = set()
        self.step_count = 0

        # Load inbox into the browser
        self._page.evaluate(
            "([emailList, ruleText, partialObs]) => loadInbox(emailList, ruleText, partialObs)",
            [self.emails, self.rule["description"], self.partial_obs],
        )

        obs = self._get_obs()
        info = {
            "rule": self.rule["description"],
            "emails": self._visible_emails_for_agent(),
            "partial_obs": self.partial_obs,
            "opened_ids": list(self.opened_ids),
        }
        return obs, info

    def step(self, action):
        email_idx, action_idx = int(action[0]), int(action[1])

        # Clamp to valid range (safety)
        email_idx = max(0, min(email_idx, len(self.emails) - 1))
        action_idx = max(0, min(action_idx, len(ACTION_NAMES) - 1))

        email = self.emails[email_idx]
        eid = email["id"]
        action_name = ACTION_NAMES[action_idx]

        if action_name == OPEN_ACTION:
            self.opened_ids.add(eid)
            self._page.evaluate(
                "([emailId]) => openEmail(emailId)",
                [eid],
            )
        else:
            self.actions_taken[eid] = action_name
            self._page.evaluate(
                "([emailId, actionName]) => act(emailId, actionName)",
                [eid, action_name],
            )

        self.step_count += 1

        # Episode ends when every email has a triage action (open alone is not enough)
        all_actioned = len(self.actions_taken) >= len(self.emails)
        truncated = self.step_count >= self.max_steps
        terminated = all_actioned

        reward = 0.0
        if terminated or truncated:
            reward = self._compute_reward()

        obs = self._get_obs()
        info = {
            "actions_taken": self.actions_taken.copy(),
            "opened_ids": list(self.opened_ids),
        }

        return obs, reward, terminated, truncated, info

    def is_email_opened(self, email_idx):
        eid = self.emails[email_idx]["id"]
        return eid in self.opened_ids

    def open_email_idx(self, email_idx):
        """Reveal sender for this email (partial observability). Returns new obs."""
        open_idx = ACTION_NAMES.index(OPEN_ACTION)
        obs, _, _, _, _ = self.step([email_idx, open_idx])
        return obs

    def _visible_emails_for_agent(self):
        """Emails as the agent sees them (senders redacted if not opened)."""
        if not self.partial_obs:
            return list(self.emails)
        out = []
        for e in self.emails:
            if e["id"] in self.opened_ids:
                out.append(dict(e))
            else:
                out.append(
                    {
                        "id": e["id"],
                        "sender_name": "[hidden]",
                        "sender_email": "[hidden]",
                        "subject": e["subject"],
                    }
                )
        return out

    def close(self):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        self._playwright = None
        self._browser = None
        self._page = None

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self):
        return self._verifier.log_verify(
            self.rule,
            self.emails,
            self.actions_taken,
            n_steps=self.step_count,
            partial_obs=self.partial_obs,
            opened_ids=self.opened_ids,
        ).total_score

    def verify_episode(self, use_dom=False):
        """Return EpisodeVerdict from log or DOM (ground truth)."""
        if use_dom and self._page is not None:
            return self._verifier.dom_verify(
                self.rule,
                self.emails,
                self._page,
                n_steps=self.step_count,
                partial_obs=self.partial_obs,
                opened_ids=self.opened_ids,
            )
        return self._verifier.log_verify(
            self.rule,
            self.emails,
            self.actions_taken,
            n_steps=self.step_count,
            partial_obs=self.partial_obs,
            opened_ids=self.opened_ids,
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        # Build a one-hot matrix: n_emails × n_actions
        # 1.0 where an action has been taken, 0.0 otherwise
        matrix = np.zeros((self.n_emails, len(ACTION_NAMES)), dtype=np.float32)
        opened = np.zeros(self.n_emails, dtype=np.float32)
        for email_idx, email in enumerate(self.emails):
            eid = email["id"]
            if eid in self.opened_ids:
                opened[email_idx] = 1.0
            if eid in self.actions_taken:
                action_name = self.actions_taken[eid]
                if action_name in ACTION_NAMES:
                    a_idx = ACTION_NAMES.index(action_name)
                    matrix[email_idx, a_idx] = 1.0
        return {"actions_taken": matrix, "opened": opened}

    # ------------------------------------------------------------------
    # Browser helpers
    # ------------------------------------------------------------------

    def _start_browser(self):
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=(self.render_mode != "human")
            )
            self._page = self._browser.new_page()
            self._page.goto(HTML_PATH)
            self._page.wait_for_load_state("domcontentloaded")
        except Exception as e:
            if "Executable doesn't exist" in str(e):
                raise RuntimeError(
                    "Playwright Chromium is not installed. In your venv run:\n"
                    "  playwright install chromium"
                ) from e
            raise

    def render(self):
        # In "human" mode the browser window is visible — no extra work needed
        pass

    # ------------------------------------------------------------------
    # Dev helper: print current inbox state
    # ------------------------------------------------------------------

    def print_state(self):
        print(f"\nRule: {self.rule['description']}")
        print(f"{'ID':<4} {'Sender':<30} {'Subject':<35} {'Action'}")
        print("-" * 85)
        for email in self.emails:
            eid = email["id"]
            taken = self.actions_taken.get(eid, "—")
            print(f"{eid:<4} {email['sender_email']:<30} {email['subject']:<35} {taken}")