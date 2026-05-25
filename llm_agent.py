# llm_agent.py — LLM triage agent (Ollama, OpenAI, Anthropic)
"""
Usage:
  LLMAgent(env, provider="ollama")   # or "openai", "anthropic"

Environment variables:
  Ollama:    OLLAMA_API_KEY, OLLAMA_HOST, OLLAMA_MODEL
  OpenAI:    OPENAI_API_KEY, OPENAI_MODEL
  Anthropic: ANTHROPIC_API_KEY, ANTHROPIC_MODEL
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from email_data import ACTION_NAMES, EXPAND_ACTION, OPEN_ACTION, PAGE_ACTION, TRIAGE_ACTIONS

PROVIDERS = ("ollama", "openai", "anthropic")

DEFAULT_MODELS = {
    "ollama":    "kimi-k2:cloud",
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
}

SYSTEM_PROMPT = """You are an email triage assistant. Apply the rule precisely.

Valid actions — respond with exactly one:
  delete
  archive
  label:newsletter
  skip

Instructions:
- Only act on emails that exactly match the rule (sender address or subject keyword).
- Similar-looking senders (e.g. promo@sale.net vs promo@sale.com) are NOT matches.
- If the email does not match the rule, respond with skip.
- Sender addresses are hidden until you open the email; you only see the real sender after it is opened.
- "only if" rules: apply the action ONLY when BOTH the primary AND the secondary condition are true; skip otherwise.
- "only if NOT" rules: apply the action only when the primary condition matches AND the subject does NOT contain the stated keyword; skip emails whose subject does contain it.
- "except if" rules: apply the action when the primary condition is true, UNLESS the exception is also true — then skip.
- "except if … or …" rules: skip if ANY of the listed exception keywords appear in the subject.

Respond with JSON only, no markdown fences:
{"action": "<one valid action>"}
"""


# ------------------------------------------------------------------
# Prompt helpers
# ------------------------------------------------------------------

def _format_email(email: dict, opened: bool = False, expanded: bool = False) -> str:
    base = f"id={email['id']} | subject: {email['subject']}"
    if opened:
        sender_info = f"{email['sender_name']} <{email['sender_email']}>"
    else:
        sender_info = "[sender hidden — open to reveal]"

    if "messages" in email:
        n = len(email["messages"])
        if expanded:
            thread_lines = "\n".join(
                f"      [{i+1}/{n}] {m['sender_name']} <{m['sender_email']}>: {m['snippet']}"
                for i, m in enumerate(email["messages"])
            )
            return f"{base} | {sender_info}\n    Thread ({n} messages):\n{thread_lines}"
        return f"{base} | {sender_info} [{n}-msg thread — expand to read full thread]"

    return f"{base} | {sender_info}"


def _build_user_prompt(env, email_idx: int) -> str:
    email     = env.emails[email_idx]
    rule_desc = env.rule.get("description", "")

    # Pagination context
    page_info = ""
    if getattr(env, "page_size", None) is not None:
        n_pages = (len(env.emails) + env.page_size - 1) // env.page_size
        page_info = f"Page {env.current_page + 1} of {n_pages} — use next_page to navigate.\n\n"

    # Show only emails visible on the current page
    page_start = env.current_page * env.page_size if getattr(env, "page_size", None) else 0
    page_end   = page_start + env.page_size if getattr(env, "page_size", None) else len(env.emails)

    inbox_lines = []
    for i, e in enumerate(env.emails):
        if not (page_start <= i < page_end):
            continue
        is_opened   = (not env.partial_obs) or env.is_email_opened(i)
        is_expanded = hasattr(env, "expanded_ids") and e["id"] in env.expanded_ids
        inbox_lines.append(
            f"  [{i}] {_format_email(e, opened=is_opened, expanded=is_expanded)}"
        )

    if env.actions_taken:
        processed_lines = [
            f"  id={e['id']}: {env.actions_taken[e['id']]}"
            for e in env.emails
            if e["id"] in env.actions_taken
        ]
    else:
        processed_lines = ["  (none yet)"]

    current_opened   = (not env.partial_obs) or env.is_email_opened(email_idx)
    current_expanded = hasattr(env, "expanded_ids") and email["id"] in env.expanded_ids
    current_from     = _format_email(
        email, opened=current_opened, expanded=current_expanded
    ).split("|", 1)[-1].strip()

    return (
        f"Rule: {rule_desc}\n\n"
        f"{page_info}"
        f"Current page inbox (senders hidden until opened):\n"
        f"{chr(10).join(inbox_lines)}\n\n"
        f"Already processed:\n"
        f"{chr(10).join(processed_lines)}\n\n"
        f"Choose triage action for this email only (delete, archive, label:newsletter, or skip):\n"
        f"  index={email_idx}, id={email['id']}\n"
        f"  {current_from}\n"
    )


# ------------------------------------------------------------------
# Response parsing
# ------------------------------------------------------------------

def _parse_action(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "skip"
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "action" in data:
            action = str(data["action"]).strip().lower()
            if action in TRIAGE_ACTIONS:
                return action
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            data   = json.loads(fenced.group(1))
            action = str(data.get("action", "")).strip().lower()
            if action in TRIAGE_ACTIONS:
                return action
        except json.JSONDecodeError:
            pass
    lowered = text.lower()
    for name in TRIAGE_ACTIONS:
        if name in lowered:
            return name
    return "skip"


# ------------------------------------------------------------------
# Ollama helpers
# ------------------------------------------------------------------

def _ollama_api_key() -> str:
    return (os.environ.get("OLLAMA_API_KEY") or "").strip()


def _ollama_default_host(use_local: bool = False) -> str:
    if use_local:
        return os.environ.get("OLLAMA_LOCAL_HOST", "http://localhost:11434")
    if _ollama_api_key():
        return os.environ.get("OLLAMA_HOST", "https://ollama.com")
    return os.environ.get("OLLAMA_LOCAL_HOST", "http://localhost:11434")


def _make_ollama_client(host=None, use_local: bool = False):
    from ollama import Client as OllamaClient

    host    = host or _ollama_default_host(use_local=use_local)
    api_key = _ollama_api_key()
    if api_key and not use_local:
        return OllamaClient(host=host, headers={"Authorization": f"Bearer {api_key}"})
    return OllamaClient(host=host)


def verify_ollama_key(model=None):
    """Test that OLLAMA_API_KEY works for /api/chat.  Raises RuntimeError on failure."""
    api_key = _ollama_api_key()
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is not set.\n"
            "  Create one at: https://ollama.com/settings/keys\n"
            "  Add to .env:   OLLAMA_API_KEY=your_key\n"
            "  Or run local:  python run.py --agent ollama --local --model llama3.2"
        )
    client = _make_ollama_client()
    try:
        client.chat(
            model=model or _default_model("ollama"),
            messages=[{"role": "user", "content": "reply with ok"}],
            options={"temperature": 0},
        )
        return True
    except Exception as e:
        err = str(e)
        if "401" in err or "unauthorized" in err.lower():
            raise RuntimeError(
                "Ollama cloud returned 401 — API key is invalid or revoked.\n"
                "  Fix: create a new key at https://ollama.com/settings/keys\n"
                "  Or use local:  python run.py --agent ollama --local --model llama3.2"
            ) from e
        raise


def _default_model(provider: str) -> str:
    env_key = {"ollama": "OLLAMA_MODEL", "openai": "OPENAI_MODEL", "anthropic": "ANTHROPIC_MODEL"}[provider]
    model   = os.environ.get(env_key)
    if model:
        return model
    if provider == "ollama" and not os.environ.get("OLLAMA_API_KEY"):
        return "llama3.2"
    return DEFAULT_MODELS[provider]


def list_ollama_models(host=None):
    host   = host or _ollama_default_host()
    client = _make_ollama_client(host=host, use_local=host.startswith("http://localhost"))
    print(f"\nOllama models on {host}\n{'-'*50}")
    try:
        resp = client.list()
    except Exception as e:
        print(f"ERROR: {e}")
        return []
    names = []
    for m in sorted(resp.get("models") or [], key=lambda x: x.get("name", "")):
        name = m.get("name") or m.get("model", "?")
        names.append(name)
        size = m.get("size")
        print(f"  {name:<42} {size/(1024**3):.1f} GB" if size else f"  {name}")
    print(f"\n{len(names)} model(s)")
    return names


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

class LLMAgent:
    """Triage agent backed by an LLM (Ollama / OpenAI / Anthropic)."""

    def __init__(self, env, provider: str = "ollama", model: str | None = None, use_local: bool = False):
        provider = provider.lower()
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}, got {provider!r}")

        self.env             = env
        self.provider        = provider
        self.use_local       = use_local
        self.model           = model or _default_model(provider)
        self.last_response: str | None = None
        self._client         = None
        self.current_episode: int = 0

        if provider == "ollama":
            self._client = _make_ollama_client(use_local=use_local)
        elif provider == "openai":
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set (add to .env or export)")
            self._client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set (add to .env or export)")
            self._client = anthropic.Anthropic(api_key=api_key)

    def act(self, obs, info) -> np.ndarray:
        """Navigate pages, open and expand emails as needed, then ask the LLM to triage."""
        mask       = info["action_mask"]
        open_idx   = ACTION_NAMES.index(OPEN_ACTION)
        expand_idx = ACTION_NAMES.index(EXPAND_ACTION)
        page_idx   = ACTION_NAMES.index(PAGE_ACTION)
        triaged    = obs["triaged"]
        visible    = obs.get("visible", np.ones(len(triaged)))
        n          = len(triaged)

        # Navigate to next page when the current page is fully processed
        if mask[0, page_idx]:
            all_visible_done = all(triaged[i] > 0.5 or visible[i] < 0.5 for i in range(n))
            if all_visible_done:
                return np.array([0, page_idx], dtype=np.int64)

        # Pick the first untriaged email on the current page
        email_idx = 0
        for i in range(n):
            if triaged[i] > 0.5 or visible[i] < 0.5:
                continue
            if mask[i].any():
                email_idx = i
                break

        # Open first if the sender is still hidden
        if mask[email_idx, open_idx]:
            return np.array([email_idx, open_idx], dtype=np.int64)

        # Expand thread if not yet revealed
        if mask[email_idx, expand_idx]:
            return np.array([email_idx, expand_idx], dtype=np.int64)

        user_prompt = _build_user_prompt(self.env, email_idx)
        content     = self._chat(user_prompt)
        self.last_response = content

        action_name = _parse_action(content)
        action_idx  = ACTION_NAMES.index(action_name)
        if not mask[email_idx, action_idx]:
            action_idx = ACTION_NAMES.index("skip")

        ep_tag = f"ep{self.current_episode}/" if self.current_episode else ""
        print(
            f"  [{ep_tag}{self.provider}/{self.model}] "
            f"email {email_idx} → {ACTION_NAMES[action_idx]}  "
            f"(raw: {content[:80]!r})"
        )
        return np.array([email_idx, action_idx], dtype=np.int64)

    # ------------------------------------------------------------------
    def _chat(self, user_prompt: str) -> str:
        if self.provider == "ollama":
            return self._chat_ollama(user_prompt)
        if self.provider == "openai":
            return self._chat_openai(user_prompt)
        return self._chat_anthropic(user_prompt)

    def _chat_ollama(self, user_prompt: str) -> str:
        try:
            resp = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                options={"temperature": 0},
            )
            return resp.message.content
        except Exception as e:
            err = str(e)
            if "401" in err or "unauthorized" in err.lower():
                raise RuntimeError(
                    "Ollama 401 on /api/chat — API key invalid.\n"
                    "  Set OLLAMA_API_KEY in .env or use --local."
                ) from e
            raise

    def _chat_openai(self, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    def _chat_anthropic(self, user_prompt: str) -> str:
        msg    = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0,
        )
        return "".join(b.text for b in msg.content if hasattr(b, "text"))
