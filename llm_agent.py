# llm_agent.py — unified LLM triage agent (Ollama, OpenAI, Anthropic)
"""
Usage:
  LLMAgent(env, provider="ollama")   # or "openai", "anthropic"

Environment variables:
  Ollama:    OLLAMA_API_KEY, OLLAMA_HOST, OLLAMA_MODEL
  OpenAI:    OPENAI_API_KEY, OPENAI_MODEL
  Anthropic: ANTHROPIC_API_KEY, ANTHROPIC_MODEL
"""

import json
import os
import re

from email_data import ACTION_NAMES, TRIAGE_ACTIONS, format_email_line

PROVIDERS = ("ollama", "openai", "anthropic")

DEFAULT_MODELS = {
    "ollama": "gpt-oss:120b",  # cloud default; llama3.2 if local only
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
}

SYSTEM_PROMPT = """You are an email triage assistant. Apply the rule precisely.

Valid actions (use exactly one string):
- delete
- archive
- label:newsletter
- skip

Rules:
- Only act on emails that match the rule exactly (sender or subject keyword as stated).
- Similar senders (e.g. promo@sale.net vs promo@sale.com) are NOT matches unless the rule says so.
- If an email does NOT match the rule, respond with skip.
- Do not label newsletters unless the rule asks for it.
- Sender addresses are hidden until an email is opened; you only see the sender for the current email after it is opened.

Respond with JSON only, no markdown:
{"action": "<one valid action>"}
"""


def _parse_action(text):
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
            data = json.loads(fenced.group(1))
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


def _build_user_prompt(env, email_idx):
    email = env.emails[email_idx]
    rule_desc = env.rule.get("description", "")

    inbox_lines = []
    for i, e in enumerate(env.emails):
        opened = (not env.partial_obs) or env.is_email_opened(i)
        inbox_lines.append(f"  [{i}] {format_email_line(e, opened=opened)}")

    processed_lines = ["  (none yet)"]
    if env.actions_taken:
        processed_lines = [
            f"  id={e['id']}: {env.actions_taken[e['id']]}"
            for e in env.emails
            if e["id"] in env.actions_taken
        ] or processed_lines

    current_opened = (not env.partial_obs) or env.is_email_opened(email_idx)
    current_from = format_email_line(email, opened=current_opened).split("|", 1)[-1].strip()

    return f"""Rule: {rule_desc}

Full inbox (senders hidden until opened):
{chr(10).join(inbox_lines)}

Already processed:
{chr(10).join(processed_lines)}

Choose triage action for this email only (delete, archive, label:newsletter, or skip):
  index={email_idx}, id={email['id']}
  {current_from}
"""


def _ollama_api_key():
    return (os.environ.get("OLLAMA_API_KEY") or "").strip()


def _ollama_default_host(use_local=False):
    if use_local:
        return os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if _ollama_api_key():
        return os.environ.get("OLLAMA_HOST", "https://ollama.com")
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _make_ollama_client(host=None, use_local=False):
    from ollama import Client as OllamaClient

    host = host or _ollama_default_host(use_local=use_local)
    api_key = _ollama_api_key()
    if api_key and not use_local:
        return OllamaClient(host=host, headers={"Authorization": f"Bearer {api_key}"})
    return OllamaClient(host=host)


def verify_ollama_key(model=None):
    """
    Test that OLLAMA_API_KEY works for /api/chat (listing models does NOT require auth).
    Returns True if ok; raises RuntimeError with fix instructions otherwise.
    """
    api_key = _ollama_api_key()
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is not set. Cloud chat requires a valid key.\n"
            "  Create one: https://ollama.com/settings/keys\n"
            "  Then add to .env: OLLAMA_API_KEY=your_key\n"
            "Or use local Ollama: python run.py --agent ollama --local --model kimi-k2.6:cloud"
        )

    model = model or _default_model("ollama")
    client = _make_ollama_client()
    try:
        client.chat(
            model=model,
            messages=[{"role": "user", "content": "reply with ok"}],
            options={"temperature": 0},
        )
        return True
    except Exception as e:
        err = str(e)
        if "401" in err or "unauthorized" in err.lower():
            raise RuntimeError(
                "Ollama cloud returned 401 Unauthorized — your API key is invalid or revoked.\n"
                "  Fix: create a new key at https://ollama.com/settings/keys\n"
                "  Put it in .env (no quotes): OLLAMA_API_KEY=your_key_here\n"
                "  Or use local: ollama signin && ollama pull kimi-k2.6:cloud\n"
                "       python run.py --agent ollama --local --model kimi-k2.6:cloud"
            ) from e
        raise


def _default_model(provider):
    env_key = {
        "ollama": "OLLAMA_MODEL",
        "openai": "OPENAI_MODEL",
        "anthropic": "ANTHROPIC_MODEL",
    }[provider]
    model = os.environ.get(env_key)
    if model:
        return model
    if provider == "ollama" and not os.environ.get("OLLAMA_API_KEY"):
        return "llama3.2"
    return DEFAULT_MODELS[provider]


def list_ollama_models(host=None):
    """List models from Ollama (cloud or local)."""
    host = host or _ollama_default_host()
    client = _make_ollama_client(host=host, use_local=host.startswith("http://localhost"))

    print(f"\nOllama models on {host}\n{'-' * 50}")
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
        if size:
            print(f"  {name:<42} {size / (1024**3):.1f} GB")
        else:
            print(f"  {name}")
    print(f"\n{len(names)} model(s)")
    return names


def print_provider_hints():
    print("\nSuggested models (set via --model or .env):")
    print("  openai:    gpt-4o-mini, gpt-4o, gpt-4.1-mini")
    print("  anthropic: claude-3-5-haiku-latest, claude-sonnet-4-20250514")


class LLMAgent:
    """One agent class for ollama | openai | anthropic."""

    def __init__(self, env, provider="ollama", model=None, use_local=False):
        provider = provider.lower()
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}, got {provider!r}")

        self.env = env
        self.provider = provider
        self.use_local = use_local
        self.model = model or _default_model(provider)
        self.last_response = None
        self._client = None

        if provider == "ollama":
            if use_local and ":cloud" not in self.model:
                self.model = f"{self.model}:cloud"
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

    def act(self, obs, email_idx):
        user_prompt = _build_user_prompt(self.env, email_idx)
        content = self._chat(user_prompt)
        self.last_response = content
        action_name = _parse_action(content)
        print(
            f"  [{self.provider}] email {email_idx} → {action_name}  "
            f"(raw: {content[:80]!r}...)"
        )
        return ACTION_NAMES.index(action_name)

    def _chat(self, user_prompt):
        if self.provider == "ollama":
            return self._chat_ollama(user_prompt)
        if self.provider == "openai":
            return self._chat_openai(user_prompt)
        return self._chat_anthropic(user_prompt)

    def _chat_ollama(self, user_prompt):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0},
            )
            return response.message.content
        except Exception as e:
            err = str(e)
            if "401" in err or "unauthorized" in err.lower():
                raise RuntimeError(
                    "Ollama 401 Unauthorized on /api/chat.\n"
                    "  --list-models does NOT prove your key works (tags are public).\n"
                    "  Set a valid OLLAMA_API_KEY in .env, or use --local with ollama signin."
                ) from e
            raise

    def _chat_openai(self, user_prompt):
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    def _chat_anthropic(self, user_prompt):
        message = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0,
        )
        parts = []
        for block in message.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)
