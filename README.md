# Email Triage RL

A Gymnasium-style environment for evaluating LLM agents on synthetic email triage. Each episode presents a natural-language rule, a batch of emails in a Playwright inbox UI, and a fixed action set (delete, archive, label, skip, open). A separate verifier scores triage decisions and optional step costs.

Best suited for **comparing LLM models** and demos—not for training classical RL policies out of the box (sparse rewards, scripted email order, text lives in prompts not in `obs`).

## Features

- **Playwright inbox** (`inbox.html`) — actions update the browser; optional DOM verification
- **Partial observability** (default) — senders hidden until an email is opened
- **Difficulty levels** — decoys, guaranteed rule matches, subject-keyword rules on hard mode
- **Unified LLM agent** — Ollama (cloud or local), OpenAI, Anthropic
- **Strict verifier** — per-email rewards, step penalty, triage accuracy vs reward efficiency
- **Trajectory logging** — `trajectories/latest.jsonl` (one JSON object per episode)

## Requirements

- Python 3.9+
- Chromium via Playwright

## Setup

```bash
cd "email triage RL"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Copy environment template and add API keys (never commit `.env`):

```bash
cp .env.example .env
```

| Provider | Variables |
|----------|-----------|
| Ollama cloud | `OLLAMA_API_KEY`, `OLLAMA_HOST`, `OLLAMA_MODEL` |
| Ollama local | `OLLAMA_HOST=http://localhost:11434`, run `ollama serve` |
| OpenAI | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| Anthropic | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |

## Quick start

**Ollama cloud (default agent):**

```bash
python run.py --agent ollama --model kimi-k2.6 --difficulty hard --episodes 5
```

**Rule-based baseline (sanity check / high triage accuracy):**

```bash
python run.py --agent rule --difficulty hard --episodes 3
```

**Full observability (all senders visible):**

```bash
python run.py --agent ollama --no-partial-obs --episodes 2
```

**List Ollama models:**

```bash
python run.py --list-models
```

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--agent` | `ollama` | `random`, `rule`, `ollama`, `openai`, `anthropic`, `dom`, `all` |
| `--episodes` | `2` | Episodes per run |
| `--model` | env default | Override provider model name |
| `--difficulty` | `easy` | `easy`, `medium`, `hard` |
| `--partial-obs` / `--no-partial-obs` | on | Hide senders until opened |
| `--headless` | off | Browser without visible window |
| `--local` | off | Ollama at `localhost:11434` instead of cloud |
| `--no-log-trajectories` | off | Disable JSONL logging |
| `--trajectory-log PATH` | `trajectories/latest.jsonl` | Custom log path |

## Scoring

Per-email points (shown in the episode summary table):

- Correct action on matching email: **+0.04**
- Correct skip: **+0.01**
- Wrong action / false positive / missed match: penalties

Episode-level:

- **Step penalty**: −0.002 per environment step (open + triage under partial obs)
- **Blind triage penalty**: −0.05 if acting without opening under partial obs
- **Score cap**: episode reward clamped to roughly **−0.20 … 0.40**

Summary metrics:

- **Triage accuracy** — fraction of emails with the correct action
- **Reward efficiency** — final score ÷ max per-email score (includes step cost)

## Project layout

```
email_data.py        # Rules, inbox generation, email_matches_rule
env.py               # Gymnasium env + Playwright
verifier.py          # log_verify / dom_verify, rewards
llm_agent.py         # LLM providers + prompts
agent.py             # RandomAgent, RuleBasedAgent
run.py               # CLI entrypoint
inbox.html           # Browser UI
trajectory_logger.py # JSONL episode logs
```

## Trajectories

Each run overwrites `trajectories/latest.jsonl` (unless `--trajectory-log` is set). Each line includes the rule, emails, per-step actions, verifier metrics, and optional raw LLM responses.

## Git / secrets

- `.env` and `venv/` are gitignored
- Use `.env.example` as the template for collaborators
- Rotate any API key that was ever committed by mistake

## License

Add a license file if you publish this repo publicly.
