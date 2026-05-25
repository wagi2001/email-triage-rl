# Email Triage RL

A Gymnasium environment for training RL policies on email triage.  Each episode presents a natural-language rule, a batch of emails, and a fixed action set (`delete`, `archive`, `label:newsletter`, `skip`, `open`).  A verifier scores per-step triage decisions and applies a step penalty.

## Overview

The agent's job: read the rule, open emails to reveal hidden senders (partial observability), and apply the correct action to every email.

Difficulty scales from random inboxes up to compound AND / EXCEPT rules with deliberately counterintuitive trap emails — designed so a policy cannot rely on surface-level heuristics.

## Setup

```bash
cd "email triage RL"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium   # only needed for --render (browser UI)
```

## Quick start

```bash
# Validate the environment — rule-based oracle should score near 100% triage accuracy
python run.py --agent rule --difficulty expert --episodes 5

# Both agents, default settings
python run.py

# Expert compound rules, no browser (fast)
python run.py --agent all --difficulty expert --episodes 10 --quiet

# Show browser UI
python run.py --agent rule --difficulty hard --render
```

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--agent` | `all` | `random`, `rule`, `all` |
| `--episodes` | `2` | Episodes per agent |
| `--difficulty` | `hard` | `easy`, `medium`, `hard`, `expert` |
| `--partial-obs` / `--no-partial-obs` | on | Hide senders until each email is opened |
| `--render` | off | Open Playwright browser UI (default: no browser) |
| `--normalize-reward` | off | Wrap env with `NormalizeReward` (for RL training loops) |
| `--quiet` | off | Structured log output; suppress verbose episode tables |
| `--no-log-trajectories` | off | Disable JSONL trajectory logging |

## Difficulty levels

| Level | Inbox size | Rule pool | Guaranteed matches | Traps |
|-------|-----------|-----------|-------------------|-------|
| `easy` | 8 | Simple sender / action rules | 0 | 0 |
| `medium` | 8 | Simple rules | 2–3 | Decoy lookalikes |
| `hard` | 12 | Simple + subject-keyword rules | 3–5 | Decoys |
| `expert` | 12 | Compound AND / EXCEPT rules only | 2 full matches | 3–4 traps from same sender |

**Expert rule types:**

- `AND` — action only when primary AND secondary condition both hold (e.g. "archive from `noreply@github.com` only if subject contains 'PR'")
- `EXCEPT` — action unless an exception condition fires (e.g. "delete from `promo@sale.com` except if subject contains 'won'")
- `EXCEPT_ANY` — action unless any of several exception keywords appear
- Negated AND — action only if subject does NOT contain a keyword (counterintuitive: spares the obvious emails)

Trap emails match only the primary condition (right sender, wrong subject), requiring the policy to correctly apply both parts of the rule.

## RL interface

```python
import gymnasium as gym
from env import EmailTriageEnv

env = EmailTriageEnv(n_emails=12, difficulty="expert", partial_obs=True)
obs, info = env.reset(seed=0)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

Or via Gymnasium registry:

```python
env = gym.make("EmailTriage-v0", difficulty="expert")
```

### Observation space

All arrays are `float32`.

| Key | Shape | Meaning |
|-----|-------|---------|
| `rule_action` | `(4,)` | One-hot: which triage action the rule demands |
| `rule_compound` | `(4,)` | One-hot: rule type — none / AND / EXCEPT / EXCEPT_ANY |
| `primary_match` | `(n_emails,)` | 1 if primary filter matches; 0 until opened in partial-obs |
| `secondary_match` | `(n_emails,)` | 1 if secondary condition allows the action (always visible) |
| `opened` | `(n_emails,)` | 1 if this email has been opened |
| `triaged` | `(n_emails,)` | 1 if this email has received a triage action |

**Decision logic for the policy:**
```
full_match = primary_match[i] == 1 AND secondary_match[i] == 1
action     = rule_action if full_match else skip
```
`secondary_match` is always 1 for simple rules, so this formula works uniformly across all rule types.

### Action space

`MultiDiscrete([n_emails, n_actions])` — choose an email index and an action index.

Actions: `delete (0)`, `archive (1)`, `label:newsletter (2)`, `skip (3)`, `open (4)`

`info["action_mask"]` is a `(n_emails, n_actions)` boolean array of legal moves.

### Reward

Per-step, unclamped (no episode-level cap).

| Event | Reward |
|-------|--------|
| Correct triage action | +0.04 |
| Correct skip | +0.01 |
| Missed required action | −0.03 |
| Wrong action applied | −0.06 |
| False positive | −0.04 |
| Step penalty | −0.002 per step |
| Invalid action attempt | −0.01 |

### Reward normalisation (optional)

For training with PPO / similar algorithms, wrap the env:

```python
from training_wrappers import NormalizeReward
env = NormalizeReward(EmailTriageEnv(...))
```

Or pass `--normalize-reward` to `run.py`.  Raw reward is stored in `info["raw_reward"]`.

## Metrics

After each episode the verifier reports:

- **Triage accuracy** — fraction of emails with the correct action (100% = all right)
- **Reward efficiency** — `total_score / max_possible` (penalised by step cost)

Example: a perfect policy on 12 emails uses 24 steps (open + triage each), paying `24 × 0.002 = 0.048` in step penalties, so reward efficiency is ~73–82% even at 100% triage accuracy.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRIAGE_DIFFICULTY` | `hard` | Default difficulty |
| `TRIAGE_EPISODES` | `2` | Default episode count |
| `TRIAGE_PARTIAL_OBS` | `true` | Partial observability |
| `TRIAGE_NORMALIZE_REWARD` | `false` | NormalizeReward wrapper |
| `TRIAGE_TRAJECTORY_DIR` | `trajectories` | Trajectory log directory |
| `TRIAGE_LOG_LEVEL` | `INFO` | Logging level |

## Trajectory logging

Each run clears old `*.jsonl` files in `trajectories/` and writes `trajectories/latest.jsonl`. Each line is one episode and includes the rule, emails, per-step actions, rewards, and final metrics. Disable with `--no-log-trajectories`.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

No browser or API keys required.  CI runs on push via `.github/workflows/ci.yml`.

## Project layout

```
email_data.py        # Rules, inbox generation, email_matches_rule
obs_encoding.py      # Structured RL observation (match features, rule one-hots)
env.py               # Gymnasium env (EmailTriage-v0)
verifier.py          # Per-step and episode scoring
agent.py             # RandomAgent, RuleBasedAgent (validation baselines)
run.py               # CLI runner
training_wrappers.py # NormalizeReward wrapper
config.py            # Settings and env-var defaults
logging_config.py    # Structured logging
trajectory_logger.py # JSONL episode logging
inbox.html           # Browser UI (render_mode="human" only)
compare_models.py    # Side-by-side agent comparison table
tests/               # Unit tests
```

## Git / secrets

`.env` and `venv/` are gitignored. Rotate any key that was ever committed by mistake.
