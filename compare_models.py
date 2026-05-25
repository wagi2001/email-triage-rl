#!/usr/bin/env python3
"""
Compare multiple agents/models on the same seeds and print a leaderboard.

Examples:
  python compare_models.py kimi-k2.6 local/llama3.2:1b local/llama3.2:latest --episodes 5
  python compare_models.py rule kimi-k2.6 --difficulty hard --no-browser
  python compare_models.py openai/gpt-4o-mini anthropic/claude-3-5-haiku-latest --episodes 3
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from run import (
    _env_kwargs,
    _load_dotenv,
    _wrap_env,
    run_episode,
)
from config import Settings, project_root
from env import EmailTriageEnv
from agent import RuleBasedAgent
from verifier import Verifier


@dataclass
class ModelSpec:
    label: str
    kind: str  # llm | rule
    provider: str = "ollama"
    model: str | None = None
    use_local: bool = False


def parse_model_token(token: str) -> ModelSpec:
    token = token.strip()
    if token in ("rule", "rule-based", "oracle"):
        return ModelSpec(label="rule-based", kind="rule")

    if token.startswith("local/"):
        name = token.split("/", 1)[1]
        return ModelSpec(label=f"local/{name}", kind="llm", provider="ollama", model=name, use_local=True)

    if "/" in token:
        provider, name = token.split("/", 1)
        if provider not in ("ollama", "openai", "anthropic"):
            raise ValueError(f"Unknown provider {provider!r} in {token!r}")
        return ModelSpec(label=token, kind="llm", provider=provider, model=name, use_local=False)

    # Bare name → Ollama cloud (requires OLLAMA_API_KEY)
    return ModelSpec(label=token, kind="llm", provider="ollama", model=token, use_local=False)


def _benchmark_rule(n_episodes, difficulty, partial_obs, use_browser, quiet):
    env = EmailTriageEnv(**_env_kwargs(None, difficulty, partial_obs, use_browser))
    agent = RuleBasedAgent()
    verifier = Verifier()
    scores, triage, eff = [], [], []
    try:
        for ep in range(n_episodes):
            v = run_episode(env, agent, verifier, ep, mode="rule_based", quiet=quiet)
            scores.append(v.total_score)
            triage.append(v.triage_accuracy)
            eff.append(v.reward_efficiency)
    finally:
        env.close()
    return _summary("rule-based", "rule", None, False, scores, triage, eff)


def _benchmark_llm(spec: ModelSpec, n_episodes, difficulty, partial_obs, use_browser, quiet):
    from llm_agent import LLMAgent, verify_ollama_key

    if spec.provider == "ollama":
        if spec.use_local:
            if not quiet:
                print(f"\n>>> {spec.label} (local Ollama)")
        elif os.environ.get("OLLAMA_API_KEY"):
            if not quiet:
                print(f"\n>>> {spec.label} (Ollama cloud)")
            verify_ollama_key(model=spec.model)
        else:
            spec = ModelSpec(
                label=spec.label, kind="llm", provider="ollama",
                model=spec.model, use_local=True,
            )
            if not quiet:
                print(f"\n>>> {spec.label} (local — no OLLAMA_API_KEY)")

    env = _wrap_env(
        EmailTriageEnv(**_env_kwargs(None, difficulty, partial_obs, use_browser)),
        normalize_reward=False,
    )
    agent = LLMAgent(env, provider=spec.provider, model=spec.model, use_local=spec.use_local)
    verifier = Verifier()
    scores, triage, eff = [], [], []
    try:
        for ep in range(n_episodes):
            v = run_episode(env, agent, verifier, ep, mode="llm", quiet=quiet)
            scores.append(v.total_score)
            triage.append(v.triage_accuracy)
            eff.append(v.reward_efficiency)
    finally:
        env.close()
    return _summary(spec.label, spec.provider, agent.model, spec.use_local, scores, triage, eff)


def _summary(label, provider, model, use_local, scores, triage, eff):
    n = len(scores) or 1
    return {
        "label": label,
        "provider": provider,
        "model": model,
        "use_local": use_local,
        "episodes": len(scores),
        "scores": scores,
        "avg_score": sum(scores) / n,
        "avg_triage_accuracy": sum(triage) / n,
        "avg_reward_efficiency": sum(eff) / n,
        "triage_accuracies": triage,
        "reward_efficiencies": eff,
    }


def _print_table(rows: list[dict]):
    headers = ("Model", "Avg score", "Triage %", "Efficiency %", "Scores")
    print("\n" + "=" * 88)
    print("MODEL COMPARISON (same seeds 0..n-1 per model)")
    print("=" * 88)
    print(f"{headers[0]:<28} {headers[1]:>10} {headers[2]:>10} {headers[3]:>12}  {headers[4]}")
    print("-" * 88)
    for r in sorted(rows, key=lambda x: x["avg_score"], reverse=True):
        scores_str = ", ".join(f"{s:.3f}" for s in r["scores"])
        print(
            f"{r['label']:<28} {r['avg_score']:>10.3f} "
            f"{r['avg_triage_accuracy']*100:>9.1f}% "
            f"{r['avg_reward_efficiency']*100:>11.1f}%  [{scores_str}]"
        )
    print("=" * 88)


def main():
    _load_dotenv()
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Compare email triage agents/models")
    parser.add_argument(
        "models",
        nargs="+",
        help="Models: kimi-k2.6 (cloud), local/llama3.2:1b, openai/gpt-4o-mini, rule",
    )
    parser.add_argument("--episodes", type=int, default=settings.default_episodes)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=settings.default_difficulty)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Sim only, no Playwright (default for compare)",
    )
    parser.add_argument("--browser", action="store_true", help="Show Playwright inbox")
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true", help="Print per-episode details")
    parser.add_argument(
        "--output",
        default=None,
        help="JSONL path (default: trajectories/compare_<timestamp>.jsonl)",
    )
    args = parser.parse_args()

    quiet = not args.verbose
    use_browser = args.browser and not args.no_browser  # default: no browser (faster)
    specs = [parse_model_token(t) for t in args.models]

    print(f"Difficulty: {args.difficulty}  |  Episodes per model: {args.episodes}")
    print(f"Browser: {'on' if use_browser else 'off (sim)'}")

    rows = []
    for spec in specs:
        if spec.kind == "rule":
            rows.append(_benchmark_rule(args.episodes, args.difficulty, True, use_browser, quiet))
        else:
            rows.append(_benchmark_llm(spec, args.episodes, args.difficulty, True, use_browser, quiet))

    _print_table(rows)

    out = args.output
    if out is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = os.path.join(project_root(), "trajectories", f"compare_{ts}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        meta = {
            "type": "compare_run",
            "difficulty": args.difficulty,
            "episodes": args.episodes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        f.write(json.dumps(meta) + "\n")
        for r in rows:
            f.write(json.dumps({"type": "model_summary", **r}) + "\n")
    print(f"\nComparison saved to: {out}")


if __name__ == "__main__":
    main()
