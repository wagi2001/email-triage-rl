# run.py
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone

from config import Settings, project_root
from env import EmailTriageEnv
from agent import RandomAgent, RuleBasedAgent
from logging_config import setup_logging
from verifier import Verifier

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------------

def run_episode(env, agent, verifier, ep_num, mode="random", trajectory_logger=None, quiet=False):
    obs, info = env.reset(seed=ep_num)

    if trajectory_logger:
        trajectory_logger.begin_episode(
            episode=ep_num,
            seed=ep_num,
            rule=env.rule,
            emails=env.emails,
            mode=mode,
            agent_meta={"mode": mode},
        )

    print(f"\n{'='*60}")
    print(f"Episode {ep_num + 1}  |  mode: {mode}")
    print(f"Rule: {info['rule']}")
    if not quiet:
        print(f"{'='*60}")

    # Let agents that care (e.g. LLMAgent) know the current episode number
    if hasattr(agent, "current_episode"):
        agent.current_episode = ep_num + 1

    done     = False
    step_idx = 0

    while not done:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        if trajectory_logger:
            trajectory_logger.log_step(
                step_idx, int(action[0]), int(action[1]),
                info.get("raw_reward", reward), terminated, truncated,
            )
        step_idx += 1
        done = terminated or truncated

    verdict = verifier.log_verify(
        env.rule, env.emails, env.actions_taken,
        partial_obs=env.partial_obs,
        opened_ids=env.opened_ids,
    )

    if trajectory_logger:
        trajectory_logger.end_episode(
            env.actions_taken, env.episode_return, verdict,
            opened_ids=env.opened_ids,
        )

    if quiet:
        print(
            f"  score={verdict.total_score:.3f}  "
            f"triage={verdict.triage_accuracy*100:.1f}%  "
            f"efficiency={verdict.reward_efficiency*100:.1f}%"
        )
    else:
        print(verdict.summary())

    return verdict


# ------------------------------------------------------------------
# Agent runners
# ------------------------------------------------------------------

def _env_kwargs(render_mode, difficulty, partial_obs=True,
                use_threads=False, page_size=None):
    n_emails = 12 if difficulty in ("hard", "expert") else 8
    return dict(n_emails=n_emails, render_mode=render_mode,
                difficulty=difficulty, partial_obs=partial_obs,
                use_threads=use_threads, page_size=page_size)


def _wrap_env(env, normalize_reward: bool):
    if normalize_reward:
        from training_wrappers import NormalizeReward
        env = NormalizeReward(env)
        logger.debug("NormalizeReward wrapper applied")
    return env


def run_random(n_episodes=3, trajectory_path=None, render_mode=None,
               difficulty="easy", partial_obs=True, normalize_reward=False, quiet=False,
               use_threads=False, page_size=None):
    print("\n" + "=" * 60)
    print("RANDOM AGENT")
    print("=" * 60)

    env      = _wrap_env(EmailTriageEnv(**_env_kwargs(render_mode, difficulty, partial_obs, use_threads, page_size)), normalize_reward)
    agent    = RandomAgent(env.action_space)
    verifier = Verifier()
    traj     = _make_traj_logger(trajectory_path, "random")

    scores = []
    for ep in range(n_episodes):
        scores.append(run_episode(env, agent, verifier, ep, mode="random", trajectory_logger=traj, quiet=quiet).total_score)
    env.close()

    _print_summary("Random", scores, n_episodes)


def run_llm(
    n_episodes=2, model=None, provider="ollama", use_local=False,
    trajectory_path=None, render_mode=None,
    difficulty="easy", partial_obs=True, normalize_reward=False, quiet=False,
    use_threads=False, page_size=None,
):
    from llm_agent import LLMAgent, verify_ollama_key

    if provider == "ollama":
        if use_local:
            from llm_agent import _ollama_default_host
            print(f"Local Ollama at {_ollama_default_host(use_local=True)}")
        elif os.environ.get("OLLAMA_API_KEY"):
            print("Verifying OLLAMA_API_KEY …")
            verify_ollama_key(model=model)
            print("API key OK.")
        else:
            print("OLLAMA_API_KEY not set — falling back to local Ollama.")
            use_local = True

    print("\n" + "=" * 60)
    print(f"{provider.upper()} LLM AGENT")
    print("=" * 60)

    env      = _wrap_env(EmailTriageEnv(**_env_kwargs(render_mode, difficulty, partial_obs, use_threads, page_size)), normalize_reward)
    agent    = LLMAgent(env, provider=provider, model=model, use_local=use_local)
    verifier = Verifier()
    traj     = _make_traj_logger(trajectory_path, provider)

    print(f"Model: {agent.model}")
    scores, triage, eff = [], [], []
    for ep in range(n_episodes):
        v = run_episode(env, agent, verifier, ep, mode="llm", trajectory_logger=traj, quiet=quiet)
        scores.append(v.total_score)
        triage.append(v.triage_accuracy)
        eff.append(v.reward_efficiency)
    env.close()

    print(f"\n{provider} ({agent.model}) summary over {n_episodes} episodes:")
    print(f"  Scores:             {[round(s, 3) for s in scores]}")
    print(f"  Triage accuracy:    {[f'{a*100:.1f}%' for a in triage]}")
    print(f"  Reward efficiency:  {[f'{a*100:.1f}%' for a in eff]}")
    print(f"  Avg score:          {sum(scores)/len(scores):.3f}")


def run_rule_based(n_episodes=3, trajectory_path=None, render_mode=None,
                   difficulty="easy", partial_obs=True, normalize_reward=False, quiet=False,
                   use_threads=False, page_size=None):
    print("\n" + "=" * 60)
    print("RULE-BASED AGENT  (should score near max)")
    print("=" * 60)

    env      = _wrap_env(EmailTriageEnv(**_env_kwargs(render_mode, difficulty, partial_obs, use_threads, page_size)), normalize_reward)
    agent    = RuleBasedAgent()
    verifier = Verifier()
    traj     = _make_traj_logger(trajectory_path, "rule")

    scores, triage, eff = [], [], []
    for ep in range(n_episodes):
        v = run_episode(env, agent, verifier, ep, mode="rule_based", trajectory_logger=traj, quiet=quiet)
        scores.append(v.total_score)
        triage.append(v.triage_accuracy)
        eff.append(v.reward_efficiency)
    env.close()

    print(f"\nRule-based summary over {n_episodes} episodes:")
    print(f"  Scores:             {[round(s, 3) for s in scores]}")
    print(f"  Triage accuracy:    {[f'{a*100:.1f}%' for a in triage]}")
    print(f"  Reward efficiency:  {[f'{a*100:.1f}%' for a in eff]}")
    print(f"  Avg score:          {sum(scores)/len(scores):.3f}")


# ------------------------------------------------------------------
# Trajectory helpers
# ------------------------------------------------------------------

def _trajectories_dir():
    return os.path.join(project_root(), Settings.from_env().trajectory_dir)


def _make_traj_logger(path, agent_name):
    if path is None:
        return None
    from trajectory_logger import TrajectoryLogger
    run_meta = {"agent": agent_name, "started_at": datetime.now(timezone.utc).isoformat()}
    return TrajectoryLogger(path, run_meta=run_meta)


def _print_summary(name, scores, n):
    print(f"\n{name} summary over {n} episodes:")
    print(f"  Scores:  {[round(s, 3) for s in scores]}")
    print(f"  Average: {sum(scores)/len(scores):.3f}")


def _reset_trajectory_file(path):
    dirpath = _trajectories_dir()
    os.makedirs(dirpath, exist_ok=True)
    for name in os.listdir(dirpath):
        if name.endswith(".jsonl"):
            os.remove(os.path.join(dirpath, name))
    open(path, "w", encoding="utf-8").close()
    return path


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _load_dotenv():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (value or key not in os.environ):
                os.environ[key] = value


def main():
    _load_dotenv()
    settings = Settings.from_env()

    parser = argparse.ArgumentParser(description="Email triage RL environment runner")
    parser.add_argument(
        "--agent",
        choices=["random", "rule", "all", "ollama", "openai", "anthropic"],
        default="all",
        help="Which agent to run (default: all)",
    )
    parser.add_argument(
        "--model", default=None,
        help="LLM model name (overrides OLLAMA_MODEL / OPENAI_MODEL / ANTHROPIC_MODEL)",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Use local Ollama at localhost:11434 instead of cloud",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available Ollama models and exit",
    )
    parser.add_argument(
        "--episodes", type=int, default=settings.default_episodes,
        help="Episodes per agent",
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard", "expert"],
        default=settings.default_difficulty,
        help="Episode difficulty",
    )
    parser.add_argument(
        "--partial-obs", action=argparse.BooleanOptionalAction, default=True,
        help="Hide sender until opened (default: on)",
    )
    parser.add_argument(
        "--normalize-reward", action="store_true", default=settings.normalize_reward,
        help="Wrap env with NormalizeReward (for RL training loops)",
    )
    parser.add_argument(
        "--render", action="store_true", default=False,
        help="Open browser UI (default: headless / no browser)",
    )
    parser.add_argument(
        "--threads", action="store_true", default=False,
        help="Each email is a thread; agent must expand before triaging",
    )
    parser.add_argument(
        "--page-size", type=int, default=None, metavar="N",
        help="Split inbox into pages of N emails; agent must navigate with next_page",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Structured log output instead of verbose console",
    )
    parser.add_argument(
        "--no-log-trajectories", action="store_true",
        help="Disable JSONL trajectory logging",
    )
    args = parser.parse_args()
    setup_logging(settings.log_level, quiet=args.quiet)

    if args.list_models:
        from llm_agent import list_ollama_models
        list_ollama_models()
        print("\nLocal Ollama (if ollama serve is running):")
        list_ollama_models(host="http://localhost:11434")
        return

    render_mode = "human" if args.render else None

    traj_path = None
    if not args.no_log_trajectories:
        dirpath   = _trajectories_dir()
        os.makedirs(dirpath, exist_ok=True)
        traj_path = _reset_trajectory_file(os.path.join(dirpath, "latest.jsonl"))
        print(f"Trajectory log: {traj_path}")

    diff = args.difficulty
    pobs = args.partial_obs
    norm = args.normalize_reward

    if diff != "easy":
        print(f"Difficulty: {diff}")
    print(f"Partial obs: {'on' if pobs else 'off'}")
    if norm:
        print("Reward normalisation: enabled")
    if render_mode:
        print("Browser: visible")

    kw = dict(
        n_episodes=args.episodes,
        trajectory_path=traj_path,
        render_mode=render_mode,
        difficulty=diff,
        partial_obs=pobs,
        normalize_reward=norm,
        quiet=args.quiet,
        use_threads=args.threads,
        page_size=args.page_size,
    )

    if args.threads:
        print("Threads: on")
    if args.page_size:
        print(f"Page size: {args.page_size}")

    if args.agent in ("random", "all"):
        run_random(**kw)
    if args.agent in ("rule", "all"):
        run_rule_based(**kw)
    if args.agent in ("ollama", "openai", "anthropic"):
        run_llm(**kw, model=args.model, provider=args.agent, use_local=args.local)


if __name__ == "__main__":
    main()
