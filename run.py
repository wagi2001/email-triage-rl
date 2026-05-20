# run.py
import argparse
import os
from datetime import datetime, timezone

from email_data import OPEN_ACTION, ACTION_NAMES

from env import EmailTriageEnv
from agent import RandomAgent, RuleBasedAgent
from verifier import Verifier


def _agent_meta(agent, mode):
    meta = {"mode": mode}
    if hasattr(agent, "provider"):
        meta["provider"] = agent.provider
    if hasattr(agent, "model"):
        meta["model"] = agent.model
    if hasattr(agent, "use_local"):
        meta["use_local"] = agent.use_local
    return meta


def _log_step(trajectory_logger, step_idx, email_idx, action_idx, reward, terminated, truncated, llm_raw=None):
    if trajectory_logger:
        trajectory_logger.log_step(
            step_idx, email_idx, action_idx, reward, terminated, truncated,
            llm_raw_response=llm_raw,
        )


def _open_email_if_needed(env, obs, email_idx, trajectory_logger, step_idx):
    """Reveal sender for email_idx when partial observability is on."""
    if not env.partial_obs or env.is_email_opened(email_idx):
        return obs, 0.0, False, False, step_idx
    open_idx = ACTION_NAMES.index(OPEN_ACTION)
    obs, reward, terminated, truncated, _ = env.step([email_idx, open_idx])
    _log_step(trajectory_logger, step_idx, email_idx, open_idx, reward, terminated, truncated)
    return obs, reward, terminated, truncated, step_idx + 1


def _llm_response(agent):
    return getattr(agent, "last_response", None)


def run_episode(
    env,
    agent,
    verifier,
    ep_num,
    mode="random",
    trajectory_logger=None,
):
    obs, info = env.reset(seed=ep_num)

    if trajectory_logger:
        meta = _agent_meta(agent, mode)
        meta["partial_obs"] = env.partial_obs
        trajectory_logger.begin_episode(
            episode=ep_num,
            seed=ep_num,
            rule=env.rule,
            emails=env.emails,
            mode=mode,
            agent_meta=meta,
        )

    print(f"\n{'='*60}")
    print(f"Episode {ep_num+1} — mode: {mode}")
    print(f"Rule: {info['rule']}")
    print(f"{'='*60}")

    total_reward = 0.0
    done = False
    step_idx = 0

    if mode == "random":
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            if trajectory_logger:
                trajectory_logger.log_step(
                    step_idx,
                    int(action[0]),
                    int(action[1]),
                    reward,
                    terminated,
                    truncated,
                )
            step_idx += 1
            done = terminated or truncated
            total_reward += reward

    elif mode in ("rule_based", "llm"):
        for email_idx in range(len(env.emails)):
            obs, r, term, trunc, step_idx = _open_email_if_needed(
                env, obs, email_idx, trajectory_logger, step_idx
            )
            total_reward += r
            if term or trunc:
                done = True
                break

            action_idx = agent.act(obs, email_idx)
            obs, reward, terminated, truncated, _ = env.step([email_idx, action_idx])
            _log_step(
                trajectory_logger, step_idx, email_idx, action_idx,
                reward, terminated, truncated,
                llm_raw=_llm_response(agent) if mode == "llm" else None,
            )
            step_idx += 1
            total_reward += reward
            if terminated or truncated:
                break

    # --- Verify ---
    verdict = verifier.log_verify(
        env.rule,
        env.emails,
        env.actions_taken,
        n_steps=step_idx,
        partial_obs=env.partial_obs,
        opened_ids=env.opened_ids,
    )

    # Consistency check: verifier score must match env reward
    try:
        verifier.assert_reward_consistency(
            env_reward=total_reward,
            rule=env.rule,
            emails=env.emails,
            actions_taken=env.actions_taken,
            n_steps=step_idx,
            partial_obs=env.partial_obs,
            opened_ids=env.opened_ids,
        )
        consistency = "PASS"
    except AssertionError as e:
        consistency = f"FAIL — {e}"

    if trajectory_logger:
        trajectory_logger.end_episode(
            env.actions_taken,
            total_reward,
            verdict,
            reward_consistency=consistency,
            opened_ids=env.opened_ids,
        )

    print(verdict.summary())
    print(f"\nReward consistency check: {consistency}")

    return verdict


def _trajectories_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories")


def _default_trajectory_path(agent, model=None):
    """Single file overwritten each run (no timestamped history)."""
    return os.path.join(_trajectories_dir(), "latest.jsonl")


def _resolve_trajectory_path(path):
    dirpath = _trajectories_dir()
    os.makedirs(dirpath, exist_ok=True)
    if os.path.isabs(path):
        return path
    return os.path.join(dirpath, os.path.basename(path))


def _reset_trajectory_file(path):
    """Delete all old JSONL logs and start a fresh file for this run."""
    dirpath = _trajectories_dir()
    os.makedirs(dirpath, exist_ok=True)
    for name in os.listdir(dirpath):
        if name.endswith(".jsonl"):
            os.remove(os.path.join(dirpath, name))
    path = _resolve_trajectory_path(path)
    open(path, "w", encoding="utf-8").close()
    return path


def _make_trajectory_logger(path, agent_name, model=None, provider=None):
    from trajectory_logger import TrajectoryLogger

    path = _resolve_trajectory_path(path)
    run_meta = {
        "agent": agent_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if model:
        run_meta["model"] = model
    if provider:
        run_meta["provider"] = provider
    return TrajectoryLogger(path, run_meta=run_meta), path


def _env_kwargs(render_mode, difficulty, partial_obs=True):
    n_emails = 12 if difficulty == "hard" else 8
    return {
        "n_emails": n_emails,
        "render_mode": render_mode,
        "difficulty": difficulty,
        "partial_obs": partial_obs,
    }


def run_random(n_episodes=3, trajectory_path=None, render_mode="human", difficulty="easy", partial_obs=True):
    print("\n" + "="*60)
    print("RANDOM AGENT")
    print("="*60)

    env = EmailTriageEnv(**_env_kwargs(render_mode, difficulty, partial_obs))
    agent = RandomAgent(env.action_space)
    verifier = Verifier()
    logger = None
    if trajectory_path:
        logger, trajectory_path = _make_trajectory_logger(trajectory_path, "random")

    scores = []
    for ep in range(n_episodes):
        verdict = run_episode(
            env, agent, verifier, ep, mode="random", trajectory_logger=logger
        )
        scores.append(verdict.total_score)

    env.close()

    if logger:
        print(f"\nTrajectories logged to: {trajectory_path}")

    print(f"\nRandom agent summary over {n_episodes} episodes:")
    print(f"  Scores:  {[round(s, 3) for s in scores]}")
    print(f"  Average: {sum(scores)/len(scores):.3f}")


def run_rule_based(n_episodes=3, trajectory_path=None, render_mode="human", difficulty="easy", partial_obs=True):
    print("\n" + "="*60)
    print("RULE-BASED AGENT (should score near max)")
    print("="*60)

    env = EmailTriageEnv(**_env_kwargs(render_mode, difficulty, partial_obs))
    agent = RuleBasedAgent(env)
    verifier = Verifier()
    logger = None
    if trajectory_path:
        logger, trajectory_path = _make_trajectory_logger(trajectory_path, "rule")

    scores = []
    triage_accs = []
    eff_accs = []
    for ep in range(n_episodes):
        verdict = run_episode(
            env, agent, verifier, ep, mode="rule_based", trajectory_logger=logger
        )
        scores.append(verdict.total_score)
        triage_accs.append(verdict.triage_accuracy)
        eff_accs.append(verdict.reward_efficiency)

    env.close()

    if logger:
        print(f"\nTrajectories logged to: {trajectory_path}")

    print(f"\nRule-based agent summary over {n_episodes} episodes:")
    print(f"  Scores:             {[round(s, 3) for s in scores]}")
    print(f"  Triage accuracy:    {[f'{a*100:.1f}%' for a in triage_accs]}")
    print(f"  Reward efficiency:  {[f'{a*100:.1f}%' for a in eff_accs]}")
    print(f"  Avg score:          {sum(scores)/len(scores):.3f}")


def run_llm(
    n_episodes=2,
    model=None,
    render_mode="human",
    provider="ollama",
    use_local=False,
    trajectory_path=None,
    difficulty="easy",
    partial_obs=True,
):
    """Run unified LLMAgent (provider: ollama | openai | anthropic)."""
    from llm_agent import LLMAgent, verify_ollama_key

    if provider == "ollama":
        if use_local:
            print("Using local Ollama at http://localhost:11434 (run `ollama serve`).")
        elif os.environ.get("OLLAMA_API_KEY"):
            print("Verifying OLLAMA_API_KEY with a test chat request...")
            verify_ollama_key(model=model)
            print("API key OK.")
        else:
            print(
                "Note: OLLAMA_API_KEY not set — using local Ollama at "
                "http://localhost:11434 (run `ollama serve` and pull a model)."
            )
            use_local = True

    print(f"\nProvider: {provider}  |  model: {model or '(from env/default)'}")

    print("\n" + "=" * 60)
    print(f"{provider.upper()} LLM AGENT")
    print("=" * 60)

    env = EmailTriageEnv(**_env_kwargs(render_mode, difficulty, partial_obs))
    agent = LLMAgent(env, provider=provider, model=model, use_local=use_local)
    verifier = Verifier()
    logger = None
    if trajectory_path:
        logger, trajectory_path = _make_trajectory_logger(
            trajectory_path, provider, model=agent.model
        )

    scores = []
    triage_accs = []
    eff_accs = []
    for ep in range(n_episodes):
        verdict = run_episode(
            env, agent, verifier, ep, mode="llm", trajectory_logger=logger
        )
        scores.append(verdict.total_score)
        triage_accs.append(verdict.triage_accuracy)
        eff_accs.append(verdict.reward_efficiency)

    env.close()

    if logger:
        print(f"\nTrajectories logged to: {trajectory_path}")

    print(f"\n{provider} summary over {n_episodes} episodes:")
    print(f"  Model:              {agent.model}")
    print(f"  Scores:             {[round(s, 3) for s in scores]}")
    print(f"  Triage accuracy:    {[f'{a*100:.1f}%' for a in triage_accs]}")
    print(f"  Reward efficiency:  {[f'{a*100:.1f}%' for a in eff_accs]}")
    print(f"  Avg score:          {sum(scores)/len(scores):.3f}")


def run_dom_verify(n_episodes=2, trajectory_path=None, render_mode="human", difficulty="easy", partial_obs=True):
    """
    Runs the rule-based agent then cross-checks with DOM verification.
    Both log_verify and dom_verify should produce identical scores.
    """
    print("\n" + "="*60)
    print("DOM VERIFICATION CROSS-CHECK")
    print("="*60)

    env = EmailTriageEnv(**_env_kwargs(render_mode, difficulty, partial_obs))
    agent = RuleBasedAgent(env)
    verifier = Verifier()
    logger = None
    if trajectory_path:
        logger, trajectory_path = _make_trajectory_logger(trajectory_path, "dom")

    for ep in range(n_episodes):
        run_episode(
            env, agent, verifier, ep, mode="rule_based", trajectory_logger=logger
        )
        kw = dict(
            n_steps=env.step_count,
            partial_obs=env.partial_obs,
            opened_ids=env.opened_ids,
        )
        dom_verdict = verifier.dom_verify(env.rule, env.emails, env._page, **kw)
        log_verdict = verifier.log_verify(
            env.rule, env.emails, env.actions_taken, **kw
        )

        log_score = round(log_verdict.total_score, 4)
        dom_score = round(dom_verdict.total_score, 4)
        match = "MATCH" if log_score == dom_score else "MISMATCH"

        print(f"  dom_verify score: {dom_score}")
        print(f"  DOM vs log: {match}")

    env.close()

    if logger:
        print(f"\nTrajectories logged to: {trajectory_path}")


def _load_dotenv():
    """Load .env into os.environ (does not override existing vars)."""
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
            if key:
                # .env should win over empty shell exports; never override a real key
                if value or key not in os.environ:
                    os.environ[key] = value.strip()


def main():
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Email triage RL demo runners")
    parser.add_argument(
        "--agent",
        choices=["random", "rule", "ollama", "openai", "anthropic", "dom", "all"],
        default="ollama",
        help="Which agent to run (default: ollama)",
    )
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per agent")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (overrides OPENAI_MODEL / ANTHROPIC_MODEL / OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser without visible window",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Ollama models and exit",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local Ollama (localhost:11434) instead of ollama.com cloud API",
    )
    parser.add_argument(
        "--no-log-trajectories",
        action="store_true",
        help="Disable automatic JSONL trajectory logging",
    )
    parser.add_argument(
        "--trajectory-log",
        default=None,
        metavar="PATH",
        help="Override trajectory file (default: trajectories/latest.jsonl; old logs cleared each run)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="easy",
        help="easy=random inbox; medium=guaranteed matches+decoys; hard=subject rules+12 emails",
    )
    parser.add_argument(
        "--partial-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide sender until each email is opened (default: on). "
        "Use --no-partial-obs to show all senders.",
    )
    args = parser.parse_args()

    if args.list_models:
        from llm_agent import list_ollama_models, print_provider_hints

        list_ollama_models()
        print("\nNote: listing cloud models does NOT require a valid API key.")
        print("Chat (/api/chat) does — run with your key or use --local.")
        print("\nLocal Ollama (if ollama serve is running):")
        list_ollama_models(host="http://localhost:11434")
        print_provider_hints()
        return

    traj_path = None
    if not args.no_log_trajectories:
        traj_path = args.trajectory_log or _default_trajectory_path(
            args.agent, args.model
        )
        traj_path = _reset_trajectory_file(traj_path)
        print(f"Trajectory log: {traj_path}")

    render = None if args.headless else "human"
    diff = args.difficulty
    pobs = args.partial_obs
    if diff != "easy":
        print(f"Difficulty: {diff}")
    if pobs:
        print("Partial observability: senders hidden until opened (default)")
    else:
        print("Full observability: all senders visible")

    kw = dict(
        n_episodes=args.episodes,
        trajectory_path=traj_path,
        render_mode=render,
        difficulty=diff,
        partial_obs=pobs,
    )

    if args.agent in ("random", "all"):
        run_random(**kw)
    if args.agent in ("rule", "all"):
        run_rule_based(**kw)
    if args.agent == "ollama":
        run_llm(**kw, model=args.model, provider="ollama", use_local=args.local)
    if args.agent == "openai":
        run_llm(**kw, model=args.model, provider="openai")
    if args.agent == "anthropic":
        run_llm(**kw, model=args.model, provider="anthropic")
    if args.agent == "dom":
        run_dom_verify(**kw)
    if args.agent == "all":
        run_dom_verify(**{**kw, "n_episodes": min(2, args.episodes)})


if __name__ == "__main__":
    main()