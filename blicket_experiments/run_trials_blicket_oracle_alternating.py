import datetime
import hashlib
import json
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

import hydra
import numpy as np
from omegaconf import DictConfig
from openai.types.completion_usage import CompletionUsage

import env.blicket_text as blicket_text
from agent.alternating_blicket_oracle_llm import AlternatingBlicketOracleAgent, strip_thinking

eLog = logging.getLogger(__name__)

ANN_OR_TEST_LINE = re.compile(r"(?mi)^\s*(Announce:.*|Test:.*)\s*$")

def assistant_one_line(text: str) -> str:
    t = strip_thinking(text or "") or ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""
    for ln in lines:
        if ln.startswith("Announce:") or ln.startswith("Test:"):
            return ln
    m = ANN_OR_TEST_LINE.search(t)
    if m:
        return m.group(1).strip()
    return lines[-1]

def strip_np(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, list):
        return [strip_np(y) for y in x]
    if isinstance(x, bool):
        return str(x).lower()
    return x

def deterministic_seed(base_seed: int, episode_id: int) -> int:
    seed_str = f"{base_seed}-{episode_id}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)

def _oracle_on_off(env: blicket_text.BlicketTextEnv, objects_on_machine: list[str]) -> bool:
    """
    Returns True if machine would be ON given exactly this set on the machine.
    """
    state = np.zeros((env.num_objects + 1,), dtype=bool)
    for name in objects_on_machine:
        i = int(name.split()[-1])
        state[i] = True
    # rule fn reads only indices in env.blicket_indices
    return bool(env._rule_fn(state, np.array(env.blicket_indices)))

def _feedback_block_on_off(is_on: bool) -> str:
    """
    Minimal feedback, no configuration text.
    Kept as a USER message injected on the next turn.
    """
    return ("ON" if is_on else "OFF")


def run_trial(CFG: DictConfig, CWD: str, trial_idx: int, base_seed: int):
    start_time = datetime.datetime.now()

    seed = deterministic_seed(base_seed, trial_idx)
    random.seed(seed)
    np.random.seed(seed)

    # Use env only for sampling blickets + rule, and initial configuration
    env = blicket_text.BlicketTextEnv(**CFG.env_kwargs, seed=seed)

    n_test = int(CFG.max_actions_per_trial)
    agent: AlternatingBlicketOracleAgent = hydra.utils.instantiate(CFG.agent, horizon_tests=n_test)

    total_turns = 2 * n_test + 1  # start Announce, end Announce

    eLog.info(
        f"Trial: {trial_idx}. num_objects={env.num_objects}. "
        f"Object names: {env.object_names}. Blicket indices: {env.blicket_indices}. Rule: {env.rule}"
    )

    game_state = env.reset()
    initial_configuration = game_state["feedback"]
    agent.init_episode(initial_configuration=initial_configuration)

    done = False
    turns = 0
    test_steps = 0
    total_reward = 0.0
    num_api_errors = 0
    last_announce = None

    # Injected only on the NEXT turn after a test
    obs_after_test: str | None = None

    while (not done) and (turns < total_turns):
        output_line, act_info = agent.act(
            num_objects=env.num_objects,
            obs_after_test=obs_after_test,  # None unless last turn was a TEST
        )
        turn_type = act_info.get("expect", "announce")

        # log current "game_state" = env.reset output only (static),
        # plus the oracle feedback is in obs_after_test.
        if CFG.save_trial_log:
            log_entry = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "agent_class": str(agent),
                "trial_idx": trial_idx,
                "turns": turns,
                "turn_type": turn_type,
                "test_steps": test_steps,
                "action": output_line,
                "game_state": {
                    "initial_configuration": initial_configuration,
                    "object_names": list(env.object_names),
                    "blicket_indices": [int(x) for x in env.blicket_indices],
                    "true_rule": str(env.rule),
                },
            }

            for k in act_info:
                if "usage" in k and isinstance(act_info[k], CompletionUsage):
                    log_entry[k] = {k1: v for k1, v in dict(act_info[k]).items() if isinstance(v, (int, float))}
                else:
                    log_entry[k] = strip_np(act_info[k])

            with open(os.path.join(CWD, f"action_log_trial-{trial_idx}.jsonl"), "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        if turn_type == "test":
            # Oracle evaluation from parsed list
            objs = act_info.get("parsed_test_objects") or []
            is_on = _oracle_on_off(env, objs)
            obs_after_test = _feedback_block_on_off(is_on)
            test_steps += 1
        else:
            last_announce = output_line
            obs_after_test = None

        if "api_error" in act_info:
            num_api_errors += int(bool(act_info["api_error"]))

        turns += 1
        agent.next_turn()

    trial_duration = (datetime.datetime.now() - start_time).total_seconds()

    trial_data = {
        "trial_idx": trial_idx,
        "seed": int(seed),
        "env_rule_setting": CFG.env_kwargs.get("rule", None),
        "object_names": list(env.object_names),
        "blicket_indices": [int(x) for x in env.blicket_indices],
        "num_test_steps": test_steps,
        "num_total_turns": turns,
        "total_reward": total_reward,
        "trial_duration": trial_duration,
        "num_traj_api_errors": num_api_errors,
        "cost_estimate": float(getattr(agent, "total_cost", 0.0)),
        "final_announce": last_announce,
    }

    with open(os.path.join(CWD, "results.jsonl"), "a") as f:
        f.write(json.dumps(trial_data) + "\n")

    # Conversation dump: show user messages + one-line assistant messages
    conv_path = os.path.join(CWD, f"conversation_trial-{trial_idx}.txt")
    with open(conv_path, "w", encoding="utf-8") as f:
        for m in getattr(agent, "messages", []):
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                f.write(content.rstrip() + "\n\n")
            elif role == "assistant":
                f.write(assistant_one_line(content).rstrip() + "\n\n")

    return trial_data


@hydra.main(version_base=None, config_path=".", config_name="run_trials_blicket_oracle_alternating")
def main(CFG: DictConfig):
    CWD = os.getcwd()
    eLog.info(f"Current working directory: {CWD}")

    start_time = datetime.datetime.now()

    if CFG.use_threadpool:
        executor = ThreadPoolExecutor(max_workers=CFG.tp_max_workers)

        def _run_trial(trial_idx):
            return run_trial(CFG, CWD, trial_idx, CFG.seed)

        results = list(executor.map(_run_trial, range(CFG.num_trials)))
        executor.shutdown()
    else:
        results = [run_trial(CFG, CWD, trial_idx, CFG.seed) for trial_idx in range(CFG.num_trials)]

    total_time = (datetime.datetime.now() - start_time).total_seconds()
    total_cost = sum(float(r.get("cost_estimate", 0.0) or 0.0) for r in results)
    eLog.info(f"Num trials: {CFG.num_trials}. Total time: {total_time:.2f}s. Estimated total cost: {total_cost}.")


if __name__ == "__main__":
    main()
