from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


AlphaType = Literal["constant", "sample_average"]


@dataclass(frozen=True)
class BanditSpec:
    k: int = 10
    reward_std: float = 1.0
    mean_std: float = 1.0


@dataclass(frozen=True)
class AgentSpec:
    epsilon: float
    q_init: float
    alpha_type: AlphaType
    alpha_constant: float = 0.1


@dataclass(frozen=True)
class RunSpec:
    steps: int = 2000
    runs: int = 200
    seed: int = 0


def _validate_specs(bandit: BanditSpec, agent: AgentSpec, run: RunSpec) -> None:
    if bandit.k <= 0:
        raise ValueError("k must be > 0")
    if run.steps <= 0:
        raise ValueError("steps must be > 0")
    if run.runs <= 0:
        raise ValueError("runs must be > 0")
    if not (0.0 <= agent.epsilon <= 1.0):
        raise ValueError("epsilon must be in [0, 1]")
    if agent.alpha_type not in ("constant", "sample_average"):
        raise ValueError("alpha_type must be 'constant' or 'sample_average'")
    if agent.alpha_type == "constant" and not (0.0 < agent.alpha_constant <= 1.0):
        raise ValueError("alpha_constant must be in (0, 1] for constant alpha")
    if bandit.reward_std <= 0.0:
        raise ValueError("reward_std must be > 0")
    if bandit.mean_std <= 0.0:
        raise ValueError("mean_std must be > 0")


def simulate_eps_greedy(
    *,
    bandit: BanditSpec,
    agent: AgentSpec,
    run: RunSpec,
) -> dict[str, np.ndarray]:
    """Simulate a stationary K-armed bandit with an ε-greedy agent.

    Environment (per run):
      - true means mu[a] ~ Normal(0, 1) sampled once at start
      - reward R_t ~ Normal(mu[action_t], 1)

    Agent maintains action-value estimates Q[a] and updates with:
      Q <- Q + alpha * (R - Q)

    Returns per-timestep aggregates across independent runs:
      - avg_reward[t]
      - pct_optimal_action[t]
      - avg_regret[t] (expected instantaneous regret)
      - cumulative_regret[t]
    """

    _validate_specs(bandit, agent, run)

    rng = np.random.default_rng(run.seed)

    # True means per run and per arm (stationary throughout each run)
    true_means = rng.normal(loc=0.0, scale=bandit.mean_std, size=(run.runs, bandit.k))
    optimal_arm = np.argmax(true_means, axis=1)
    optimal_mean = true_means[np.arange(run.runs), optimal_arm]

    q = np.full((run.runs, bandit.k), fill_value=float(agent.q_init), dtype=np.float64)
    n = np.zeros((run.runs, bandit.k), dtype=np.int64)

    avg_reward = np.zeros(run.steps, dtype=np.float64)
    pct_opt = np.zeros(run.steps, dtype=np.float64)
    avg_regret = np.zeros(run.steps, dtype=np.float64)
    cum_regret = np.zeros(run.steps, dtype=np.float64)

    # tie-breaking noise to avoid deterministic argmax with equal Q
    tie_noise_scale = 1e-8

    cum = 0.0
    run_idx = np.arange(run.runs)

    for t in range(run.steps):
        explore = rng.random(run.runs) < agent.epsilon

        greedy_actions = np.argmax(q + tie_noise_scale * rng.standard_normal(q.shape), axis=1)
        random_actions = rng.integers(low=0, high=bandit.k, size=run.runs)
        actions = np.where(explore, random_actions, greedy_actions)

        chosen_means = true_means[run_idx, actions]
        rewards = rng.normal(loc=chosen_means, scale=bandit.reward_std, size=run.runs)

        # Expected regret: E[R* - R_a] = mu* - mu_a (noise cancels in expectation)
        inst_regret = optimal_mean - chosen_means
        cum += float(np.sum(inst_regret))

        # Update counts first so sample-average uses 1/N_new
        n[run_idx, actions] += 1

        if agent.alpha_type == "constant":
            alpha = agent.alpha_constant
        else:
            alpha = 1.0 / n[run_idx, actions]

        q_selected = q[run_idx, actions]
        q[run_idx, actions] = q_selected + alpha * (rewards - q_selected)

        avg_reward[t] = float(np.mean(rewards))
        pct_opt[t] = float(np.mean(actions == optimal_arm))
        avg_regret[t] = float(np.mean(inst_regret))
        cum_regret[t] = cum / float(run.runs)

    return {
        "avg_reward": avg_reward,
        "pct_optimal_action": pct_opt,
        "avg_regret": avg_regret,
        "cumulative_regret": cum_regret,
    }
