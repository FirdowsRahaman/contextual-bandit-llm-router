"""Online RL Contextual Bandit Routers (LinUCB & Thompson Sampling) for LLM Routing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelArm:
    """Candidate LLM endpoint with cost and latency characteristics."""

    model_id: str
    cost_per_1k_tokens: float
    avg_latency_ms: float
    sovereign: bool = True


DEFAULT_ARMS: tuple[ModelArm, ...] = (
    ModelArm("mistralai/Mistral-Small-24B", cost_per_1k_tokens=0.0002, avg_latency_ms=180.0, sovereign=True),
    ModelArm("mistralai/Mistral-Large-2407", cost_per_1k_tokens=0.0020, avg_latency_ms=420.0, sovereign=True),
    ModelArm("anthropic/claude-sonnet-4", cost_per_1k_tokens=0.0030, avg_latency_ms=520.0, sovereign=False),
)


def compute_composite_reward(
    quality_score: float,
    latency_ms: float,
    cost_usd: float,
    latency_weight: float = 0.15,
    cost_weight: float = 0.20,
) -> float:
    """Scalarized multi-objective RL reward in [0, 1]."""
    norm_latency = min(1.0, latency_ms / 2000.0)
    norm_cost = min(1.0, cost_usd / 0.01)
    raw = quality_score - (latency_weight * norm_latency) - (cost_weight * norm_cost)
    return float(np.clip(raw, 0.0, 1.0))


class LinUCBLLMRouter:
    """Disjoint LinUCB contextual bandit for prompt-aware LLM routing."""

    def __init__(self, arms: tuple[ModelArm, ...] = DEFAULT_ARMS, context_dim: int = 4, alpha: float = 0.5) -> None:
        self.arms = arms
        self.context_dim = context_dim
        self.alpha = alpha
        self._a: dict[str, np.ndarray] = {arm.model_id: np.eye(context_dim) for arm in arms}
        self._b: dict[str, np.ndarray] = {arm.model_id: np.zeros((context_dim, 1)) for arm in arms}

    def select_arm(self, context: np.ndarray, require_sovereign: bool = False) -> ModelArm:
        """Select the model arm maximizing the upper confidence bound."""
        x = context.reshape(self.context_dim, 1)
        candidates = [arm for arm in self.arms if not require_sovereign or arm.sovereign]
        if not candidates:
            raise ValueError("No eligible model arms satisfy sovereign constraint.")

        best_arm = candidates[0]
        best_ucb = -float("inf")
        for arm in candidates:
            a_inv = np.linalg.inv(self._a[arm.model_id])
            theta = a_inv @ self._b[arm.model_id]
            mean_est = float((theta.T @ x).item())
            uncertainty = float(self.alpha * np.sqrt((x.T @ a_inv @ x).item()))
            ucb = mean_est + uncertainty
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm
        return best_arm

    def update(self, model_id: str, context: np.ndarray, reward: float) -> None:
        """Update ridge-regression bandit parameters with observed reward."""
        x = context.reshape(self.context_dim, 1)
        self._a[model_id] = self._a[model_id] + (x @ x.T)
        self._b[model_id] = self._b[model_id] + (reward * x)


class ThompsonSamplingRouter:
    """Beta-Bernoulli Thompson Sampling router for binary quality SLA tracking."""

    def __init__(self, arms: tuple[ModelArm, ...] = DEFAULT_ARMS, seed: int = 42) -> None:
        self.arms = arms
        self._rng = np.random.default_rng(seed)
        self._alpha: dict[str, float] = {arm.model_id: 1.0 for arm in arms}
        self._beta: dict[str, float] = {arm.model_id: 1.0 for arm in arms}

    def select_arm(self) -> ModelArm:
        """Sample posterior success probability for each arm and pick highest."""
        samples = {
            arm.model_id: self._rng.beta(self._alpha[arm.model_id], self._beta[arm.model_id])
            for arm in self.arms
        }
        best_id = max(samples, key=samples.__getitem__)
        return next(arm for arm in self.arms if arm.model_id == best_id)

    def update(self, model_id: str, sla_met: bool) -> None:
        if sla_met:
            self._alpha[model_id] += 1.0
        else:
            self._beta[model_id] += 1.0
