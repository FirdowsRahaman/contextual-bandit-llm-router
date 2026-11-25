"""Tests for LinUCB and Thompson Sampling contextual bandit LLM routers."""

import numpy as np

from src.router.bandit_router import (
    DEFAULT_ARMS,
    LinUCBLLMRouter,
    ThompsonSamplingRouter,
    compute_composite_reward,
)


def test_linucb_learns_context_specific_routing():
    router = LinUCBLLMRouter(alpha=0.1)
    simple_ctx = np.array([1.0, 0.1, 0.0, 0.0])
    complex_ctx = np.array([0.1, 1.0, 1.0, 0.9])

    for _ in range(15):
        router.update("mistralai/Mistral-Small-24B", simple_ctx, compute_composite_reward(0.95, 180.0, 0.0002))
        router.update("mistralai/Mistral-Large-2407", simple_ctx, compute_composite_reward(0.90, 420.0, 0.0020))
        router.update("mistralai/Mistral-Small-24B", complex_ctx, compute_composite_reward(0.45, 180.0, 0.0002))
        router.update("anthropic/claude-sonnet-4", complex_ctx, compute_composite_reward(0.98, 520.0, 0.0030))

    assert router.select_arm(simple_ctx).model_id == "mistralai/Mistral-Small-24B"
    assert router.select_arm(complex_ctx).model_id == "anthropic/claude-sonnet-4"
    assert router.select_arm(complex_ctx, require_sovereign=True).sovereign is True


def test_thompson_sampling_converges_to_best_arm():
    router = ThompsonSamplingRouter(arms=DEFAULT_ARMS, seed=7)
    for _ in range(30):
        router.update("mistralai/Mistral-Large-2407", sla_met=True)
        router.update("mistralai/Mistral-Small-24B", sla_met=False)
    selected = router.select_arm()
    assert selected.model_id in {arm.model_id for arm in DEFAULT_ARMS}
