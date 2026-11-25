# Contextual Bandit LLM Router (Online RL)

<p align="left">
  <img src="https://img.shields.io/badge/Online%20RL-LinUCB%20%7C%20Thompson%20Sampling-10B981?style=flat-square" alt="Online RL">
  <img src="https://img.shields.io/badge/Routing-Cost%20%7C%20Latency%20%7C%20Quality-0EA5E9?style=flat-square" alt="Multi-Objective">
  <img src="https://img.shields.io/badge/Governance-Sovereign%20EU%20Constraints-8B5CF6?style=flat-square" alt="Sovereign">
  <img src="https://img.shields.io/badge/CI-Ruff%20%7C%20Bandit%20%7C%20Pytest-F59E0B?style=flat-square" alt="CI">
</p>

An **Online Reinforcement Learning (RL)** routing engine using **Disjoint LinUCB** and **Beta-Bernoulli Thompson Sampling** contextual bandits to dynamically route enterprise prompts across sovereign and frontier LLMs—jointly optimizing **output quality, latency, and token cost** under optional **EU sovereign residency constraints**.

---

## Table of Contents
- [Why Online RL for LLM Routing?](#why-online-rl-for-llm-routing)
- [Architecture](#architecture)
- [Routing Policies & Reward Function](#routing-policies--reward-function)
- [Quick Start](#quick-start)
- [Usage Example](#usage-example)
- [License](#license)

---

## Why Online RL for LLM Routing?

Sending every enterprise prompt to a frontier reasoning model wastes 70–85% of inference budget on routine classification or extraction tasks, while static if/else routers cannot adapt to model drift or latency spikes. **Contextual bandits** learn per-prompt-feature reward curves online with provable regret bounds.

---

## Architecture

```
Prompt Context Vector x ∈ ℝ⁴ (complexity, domain, length, compliance_tier)
                         │
                         ▼
           ┌───────────────────────────┐
           │   LinUCB / Thompson       │ ──▶ Applies `require_sovereign` filter
           │   Contextual Bandit       │ ──▶ Computes UCB = θᵀx + α√(xᵀA⁻¹x)
           └───────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   Mistral-Small   Mistral-Large   Claude-Sonnet
   (Sovereign EU)  (Sovereign EU)   (Frontier)
          │              │              │
          └──────────────┴──────────────┘
                         │
                         ▼
           Composite Reward Feedback Loop
        R = quality - w₁·latency - w₂·cost
```

---

## Routing Policies & Reward Function

- **`LinUCBLLMRouter`**: Maintains per-arm ridge-regression matrices $(A_a, b_a)$ and balances exploitation with UCB exploration ($\alpha$). Supports hard `require_sovereign=True` filtering for EU zero-egress workloads.
- **`ThompsonSamplingRouter`**: Beta-Bernoulli posterior sampling for binary SLA compliance tracking.
- **`compute_composite_reward`**: Scalarizes quality, normalized latency, and normalized cost into $[0, 1]$.

---

## Quick Start

```bash
git clone https://github.com/FirdowsRahaman/contextual-bandit-llm-router.git
cd contextual-bandit-llm-router
pip install -r requirements.txt
PYTHONPATH=. pytest tests -v
```

---

## Usage Example

```python
import numpy as np
from src.router.bandit_router import LinUCBLLMRouter, compute_composite_reward

router = LinUCBLLMRouter(alpha=0.25)
ctx = np.array([0.8, 0.2, 0.1, 0.0])
arm = router.select_arm(ctx, require_sovereign=True)
reward = compute_composite_reward(quality_score=0.94, latency_ms=190.0, cost_usd=0.0002)
router.update(arm.model_id, ctx, reward)
```

---

## License
This project is licensed under the MIT License.
