"""Train a drifting policy for a published F1TENTH-compatible race car."""

from __future__ import annotations

from typing import Any

from task import _train, app, simulo


@app.job(
    type="train",
    system=simulo.SystemType.TIER_1,
    timeout=8 * 60 * 60,
    retries=2,
)
def train(num_envs: int = 256, max_iterations: int = 500) -> dict[str, Any]:
    """Train the drift policy."""
    return _train(num_envs=num_envs, max_iterations=max_iterations)
