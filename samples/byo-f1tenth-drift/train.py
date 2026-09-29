"""Bring your own robot: train an F1TENTH-compatible race car to drift.

This file holds the one job you submit. The application, the drift task, and the
training body it calls, ``_train``, live in ``task.py`` next to it. Publish
``assets/robot/f1tenth/`` to your organization's catalog first; ``README.md`` has the
command.

Run it
------
Sign in with ``simulo login``, publish ``assets/robot/f1tenth/``, then::

    simulo run samples/byo-f1tenth-drift/train.py --num-envs 256 --max-iterations 500

``--num-envs`` and ``--max-iterations`` are ``train``'s own parameters. See
``README.md`` for what to expect from a run.
"""

from __future__ import annotations

from typing import Any

from task import _train, app, simulo


# retries=2 reruns the job after a failure. A training job saves its latest
# checkpoint automatically every 50 iterations, and a rerun picks up from it
# instead of starting over.
@app.job(
    type="train",
    # Tier 1: T4 GPU, 16 GB VRAM. Run `simulo systems` for the full four-tier catalog.
    system=simulo.SystemType.TIER_1,
    timeout=8 * 60 * 60,
    retries=2,
)
def train(num_envs: int = 256, max_iterations: int = 500) -> dict[str, Any]:
    """Train the drift policy. Simulo saves its checkpoints automatically."""
    return _train(num_envs=num_envs, max_iterations=max_iterations)
