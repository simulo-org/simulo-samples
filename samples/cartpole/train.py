"""Cartpole: train a pole-balancing policy with PPO.

This file holds the one job you submit. The application and the task it trains live
in ``task.py`` next to it.

What this file shows
--------------------
* ``@app.job(type="train")`` declares a training job. A training job saves its
  checkpoints on its own: every 50 iterations, and at the last iteration, it replaces
  the policy's ``latest`` checkpoint, and it replaces ``best`` whenever the mean
  episode reward improves. There is no callback to add and no file to write.
* ``retries=2`` reruns the job after a failure. A rerun picks up from the job's own
  latest checkpoint instead of starting over.
* The function's parameters are the job's command-line flags: ``num_envs`` is
  ``--num-envs`` and ``max_iterations`` is ``--max-iterations``.

Run it
------
Sign in once with ``simulo login``, then::

    simulo run samples/cartpole/train.py --num-envs 4096 --max-iterations 200

Use ``--num-envs 64 --max-iterations 2`` for a quick check that the job launches. To
keep training a policy you already have, start a new job from one of its checkpoints::

    simulo run samples/cartpole/train.py --max-iterations 400 --from <policy-id>:best
"""

from __future__ import annotations

from typing import Any

import simulo

from task import CartpoleTask, app


@app.job(
    type="train",
    # Tier 1: T4 GPU, 16 GB VRAM. Run `simulo systems` for the full four-tier catalog.
    system=simulo.SystemType.TIER_1,
    timeout=8 * 60 * 60,
    retries=2,
)
def train(num_envs: int = 4096, max_iterations: int = 200) -> dict[str, Any]:
    """Train a cartpole-balancing policy with PPO.

    Everything here runs in the Simulo cloud: ``simulo.LearningEnv`` builds ``num_envs``
    parallel copies of ``CartpoleTask`` on the GPU and ``simulo.RLTrainer`` trains a PPO
    policy against them. Simulo saves the policy's checkpoints automatically.

    Args:
        num_envs: Number of parallel environments to simulate.
        max_iterations: Number of PPO policy-update iterations.

    Returns:
        A JSON-serialisable dict of training statistics, such as ``iterations`` and
        ``best_reward``.
    """
    env = simulo.LearningEnv(
        task=CartpoleTask(),
        num_envs=num_envs,
        device="cuda",
        env_spacing=4.0,
        headless=True,
        seed=42,
    )
    trainer = simulo.RLTrainer(env=env, algorithm="PPO", device="cuda", seed=42)

    stats = trainer.train(max_iterations=max_iterations)

    # Close the trainer before the environment so the RL library releases its resources.
    trainer.close()
    env.close()

    return {"num_envs": num_envs, **stats}
