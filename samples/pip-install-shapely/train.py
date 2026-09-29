"""Install a PyPI dependency: train the Shapely target-zone JetBot policy.

This file holds the one job you submit. The application, its runtime with Shapely and
``DEMO_ZONE_CENTER_X`` layered on, and the task it trains live in ``task.py`` next to it.

Run it
------
Sign in once with ``simulo login``, then::

    simulo run samples/pip-install-shapely/train.py --num-envs 64 --max-iterations 300

Training starts once the runtime and catalog asset are prepared. A cache miss can
delay the first log line on any run.
"""

from __future__ import annotations

import os
from typing import Any

import simulo

from task import ShapelyZoneTask, app


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
def train(num_envs: int = 64, max_iterations: int = 300) -> dict[str, Any]:
    """Train the shapely target-zone JetBot policy with PPO.

    Simulo saves the policy's checkpoints automatically. The job reads
    ``DEMO_ZONE_CENTER_X`` from ``os.environ`` at job start and logs it. Simulo applies
    this app's ``Runtime.env({"DEMO_ZONE_CENTER_X": "2.5"})`` layer to the job's
    environment before the body runs, so the printed line is the job-log proof that
    ``env()`` reached the job, not just that submit recorded it. The value positions the
    target zone ``ShapelyZoneTask`` drives the robot into; see ``_target_zone_vertices``
    in ``task.py``.

    Args:
        num_envs: Number of parallel environments to simulate. Kept modest (64) by
            default: this app's point is showing that ``pip_install("shapely")`` works
            end to end in a real job, not maximising throughput.
        max_iterations: Number of PPO policy-update iterations.

    Returns:
        A JSON-serialisable dict of training statistics, such as ``iterations`` and
        ``best_reward``.
    """
    zone_center_x = float(os.environ.get("DEMO_ZONE_CENTER_X", "2.5"))
    print(f"[demo] target-zone center X from Runtime.env(): {zone_center_x}")

    env = simulo.LearningEnv(
        task=ShapelyZoneTask(zone_center_x=zone_center_x),
        num_envs=num_envs,
        device="cuda",
        # Wider than jetbot's 2.0: the target zone reaches out to 3.0 m ahead of
        # each env's spawn point (plus spawn jitter and manoeuvring room), so
        # the grid cell needs more clearance. See "Zone geometry" in task.py for
        # the zone's exact offset.
        env_spacing=8.0,
        headless=True,
        seed=42,
    )
    trainer = simulo.RLTrainer(env=env, algorithm="PPO", device="cuda", seed=42)

    stats = trainer.train(max_iterations=max_iterations)

    # Close the trainer before the environment so the RL library releases its resources.
    trainer.close()
    env.close()

    return {"num_envs": num_envs, **stats}
