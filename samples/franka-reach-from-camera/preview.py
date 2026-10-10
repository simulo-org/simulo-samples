# ruff: noqa: I001
"""Preview the Franka camera-reach scene and its action mapping before training."""

from task import FrankaReachFromCameraTask, app

import simulo


@app.job(type="preview", system=simulo.SystemType.TIER_1, timeout=15 * 60)
def preview(num_envs: int = 16, checks: str = "all", seed: int = 42):
    return simulo.preview(
        FrankaReachFromCameraTask(), num_envs=num_envs, checks=checks, video=True, seed=seed
    )
