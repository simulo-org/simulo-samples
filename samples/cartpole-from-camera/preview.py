# ruff: noqa: I001
"""Preview the cartpole scene and action mapping before training."""

from task import CartpoleFromCameraTask, app

import simulo


@app.job(type="preview", system=simulo.SystemType.TIER_1, timeout=15 * 60)
def preview(num_envs: int = 4, checks: str = "all", seed: int = 42):
    return simulo.preview(
        CartpoleFromCameraTask(), num_envs=num_envs, checks=checks, video=True, seed=seed
    )
