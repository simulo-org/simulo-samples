"""Preview the Cartpole task and its action mapping before training."""

import simulo

from task import CartpoleTask, app


@app.job(type="preview", system=simulo.SystemType.TIER_1, timeout=15 * 60)
def preview(num_envs: int = 16, checks: str = "all"):
    return simulo.preview(CartpoleTask(), num_envs=num_envs, checks=checks, video=True)
