"""Check the Humanoid task before training; this preview makes no policy."""

import simulo

from task import HumanoidTask, app


@app.job(type="preview", system=simulo.SystemType.TIER_1, timeout=15 * 60)
def preview(num_envs: int = 16, checks: str = "all"):
    return simulo.preview(HumanoidTask(), num_envs=num_envs, checks=checks, video=True)
