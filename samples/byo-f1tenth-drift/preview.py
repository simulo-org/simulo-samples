"""Preview the F1TENTH drift task and action mapping before training."""

import simulo

from task import F1TenthDriftTask, app


@app.job(type="preview", system=simulo.SystemType.TIER_1, timeout=15 * 60)
def preview(num_envs: int = 16, checks: str = "all"):
    return simulo.preview(F1TenthDriftTask(), num_envs=num_envs, checks=checks, video=True)
