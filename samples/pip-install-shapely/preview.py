"""Check the Shapely target-zone task before training; this preview makes no policy."""

import os

import simulo

from task import ShapelyZoneTask, app


@app.job(type="preview", system=simulo.SystemType.TIER_1, timeout=15 * 60)
def preview(num_envs: int = 16, checks: str = "all"):
    zone_center_x = float(os.environ.get("DEMO_ZONE_CENTER_X", "2.5"))
    return simulo.preview(
        ShapelyZoneTask(zone_center_x=zone_center_x),
        num_envs=num_envs,
        checks=checks,
        video=True,
    )
