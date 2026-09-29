"""Cartpole: preview the robot inside the task before you train.

This file holds a preview job for the task in ``task.py``. A preview checks the
cartpole in this task the way training will run it: it lets the robot settle, sweeps
each driven joint, runs random actions through the real training loop, and checks
observations, rewards, episodes and resets. It prints a report and saves a recording.
A preview makes no policy.

Run it
------
Sign in once with ``simulo login``, then::

    simulo run samples/cartpole/preview.py

Run only some checks with ``--checks settle,joint-sweep``.
"""

import simulo

from task import CartpoleTask, app


@app.job(type="preview", system=simulo.SystemType.TIER_1, timeout=15 * 60)
def preview(num_envs: int = 16, checks: str = "all"):
    return simulo.preview(CartpoleTask(), num_envs=num_envs, checks=checks, video=True)
