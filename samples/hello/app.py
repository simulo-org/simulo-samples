"""Hello: the smallest Simulo application.

One job, no GPU, no simulation. ``hello`` prints a few greeting lines and returns a
small dictionary. Run it first: it proves that your sign-in, the upload, the job, its
log stream, and its result all work, and it finishes in seconds.

What this sample shows
----------------------
* ``simulo.App`` names the application; ``@app.job`` turns an ordinary typed Python
  function into a job the Simulo cloud can run.
* With a single job, the function's parameters become command-line flags: ``name``
  is ``--name`` and ``repeat`` is ``--repeat``.
* Anything the job prints becomes its log stream (``simulo logs``); the dictionary it
  returns becomes its result (``simulo result``).
* ``timeout=5 * 60`` bounds the job at five minutes of wall-clock time.

Run it
------
Sign in once with ``simulo login``, then::

    simulo run samples/hello/app.py --name robot --repeat 5

``simulo run`` uploads the application, creates a job, and follows its logs until it
finishes. Without a sign-in it only writes a package to a ``.simulo/`` directory next
to this file and runs nothing.
"""

from __future__ import annotations

import time
from typing import Any

import simulo

app = simulo.App("hello")


@app.job(timeout=5 * 60)
def hello(name: str = "world", repeat: int = 3) -> dict[str, Any]:
    """Print a few greeting lines (the job log) and return a small dictionary (the result)."""
    for index in range(repeat):
        print(f"[hello] {index + 1}/{repeat}: hello, {name}!", flush=True)
        time.sleep(0.3)  # long enough for `simulo logs --follow` to visibly stream
    return {"greeting": f"hello, {name}", "repeat": repeat}
