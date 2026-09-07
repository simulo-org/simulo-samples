# Hello

## What this shows

The smallest Simulo application: one job that prints a few greeting lines and returns a small
dictionary. It requests no GPU and runs no simulation, and it is the right first thing to run,
because a successful run checks your sign-in, the upload, the job, its log stream, and its result.
The output is deterministic, and a run takes about a minute end to end.

You will learn how an application is declared (`simulo.App` plus one `@app.job` function), how a
job's parameters become command-line flags, and where a job's printed output and return value go.

## Prerequisites

- Python 3.11 or newer and the Simulo client: `python -m pip install "simulo>=0.26.0,<0.27"`.
- A Simulo account, signed in once with `simulo login`. Without a sign-in, `simulo run` writes a
  package to a `.simulo/` directory next to `app.py` and runs nothing.
- A clone of this repository. The commands below run from its root.
- No local GPU is needed, and the job does not request a cloud GPU. That describes the
  job's own request only, not queue priority: it still waits on the same shared GPU
  fleet every sample in this repository runs on.

## Assets

None. The job uses no robot, world, or runtime beyond the default.

## Files and APIs

- `app.py`: the whole sample application.
- `.simuloignore`: files `simulo run` leaves out of the uploaded package.

Simulo names it uses:

- `simulo.App("hello")` names the application.
- `@app.job(timeout=5 * 60)` registers `hello` as a job with a five-minute wall-clock limit.
- `hello(name, repeat)`: an ordinary typed Python function. Its parameters are the job's flags.

## Run it

```bash
simulo login
simulo run samples/hello/app.py --name robot --repeat 5
```

`--name` and `--repeat` are the job's own parameters, mapped onto flags because the file declares
exactly one job. `simulo run samples/hello/app.py -h` prints them.

The job has a five-minute wall-clock limit. Pressing Ctrl-C while logs are being followed only
detaches your terminal; it does not stop the job. To stop a queued or running job, use the id
printed by `simulo run`:

```bash
simulo cancel <job-id>
```

## What to expect

`simulo run` uploads the application, creates a job, and follows its log until the job finishes.
The job emits five lines, sleeping 0.3 seconds between them; delivery to your terminal may be
batched:

```
[hello] 1/5: hello, robot!
[hello] 2/5: hello, robot!
[hello] 3/5: hello, robot!
[hello] 4/5: hello, robot!
[hello] 5/5: hello, robot!
```

The job then completes with this result:

```json
{"greeting": "hello, robot", "repeat": 5}
```

The output is fully determined by the flags; there is nothing random in this sample. A run takes
about a minute end to end, most of it queueing and startup rather than the job itself.

## Inspecting results

```bash
simulo jobs            # the job and its status
simulo logs            # the five greeting lines
simulo result          # the returned dictionary
simulo cancel <job-id> # stop the job if it is still queued or running
```

Each command defaults to your most recent job and accepts a job id, which `simulo run` prints.

## Troubleshooting

- `simulo run` prints "This wrote a local package only" and exits without running anything: you
  are not signed in. Run `simulo login`, then run the command again.
- The job stays `queued` in `simulo jobs` for a long time: the cloud is waiting for capacity.
  `simulo logs --follow` attaches as soon as it starts.
- `python samples/hello/app.py` prints the matching `simulo run` command and exits with status 2.
  That is deliberate: applications are submitted with `simulo run`, never run with `python`.
- `simulo result` reports that the job has not completed: it only has a result once the job
  finishes successfully. Use `simulo logs --follow` to wait for it.

## Extending it

- Add a second parameter to `hello` and it appears as a flag automatically.
- Add a second `@app.job` function. With more than one job, `simulo run` needs `--job NAME` to
  choose which one to submit; see [Cartpole Eval](../cartpole-eval/).
- Return a richer dictionary. Anything JSON-serialisable becomes the job's result.
- Move on to [Cartpole](../cartpole/) for the same submit-and-observe loop around a real training
  job.

## Assets, licensing, attribution

The sample code is available under the [MIT License](../../LICENSE). This sample references no
catalog assets and installs no third-party packages.

### Attribution

Copyright (c) 2026 Simulo LLC.
