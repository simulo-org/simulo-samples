# Simulo samples

This repository collects runnable Simulo robotics projects for engineers learning how to
define cloud simulation and reinforcement learning jobs. Each sample is a small application
that you submit with the Simulo client.

## Prerequisites and compatibility

Install Python 3.11 or newer and the Simulo client. Every sample works with Simulo client
versions `>=0.23.1,<0.25`.

```bash
python -m pip install "simulo>=0.23.1,<0.25"
```

Samples run in the Simulo cloud. Sign in before submitting a job:

```bash
simulo login
```

## Get the samples

Clone this repository:

```bash
git clone https://github.com/simulo-org/simulo-samples.git
cd simulo-samples
```

## Quick start

Start with [Hello](samples/hello/): it is deterministic, requests no GPU, uses no catalog
assets, and has an exact expected result.

```bash
simulo run samples/hello/app.py --name robot --repeat 5
```

Each sample README explains its prerequisites, assets, expected results, and runtime.

## Sample index

<!-- BEGIN INDEX -->
No sample directories are available in this checkout.
<!-- END INDEX -->

## Assets

Most samples train against robots the Simulo catalog already carries, named with a
publisher: `simulo/robot/cartpole:v1`. A sample can also use a robot, world, or prop that
this repository ships and you publish to your own organization's catalog, named without
one: `robot/byo-urdf-arm:v1`. Those files live under [`assets/`](assets/), one directory
per asset, and that directory's path is the reference it publishes as.

## Update a clone

Pull the latest sample catalog and files from your existing clone:

```bash
git pull
```

## Support

Open an issue in this repository for bugs and sample requests. The documentation site at
[docs.simulo.ai](https://docs.simulo.ai) covers the client and cloud workflow. Pull requests
are not accepted yet.

## Licensing and attribution

Sample code is available under the [MIT License](LICENSE). Copyright (c) 2026 Simulo LLC.
