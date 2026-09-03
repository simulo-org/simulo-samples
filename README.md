# Simulo samples

This repository collects runnable Simulo robotics projects for engineers learning how to
define cloud simulation and reinforcement learning jobs. Each sample is a small application
that you submit with the Simulo client.

## Prerequisites and compatibility

Install Python 3.11 or newer and the Simulo client. This repository targets Simulo client
versions `>=0.23.1,<0.24`.

```bash
python -m pip install "simulo>=0.23.1,<0.24"
```

Samples run in the Simulo cloud. Sign in before submitting a job:

```bash
simulo login
```

## Get the samples

Clone this repository to use the samples available today:

```bash
git clone https://github.com/simulo-org/simulo-samples.git
cd simulo-samples
```

A future Simulo client release will include `simulo install samples`. That command is not
available in Simulo 0.23.1, so use `git clone` until you have a client version that carries it.

## Quick start

Choose a published sample from the index, then run one of its listed jobs:

```bash
simulo run samples/<sample>/app.py --job <job>
```

Each sample README explains its prerequisites, assets, expected results, and estimated runtime.

## Sample index

<!-- BEGIN INDEX -->
No samples are published yet.
<!-- END INDEX -->

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

Sample code is available under the [MIT License](LICENSE). Catalog assets retain their own
terms, which each sample names in its README.
