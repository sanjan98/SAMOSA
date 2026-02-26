# SAMOSA

<table>
  <tr>
    <td align="center" width="250">
      <img src="samosa_logo.svg" width="150" alt="SAMOSA logo" />
    </td>
    <td>
      <strong>SAMOSA</strong> (Space-Aligned Multi-fidelity Open Sampling Architecture) is a Python package for MCMC sampling with multi-fidelity and coupled-chain support. Multi-fidelity MCMC (e.g. MLMC) features are included.
    </td>
  </tr>
</table>

[![CI](https://github.com/sanjan98/SAMOSA/actions/workflows/ci.yml/badge.svg)](https://github.com/sanjan-m/SAMOSA/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/sanjan-m/SAMOSA.git
cd SAMOSA
pip install -r requirements.txt
pip install -e .
```

Core dependencies (numpy, scipy) are listed in `requirements.txt` and in `setup.py`. For running tests, use `pip install -e ".[test]"` (installs pytest).

### Optional: Transport maps

To use **transport maps** (`LowerTriangularMap`, `Normalizingflow`, `RealNVP` in `samosa.maps`), you need **MParT**, **normflows**, and **PyTorch**. These packages have non-trivial build or platform requirements and are best installed separately according to their official documentation, not necessarily via a single `pip` command:

- **MParT** — [GitHub](https://github.com/MeasureTransport/MParT). Follow the project’s install instructions (e.g. `pip install MParT` or `conda install -c conda-forge mpart` where supported).
- **normflows** (and **PyTorch**) — [GitHub](https://github.com/VincentStimper/normalizing-flows). Install PyTorch for your platform first, then `pip install normflows` as described in the normflows docs.



## Quick start

```python
import numpy as np
from samosa import (
    GaussianRandomWalk,
    MetropolisHastingsKernel,
    SingleChainSampler,
)

def model(params):
    return {"log_posterior": float(-0.5 * np.sum(params**2))}

proposal = GaussianRandomWalk(mu=np.zeros((2, 1)), cov=0.1 * np.eye(2))
kernel = MetropolisHastingsKernel(model=model, proposal=proposal)
sampler = SingleChainSampler(kernel, initial_position=np.zeros((2, 1)), n_iterations=1000)
sampler.run("output")
```

## How sampling works

In MCMC we use a **Markov chain** to generate samples from a target distribution (e.g. a Bayesian posterior). The chain has a transition rule: from the current state we propose a new state and then accept or reject it so that, in the limit, the chain’s stationary distribution is the target. SAMOSA is built around a small set of pieces that you combine to build such a chain.

**Building a chain.** The main ingredients of any sampling algorithm are a **proposal** (how we suggest new states) and an **acceptance rule** (whether we keep or reject the proposal). Together they form a **kernel**: one step of the chain. The kernel also takes a **model** that defines the target (e.g. log-posterior). A **sampler** then runs the kernel repeatedly from an initial position to produce a sequence of samples.

**Proposals, adapters, and transport maps.** Proposals can be as simple as a Gaussian random walk. A **base proposal** can be wrapped with an **adapter** so that its parameters (e.g. covariance or scale) are updated during the run from the history of the chain. Alternatively (or in addition), a proposal can be wrapped with a **transport map**. Transport maps are bijection operators that transport measures. In simple words, they map points from a complex distribution to a reference distribution. In MCMC sampling, we map the current point from target space to reference space, propose in the reference space (e.g. standard Gaussian), then map the point into the target space. That often improves efficiency on difficult posteriors. So in SAMOSA you can use base proposals alone, or combine them with adapters and/or transport maps.

**Kernels and samplers.** SAMOSA provides **three kernels**: `MetropolisHastingsKernel` (standard accept/reject), `DelayedRejectionKernel` (optional second-stage proposal after a reject), and `CoupledKernel` (for two chains with a shared accept/reject). The **two main samplers** are `SingleChainSampler` (one chain) and `CoupledChainSampler` (two chains, used e.g. for multilevel or coupling). You plug a kernel into a sampler and run it for a given number of iterations.

## Examples

- **single_chain** — Single-chain sampling on a banana posterior with several strategies (base, adaptation, delayed rejection, transport maps).
- **coupled_chain** — Coupled sampling with Independent, Maximal, and Synce coupling (including transport maps).
- **mlmcmc** — Multilevel MCMC: build kernels per level, run sampling, and estimate MLMC statistics.

## License

See [LICENSE](LICENSE).
