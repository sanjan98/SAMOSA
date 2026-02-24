"""
# Change this name
SAMOSA: Sampling and Multilevel Uncertainty Quantification.

Top-level re-exports for proposals and maps so users can import from the package:

  from samosa import GaussianRandomWalk, SynceCoupling, LowerTriangularMap
"""

from samosa.core import (
    AdaptiveProposal,
    Proposal,
    ProposalBase,
    TransportMap,
    TransportProposalBase,
    CoupledKernelBase,
    CoupledKernel,
)
from samosa.maps import (
    LinearOptimalTransportMap,
    LowerTriangularMap,
    Normalizingflow,
    RealNVPMap,
)
from samosa.proposals import (
    GaussianRandomWalk,
    GlobalAdapter,
    HaarioAdapter,
    IndependentCoupling,
    IndependentProposal,
    MaximalCoupling,
    PreCrankNicolson,
    SynceCoupling,
)
from samosa.utils import (
    load_samples,
    get_position_from_states,
    scatter_matrix,
    plot_trace,
    plot_lag,
    joint_plots,
    lognormpdf,
    sample_multivariate_gaussian,
    laplace_approx,
    log_banana,
    log_quartic,
)
from samosa.samplers import SingleChainSampler, CoupledChainSampler
from samosa.kernels import MetropolisHastingsKernel, DelayedRejectionKernel

__all__ = [
    # Core (proposals / maps)
    "AdaptiveProposal",
    "Proposal",
    "ProposalBase",
    "TransportMap",
    "TransportProposalBase",
    "CoupledKernelBase",
    "CoupledKernel",
    # Proposals
    "GaussianRandomWalk",
    "GlobalAdapter",
    "HaarioAdapter",
    "IndependentCoupling",
    "IndependentProposal",
    "MaximalCoupling",
    "PreCrankNicolson",
    "SynceCoupling",
    # Maps
    "LinearOptimalTransportMap",
    "LowerTriangularMap",
    "Normalizingflow",
    "RealNVPMap",
    # Samplers
    "SingleChainSampler",
    "CoupledChainSampler",
    # Kernels
    "MetropolisHastingsKernel",
    "DelayedRejectionKernel",
    # Utils
    "load_samples",
    "get_position_from_states",
    "scatter_matrix",
    "plot_trace",
    "plot_lag",
    "joint_plots",
    "lognormpdf",
    "sample_multivariate_gaussian",
    "laplace_approx",
    "log_banana",
    "log_quartic",
]
