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
    TransportProposal,
    TransportProposalBase,
    CoupledKernelBase,
    CoupledKernel,
    create_mlmc_levels,
    MLMCSampler,
    MLMCCalculator,
    MLMCPostProcessor,
)
from samosa.maps import LinearOptimalTransportMap

try:
    from samosa.maps import LowerTriangularMap
except ImportError:
    LowerTriangularMap = None  # optional: requires MParT
try:
    from samosa.maps import Normalizingflow, RealNVPMap
except ImportError:
    Normalizingflow = None  # optional: requires torch
    RealNVPMap = None
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
    get_reference_position_from_states,
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
    "TransportProposal",
    "TransportProposalBase",
    "CoupledKernelBase",
    "CoupledKernel",
    "create_mlmc_levels",
    "MLMCSampler",
    "MLMCCalculator",
    "MLMCPostProcessor",
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
    "get_reference_position_from_states",
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
