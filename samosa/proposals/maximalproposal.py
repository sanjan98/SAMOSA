"""
Maximal coupling used for the resynchronization of two chains
"""

import numpy as np
from typing import Tuple
from samosa.core.proposal import ProposalProtocol
from samosa.core.state import ChainState
from samosa.utils.tools import sample_multivariate_gaussian, lognormpdf

class MaximalCoupling():
    """Try to force two chains to sample the same point using maximal coupling"""

    def __init__(self, proposal_coarse: ProposalProtocol, proposal_fine: ProposalProtocol):
        self.proposal_coarse = proposal_coarse
        self.proposal_fine = proposal_fine
    
    def sample(self, coarse_state: ChainState, fine_state: ChainState) -> ChainState:
        
        dim = coarse_state.position.shape[0]
        assert dim == fine_state.position.shape[0], "The dimensions of the two chains must be the same."

        x = self.proposal_fine.sample(fine_state).position
        px = np.exp(self.proposal_fine.proposal_logpdf(fine_state, ChainState(position=x))[0])
        qx = np.exp(self.proposal_coarse.proposal_logpdf(coarse_state, ChainState(position=x))[0])
        w = np.random.uniform(0, px)
        if w <= qx:
            Xstar = x
            Ystar = x
            return Xstar, Ystar
        else:
            wstar = -1000
            y = np.ones((dim,1))
            while(wstar < np.exp(self.proposal_fine.proposal_logpdf(fine_state, ChainState(position=y))[0])):
                y = self.proposal_coarse.sample(coarse_state).position
                wstar = np.random.uniform(0, np.exp(self.proposal_coarse.proposal_logpdf(coarse_state, ChainState(position=y))[0]))
            Xstar = x
            Ystar = y
        
        return Xstar, Ystar