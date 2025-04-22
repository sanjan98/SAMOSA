"""
Class file for a single chain MCMC sampler.
"""

class MCMCsampler:
    """
    Class for a single chain MCMC sampler.
    
    Attributes:
        kernel (TransitionKernel): The transition kernel used for sampling.
        proposal (Proposal): The proposal distribution used for generating candidate states.
        model (BaseModel): The model used for computing posterior values.
        initial_state (ChainState): The initial state of the chain.
        n_iterations (int): Number of iterations to run the sampler.
    """
    
    def __init__(self, kernel, proposal, model, initial_state, n_iterations):
        self.kernel = kernel
        self.proposal = proposal
        self.model = model
        self.initial_state = initial_state
        self.n_iterations = n_iterations

    # STOPPED HERE
    # First figuring out kernel