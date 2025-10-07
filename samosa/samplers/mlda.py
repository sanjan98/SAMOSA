"""
The Multilevel Delayed Acceptance sampler.
Reference: Lykkegaard, M. B., T. J. Dodwell, C. Fox, G. Mingas, and R. Scheichl. “Multilevel Delayed Acceptance MCMC.” SIAM/ASA Journal on Uncertainty Quantification 11, no. 1 (2023): 1-30. https://doi.org/10.1137/22M1476770.
Also implemented in pyMC3 (https://www.pymc.io/projects/examples/en/2021.11.0/samplers/MLDA_simple_linear_regression.html).
"""

import numpy as np
import os

from samosa.core.state import ChainState
from samosa.core.kernel import KernelProtocol
from samosa.core.proposal import ProposalProtocol
from samosa.core.model import ModelProtocol

from samosa.samplers.single_chain import MCMCsampler
from samosa.utils.post_processing import load_samples

class MLDASampler:
    """
    Multilevel Delayed Acceptance Sampler (recursive, any L-level, with AEM support).
    """
    def __init__(self, models: list[ModelProtocol], kernel: KernelProtocol, proposal: ProposalProtocol, initial_position: np.ndarray, n_iterations: int, subchain_lengths: list[int], use_aem: bool = False, print_iteration: int = 1000):
        """
        Parameters:
            models: list of models for each level (coarsest to finest)
            kernel: kernel for the coarsest level
            proposal: proposal for the coarsest level
            initial_position: initial position for the finest level
            n_iterations: number of MLDA iterations to run (finest level)
            subsampling_lengths: list of subchain lengths for each level (length L-1)
            use_aem: whether to enable Adaptive Error Model bias correction
            print_iteration: print every this many iterations

        """
        dim = initial_position.shape[0]
        self.dim = dim
        L = len(models)
        self.L = L
        self.models = models
        self.kernel = kernel
        self.proposal = proposal
        
        self.initial_state = ChainState(position=initial_position, **self.models[-1](initial_position), metadata={
                'acceptance_probability': 0.0,
                'iteration': 1
                })
        self.start_iteration = 1

        self.n_iterations = n_iterations
        self.subchain_lengths = subchain_lengths
        self.use_aem = use_aem
        self.print_iteration = print_iteration

    def run(self, output_dir: str):
        """
        Run the MLDA sampler for the specified number of iterations.
        Parameters:
            output_dir: directory to save samples

        Returns:
            acceptance_rate: acceptance rate of the finest level
            Also saves the samples to files.
        """

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok = True)

        samples = [None for _ in range(self.L)]
        sizes = [None for _ in range(self.L)]

        sizes[-1] = self.n_iterations
        for l in range(self.L - 1):
            count = self.n_iterations
            for k in range(l, self.L - 1):
                count *= self.subchain_lengths[k]
            sizes[l] = count

        for l in range(self.L):
            samples[l] = np.zeros((self.dim, sizes[l]))

        samples[-1] = self._recursive_mlda(self.models, self.kernel, self.proposal, self.subchain_lengths, self.initial_state, self.L, self.n_iterations)

    def _recursive_mlda(self, models, kernel, proposal, subchain_lengths, initial_state, L, N):

        for j in range(N):

            if L == 1:
                sampler = MCMCsampler(kernel, proposal, initial_state.position, N, print_iteration=100000, save_iteration=10000000000) 
                _ = sampler.run(f'{self.output_dir}/temp')

                subchain = load_samples(f'{self.output_dir}/temp')

            else:

                subchain = self._recursive_mlda(models[:-1], kernel, proposal, subchain_lengths[:-1], initial_state, L-1, subchain_lengths[-1])

                current_state = initial_state
                proposed_position = subchain[-1].position

                model_result = models[-1](proposed_position)
                proposed_state = ChainState(position=proposed_position, **model_result, metadata=current_state.metadata.copy())

                


if __name__ == "__main__":

    from samosa.kernels.metropolis import MetropolisHastingsKernel
    from samosa.proposals.gaussianproposal import GaussianRandomWalk
    from samosa.utils.tools import lognormpdf

    def gaussian_model(x: np.ndarray, level: int) -> dict[str, any]:
        """
        Banana model function
        """
        output = {}
        # Just use the log_banana function to compute the log posterior
        log_posterior = lognormpdf(x, mean=np.array([[2**(-level+2)],[3**(-level+2)]]), cov=np.array([[2, 2**(-level)], [2**(-level), 1]]))

        output['log_posterior'] = log_posterior

        # If you want to compute the qoi, cost_model_output etc. you can do it like this
        cost = 1
        qoi = x[0]

        output['cost'] = cost
        output['qoi'] = qoi

        return output

    dim = 2

    L = 3  # Number of levels (0, 1, 2)
    
    models_list = [lambda x, level=i: gaussian_model(x, level) for i in range(L)]
    kernel = MetropolisHastingsKernel(model=models_list[0])
    proposal = GaussianRandomWalk(mu = np.zeros((dim,1)), sigma = np.eye(dim))
    initial_position = np.zeros((dim, 1))

    n_iterations = 5
    subchain_lengths = [2, 2]  # For levels 0 and 1

    mlda_sampler = MLDASampler(models=models_list, kernel=kernel, proposal=proposal, initial_position=initial_position, n_iterations=n_iterations, subchain_lengths=subchain_lengths, use_aem=False, print_iteration=1)
    mlda_sampler.run(output_dir='mlda_test_output')
    

    

        


        