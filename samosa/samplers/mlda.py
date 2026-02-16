"""
The Multilevel Delayed Acceptance sampler.
Reference: Lykkegaard, M. B., T. J. Dodwell, C. Fox, G. Mingas, and R. Scheichl. “Multilevel Delayed Acceptance MCMC.” SIAM/ASA Journal on Uncertainty Quantification 11, no. 1 (2023): 1-30. https://doi.org/10.1137/22M1476770.
Also implemented in pyMC3 (https://www.pymc.io/projects/examples/en/2021.11.0/samplers/MLDA_simple_linear_regression.html).
"""

import numpy as np
import os
import pickle

from typing import Optional

from samosa.core.state import ChainState
from samosa.core.kernel import KernelProtocol
from samosa.core.proposal import ProposalBase
from samosa.core.model import ModelProtocol

from samosa.samplers.single_chain import MCMCsampler
from samosa.utils.post_processing import load_samples
from samosa.core.mlmc import MLMCCalculator, MLMCPostProcessor


class MLDASampler:
    """
    Multilevel Delayed Acceptance Sampler (recursive, any L-level, with AEM support).
    """

    def __init__(
        self,
        models: list[ModelProtocol],
        kernel: KernelProtocol,
        proposal: ProposalBase,
        initial_position: np.ndarray,
        n_iterations: int,
        subchain_lengths: list[int],
        use_aem: bool = False,
        print_iteration: Optional[int] = None,
    ):
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
        L = len(models) - 1
        self.L = L
        self.models = models
        self.kernel = kernel
        self.proposal = proposal
        self.initial_position = initial_position
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
        os.makedirs(self.output_dir, exist_ok=True)

        samples = [[] for _ in range(self.L + 1)]

        initial_state = ChainState(
            position=self.initial_position,
            **self.models[-1](self.initial_position),
            metadata={"acceptance_probability": 0.0, "iteration": 1},
        )

        _ = self._recursive_mlda(
            self.models,
            self.kernel,
            self.proposal,
            self.subchain_lengths,
            initial_state,
            self.L,
            self.n_iterations,
            samples,
        )

        self._save_samples_levelwise(samples, self.subchain_lengths, output_dir)

    def _recursive_mlda(
        self, models, kernel, proposal, subchain_lengths, initial_state, L, N, samples
    ):

        current_state = initial_state

        for j in range(1, N + 1):
            if self.print_iteration is not None and j % self.print_iteration == 0:
                print(f"MLDA Level {L} - Iteration {j}/{N}")

            if L == 1:
                sampler = MCMCsampler(
                    kernel,
                    proposal,
                    current_state.position,
                    subchain_lengths[0],
                    print_iteration=self.print_iteration,
                )
                _ = sampler.run(f"{self.output_dir}/temp")

                subchain = load_samples(f"{self.output_dir}/temp")
                samples[0].extend(subchain)

            else:
                subchain = self._recursive_mlda(
                    models[:-1],
                    kernel,
                    proposal,
                    subchain_lengths[:-1],
                    current_state,
                    L - 1,
                    subchain_lengths[-1],
                    samples,
                )

            proposed_state = ChainState(
                position=subchain[-1].position,
                **models[-1](subchain[-1].position),
                metadata=current_state.metadata.copy(),
            )

            current_coarse_logpost = models[-2](current_state.position)["log_posterior"]
            proposed_coarse_logpost = subchain[-1].log_posterior

            ar = self._mlda_ar(
                current_state,
                proposed_state,
                current_coarse_logpost,
                proposed_coarse_logpost,
            )

            if ar == 1 or np.random.rand() < ar:
                current_state = proposed_state

            current_state.metadata["acceptance_probability"] = ar
            current_state.metadata["iteration"] = j

            samples[L].append(current_state)

        return samples[L]

    def _mlda_ar(
        self,
        current_state: ChainState,
        proposed_state: ChainState,
        current_coarse: np.ndarray,
        proposed_coarse: np.ndarray,
    ) -> float:
        """
        Compute the MLDA acceptance ratio.

        Parameters:
            current_state: current state of the chain
            proposed_state: proposed state of the chain
            current_coarse: log posterior of the coarse model at current position
            proposed_coarse: log posterior of the coarse model at proposed position

        Returns:
            ar: acceptance ratio
        """

        log_numerator = proposed_state.log_posterior + current_coarse
        log_denominator = current_state.log_posterior + proposed_coarse

        check = log_numerator - log_denominator

        if check > 0:
            ar = 1.0
        else:
            ar = np.exp(check)

        return ar

    def _save_samples_levelwise(self, samples, subchain_lengths, output_dir) -> None:
        """
        Save samples for each level to files.

        Parameters:
            samples: list of samples for each level
            subchain_lengths: list of subchain lengths for each level
            output_dir: directory to save the samples
        """

        for ell in range(self.L + 1):
            level_dir = os.path.join(output_dir, f"level_{ell}")
            os.makedirs(level_dir, exist_ok=True)

            if ell == 0:
                # Save all the samples for the coarsest level
                with open(f"{level_dir}/samples.pkl", "wb") as f:
                    pickle.dump(samples[ell], f)

            else:
                # Fine level samples are all the samples in the current level
                with open(f"{level_dir}/samples_fine.pkl", "wb") as f:
                    pickle.dump(samples[ell], f)

                # Coarse level samples are subsampled from the previous level
                coarse_samples = []
                n_fine = len(samples[ell])
                for i in range(n_fine):
                    end = (i + 1) * subchain_lengths[ell - 1]
                    coarse_samples.append(samples[ell - 1][end - 1])
                with open(f"{level_dir}/samples_coarse.pkl", "wb") as f:
                    pickle.dump(coarse_samples, f)


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
        log_posterior = lognormpdf(
            x,
            mean=np.array([[2 ** (-level + 2)], [3 ** (-level + 2)]]),
            cov=np.array([[2, 2 ** (-level)], [2 ** (-level), 1]]),
        )

        output["log_posterior"] = log_posterior

        # If you want to compute the qoi, cost_model_output etc. you can do it like this
        cost = 1
        qoi = x[0]

        output["cost"] = cost
        output["qoi"] = qoi

        return output

    dim = 2

    L = 3  # Number of levels (0, 1, 2)

    models_list = [lambda x, level=i: gaussian_model(x, level) for i in range(L)]
    kernel = MetropolisHastingsKernel(model=models_list[0])
    proposal = GaussianRandomWalk(mu=np.zeros((dim, 1)), sigma=np.eye(dim))
    initial_position = np.zeros((dim, 1))

    n_iterations = 1000
    subchain_lengths = [10, 5]  # For levels 0

    mlda_sampler = MLDASampler(
        models=models_list,
        kernel=kernel,
        proposal=proposal,
        initial_position=initial_position,
        n_iterations=n_iterations,
        subchain_lengths=subchain_lengths,
        use_aem=False,
        print_iteration=1000,
    )
    mlda_sampler.run(output_dir="mlda_test_output")

    mlmc_calc = MLMCCalculator(
        output_dir="mlda_test_output", num_levels=L, print_progress=True
    )
    mlmc_calc.compute_mlmc_estimator(burnin_fraction=0.3)
    mlmc_calc.print_mlmc_summary()

    mlmc_post = MLMCPostProcessor(output_dir="mlda_test_output", print_progress=True)
    mlmc_post.process_levels(levels=list(range(L)), burnin_fraction=0.3)
