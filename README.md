# SAMOSA

MCMC package — ML-MCMC functionalities (e.g. MLDA) will be added soon.

## Package hierarchy

The `samosa/` package is organized as follows:

- **core/** — State (`ChainState`), model protocol, and proposal base classes: `ProposalBase`, `AdaptiveProposal`, `TransportProposalBase`. Also the transport map interface and MLMC utilities.

- **kernels/** — Transition kernels: `MetropolisHastingsKernel`, `DelayedRejectionKernel`, plus SyncE and transport variants.

- **proposals/** — Base proposals (e.g. `GaussianRandomWalk`), adapters (`HaarioAdapter`, `GlobalAdapter`), and wiring via `AdaptiveProposal`.

- **samplers/** — `SingleChainSampler` (and coupled / MLDA samplers for multi-level and coupling).

- **maps/** — Transport maps (`LowerTriangularMap`, `RealNVPMap`, `LinearOptimalTransportMap`) used with `TransportProposalBase`.

- **utils/** — `tools` (e.g. `laplace_approx`, `log_banana`) and `post_processing` (`load_samples`, `get_position_from_states`, `scatter_matrix`, `plot_trace`, `plot_lag`).

## Sampling strategy progression

Examples are ordered by increasing complexity:

1. **Simple** — Fixed proposal (e.g. Gaussian random walk), optionally tuned with a Laplace approximation at the MAP.

2. **Adaptation** — Same base proposal wrapped with an adapter (e.g. Haario or Global) so covariance or scale is adapted during the run.

3. **Delayed rejection** — After a rejected first stage, propose again (e.g. with scaled covariance) to improve acceptance.

4. **Transport maps** — Propose in a reference space (e.g. N(0,I)) and map back to the target; the map is often pre-adapted using samples from a previous run.
