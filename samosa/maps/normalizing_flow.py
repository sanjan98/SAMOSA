"""
Class file for Normalizing flow transport maps using normflows python package "https://vincentstimper.com/normalizing-flows/"
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

import normflows as nf

from samosa.core.model import ModelProtocol
from samosa.core.map import TransportMap
from samosa.core.state import ChainState
from samosa.utils.post_processing import get_position_from_states
from typing import Any, List, Optional, Tuple


class Normalizingflow(TransportMap):
    """
    Normalizing-flow-based transport map using the ``normflows`` package.

    This class implements a transport map compatible with the
    :class:`TransportMapBase` interface: it provides `forward`, `inverse`,
    and `adapt` methods and uses the shared adaptation and density helpers
    from the base class.
    """

    def __init__(
        self,
        dim: int,
        flows: List[Any],
        learning_rate: float = 1e-3,
        num_epochs: int = 100,
        batch_size: int = 500,
        adapt_start: int = 500,
        adapt_end: int = 1000,
        adapt_interval: int = 100,
        reference_model: Optional[ModelProtocol] = None,
        force_cpu: bool = True,
    ) -> None:
        """
        Initialize the normalizing-flow transport map.

        Args:
            dim: Dimension of the input/parameter space.
            flows: List of flow modules as defined in the ``normflows`` package.
            learning_rate: Learning rate for the optimizer.
            num_epochs: Number of epochs for full adaptation.
            batch_size: Batch size for training.
            adapt_start: Iteration at which to start adaptation.
            adapt_end: Iteration at which to stop adaptation.
            adapt_interval: Adapt every ``adapt_interval`` iterations.
            reference_model: Optional reference distribution in the reference
                space. If ``None``, a standard Gaussian is used.
            force_cpu: If ``True``, force computation on CPU even if a GPU is
                available.
        """

        super().__init__(
            dim=dim,
            adapt_start=adapt_start,
            adapt_end=adapt_end,
            adapt_interval=adapt_interval,
        )

        self.flows = flows
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.reference_model = reference_model

        if self.reference_model is None:
            # If no reference model is provided, we use a standard Gaussian distribution
            q0 = nf.distributions.DiagGaussian(self.dim)
        else:
            q0 = reference_model

        self.q0 = q0

        if force_cpu is True:
            device = torch.device("cpu")
            print("Forcing CPU usage")
        else:
            # Check GPU availability and set device
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("Using GPU (CUDA backend)")
            elif torch.backends.mps.is_available():
                # For Apple Silicon Macs
                device = torch.device("mps")
                print("Using Apple GPU (MPS backend)")
            else:
                device = torch.device("cpu")
                print("Using CPU, no GPU support available for now")

        self.device = device

        # Add a loss attribute to track the loss during training
        self.losses = []

        self._define_map()

    def _define_map(self) -> None:
        """
        Define the map according to normflow's convention. Also add initialization of the optimizer and scheduler.
        """
        self.nfm = nf.NormalizingFlow(q0=self.q0, flows=self.flows).to(self.device)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward pass through the RealNVP map (Assuming target space -> reference space).
        Args:
            x (np.ndarray): Input data in the target space.
        Returns:
            Tuple[np.ndarray, float]: Transformed data in the reference space and log-determinant of the transformation.
        """
        # Scale the input data first using the base-class helper
        xscaled = self._scale_points(x, normalize=True)
        # Convert to torch tensor and move to device, shape (N, dim)
        x_scaled_tensor = torch.tensor(xscaled.T, dtype=torch.float32).to(self.device)

        # Evaluate the RealNVP map
        r, logdet = self.nfm.inverse_and_log_det(x_scaled_tensor)

        # Convert the result back to numpy
        r = r.detach().cpu().numpy()
        logdet = logdet.detach().cpu().numpy()

        # Back to shape (dim, N)
        r = r.T
        # Add the log-determinant of the scaling to the log-determinant of the transformation
        logdet += np.log(np.prod(1 / self.norm_std))

        return r, logdet

    def inverse(self, r: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Inverse pass through the RealNVP map (Assuming reference space -> target space).
        Args:
            r (np.ndarray): Input data in the reference space.
        Returns:
            Tuple[np.ndarray, float]: Transformed data in the target space and log-determinant of the transformation.
        """
        # Convert to torch tensor and move to device, shape (N, dim)
        r_tensor = torch.tensor(r.T, dtype=torch.float32).to(self.device)

        # Evaluate the RealNVP map
        xscaled, logdet = self.nfm.forward_and_log_det(r_tensor)

        # Convert the result back to numpy and scale it back
        xscaled = xscaled.detach().cpu().numpy()
        logdet = logdet.detach().cpu().numpy()

        # Back to shape (dim, N)
        xscaled = xscaled.T
        # Scale the output data back using the base-class helper
        x = self._scale_points(xscaled, normalize=False)
        # Add the log-determinant of the scaling to the log-determinant of the transformation
        logdet += np.log(np.prod(self.norm_std))

        return x, logdet

    def adapt(
        self,
        samples: List[ChainState],
        force_adapt: bool = False,
        paired_samples: Optional[List[ChainState]] = None,
    ) -> None:
        """
        Adapt the map to new samples.

        Args:
            samples: New samples to adapt the map to.
            force_adapt: If ``True``, bypass the adaptation window/interval checks.
            paired_samples: Optional paired samples (not used for this map).
        """
        del paired_samples  # Unused for this map.

        if not self._should_adapt(samples, force_adapt=force_adapt):
            return None

        iteration = self._extract_iteration(samples)
        self.iteration = iteration  # Useful for plotting loss curves later
        print(f"Adapting RealNVP map at iteration {iteration}")

        # Get positions from states
        positions = get_position_from_states(samples)

        # Fit or reset standardization depending on presence of reference_model
        if self.reference_model is None:
            self.norm_mean, self.norm_std = self._fit_standardization(positions)
        else:
            self.norm_mean = np.zeros((self.dim, 1))
            self.norm_std = np.ones((self.dim, 1))

        if force_adapt:
            # Use all positions for adaptation
            self.x = positions
            self._optimize_map_forceadapt()
        else:
            # Use the last adapt_interval samples for adaptation (in sample axis)
            if positions.shape[1] > self.adapt_interval:
                self.x = positions[:, -self.adapt_interval :]
            else:
                self.x = positions
            self._optimize_map()

    def _optimize_map_forceadapt(self) -> None:
        """
        Optimize the RealNVP map using the provided samples.
        Force adaptation - use all samples for training.
        """

        # Define the optimizer, new one every time we adapt
        optimizer = optim.Adam(self.nfm.parameters(), lr=self.learning_rate)
        scheduler = None
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.75)

        # Standardize the input data
        # I am doing this as I cannot break the computation graph for the backward pass by using the forward and inverse methods
        # This is in contrast to the lower-traingular map where I am not using a backward pass
        self.x = self._scale_points(self.x, normalize=True)

        # Convert x to torch tensor and move to device
        x_tensor = torch.tensor(self.x.T, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(x_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # Training loop with early stopping
        best_loss = float("inf")
        patience = 50
        patience_counter = 0

        # Optimizer
        print_interval = max(1, self.num_epochs // 10)  # Print every 10% of epochs
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch in dataloader:
                optimizer.zero_grad()

                x_batch = batch[0].to(self.device)

                # Add noise for regularization
                noise = torch.randn_like(x_batch) * 0.01
                x_batch_noisy = x_batch + noise

                loss = self.nfm.forward_kld(
                    x_batch_noisy
                )  # Remember that we are ignoring the logdet term of the affine standardization layer as it is not dependent on the parameters of the NF

                l2_reg = sum(p.pow(2.0).sum() for p in self.nfm.parameters())
                loss += 1e-6 * l2_reg

                # Backpropagation
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.nfm.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            # Print training progress
            avg_loss = epoch_loss / num_batches

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % print_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}], Avg Loss: {avg_loss:.5f} Learning rate: {optimizer.param_groups[0]['lr']:.5f}"
                )

                # Update learning rate
                if scheduler is not None:
                    scheduler.step()

        # Store the loss for analysis
        self.losses.append(best_loss)

        return None  # Return None to indicate adaptation is complete

    def _optimize_map(self) -> None:
        """
        Optimize the RealNVP map using the provided samples.
        Step optimization - use the last adapt_interval samples for training.
        """

        optimizer = optim.Adam(self.nfm.parameters(), lr=self.learning_rate)
        scheduler = None

        # Standardize the input data
        # I am doing this as I cannot break the computation graph for the backward pass by using the forward and inverse methods
        # This is in contrast to the lower-traingular map where I am not using a backward pass
        self.x = self._scale_points(self.x, normalize=True)

        # Convert x to torch tensor and move to device
        x_tensor = torch.tensor(self.x.T, dtype=torch.float32).to(self.device)

        # One optimizer step on new data
        optimizer.zero_grad()
        # Add noise for regularization
        noise = torch.randn_like(x_tensor) * 0.01
        x_batch_noisy = x_tensor + noise

        loss = self.nfm.forward_kld(
            x_batch_noisy
        )  # Remember that we are ignoring the logdet term of the affine standardization layer as it is not dependent on the parameters of the NF

        l2_reg = sum(p.pow(2.0).sum() for p in self.nfm.parameters())
        loss += 1e-6 * l2_reg

        loss.backward()
        optimizer.step()
        # Update the learning rate
        if scheduler is not None:
            scheduler.step()
        # Store the loss for analysis
        self.losses.append(loss.item())
        print("Adaptation step completed with loss:", loss.item())

        return None  # Return None to indicate adaptation is complete

    def plot_loss(self):
        """
        Return the fig and ax of the loss curve.
        This is useful for plotting the loss curve after training.
        """

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(self.losses, label="Training loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend()

        return fig, ax

    def checkpoint_model(self, filepath: str):
        """
        Save only the normalizing flow model for later plotting/analysis

        Args:
            filepath: Path to save the model (use .pth extension)
        """

        filepath = filepath + ".pth"

        model_data = {
            "model_state_dict": self.nfm.state_dict(),
            "mean": self.norm_mean,
            "std": self.norm_std,
            "dim": self.dim,
            "flows_config": self.flows,  # Optional: save flow architecture info
        }

        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load the normalizing flow model from a checkpoint file.

        Args:
            filepath: Path to the checkpoint file (use .pth extension)
        """

        filepath = filepath + ".pth"

        model_data = torch.load(filepath, map_location=self.device)

        # Load the model state
        self.nfm.load_state_dict(model_data["model_state_dict"])

        # Load mean and std
        self.norm_mean = model_data["mean"]
        self.norm_std = model_data["std"]
        self.dim = model_data["dim"]

        # Optionally load flow architecture info
        self.flows = model_data.get("flows_config", [])

        print(f"Model loaded from {filepath}")


# ------------------


def check_reference_model(ref_model: Any) -> bool:

    # Check if ref_model is an instance of torch.nn.Module
    if not isinstance(ref_model, nf.distributions.BaseDistribution):
        raise TypeError("Reference model must be a subclass of torch.nn.Module")

    # Check if it has a callable 'log_prob' method
    if not hasattr(ref_model, "log_prob") or not callable(
        getattr(ref_model, "log_prob")
    ):
        raise AttributeError("Reference model must have a callable 'log_prob' method")

    # Check if it has a callable 'forward' method
    if not hasattr(ref_model, "forward") or not callable(getattr(ref_model, "forward")):
        raise AttributeError("Reference model must have a callable 'forward' method")

    return True
