"""
Class file for the RealNVP transport map using "https://github.com/xqding/RealNVP"
This repo is used for density estimation that is exactly what I need
"""

# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from samosa.core.model import ModelProtocol
from samosa.core.map import TransportMap
from samosa.core.state import ChainState
from samosa.utils.post_processing import get_position_from_states
from typing import List, Optional, Tuple


# Need this to preserve the computation graph for the backward pass
def logpdf_multivariate_normal(x, mu, cov):
    # x: (batch_size, dim)
    mvn = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)
    return mvn.log_prob(x)


class RealNVPMap(TransportMap):
    """
    RealNVP-based transport map implemented in PyTorch.

    This class is compatible with :class:`TransportMapBase` and exposes
    `forward`, `inverse`, and `adapt` methods while reusing the common
    adaptation and density helpers from the base class.
    """

    def __init__(
        self,
        dim: int,
        masks: List[np.ndarray],
        hidden_dim: int = 32,
        learning_rate: float = 1e-3,
        num_epochs: int = 100,
        batch_size: int = 500,
        adapt_start: int = 500,
        adapt_end: int = 1000,
        adapt_interval: int = 100,
        reference_model: Optional[ModelProtocol] = None,
    ) -> None:
        """
        Initialize the RealNVP transport map.

        Args:
            dim: Dimension of the input space.
            masks: List of masks for the affine coupling layers.
            hidden_dim: Hidden dimension for the neural networks in the
                affine coupling layers.
            learning_rate: Learning rate for the optimizer.
            num_epochs: Number of epochs for full adaptation.
            batch_size: Batch size for training.
            adapt_start: Iteration to start adaptation.
            adapt_end: Iteration to end adaptation.
            adapt_interval: Adapt every ``adapt_interval`` iterations.
            reference_model: Optional reference model in the reference space.
        """

        super().__init__(
            dim=dim,
            adapt_start=adapt_start,
            adapt_end=adapt_end,
            adapt_interval=adapt_interval,
        )

        self.masks = masks
        self.hidden_dim = hidden_dim
        self.reference_model = reference_model
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

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
        # Convert to torch tensor and move to device (shape (N, dim))
        x_scaled_tensor = torch.tensor(xscaled.T, dtype=torch.float32).to(self.device)

        # Evaluate the RealNVP map
        r, logdet = self.realNVP.inverse(x_scaled_tensor)

        # Convert the result back to numpy and scale it back
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
        # Convert to torch tensor and move to device (shape (N, dim))
        r_tensor = torch.tensor(r.T, dtype=torch.float32).to(self.device)

        # Evaluate the RealNVP map
        xscaled, logdet = self.realNVP.forward(r_tensor)

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

    def _define_map(self) -> None:
        """
        Define the realNVP map using the Affine coupling layers and the Github implementation.
        """
        self.realNVP = RealNVP(self.masks, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.realNVP.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

    def _optimize_map_forceadapt(self) -> None:
        """
        Optimize the RealNVP map using the provided samples.
        Force adaptation - use all samples for training.
        """

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

        # Optimizer
        print_interval = max(1, self.num_epochs // 10)  # Print every 10% of epochs
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch in dataloader:
                x_batch = batch[0].to(self.device)
                # Forward pass through the RealNVP map (Dont use the forward method here as it breaks the computation graph)
                r, logdet = self.realNVP.inverse(x_batch)
                # Add the standardization log-determinant
                logdet_std = -torch.sum(
                    torch.log(torch.tensor(self.norm_std, dtype=torch.float32))
                ).to(self.device)
                logdet += logdet_std

                # Compute the logpdf of the reference
                if self.reference_model is None:
                    mu = torch.zeros(self.dim, device=self.device, dtype=x_batch.dtype)
                    cov = torch.eye(self.dim, device=self.device, dtype=x_batch.dtype)
                    log_rho = logpdf_multivariate_normal(r, mu, cov)
                else:
                    log_rho = self.reference_model(r)["log_posterior"]

                # Compute the loss
                loss = -torch.mean(log_rho + logdet)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            # Print training progress
            avg_loss = epoch_loss / num_batches
            if (epoch + 1) % print_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}], Avg Loss: {avg_loss:.5f}"
                )

        # Store the loss for analysis
        self.losses.append(avg_loss)

        return None  # Return None to indicate adaptation is complete

    def _optimize_map(self) -> None:
        """
        Optimize the RealNVP map using the provided samples.
        Step optimization - use the last adapt_interval samples for training.
        """

        # Standardize the input data
        # I am doing this as I cannot break the computation graph for the backward pass by using the forward and inverse methods
        # This is in contrast to the lower-traingular map where I am not using a backward pass
        self.x = self._scale_points(self.x, normalize=True)

        # Convert x to torch tensor and move to device
        x_tensor = torch.tensor(self.x.T, dtype=torch.float32).to(self.device)

        # One optimizer step on new data
        self.optimizer.zero_grad()
        r, logdet = self.realNVP.inverse(x_tensor)
        # Add the standardization log-determinant
        logdet_std = -torch.sum(
            torch.log(torch.tensor(self.norm_std, dtype=torch.float32))
        ).to(self.device)
        logdet += logdet_std

        # Compute the logpdf of the reference
        if self.reference_model is None:
            mu = torch.zeros(self.dim, device=self.device, dtype=x_tensor.dtype)
            cov = torch.eye(self.dim, device=self.device, dtype=x_tensor.dtype)
            log_rho = logpdf_multivariate_normal(r, mu, cov)
        else:
            log_rho = self.reference_model(r)["log_posterior"]

        # Compute the loss
        loss = -torch.mean(log_rho + logdet)

        loss.backward()
        self.optimizer.step()
        # Store the loss for analysis
        self.losses.append(loss.item())
        print("Adaptation step completed with loss:", loss.item())

        return None  # Return None to indicate adaptation is complete


# ----------------------------


class RealNVP(nn.Module):
    """
    A vanilla RealNVP class
    """

    def __init__(self, masks, hidden_dim):
        """
        initialized with a list of masks. each mask define an affine coupling layer
        """
        super(RealNVP, self).__init__()
        self.hidden_dim = hidden_dim
        self.masks = nn.ParameterList(
            [nn.Parameter(torch.Tensor(m), requires_grad=False) for m in masks]
        )

        self.affine_couplings = nn.ModuleList(
            [
                Affine_Coupling(self.masks[i], self.hidden_dim)
                for i in range(len(self.masks))
            ]
        )

    def forward(self, x):
        ## convert latent space variables into observed variables
        y = x
        logdet_tot = 0
        for i in range(len(self.affine_couplings)):
            y, logdet = self.affine_couplings[i](y)
            logdet_tot = logdet_tot + logdet

        return y, logdet_tot

    def inverse(self, y):
        ## convert observed variables into latent space variables
        x = y
        logdet_tot = 0

        ## inverse affine coupling layers
        for i in range(len(self.affine_couplings) - 1, -1, -1):
            x, logdet = self.affine_couplings[i].inverse(x)
            logdet_tot = logdet_tot + logdet

        return x, logdet_tot


class Affine_Coupling(nn.Module):
    def __init__(self, mask, hidden_dim):
        super(Affine_Coupling, self).__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim

        ## mask to seperate positions that do not change and positions that change.
        ## mask[i] = 1 means the ith position does not change.
        self.mask = nn.Parameter(mask, requires_grad=False)

        ## layers used to compute scale in affine transformation
        self.scale_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.scale_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_fc3 = nn.Linear(self.hidden_dim, self.input_dim)
        self.scale = nn.Parameter(torch.Tensor(self.input_dim))
        init.normal_(self.scale)

        ## layers used to compute translation in affine transformation
        self.translation_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.translation_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.translation_fc3 = nn.Linear(self.hidden_dim, self.input_dim)

    def _compute_scale(self, x):
        ## compute scaling factor using unchanged part of x with a neural network
        s = torch.relu(self.scale_fc1(x * self.mask))
        s = torch.relu(self.scale_fc2(s))
        s = torch.relu(self.scale_fc3(s)) * self.scale
        return s

    def _compute_translation(self, x):
        ## compute translation using unchanged part of x with a neural network
        t = torch.relu(self.translation_fc1(x * self.mask))
        t = torch.relu(self.translation_fc2(t))
        t = self.translation_fc3(t)
        return t

    def forward(self, x):
        ## convert latent space variable to observed variable
        s = self._compute_scale(x)
        t = self._compute_translation(x)

        y = self.mask * x + (1 - self.mask) * (x * torch.exp(s) + t)
        logdet = torch.sum((1 - self.mask) * s, -1)

        return y, logdet

    def inverse(self, y):
        ## convert observed varible to latent space variable
        s = self._compute_scale(y)
        t = self._compute_translation(y)

        x = self.mask * y + (1 - self.mask) * ((y - t) * torch.exp(-s))
        logdet = torch.sum((1 - self.mask) * (-s), -1)

        return x, logdet
