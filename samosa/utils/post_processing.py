"""
MCMC plotting and post-processing utilities.

Load samples, extract positions, and produce publication-quality diagnostic
plots (scatter matrix, trace, autocorrelation). Modified from Alex Gorodetsky
2020 script. License: MIT.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde

from samosa.core.state import ChainState

logger = logging.getLogger(__name__)

# Default image kwargs for publication-style plots (skill: post_processing)
_DEFAULT_IMG_KWARGS = {
    "label_fontsize": 24,
    "title_fontsize": 20,
    "tick_fontsize": 20,
    "legend_fontsize": 16,
    "img_format": "pdf",
}


def _default_img_kwargs() -> Dict[str, Any]:
    """Return a copy of default image kwargs for plots."""
    return _DEFAULT_IMG_KWARGS.copy()


def _setup_plot_style(
    img_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Set matplotlib/seaborn style and apply font sizes from img_kwargs.

    Parameters
    ----------
    img_kwargs : dict, optional
        Font sizes and format. If None, uses _DEFAULT_IMG_KWARGS.

    Returns
    -------
    dict
        The img_kwargs used (for downstream use).
    """
    if img_kwargs is None:
        img_kwargs = _default_img_kwargs()
    sns.set_style("white")
    sns.set_context("talk")
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams.update(
        {
            "axes.labelsize": img_kwargs.get(
                "label_fontsize", _DEFAULT_IMG_KWARGS["label_fontsize"]
            ),
            "axes.titlesize": img_kwargs.get(
                "title_fontsize", _DEFAULT_IMG_KWARGS["title_fontsize"]
            ),
            "xtick.labelsize": img_kwargs.get(
                "tick_fontsize", _DEFAULT_IMG_KWARGS["tick_fontsize"]
            ),
            "ytick.labelsize": img_kwargs.get(
                "tick_fontsize", _DEFAULT_IMG_KWARGS["tick_fontsize"]
            ),
            "legend.fontsize": img_kwargs.get(
                "legend_fontsize", _DEFAULT_IMG_KWARGS["legend_fontsize"]
            ),
        }
    )
    return img_kwargs


def _validate_samples(
    samples: Union[List[np.ndarray], np.ndarray],
    *,
    expected_ndim: int = 2,
    layout_hint: str = "(d, N) with d dimensions, N samples",
) -> List[np.ndarray]:
    """
    Validate and normalize sample inputs for plotting.

    Parameters
    ----------
    samples : list of numpy.ndarray or numpy.ndarray
        Single array or list of arrays.
    expected_ndim : int
        Expected number of dimensions per array (default 2).
    layout_hint : str
        Description for error messages.

    Returns
    -------
    list of numpy.ndarray
        List of 2D arrays in (d, N) format.

    Raises
    ------
    ValueError
        If samples are empty, not arrays, or have wrong shape.
    """
    if isinstance(samples, np.ndarray):
        samples = [samples]
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError(
            "Samples must be a non-empty list of arrays or a single array."
        )
    if not all(isinstance(s, np.ndarray) for s in samples):
        raise ValueError("All samples must be numpy arrays.")
    if not all(s.ndim == expected_ndim for s in samples):
        raise ValueError(f"All samples must be {expected_ndim}D arrays.")
    if not all(s.shape[0] <= s.shape[1] for s in samples):
        raise ValueError(f"Samples must be in {layout_hint}.")
    if not all(s.shape[0] == samples[0].shape[0] for s in samples):
        raise ValueError("All samples must have the same number of dimensions.")
    return samples


def load_samples(
    output_dir: str,
    iteration: Optional[int] = None,
) -> List[ChainState]:
    """
    Load MCMC samples from a pickle file.

    Looks for ``output_dir/samples.pkl`` when iteration is None, otherwise
    ``output_dir/samples/samples_{iteration}.pkl`` or
    ``output_dir/samples_{iteration}.pkl``.

    Parameters
    ----------
    output_dir : str
        Directory where samples are saved.
    iteration : int, optional
        Iteration number for a checkpoint. If None, loads final samples.

    Returns
    -------
    list of ChainState
        Loaded MCMC samples.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist.
    """
    if iteration is None:
        with open(f"{output_dir}/samples.pkl", "rb") as f:
            samples = pickle.load(f)
            return samples
    else:
        file_path = os.path.join(output_dir, "samples", f"samples_{iteration}.pkl")
        if not os.path.exists(file_path):
            file_path = f"{output_dir}/samples_{iteration}.pkl"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, "rb") as f:
            samples = pickle.load(f)
        return samples


def load_coupled_samples(
    output_dir: str,
    iteration: Optional[int] = None,
) -> Tuple[List[ChainState], List[ChainState]]:
    """
    Load coupled (coarse and fine) MCMC samples from pickle files.

    When iteration is None, loads ``samples_coarse.pkl`` and ``samples_fine.pkl``
    from output_dir. Otherwise tries ``output_dir/samples/samples_coarse_{iteration}.pkl``
    (and fine), then legacy ``output_dir/samples_coarse{iteration}.pkl``.

    Parameters
    ----------
    output_dir : str
        Directory where samples are saved.
    iteration : int, optional
        Iteration number for a checkpoint. If None, loads final samples.

    Returns
    -------
    samples_coarse : list of ChainState
        Coarse-chain samples.
    samples_fine : list of ChainState
        Fine-chain samples.

    Raises
    ------
    FileNotFoundError
        If the requested files do not exist.
    """
    if iteration is None:
        with open(f"{output_dir}/samples_coarse.pkl", "rb") as f:
            samples_coarse = pickle.load(f)
        with open(f"{output_dir}/samples_fine.pkl", "rb") as f:
            samples_fine = pickle.load(f)
            return samples_coarse, samples_fine
    else:
        # New layout: output_dir/samples/samples_coarse_{iteration}.pkl
        path_c = os.path.join(output_dir, "samples", f"samples_coarse_{iteration}.pkl")
        path_f = os.path.join(output_dir, "samples", f"samples_fine_{iteration}.pkl")
        if not os.path.exists(path_c):
            path_c = f"{output_dir}/samples_coarse{iteration}.pkl"
            path_f = f"{output_dir}/samples_fine{iteration}.pkl"
        if not os.path.exists(path_c):
            raise FileNotFoundError(f"The file {path_c} does not exist.")
        if not os.path.exists(path_f):
            raise FileNotFoundError(f"The file {path_f} does not exist.")
        with open(path_c, "rb") as f:
            samples_coarse = pickle.load(f)
        with open(path_f, "rb") as f:
            samples_fine = pickle.load(f)
        return samples_coarse, samples_fine


# Attributes that are arrays (shape (d, 1) per state → stacked as (d, N))
_EXTRACT_ARRAY_ATTRS = ("position", "reference_position")
# Attributes that are scalars (→ stacked as (1, N))
_EXTRACT_SCALAR_ATTRS = ("log_posterior", "log_prior", "log_likelihood", "cost")
# Use .posterior property for log_posterior so prior+likelihood is supported
_EXTRACT_POSTERIOR_PROP = "log_posterior"


def extract_from_states(
    samples: List[ChainState],
    attribute: Literal[
        "position",
        "reference_position",
        "log_posterior",
        "log_prior",
        "log_likelihood",
        "cost",
    ] = "position",
    burnin: float = 0.0,
) -> np.ndarray:
    """
    Extract a single attribute from a list of ChainState objects.

    One function for position, reference_position, log_posterior, log_prior,
    log_likelihood, and cost. Validates samples and applies burn-in once.
    Array attributes return shape (d, N); scalar attributes return (1, N).

    Parameters
    ----------
    samples : list of ChainState
        MCMC chain states.
    attribute : str, optional
        Attribute to extract: ``'position'``, ``'reference_position'``,
        ``'log_posterior'``, ``'log_prior'``, ``'log_likelihood'``, ``'cost'``.
        For ``'log_posterior'`` uses the ``.posterior`` property when needed.
        Default is ``'position'``.
    burnin : float, optional
        Fraction of samples to discard as burn-in (0 to 1). Default is 0.0.

    Returns
    -------
    numpy.ndarray
        For ``'position'`` or ``'reference_position'``: shape (d, N).
        For scalar attributes: shape (1, N).

    Raises
    ------
    ValueError
        If samples is empty, not ChainState, burnin is negative, or any
        state has None for the requested attribute (e.g. reference_position
        when not from a transport kernel).
    """
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("Samples must be a non-empty list of ChainState objects.")
    if not all(isinstance(s, ChainState) for s in samples):
        raise ValueError("All elements of samples must be ChainState objects.")
    if burnin < 0:
        raise ValueError("burnin must be non-negative.")

    if burnin > 0:
        n_burnin = int(len(samples) * burnin)
        samples = samples[n_burnin:]

    if attribute in _EXTRACT_ARRAY_ATTRS:
        values = []
        for s in samples:
            v = getattr(s, attribute)
            if v is None:
                raise ValueError(
                    f"State at index {len(values)} has {attribute}=None. "
                    "Use extract_from_states only for chains where this attribute is set."
                )
            values.append(np.ravel(v))
        return np.column_stack(values)

    if attribute in _EXTRACT_SCALAR_ATTRS:
        if attribute == _EXTRACT_POSTERIOR_PROP:
            values = [getattr(s, "posterior") for s in samples]
        else:
            values = []
            for s in samples:
                v = getattr(s, attribute)
                if v is None:
                    raise ValueError(
                        f"State at index {len(values)} has {attribute}=None."
                    )
                values.append(float(v))
        return np.reshape(np.array(values), (1, -1))

    raise ValueError(
        f"attribute must be one of {_EXTRACT_ARRAY_ATTRS + _EXTRACT_SCALAR_ATTRS}, got {attribute!r}."
    )


def get_position_from_states(
    samples: List[ChainState],
    burnin: float = 0.0,
) -> np.ndarray:
    """
    Extract positions from a list of ChainState objects.

    Convenience wrapper around ``extract_from_states(samples, 'position', burnin)``.

    Parameters
    ----------
    samples : list of ChainState
        MCMC chain states.
    burnin : float, optional
        Fraction of samples to discard as burn-in (0 to 1). Default is 0.0.

    Returns
    -------
    numpy.ndarray of shape (d, N)
        Positions with d dimensions and N (post-burn-in) samples.
    """
    return extract_from_states(samples, attribute="position", burnin=burnin)


def get_reference_position_from_states(
    samples: List[ChainState],
    burnin: float = 0.0,
) -> np.ndarray:
    """
    Extract reference positions from a list of ChainState objects.

    Convenience wrapper around ``extract_from_states(samples, 'reference_position', burnin)``.
    Use only for chains that have reference positions (e.g. transport-map kernels).

    Parameters
    ----------
    samples : list of ChainState
        MCMC chain states (e.g. from transport-map kernels).
    burnin : float, optional
        Fraction of samples to discard as burn-in (0 to 1). Default is 0.0.

    Returns
    -------
    numpy.ndarray of shape (d, N)
        Reference positions with d dimensions and N (post-burn-in) samples.
    """
    return extract_from_states(samples, attribute="reference_position", burnin=burnin)


def scatter_matrix(
    samples: List[np.ndarray],
    mins: Optional[np.ndarray] = None,
    maxs: Optional[np.ndarray] = None,
    upper_right: Optional[Any] = None,
    specials: Optional[Any] = None,
    hist_plot: Optional[bool] = True,
    nbins: Optional[int] = 100,
    img_kwargs: Optional[Dict[str, int]] = None,
    labels: Optional[List[str]] = None,
    sample_labels: Optional[List[str]] = None,
) -> Tuple[plt.Figure, List[plt.Axes], GridSpec]:
    """
    Create a nice scatter plot matrix of the samples.
    The marginals are on the diagonal and the joint distributions are on the off-diagonal.

    Parameters
    ----------
    samples : list of np.ndarray
        List of samples from different chains. Each sample is a 2D array with shape (n_dim, n_samples).
    mins : np.ndarray, optional
        Minimum values for each dimension. If None, they will be calculated from the samples.
    maxs : np.ndarray, optional
        Maximum values for each dimension. If None, they will be calculated from the samples.
    upper_right : dict, optional
        Dictionary containing the name and values for the upper right plot. Should contain 'name' and 'vals' keys.
    specials : list of dict, optional
        List of special points to plot. Each dict should contain 'vals' and optionally 'color' keys.
    hist_plot : bool, optional
        If True, use contour plots for the joint distributions. If False, use scatter plots.
    nbins : int, optional
        Number of bins for the histogram. Default is 100.
    img_kwargs : dict, optional
        Dictionary containing image parameters such as label_fontsize, title_fontsize, tick_fontsize, legend_fontsize, and img_format.
    labels : list of str, optional
        List of labels for each dimension. If None, they will be set to '\theta_1', '\theta_2', etc.
    sample_labels : list of str, optional
        List of labels for each sample chain. If None, they will be set to 'Chain 1', 'Chain 2', etc.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the scatter plot matrix.
    axs : list of matplotlib.axes.Axes
        List of axes objects for each subplot.
    gs : matplotlib.gridspec.GridSpec
        The GridSpec object used for the layout of the subplots.
    """

    samples = _validate_samples(samples)
    img_kwargs = _setup_plot_style(img_kwargs)
    nchains = len(samples)
    dim = samples[0].shape[0]

    if labels is None:
        labels = [rf"$\theta_{{{ii + 1}}}$" for ii in range(dim)]

    # Define a list of different colormaps to use for different samples
    cmap_list = [
        plt.cm.viridis,
        plt.cm.plasma,
        plt.cm.cividis,
        plt.cm.inferno,
        plt.cm.magma,
    ]

    # Get the default Seaborn color palette
    colors = sns.color_palette("tab10")

    if mins is None:
        mins = np.zeros((dim))
        maxs = np.zeros((dim))

        for ii in range(dim):
            mm = [np.quantile(samp[ii, :], 0.01, axis=0) for samp in samples]
            mins[ii] = np.min(mm)
            mm = [np.quantile(samp[ii, :], 0.99, axis=0) for samp in samples]
            maxs[ii] = np.max(mm)

            if specials is not None:
                if isinstance(specials, list):
                    minspec = np.min(
                        [
                            spec["vals"][ii]
                            for spec in specials
                            if spec["vals"][ii] is not None
                        ]
                    )
                    maxspec = np.max(
                        [
                            spec["vals"][ii]
                            for spec in specials
                            if spec["vals"][ii] is not None
                        ]
                    )
                else:
                    minspec = specials["vals"][ii]
                    maxspec = specials["vals"][ii]
                mins[ii] = min(mins[ii], minspec)
                maxs[ii] = max(maxs[ii], maxspec)

    deltas = (maxs - mins) / 10.0
    use_mins = mins - deltas
    use_maxs = maxs + deltas

    fig = plt.figure(figsize=(18, 10))
    if upper_right is None:
        gs = GridSpec(dim, dim, figure=fig)
        axs = [None] * dim * dim
        start = 0
        grid_size = dim
    else:
        gs = GridSpec(dim + 1, dim + 1, figure=fig)
        axs = [None] * (dim + 1) * (dim + 1)
        start = 1
        grid_size = dim + 1

    _ = np.array(
        [[np.mean(samples[kk][ii, :]) for ii in range(dim)] for kk in range(nchains)]
    )

    def one_decimal(x, pos):
        return f"{x:.1f}"

    formatter = FuncFormatter(one_decimal)

    for ii in range(dim):
        axs[ii] = fig.add_subplot(gs[ii + start, ii])
        ax = axs[ii]

        ax.grid(False)  # Disable gridlines on the diagonal plots

        if ii < dim - 1:
            ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False)
        else:
            ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True)
            if labels:
                ax.set_xlabel(labels[ii], fontsize=img_kwargs["label_fontsize"])

        ax.tick_params(axis="y", left=False, right=False, labelleft=False)
        ax.set_frame_on(False)

        # Use Gaussian KDE for the diagonal plots
        for kk in range(nchains):
            sampii = samples[kk][ii, :]
            kde = gaussian_kde(sampii)
            x_grid = np.linspace(use_mins[ii], use_maxs[ii], 1000)
            ax.fill_between(
                x_grid, 0, kde(x_grid), color=colors[kk % len(colors)], alpha=0.3
            )
            ax.plot(x_grid, kde(x_grid), color=colors[kk % len(colors)], alpha=0.7)
            # Plot vertical line for mean of this chain and dimension
            # ax.axvline(means[kk, ii], color=colors[kk % len(colors)], linestyle='--', lw=2)

        ax.set_xlim((use_mins[ii], use_maxs[ii]))
        ax.set_ylim(0)  # Start the y-axis at zero for alignment with lower plots

        if specials is not None:
            for special in specials:
                if special["vals"][ii] is not None:
                    if "color" in special:
                        ax.axvline(special["vals"][ii], color=special["color"], lw=2)
                    else:
                        ax.axvline(special["vals"][ii], lw=2)

        # ax.axvline(means[ii], color='red', linestyle='--', lw=2, label=f'Mean: {means[ii]:.2f}')
        ax.set_xlim((use_mins[ii] - 1e-10, use_maxs[ii] + 1e-10))

        diff = 0.2 * (use_maxs[ii] - use_mins[ii])
        if (use_maxs[ii] - use_mins[ii]) < 0.25:  # Adjust threshold as needed
            xticks = [np.mean([use_mins[ii], use_maxs[ii]])]
        else:
            xticks = np.linspace(use_mins[ii] + diff, use_maxs[ii] - diff, 2)
        yticks = ax.get_yticks()
        if len(yticks) >= 2:
            if (yticks[1] - yticks[0]) < 0.25:
                yticks = [yticks[0]]
            else:
                yticks = np.linspace(yticks[0], yticks[-1], 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis="both", labelsize=img_kwargs["tick_fontsize"])

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        for jj in range(ii + 1, dim):
            axs[jj * grid_size + ii] = fig.add_subplot(gs[jj + start, ii])
            ax = axs[jj * grid_size + ii]

            ax.grid(False)

            if jj < dim - 1:
                ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False)
            else:
                ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True)
                if labels:
                    ax.set_xlabel(labels[ii], fontsize=img_kwargs["label_fontsize"])
            if ii > 0:
                ax.tick_params(axis="y", left=False, right=False, labelleft=False)
            else:
                ax.tick_params(axis="y", left=True, right=False, labelleft=True)
                if labels:
                    ax.set_ylabel(labels[jj], fontsize=img_kwargs["label_fontsize"])

            ax.set_frame_on(True)

            for kk in range(nchains):
                cmap = cmap_list[
                    kk % len(cmap_list)
                ]  # Choose a colormap for each sample set
                data = np.vstack([samples[kk][ii, :], samples[kk][jj, :]])
                kde = gaussian_kde(data)
                x_grid = np.linspace(use_mins[ii], use_maxs[ii], nbins)
                y_grid = np.linspace(use_mins[jj], use_maxs[jj], nbins)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

                if hist_plot:
                    max_z = Z.max()
                    min_z = Z.min()
                    all_levels = np.linspace(min_z, max_z, 6)

                    # Skip the first (lowest density) level to remove the outermost contour
                    levels = all_levels[
                        1:
                    ]  # This gives you 5 levels, excluding the lowest
                    ax.contour(X, Y, Z, levels=levels, cmap=cmap, linewidths=1.0)
                    ax.plot(
                        samples[kk][ii, :],
                        samples[kk][jj, :],
                        "o",
                        ms=1,
                        alpha=0.01,
                        color=colors[kk % len(colors)],
                        label=sample_labels[kk]
                        if sample_labels is not None
                        else f"Chain {kk + 1}",
                        rasterized=True,
                    )
                else:
                    ax.plot(
                        samples[kk][ii, :],
                        samples[kk][jj, :],
                        "o",
                        ms=1,
                        alpha=0.2,
                        color=colors[kk % len(colors)],
                        label=sample_labels[kk]
                        if sample_labels is not None
                        else f"Chain {kk + 1}",
                        rasterized=True,
                    )
            if specials is not None:
                for special in specials:
                    if "color" in special:
                        ax.plot(
                            special["vals"][ii],
                            special["vals"][jj],
                            "x",
                            color=special["color"],
                            ms=2,
                            mew=2,
                            rasterized=True,
                        )
                    else:
                        ax.plot(
                            special["vals"][ii],
                            special["vals"][jj],
                            "x",
                            ms=2,
                            mew=2,
                            rasterized=True,
                        )

            ax.set_xlim((use_mins[ii], use_maxs[ii]))
            ax.set_ylim((use_mins[jj] - 1e-10, use_maxs[jj] + 1e-10))

            diff = 0.2 * (use_maxs[ii] - use_mins[ii])
            if (use_maxs[ii] - use_mins[ii]) < 0.25:  # Adjust threshold as needed
                xticks = [np.mean([use_mins[ii], use_maxs[ii]])]
            else:
                xticks = np.linspace(use_mins[ii] + diff, use_maxs[ii] - diff, 2)
            if (use_maxs[jj] - use_mins[jj]) < 0.25:  # Adjust threshold as needed
                yticks = [np.mean([use_mins[jj], use_maxs[jj]])]
            else:
                yticks = np.linspace(use_mins[jj] + diff, use_maxs[jj] - diff, 2)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.tick_params(axis="both", labelsize=img_kwargs["tick_fontsize"])

            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout(pad=0.01)

    if upper_right is not None:
        size_ur = int(dim / 2)

        name = upper_right["name"]
        vals = upper_right["vals"]
        if "log_transform" in upper_right:
            log_transform = upper_right["log_transform"]
        else:
            log_transform = None
        ax = fig.add_subplot(
            gs[0 : int(dim / 2), size_ur + 1 : size_ur + int(dim / 2) + 1]
        )

        lb = np.min([np.quantile(val, 0.01) for val in vals])
        ub = np.max([np.quantile(val, 0.99) for val in vals])
        for kk in range(nchains):
            if log_transform is not None:
                pv = np.log10(vals[kk])
                ra = (np.log10(lb), np.log10(ub))
            else:
                pv = vals[kk]
                ra = (lb, ub)
            ax.hist(
                pv,
                density=True,
                range=ra,
                edgecolor="black",
                stacked=True,
                bins="auto",
                alpha=0.5,
            )  # Adjust transparency
        ax.tick_params(axis="x", bottom="both", top=False, labelbottom=True)
        ax.tick_params(axis="y", left="both", right=False, labelleft=False)
        ax.set_frame_on(True)
        ax.set_xlabel(name, fontsize=img_kwargs["label_fontsize"])

        diff = 0.2 * (ra[1] - ra[0])
        xticks = np.linspace(ra[0] + diff, ra[1] - diff, 2)
        yticks = ax.get_yticks()
        if len(yticks) >= 2:
            yticks = np.linspace(yticks[0], yticks[-1], 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis="both", labelsize=img_kwargs["tick_fontsize"])

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    if sample_labels is not None:
        # Place the legend in the upper right whitespace
        # [left, bottom, width, height] in figure coordinates; adjust as needed
        left = 0.6
        bottom = 0.7
        width = 0.20
        nlabels = len(sample_labels)
        # Dynamically set the height based on number of labels
        height = max(
            0.08, min(0.04 * nlabels, 0.25)
        )  # min and max for reasonable bounds
        legend_ax = fig.add_axes([left, bottom, width, height])
        legend_ax.axis("off")
        handles = []
        for kk in range(nchains):
            color = colors[kk % len(colors)]
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=sample_labels[kk],
                    markerfacecolor=color,
                    markersize=10,
                )
            )
        legend_ax.legend(
            handles=handles,
            loc="center",
            frameon=False,
            fontsize=img_kwargs["legend_fontsize"],
        )

    plt.subplots_adjust(left=0.15, right=0.95)

    return fig, axs, gs


def joint_plots(
    samples: List[np.ndarray],
    img_kwargs: Optional[Dict[str, int]] = None,
    labels: Optional[List[str]] = None,
    bins: int = 30,
) -> List[plt.Figure]:
    """
    Create joint plots for pairs of dimensions (e.g., consecutive samples or different chains).

    Parameters
    ----------
    samples : list of np.ndarray
        List of samples from different chains or levels. Each sample is a 2D array with shape (n_dim, n_samples).
    img_kwargs : dict, optional
        Dictionary containing image parameters.
    labels : list of str, optional
        List of labels for each dimension.
    bins : int, optional
        Number of bins for marginal histograms.

    Returns
    -------
    figures : list of matplotlib.figure.Figurexs
        List of figure objects for each joint plot.
    """

    if len(samples) != 2:
        raise ValueError("Need 2 samples for pairwise plotting.")
    samples = _validate_samples(samples)
    img_kwargs = _setup_plot_style(img_kwargs)
    dim, nsamples = samples[0].shape

    if labels is None:
        labels = [rf"$\theta_{{{ii + 1}}}$" for ii in range(dim)]

    figures = []

    # Create joint plots for each dimension, comparing consecutive samples
    for dd in range(dim):
        x = samples[0][dd, :]
        y = samples[1][dd, :]

        # Create joint plot using seaborn
        g = sns.jointplot(
            x=x,
            y=y,
            kind="scatter",
            marginal_kws=dict(bins=bins, fill=True),
            alpha=0.6,
            s=30,
            linewidth=0,
            joint_kws={"rasterized": True},
        )
        g.figure.set_size_inches((8, 8))

        # Set labels
        g.set_axis_labels(f"{labels[dd]} - coarse", f"{labels[dd]} - fine")

        # Adjust tick parameters
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        ticks_x = np.linspace(x_min, x_max, 4)
        ticks_y = np.linspace(y_min, y_max, 4)
        g.ax_joint.set_xticks([round(tick, 2) for tick in ticks_x])
        g.ax_joint.set_yticks([round(tick, 2) for tick in ticks_y])

        g.ax_joint.grid(False)
        g.ax_joint.spines["top"].set_color("black")
        g.ax_joint.spines["top"].set_linewidth(2)
        g.ax_joint.spines["right"].set_color("black")
        g.ax_joint.spines["right"].set_linewidth(2)
        g.ax_joint.spines["bottom"].set_color("black")
        g.ax_joint.spines["bottom"].set_linewidth(2)
        g.ax_joint.spines["left"].set_color("black")
        g.ax_joint.spines["left"].set_linewidth(2)

        if g.ax_joint.get_legend() is not None:
            for label in (
                g.ax_joint.get_xticklabels()
                + g.ax_joint.get_yticklabels()
                + g.ax_joint.get_legend().get_texts()
                + [g.ax_joint.xaxis.label, g.ax_joint.yaxis.label]
            ):
                label.set_color("black")
        else:
            for label in (
                g.ax_joint.get_xticklabels()
                + g.ax_joint.get_yticklabels()
                + [g.ax_joint.xaxis.label, g.ax_joint.yaxis.label]
            ):
                label.set_color("black")

        figures.append(g.figure)

    return figures


def plot_trace(
    samples: Union[np.ndarray, List[np.ndarray]],
    img_kwargs: Optional[Dict] = None,
    labels: Optional[List] = None,
    sample_labels: Optional[List[str]] = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot the trace of the samples.
    Parameters
    ----------
    samples : np.ndarray or list of np.ndarray
        The samples to plot. Should be of shape (n_dim, n_samples).
    img_kwargs : dict, optional
        Dictionary containing image parameters such as label_fontsize, title_fontsize, tick_fontsize, legend_fontsize, and img_format.
    labels : list of str, optional
        List of labels for each dimension. If None, they will be set to '\theta_1', '\theta_2', etc.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the trace plot.
    axs : matplotlib.axes.Axes
        The axes object for the trace plot.
    """

    # If samples is a single numpy array, convert it to a list
    if isinstance(samples, np.ndarray):
        samples = [samples]

    # Check if the samples are in the correct format
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError(
            "Samples should be a numpy array or non empty list of numoy arrays."
        )

    samples = _validate_samples(samples)
    img_kwargs = _setup_plot_style(img_kwargs)
    dim = samples[0].shape[0]

    if labels is None:
        labels = [rf"$\theta_{{{ii + 1}}}$" for ii in range(dim)]

    if sample_labels is None and len(samples) > 1:
        sample_labels = [f"Chain {i + 1}" for i in range(len(samples))]
    elif sample_labels is None:
        sample_labels = [None]

    # Get default color palette
    colors = sns.color_palette("tab10")

    fig, axs = plt.subplots(dim, 1, figsize=(16, 8), sharex=True)

    if dim == 1:
        axs = [axs]  # Ensure axs is a list when dim=1

    for i in range(dim):
        for j, samp in enumerate(samples):
            label = sample_labels[j] if j < len(sample_labels) else None
            axs[i].plot(
                samp[i, :],
                alpha=0.6,
                color=colors[j % len(colors)],
                linewidth=0.8,
                label=label,
            )

        axs[i].set_ylabel(f"{labels[i]}")

        if i == 0 and len(samples) > 1:
            axs[i].legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

    axs[dim - 1].set_xlabel("Sample Number")

    for i in range(dim):
        axs[i].grid(False)
        axs[i].spines["top"].set_color("black")
        axs[i].spines["top"].set_linewidth(2)
        axs[i].spines["right"].set_color("black")
        axs[i].spines["right"].set_linewidth(2)
        axs[i].spines["bottom"].set_color("black")
        axs[i].spines["bottom"].set_linewidth(2)
        axs[i].spines["left"].set_color("black")
        axs[i].spines["left"].set_linewidth(2)
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if axs[i].get_legend() is not None:
            for label in (
                axs[i].get_xticklabels()
                + axs[i].get_yticklabels()
                + axs[i].get_legend().get_texts()
                + [axs[i].xaxis.label, axs[i].yaxis.label]
            ):
                label.set_color("black")
        else:
            for label in (
                axs[i].get_xticklabels()
                + axs[i].get_yticklabels()
                + [axs[i].xaxis.label, axs[i].yaxis.label]
            ):
                label.set_color("black")
        plt.tight_layout()

    return fig, axs


def plot_lag(
    samples: Union[np.ndarray, List[np.ndarray]],
    maxlag: Optional[int] = 500,
    step: Optional[int] = 1,
    img_kwargs: Optional[Dict[str, int]] = None,
    labels: Optional[List[str]] = None,
    sample_labels: Optional[List[str]] = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot the autocorrelation of the samples.
    Parameters
    ----------
    samples : np.ndarray or list of np.ndarray
        The samples to plot. Should be of shape (n_dim, n_samples).
    maxlag : int, optional
        The maximum lag to compute the autocorrelation for. Default is 500.
    step : int, optional
        The step size for the lag. Default is 1.
    img_kwargs : dict, optional
        Dictionary containing image parameters such as label_fontsize, title_fontsize, tick_fontsize, legend_fontsize, and img_format.
    labels : list of str, optional
        List of labels for each dimension. If None, they will be set to '\theta_1', '\theta_2', etc.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the autocorrelation plot.
    axs : list of matplotlib.axes.Axes
        List of axes objects for each subplot.
    """

    # Convert single array to list for uniform handling
    if isinstance(samples, np.ndarray):
        samples = [samples]

    samples = _validate_samples(samples)
    img_kwargs = _setup_plot_style(img_kwargs)

    # Compute autocorrelation for all sample sets
    all_autos = []
    all_taus = []
    all_ess = []

    for samp in samples:
        autos, taus, ess = autocorrelation(samp)
        all_autos.append(autos)
        all_taus.append(taus)
        all_ess.append(ess)

    if labels is None:
        labels = [rf"$\theta_{ii + 1}$" for ii in range(samples[0].shape[0])]

    if sample_labels is None and len(samples) > 1:
        sample_labels = [f"Chain {i + 1}" for i in range(len(samples))]
    elif sample_labels is None:
        sample_labels = [None]

    # Set the style of the visualization
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.3)

    markers = [
        "o",
        "s",
        "D",
        "^",
        "v",
        "<",
        ">",
        "p",
        "*",
        "h",
        "H",
        "+",
        "x",
        "d",
        "|",
        "_",
    ]
    colors = sns.color_palette("tab10")

    dim = samples[0].shape[0]
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), sharex=True)

    lags = np.arange(1, maxlag + 1, step)
    for i in range(dim):
        for j, autos in enumerate(all_autos):
            marker = markers[i % len(markers)]
            color = colors[(i * len(samples) + j) % len(colors)]

            # Create label combining dimension and sample labels
            if len(samples) > 1 and sample_labels[j] is not None:
                label = f"{labels[i]} ({sample_labels[j]})"
            else:
                label = f"{labels[i]}"

            axs.plot(
                lags,
                autos[i, :maxlag],
                marker,
                label=label,
                alpha=0.2,
                markersize=7,
                linewidth=4,
                color=color,
            )

    axs.set_ylabel("Autocorrelation")
    axs.set_xlabel("Lag")

    # Position legend outside plot area if many entries
    num_legend_entries = dim * len(samples)
    if num_legend_entries > 6:
        axs.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)
    else:
        axs.legend(loc="best", frameon=True)

    axs.grid(False)
    axs.spines["top"].set_color("black")
    axs.spines["top"].set_linewidth(2)
    axs.spines["right"].set_color("black")
    axs.spines["right"].set_linewidth(2)
    axs.spines["bottom"].set_color("black")
    axs.spines["bottom"].set_linewidth(2)
    axs.spines["left"].set_color("black")
    axs.spines["left"].set_linewidth(2)
    axs.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if axs.get_legend() is not None:
        for label in (
            axs.get_xticklabels()
            + axs.get_yticklabels()
            + axs.get_legend().get_texts()
            + [axs.xaxis.label, axs.yaxis.label]
        ):
            label.set_color("black")
    else:
        for label in (
            axs.get_xticklabels()
            + axs.get_yticklabels()
            + [axs.xaxis.label, axs.yaxis.label]
        ):
            label.set_color("black")

    plt.tight_layout()

    return fig, axs, all_ess, all_taus


def next_pow_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    i = 1
    while i < n:
        i = i << 1
    return i


def function_1d(x: np.ndarray) -> np.ndarray:
    """FFT-based autocorrelation function, normalized (lag 0 = 1)."""
    x = np.atleast_1d(x)
    n = next_pow_two(len(x))
    x_mean = np.mean(x)
    f = np.fft.fft(x - x_mean, n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= acf[0]
    return acf


def auto_window(taus: np.ndarray, c: int) -> int:
    """Emcee-style window: first lag where lag < c * tau."""
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return int(np.argmin(m))
    return len(taus) - 1


def autocorrelation(
    samples: np.ndarray, c: Optional[int] = 5, tol: Optional[int] = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the correlation of a set of samples
    New implementation based on emcee's autocorrelation function.

    Estimate the integrated autocorrelation time of a time series.

    This estimate uses the iterative procedure described on page 16 of
    `Sokal's notes <https://www.semanticscholar.org/paper/Monte-Carlo-Methods-in-Statistical-Mechanics%3A-and-Sokal/0bfe9e3db30605fe2d4d26e1a288a5e2997e7225>`_ to
    determine a reasonable window size.

    Parameters
    ----------
    samples : np.ndarray
        The samples to compute the autocorrelation for. Should be of shape (n_dim, n_samples).
    c : int
        The step size for window search
    tol : int
        The minimum number of autocorrelation times needed to trust the estimate.

    Returns
    -------
    autos : np.ndarray of shape (ndim, nsamples)
        Autocorrelation function per dimension.
    taus : np.ndarray of shape (ndim,)
        Integrated autocorrelation time per dimension.
    ess : np.ndarray of shape (ndim,)
        Effective sample size per dimension.
    """
    ndim, nsamples = samples.shape

    # Check if the samples are in the correct format
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise ValueError("Samples should be a 2D numpy array.")
    if samples.shape[0] > samples.shape[1]:
        raise ValueError(
            "Samples should be in the format (d, N), where d is the number of dimensions and N is the number of samples."
        )

    autos = np.empty((ndim, nsamples))
    taus = np.empty(ndim)
    windows = np.empty(ndim, dtype=int)

    for d in range(ndim):
        acf = function_1d(samples[d])
        autos[d, :] = acf
        tau_seq = 2.0 * np.cumsum(acf) - 1.0
        window = auto_window(tau_seq, c)
        taus[d] = tau_seq[window]
        windows[d] = window

    # Check convergence
    flag = tol * taus > nsamples

    # Warn or raise in the case of non-convergence
    if np.any(flag):
        msg = (
            "The chain is shorter than {0} times the integrated "
            "autocorrelation time for {1} parameter(s). Use this estimate "
            "with caution and run a longer chain!\n"
        ).format(tol, np.sum(flag))
        msg += "N/{0} = {1:.0f};\ntau: {2}".format(tol, nsamples / tol, taus)

    ess = np.ceil(nsamples / taus)
    return autos, taus, ess


def effective_sample_size(acf_1d: np.ndarray) -> float:
    """
    Effective sample size from a 1D autocorrelation function.

    Uses the identity ESS = n / tau with integrated autocorrelation time
    tau = 1 + 2 * sum(acf[1:]).

    Parameters
    ----------
    acf_1d : numpy.ndarray of shape (L,)
        Autocorrelation at lags 0, 1, ..., L-1 (acf[0] should be 1.0).

    Returns
    -------
    float
        Effective sample size (n / tau).
    """
    acf_1d = np.asarray(acf_1d).flatten()
    n = len(acf_1d)
    if n <= 1:
        return 1.0
    tau = 1.0 + 2.0 * np.sum(acf_1d[1:])
    if tau <= 0:
        return float(n)
    return float(n / tau)
