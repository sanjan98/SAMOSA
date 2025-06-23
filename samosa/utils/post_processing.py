"""Markov Chain Monte Carlo Plotting and Utilities
Modified from - Alex Gorodetsky, 2020's script
License: MIT
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde

from samosa.core.state import ChainState

from typing import List, Any, Optional, Dict, Tuple

def load_samples(output_dir: str, iteration: int = None) -> List[ChainState]:
    """
    Load MCMC samples from a pickle file.

    Parameters:
    ----------
        output_dir (str): Directory where the samples are saved.
        iteration (int, optional): Iteration number to load specific samples. If None, loads all samples.

    Returns:
    -------
        samples (list): List of ChainState objects representing the MCMC samples.
    """
    if iteration is None:
        with open(f'{output_dir}/samples.pkl', "rb") as f:
            samples = pickle.load(f)
            return samples
    else:
        file_path = f'{output_dir}/samples_{iteration}.pkl'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(f'{output_dir}/samples_{iteration}.pkl', "rb") as f:
            samples = pickle.load(f)
            return samples

def load_coupled_samples(output_dir: str, iteration: int = None) -> List[ChainState]:
    """
    Load coupled MCMC samples from a pickle file.

    Parameters:
    ----------
        output_dir (str): Directory where the samples are saved.
        iteration (int, optional): Iteration number to load specific samples. If None, loads all samples.

    Returns:
    -------
        samples_coarse (list): List of ChainState objects representing the MCMC samples.
        samples_fine (list): List of ChainState objects representing the MCMC samples.
    """
    if iteration is None:
        with open(f'{output_dir}/samples_coarse.pkl', "rb") as f:
            samples_coarse = pickle.load(f)
        with open(f'{output_dir}/samples_fine.pkl', "rb") as f:
            samples_fine = pickle.load(f)
            return samples_coarse, samples_fine
    else:
        file_path_coarse = f'{output_dir}/samples_coarse{iteration}.pkl'
        file_path_fine = f'{output_dir}/samples_fine{iteration}.pkl'
        if not os.path.exists(file_path_coarse):
            raise FileNotFoundError(f"The file {file_path_coarse} does not exist.")
        if not os.path.exists(file_path_fine):
            raise FileNotFoundError(f"The file {file_path_fine} does not exist.")
        with open(f'{file_path_coarse}', "rb") as f:
            samples_coarse = pickle.load(f)
        with open(f'{file_path_fine}', "rb") as f:
            samples_fine = pickle.load(f)
            return samples_coarse, samples_fine

def get_position_from_states(samples: List[ChainState], burnin: Optional[float] = 0.0) -> np.ndarray:
    """
    From a list of ChainState objects, extract the position of each state.

    Parameters
    ----------
    samples : list of ChainState
        List of ChainState objects.
    burnin : float, optional
        Fraction of samples to discard as burn-in. Default is 0.25.

    Returns
    -------
    positions : np.ndarray
        2D numpy array of shape (d, N), where d is the number of dimensions and N is the number of samples.
    """

    # Check if the samples are in the correct format
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("Samples should be a non-empty list of ChainState objects.")
    if not all(isinstance(s, ChainState) for s in samples):
        raise ValueError("All samples should be ChainState objects.")
    
    # Extract the position from each ChainState object
    positions = [np.ravel(s.position) for s in samples]

    # Discard the burn-in samples
    if burnin >= 0:
        n_burnin = int(len(positions) * burnin)
        positions = positions[n_burnin:]
    elif burnin < 0:
        raise ValueError("Burn-in must be a positive value.")
    
    # Convert to numpy array and transpose to (d, N) format
    return np.column_stack(positions)

def get_reference_position_from_states(samples: List[ChainState], burnin: Optional[float] = 0.0) -> np.ndarray:
    """
    From a list of ChainState objects, extract the position of each state.

    Parameters
    ----------
    samples : list of ChainState
        List of ChainState objects.
    burnin : float, optional
        Fraction of samples to discard as burn-in. Default is 0.25.

    Returns
    -------
    reference_positions : np.ndarray
        2D numpy array of shape (d, N), where d is the number of dimensions and N is the number of samples.
    """

    # Check if the samples are in the correct format
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("Samples should be a non-empty list of ChainState objects.")
    if not all(isinstance(s, ChainState) for s in samples):
        raise ValueError("All samples should be ChainState objects.")
    
    # Extract the position from each ChainState object
    reference_positions = [np.ravel(s.reference_position) for s in samples]

    # Discard the burn-in samples
    if burnin >= 0:
        n_burnin = int(len(reference_positions) * burnin)
        reference_positions = reference_positions[n_burnin:]
    elif burnin < 0:
        raise ValueError("Burn-in must be a positive value.")
    
    # Convert to numpy array and transpose to (d, N) format
    return np.column_stack(reference_positions)
    
def scatter_matrix(samples: List[np.ndarray], mins: Optional[np.ndarray] = None, maxs: Optional[np.ndarray] = None, upper_right: Optional[Any] = None, specials: Optional[Any] = None, hist_plot: Optional[bool] = True, nbins: Optional[int] = 100, img_kwargs: Optional[Dict[str, int]] = None, labels: Optional[List[str]] = None, sample_labels: Optional[List[str]] = None) -> Tuple[plt.Figure, List[plt.Axes], GridSpec]:

    """
    Create a nice scatter plot matrix of the samples.
    The marginals are one the digaonal and the joint distributions are on the off-diagonal.

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

    nchains = len(samples)
    dim = samples[0].shape[0]

    # Check if the samples are in the correct format
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("Samples should be a non-empty list of numpy arrays.")
    if not all(isinstance(s, np.ndarray) for s in samples):
        raise ValueError("All samples should be numpy arrays.")
    if not all(s.ndim == 2 for s in samples):
        raise ValueError("All samples should be 2D arrays.")
    if not all(s.shape[0] <= s.shape[1] for s in samples):
        raise ValueError("Samples should be in the format (d, N), where d is the number of dimensions and N is the number of samples.")
    if not all(s.shape[0] == samples[0].shape[0] for s in samples):
        raise ValueError("All samples should have the same number of dimensions.")
    
    # Set some default values for img_kwargs if not provided
    if img_kwargs is None:
        img_kwargs = {
            'label_fontsize': 18,
            'title_fontsize': 20,
            'tick_fontsize': 16,
            'legend_fontsize': 16,
            'img_format': 'png'
        }

    plt.rcParams.update({
    'axes.labelsize': img_kwargs['label_fontsize'],
    'axes.titlesize': img_kwargs['title_fontsize'],
    'xtick.labelsize': img_kwargs['tick_fontsize'],
    'ytick.labelsize': img_kwargs['tick_fontsize'],
    'legend.fontsize': img_kwargs['legend_fontsize']
    })

    # Set some default labels if none are provided
    if labels is None:
        labels = [rf'$\theta_{ii+1}$' for ii in range(dim)]

    # Set the style using Seaborn and configure LaTeX font
    sns.set_style("white")
    sns.set_context("talk")
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Define a list of different colormaps to use for different samples
    cmap_list = [plt.cm.viridis, plt.cm.plasma, plt.cm.cividis, plt.cm.inferno, plt.cm.magma]

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
                    minspec = np.min([spec['vals'][ii] for spec in specials if spec['vals'][ii] is not None])
                    maxspec = np.max([spec['vals'][ii] for spec in specials if spec['vals'][ii] is not None])
                else:
                    minspec = specials['vals'][ii]
                    maxspec = specials['vals'][ii]
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
        end = dim
        l = dim
    else:
        gs = GridSpec(dim + 1, dim + 1, figure=fig)
        axs = [None] * (dim + 1) * (dim + 1)
        start = 1
        end = dim + 1
        l = dim + 1

    # means = [np.mean(np.concatenate([samples[kk][ii, :] for kk in range(nchains)])) for ii in range(dim)]
    means = np.array([[np.mean(samples[kk][ii, :]) for ii in range(dim)] for kk in range(nchains)])

    def one_decimal(x, pos):
        return f'{x:.1f}'

    formatter = FuncFormatter(one_decimal)

    for ii in range(dim):
        axs[ii] = fig.add_subplot(gs[ii + start, ii])
        ax = axs[ii]

        ax.grid(False)  # Disable gridlines on the diagonal plots

        if ii < dim - 1:
            ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
        else:
            ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True)
            if labels:
                ax.set_xlabel(labels[ii])

        ax.tick_params(axis='y', left=False, right=False, labelleft=False)
        ax.set_frame_on(False)

        # Use Gaussian KDE for the diagonal plots
        for kk in range(nchains):
            sampii = samples[kk][ii, :]
            kde = gaussian_kde(sampii)
            x_grid = np.linspace(use_mins[ii], use_maxs[ii], 1000)
            ax.fill_between(x_grid, 0, kde(x_grid), color=colors[kk % len(colors)], alpha=0.3)
            ax.plot(x_grid, kde(x_grid), color=colors[kk % len(colors)], alpha=0.7)
            # Plot vertical line for mean of this chain and dimension
            ax.axvline(means[kk, ii], color=colors[kk % len(colors)], linestyle='--', lw=2)

        ax.set_xlim((use_mins[ii], use_maxs[ii]))
        ax.set_ylim(0)  # Start the y-axis at zero for alignment with lower plots

        if specials is not None:
            for special in specials:
                if special['vals'][ii] is not None:
                    if 'color' in special:
                        ax.axvline(special['vals'][ii], color=special['color'], lw=2)
                    else:
                        ax.axvline(special['vals'][ii], lw=2)

        # ax.axvline(means[ii], color='red', linestyle='--', lw=2, label=f'Mean: {means[ii]:.2f}')
        ax.set_xlim((use_mins[ii] - 1e-10, use_maxs[ii] + 1e-10))

        diff = 0.2 * (use_maxs[ii] - use_mins[ii])
        xticks = np.linspace(use_mins[ii] + diff, use_maxs[ii] - diff, 2)
        yticks = ax.get_yticks()
        if len(yticks) >= 2:
            yticks = np.linspace(yticks[0], yticks[-1], 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        for jj in range(ii + 1, dim):
            axs[jj * l + ii] = fig.add_subplot(gs[jj + start, ii])
            ax = axs[jj * l + ii]

            ax.grid(False)  # Disable gridlines on the joint plots

            if jj < dim - 1:
                ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
            else:
                ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True)
                if labels:
                    ax.set_xlabel(labels[ii])
            if ii > 0:
                ax.tick_params(axis='y', left=False, right=False, labelleft=False)
            else:
                ax.tick_params(axis='y', left=True, right=False, labelleft=True)
                if labels:
                    ax.set_ylabel(labels[jj])

            ax.set_frame_on(True)

            for kk in range(nchains):
                cmap = cmap_list[kk % len(cmap_list)]  # Choose a colormap for each sample set
                data = np.vstack([samples[kk][ii, :], samples[kk][jj, :]])
                kde = gaussian_kde(data)
                x_grid = np.linspace(use_mins[ii], use_maxs[ii], nbins)
                y_grid = np.linspace(use_mins[jj], use_maxs[jj], nbins)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

                if hist_plot:
                    ax.contour(X, Y, Z, levels=5, cmap=cmap, linewidths=1.0)
                    ax.plot(samples[kk][ii, :], samples[kk][jj, :], 'o', ms=1, alpha=0.01, color=colors[kk % len(colors)], label=sample_labels[kk] if sample_labels is not None else f'Chain {kk+1}')
                else:
                    ax.plot(samples[kk][ii, :], samples[kk][jj, :], 'o', ms=1, alpha=0.2, color=colors[kk % len(colors)], label=sample_labels[kk] if sample_labels is not None else f'Chain {kk+1}')
            if specials is not None:
                for special in specials:
                    if 'color' in special:
                        ax.plot(special['vals'][ii], special['vals'][jj], 'x', color=special['color'], ms=2, mew=2)
                    else:
                        ax.plot(special['vals'][ii], special['vals'][jj], 'x', ms=2, mew=2)

            if jj == ii + 1:
                leg = ax.legend(loc='best', fontsize=img_kwargs['legend_fontsize'], markerscale=6)
                for lh in leg.legend_handles:
                    lh.set_alpha(1)

            ax.set_xlim((use_mins[ii], use_maxs[ii]))
            ax.set_ylim((use_mins[jj] - 1e-10, use_maxs[jj] + 1e-10))

            diff = 0.2 * (use_maxs[ii] - use_mins[ii])
            xticks = np.linspace(use_mins[ii] + diff, use_maxs[ii] - diff, 2)
            yticks = np.linspace(use_mins[jj] + diff, use_maxs[jj] - diff, 2)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)

            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout(pad=0.01)

    if upper_right is not None:
        size_ur = int(dim / 2)

        name = upper_right['name']
        vals = upper_right['vals']
        if 'log_transform' in upper_right:
            log_transform = upper_right['log_transform']
        else:
            log_transform = None
        ax = fig.add_subplot(gs[0:int(dim / 2),
                                size_ur + 1:size_ur + int(dim / 2) + 1])

        lb = np.min([np.quantile(val, 0.01) for val in vals])
        ub = np.max([np.quantile(val, 0.99) for val in vals])
        for kk in range(nchains):
            if log_transform is not None:
                pv = np.log10(vals[kk])
                ra = (np.log10(lb), np.log10(ub))
            else:
                pv = vals[kk]
                ra = (lb, ub)
            ax.hist(pv,
                    density=True,
                    range=ra,
                    edgecolor='black',
                    stacked=True,
                    bins='auto',
                    alpha=0.5)  # Adjust transparency
        ax.tick_params(axis='x', bottom='both', top=False, labelbottom=True)
        ax.tick_params(axis='y', left='both', right=False, labelleft=False)
        ax.set_frame_on(True)
        ax.set_xlabel(name)

        diff = 0.2 * (ra[1] - ra[0])
        xticks = np.linspace(ra[0] + diff, ra[1] - diff, 2)
        yticks = ax.get_yticks()
        if len(yticks) >= 2:
            yticks = np.linspace(yticks[0], yticks[-1], 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    plt.subplots_adjust(left=0.15, right=0.95)

    return fig, axs, gs

def plot_trace(samples: np.ndarray, img_kwargs: Optional[Dict] = None, labels: Optional[List] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot the trace of the samples.
    Parameters
    ----------
    samples : np.ndarray
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

    # Check if the samples are in the correct format
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise ValueError("Samples should be a 2D numpy array.")
    if samples.shape[0] > samples.shape[1]:
        raise ValueError("Samples should be in the format (d, N), where d is the number of dimensions and N is the number of samples.")

    # Set some default values for img_kwargs if not provided
    if img_kwargs is None:
        img_kwargs = {
            'label_fontsize': 18,
            'title_fontsize': 20,
            'tick_fontsize': 16,
            'legend_fontsize': 16,
            'img_format': 'png'
        }
    plt.rcParams.update({
    'axes.labelsize': img_kwargs['label_fontsize'],
    'axes.titlesize': img_kwargs['title_fontsize'],
    'xtick.labelsize': img_kwargs['tick_fontsize'],
    'ytick.labelsize': img_kwargs['tick_fontsize'],
    'legend.fontsize': img_kwargs['legend_fontsize']
    })
    if labels is None:
        labels = [rf'$\theta_{ii+1}$' for ii in range(samples.shape[1])]
 
    sns.set_style("white")
    sns.set_context("talk")

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    dim = samples.shape[0]

    fig, axs = plt.subplots(dim, 1, figsize=(16,8), sharex=True)
    if dim == 1:
        axs = [axs]  # Ensure axs is a list when dim=1
    for i in range(dim):
        axs[i].plot(samples[i, :], alpha=0.3)
        axs[i].set_ylabel(f'{labels[i]}')
        
    axs[dim-1].set_xlabel('Sample Number')

    for i in range(dim):
        axs[i].grid(False)
        axs[i].spines['top'].set_color('black')
        axs[i].spines['top'].set_linewidth(2)
        axs[i].spines['right'].set_color('black')
        axs[i].spines['right'].set_linewidth(2)
        axs[i].spines['bottom'].set_color('black')
        axs[i].spines['bottom'].set_linewidth(2)
        axs[i].spines['left'].set_color('black')
        axs[i].spines['left'].set_linewidth(2)
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if axs[i].get_legend() is not None:
            for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels() + axs[i].get_legend().get_texts() + [axs[i].xaxis.label, axs[i].yaxis.label]):
                label.set_color('black')
        else:
            for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels() + [axs[i].xaxis.label, axs[i].yaxis.label]):
                label.set_color('black')

    return fig, axs

def plot_lag(samples: np.ndarray, maxlag: Optional[int] = 500, step: Optional[int] = 1, img_kwargs: Optional[Dict[str, int]] = None, labels: Optional[List[str]] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot the autocorrelation of the samples.
    Parameters
    ----------
    samples : np.ndarray
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

    # Check if the samples are in the correct format
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise ValueError("Samples should be a 2D numpy array.")
    if samples.shape[0] > samples.shape[1]:
        raise ValueError("Samples should be in the format (d, N), where d is the number of dimensions and N is the number of samples.")
    
    # Set some default values for img_kwargs if not provided
    if img_kwargs is None:
        img_kwargs = {
            'label_fontsize': 18,
            'title_fontsize': 20,
            'tick_fontsize': 16,
            'legend_fontsize': 16,
            'img_format': 'png'
        }

    # Setting font sizes with rcParams
    plt.rcParams.update({
        'axes.labelsize': img_kwargs['label_fontsize'],
        'axes.titlesize': img_kwargs['title_fontsize'],
        'xtick.labelsize': img_kwargs['tick_fontsize'],
        'ytick.labelsize': img_kwargs['tick_fontsize'],
        'legend.fontsize': img_kwargs['legend_fontsize']
    })

    if labels is None:
        labels = [rf'$\theta_{ii+1}$' for ii in range(samples.shape[1])]

    # Set the style of the visualization
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.3)

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']

    lags, autolag = autocorrelation(samples, maxlag=maxlag,step=step)
    ess = []
    dim = samples.shape[0]
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), sharex=True)

    for i in range(dim):
        ess.append(effective_sample_size(autolag[i, :]))
        axs.plot(lags, autolag[i, :], markers[i], label=f'{labels[i]}, ess = {ess[i]:0.2f}', alpha=0.7, markersize=7, linewidth=4)
    axs.set_ylabel(f'Autocorrelation')
    # Set the xlabel for the last subplot (shared x-axis)
    axs.set_xlabel('Lag')
    axs.legend()
    
    axs.grid(False)
    axs.spines['top'].set_color('black')
    axs.spines['top'].set_linewidth(2)
    axs.spines['right'].set_color('black')
    axs.spines['right'].set_linewidth(2)
    axs.spines['bottom'].set_color('black')
    axs.spines['bottom'].set_linewidth(2)
    axs.spines['left'].set_color('black')
    axs.spines['left'].set_linewidth(2)
    axs.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if axs.get_legend() is not None:
        for label in (axs.get_xticklabels() + axs.get_yticklabels() + axs.get_legend().get_texts() + [axs.xaxis.label, axs.yaxis.label]):
            label.set_color('black')
    else:
        for label in (axs.get_xticklabels() + axs.get_yticklabels() + [axs.xaxis.label, axs.yaxis.label]):
            label.set_color('black')

    return fig, axs

def autocorrelation(samples: np.ndarray, maxlag: Optional[int] = 100, step: Optional[int] = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the correlation of a set of samples
    
    Parameters
    ----------
    samples : np.ndarray
        The samples to compute the autocorrelation for. Should be of shape (n_dim, n_samples).
    maxlag : int
        The maximum lag to compute the autocorrelation for.
    step : int
        The step size for the lag. Default is 1.

    Returns
    -------
    lags : np.ndarray
        The lags for which the autocorrelation is computed.
    autos : np.ndarray
        The autocorrelation values for each dimension at each lag.
    """
    
    ndim, nsamples = samples.shape

    # Check if the samples are in the correct format
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise ValueError("Samples should be a 2D numpy array.")
    if samples.shape[0] > samples.shape[1]:
        raise ValueError("Samples should be in the format (d, N), where d is the number of dimensions and N is the number of samples.")
    
    # Compute the mean
    mean = np.mean(samples, axis=1)
    
    # Compute the denominator, which is variance
    denominator = np.zeros((ndim))
    for ii in range(nsamples):
        denominator = denominator + (samples[:, ii] - mean)**2
    
    lags = np.arange(0, maxlag, step)
    autos = np.zeros((ndim, len(lags)))

    for zz, lag in enumerate(lags):
        autos[:, zz] = np.zeros((ndim))
        # compute the covariance between all samples *lag apart*
        for ii in range(nsamples - lag):
            autos[:, zz] = autos[:, zz] + (samples[:, ii] - mean) * (samples[:, ii + lag] - mean)
        autos[:, zz] = autos[:, zz] / denominator

    return lags, autos

def effective_sample_size(auto_corrs: np.ndarray) -> float:
    """
    Estimate the effective sample size for an array of samples.
    Parameters
    ----------
    auto_corrs : np.ndarray
        The autocorrelation values for which to compute the effective sample size.
    
    Returns
    -------
    ess : float
        The effective sample size.
    """
    n = len(auto_corrs)

    # Sum the sequence of autocorrelations
    negative_autocorr = auto_corrs[auto_corrs < 0]
    if len(negative_autocorr) > 0:  # truncate the sum at first negative autocorrelation
        first_negative = np.where(auto_corrs < 0)[0][0]
    else:
        first_negative = len(auto_corrs)

    ess = n / (1 + 2 * np.sum(auto_corrs[:first_negative]))
    return ess

