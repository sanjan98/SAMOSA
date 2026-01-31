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

from typing import List, Any, Optional, Dict, Tuple, Union

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
            'label_fontsize': 24,
            'title_fontsize': 20,
            'tick_fontsize': 20,
            'legend_fontsize': 24,
            'img_format': 'png'
        }

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
                ax.set_xlabel(labels[ii], fontsize=img_kwargs['label_fontsize'])

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
            # ax.axvline(means[kk, ii], color=colors[kk % len(colors)], linestyle='--', lw=2)

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
        ax.tick_params(axis='both', labelsize=img_kwargs['tick_fontsize'])

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        for jj in range(ii + 1, dim):
            axs[jj * l + ii] = fig.add_subplot(gs[jj + start, ii])
            ax = axs[jj * l + ii]

            ax.grid(False) 

            if jj < dim - 1:
                ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
            else:
                ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True)
                if labels:
                    ax.set_xlabel(labels[ii], fontsize=img_kwargs['label_fontsize'])
            if ii > 0:
                ax.tick_params(axis='y', left=False, right=False, labelleft=False)
            else:
                ax.tick_params(axis='y', left=True, right=False, labelleft=True)
                if labels:
                    ax.set_ylabel(labels[jj], fontsize=img_kwargs['label_fontsize'])

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
                    max_z = Z.max()
                    min_z = Z.min()
                    all_levels = np.linspace(min_z, max_z, 6)
                    
                    # Skip the first (lowest density) level to remove the outermost contour
                    levels = all_levels[1:]  # This gives you 5 levels, excluding the lowest
                    ax.contour(X, Y, Z, levels=levels, cmap=cmap, linewidths=1.0)
                    ax.plot(samples[kk][ii, :], samples[kk][jj, :], 'o', ms=1, alpha=0.01, color=colors[kk % len(colors)], label=sample_labels[kk] if sample_labels is not None else f'Chain {kk+1}', rasterized=True)
                else:
                    ax.plot(samples[kk][ii, :], samples[kk][jj, :], 'o', ms=1, alpha=0.2, color=colors[kk % len(colors)], label=sample_labels[kk] if sample_labels is not None else f'Chain {kk+1}', rasterized=True)
            if specials is not None:
                for special in specials:
                    if 'color' in special:
                        ax.plot(special['vals'][ii], special['vals'][jj], 'x', color=special['color'], ms=2, mew=2, rasterized=True)
                    else:
                        ax.plot(special['vals'][ii], special['vals'][jj], 'x', ms=2, mew=2, rasterized=True)

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
            ax.tick_params(axis='both', labelsize=img_kwargs['tick_fontsize'])

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
        ax.set_xlabel(name, fontsize=img_kwargs['label_fontsize'])

        diff = 0.2 * (ra[1] - ra[0])
        xticks = np.linspace(ra[0] + diff, ra[1] - diff, 2)
        yticks = ax.get_yticks()
        if len(yticks) >= 2:
            yticks = np.linspace(yticks[0], yticks[-1], 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis='both', labelsize=img_kwargs['tick_fontsize'])

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    
    if sample_labels is not None:
        # Place the legend in the upper right whitespace
        # [left, bottom, width, height] in figure coordinates; adjust as needed
        left=0.6; bottom=0.7; width=0.20
        nlabels = len(sample_labels)
        # Dynamically set the height based on number of labels
        height = max(0.08, min(0.04 * nlabels, 0.25))  # min and max for reasonable bounds
        legend_ax = fig.add_axes([left, bottom, width, height])
        legend_ax.axis('off')
        handles = []
        for kk in range(nchains):
            color = colors[kk % len(colors)]
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                    label=sample_labels[kk], markerfacecolor=color, markersize=10))
        legend_ax.legend(handles=handles, loc='center', frameon=False, fontsize=img_kwargs['legend_fontsize'])

    plt.subplots_adjust(left=0.15, right=0.95)

    return fig, axs, gs

def joint_plots(samples: List[np.ndarray], img_kwargs: Optional[Dict[str, int]] = None, labels: Optional[List[str]] = None, bins: int = 30) -> List[plt.Figure]:

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

    # Validation
    if len(samples) != 2:
        raise ValueError("Need 2 samples for pairwise plotting.")

    dim, nsamples = samples[0].shape

    # Set defaults
    if img_kwargs is None:
        img_kwargs = {
            'label_fontsize': 24,
            'title_fontsize': 20,
            'tick_fontsize': 20,
            'legend_fontsize': 16,
            'img_format': 'png'
        }

    if labels is None:
        labels = [rf'$\theta_{ii+1}$' for ii in range(dim)]

    # Set style
    sns.set_style("white")
    sns.set_context("talk")
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.rcParams.update({
        'axes.labelsize': img_kwargs['label_fontsize'],
        'axes.titlesize': img_kwargs['title_fontsize'],
        'xtick.labelsize': img_kwargs['tick_fontsize'],
        'ytick.labelsize': img_kwargs['tick_fontsize'],
        'legend.fontsize': img_kwargs['legend_fontsize']
    })

    figures = []

    # Create joint plots for each dimension, comparing consecutive samples
    for dd in range(dim):
                
        x = samples[0][dd, :]
        y = samples[1][dd, :]
        
        # Create joint plot using seaborn
        g = sns.jointplot(x=x, y=y, kind='scatter', marginal_kws=dict(bins=bins, fill=True), alpha=0.6, s=30, linewidth=0, joint_kws={'rasterized': True})
        g.figure.set_size_inches((8, 8))
        
        # Set labels
        g.set_axis_labels(f'{labels[dd]} - coarse', f'{labels[dd]} - fine')
        
        # Adjust tick parameters
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        ticks_x = np.linspace(x_min, x_max, 4)
        ticks_y = np.linspace(y_min, y_max, 4)
        g.ax_joint.set_xticks([round(tick, 2) for tick in ticks_x])
        g.ax_joint.set_yticks([round(tick, 2) for tick in ticks_y])
        
        g.ax_joint.grid(False)
        g.ax_joint.spines['top'].set_color('black')
        g.ax_joint.spines['top'].set_linewidth(2)
        g.ax_joint.spines['right'].set_color('black')
        g.ax_joint.spines['right'].set_linewidth(2)
        g.ax_joint.spines['bottom'].set_color('black')
        g.ax_joint.spines['bottom'].set_linewidth(2)
        g.ax_joint.spines['left'].set_color('black')
        g.ax_joint.spines['left'].set_linewidth(2)

        if g.ax_joint.get_legend() is not None:
            for label in (g.ax_joint.get_xticklabels() + g.ax_joint.get_yticklabels() + g.ax_joint.get_legend().get_texts() + [g.ax_joint.xaxis.label, g.ax_joint.yaxis.label]):
                label.set_color('black')
        else:
            for label in (g.ax_joint.get_xticklabels() + g.ax_joint.get_yticklabels() + [g.ax_joint.xaxis.label, g.ax_joint.yaxis.label]):
                label.set_color('black')

        figures.append(g.figure)

    return figures

def plot_trace(samples: Union[np.ndarray, List[np.ndarray]], img_kwargs: Optional[Dict] = None, labels: Optional[List] = None, sample_labels: Optional[List[str]] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
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
        raise ValueError("Samples should be a numpy array or non empty list of numoy arrays.")
    
    for i, samp in enumerate(samples):
        if not isinstance(samp, np.ndarray) or samp.ndim != 2:
            raise ValueError(f"Sample {i} should be a 2D numpy array.")
        if samp.shape[0] > samp.shape[1]:
            raise ValueError(f"Sample {i} should be in the format (d, N), where d is the number of dimensions and N is the number of samples.")
    
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

    dim = samples[0].shape[0]

    if labels is None:
        labels = [rf'$\theta_{ii+1}$' for ii in range(dim)]

    if sample_labels is None and len(samples) > 1:
        sample_labels = [f'Chain {i+1}' for i in range(len(samples))]
    elif sample_labels is None:
        sample_labels = [None]
 
    sns.set_style("white")
    sns.set_context("talk")

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Get default color palette
    colors = sns.color_palette("tab10")

    fig, axs = plt.subplots(dim, 1, figsize=(16,8), sharex=True)
    
    if dim == 1:
        axs = [axs]  # Ensure axs is a list when dim=1
    
    for i in range(dim):
        for j, samp in enumerate(samples):
            label = sample_labels[j] if j < len(sample_labels) else None
            axs[i].plot(samp[i, :], alpha=0.6, color=colors[j % len(colors)], linewidth=0.8, label=label)

        axs[i].set_ylabel(f'{labels[i]}')

        if i == 0 and len(samples) > 1:
            axs[i].legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

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
        plt.tight_layout()

    return fig, axs

def plot_lag(samples: Union[np.ndarray, List[np.ndarray]], maxlag: Optional[int] = 500, step: Optional[int] = 1, img_kwargs: Optional[Dict[str, int]] = None, labels: Optional[List[str]] = None, sample_labels: Optional[List[str]] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
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

    # Validate samples
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("Samples should be a numpy array or a non-empty list of numpy arrays.")

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
        labels = [rf'$\theta_{ii+1}$' for ii in range(samples[0].shape[0])]

    if sample_labels is None and len(samples) > 1:
        sample_labels = [f'Chain {i+1}' for i in range(len(samples))]
    elif sample_labels is None:
        sample_labels = [None]

    # Set the style of the visualization
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.3)

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
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
                label = f'{labels[i]} ({sample_labels[j]})'
            else:
                label = f'{labels[i]}'
            
            axs.plot(lags, autos[i, :maxlag], marker, label=label, alpha=0.2, 
                    markersize=7, linewidth=4, color=color)
    
    axs.set_ylabel('Autocorrelation')
    axs.set_xlabel('Lag')
    
    # Position legend outside plot area if many entries
    num_legend_entries = dim * len(samples)
    if num_legend_entries > 6:
        axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    else:
        axs.legend(loc='best', frameon=True)
    
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

    plt.tight_layout()

    return fig, axs, all_ess, all_taus

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def function_1d(x):
    """
    FFT-based autocorrelation function, normalized
    """
    x = np.atleast_1d(x)
    n = next_pow_two(len(x))
    x_mean = np.mean(x)
    f = np.fft.fft(x - x_mean, n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= acf[0]
    return acf

def auto_window(taus, c):
    """
    Windowing as in emcee: stop at first lag where lag < c * tau
    """
    
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorrelation(samples: np.ndarray, c: Optional[int] = 5, tol: Optional[int] = 50) -> Tuple[np.ndarray, np.ndarray]:
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

