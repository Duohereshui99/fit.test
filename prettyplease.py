# Copyright 2021-2024 Isak Svensson

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3

import decimal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
import warnings

def corner(data, bins=20, quantiles=[0.16, 0.84], weights=None, **kwargs):
    """
    Create a pretty corner plot.

    :param bins:
        The number of bins to use in both the 1D and 2D histograms.
        Default: 20

    :param levels:
        The levels of the 2D contours showed in the lower triangle.
        Default: [0.393, 0.865]

    :param quantiles:
        The quantiles used to compute uncertainty in the 1D marginalizations.
        Must be of length 2.
        Default: [0.16, 0.84].

    :param plot_type_2d:
        The type of plot to show in the lower triangle.
        Values: 'hist', 'scatter'
        Default: 'hist'

    :param weights:
        Array of weights for each sample. Passed to the histogramming functions
        in numpy. Should have shape (len(data),).
        If None, all samples have equal weight.
        Default: None

    :param n_uncertainty_digits:
        Determines to how many significant digits the uncertainty is computed.
        This directly affects how estimates and ticks are displayed.
        Default: 1

    :param error_style:
        Defines how the error estimates are displayed.
        Options are: 'plusminus', 'parenthesis', and None.
        Default: 'plusminus'

    :param labels:
        List of parameter labels. Should be the same length as data.shape[1]
        Default: None

    :param show_estimates:
        Whether to display parameter estimates above the diagonal.
        Uses the median as the central value and uncertainty based
        on the quantiles argument.
        Default: True

    :param plot_estimates:
        Whether to add vertical lines at the quantiles in the 1D plots.
        Default: False

    :param colors:
        Synonym for 'colors'.

    :param colors:
        Color scheme to use. May be either a single color string
        or a list of colors.
        Default: ['whitesmoke', 'xkcd:royal']

    :param n_ticks:
        Number of ticks to show on each axis.
        Must currently be >= 2 or None, or a list of such values.
        Choosing None results in a considerable speedup, with
        the obvious downside of not showing any tick marks.
        Default: 2

    :param figsize:
        The figsize. Either a tuple or None.
        Default: None.

    :param fontsize:
        Fontsize to use. Affects all text.
        Default: 10

    :param title_loc:
        Determines the horizontal alignment of the axes titles.
        Can be either a string or a list of strings.
        Possible string values are 'left', 'center', 'right'.
        Default: 'left'

    :param title_x:
        Determines the horizontal position of the axes titles.
        Can be either a float or a list of floats.
        Default: None

    :param linewidth:
        Linewidth to use in plots.
        Default: 0.6

    :param limits:
        Axis limits for each parameter.
        A list of tuples with min and max for each parameter.
        Default: None

    :param printrowcol:
        Debugging tool to print (row, col) in each square.
        Mainly useful if you want manually to add things to the plots.
        Default: False

    :param crosshairs:
        A list of values where a vertical or horizontal line will be added,
        as appropriate. This will thus form a crosshair in the 2D plots.
        Default: None

    :param crosshairs_color:
        The color of the crosshairs.
        Default: (The last color in) colors.

    :param crosshairs_alpha:
        The opacity of the crosshairs.
        Default: 0.5.

    :param return_axes:
        If True, the function returns the (fig, axes) tuple.
        If False, the return value is fig.
        Default: False
    """

    warnings.formatwarning = format_warning

    # The length of quantiles must be 2
    assert(len(quantiles) == 2)

    # Get number of dimensions immediately
    ndim = data.shape[1]

    # POP KEYWORD ARGUMENTS
    # Color scheme. If colors is a string then the color scheme is
    # "whitesmoke plus the specified color".
    # If colors is a list the user has completely
    # specified the color scheme they want.
    colors = ['whitesmoke', 'xkcd:royal']
    if kwargs.get('colors') is not None:
        colors = kwargs.pop('colors')
    elif kwargs.get('color') is not None:
        colors = kwargs.pop('color')
    if type(colors) is str:
        colors = ['whitesmoke', colors]
    # 2D contour levels to draw using solid lines
    levels = kwargs.pop('levels', compute_sigma_levels([1.0, 2.0]))
    plot_type_2d = kwargs.pop('plot_type_2d', 'hist')
    labels = kwargs.pop('labels', None)
    plot_estimates = kwargs.pop('plot_estimates', False) # Show vertical lines at quantiles?
    show_estimates = kwargs.pop('show_estimates', True) # Show median and uncertainty above diagonal
    error_style = kwargs.pop('error_style', 'plusminus') # How the error estimates appear
    n_uncertainty_digits = kwargs.pop('n_uncertainty_digits', 1)
    if n_uncertainty_digits > 1 and error_style == 'parenthesis':
        warnings.warn("Using n_uncertainty_digits > 1 with error_style == \'parenthesis\' may cause ambiguous forms for the error estimates. Check these carefully.")
    n_ticks = kwargs.pop('n_ticks', 2)
    xticklabel_rotation = kwargs.pop('xticklabel_rotation', 45)
    figsize = kwargs.pop('figsize', None)
    fontsize = kwargs.pop('fontsize', 10)
    lw = kwargs.pop('linewidth', 0.7)
    title_loc = kwargs.pop('title_loc', 'left')
    title_x = kwargs.pop('title_x', None)
    if type(title_loc) is str:
        title_loc = [title_loc] * ndim
    if title_x is None:
        title_x = [0.1 if tmp == 'left' else 0.5 for tmp in title_loc]
    if type(title_x) is float:
        title_x = [title_x] * ndim
    limits = kwargs.pop('limits', None)
    printrowcol = kwargs.pop('printrowcol', False)
    crosshairs = kwargs.pop('crosshairs', None)
    crosshairs_color = kwargs.pop('crosshairs_color', colors[-1])
    crosshairs_alpha = kwargs.pop('crosshairs_alpha', 0.5)
    return_axes = kwargs.pop('return_axes', False)

    # Warn about ignored keywords
    if len(kwargs.keys()) > 0:
        warnings.warn(f'Unused keyword argument(s) {list(kwargs.keys())}')

    density_cmap = LinearSegmentedColormap.from_list("density_cmap", colors=colors)

    # Determine suitable number of digits to show
    decimals = [determine_num_decimals(column, n_uncertainty_digits, weights) for column in data.T]

    # n_ticks can be either an int or a list
    if type(n_ticks) is int:
        n_ticks = [n_ticks] * ndim
    if n_ticks is None:
        n_ticks = [None] * ndim
    assert(len(n_ticks) == ndim)

    if figsize is None:
        figsize = automatic_figsize(ndim)

    tmp = {}
    tmp['squeeze'] = False
    tmp['sharex'] = 'col' # We cannot sharey because of the ylim on the diagonal
    tmp['figsize'] = figsize
    fig, axes = plt.subplots(nrows=ndim, ncols=ndim, **tmp)

    # Upper triangle
    for col in range(ndim):
        for row in range(col):
            axes[row, col].axis('off')
    # Diagonal
    for i in range(ndim):
        x = data[:, i].flatten()
        ax = axes[i, i]
        r = limits[i] if limits is not None else None
        ax.hist(x, bins=bins, color=colors[-1], histtype='step', linewidth=lw, density=True, weights=weights, range=r)
        ax.set_xticks([])
        ax.set_yticks([])
        if show_estimates:
            n_dec = decimals[i]
            low, median, high = low_median_high(x, quantiles, weights)
            label = labels[i] if labels is not None else ''
            title = diagonal_title(label, error_style, median, low, high, n_dec, n_uncertainty_digits)
            ax.set_title(title, fontsize=fontsize, loc=title_loc[i], x=title_x[i])
        if plot_estimates:
            c = colors[-1]
            [ax.axvline(weighted_quantile(x, [q], weights), ls='-.', color=c, lw=lw) for q in quantiles]
    # Lower triangle
    for col in range(ndim):
        for row in range(col+1, ndim):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            x1 = data[:, col]
            x2 = data[:, row]
            if plot_type_2d == 'hist':
                # The user wants 2D histograms
                r = [limits[col], limits[row]] if limits is not None else None
                plot_joint_distribution(ax, x1, x2, bins, density_cmap, levels, weights, lw, limits=r)
            elif plot_type_2d == 'scatter':
                # The user wants 2D scatter plots.
                plot_joint_scatter(ax, x1, x2, colors[-1], weights)
            else:
                raise ValueError(f'Unrecognized 2D plot type {plot_type_2d}.')

    # Add optional crosshairs
    if crosshairs is not None:
        for i in range(ndim):
            axes[i,i].axvline(crosshairs[i], color=crosshairs_color, lw=lw, alpha=crosshairs_alpha)
        for col in range(ndim):
            for row in range(col+1, ndim):
                axes[row,col].axvline(crosshairs[col], color=crosshairs_color, lw=lw, alpha=crosshairs_alpha)
                axes[row,col].axhline(crosshairs[row], color=crosshairs_color, lw=lw, alpha=crosshairs_alpha)


    # x and y limits
    if limits is not None:
        # x-limits
        for col in range(ndim):
            axes[-1, col].set_xlim(limits[col])
        # y-limits
        for col in range(ndim):
            for row in range(col+1, ndim):
                axes[row, col].set_ylim(limits[row])

    # Bottom labels
    for col in range(ndim):
        if labels is not None:
            axes[-1, col].set_xlabel(labels[col], fontsize=fontsize)
    # Left labels
    for row in range(1,ndim):
        if labels is not None:
            axes[row, 0].set_ylabel(labels[row], fontsize=fontsize)

    # Ticks
    formatters = [ticker.FormatStrFormatter(rf'$%.{max(dec-1,1)}f$') for dec in decimals]
    for i in range(ndim):
        xlim = axes[-1, i].get_xlim()
        ticks = nice_ticks(xlim, n_ticks[i]) if n_ticks[i] is not None else None
        if ticks is not None:
            axes[-1, i].set_xticks(ticks)
            if i >= 1:
                axes[i, 0].set_yticks(ticks)
        if decimals[i] >= 5:
            axes[-1, i].xaxis.set_major_formatter(formatters[i])
            axes[i, 0].yaxis.set_major_formatter(formatters[i])

    # Ticklabel rotation, alignment, fontsize
    for i in range(ndim):
        ax = axes[-1,i]
        [l.set_fontsize(fontsize) for l in ax.get_xticklabels()]
        [l.set_rotation(xticklabel_rotation) for l in ax.get_xticklabels()]
        [l.set_horizontalalignment('right') for l in ax.get_xticklabels()]
    for i in range(1,ndim):
        ax = axes[i,0]
        [l.set_fontsize(fontsize) for l in ax.get_yticklabels()]


    # This is just for debugging/temporary use.
    # Print the row and column in each square.
    if printrowcol:
        for row in range(ndim):
            for col in range(row+1):
                ax = axes[row,col]
                ax.text(0.1, 0.8, f'({row},{col})', transform=ax.transAxes)

    # Make sure the axis limits are the same
    for row in range(1,ndim):
        bottom, top = axes[row, 0].get_ylim()
        for col in range(1,row):
            bottom2, top2 = axes[row, col].get_ylim()
            try:
                assert(np.abs(bottom2-bottom) < 1e-14)
                assert(np.abs(top2-top) < 1e-14)
            except AssertionError:
                raise ValueError(f'Row {row} has inconsistent y-limits')

    # Adjust plot
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.align_labels()
    if return_axes:
        return (fig, axes)
    else:
        return fig


def compute_sigma_levels(sigmas):
    return 1.0 - np.exp(-0.5 * np.array(sigmas) ** 2)

def contour_levels(grid, levels=compute_sigma_levels([1.0, 2.0])):
    """Compute contour levels for a gridded 2D posterior"""
    sorted_ = np.flipud(np.sort(grid.ravel()))
    pct = np.cumsum(sorted_) / np.sum(sorted_)
    cutoffs = np.searchsorted(pct, np.array(levels))
    return np.sort(sorted_[cutoffs])

def weighted_quantile(values, quantiles, weights=None,
                      values_sorted=False, old_style=False):
    """
    Very close to numpy.percentile, but supports weights.
    :param values: numpy.array with data
    :param weights: array-like of the same length as `values'
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if weights is None:
        return np.quantile(values, quantiles)
    weights = np.array(weights)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]

    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(weights)
    return np.interp(quantiles, weighted_quantiles, values)

def automatic_figsize(ndim):
    base = 4.8
    size = base + base * (ndim // 8)
    return (size, size)

def determine_num_decimals(x, n_uncertainty_digits, weights):
    n_extra_digits = n_uncertainty_digits - 1
    low, median, high = low_median_high(x, [0.16, 0.84], weights)
    decimals_low = -int(np.floor(np.log10(np.abs(low)))) + n_extra_digits
    decimals_high = -int(np.floor(np.log10(np.abs(high)))) + n_extra_digits
    return max(decimals_low, decimals_high)

def low_median_high(x, quantiles, weights):
    median = weighted_quantile(x, [0.5], weights=weights)[0]
    tmp = weighted_quantile(x, quantiles, weights=weights)
    low = tmp[0] - median
    high = tmp[1] - median
    return (low, median, high)

def float_to_leading_integers(val, n_dig):
    '''
    Discards any decimal points and leading zeros
    and returns an integer with n_dig digits.
    '''
    exp = decimal.Decimal(val).adjusted()
    leading = np.abs(val/10**(exp-(n_dig-1)))
    return decimal.Decimal(leading).to_integral_value()

def diagonal_title(label, error_style, mid, low, high, n_dec, n_uncertainty_digits):
    fmt = f'.{max(n_dec,0)}f'
    mid = np.around(mid, n_dec)
    mid = f'{mid:{fmt}}'
    label = label + '\n' + rf'${mid}$' if label is not None else ''
    if error_style == 'parenthesis':
        low = float_to_leading_integers(low, n_uncertainty_digits)
        high = float_to_leading_integers(high, n_uncertainty_digits)
        if n_dec < 0:
            low *= 10**(-n_dec)
            high *= 10**(-n_dec)
        label += rf'$(_{{{low}}}^{{{high}}})$'
    elif error_style == 'plusminus':
        low = np.around(low, n_dec)
        low = f'{low:{fmt}}'
        high = np.around(high, n_dec)
        high = f'+{high:{fmt}}'
        label += rf'$_{{{low}}}^{{{high}}}$'
    elif error_style is None:
        pass
    else:
        warnings.warn(f"Unknown error_style \'{error_style}\', ignoring")
    return label

def nice_ticks(lim, n):
    '''
    Find suitable tick locations.
    The builtins (MaxNLocator etc.) are not suitable here.
    '''
    # Total length, division points, division size, division midpoints
    diff = lim[1] - lim[0]
    div_size = diff / (n+1)
    div_points = []
    for i in range(1, n+1):
        div_points.append(lim[0]+i*div_size)

    # Compute minimum number of decimals needed
    # Negative decimals means we round integer numbers
    n_dec = -int(np.floor(np.log10(np.abs(diff))))

    # Find nice ticks
    ticks = None
    ok = False
    counter = -1
    while not ok:
        counter += 1
        if counter > 3:
            print('Unable to find suitable ticks, using whatever matplotlib decides')
            return None
        tmp = n_dec + counter
        ticks = [np.around(div_points[i], decimals=tmp) for i in range(n)]
        margin = (div_points[1]-div_points[0]) * 0.15
        all_ok = True
        for i in range(len(ticks)):
            low = div_points[i]-margin
            high = div_points[i]+margin
            if ticks[i] < low or ticks[i] > high:
                all_ok = False
        ok = all_ok
    n_dec = n_dec + counter

    # Make sure the ticks are evenly spaced.
    # If not, increase number of decimals by one and take averages.
    uneven = False
    for i in range(2, n, 2):
        tmp1 = ticks[i-2]
        tmp2 = ticks[i]
        midtick = ticks[i-2] + (ticks[i]-ticks[i-2])/2
        rounded_midtick = np.around(midtick, decimals=n_dec+1)
        if rounded_midtick != ticks[i-1]:
            uneven = True
        if uneven:
            ticks[i-1] = rounded_midtick
    return ticks

def plot_joint_distribution(ax, x1, x2, bins, cmap, levels, weights, lw, limits=None, n_contourf_levels=30):
    hist, xedges, yedges = np.histogram2d(x1, x2, bins=bins, weights=weights, range=limits)
    hist = hist.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    locator = ticker.MaxNLocator(n_contourf_levels, min_n_ticks=1)
    lev = locator.tick_values(np.min(hist), np.max(hist))
    lev[0] = lev[0] if lev[0] > 0 else 1
    # This is a confusing bit of code, I assume contourf will fail.
    # If it doesn't, I set scatter_fallback to False and continue by calling contour.
    # Otherwise contour would still be called even if contourf failed, which doesn't make much sense.
    # TODO reimplement using a loop instead of recursion.
    scatter_fallback = True
    try:
        ax.contourf(hist, extent=extent, cmap=cmap, levels=lev, extend='max')
        scatter_fallback = False
    except ValueError:
        # ax.contourf failed
        #warnings.warn(f'contourf failed, decreasing n_contourf_levels to {n_contourf_levels}')
        n_contourf_levels -= 1
        if n_contourf_levels > 5:
            plot_joint_distribution(ax, x1, x2, bins, cmap, levels, weights, lw, limits=limits, n_contourf_levels=n_contourf_levels)
        else:
            warnings.warn('Could not compute contourf levels, falling back to scatter plot')
            plot_joint_scatter(ax, x1, x2, 'gray', weights)
    tmp = contour_levels(hist, levels)
    if levels is not None and scatter_fallback is False:
        try:
            ax.contour(hist, extent=extent, colors='xkcd:charcoal gray', linewidths=lw, levels=tmp, alpha=0.5)
        except ValueError:
            warnings.warn('Could not compute increasing contour levels, omitting contours')

def plot_joint_scatter(ax, x1, x2, color, weights):
    ax.scatter(x1, x2, color=color, marker='.', alpha=0.2)
    if weights is not None:
        warnings.warn('The specified weights will be disregarded for scatter plots!')

# Nicer warning messages
def format_warning(message, category, filename, lineno, file=None, line=None):
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'
