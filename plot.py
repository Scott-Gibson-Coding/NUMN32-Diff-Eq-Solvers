import matplotlib.pyplot as plt
import numpy as np

def decorate_axis(
        ax: plt.Axes,
        title = "",
        xlabel = "",
        ylabel = "",
    ):
    """
    ## Description

    Wrapper for setting up styling for the axis since it is reused in a few places.
    """

    if title:
        ax.set_title(title, fontsize=16, fontweight=700)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight=700)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight=700)

def setup_plot(
        title="",
        xlabel="",
        ylabel="",
        draw_axes_lines=False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    ## Description

    Set up plot with drawn axis lines and grid.

    :param title: Title shown above plot.
    :param xlabel: Label on x axis.
    :param ylabel: Label on y axis.
    :param draw_axes_lines: Set to True to draw the lines for x=0, y=0. Defaults
    to False.

    :returns Figure, Axes: `fig`, `ax` from `plt.subplots()`
    """
    fig, ax = plt.subplots()

    # Draw title & axis labels
    decorate_axis(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    # Draw axis lines
    if draw_axes_lines:
        ax.axhline(0, color='black', linewidth=1.2)
        ax.axvline(0, color='black', linewidth=1.2)

    # Draw light grid lines
    ax.grid(True, linestyle="--", color="gray", alpha=0.2)
    
    return fig, ax
