import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_tick


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    return unique


def off_margin_ticks(*axs, axis='x'):
    # Turn off tick visibility for the measure axis on the marginal plots
    f = f"get_{axis}ticklabels"

    for ax in axs:
        plt.setp(getattr(ax, f)(), visible=False)
        plt.setp(getattr(ax, f)(minor=True), visible=False)


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def add_bar_value_labels(ax, spacing=5, format='.2f', **kwargs):
    """Add labels to the end of each bar in a bar chart.
    https://stackoverflow.com/a/48372659
    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
        format (str): format of the value of the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = ("{:" + format + "}").format(y_value)

        # Create annotation
        ax.annotate(
            label,                       # Use `label` as label
            (x_value, y_value),          # Place label at end of the bar
            xytext=(0, space),           # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha='center',                 # Horizontally center label
            va=va,                       # Vertically align label differently for
            **kwargs)                    # positive and negative values.


def annotate_brackets(ax, num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, same_h=True, fs=None):
    """
    https://stackoverflow.com/a/52333561

    Annotate plot (barplot, boxplot, ...) with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param same_h: if the bracket edges should have the same length or not (if not the ends get close to the bars)
    :param fs: font size
    """

    text = f"p = {data:.3f}" if data >= 0.001 else f"p < 0.001"

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    dh *= max(ly, ry)
    barh *= max(ly, ry)

    if same_h:
        y = max(ly, ry) + dh
        y0, y1 = y, y

        barhl = {0: barh, 1: barh}
    else:
        y0 = ly + dh
        y1 = ry + dh

        barhl = dict.fromkeys([0, 1])
        if y0 > y1:
            barhl[1] = y0 + barh - y1
            barhl[0] = barh
        else:
            barhl[0] = y1 + barh - y0
            barhl[1] = barh

    barx = [lx, lx, rx, rx]
    bary = [y0, y0+barhl[0], y1+barhl[1], y1]
    mid = ((lx+rx)/2, max(y0, y1)+barh)

    ax.plot(barx, bary, c='black')
    # ax.plot([lx + 0.001, rx - 0.001], [max(y0, y1), max(y0, y1)], c='black', ls='--', lw=0.6)

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    return ax.text(*mid, text, **kwargs)


def hierarchical_labels(ax, axis="x", tick_pos=None, label_sep='-', offset=-0.1):
    if axis == "x":
        twin = "twiny"
        _axis = "xaxis"
        spine = "bottom"
        ticks = "xticklabels"
        offset = -abs(offset)
    else:
        twin = "twinx"
        _axis = "yaxis"
        spine = "left"
        ticks = "yticklabels"
        offset = abs(offset)

    labels = getattr(ax, f"get_{ticks}")()
    levels = list(zip(*map(lambda x: x.get_text().split(label_sep), labels)))[::-1]
    tick_pos = [np.linspace(0, 1, len(l) + 1) for l in levels] if tick_pos is None else tick_pos
    for level, (level_labels, tick_p) in enumerate(zip(levels, tick_pos)):
        twin_ax = getattr(ax, twin)() if level > 0 else ax

        twin_ax.spines[spine].set_position(("axes", offset * (level + 1)))
        twin_ax.tick_params('both', length=0, width=0, which='minor')
        twin_ax.tick_params('both', direction='in', which='major')
        getattr(twin_ax, _axis).set_ticks_position(spine)
        getattr(twin_ax, _axis).set_label_position(spine)

        twin_ax.set_xticks(tick_p)
        getattr(twin_ax, _axis).set_major_formatter(mpl_tick.NullFormatter())
        loc = [np.mean(tick_p[i - 1:i + 1]) for i in range(1, len(tick_p))]
        getattr(twin_ax, _axis).set_minor_locator(mpl_tick.FixedLocator(loc))
        getattr(twin_ax, _axis).set_minor_formatter(mpl_tick.FixedFormatter(level_labels))
