import matplotlib.pyplot as plt
import polars as pl
import ptitprince as pt


def make_raindrop_plot(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    hue_col=None,
    palette=None,
    figsize=(10, 10),
    orient="v",
    alpha=0.25,
) -> tuple:
    """
    Construct raindrop plots.
    """
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    fig, ax = plt.subplots(figsize=figsize)
    pt.RainCloud(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette,
        orient=orient,
        alpha=alpha,
    )
    return fig, ax


def set_defaults_figures(
    fig,
    ax,
    title=None,
    labelrotation=None,
    grid_on=True,
    labels=None,
):
    """default figure settings"""
    fig.set_dpi(1000)
    fig.patch.set_facecolor("white")
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.spines["bottom"].set_color("#412970")
    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    if grid_on:
        plt.grid(
            axis="y",
            which="major",
            color="grey",
            alpha=0.2,
            linestyle="-",
            linewidth=0.5,
        )
        ax.set_axisbelow(True)
    # ax.margins(x=0, y=0)
    # Grid Spacing
    # ax.yaxis.set_major_locator(MultipleLocator(500))
    ax.tick_params(
        axis="y",
        direction="out",
        length=2,
        width=0.6,
        color="#412970",
        labelsize=16.5,
        labelcolor="#412970",
    )
    ax.tick_params(
        axis="x",
        labelcolor="#412970",
        direction="out",
        length=3,
        width=0.7,
        colors="#412970",
        labelsize=16.5,
        labelrotation=labelrotation,
    )
    xticks, xlabels = plt.xticks()
    ax.set_xticks(xticks, xlabels, fontweight="bold")
    if labels:
        ax.set_xlabel(
            labels[0],
            fontweight="bold",
            color="#412970",
            fontsize=16.5,
            labelpad=15,
        )
        ax.set_ylabel(
            labels[1],
            fontweight="bold",
            color="#412970",
            fontsize=16.5,
            labelpad=15,
        )

    # Title
    if title:
        plt.title(title, fontsize=24, loc="center", pad=30, color="#412970")
