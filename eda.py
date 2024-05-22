import os
from typing import List

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from config.core import DATASET_DIR, FIGURES_DIR, config
from utils.utils_visualizations import make_raindrop_plot, set_defaults_figures


def melt_and_prep(df_in: pl.DataFrame, id_vars: str, subset: List[str]) -> pl.DataFrame:
    df_melt = df_in.select(pl.col(subset))
    return df_melt.melt(id_vars=id_vars)


def main() -> None:
    # load df
    df_main = pl.read_parquet(
        os.path.join(DATASET_DIR, config["preprocessed_file_name"].data)
    )

    # prepare for raindrop plots (want treatment type as hue, so melt and join)
    df_melt = melt_and_prep(
        df_main,
        id_vars="id",
        subset=[
            "id",
            "treatment_conv_cur",
            "treatment_aug_cur",
            "treatment_adt_only_cur",
        ],
    )
    #  todo: check that no one falls into more than one treatment type (could add test with assert)
    # todo: double check that lost records reprsent ids w/o any treatment (adt)
    df_melt = df_melt.filter(pl.col("value") == 1).drop("value")

    df_plot_rd = (
        df_melt.group_by("id")
        .agg(pl.max("variable").alias("treatment_type"))
        .join(df_main, on="id", how="inner")
    )

    # raindrop plots
    cols = [
        "gaf_lv_24",
        "bmi_cur",
        "age_years",
        "cgis_change_mean",
    ]
    # todo: code gender/race as num to plot as raindrop plots as below
    for col in cols:
        fig, ax = make_raindrop_plot(
            df_plot_rd,
            x_col="symptom_severity_cur",
            y_col=col,
            hue_col="treatment_type",
        )
        set_defaults_figures(fig, ax, labels=("symptom_severity_cur", col))
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, col + "_raindrop.png"))
        plt.close("all")

    cols_num = [
        "gaf_lv_24",
        "ips_duration_mean",
        "bmi_cur",
        "age_years",
        "cgis_change_mean",
        "symptom_severity_cur",
    ]
    cols_cat = [
        "history_phys_24",
        "history_mh_24",
        "treatment_adt_only_cur",
        "treatment_conv_cur",
        "treatment_aug_cur",
        "gender",
        "race",
    ]

    # pairplot of num features
    df_num = df_main.select(pl.col(cols_num))
    sns.pairplot(df_num.to_pandas())
    plt.savefig(os.path.join(FIGURES_DIR, "pairplot_num.png"))

    df_cat = df_main.select(pl.col(cols_cat))
    fig, ax = plt.subplots(figsize=(10, 10))
    for col in cols_cat:
        val_df = df_cat[col].value_counts(sort=True)
        sns.barplot(data=val_df.to_pandas(), x=col, y="count", ax=ax)
        fig.savefig(os.path.join(FIGURES_DIR, col + "_bar.png"))
        plt.cla()

    # plot predictors vs outcome
    # for categorical: group by cat, count symptom severity
    for col in df_main.select(pl.col(cols_cat)).drop("race").columns:
        val_df = (
            df_main.group_by("symptom_severity_cur")
            .agg(pl.sum(col) / df_main.shape[0])
            .sort(by=col, descending=True)
        )
        sns.barplot(data=val_df.to_pandas(), x="symptom_severity_cur", y=col, ax=ax)
        fig.savefig(os.path.join(FIGURES_DIR, col + "_symptom_severity_bar.png"))
        plt.cla()

    plt.close("all")
    return None


if __name__ == "__main__":
    main()
