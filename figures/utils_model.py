from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.model_selection import RepeatedKFold, cross_validate


def cross_validate_res(
    clf,
    figure_path: str,
    palette: List[str],
    X_train: np.array,
    y_train: np.array,
    column_names: List[str],
) -> None:
    cv_model = cross_validate(
        clf,
        X_train,
        y_train,
        cv=RepeatedKFold(n_splits=5, n_repeats=5),
        return_estimator=True,
        n_jobs=2,
        scoring="f1_weighted",
    )
    coefs = pd.DataFrame(
        [m.coef_[0] for m in cv_model["estimator"]],
        columns=column_names,
    )
    plt.figure(figsize=(9, 7))
    sns.boxplot(data=coefs, orient="h", color=palette[1], saturation=0.5)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficient size")
    plt.title("Variability of coefficient size")
    plt.subplots_adjust(left=0.3)
    plt.savefig(figure_path)
    return None


def standard_scale(df: pl.DataFrame, subset: List[str], id_col="id") -> tuple:
    df_select = df.select(pl.col(subset))
    schema = df_select.columns
    X = df_select.to_numpy()
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std
    df_scaled = pl.DataFrame(X_scaled, schema=schema).with_columns(df[id_col])
    return df.drop(subset).join(df_scaled, on=id_col, how="inner"), X_mean, X_std


def get_binary_cols(df: pl.DataFrame) -> List[str]:
    """
    Get columns with binary values
    """
    binary_cols = []
    for col in df.columns:
        if df[col].unique().sum() == 1:
            binary_cols.append(col)
        else:
            continue
    return binary_cols


def train_test_split(df: pl.DataFrame, test_size=0.3, id_col="id") -> pl.DataFrame:
    """
    Split into training and test sets
    """
    # select fraction = test_size at random
    df_test = df.sample(fraction=test_size, shuffle=True)
    df_train = df.filter(~pl.col(id_col).is_in(df_test[id_col].to_list()))
    return df_train, df_test
