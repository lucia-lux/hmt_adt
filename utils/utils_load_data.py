import os
from datetime import datetime
from typing import List

import polars as pl

from utils.utils_general import listify


def read_file_names(path: str, extension="csv") -> List[str]:
    dir_contents = os.listdir(path)
    if len(dir_contents) < 1:
        raise ValueError("Folder is empty.")
    else:
        return [f for f in dir_contents if f[-len(extension) :] == extension]


def convert_to_date(df_in: pl.DataFrame, cols=List[str], fmt="%d/%m/%y"):
    for col in cols:
        df_in = df_in.with_columns(
            [pl.col(col).str.strptime(pl.Date, format=fmt).cast(pl.Datetime).alias(col)]
        )
    return df_in


def read_files(path: str, files_in: List[str]) -> list:
    if len(files_in) < 1:
        raise ValueError("List is empty.")
    else:
        dfs = []
        for file in files_in:
            dfs.append(pl.read_csv(os.path.join(path, file)))
        return dfs


def remove_duplicates(df_in: pl.DataFrame, subset=None) -> pl.DataFrame:
    return df_in.unique(subset=subset)


def convert_to_age(df_in: pl.DataFrame, col_name: str) -> pl.DataFrame:
    # calculate age from date_of_birth and current year
    current_date = datetime.now().date()
    df_in = df_in.with_columns(
        [(current_date.year - pl.col(col_name).dt.year()).alias("age_years")]
    )
    return df_in


def compound_cols_or(
    df_in: pl.DataFrame, cols: List[str], new_col_name: str, func=any, drop_cols=False
) -> pl.DataFrame:
    # classic vs classic+
    # listify
    cols_zipped = listify(df_in, cols)
    cols_any = [func(n) * 1 for n in cols_zipped]
    df_in = df_in.with_columns([pl.Series(new_col_name, cols_any)])
    if drop_cols:
        return df_in.drop(cols)
    else:
        return df_in


def select_df(dfs_in: list, check_str: str) -> pl.DataFrame:
    df_selected = [val for val in dfs_in if any(check_str in s for s in val.columns)]
    try:
        return df_selected[0]
    except KeyError:
        print(f"No df contains columns matching condition {check_str}")


def convert_str_col(df_in: pl.DataFrame, col_name: str) -> pl.DataFrame:
    """
    Clean up col and convert to numeric type
    """
    df_in = df_in.with_columns(
        [
            pl.when(pl.col(col_name).str.contains(r"(?i)yes"))
            .then(pl.lit("1"))
            .when(pl.col(col_name).str.contains(r"(?i)No"))
            .then(pl.lit("0"))
            .otherwise(pl.col(col_name))
            .alias(col_name)
        ]
    )
    return df_in.with_columns(pl.col(col_name).cast(pl.Float64))


def get_bmi(df_in: pl.DataFrame, col_height: str, col_weight: str) -> pl.DataFrame:
    """
    calculate BMI
    """
    return df_in.with_columns(
        [(pl.col(col_weight) / (pl.col(col_height) / 100) ** 2).alias("bmi")]
    )
