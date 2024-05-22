import argparse
import os

import polars as pl

from config.core import DATASET_DIR, config
from utils.utils_general import listify
from utils.utils_load_data import (compound_cols_or, convert_str_col,
                                   convert_to_age, convert_to_date, get_bmi,
                                   read_file_names, read_files,
                                   remove_duplicates, select_df)


def main(extension: str) -> None:
    files = read_file_names(DATASET_DIR, extension)
    # read files & get dfs of interest
    files_read = read_files(DATASET_DIR, files)
    df_demographics = select_df(files_read, check_str="gender")
    df_main = select_df(files_read, check_str="trt_adt")

    ###################################################################
    # PREPROCESS MAIN
    ###################################################################
    # convert hbp to numeric type
    df_main = convert_str_col(df_main, col_name="medical_history_hbp")
    # date of inpatient stays - convert and get duration
    df_main = convert_to_date(df_main, cols=["date_of_admission", "date_of_discharge"])
    # get timedelta
    df_main = df_main.with_columns(
        [
            (pl.col("date_of_discharge") - pl.col("date_of_admission")).alias(
                "ips_duration"
            )
        ]
    )

    # get summary values for medical history (physical conditions),
    # medical history (mental health),
    # treatment types
    compound_cands = [
        (
            [
                "medical_history_dia",
                "medical_history_hbp",
                "medical_history_ren",
                "medical_history_tum",
            ],
            "history_phys",
        ),
        (
            ["medical_history_sud", "medical_history_anx", "medical_history_mood"],
            "history_mh",
        ),
        (["trt_adt", "trt_ssr"], "treatment_adt"),
        (["trt_anx", "trt_con", "trt_oth"], "treatment_other_med"),
    ]

    for cols, new_col_name in compound_cands:
        df_main = compound_cols_or(df_main, cols, new_col_name)

    # create columns to indicate 'conventional' or augmented
    # treatment types
    df_main = df_main.with_columns(
        [
            pl.when(
                (pl.col("treatment_adt") == 1)
                & (pl.col("trt_the") == 1)
                & (pl.col("treatment_other_med") == 0)
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("treatment_conv"),  # conventional: adt(any) + therapy
            pl.when(
                (pl.col("treatment_adt") == 1) & (pl.col("treatment_other_med") == 1)
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias(
                "treatment_aug"
            ),  # treatment augmented: ad + other meds (maybe + therapy)
            pl.when(
                (pl.col("treatment_adt") == 1)
                & (pl.col("treatment_other_med") == 0)
                & (pl.col("trt_the") == 0)
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias(
                "treatment_adt_only"
            ),  # treatment augmented: ad + other meds (maybe + therapy)
        ]
    )

    # global functioning - change during inpatient stay
    df_main = df_main.with_columns(
        [(pl.col("cgis_dis") - pl.col("cgis_adm")).alias("cgis_change")]
    )

    # symptom severity
    sum_cols = [
        sum(n)
        for n in listify(
            df_main, cols=[col for col in df_main.columns if "symptom_" in col]
        )
    ]
    df_main = df_main.with_columns([pl.Series("symptom_severity", sum_cols)])
    # add bmi
    df_main = get_bmi(df_main, "height", "weight")

    # group by patient id
    df_main_grouped = df_main.group_by("id").agg(
        [
            pl.max("gaf_lv").alias("gaf_lv_24"),
            pl.mean("ips_duration").alias("ips_duration_mean"),
            pl.max("history_phys").alias("history_phys_24"),
            pl.max("history_mh").alias("history_mh_24"),
            pl.max("treatment_adt_only").alias("treatment_adt_only_cur"),
            #  pl.max("treatment_other_med").alias("treatment_other_med_cur"),
            pl.max("treatment_conv").alias("treatment_conv_cur"),
            pl.max("treatment_aug").alias("treatment_aug_cur"),
            pl.mean("cgis_change").alias("cgis_change_mean"),
            pl.max("symptom_severity").alias("symptom_severity_cur"),
            pl.max("bmi").alias("bmi_cur"),
        ]
    )

    ###################################################################
    # PREPROCESS DEMOGRAPHICS
    ###################################################################

    df_demographics = convert_to_date(
        df_demographics, cols=["date_of_birth"], fmt="%Y-%m-%d"
    )
    # gender, race

    # omitting resident status.
    # Would be interesting if answering questions surrounding billing
    # But this is not what I am looking at here.
    df_demographics = df_demographics.with_columns(
        [
            pl.when(pl.col("gender").str.contains(r"(?i)f"))
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("gender"),
            pl.when(pl.col("race").str.contains(r"(?i)chin"))
            .then(pl.lit("Chinese"))
            .when(pl.col("race").str.contains(r"(?i)ind"))
            .then(pl.lit("Indian"))
            .otherwise(pl.col("race"))
            .alias("race"),
        ]
    )

    # add age
    df_demographics = convert_to_age(df_demographics, "date_of_birth")

    # join and write to disk
    df_out = df_main_grouped.join(
        df_demographics.select(
            pl.col(["patient_id", "gender", "race", "age_years"])
        ).rename({"patient_id": "id"}),
        on="id",
        how="inner",
    )
    ###################################################################
    # FINAL CHECKS, WRITE
    ###################################################################
    # check for duplicates
    df_out = remove_duplicates(df_out)

    df_out.write_parquet(
        os.path.join(DATASET_DIR, config["preprocessed_file_name"].data)
    )

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="enter file directory")
    parser.add_argument("extension", type=str, help="Extension type to be included")
    args = parser.parse_args()
    extension = args.extension
    main(extension=extension)
