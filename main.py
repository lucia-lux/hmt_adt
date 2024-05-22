import os
import polars as pl
from sklearn.linear_model import LogisticRegression

from config.core import DATASET_DIR, config
from utils.utils_model import (get_binary_cols,
                               standard_scale, train_test_split
)


def main() -> None:
    # set some constants
    random_state = int(config["random_state"].data)  # set seed
    palette = config["palette"].data  # for plotting if needed

    #########################################################
    # load data
    #########################################################
    df_main = pl.read_parquet(
        os.path.join(DATASET_DIR, config["preprocessed_file_name"].data)
    )

    #########################################################
    # preprocess
    #########################################################
    # get dummies for race
    df_main = df_main.to_dummies(columns="race", drop_first=True)
    # todo: add interaction terms
    # use patsy to specify formula & add interaction terms

    # split into training - test - validate
    df_train, df_test = train_test_split(df_main)
    df_test, df_valid = train_test_split(df_test, test_size=0.5)

    # scale
    binary_cols = get_binary_cols(df_main)
    cols_to_scale = [
        c
        for c in df_main.columns
        if c not in binary_cols + ["symptom_severity_cur", "id"]
    ]

    # todo: check ips duration/dtype
    # return mean and std to scale validation/test data
    df_train_scaled, X_mean, X_std = standard_scale(df_train, cols_to_scale)

    X_train = df_train_scaled.drop(["symptom_severity_cur", "id"]).to_numpy()
    y_train = df_train_scaled["symptom_severity_cur"].to_numpy()

    #########################################################
    # model
    #########################################################
    # try logistic regression - don't want cont preds
    # todo: check coef size, validation data set
    # could reframe a as a regression problem/try alternative models
    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        class_weight="balanced",
        multi_class="auto",
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    #########################################################
    # checks/validation
    #########################################################
    # todo: obtain preds for validation data
    # plot confusion matrix
    # refine & repeat
    # once happy, repeat checks on test data
    return None


if __name__ == "__main__":
    main()
