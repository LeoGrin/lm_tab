from skrub.datasets import fetch_employee_salaries, fetch_drug_directory
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#taken from skrub
def _replace_false_missing(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Takes a DataFrame or a Series, and replaces the "false missing", that is,
    strings that designate a missing value, but do not have the corresponding
    type. We convert these strings to np.nan.
    Also replaces `None` to np.nan.
    """
    # Should not replace "missing" (the string used for imputation in
    # categorical features).
    STR_NA_VALUES = [
        "null",
        "",
        "1.#QNAN",
        "#NA",
        "nan",
        "#N/A N/A",
        "-1.#QNAN",
        "<NA>",
        "-1.#IND",
        "-nan",
        "n/a",
        "-NaN",
        "1.#IND",
        "NULL",
        "NA",
        "N/A",
        "#N/A",
        "NaN",
    ]  # taken from pandas.io.parsers (version 1.1.4)
    df = df.replace(STR_NA_VALUES + [None, "?", "..."], np.nan)
    df = df.replace(r"^\s+$", np.nan, regex=True)  # Replace whitespaces
    return df

def remove_missing_values(X, y, threshold=0.7):
    X = _replace_false_missing(X)
    """Remove columns where most values are missing, then remove any row with missing values"""
    missing_cols_mask = pd.isnull(X).mean(axis=0) > threshold
    print("Removed {} columns with missing values on {} columns".format(sum(missing_cols_mask), X.shape[1]))
    X = X.drop(X.columns[missing_cols_mask], axis=1)
    missing_rows_mask = pd.isnull(X).any(axis=1)
    print("Removed {} rows with missing values on {} rows".format(sum(missing_rows_mask), X.shape[0]))
    X = X[~missing_rows_mask]
    y = y[~missing_rows_mask]
    # reset index
    X = X.reset_index(drop=True)
    res = (X, y, missing_cols_mask, missing_rows_mask)
    return res


default_column = {"employee_salary": "employee_position_title",
                    "drug_directory": "SUBSTANCENAME",
                    #"readability": "excerpt",
                    }

skrub_functions = {"employee_salary": fetch_employee_salaries,
                     "drug_directory": fetch_drug_directory
}

def load_data(data_name, max_rows=None, include_all_columns=False, remove_missing=True):
    # load the data
    if data_name in skrub_functions.keys():
        ds = skrub_functions[data_name]()
        X, y = ds.X, ds.y
    else:
        df = pd.read_parquet("../data/{}.parquet".format(data_name)) #FIXME
        X = df.drop("target", axis=1)
        y = df["target"]
    # get the default column
    if data_name in default_column.keys():
        default_col = default_column[data_name]
    else:
        default_col = "name"

    # infer task
    task = "classification" if len(np.unique(y)) < 20 else "regression"
    print(f"Original task: {task} for {data_name}")
    if task == "regression":
        print("Converting to binary classification")
        y = y > np.median(y)

    assert len(np.unique(y)) < min(20, len(y)), "probably a problem with y"
    
    X_text = X[default_col]
    X_rest = X.drop(default_col, axis=1)
    if remove_missing:
        X_rest, y, missing_cols_mask, missing_rows_mask = remove_missing_values(X_rest, y)
        X_text = X_text[~missing_rows_mask]
        print("Removed {} rows with missing values on {} rows".format(sum(missing_rows_mask), X.shape[0]))
        print("Removed {} columns with missing values on {} columns".format(sum(missing_cols_mask), X.shape[1] - 1))
        print("New shape: {}".format(X_rest.shape))

    if max_rows is not None:
        # shuffle the data
        rng = np.random.default_rng(42)
        indices = np.arange(len(X_rest))
        rng.shuffle(indices)
        X_rest = X_rest.iloc[indices]
        y = y.iloc[indices]
        X_text = X_text.iloc[indices]
        X_rest = X_rest[:max_rows]
        y = y[:max_rows]
        X_text = X_text[:max_rows]

    # label encode the target
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = y.astype(np.int64) # for skorch

    # print shapes
    print(f"X_text shape: {X_text.shape}, X_rest shape: {X_rest.shape}, y shape: {y.shape}")

    if include_all_columns:
        return X_text, X_rest, y
    else:
        return X_text, y

