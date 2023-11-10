from skrub.datasets import fetch_employee_salaries, fetch_drug_directory, fetch_medical_charge, fetch_road_safety, fetch_traffic_violations
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

def balance(x_train, x_test, y_train, y_test, rng):
    indices_train = [(y_train == 0), (y_train == 1)]
    max_class = np.argmax(list(map(sum, indices_train)))
    min_class = np.argmin(list(map(sum, indices_train)))
    n_samples_min_class = sum(indices_train[min_class])
    indices_max_class = rng.choice(np.where(indices_train[max_class])[0], n_samples_min_class, replace=False)
    indices_min_class = np.where(indices_train[min_class])[0]
    total_indices_train = np.concatenate((indices_max_class, indices_min_class))

    indices_test = [(y_train == 0), (y_train == 1)]
    max_class = np.argmax(list(map(sum, indices_test)))
    min_class = np.argmin(list(map(sum, indices_test)))
    n_samples_min_class = sum(indices_test[min_class])
    indices_max_class = rng.choice(np.where(indices_test[max_class])[0], n_samples_min_class, replace=False)
    indices_min_class = np.where(indices_test[min_class])[0]
    total_indices_test = np.concatenate((indices_max_class, indices_min_class))
    return x_train[total_indices_train], x_test[total_indices_test], y_train[total_indices_train], y_test[
        total_indices_test]


default_column = {"employee_salary": "employee_position_title",
                    "drug_directory": "NONPROPRIETARYNAME",
                    "medical_charge": "Provider_Name",
                    "traffic_violations": "description",
                    #"readability": "excerpt",
                    }

skrub_functions = {"employee_salary": fetch_employee_salaries,
                     "drug_directory": fetch_drug_directory,
                     "medical_charge": fetch_medical_charge,
                     "traffic_violations": fetch_traffic_violations,
}

def load_data_1_text(data_name, max_rows=None, include_all_columns=False, remove_missing=True, regression=False):
    rng = np.random.default_rng(42)
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

    X_text = X[default_col]
    X_rest = X.drop(default_col, axis=1)
    # remove missing in y
    indices = pd.isnull(y)
    y = y[~indices]
    X_text = X_text[~indices]
    X_rest = X_rest[~indices]
    if remove_missing:
        X_rest, y, missing_cols_mask, missing_rows_mask = remove_missing_values(X_rest, y)
        X_text = X_text[~missing_rows_mask]
        print("Removed {} rows with missing values on {} rows".format(sum(missing_rows_mask), X.shape[0]))
        print("Removed {} columns with missing values on {} columns".format(sum(missing_cols_mask), X.shape[1] - 1))
        print("New shape: {}".format(X_rest.shape))

    # infer task
    task = "classification" if len(np.unique(y)) <= 20 else "regression"
    print(f"Original task: {task} for {data_name}")
    if task == "regression":
        if not regression:
            print("Converting to binary classification")
            y = y > np.median(y)
    if not regression:
        # label encode the target
        le = LabelEncoder()
        y = le.fit_transform(y)
        y = y.astype(np.int64) # for skorch
        if len(np.unique(y)) > 2:
            print("More than 2 classes, converting to binary classification")

    if regression:
        # convert y to numpy
        y = y.to_numpy()
        y = y.astype(np.float32)
        


    assert (len(np.unique(y)) < min(20, len(y))) or regression, "probably a problem with y"
    if not regression:
        print("Classes", np.unique(y, return_counts=True))
    


    if max_rows is not None:
        # shuffle the data
        indices = np.arange(len(X_rest))
        rng.shuffle(indices)
        X_rest = X_rest.iloc[indices]
        y = y[indices]
        X_text = X_text.iloc[indices]
        X_rest = X_rest[:max_rows]
        y = y[:max_rows]
        X_text = X_text[:max_rows]



    # print shapes
    print(f"X_text shape: {X_text.shape}, X_rest shape: {X_rest.shape}, y shape: {y.shape}")

    if include_all_columns:
        return X_text, X_rest, y
    else:
        return X_text, y

def load_data(data_name, max_rows=None, remove_missing=True, regression=False):
    rng = np.random.default_rng(42)
    # load the data
    if data_name in skrub_functions.keys():
        ds = skrub_functions[data_name]()
        X, y = ds.X, ds.y
    else:
        df = pd.read_parquet("../data/{}.parquet".format(data_name)) #FIXME
        X = df.drop("target", axis=1)
        y = df["target"]


    # remove missing in y
    indices = pd.isnull(y)
    y = y[~indices]
    X = X[~indices]
    if remove_missing:
        X, y, missing_cols_mask, missing_rows_mask = remove_missing_values(X, y)
        print("Removed {} rows with missing values on {} rows".format(sum(missing_rows_mask), X.shape[0]))
        print("Removed {} columns with missing values on {} columns".format(sum(missing_cols_mask), X.shape[1] - 1))
        print("New shape: {}".format(X.shape))

    # infer task
    task = "classification" if len(np.unique(y)) <= 20 else "regression"
    print(f"Original task: {task} for {data_name}")
    if task == "regression":
        if not regression:
            print("Converting to binary classification")
            y = y > np.median(y)
    if not regression:
        # label encode the target
        le = LabelEncoder()
        y = le.fit_transform(y)
        y = y.astype(np.int64) # for skorch
        if len(np.unique(y)) > 2:
            print("More than 2 classes, converting to binary classification")

    if regression:
        # convert y to numpy
        y = y.to_numpy()
        y = y.astype(np.float32)
        


    assert (len(np.unique(y)) < min(20, len(y))) or regression, "probably a problem with y"
    if not regression:
        print("Classes", np.unique(y, return_counts=True))
    


    if max_rows is not None:
        # shuffle the data
        indices = np.arange(len(X))
        rng.shuffle(indices)
        X = X.iloc[indices]
        y = y[indices]
        X = X[:max_rows]
        y = y[:max_rows]



    # print shapes
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y

