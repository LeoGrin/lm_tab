from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from skrub import MinHashEncoder, TableVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import submitit
from src.data_loading import load_data
from src.utils import FixedSizeSplit
from itertools import product
import time
import os

def run_catboost(X, y, cv, features, **kwargs):
    """
    X_text: np array of shape (n_samples, 1), the text feature
    X_rest: np array of shape (n_samples, n_features), additional tabular data
    y: np array of shape (n_samples,), the classifcation target
    dim_reduction_name: str, the name of the dim reduction method
    dim_reduction: sklearn transformer, the dim reduction method
    model_name: str, the name of the model
    model: sklearn model, the model
    encoding: str, the name of the encoding which was used to create X_enc
    cv: sklearn cross validator, the cross validator to use
    """
    tb = TableVectorizer(cardinality_threshold=30, #TODO: don't hardcode
                        high_card_cat_transformer = "passthrough",
                        low_card_cat_transformer = "passthrough",
                        numerical_transformer = "passthrough",
                        datetime_transformer = "passthrough",
    ) #just to get the high cardinality columns
    tb.fit(X)
    # get high cardinality columns
    high_cardinality_columns = []
    low_cardinality_columns = []
    for name, trans, cols in tb.transformers_:
        print(name, cols)
        if "high" in name:
            high_cardinality_columns.extend(cols)
        elif "low" in name:
            low_cardinality_columns.extend(cols)
    print("Low cardinality columns", low_cardinality_columns)
    print("High cardinality columns", high_cardinality_columns)
    

    model = CatBoostClassifier( tokenizers = [{
        "tokenizer_id" : "Space",
        "separator_type" : "ByDelimiter",
        "delimiter" : ""
    }],

    dictionaries = [{
        "dictionary_id" : "BiGram",
        "max_dictionary_size" : "50000",
        "occurrence_lower_bound" : "3",
        "gram_order" : "2"
    }, {
        "dictionary_id" : "Word",
        "max_dictionary_size" : "50000",
        "occurrence_lower_bound" : "3",
        "gram_order" : "1"
    }])

    if features == "text_only":
        X = X.drop(columns=[col for col in X.columns if (col not in high_cardinality_columns)])
    else:
        assert features == "all"
    
    # do cv by hand
    accuracies = []
    roc_aucs = []
    #TODO, maybe I can specify in the model and use sklearn cross_val score
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train, cat_features=low_cardinality_columns, text_features=high_cardinality_columns)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        accuracies.append(accuracy)
        roc_aucs.append(roc_auc)

    return {
        'encoding': "catboost",
        'dim_reduction': "none",
        'model': "catboost",
        'accuracies': accuracies,
        'roc_auc': roc_aucs,
        'n_train': cv.n_train,
        'n_test': cv.n_test,
        **kwargs
    }

def pipeline(config):#dataset, encoding, n_test, dim_reduction_name, model_name, n_train, features):
    print(config)
    dataset, n_test, n_train, features = config

    X, y = load_data(dataset, max_rows=10000)
    cv = FixedSizeSplit(n_splits=7, n_train=n_train, n_test=n_test, random_state=42)
    return run_catboost(X, y, cv, dataset=dataset, features=features)



#print("Number of iterations: ", len(datasets) * len(encodings) * len(dim_reductions) * len(models))
jobs = []

executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(timeout_min=100, slurm_partition='parietal,normal', slurm_array_parallelism=40, cpus_per_task=2,
                            exclude="margpu009")
# change name of job
executor.update_parameters(name="catboost")

datasets = ['bikewale', 'clear_corpus', 'company_employees',
       'employee-remuneration-and-expenses-earning-over-75000',
       'employee_salary', 'goodreads', 'journal_jcr_cls', 'ramen_ratings',
       'spotify', 'us_accidents_counts', 'us_accidents_severity',
       'us_presidential', 'wine_review', 'zomato']

n_trains = [500, 1000, 2000, 3000, 4000, 5000]
features_list = ["all", "text_only"]
n_test = 500

# Generate all combinations of parameters
param_combinations = list(product(datasets, [n_test], n_trains, features_list))

# Chunk your jobs
CHUNK_SIZE = 500  # Choose a suitable chunk size
chunks = [param_combinations[i:i + CHUNK_SIZE] for i in range(0, len(param_combinations), CHUNK_SIZE)]

jobs = []
# Submit jobs chunk by chunk

for i, chunk in enumerate(chunks):
    while True:
        try:
            jobs.extend(executor.map_array(pipeline, chunk))
            break
        except Exception as e:
            print(e)
            print("Sleeping 200 seconds")
            time.sleep(200)
    print(f"Submitted chunk {i+1} of {len(chunks)}, {len(jobs)} jobs to the cluster.")

import pandas as pd

# Define the columns of your dataframe
# Open a file to write the results
name = "results_catboost_21_11"
for job in jobs:
    try:
        result = job.result()
        if result is not None:
            res = result
            # Flatten
            #print(result)
            #result = [item for sublist in result for item in sublist if item is not None]
            # Remove element if it is (_, _, None)
            #print(result)
            # Merge with n_train, features
            #result = [{"n_train": n_train, "features": features, **r} for n_train, features, r in result]
            print(res)
            df = pd.DataFrame(res)
            if len(df):
                # Explode 'accuracies' and 'roc_auc'
                df = df.explode(['accuracies', "roc_auc"])
                # check if the file exists
                if not os.path.isfile(f"../results/{name}.csv"):
                    # Create a new file
                    df.to_csv(f"../results/{name}.csv", index=False)
                else:
                    # Append to the file
                    df.to_csv(f"../results/{name}.csv", mode='a', header=False, index=False)
    except Exception as e:
        print(f"Job {job.job_id} failed with exception: {e}")