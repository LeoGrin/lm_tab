from src.data_loading import load_data
from skrub import TableVectorizer, MinHashEncoder
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.base import clone
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, TruncatedSVD
from src.utils import FixedSizeSplit
import submitit
from itertools import product
import time
import os
import pandas as pd

def run_with_hv(dataset, analyzer, ngram_range, dim_reduction_name, dim_reduction, cv, model_name, model, features, cardinality_threshold=30, tf_idf=False, **kwargs):
    X, y = load_data(dataset, max_rows=10000)
    tb = TableVectorizer(cardinality_threshold=cardinality_threshold,
                        high_card_cat_transformer = "passthrough",
                        low_card_cat_transformer = "passthrough",
                        numerical_transformer = "passthrough",
                        datetime_transformer = "passthrough",
    ) #just to get the high cardinality columns
    tb.fit(X)
    # get high cardinality columns
    high_cardinality_columns = []
    for name, trans, cols in tb.transformers_:
        if "high" in name:
            high_cardinality_columns.extend(cols)
            break
    print("High cardinality columns", high_cardinality_columns)
    all_enc_cols = []
    for col in high_cardinality_columns:
        if tf_idf:
            vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
        else:
            vectorizer = HashingVectorizer(analyzer=analyzer, ngram_range=ngram_range)
        res = vectorizer.fit_transform(X[col])
        all_enc_cols.append(res)

    X_rest = X.drop(high_cardinality_columns, axis=1)

    rest_trans = TableVectorizer(high_card_cat_transformer = MinHashEncoder(n_components=10, analyzer='char'),
                            cardinality_threshold=30)
    # cv by hand
    # split X
    accuracies = []
    roc_aucs = []
    for i, (train, test) in enumerate(cv.split(X, y)):
        print(f"Fold {i}")
        X_rest_train = rest_trans.fit_transform(X_rest.iloc[train])
        X_rest_test = rest_trans.transform(X_rest.iloc[test])
        X_high_train, X_high_test = [], []
        for j, col in enumerate(high_cardinality_columns):
            dim_rec = clone(dim_reduction)
            X_high_train.append(dim_rec.fit_transform(all_enc_cols[j][train]))
            X_high_test.append(dim_rec.transform(all_enc_cols[j][test]))
        X_high_train = np.concatenate(X_high_train, axis=1)
        X_high_test = np.concatenate(X_high_test, axis=1)
        if features == "all":
            X_train = np.concatenate([X_rest_train, X_high_train], axis=1)
            X_test = np.concatenate([X_rest_test, X_high_test], axis=1)
        elif features == "text_only":
            X_train = X_high_train
            X_test = X_high_test
        else:
            raise NotImplementedError()
        print("X_train shape", X_train.shape)
        print("X_test shape", X_test.shape)
        model.fit(X_train, y[train])
        y_pred = model.predict(X_test)
        # accuracy and roc_auc
        accuracy = accuracy_score(y[test], y_pred)
        roc_auc = roc_auc_score(y[test], y_pred)
        accuracies.append(accuracy)
        roc_aucs.append(roc_auc)
    
    return {
        "dataset": dataset,
        "analyzer": analyzer,
        "ngram_range": str(ngram_range),
        "dim_reduction_name": dim_reduction_name,
        #"dim_reduction": dim_reduction,
        "cv": cv,
        "model_name": model_name,
        #"model": model,
        "features": features,
        "accuracy": np.array(accuracies),
        "roc_auc": np.array(roc_aucs),
        "tfidf": tf_idf,
        **kwargs
    }


datasets = ['bikewale', 'clear_corpus', 'company_employees',
       'employee-remuneration-and-expenses-earning-over-75000',
       'employee_salary', 'goodreads', 'journal_jcr_cls', 'ramen_ratings',
       'spotify', 'us_accidents_counts', 'us_accidents_severity',
       'us_presidential', 'wine_review', 'zomato']
#datasets = ["drug_directory", "met_objects"] #TODO
# datasets = ['michelin',
#  'colleges',
#  'goodreads',
#  'coffee_fix',
#  'coffee_analysis',
#  'ramen_ratings']
print(len(datasets))
#datasets = ["agora"]




#print("Number of iterations: ", len(datasets) * len(encodings) * len(dim_reductions) * len(models))
jobs = []

executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(timeout_min=400, slurm_partition='parietal,normal', slurm_array_parallelism=30, cpus_per_task=2,
                            exclude="margpu009")
# change name of job
executor.update_parameters(name="hv")


def pipeline(config):
    model_dic = {"GradientBoostingClassifier": GradientBoostingClassifier()}
    dim_reduction_dic = {"SVD_30": TruncatedSVD(n_components=30)}
    dataset, analyzer, ngram_range, dim_reduction_name, model_name, features, n_train, tf_idf = config
    n_test = 500
    cv = FixedSizeSplit(n_splits=7, n_train=n_train, n_test=n_test, random_state=42)
    model = model_dic[model_name]
    dim_reduction = dim_reduction_dic[dim_reduction_name]
    return run_with_hv(dataset, analyzer, ngram_range, dim_reduction_name, dim_reduction, cv, model_name, model, features, tf_idf=tf_idf, n_train=n_train, n_test=n_test)

n_trains = [500, 1000, 2000, 3000, 4000, 5000]
features_list = ["all", "text_only"]
model_names = ["GradientBoostingClassifier"]
dim_reduction_names = ["SVD_30"]
analyzers = ["char", "word"]
ngram_ranges = [(1, 1), (1, 2), (1, 3), (2, 3), (2, 4)]
tf_idf_list = [True, False]
n_test = 500

# Generate all combinations of parameters
param_combinations = list(product(datasets, analyzers, ngram_ranges, dim_reduction_names, model_names, features_list, n_trains, tf_idf_list))

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


# Define the columns of your dataframe
# Open a file to write the results
name = "results_hashing_vectorizer_01_12"
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
            print("df done")
            if len(df):
                # Explode 'accuracies' and 'roc_auc'
                df = df.explode(['accuracy', "roc_auc"])
                print("explode done")
                # check if the file exists
                if not os.path.isfile(f"../results/{name}.csv"):
                    # Create a new file
                    df.to_csv(f"../results/{name}.csv", index=False)
                else:
                    # Append to the file
                    df.to_csv(f"../results/{name}.csv", mode='a', header=False, index=False)
    except Exception as e:
        print(f"Job {job.job_id} failed with exception: {e}")