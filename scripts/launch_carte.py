from src.data_loading import load_data
from skrub import MinHashEncoder
from sklearn.decomposition import PCA
from src.utils import FeaturesExtractor, FixedSizeSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from tabpfn import TabPFNClassifier
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import submitit
from functools import partial
from itertools import product
import time
# Add the path to the other project
#import sys
#sys.path.append('/scratch/lgrinszt/carte')
#from src_carte.carte_table_to_graph import Table2GraphTransformer
#from src_carte.carte_estimator import CARTERegressor, CARTEClassifier
#from configs.directory import config_directory

from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score



# datasets = ["journal_jcr_cls", "movies", "michelin", "spotify", "employee_salary", "museums", "fifa_footballplayers_22", "jp_anime", "clear_corpus", "company_employees", "us_presidential", "us_accidents_severity", "us_accidents_counts", "wine_review"]
# datasets.extend(["building_permits", "public", "kickstarter", "colleges", "medical_charge", "traffic_violations"]) #  "agora"
# datasets.extend(["bikewale", "goodreads", "zomato", "coffee_fix", "nfl_contract", "employee-remuneration-and-expenses-earning-over-75000", "coffee_analysis", "ramen_ratings", "beer_profile_and_ratings", "adult"])
# datasets = ['bikewale', 'clear_corpus', 'company_employees',
#        'employee-remuneration-and-expenses-earning-over-75000',
#        'employee_salary', 'goodreads', 'journal_jcr_cls', 'ramen_ratings',
#        'spotify', 'us_accidents_counts', 'us_accidents_severity',
#        'us_presidential', 'wine_review', 'zomato']
# datasets = ['prod',
#  'airbnb',
#  'channel',
#  'wine',
#  'imdb',
#  'jigsaw',
#  'fake',
#  'kick',
#  'ae',
#  'qaa',
#  'qaq',
#  'cloth',
#  'mercari',
#  'jc',
#  'pop',
#  'book',
#  'salary',
#  'house'
# ]
# datasets = new_datasets + datasets
# datasets = ['wine_review',
#  'prod',
#  'airbnb',
#  'channel',
#  'wine',
#  'ae',
#  'qaa',
#  'qaq',
#  'cloth',
#  'salary']
datasets = ["bikewale", "employee-remuneration-and-expenses-earning-over-75000"]
#datasets = [f"companies_{year}" for year in range(2012, 2024)]

#datasets = ["drug_directory", "met_objects"] #TODO
# datasets = ['michelin',
#  'colleges',
#  'goodreads',
#  'coffee_fix',
#  'coffee_analysis',
#  'ramen_ratings']
print(len(datasets))
#datasets = ["agora"]

def run_carte(X, y, cv):
    import sys
    sys.path.append('/scratch/lgrinszt/carte')
    from src_carte.carte_table_to_graph import Table2GraphTransformer
    from src_carte.carte_estimator import CARTERegressor, CARTEClassifier
    from configs.directory import config_directory

    # Define some parameters
    fixed_params = dict()
    fixed_params["num_model"] = 10 # 10 models for the bagging strategy
    fixed_params["disable_pbar"] = False # True if you want cleanness
    fixed_params["random_state"] = 0
    fixed_params["device"] = "cpu"
    fixed_params["n_jobs"] = 10

    accs = []
    roc_aucs = []
    balanced_accs = []
    for train_idx, test_idx in cv.split(X):
        preprocessor = Table2GraphTransformer()
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]
        X_train = preprocessor.fit_transform(X_train, y=y_train)
        X_test = preprocessor.transform(X_test)

        is_binary = len(np.unique(y)) == 2

        # Define the estimator and run fit/predict
        estimator = CARTEClassifier(**fixed_params,
        loss="binary_crossentropy" if is_binary else "categorical_crossentropy") # CARTERegressor for Regression
        estimator.fit(X=X_train, y=y_train)
        y_pred_proba = estimator.predict_proba(X_test)
        y_pred = y_pred_proba > 0.5

        # Obtain the r2 score on predictions
        try:
            score = roc_auc_score(y_test, y_pred_proba)
            print(f"\nThe AUROC for CARTE:", "{:.4f}".format(score))
            roc_aucs.append(score)
        except Exception as e:
            print(f"Error computing AUROC: {e}")
            roc_aucs.append(None)

        acc = accuracy_score(y_test, y_pred)
        print(f"\nThe accuracy for CARTE:", "{:.4f}".format(acc))
        accs.append(acc)

        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        print(f"\nThe balanced accuracy for CARTE:", "{:.4f}".format(balanced_acc))
        balanced_accs.append(balanced_acc)

    print(f"\nThe mean accuracy for CARTE:", "{:.4f}".format(np.mean(accs)))
    print(f"\nThe mean balanced accuracy for CARTE:", "{:.4f}".format(np.mean(balanced_accs)))
    res = {}
    if roc_aucs and roc_aucs[0] is not None:
        res["roc_auc"] = roc_aucs
    res["accuracy"] = accs
    res["balanced_accuracy"] = balanced_accs
    print("returning ", res)
    return res





#print("Number of iterations: ", len(datasets) * len(encodings) * len(dim_reductions) * len(models))


def pipeline(config):#dataset, encoding, n_test, dim_reduction_name, model_name, n_train, features
    print(config)
    dataset, n_test, n_train, features = config
    
    X, y = load_data(dataset, max_rows=10000)
    if len(X) < n_train + n_test:
        return (n_train, features, None)
    cv = FixedSizeSplit(n_splits=7, n_train=n_train, n_test=n_test, random_state=42)
    if features == "all":
        res_scores = run_carte(X, y, cv)
        res_scores["n_train"] = n_train
        res_scores["n_test"] = n_test
        res_scores["features"] = features
        res_scores["dataset"] = dataset
        return (n_train, features, res_scores)
    elif features == "text_only":
        raise Exception("Not implemented")
        #return (n_train, features, time_limit, preset, run_on_encoded_data(X_enc, None, y, dim_reduction_name, dim_reduction, model_name, model, encoding, cv, dataset=dataset, features=features))
    elif features == "rest_only":
        raise Exception("Not implemented")
        #return (n_train, features, run_on_encoded_data(None, X_rest, y, dim_reduction_name, dim_reduction, model_name, model, encoding, cv, dataset=dataset, features=features))

n_trains = [64, 128, 256, 1000, 2000, 3000, 4000, 5000]#[500, 1000, 2000, 3000]#, 4000, 5000]
features_list = ["all"]#, "rest_only"]
n_test = 500

# Generate all combinations of parameters
param_combinations = list(product(datasets, [n_test], n_trains, features_list))

# Chunk your jobs
CHUNK_SIZE = 500  # Choose a suitable chunk size
chunks = [param_combinations[i:i + CHUNK_SIZE] for i in range(0, len(param_combinations), CHUNK_SIZE)]

array_parallelism_total = 50
array_parallelism = array_parallelism_total // len(chunks)
print(f"Using {array_parallelism} array parallelism")

jobs = []

executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(timeout_min=2000, slurm_partition='parietal,normal', slurm_array_parallelism=array_parallelism, cpus_per_task=16, mem_gb=64)
# change name of job
executor.update_parameters(name="pipeline")
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
name = "results_carte_01_09"
for job in jobs:
    try:
        print("retrieving result")
        result = job.result()
        print("retrieved")
        print(result)
        if result is not None:
            if result[2] is None:
                continue
            res = result[2]
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
                #df = df.explode(['accuracies', "roc_auc"])
                # explode columns with scores
                columns = list(df.columns)
                columns_scores = [col for col in columns if col not in ["n_train", "n_test" "features", "dataset"]]
                # check if the file exists
                if not os.path.isfile(f"../results/{name}.csv"):
                    # Create a new file
                    df.to_csv(f"../results/{name}.csv", index=False)
                else:
                    # Append to the file
                    df.to_csv(f"../results/{name}.csv", mode='a', header=False, index=False)
    except Exception as e:
        print(f"Job {job.job_id} failed with exception: {e}")

