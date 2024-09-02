from src.data_loading import load_data
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import submitit
from functools import partial
from itertools import product
import time
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from src.openai_api_utils import AnswerClassifier
from src.data_loading import load_data
from src.utils import FixedSizeSplit, extract_high_cardinality_features



# datasets = ["journal_jcr_cls", "movies", "michelin", "spotify", "employee_salary", "museums", "fifa_footballplayers_22", "jp_anime", "clear_corpus", "company_employees", "us_presidential", "us_accidents_severity", "us_accidents_counts", "wine_review"]
# datasets.extend(["building_permits", "public", "kickstarter", "colleges", "medical_charge", "traffic_violations"]) #  "agora"
# datasets.extend(["bikewale", "goodreads", "zomato", "coffee_fix", "nfl_contract", "employee-remuneration-and-expenses-earning-over-75000", "coffee_analysis", "ramen_ratings", "beer_profile_and_ratings", "adult"])
datasets = ['bikewale', 'clear_corpus', 'company_employees',
       'employee-remuneration-and-expenses-earning-over-75000',
       'employee_salary', 'goodreads', 'journal_jcr_cls', 'ramen_ratings',
       'spotify', 'us_accidents_counts', 'us_accidents_severity']
       #'us_presidential', 'wine_review', 'zomato']
# new_datasets = ['prod',
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

def run_llm_icl(X, y, cv, test_batch_size, n_ensemble):
    accs = []
    balanced_accs = []
    for train_idx, test_idx in cv.split(X):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        # Define the estimator and run fit/predict
        estimator = AnswerClassifier(temperature=0.1, n_ensembles=n_ensemble, test_batch_size=test_batch_size) # CARTERegressor for Regression
        estimator.fit(X=X_train, y=y_train)
        y_pred = estimator.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\nThe accuracy for LLM ICL:", "{:.4f}".format(acc))
        accs.append(acc)

        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        print(f"\nThe balanced accuracy for LLM ICL:", "{:.4f}".format(balanced_acc))
        balanced_accs.append(balanced_acc)

    print(f"\nThe mean accuracy for LLM ICL:", "{:.4f}".format(np.mean(accs)))
    print(f"\nThe mean balanced accuracy for LLM ICL:", "{:.4f}".format(np.mean(balanced_accs)))
    res = {}
    res["accuracy"] = accs
    res["balanced_accuracy"] = balanced_accs
    print("returning ", res)
    return res





#print("Number of iterations: ", len(datasets) * len(encodings) * len(dim_reductions) * len(models))


def pipeline(config):#dataset, encoding, n_test, dim_reduction_name, model_name, n_train, features
    print(config)
    dataset, n_test, n_train, features, test_batch_size, n_ensemble = config
    
    X, y = load_data(dataset, max_rows=10000)
    if len(X) < n_train + n_test:
        return (n_train, features, None)
    cv = FixedSizeSplit(n_splits=7, n_train=n_train, n_test=n_test, random_state=42)
    if features == "all":
        res_scores = run_llm_icl(X, y, cv, test_batch_size, n_ensemble)
        res_scores["n_train"] = n_train
        res_scores["n_test"] = n_test
        res_scores["features"] = features
        res_scores["dataset"] = dataset
        return (n_train, features, res_scores)
    elif features == "text_only":
        X_text, X_rest = extract_high_cardinality_features(X, dataset_name=dataset, override_cache=False, cardinality_threshold=30, fail_if_not_cached=True)
        res_scores = run_llm_icl(X_text, y, cv, test_batch_size, n_ensemble)
        res_scores["n_train"] = n_train
        res_scores["n_test"] = n_test
        res_scores["features"] = features
        res_scores["dataset"] = dataset
        return (n_train, features, res_scores)
        #return (n_train, features, time_limit, preset, run_on_encoded_data(X_enc, None, y, dim_reduction_name, dim_reduction, model_name, model, encoding, cv, dataset=dataset, features=features))
    elif features == "rest_only":
        raise Exception("Not implemented")
        #return (n_train, features, run_on_encoded_data(None, X_rest, y, dim_reduction_name, dim_reduction, model_name, model, encoding, cv, dataset=dataset, features=features))

n_trains = [64, 128]#, 2000]#, 3000, 4000, 5000]
features_list = ["all", "text_only"]#, "rest_only"]
test_batch_size_list = [6]
n_ensemble_list = [4]
n_test = 200

# Generate all combinations of parameters
param_combinations = list(product(datasets, [n_test], n_trains, features_list, test_batch_size_list, n_ensemble_list))

# Chunk your jobs
CHUNK_SIZE = 500  # Choose a suitable chunk size
chunks = [param_combinations[i:i + CHUNK_SIZE] for i in range(0, len(param_combinations), CHUNK_SIZE)]

array_parallelism_total = 4
array_parallelism = array_parallelism_total // len(chunks)
print(f"Using {array_parallelism} array parallelism")

jobs = []

executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(timeout_min=2000, slurm_partition='parietal,gpu,gpu-best', slurm_array_parallelism=array_parallelism, cpus_per_task=16, mem_gb=64)
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
name = "results_llm_31_08"
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

