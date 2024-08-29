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
from sentence_transformers import SentenceTransformer
from src.encodings import encode_high_cardinality_features
from src.utils import run_on_encoded_data, FeaturesExtractor
from skrub import TableVectorizer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.random_projection import GaussianRandomProjection
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import submitit
from functools import partial
from itertools import product
import time
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.multimodal import MultiModalPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config



# datasets = ["journal_jcr_cls", "movies", "michelin", "spotify", "employee_salary", "museums", "fifa_footballplayers_22", "jp_anime", "clear_corpus", "company_employees", "us_presidential", "us_accidents_severity", "us_accidents_counts", "wine_review"]
# datasets.extend(["building_permits", "public", "kickstarter", "colleges", "medical_charge", "traffic_violations"]) #  "agora"
# datasets.extend(["bikewale", "goodreads", "zomato", "coffee_fix", "nfl_contract", "employee-remuneration-and-expenses-earning-over-75000", "coffee_analysis", "ramen_ratings", "beer_profile_and_ratings", "adult"])
datasets = ['bikewale', 'clear_corpus', 'company_employees',
       'employee-remuneration-and-expenses-earning-over-75000',
       'employee_salary', 'goodreads', 'journal_jcr_cls', 'ramen_ratings',
       'spotify', 'us_accidents_counts', 'us_accidents_severity',
       'us_presidential', 'wine_review', 'zomato']
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

def run_autogluon(X, y, cv, time_limit=180, presets="medium_quality"):
    # Prepare the data for AutoGluon
    data = pd.DataFrame(X)
    data['target'] = y

    all_scores = []

    # Use cv to split the data and fit the model
    for train_idx, test_idx in cv.split(data):
        predictor = TabularPredictor(label='target')
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Fit the model using the training data
        predictor.fit(train_data=train_data, time_limit=time_limit, num_gpus=1,
                      presets=presets,
            hyperparameters = get_hyperparameter_config('multimodal'))
        
        # Evaluate the model using the test data
        performance = predictor.evaluate(test_data)
        print("Model performance:", performance)
        all_scores.append(performance)

    res = {}
    res["roc_auc"] = [score["roc_auc"] for score in all_scores]
    res["accuracies"] = [score["accuracy"] for score in all_scores]
    return res





#print("Number of iterations: ", len(datasets) * len(encodings) * len(dim_reductions) * len(models))


def pipeline(config):#dataset, encoding, n_test, dim_reduction_name, model_name, n_train, features
    print(config)
    dataset, n_test, n_train, features, time_limit, preset = config
    

    X, y = load_data(dataset, max_rows=10000)
    if len(X) < n_train + n_test:
        return (n_train, features, None)
    cv = FixedSizeSplit(n_splits=7, n_train=n_train, n_test=n_test, random_state=42)
    if features == "all":
        res_scores = run_autogluon(X, y, cv, time_limit, preset)
        res_scores["n_train"] = n_train
        res_scores["n_test"] = n_test
        res_scores["features"] = features
        res_scores["dataset"] = dataset
        res_scores["time_limit"] = time_limit
        res_scores["preset"] = preset
        return (n_train, features, res_scores)
    elif features == "text_only":
        raise Exception("Not implemented")
        #return (n_train, features, time_limit, preset, run_on_encoded_data(X_enc, None, y, dim_reduction_name, dim_reduction, model_name, model, encoding, cv, dataset=dataset, features=features))
    elif features == "rest_only":
        raise Exception("Not implemented")
        #return (n_train, features, run_on_encoded_data(None, X_rest, y, dim_reduction_name, dim_reduction, model_name, model, encoding, cv, dataset=dataset, features=features))

n_trains = [500, 1000, 2000]#, 3000, 4000, 5000]
time_limit = [180]
presets = ["medium_quality"]
features_list = ["all"]#, "rest_only"]
n_test = 500

# Generate all combinations of parameters
param_combinations = list(product(datasets, [n_test], n_trains, features_list, time_limit, presets))

# Chunk your jobs
CHUNK_SIZE = 500  # Choose a suitable chunk size
chunks = [param_combinations[i:i + CHUNK_SIZE] for i in range(0, len(param_combinations), CHUNK_SIZE)]

array_parallelism_total = 4
array_parallelism = array_parallelism_total // len(chunks)
print(f"Using {array_parallelism} array parallelism")

jobs = []

executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(timeout_min=2000, slurm_partition='parietal,gpu,gpu-best', slurm_array_parallelism=array_parallelism,# cpus_per_task=128,
                           gpus_per_node=1)#, mem_gb=128)
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
name = "new_results_autogluon_29_08"
for job in jobs:
    try:
        result = job.result()
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
