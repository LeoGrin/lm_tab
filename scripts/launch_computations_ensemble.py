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
from src.utils import run_on_encoded_data_ensemble
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
#encodings = ["skrub__minhash_100", "openai__"]
encodings = []
model_names = [
      #"BAAI/bge-large-en-v1.5",
#     "BAAI/bge-base-en-v1.5",
      #"llmrails/ember-v1",
    #  "thenlper/gte-large",
    #  "thenlper/gte-base",
    #  "intfloat/e5-large-v2",
    # "BAAI/bge-small-en-v1.5",
    # "hkunlp/instructor-xl",
    # "hkunlp/instructor-large",
    # "intfloat/e5-base-v2",
    # "intfloat/multilingual-e5-large",
    # "intfloat/e5-large",
    # "thenlper/gte-small",
    # "intfloat/e5-base",
    # "intfloat/e5-small-v2",
    # "hkunlp/instructor-base",
    # #"sentence-t5-xxl",
    # "intfloat/multilingual-e5-base",
    # #"XLM-3B5-embedding",
    # #"gtr-t5-xxl",
    # #"SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
    # "intfloat/e5-small",
    # "TaylorAI/gte-tiny",
    # #"gtr-t5-xl",
    # "gtr-t5-large",
    # #"XLM-0B6-embedding",
    # "intfloat/multilingual-e5-small",
    # #"sentence-t5-xl",
    # "all-mpnet-base-v2",
    # #"sgpt-bloom-7b1-msmarco",
    # "jinaai/jina-embedding-l-en-v1",
    # #"SGPT-2.7B-weightedmean-msmarco-specb-bitfit",
    # "sentence-t5-large",
    # #"MegatronBert-1B3-embedding",
    # "TaylorAI/bge-micro-v2",
    # "all-MiniLM-L12-v2",
    # "all-MiniLM-L6-v2",
    # "jinaai/jina-embedding-b-en-v1",
    # #"SGPT-1.3B-weightedmean-msmarco-specb-bitfit",
    # "gtr-t5-base",
    # "nthakur/contriever-base-msmarco",
    # "TaylorAI/bge-micro",
    # "sentence-t5-base",
    # "paraphrase-multilingual-mpnet-base-v2",
    # "Hum-Works/lodestone-base-4096-v1",
    # #"SGPT-5.8B-weightedmean-nli-bitfit",
    # "paraphrase-multilingual-MiniLM-L12-v2",
    # "msmarco-bert-co-condensor",
    # "jinaai/jina-embedding-s-en-v1"
]

# for model_name in model_names:
#     encodings.append(f"lm__{model_name}")

model_names = [
    # "EleutherAI/pythia-70m",
    # "EleutherAI/pythia-160m",
    # "EleutherAI/pythia-410m",
    # "EleutherAI/pythia-1b",
    # "EleutherAI/pythia-1.4b",
    # "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Llama-2-7b-hf",
    # "huggyllama/llama-7b"
    # "bert-base-cased",
    # "bert-large-cased",
    # "roberta-base",
    # "roberta-large",
    # "microsoft/deberta-v3-base",
    # "microsoft/deberta-v3-large"
]
for model_name in model_names:
    encodings.append(f"hf__{model_name}")




#encodings.extend(["skrub__minhash_30_word_none"])
#encodings = ["skrub__minhash_30"]


# datasets = ["journal_jcr_cls", "movies", "michelin", "spotify", "employee_salary", "museums", "fifa_footballplayers_22", "jp_anime", "clear_corpus", "company_employees", "us_presidential", "us_accidents_severity", "us_accidents_counts", "wine_review"]
# datasets.extend(["building_permits", "public", "kickstarter", "colleges", "medical_charge", "traffic_violations"]) #  "agora"
# datasets.extend(["bikewale", "goodreads", "zomato", "coffee_fix", "nfl_contract", "employee-remuneration-and-expenses-earning-over-75000", "coffee_analysis", "ramen_ratings", "beer_profile_and_ratings", "adult"])
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
executor.update_parameters(timeout_min=100, slurm_partition='parietal,normal', slurm_array_parallelism=10, cpus_per_task=2,
                            exclude="margpu009")
# change name of job
executor.update_parameters(name="pipeline")

def pipeline(config):#dataset, encoding, n_test, dim_reduction_name, model_name, n_train, features):
    print(config)
    dataset, encoding, n_test, dim_reduction_name, model_name, n_train, features, aggregation = config
    models = {"LogisticRegression": LogisticRegression(), "GradientBoostingClassifier": GradientBoostingClassifier()}
    dim_reductions = {"PCA_30": PCA(n_components=30),
                    "PCA_60": PCA(n_components=60),
                    "PCA_100": PCA(n_components=100),
                    "PCA_200": PCA(n_components=200), "passthrough": "passthrough"}
    
    dim_reduction = dim_reductions[dim_reduction_name]
    model = models[model_name]

    X, y = load_data(dataset, max_rows=10000)
    try:
        X_enc, X_rest = encode_high_cardinality_features(X, encoding, dataset_name=dataset, override_cache=False, cardinality_threshold=30, fail_if_not_cached=True)
    except:
        return
    if len(X_enc) < n_train + n_test:
        return (n_train, features, None)
    cv = FixedSizeSplit(n_splits=7, n_train=n_train, n_test=n_test, random_state=42)
    if features == "all":
        return (n_train, features, run_on_encoded_data_ensemble(X_enc, X_rest, y, dim_reduction_name, dim_reduction, 
                                                                enc_model_name=model_name, enc_model=model,
                                                                rest_model_name="GradientBoostingClassifier", rest_model=GradientBoostingClassifier(),
                                                                aggregation=aggregation,
                                                                encoding=encoding,
                                                                cv=cv, dataset=dataset, features=features))
    else:
        raise NotImplementedError()

n_trains = [500, 1000, 2000, 3000, 4000, 5000]#, 4000, 5000]
features_list = ["all"]#, "rest_only"]
aggegation_list = ["voting", "stacking"]
model_names = ["LogisticRegression"]
dim_reduction_names = ["PCA_30", "PCA_100", "passthrough"]
n_test = 500

# Generate all combinations of parameters
param_combinations = list(product(datasets, encodings, [n_test], dim_reduction_names, model_names, n_trains, features_list, aggegation_list))

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
name = "results_ensemble_30_11_decoder"
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