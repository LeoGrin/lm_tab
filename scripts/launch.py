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
from src.encodings import encode
from src.utils import run_on_encoded_data, FeaturesExtractor
from skrub import TableVectorizer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.random_projection import GaussianRandomProjection
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import submitit
from functools import partial
import time
encodings = ["skrub__minhash_30", "lm__all-distilroberta-v1", "lm__all-mpnet-base-v2", "openai__"]

model_names = [
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "llmrails/ember-v1",
    "thenlper/gte-large",
    "thenlper/gte-base",
    "intfloat/e5-large-v2",
    "BAAI/bge-small-en-v1.5",
    "hkunlp/instructor-xl",
    "hkunlp/instructor-large",
    "intfloat/e5-base-v2",
    "intfloat/multilingual-e5-large",
    "intfloat/e5-large",
    "thenlper/gte-small",
    "intfloat/e5-base",
    "intfloat/e5-small-v2",
    "hkunlp/instructor-base",
    #"sentence-t5-xxl",
    "intfloat/multilingual-e5-base",
    #"XLM-3B5-embedding",
    #"gtr-t5-xxl",
    #"SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
    "intfloat/e5-small",
    "TaylorAI/gte-tiny",
    #"gtr-t5-xl",
    "gtr-t5-large",
    #"XLM-0B6-embedding",
    "intfloat/multilingual-e5-small",
    #"sentence-t5-xl",
    "all-mpnet-base-v2",
    #"sgpt-bloom-7b1-msmarco",
    "jinaai/jina-embedding-l-en-v1",
    #"SGPT-2.7B-weightedmean-msmarco-specb-bitfit",
    "sentence-t5-large",
    #"MegatronBert-1B3-embedding",
    "TaylorAI/bge-micro-v2",
    "all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2",
    "jinaai/jina-embedding-b-en-v1",
    #"SGPT-1.3B-weightedmean-msmarco-specb-bitfit",
    "gtr-t5-base",
    "nthakur/contriever-base-msmarco",
    "TaylorAI/bge-micro",
    "sentence-t5-base",
    "paraphrase-multilingual-mpnet-base-v2",
    "Hum-Works/lodestone-base-4096-v1",
    #"SGPT-5.8B-weightedmean-nli-bitfit",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "msmarco-bert-co-condensor",
    "jinaai/jina-embedding-s-en-v1"
]
for model_name in model_names:
    encodings.append(f"lm__{model_name}")

encodings.extend(["skrub__minhash_30_word_none", "skrub__minhash_30_tokenizer_gpt2"])

datasets = ["journal_jcr_cls", "movies", "michelin", "spotify", "employee_salary", "museums", "fifa_footballplayers_22", "jp_anime", "clear_corpus", "company_employees", "us_presidential", "us_accidents_severity", "us_accidents_counts", "wine_review"]
datasets.extend(["building_permits", "agora", "public", "kickstarter", "colleges", "medical_charge", "traffic_violations"])
#datasets = ["drug_directory", "met_objects"]
print(len(datasets))

models = {"LogisticRegression": LogisticRegression(), "GradientBoostingClassifier": GradientBoostingClassifier()}
       # "TabPFNClassifier_basic": TabPFNClassifier(device="cpu", N_ensemble_configurations=1, no_preprocess_mode=True)}

dim_reductions = {"PCA_30": PCA(n_components=30),
                  #"PCA_10": PCA(n_components=10),
                   "passthrough": "passthrough"}

print("Number of iterations: ", len(datasets) * len(encodings) * len(dim_reductions) * len(models))
jobs = []

executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(timeout_min=100, slurm_partition='parietal,normal', slurm_array_parallelism=300, cpus_per_task=2,
                            exclude="margpu009")



n_trains = [1000, 2000, 3000, 4000, 5000]
for dataset in tqdm(datasets):
    #print(f"Dataset: {dataset}, Encoding: {encoding}")
    X_text, X_rest, y = load_data(dataset, max_rows=10000, include_all_columns=True)
    for encoding in tqdm(encodings, leave=False):
        print("Dataset", dataset)
        print("Encoding", encoding)
        #if len(X_text) > n_train + n_test:
        try:
            # only if the cache exists
            res = np.load(f"cache/{dataset}_{encoding.replace('/', '_')}.npy")
            X_enc = encode(X_text, encoding, dataset_name=dataset)
        except:
            print(f"Encoding {encoding} failed for dataset {dataset}")
            continue
        for n_train in n_trains:
            n_test = 2000
            cv = FixedSizeSplit(n_splits=5, n_train=n_train, n_test=n_test, random_state=42)
            if len(X_text) > n_train + n_test:
                for dim_reduction_name, dim_reduction in dim_reductions.items():
                    for model_name, model in models.items():
                        job_func = partial(run_on_encoded_data, X_enc, X_rest, y, dim_reduction_name, dim_reduction, model_name, model, encoding, cv,
                                        dataset = dataset, features = "all")
                        job = executor.submit(job_func)
                        jobs.append(job)
                        print(f"Submitted job {job.job_id} to the cluster.")
                
                #just X_enc
                for dim_reduction_name, dim_reduction in dim_reductions.items():
                    for model_name, model in models.items():
                        job_func = partial(run_on_encoded_data, X_enc, None, y, dim_reduction_name, dim_reduction, model_name, model, encoding, cv,
                                        dataset = dataset, features = "text_only") #TODO use the features argument?
                        job = executor.submit(job_func)
                        jobs.append(job)
                        print(f"Submitted job {job.job_id} to the cluster.")
                
                #just X_rest
                for dim_reduction_name, dim_reduction in dim_reductions.items():
                    for model_name, model in models.items():
                        job_func = partial(run_on_encoded_data, None, X_rest, y, dim_reduction_name, dim_reduction, model_name, model, encoding, cv,
                                        dataset = dataset, features = "rest_only")
                        job = executor.submit(job_func)
                        jobs.append(job)
                        print(f"Submitted job {job.job_id} to the cluster.")

results = []

for job in jobs:
    try:
        result = job.result()
        results.append(result)
    except Exception as e:
        print(f"Job {job.job_id} failed with exception: {e}")

# remove None
print(len(results))
results = [r for r in results if r is not None]
print(len(results))
df = pd.DataFrame(results)
melted_results = df.explode(['accuracies', "roc_auc"])
#melted_results = melted_results.drop(columns=["scores"])
#melted_results.to_csv("../results/results_all_01_10.csv", index=False)
#melted_results.to_csv("../results/results_all_02_10_bert_pooling.csv", index=False)
# append to "../results/results_all_02_10_bert_pooling.csv"
#melted_results.to_csv("../results/results_all_04_10.csv", index=False)
melted_results.to_csv("../results/results_18_10.csv", index=False)