import submitit
from functools import partial
from src.data_loading import load_data
from src.encodings import encode_high_cardinality_features
from tqdm import tqdm

encodings = []
model_names = [
    #"EleutherAI/pythia-70m",
    #"EleutherAI/pythia-160m",
    #"EleutherAI/pythia-410m",
    #"EleutherAI/pythia-1b",
    #"EleutherAI/pythia-1.4b",
    #"EleutherAI/pythia-2.8b",
    #"EleutherAI/pythia-6.9b",
    "mistralai/Mistral-7B-v0.1",
    #"meta-llama/Llama-2-7b-hf",
    "huggyllama/llama-7b"
]
for model_name in model_names:
    encodings.append(f"hf__{model_name}")


datasets = ['bikewale', 'clear_corpus', 'company_employees',
       'employee-remuneration-and-expenses-earning-over-75000',
       'employee_salary', 'goodreads', 'journal_jcr_cls', 'ramen_ratings',
       'spotify', 'us_accidents_counts', 'us_accidents_severity',
       'us_presidential', 'wine_review', 'zomato']



executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(timeout_min=300, slurm_partition='parietal,normal,gpu',
                           exclude="margpu001,margpu002,margpu003,margpu004",
                           gpus_per_node=1,
                           slurm_additional_parameters={"nodelist": "margpu009"})


def encoding_dataset(dataset, encoding):
    X, y = load_data(dataset, max_rows=10000)
    #X_enc = encode(X_text, encoding, dataset_name=dataset, override_cache=False)
    X_enc, X_rest = encode_high_cardinality_features(X, encoding, dataset_name=dataset, override_cache=False, cardinality_threshold=30)
    #return X_enc, X_rest, y
    

jobs = []

for dataset in tqdm(datasets):
    for encoding in tqdm(encodings, leave=False):
        job = executor.submit(encoding_dataset, dataset, encoding)
        jobs.append(job)