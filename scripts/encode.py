import submitit
from functools import partial
from src.data_loading import load_data
from src.encodings import encode_high_cardinality_features
from tqdm import tqdm

#encodings = ["openai__", "skrub__minhash_30"]
#encodings = ["skrub__minhash_10", "skrub__minhash_20", "skrub__minhash_30", "skrub__minhash_60", "skrub__minhash_100", "skrub__minhash_200"]
encodings = ["skrub__minhash_300", "skrub__minhash_400", "skrub__minhash_500", "skrub__minhash_600"]
#encodings = ["fasttext__30"]
#encodings = []
model_names = [
    # "EleutherAI/pythia-70m",
    # "EleutherAI/pythia-160m",
    # "EleutherAI/pythia-410m",
    # "EleutherAI/pythia-1b",
    # "EleutherAI/pythia-1.4b",
    # "EleutherAI/pythia-2.8b",
    # "EleutherAI/pythia-6.9b",
    # "mistralai/Mistral-7B-v0.1",
    # "meta-llama/Llama-2-7b-hf",
    # "huggyllama/llama-7b"
    #"bert-base-cased",
    #"bert-large-cased",
    #"roberta-base",
    #"roberta-large",
    #"microsoft/deberta-v3-base",
    #"microsoft/deberta-v3-large"
]
for model_name in model_names:
    encodings.append(f"hf__{model_name}")

#encodings = []
model_names = [
    #"BAAI/bge-large-en-v1.5",
    #"BAAI/bge-base-en-v1.5",
    #"llmrails/ember-v1",
    # "thenlper/gte-large",
    # "thenlper/gte-base",
    #"intfloat/e5-large-v2",
    # "BAAI/bge-small-en-v1.5",
    # "hkunlp/instructor-xl",
    # "hkunlp/instructor-large",
    #"intfloat/e5-base-v2",
    #"intfloat/multilingual-e5-large",
    #"intfloat/e5-large",
    # "thenlper/gte-small",
    #"intfloat/e5-base",
    #"intfloat/e5-small-v2",
    # "hkunlp/instructor-base",
    #"sentence-t5-xxl",
    #"intfloat/multilingual-e5-base",
    #"lixsh6/XLM-3B5-embedding",
    #"gtr-t5-xxl",
    #"Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
   # "intfloat/e5-small",
    #"TaylorAI/gte-tiny",
    #"gtr-t5-xl",
    #"gtr-t5-large",
    #"lixsh6/XLM-0B6-embedding",
    #"intfloat/multilingual-e5-small",
    #"sentence-t5-xl",
    #"all-mpnet-base-v2",
    #"bigscience/sgpt-bloom-7b1-msmarco",
    #"jinaai/jina-embedding-l-en-v1",
    #"Muennighoff/SGPT-2.7B-weightedmean-msmarco-specb-bitfit",
    #"sentence-t5-large",
    #"lixsh6/MegatronBert-1B3-embedding",
    #"TaylorAI/bge-micro-v2",
    #"all-MiniLM-L12-v2",
    #"all-MiniLM-L6-v2",
    #"jinaai/jina-embedding-b-en-v1",
    #"Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit",
    #"gtr-t5-base",
    #"nthakur/contriever-base-msmarco",
    #"TaylorAI/bge-micro",
    #"sentence-t5-base",
    #"paraphrase-multilingual-mpnet-base-v2",
    #"Hum-Works/lodestone-base-4096-v1",
    #"Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit",
    #"gpt2",
    #"paraphrase-multilingual-MiniLM-L12-v2",
    #"msmarco-bert-co-condensor",
    #"jinaai/jina-embedding-s-en-v1"
]

for model_name in model_names:
    #if "e5" in model_name:
        encodings.append("lm__" + model_name)

print("encodings", encodings)

datasets = ['bikewale', 'clear_corpus', 'company_employees',
       'employee-remuneration-and-expenses-earning-over-75000',
       'employee_salary', 'goodreads', 'journal_jcr_cls', 'ramen_ratings',
       'spotify', 'us_accidents_counts', 'us_accidents_severity',
       'us_presidential', 'wine_review', 'zomato']
#datasets = [f"companies_{year}" for year in range(2012, 2024)]



executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(timeout_min=300, slurm_partition='parietal,normal,gpu',
                           exclude="margpu001,margpu002,margpu003,margpu004",
                           slurm_array_parallelism=150,
                           cpus_per_task=4)
                           #gpus_per_node=1)
                           #slurm_additional_parameters={"nodelist": "margpu009"})


def encoding_dataset(dataset, encoding):
    X, y = load_data(dataset, max_rows=10000)
    #X_enc = encode(X_text, encoding, dataset_name=dataset, override_cache=False)
    X_enc, X_rest = encode_high_cardinality_features(X, encoding, dataset_name=dataset, override_cache=False, cardinality_threshold=30)
    #return X_enc, X_rest, y
    

jobs = []

with executor.batch():
    for dataset in tqdm(datasets):
        for encoding in tqdm(encodings, leave=False):
            job = executor.submit(encoding_dataset, dataset, encoding)
            jobs.append(job)

for job in jobs:
    print(job.result())