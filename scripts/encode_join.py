import submitit
from autofj.datasets import load_data
#from skrub import fuzzy_join
from src.fuzzy_join_custom import fuzzy_join
from src.encodings import encode, get_batch_embeddings
import time
executor = submitit.AutoExecutor(folder="logs")
# executor.update_parameters(timeout_min=100, slurm_partition='parietal,normal', slurm_array_parallelism=15, cpus_per_task=2,
#                             exclude="margpu009,marg00[1-9],marg0[11-12],marg0[14-15],marg0[16-20],marg0[25-32]")
executor.update_parameters(timeout_min=300, slurm_partition='parietal,normal,gpu',
                           exclude="margpu001,margpu002,margpu003,margpu004",
                           slurm_array_parallelism=4,
                           cpus_per_task=4,
                           gpus_per_node=1)
# change name of job
executor.update_parameters(name="pipeline")

#encodings = ["openai__", "lm__BAAI/bge-large-en-v1.5", "lm__llmrails/ember-v1"]


encodings = []# "skrub__minhash_30"]
#encodings = []
#encodings = []
model_names = [
    # "EleutherAI/pythia-70m",
    # "EleutherAI/pythia-160m",
    # "EleutherAI/pythia-410m",
    # "EleutherAI/pythia-1b",
    # "EleutherAI/pythia-1.4b",
    # "EleutherAI/pythia-2.8b",
    #"EleutherAI/pythia-6.9b",
    #"mistralai/Mistral-7B-v0.1",
    #"meta-llama/Llama-2-7b-hf",
    #"huggyllama/llama-7b"
    #"bert-base-cased",
    #"bert-large-cased",
    #"roberta-base",
    # "roberta-large",
    # "microsoft/deberta-v3-base",
    #"microsoft/deberta-v3-large"
]
for model_name in model_names:
    encodings.append(f"hf__{model_name}")

#encodings = []
model_names = [
    # "BAAI/bge-large-en-v1.5",
    # "BAAI/bge-base-en-v1.5",
    #  "llmrails/ember-v1",
    # "thenlper/gte-large",
    # "thenlper/gte-base",
    # "intfloat/e5-large-v2",
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
    #"sentence-t5-xxl",
    # "intfloat/multilingual-e5-base",
    # #"XLM-3B5-embedding",
    #"gtr-t5-xxl",
    # #"SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
    # "intfloat/e5-small",
    # "TaylorAI/gte-tiny",
    #"gtr-t5-xl",
    #"gtr-t5-large",
    # #"XLM-0B6-embedding",
    # "intfloat/multilingual-e5-small",
    #"sentence-t5-xl",
    # "all-mpnet-base-v2",
    # #"sgpt-bloom-7b1-msmarco",
    #"jinaai/jina-embedding-l-en-v1",
    # #"SGPT-2.7B-weightedmean-msmarco-specb-bitfit",
    #"sentence-t5-large",
    # #"MegatronBert-1B3-embedding",
    # "TaylorAI/bge-micro-v2",
    # "all-MiniLM-L12-v2",
    # "all-MiniLM-L6-v2",
    # "jinaai/jina-embedding-b-en-v1",
    # #"SGPT-1.3B-weightedmean-msmarco-specb-bitfit",
    #"gtr-t5-base",
    # "nthakur/contriever-base-msmarco",
    # "TaylorAI/bge-micro",
    #"sentence-t5-base",
    # "paraphrase-multilingual-mpnet-base-v2",
    # "Hum-Works/lodestone-base-4096-v1",
    # #"SGPT-5.8B-weightedmean-nli-bitfit",
    # "paraphrase-multilingual-MiniLM-L12-v2",
    # "msmarco-bert-co-condensor",
    # "jinaai/jina-embedding-s-en-v1",
    "all-distilroberta-v1"
]

for model_name in model_names:
    #if "e5" in model_name:
    encodings.append("lm__" + model_name)

from urllib.request import urlopen
import json
from itertools import product
url = "https://api.github.com/repos/chu-data-lab/AutomaticFuzzyJoin/contents/src/autofj/benchmark"
response = urlopen(url)
data = json.loads(response.read())
print("Available datasets:")
dataset_list = []
for d in data:
    dataset_list.append(d["name"])


def encode_join(config):
    """Encode a dataset and save it to disk"""
    dataset, column, encoder_name = config
    left_table, right_table, gt = load_data(dataset)
    # encode both tables with fasttext (column title)
    main_str_enc = encode(left_table, column, encoder_name=encoder_name, dataset_name="join_left_" + dataset)
    aux_str_enc = encode(right_table, column, encoder_name=encoder_name, dataset_name="join_right_" + dataset)
    return main_str_enc, aux_str_enc

# with executor.batch():
#     for encoding in encodings:
#         for dataset in dataset_list:
#             for column in ["title"]:
#                 executor.submit(encode_join, dataset, column, encoding)

# Generate all combinations of parameters
print("encodings", encodings)
param_combinations = list(product(dataset_list, ["title"], encodings))

# Chunk your jobs
CHUNK_SIZE = 500  # Choose a suitable chunk size
chunks = [param_combinations[i:i + CHUNK_SIZE] for i in range(0, len(param_combinations), CHUNK_SIZE)]

jobs = []
# Submit jobs chunk by chunk

for i, chunk in enumerate(chunks):
    while True:
        try:
            jobs.extend(executor.map_array(encode_join, chunk))
            break
        except Exception as e:
            print(e)
            print("Sleeping 200 seconds")
            time.sleep(200)
    print(f"Submitted chunk {i+1} of {len(chunks)}, {len(jobs)} jobs to the cluster.")
