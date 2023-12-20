from autofj.datasets import load_data
#from skrub import fuzzy_join
from src.fuzzy_join_custom import fuzzy_join
from src.encodings import encode, get_batch_embeddings
from autofj import AutoFJ
import pandas as pd
import time

def autofj_merge(left, right, target=0.9):
    """Merging using AutomaticFuzzyJoin"""
    autofj = AutoFJ(precision_target=target, verbose=True)
    autofj_joins = autofj.join(left, right, id_column="id")
    return autofj_joins



dataset_list = ['Amphibian',
 'ArtificialSatellite',
 'Artwork',
 'Award',
 'BasketballTeam',
 'Case',
 'ChristianBishop',
 'ClericalAdministrativeRegion',
 'Country',
 'Device',
 'Drug',
 'Election',
 'Enzyme',
 'EthnicGroup',
 'FootballLeagueSeason',
 'FootballMatch',
 'Galaxy',
 'GivenName',
 'GovernmentAgency',
 'HistoricBuilding',
 'Hospital',
 'Legislature',
 'Magazine',
 'MemberOfParliament',
 'Monarch',
 'MotorsportSeason',
 'Museum',
 'NCAATeamSeason',
 'NationalFootballLeagueSeason',
 'NaturalEvent',
 'Noble',
 'PoliticalParty',
 'Race',
 'RailwayLine',
 'Reptile',
 'RugbyLeague',
 'ShoppingMall',
 'SoccerClubSeason',
 'SoccerLeague',
 'SoccerTournament',
 'Song',
 'SportFacility',
 'SportsLeague',
 'Stadium',
 'TelevisionStation',
 'TennisTournament',
 'Tournament',
 'UnitOfWork',
 'Venue',
 'Wrestler']

encodings = []
model_names = [
    #"EleutherAI/pythia-70m",
    #"EleutherAI/pythia-160m",
    #"EleutherAI/pythia-410m",
    #"EleutherAI/pythia-1b",
    #"EleutherAI/pythia-1.4b",
    #"EleutherAI/pythia-2.8b",
    #"EleutherAI/pythia-6.9b",
    #"mistralai/Mistral-7B-v0.1",
    #"meta-llama/Llama-2-7b-hf",
    #"huggyllama/llama-7b"
    #"bert-base-cased",
    #"bert-large-cased",
    #"roberta-base",
    #"roberta-large",
    #"microsoft/deberta-v3-base",
    #"microsoft/deberta-v3-large"
]
for model_name in model_names:
    encodings.append(f"hf__{model_name}")

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
    # #"intfloat/e5-base-v2",
    # "intfloat/multilingual-e5-large",
    # #"intfloat/e5-large",
    # "thenlper/gte-small",
    # #"intfloat/e5-base",
    # #"intfloat/e5-small-v2",
    # "hkunlp/instructor-base",
    #"sentence-t5-xxl",
    # "intfloat/multilingual-e5-base",
    # #"XLM-3B5-embedding",
    #"gtr-t5-xxl",
    # #"SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
    # #"intfloat/e5-small",
    # #"TaylorAI/gte-tiny",
    #"gtr-t5-xl",
    # #"gtr-t5-large",
    # #"XLM-0B6-embedding",
    # #"intfloat/multilingual-e5-small",
    #"sentence-t5-xl",
    # #"all-mpnet-base-v2",
    # "sgpt-bloom-7b1-msmarco",
    # "jinaai/jina-embedding-l-en-v1",
    # #"SGPT-2.7B-weightedmean-msmarco-specb-bitfit",
    # #"sentence-t5-large",
    # #"MegatronBert-1B3-embedding",
    # #"TaylorAI/bge-micro-v2",
    # #"all-MiniLM-L12-v2",
    # #"all-MiniLM-L6-v2",
    # "jinaai/jina-embedding-b-en-v1",
    # # #"SGPT-1.3B-weightedmean-msmarco-specb-bitfit",
    #  "gtr-t5-base",
    # "nthakur/contriever-base-msmarco",
    # "TaylorAI/bge-micro",
    # "sentence-t5-base",
    # "paraphrase-multilingual-mpnet-base-v2",
    # "Hum-Works/lodestone-base-4096-v1",
    # # #"SGPT-5.8B-weightedmean-nli-bitfit",
    # "paraphrase-multilingual-MiniLM-L12-v2",
    # "msmarco-bert-co-condensor",
    # "jinaai/jina-embedding-s-en-v1"
    "all-distilroberta-v1"
]

for model_name in model_names:
    #if "e5" in model_name:
    encodings.append("lm__" + model_name)



import pandas as pd
def compute_join(config):
    try:
        dataset, encoder_name, match_score = config
        left_table, right_table, gt = load_data(dataset)
        # encode both tables with fasttext (column title)
        if encoder_name == "none":
            main_str_enc = None
            aux_str_enc = None
        else:
            main_str_enc = encode(left_table, "title", encoder_name=encoder_name, dataset_name="join_left_" + dataset, fail_if_not_cached=True)
            aux_str_enc = encode(right_table, "title", encoder_name=encoder_name, dataset_name="join_right_" + dataset, fail_if_not_cached=True)

        joined_fj = fuzzy_join(left_table, right_table, left_on="title",
                    match_score=match_score,
                    right_on="title",
                    return_score=True,
                    suffixes=("_l", "_r"),
                    main_str_enc_arg=main_str_enc,
                    aux_str_enc_arg=aux_str_enc,
        )

        df_all = pd.merge(joined_fj, gt, on=["id_l"], suffixes=("", "_gt"))
        df_all["correct"] = df_all["id_r"] == df_all["id_r_gt"]
        # drop na in correct
        df_all = df_all.dropna(subset=["correct"])

        recall = df_all["correct"].sum() / len(gt)
        precision = df_all["correct"].sum() / len(df_all)
        f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "dataset": dataset,
            "encoder_name": encoder_name,
            "match_score": match_score,
            "recall": recall,
            "precision": precision,
            "f1": f1
        }
    except Exception as e:
        print("error")
        print(str(e))
        return None

# jobs = []
# with executor.batch():
#     for dataset in dataset_list:
#         for encoder_name in encodings:
#             for match_score in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
#                 jobs.append(executor.submit(compute_join, dataset, encoder_name, match_score))

# Generate all combinations of parameters
from itertools import product
print("encodings", encodings)
param_combinations = list(product(dataset_list, encodings, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]))
# Chunk your jobs
CHUNK_SIZE = 500  # Choose a suitable chunk size
chunks = [param_combinations[i:i + CHUNK_SIZE] for i in range(0, len(param_combinations), CHUNK_SIZE)]


array_parallelism_total = 200
array_parallelism = array_parallelism_total // len(chunks)
print(f"Using {array_parallelism} array parallelism")

import submitit
executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(timeout_min=100, slurm_partition='parietal,normal', slurm_array_parallelism=array_parallelism, cpus_per_task=2,
                            exclude="margpu009,marg00[1-9],marg0[11-12],marg0[14-15],marg0[16-20],marg0[25-32]")
# change name of job
executor.update_parameters(name="join")


jobs = []
# Submit jobs chunk by chunk

for i, chunk in enumerate(chunks):
    while True:
        try:
            jobs.extend(executor.map_array(compute_join, chunk))
            break
        except Exception as e:
            print(e)
            print("Sleeping 200 seconds")
            time.sleep(200)
    print(f"Submitted chunk {i+1} of {len(chunks)}, {len(jobs)} jobs to the cluster.")

results = []
for job in jobs:
    try:
        result = job.result()
        if result is not None:
            print(result)
            results.append(result)
        else:
            print("None result")
    except Exception as e:
        print(e)
        print("Job failed")
        continue

df = pd.DataFrame(results)

df.to_csv("results_join_14_12_fifth.csv", index=False)