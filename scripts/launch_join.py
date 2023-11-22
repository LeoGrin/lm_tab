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


encodings = ["openai__", "lm__BAAI/bge-large-en-v1.5", "lm__llmrails/ember-v1", "none"]


import submitit
executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(timeout_min=100, slurm_partition='parietal,normal', slurm_array_parallelism=30, cpus_per_task=2,
                            exclude="margpu009,marg00[1-9],marg0[11-12],marg0[14-15],marg0[16-20],marg0[25-32]")
# change name of job
executor.update_parameters(name="join")

import pandas as pd
def compute_join(config):
    dataset, encoder_name, match_score = config
    left_table, right_table, gt = load_data(dataset)
    # encode both tables with fasttext (column title)
    if encoder_name == "none":
        main_str_enc = None
        aux_str_enc = None
    else:
        main_str_enc = encode(left_table, "title", encoder_name=encoder_name, dataset_name="join_left_" + dataset)
        aux_str_enc = encode(right_table, "title", encoder_name=encoder_name, dataset_name="join_right_" + dataset)

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

# jobs = []
# with executor.batch():
#     for dataset in dataset_list:
#         for encoder_name in encodings:
#             for match_score in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
#                 jobs.append(executor.submit(compute_join, dataset, encoder_name, match_score))

# Generate all combinations of parameters
from itertools import product
param_combinations = list(product(dataset_list, encodings, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]))
# Chunk your jobs
CHUNK_SIZE = 500  # Choose a suitable chunk size
chunks = [param_combinations[i:i + CHUNK_SIZE] for i in range(0, len(param_combinations), CHUNK_SIZE)]

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

results = [job.result() for job in jobs]
df = pd.DataFrame(results)

df.to_csv("results_join.csv", index=False)