from autofj.datasets import load_data
#from skrub import fuzzy_join
from src.fuzzy_join_custom import fuzzy_join
from src.encodings import encode, get_batch_embeddings
from autofj import AutoFJ
import pandas as pd

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


import submitit
executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(timeout_min=700, slurm_partition='parietal,normal', slurm_array_parallelism=100, cpus_per_task=4,
                            exclude="margpu009,marg00[1-9],marg0[11-12],marg0[14-15],marg0[16-20],marg0[25-32]")
# change name of job
executor.update_parameters(name="join")

import pandas as pd
def compute_join(dataset, target):
    left_table, right_table, gt = load_data(dataset)
    # encode both tables with fasttext (column title)

        
    joined_fj = autofj_merge(
                left_table,
                right_table,
                target=target,
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
        "target": target,
        "recall": recall,
        "precision": precision,
        "f1": f1
    }

jobs = []
with executor.batch():
    for dataset in dataset_list:
            for match_score in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
                jobs.append(executor.submit(compute_join, dataset, match_score))

results = [job.result() for job in jobs]
df = pd.DataFrame(results)

df.to_csv("results_join_autofj.csv", index=False)