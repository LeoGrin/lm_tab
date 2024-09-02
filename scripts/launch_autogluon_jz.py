from src.data_loading import load_data
from src.utils import FixedSizeSplit
import pandas as pd
import numpy as np
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import submitit
from itertools import product
import time
from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from src.utils_jz import setup_submitit_executor_a100
import hashlib



# datasets = ["journal_jcr_cls", "movies", "michelin", "spotify", "employee_salary", "museums", "fifa_footballplayers_22", "jp_anime", "clear_corpus", "company_employees", "us_presidential", "us_accidents_severity", "us_accidents_counts", "wine_review"]
# datasets.extend(["building_permits", "public", "kickstarter", "colleges", "medical_charge", "traffic_violations"]) #  "agora"
# datasets.extend(["bikewale", "goodreads", "zomato", "coffee_fix", "nfl_contract", "employee-remuneration-and-expenses-earning-over-75000", "coffee_analysis", "ramen_ratings", "beer_profile_and_ratings", "adult"])
datasets = ['bikewale', 'clear_corpus', 'company_employees',
       'employee-remuneration-and-expenses-earning-over-75000',
       'employee_salary', 'goodreads', 'journal_jcr_cls', 'ramen_ratings',
       'spotify', 'us_accidents_counts', 'us_accidents_severity',
       'us_presidential', 'wine_review', 'zomato']
#new_datasets = ['prod',
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
]
#datasets = new_datasets + datasets
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

#datasets = ["prod", "imdb", "ae", "mercari"]
#datasets = ["prod", "mercari"]
#datasets = ["book", "house"]
#datasets = ["imdb", "ae"]
# datasets = [
#     "prod",
#     "airbnb",
#     "channel",
#     "wine",
#     "ae",
#     "qaa",
#     "qaq",
#     "cloth",
#     "mercari",
#     "salary"
# ]


print(len(datasets))
#datasets = ["agora"]

def run_autogluon(X, y, cv, time_limit=180, presets="medium_quality", 
                  hf_model="default", model_path="/tmp/autogluon"):
    # Prepare the data for AutoGluon
    data = pd.DataFrame(X)
    data['target'] = y

    all_scores = []

    # Use cv to split the data and fit the model
    for i, (train_idx, test_idx) in enumerate(cv.split(data)):
        predictor = TabularPredictor(label='target',
                                     path=model_path + f"_{i}")
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        hyperparameters = get_hyperparameter_config('multimodal')
        if hf_model != "default":
            hyperparameters["AG_AUTOMM"]["model.hf_text.checkpoint_name"] = hf_model
        # Fit the model using the training data
        predictor.fit(train_data=train_data, time_limit=time_limit, num_gpus=1,
                      num_cpus=8,
                      presets=presets,
                      hyperparameters=hyperparameters)
        
        # Evaluate the model using the test data
        performance = predictor.evaluate(test_data)
        print("Model performance:", performance)
        all_scores.append(performance)

    res = {}
    metrics = performance.keys()
    for metric in metrics:
        res[metric] = [score[metric] for score in all_scores]
    #res["roc_auc"] = [score["roc_auc"] for score in all_scores]
    #res["accuracies"] = [score["accuracy"] for score in all_scores]
    return res

def run_autogluon_multimodal(X, y, cv, time_limit=180, presets="medium_quality",
                            hf_model="default", model_path="/tmp/autogluon_multimodal"):
    # Prepare the data for AutoGluon
    data = pd.DataFrame(X)
    data['target'] = y

    all_scores = []

    # Use cv to split the data and fit the model
    for i, (train_idx, test_idx) in enumerate(cv.split(data)):
        predictor = MultiModalPredictor(label='target',
                                        path=model_path + f"_{i}")
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Fit the model using the training data
        if hf_model == "default":
            predictor.fit(train_data=train_data, time_limit=time_limit)
        else:
            predictor.fit(train_data=train_data, time_limit=time_limit,
                          hyperparameters={"model.hf_text.checkpoint_name": hf_model})
        
        # Evaluate the model using the test data
        is_binary = len(np.unique(y)) == 2
        if is_binary:
            performance = predictor.evaluate(test_data, metrics=['acc', 'f1', 'roc_auc', "balanced_accuracy"])
        else:
            performance = predictor.evaluate(test_data, metrics=['acc', "balanced_accuracy"])
        print("Model performance:", performance)
        all_scores.append(performance)

    res = {}
    metrics = performance.keys()
    for metric in metrics:
        res[metric] = [score[metric] for score in all_scores]
    #res["roc_auc"] = [score["roc_auc"] for score in all_scores]
    #res["accuracies"] = [score["accuracy"] for score in all_scores]
    return res






#print("Number of iterations: ", len(datasets) * len(encodings) * len(dim_reductions) * len(models))


def pipeline(config):#dataset, encoding, n_test, dim_reduction_name, model_name, n_train, features
    print(config)
    dataset, n_test, n_train, features, time_limit, preset, hf_model, only_multimodal = config
    

    # Convert the config to a string
    config_str = str(config)

    # Create a hash of the config string
    config_hash = hashlib.md5(config_str.encode()).hexdigest()

    model_path = f"/lustre/fsn1/projects/rech/ptq/ueg53am/lm_tab/models/{config_hash}"

    X, y = load_data(dataset, max_rows=10000)
    if len(X) < n_train + n_test:
        return (n_train, features, None)
    cv = FixedSizeSplit(n_splits=7, n_train=n_train, n_test=n_test, random_state=42)
    if features == "all":
        if only_multimodal:
            res_scores = run_autogluon_multimodal(X, y, cv, time_limit, preset, hf_model, model_path)
        else:
            res_scores = run_autogluon(X, y, cv, time_limit, preset, hf_model, model_path)
        #res_scores = run_autogluon_multimodal(X, y, cv, time_limit, preset, hf_model)
        res_scores["n_train"] = n_train
        res_scores["n_test"] = n_test
        res_scores["features"] = features
        res_scores["dataset"] = dataset
        res_scores["time_limit"] = time_limit
        res_scores["preset"] = preset
        res_scores["hf_model"] = hf_model
        encoding = "autogluon_multimodal" if only_multimodal else "autogluon"
        res_scores["encoding"] = encoding
        #return (n_train, features, res_scores)
    elif features == "text_only":
        raise Exception("Not implemented")
        #return (n_train, features, time_limit, preset, run_on_encoded_data(X_enc, None, y, dim_reduction_name, dim_reduction, model_name, model, encoding, cv, dataset=dataset, features=features))
    elif features == "rest_only":
        raise Exception("Not implemented")
        #return (n_train, features, run_on_encoded_data(None, X_rest, y, dim_reduction_name, dim_reduction, model_name, model, encoding, cv, dataset=dataset, features=features))

    # Save the result to a CSV file
    df = pd.DataFrame([res_scores])
    # Add the hash to the dataframe
    df['config_hash'] = config_hash
    base_path = f"/lustre/fswork/projects/rech/ptq/ueg53am/lm_tab/results_raw"
    if not os.path.isfile(f"{encoding}_{config_hash}.csv"):
        df.to_csv(f"{base_path}/{encoding}_{config_hash}.csv", index=False)
    else:
        df.to_csv(f"{base_path}/{encoding}_{config_hash}.csv", mode='a', header=False, index=False)

if __name__ == "__main__":
    n_trains = [1000, 3000, 5000]
    #n_trains = [3000, 4000, 5000]
    time_limit = [15 * 60]
    presets = ["medium_quality"]
    features_list = ["all"]#, "rest_only"]
    hf_models = ["default"]
    only_multimodal_list = [False, True]
    n_test = 500

    # Generate all combinations of parameters
    param_combinations = list(product(datasets, [n_test], n_trains, features_list, time_limit, presets, hf_models, only_multimodal_list))

    # Chunk your jobs
    CHUNK_SIZE = 500  # Choose a suitable chunk size
    chunks = [param_combinations[i:i + CHUNK_SIZE] for i in range(0, len(param_combinations), CHUNK_SIZE)]

    array_parallelism_total = 100
    array_parallelism = array_parallelism_total // len(chunks)
    print(f"Using {array_parallelism} array parallelism")

    jobs = []

    executor = setup_submitit_executor_a100("pipeline", gpus_per_node=1, cpus_per_task=8, time=10*60)
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


    # # Define the columns of your dataframe
    # # Open a file to write the results
    # name = "test_longer_autogluon_02_09"
    # for job in jobs:
    #     try:
    #         result = job.result()
    #         if result is not None:
    #             if result[2] is None:
    #                 continue
    #             res = result[2]
    #             # Flatten
    #             #print(result)
    #             #result = [item for sublist in result for item in sublist if item is not None]
    #             # Remove element if it is (_, _, None)
    #             #print(result)
    #             # Merge with n_train, features
    #             #result = [{"n_train": n_train, "features": features, **r} for n_train, features, r in result]
    #             print(res)
    #             df = pd.DataFrame(res)
    #             if len(df):
    #                 # Explode 'accuracies' and 'roc_auc'
    #                 #df = df.explode(['accuracies', "roc_auc"])
    #                 # explode columns with scores
    #                 columns = list(df.columns)
    #                 columns_scores = [col for col in columns if col not in ["n_train", "n_test" "features", 
    #                 "dataset", "time_limit", "preset", "hf_model", "encoding"]]
    #                 # check if the file exists
    #                 if not os.path.isfile(f"../results/{name}.csv"):
    #                     # Create a new file
    #                     df.to_csv(f"../results/{name}.csv", index=False)
    #                 else:
    #                     # Append to the file
    #                     df.to_csv(f"../results/{name}.csv", mode='a', header=False, index=False)
    #     except Exception as e:
    #         print(f"Job {job.job_id} failed with exception: {e}")
