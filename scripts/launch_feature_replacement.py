from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from tabpfn import TabPFNClassifier
from tabpfn.utils import normalize_data, to_ranking_low_mem, remove_outliers
from tabpfn.utils import NOP, normalize_by_used_features_f
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.base import clone
import torch
from skrub import TableVectorizer, MinHashEncoder
def run_on_encoded_data_multiple_dim_rec(X_enc, X_rest, y, original_column_names, dim_reduction_names, dim_reductions, model_name, model,
                        cv, regression=False, **kwargs):
    """
    X_enc: np array of shape (n_samples, embedding_dim), the embedded texts
    X_rest: np array of shape (n_samples, n_features), additional tabular data
    y: np array of shape (n_samples,), the classifcation target
    dim_reduction_name: str, the name of the dim reduction method
    dim_reduction: sklearn transformer, the dim reduction method
    model_name: str, the name of the model
    model: sklearn model, the model
    encoding: str, the name of the encoding which was used to create X_enc
    cv: sklearn cross validator, the cross validator to use
    regression: bool, whether to use regression or classification, default False
    """
    assert model_name in ["TabPFNClassifier", "TabPFNClassifier_basic", "LogisticRegression", "GradientBoostingClassifier", "GradientBoostingRegressor", "LinearRegression"]
    #TODO: make this cleaner
    # we want to eliminate certain combinations
    # passthrough and lm__ means taking the full lm embedding, which is slow if the model is not LogisticRegression
    # for skrub encodings, we don't want to use passthrough
    if X_rest is not None:
        # encode X_rest with the TableVectorizer
        if model_name.startswith("TabPFNClassifier"):
            # ordinal encoding for low_cardinality columns
            low_card_cat_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            low_card_cat_transformer = OneHotEncoder(handle_unknown="ignore")
        if model_name.startswith("LogisticRegression"):
            numerical_transformer = StandardScaler()
        else:
            numerical_transformer = "passthrough"
        
        rest_trans = TableVectorizer(high_card_cat_transformer = MinHashEncoder(n_components=10, analyzer='char'),
                                    low_card_cat_transformer = low_card_cat_transformer,
                                    numerical_transformer=numerical_transformer,
                                    cardinality_threshold=30)
    if X_enc is not None:
        
        # Assuming X_enc and X_rest are numpy arrays, you can get their shapes
        n_enc_columns = X_enc.shape[1]
        # names of the columns should be of format original_column_name__index
        print("__" in X_enc.columns[0])
        assert all(["__" in col for col in X_enc.columns])
        # just to check
        original_column_names_ = np.unique([col.split("__")[0] for col in X_enc.columns])
        assert len(original_column_names_) == len(dim_reduction_names) == len(dim_reductions)
        assert set(original_column_names) == set(original_column_names_)
        print("original_column_names", original_column_names)
        # original_column_names is ordered in the same order
        # get the indices of the columns for each original column
        encoded_columns_indices = {}
        for col in original_column_names:
            indices = [i for i, c in enumerate(X_enc.columns) if c.split("__")[0] == col]
            encoded_columns_indices[col] = indices
        print(encoded_columns_indices)
        print(len(encoded_columns_indices))
       

        # Create the ColumnTransformer
        #TODO: test this
        # assume already cloned
        if X_rest is not None:
            n_rest_columns = X_rest.shape[1]
            # Create column indices for X_enc and X_rest
            enc_indices = np.arange(n_enc_columns)
            rest_indices = np.arange(n_enc_columns, n_enc_columns + n_rest_columns)
            #check
            all_indices = np.concatenate(list(encoded_columns_indices.values()) + [rest_indices])
            # assert no duplicates
            assert len(all_indices) == len(np.unique(all_indices)), f"Duplicate indices: {[i for i in all_indices if list(all_indices).count(i) > 1]}"
            # assert we have all indices
            assert set(all_indices) == set(np.arange(n_enc_columns + n_rest_columns))
            transformers = [('rest_trans', rest_trans, rest_indices), *[(f"dim_reduction_{i}", dim_reductions[i], encoded_columns_indices[col_name]) for i, col_name in enumerate(original_column_names)]] 
            full_X = np.concatenate([X_enc, X_rest], axis=1)
        else:
            transformers = [*[(f"dim_reduction_{i}", dim_reductions[i], encoded_columns_indices[col_name]) for i, col_name in enumerate(original_column_names)]]
            full_X = X_enc
        print(transformers)
        complete_trans = ColumnTransformer(
            transformers=transformers,
        )
        
        print(X_enc.shape, full_X.shape)
        print(complete_trans.fit_transform(full_X).shape)
    elif X_rest is not None:
        print(X_rest.shape)
        complete_trans = rest_trans
        full_X = X_rest
    else:
        raise ValueError("At least one of X_enc and X_rest must be not None")


    pipeline = Pipeline([("encoding", complete_trans), ("model", model)])
    #scores = cross_val_score(pipeline, full_X, y, scoring="accuracy", cv=cv)
    # report both accuracy and roc_auc
    if regression:
        scores = cross_validate(pipeline, full_X, y, scoring=["neg_mean_squared_error", "r2"], cv=cv)
    else:
        scores = cross_validate(pipeline, full_X, y, scoring=["accuracy", "roc_auc_ovr"], cv=cv)

    try:
        n_train = cv.n_train
        n_test = cv.n_test
    except:
        n_train = np.nan
        n_test = np.nan
    res =  {
        'dim_reductions': dim_reduction_names,
        'model': model_name,
        #'accuracies': scores['test_accuracy'],
        #'roc_auc': scores['test_roc_auc_ovr'],
        'n_train': n_train,
        'n_test': n_test,
        **kwargs
    }
    # add the scores
    if regression:
        res['neg_mean_squared_error'] = scores['test_neg_mean_squared_error']
        res['r2'] = scores['test_r2']
    else:
        res['accuracies'] = scores['test_accuracy']
        res['roc_auc'] = scores['test_roc_auc_ovr']
    
    return res


from skrub import TableVectorizer
from src.encodings import encode
from src.data_loading import load_data
from src.utils import run_on_encoded_data, FixedSizeSplit
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.base import clone
models = {"LogisticRegression": LogisticRegression(), "GradientBoostingClassifier": GradientBoostingClassifier()}
dim_reductions = {"PCA_30": PCA(n_components=30), "passthrough": "passthrough"}
def switch_encoding(base_encoder_name="skrub__minhash_30", new_encoder_name="openai__", dataset_name=None, use_cache=True, override_cache=False, cardinality_threshold=30, fail_if_not_cached=True,
                    n_train=1000, n_test=500):
    assert base_encoder_name.startswith("skrub__")
    X, y = load_data(dataset_name, max_rows=10_000)
    print(X)
    tb = TableVectorizer(cardinality_threshold=cardinality_threshold,
                        high_card_cat_transformer = "passthrough",
                        low_card_cat_transformer = "passthrough",
                        numerical_transformer = "passthrough",
                        datetime_transformer = "passthrough",
    ) #just to get the high cardinality columns
    tb.fit(X)
    # get high cardinality columns
    high_cardinality_columns = []
    for name, trans, cols in tb.transformers_:
        print(name, cols)
        if "high" in name:
            high_cardinality_columns.extend(cols)
            break
    print("High cardinality columns", high_cardinality_columns)
    all_results = []
    assert "no_column" not in high_cardinality_columns
    for col_to_encode_with_new in high_cardinality_columns + ["no_column"]:
        # encode the high cardinality columns
        res = []
        lengths = []
        dim_reductions = []
        dim_reduction_names = []
        high_cardinality_columns_to_send = []
        for col in high_cardinality_columns:
            if col == col_to_encode_with_new:
                encoder_name = new_encoder_name
                if encoder_name != "drop":
                    dim_reduction_names.append("PCA_30")
                    dim_reductions.append(clone(PCA(n_components=30)))
            else:
                encoder_name = base_encoder_name
                dim_reduction_names.append("passthrough")
                dim_reductions.append("passthrough")
            if encoder_name != "drop":
                new_enc = encode(X, col, encoder_name, dataset_name=dataset_name, use_cache=use_cache, override_cache=override_cache, fail_if_not_cached=fail_if_not_cached)
                res.append(new_enc)
                lengths.append(new_enc.shape[1])
                high_cardinality_columns_to_send.append(col)
            else:
                continue
                
        print("Lengths", lengths)
        print(sum(lengths))
        # create a dataframe with name original_col_name__index
        df = pd.DataFrame(np.concatenate(res, axis=1))
        print(df.shape)
        new_column_names = []
        for i in range(len(res)):
            for j in range(lengths[i]):
                new_column_names.append(high_cardinality_columns_to_send[i] + "__" + str(j))

        df.columns = new_column_names
        X_enc, X_rest =  df, X.drop(high_cardinality_columns, axis=1)
        if len(X_enc) < n_train + n_test:
            print("Not enough data")
            continue
        cv = FixedSizeSplit(n_train=1000, n_test=1000, n_splits=7, random_state=42)
        all_results.append(run_on_encoded_data_multiple_dim_rec(X_enc, X_rest, y, high_cardinality_columns_to_send, dim_reduction_names, dim_reductions, "GradientBoostingClassifier", GradientBoostingClassifier(), 
                                                   cv, dataset=dataset_name, base_encoder_name=base_encoder_name, new_encoder_name=new_encoder_name, col_to_encode_with_new=col_to_encode_with_new, features="all"))
        all_results.append(run_on_encoded_data_multiple_dim_rec(X_enc, None, y, high_cardinality_columns_to_send, dim_reduction_names, dim_reductions, "GradientBoostingClassifier", GradientBoostingClassifier(), 
                                                   cv, dataset=dataset_name, base_encoder_name=base_encoder_name, new_encoder_name=new_encoder_name, col_to_encode_with_new=col_to_encode_with_new, features="text_only"))
    
    # merge results
    return all_results

                                                   
datasets = ['bikewale', 'clear_corpus', 'company_employees',
       'employee-remuneration-and-expenses-earning-over-75000',
       'employee_salary', 'goodreads', 'journal_jcr_cls', 'ramen_ratings',
       'spotify', 'us_accidents_counts', 'us_accidents_severity',
       'us_presidential', 'wine_review', 'zomato']

new_encoder_names = ["openai__", "drop"]
#new_encoder_name = ["drop"]
base_encoder_names = ["skrub__minhash_30"]

encodings = []
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
    #"jinaai/jina-embedding-b-en-v1",
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
    new_encoder_names.append("lm__" + model_name)


import submitit
import time
# Generate all combinations of parameters
from itertools import product

jobs = []

executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(timeout_min=500, slurm_partition='parietal,normal', slurm_array_parallelism=350, cpus_per_task=2,
                            exclude="margpu009")
# change name of job
executor.update_parameters(name="switch_encoding")

with executor.batch():
    for dataset_name in datasets:
        for new_encoder_name in new_encoder_names:
            #switch_encoding(new_encoder_name=new_encoder_name, dataset_name=dataset_name)
            job = executor.submit(switch_encoding, new_encoder_name=new_encoder_name, dataset_name=dataset_name)
            jobs.append(job)

# create a dataframe with all the results
res = []
for job in jobs:
    try:
        print(job.result())
        for r in job.result():
            res.append(r)
    except Exception as e:
        print("Error", str(e))
        print("Continuing")
        continue

df = pd.DataFrame(res)
df = df.explode(["accuracies", "roc_auc"])
df.to_csv("../results/column_xp_30_11.csv")