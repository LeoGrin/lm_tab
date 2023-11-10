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

class FeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=1, method="first"):
        self.n_features = n_features
        self.method = method

    def fit(self, X, y=None):
        return self  # Nothing to fit, so return self

    def transform(self, X):
        # Extract the first n_features
        # choose features to keep
        if self.method == "first":
            res = X[:, :self.n_features]
        elif self.method == "last":
            res = X[:, -self.n_features:]
        elif self.method == "middle":
            res = X[:, self.n_features//2:self.n_features//2+self.n_features]
        elif self.method == "random":
            res = X[:, np.random.choice(X.shape[1], self.n_features, replace=False)]
        elif self.method == "biggest_variance":
            indices = np.argsort(np.var(X, axis=0))[-self.n_features:]
            res = X[:, indices]
        elif self.method == "smallest_variance":
            indices = np.argsort(np.var(X, axis=0))[:self.n_features]
            res = X[:, indices]
        
        assert res.shape == (X.shape[0], self.n_features)
        return res

    

from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import numpy as np

class FixedSizeSplit(BaseCrossValidator):
    def __init__(self, n_train: int, n_test: int = None, n_splits: int = 5, random_state: int = None) -> None:
        self.n_train = n_train
        self.n_test = n_test
        self.n_splits = n_splits
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)

        if self.n_train > n_samples:
            raise ValueError(f"Cannot set n_train={self.n_train} greater than the number of samples: {n_samples}.")

        if self.n_test is not None and self.n_test > n_samples - self.n_train:
            raise ValueError(f"Cannot set n_test={self.n_test} greater than the remaining samples: {n_samples - self.n_train}.")

        indices = np.arange(n_samples)

        rng = np.random.default_rng(self.random_state)

        for i in range(self.n_splits):
            indices_copy = indices.copy()
            rng.shuffle(indices_copy)
            
            train = indices_copy[:self.n_train]
            
            if self.n_test is not None:
                test = indices_copy[self.n_train:self.n_train + self.n_test]
            else:
                test = indices_copy[self.n_train:]
            
            yield train, test


#Taken from TabPFNClassifier
#used to preprocess the data before feeding it to the tabpfn
def preprocess_input(eval_xs, eval_ys, eval_position, device, preprocess_transform="none",
                     max_features=100, normalize_with_sqrt=False, normalize_with_test=False,
                     normalize_to_ranking=False, categorical_feats=[]):
    import warnings

    if eval_xs.shape[1] > 1:
        raise Exception("Transforms only allow one batch dim - TODO")
    if preprocess_transform != 'none':
        if preprocess_transform == 'power' or preprocess_transform == 'power_all':
            pt = PowerTransformer(standardize=True)
        elif preprocess_transform == 'quantile' or preprocess_transform == 'quantile_all':
            pt = QuantileTransformer(output_distribution='normal')
        elif preprocess_transform == 'robust' or preprocess_transform == 'robust_all':
            pt = RobustScaler(unit_variance=True)

    # eval_xs, eval_ys = normalize_data(eval_xs), normalize_data(eval_ys)
    eval_xs = normalize_data(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position)

    # Removing empty features
    eval_xs = eval_xs[:, 0, :]
    sel = [len(torch.unique(eval_xs[0:eval_ys.shape[0], col])) > 1 for col in range(eval_xs.shape[1])]
    eval_xs = eval_xs[:, sel]

    warnings.simplefilter('error')
    if preprocess_transform != 'none':
        eval_xs = eval_xs.cpu().numpy()
        feats = set(range(eval_xs.shape[1])) if 'all' in preprocess_transform else set(
            range(eval_xs.shape[1])) - set(categorical_feats)
        for col in feats:
            try:
                pt.fit(eval_xs[0:eval_position, col:col + 1])
                trans = pt.transform(eval_xs[:, col:col + 1])
                # print(scipy.stats.spearmanr(trans[~np.isnan(eval_xs[:, col:col+1])], eval_xs[:, col:col+1][~np.isnan(eval_xs[:, col:col+1])]))
                eval_xs[:, col:col + 1] = trans
            except:
                pass
        eval_xs = torch.tensor(eval_xs).float()
    warnings.simplefilter('default')

    eval_xs = eval_xs.unsqueeze(1)

    # TODO: Caution there is information leakage when to_ranking is used, we should not use it
    eval_xs = remove_outliers(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position) \
            if not normalize_to_ranking else normalize_data(to_ranking_low_mem(eval_xs))
    # Rescale X
    eval_xs = normalize_by_used_features_f(eval_xs, eval_xs.shape[-1], max_features,
                                            normalize_with_sqrt=normalize_with_sqrt)

    return eval_xs.to(device)


def run_on_encoded_data(X_enc, X_rest, y, dim_reduction_name, dim_reduction, model_name, model, encoding,
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
    assert encoding.startswith("lm__") or encoding.startswith("skrub__") or encoding.startswith("bert_custom__") or encoding.startswith("openai__") or encoding.startswith("bert_custom_pooling__")
    #TODO: make this cleaner
    # we want to eliminate certain combinations
    # passthrough and lm__ means taking the full lm embedding, which is slow if the model is not LogisticRegression
    # for skrub encodings, we don't want to use passthrough
    if dim_reduction_name == "passthrough" and not (model_name in ["LogisticRegression", "LinearRegression"]) and not encoding.startswith("skrub"):
        print("Skipping {} with {} and {}".format(model_name, dim_reduction_name, encoding))
        return None
    if dim_reduction_name != "passthrough" and encoding.startswith("skrub"):
        print("Skipping {} with {} and {}".format(model_name, dim_reduction_name, encoding))
        return None
    print("Running {} with {} and {}".format(model_name, dim_reduction_name, encoding))
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
    if X_rest is not None and X_enc is not None:
        
        # Assuming X_enc and X_rest are numpy arrays, you can get their shapes
        n_enc_columns = X_enc.shape[1]
        # names of the columns should be of format original_column_name__index
        assert all(["__" in col for col in X_enc.columns])
        original_column_names = np.unique([col.split("__")[0] for col in X_enc.columns])
        # get the indices of the columns for each original column
        encoded_columns_indices = []
        for col in original_column_names:
            indices = [i for i, c in enumerate(X_enc.columns) if c.startswith(col)]
            encoded_columns_indices.append(indices)
        print(encoded_columns_indices)
        print(len(encoded_columns_indices))
        n_rest_columns = X_rest.shape[1]

        # Create column indices for X_enc and X_rest
        enc_indices = np.arange(n_enc_columns)
        rest_indices = np.arange(n_enc_columns, n_enc_columns + n_rest_columns)
        #check
        all_indices = np.concatenate(encoded_columns_indices + [rest_indices])
        # assert no duplicates
        assert len(all_indices) == len(np.unique(all_indices))
        # assert we have all indices
        assert set(all_indices) == set(np.arange(n_enc_columns + n_rest_columns))

        # Create the ColumnTransformer
        #TODO: test this
        transformers = []
        for i in range(len(original_column_names)):
            transformers.append((f"dim_reduction_{i}", dim_reduction if isinstance(dim_reduction, str) else clone(dim_reduction), encoded_columns_indices[i]))
        transformers.append(('rest_trans', rest_trans, rest_indices))
        print(transformers)
        complete_trans = ColumnTransformer(
            transformers=transformers,
        )
        

        full_X = np.concatenate([X_enc, X_rest], axis=1)
        print(X_enc.shape, X_rest.shape, full_X.shape)
        print(complete_trans.fit_transform(full_X).shape)
    elif X_rest is not None:
        complete_trans = rest_trans
        full_X = X_rest
    else:
        complete_trans = dim_reduction
        full_X = X_enc


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
        'encoding': encoding,
        'dim_reduction': dim_reduction_name,
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

def run_catboost(X_text, X_rest, y,
                        cv, **kwargs):
    """
    X_text: np array of shape (n_samples, 1), the text feature
    X_rest: np array of shape (n_samples, n_features), additional tabular data
    y: np array of shape (n_samples,), the classifcation target
    dim_reduction_name: str, the name of the dim reduction method
    dim_reduction: sklearn transformer, the dim reduction method
    model_name: str, the name of the model
    model: sklearn model, the model
    encoding: str, the name of the encoding which was used to create X_enc
    cv: sklearn cross validator, the cross validator to use
    """

    if X_rest is not None:
        rest_trans = TableVectorizer(high_card_cat_transformer = MinHashEncoder(n_components=10, analyzer="char"),
                                    low_card_cat_transformer = OneHotEncoder(handle_unknown="ignore"),
                                    numerical_transformer=StandardScaler(),
                                    cardinality_threshold=10)
        rest_trans.fit(X_rest)
        # find the categorical features
        cat_cols = []
        for trans, name, cols in rest_trans.transformers:
            if name == "low_card_cat_transformer":
                cat_cols.extend(cols)
            if name == "high_card_cat_transformer":
                cat_cols.extend(cols)
        print("cat_cols", cat_cols)




    # pipeline = Pipeline([("encoding", complete_trans), ("model", model)])
    # #scores = cross_val_score(pipeline, full_X, y, scoring="accuracy", cv=cv)
    # # report both accuracy and roc_auc
    # scores = cross_validate(pipeline, full_X, y, scoring=["accuracy", "roc_auc_ovr"], cv=cv)
    # return {
    #     'encoding': encoding,
    #     'dim_reduction': dim_reduction_name,
    #     'model': model_name,
    #     'accuracies': scores['test_accuracy'],
    #     'roc_auc': scores['test_roc_auc_ovr'],
    #     'n_train': cv.n_train,
    #     'n_test': cv.n_test,
    #     **kwargs
    # }



# Not used rn
class TabPFNClassifierBis(TabPFNClassifier):
    def fit(self, X, y, **kwargs):
        res = super().fit((X, y), **kwargs)
        super().remove_models_from_memory()
        return res
    def predict(self, X, **kwargs):
        res = super().predict(X, **kwargs)
        super().remove_models_from_memory()
        return res