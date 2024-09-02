import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from typing import Literal
import openml
import openai
import json
from pydantic import BaseModel
from openai import OpenAI
import openai
import pandas as pd

def table_to_string(table, exclude_target=False, target_name=None):
    """
    Convert a table (DataFrame) to a string row by row with \n in between, including column names.
    
    Parameters:
    table (pd.DataFrame): The table to convert.
    exclude_target (bool): Whether to exclude the target column from the string.
    target_name (str): The name of the target column to exclude if exclude_target is True.
    
    Returns:
    str: The table as a string with rows separated by \n.
    """
    print(target_name)
    if exclude_target:
        if target_name:
            if target_name in table.columns:
                table = table.drop(columns=[target_name])
            else:
                print(f"Target name {target_name} not found in the table.")
        else:
            raise ValueError("Target name must be provided if exclude_target is True.")
    
    column_names = ' '.join(table.columns)
    rows = '\n'.join(table.apply(lambda row: ' '.join(row.astype(str)), axis=1))
    return f"{column_names}\n{rows}"


def create_full_prompt(X_train, X_test, target_name):
    """
    Shuffle the dataset and create a prompt for the model to predict the target.
    
    Parameters:
    X (pd.DataFrame): The dataset including the target column.
    target_name (str): The name of the target column.
    random_state (int): The seed for the random number generator.
    
    Returns:
    str: The full prompt for the model.
    """
    prompt = f"Given this table, predict the target {target_name} for the following lines:"
    end_prompt = "Just output a list of labels."
    train_table_str = table_to_string(X_train)
    test_table_str = table_to_string(X_test, exclude_target=True, target_name=target_name)

    full_prompt = f"{train_table_str}\n\n{prompt}\n\n{test_table_str}\n\n{end_prompt}"
    #print(full_prompt)
    return full_prompt


def get_answer(client, prompt, prediction_object, temperature=0.1, show_pricing=True):
    import time

    max_retries = 10
    for attempt in range(max_retries):
        try:
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                response_format=prediction_object,
                temperature=temperature,
            )
            break
        except openai.error.RateLimitError as e:
            print(f"Rate limit error occurred: {e}. Retrying {attempt + 1}/{max_retries}...")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print("Max retries reached. Exiting.")
                return None
        except Exception as e:
            print(f"An error occurred: {e}. Exiting.")
            return None

    if show_pricing:
        print("Prompt tokens:", completion.usage.prompt_tokens)
        print("Completion tokens:", completion.usage.completion_tokens)
        print("Total tokens:", completion.usage.total_tokens)
        print("Cost:", (0.15 / 1e6) * completion.usage.prompt_tokens + (0.6 / 1e6) * completion.usage.completion_tokens)
    return completion.choices[0].message.parsed.pred

def evaluate(X, target_name, train_size=100, test_size=100, test_batch_size=10, random_state=0, temperature=0.1):
    """
    Shuffle the dataset and split it into training and testing sets, then create a prompt for the model to predict the target.
    
    Parameters:
    X (pd.DataFrame): The dataset including the target column.
    target_name (str): The name of the target column.
    train_size (int): The number of samples in the training set.
    test_size (int): The number of samples in the testing set.
    test_batch_size (int): The number of samples in each test batch.
    random_state (int): The seed for the random number generator.
    
    Returns:
    str: The full prompt for the model.
    """
    from sklearn.utils import shuffle
    
    # Shuffle the dataset
    X = shuffle(X, random_state=random_state)

    unique_labels = np.unique(X[target_name])
    class Predictions(BaseModel):
        pred: list[Literal[tuple(map(str, unique_labels))]]
    
    # Split the dataset into training and testing sets
    X_train = X.iloc[:train_size]
    answers = []
    y_test_list = []
    from math import ceil
    for i in range(ceil(test_size / test_batch_size)):
        X_test = X.iloc[train_size + i * test_batch_size:train_size + (i + 1) * test_batch_size]
        y_test = X.iloc[train_size + i * test_batch_size:train_size + (i + 1) * test_batch_size][target_name]
        # Create the full prompt
        full_prompt = create_full_prompt(X_train, X_test, target_name)
        print(full_prompt)
        answer = get_answer(full_prompt, Predictions, temperature=temperature)
        answers.append(answer)
        y_test_list.append(y_test)
        print(X_test.shape)
    # evaluate the accuracy
    y_true = np.concatenate(y_test_list).astype(str)
    y_pred = np.concatenate(answers).astype(str)
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    # compute the accuracy of a GBDT
    X_test = X.iloc[train_size:train_size + ceil(test_size / test_batch_size) * test_batch_size]
    y_test = X_test[target_name]
    y_train = X_train[target_name]
    X_test = X_test.drop(columns=[target_name])
    X_train = X_train.drop(columns=[target_name])
    # label encode the target
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    gbdt = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    gbdt.fit(X_train, y_train)
    y_pred = gbdt.predict(X_test)
    accuracy_gbdt = np.sum(y_test == y_pred) / len(y_test)

    return y_true, y_pred, accuracy, accuracy_gbdt

from sklearn.base import BaseEstimator, ClassifierMixin

class AnswerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, temperature=0.7, test_batch_size=10, n_ensembles=5):
        self.temperature = temperature
        self.test_batch_size = test_batch_size
        self.n_ensembles = n_ensembles
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.max_n_tries_ensemble = 5

    def fit(self, X, y, target_name="target"):
        self.X_train = X.copy()
        self.target_name = target_name
        #self.X_train[self.target_name] = y
        if isinstance(y, pd.Series):
            self.y_train = y.to_numpy()
        else:
            self.y_train = y
        self.unique_labels = np.unique(y)
        class Predictions(BaseModel):
            pred: list[Literal[tuple(map(str, self.unique_labels))]]
        self.prediction_object = Predictions
        return self

    def predict(self, X):
        predictions = []
        from math import ceil
        n_batches = ceil(len(X) / self.test_batch_size)
        print(f"{n_batches} batches of {self.test_batch_size} samples each")
        for i in range(n_batches):
            X_batch = X.iloc[i * self.test_batch_size:(i + 1) * self.test_batch_size]
            ensemble_preds = np.zeros((len(X_batch), self.n_ensembles)).astype(str)
            for i in range(self.n_ensembles):
                correct_prediction = False
                n_tries = 0
                while not correct_prediction and n_tries < self.max_n_tries_ensemble:
                    print(f"Try {n_tries}")
                    X_train_permuted = self.X_train.sample(frac=1).reset_index(drop=True)
                    column_permutation = np.random.permutation(X_train_permuted.columns)
                    X_train_permuted = X_train_permuted.loc[:, column_permutation]
                    X_batch_permuted = X_batch.loc[:, column_permutation]
                    X_train_permuted[self.target_name] = self.y_train
                    full_prompt = create_full_prompt(X_train_permuted, X_batch_permuted, self.target_name)
                    batch_preds = get_answer(self.client, full_prompt, self.prediction_object, temperature=self.temperature)
                    if len(batch_preds) == len(X_batch):
                        correct_prediction = True
                    else:
                        print(f"Predicted output of length {len(batch_preds)} instead of {len(X_batch)}")
                    n_tries += 1
                ensemble_preds[:, i] = batch_preds
            averaged_preds = [max(set(ensemble_preds[i]), key=list(ensemble_preds[i]).count) for i in range(len(X_batch))]
            predictions.extend(averaged_preds)
        return np.array(predictions).astype(self.y_train.dtype)

