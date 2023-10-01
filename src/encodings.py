# create a new model
import torch.nn as nn
import torch
from tabpfn import TabPFNClassifier
from utils import preprocess_input
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from skrub import MinHashEncoder
import pandas as pd

class BertEmbedder(nn.Module):
    def __init__(self, max_length=128):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.max_length = max_length
    
    def forward(self, sentences):
        # get the embeddings in one batch
        input_ids, attention_mask = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").values()
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        bert_embeddings = bert_outputs.last_hidden_state[:, 0, :]
        return bert_embeddings
    

def encode(X, encoder_name):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = np.array(X)
    encoder_type, encoder_params = encoder_name.split("__", 1)
    if encoder_type == "lm":
        encoder = SentenceTransformer(encoder_params)
        return encoder.encode(X)
    elif encoder_type == "skrub":
        if encoder_params.startswith("minhash"):
            n_components = int(encoder_params.split("_")[1])
            encoder = MinHashEncoder(n_components=n_components)
            # reshape to 2d array
            # if pandas dataframe, convert to numpy array
            X = X.reshape(-1, 1)
            return encoder.fit_transform(X)
        else:
            raise ValueError(f"Unknown skrub encoder {encoder_params}")