# create a new model
import torch.nn as nn
import torch
from tabpfn import TabPFNClassifier
from src.utils import preprocess_input
from transformers import BertModel, BertTokenizer, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from skrub import MinHashEncoder
import pandas as pd
import tiktoken
from src.models import BertAndTabPFN
import os
import openai
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv
from tqdm import tqdm
from ast import literal_eval
from tenacity import retry, stop_after_attempt, wait_random_exponential

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_batch_embeddings(texts: str, model="text-embedding-ada-002"):
    res = openai.Embedding.create(input=texts, model=model)["data"]
    return np.array([literal_eval(str(x["embedding"])) for x in res])

def encode(X, encoder_name, dataset_name=None, use_cache=True, override_cache=False):
    print("working dir", os.getcwd())
    if use_cache and dataset_name is not None and not override_cache:
        # check if the cache exists
        try:
            res = np.load(f"cache/{dataset_name}_{encoder_name.replace('/', '_')}.npy")
            print("Loaded from cache")
            return res
        except FileNotFoundError:
            print("Cache not found, computing")
            pass
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = np.array(X)
    encoder_type, encoder_params = encoder_name.split("__", 1)
    if encoder_type == "lm":
        encoder = SentenceTransformer(encoder_params)
        X = X.reshape(-1)
        res = encoder.encode(X)
    elif encoder_type == "skrub":
        if encoder_params.startswith("minhash"):
            n_components = int(encoder_params.split("_")[1])
            if len(encoder_params.split("_")) > 2:
                analyzer = encoder_params.split("_")[2]
                tokenizer = encoder_params.split("_")[3]
                if tokenizer == "none":
                    tokenizer = None
                print(f"Using {analyzer} analyser and {tokenizer} tokenizer, {n_components} components")
            else:
                analyzer = "char"
                tokenizer = None
            encoder = MinHashEncoder(n_components=n_components, analyzer=analyzer, tokenizer=tokenizer,
                                     ngram_range=(2, 4) if analyzer == "char" else (1, 3), hashing="fast" if analyzer == "char" else "murmur")
            # reshape to 2d array
            # if pandas dataframe, convert to numpy array
            res = X.reshape(-1, 1)
            res = encoder.fit_transform(res)
        else:
            raise ValueError(f"Unknown skrub encoder {encoder_params}")
    elif encoder_type == "openai":
        load_dotenv()  # take environment variables from .env.
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is not set")
        else:
            openai.api_key = openai_api_key
        embedding_model = "text-embedding-ada-002"
        embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
        #max_tokens = 8000
        #encoding = tiktoken.get_encoding(embedding_encoding)
        #n_tokens = X.combined.apply(lambda x: len(encoding.encode(x)))
        ## check that the max number of tokens is not exceeded
        #if (n_tokens > max_tokens).any():
        #    raise ValueError("The maximum number of tokens is exceeded")
        #res = np.array([get_embedding(x, engine=embedding_model) for x in X.tolist()])
        #df = pd.DataFrame(X, columns=["name"])
        #res = df.name.apply(lambda x: get_embedding(x, engine=embedding_model))
        # embed in batch of 100
        for i in tqdm(range(0, len(X), 500)):
            batch = X[i:i+500].tolist()
            res_batch = get_batch_embeddings(batch, model=embedding_model)
            if i == 0:
                res = res_batch
            else:
                res = np.concatenate([res, res_batch], axis=0)

    elif encoder_type == "bert_custom":
        #FIXME: results are not great with thisr
        transformer_name = encoder_params
        # I could instantiate just Bert but this is to check for bugs in BertAndTabPFN
        lm = BertAndTabPFN(preprocess_before_tabpfn=True, linear_translator=False, transformer_name=transformer_name,
                                dim_tabpfn=30, lora=False, disable_dropout=False).to('cuda')
        lm .eval()
        tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        texts = X.tolist()
        all_encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # print the non-padded length median and quantiles
        non_padded_lengths = np.sum(all_encoding["attention_mask"].numpy(), axis=1)
        max_length = np.quantile(non_padded_lengths, 0.95)
        all_encoding = tokenizer(texts, padding="max_length", truncation=True, max_length=int(max_length), return_tensors="pt")
        # move to gpu
        all_encoding = {k: v.to('cuda') for k, v in all_encoding.items()}
        # generate random y
        with torch.no_grad():
            res = lm (**all_encoding, y=None, return_tabpfn_input=True).cpu().detach().numpy()
    elif encoder_type == "bert_custom_pooling":
        transformer_name = encoder_params
        # I could instantiate just Bert but this is to check for bugs in BertAndTabPFN
        lm = BertAndTabPFN(preprocess_before_tabpfn=True, linear_translator=False, transformer_name=transformer_name,
                                dim_tabpfn=30, lora=False, disable_dropout=False, embedding_stragegy="mean_pooling").to('cuda')
        lm .eval()
        tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        texts = X.tolist()
        all_encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # # print the non-padded length median and quantiles
        # non_padded_lengths = np.sum(all_encoding["attention_mask"].numpy(), axis=1)
        # max_length = np.quantile(non_padded_lengths, 0.95)
        # all_encoding = tokenizer(texts, padding="max_length", truncation=True, max_length=int(max_length), return_tensors="pt")
        # move to gpu
        all_encoding = {k: v.to('cuda') for k, v in all_encoding.items()}
        # generate random y
        with torch.no_grad():
            res = lm (**all_encoding, y=None, return_tabpfn_input=True).cpu().detach().numpy()
    
    if use_cache and dataset_name is not None:
        print("Saving to cache")
        # save the cache
        np.save(f"cache/{dataset_name}_{encoder_name.replace('/', '_')}.npy", res)
    
    return res