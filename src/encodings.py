# create a new model
import torch.nn as nn
import torch
from tabpfn import TabPFNClassifier
from utils import preprocess_input
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer

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
    
