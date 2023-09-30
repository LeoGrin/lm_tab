# create a new model
import torch.nn as nn
import torch
from tabpfn import TabPFNClassifier
from src.utils import preprocess_input
from transformers import BertModel

class BertAndTabPFN(nn.Module):
    def __init__(self, linear_translator=False, dim_tabpfn=100, preprocess_before_tabpfn=False,
                 train_tabpfn=False):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tabpfn = TabPFNClassifier().model[2]
        if not train_tabpfn:
            # no requires_grad for the tabpfn
            for param in self.tabpfn.parameters():
                param.requires_grad = False
        self.dim_tabpfn = dim_tabpfn
        self.preprocess_before_tabpfn = preprocess_before_tabpfn
        if linear_translator:
            self.linear_translator = nn.Linear(768, dim_tabpfn)
    
    def forward(self, input_ids, attention_mask, y, tabular_data=None, single_eval_pos=100, **fit_params):
        print("fit_params", fit_params)
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        bert_embeddings = bert_outputs.last_hidden_state[:, 0, :]
        if hasattr(self, 'linear_translator'):
            tabpfn_input = self.linear_translator(bert_embeddings)
        else:
            tabpfn_input = bert_embeddings[:, :self.dim_tabpfn]
        tabpfn_input = tabpfn_input.reshape(tabpfn_input.shape[0], 1, tabpfn_input.shape[1])
        if self.preprocess_before_tabpfn:
            tabpfn_input = preprocess_input(tabpfn_input, y, single_eval_pos, preprocess_transform="none", device=input_ids.device)
        tabpfn_outputs = self.tabpfn((tabpfn_input, y), single_eval_pos=single_eval_pos)
        return tabpfn_outputs
    