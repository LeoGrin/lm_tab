import torch.nn as nn
import torch
from tabpfn import TabPFNClassifier
from src.utils import preprocess_input
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin

peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
 )

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def torchPCA(X, n_components=2, return_Vt=False):
    X = X - X.mean(dim=0)
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    #return U[:, :n_components] * S[:n_components]
    if return_Vt:
        return Vt
    return X @ Vt.t()[:, :n_components]

class TorchPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
    
    def fit(self, X, y=None):
        X = torch.from_numpy(X.astype(float)).float()
        X = X - X.mean(dim=0)
        self.U, self.S, self.Vt = torch.linalg.svd(X, full_matrices=False)
        return self
    
    def transform(self, X):
        X = torch.from_numpy(X.astype(float)).float()
        X = X - X.mean(dim=0)
        X_new = X @ self.Vt.t()[:, :self.n_components]
        return X_new.numpy()

class BertAndTabPFN(nn.Module):
    #TODO I realized that we can make this simpler by using 
    # TabPFNClassifier with no_grad=False
    def __init__(self, dimension_reduction="subset", dim_tabpfn=100, preprocess_before_tabpfn=True,
                 train_tabpfn=False, transformer_name="distilroberta-base", lora=False, disable_dropout=False, embedding_strategy="cls"):
        super().__init__()
        #self.bert = BertModel.from_pretrained('bert-base-uncased')
        #self.bert = BertModel.from_pretrained('distilbert-base-uncased')
        self.bert = AutoModel.from_pretrained(transformer_name)
        # disable dropout
        if disable_dropout:
            print("Disabling dropout")
            for module in self.bert.modules():
                if isinstance(module, nn.Dropout):
                    print("Found dropout, setting p=0")
                    module.p = 0
        if lora:
            print("Using LoRA")
            self.bert = get_peft_model(self.bert, peft_config)
            self.bert.print_trainable_parameters()
        self.raw_tabpfn = TabPFNClassifier()
        self.tabpfn = self.raw_tabpfn.model[2]
        if not train_tabpfn:
            # no requires_grad for the tabpfn
            for param in self.tabpfn.parameters():
                param.requires_grad = False
        self.dim_tabpfn = dim_tabpfn
        self.preprocess_before_tabpfn = preprocess_before_tabpfn
        if dimension_reduction == "linear":
            self.linear_translator = nn.Linear(768, dim_tabpfn)
        if dimension_reduction == "pca_fixed":
            self.Vt = None
        self.dimension_reduction = dimension_reduction
        self.embedding_strategy = embedding_strategy
    
    def forward(self, input_ids, attention_mask, y, tabular_data=None, single_eval_pos=100, return_tabpfn_input=False):
        # get bert embeddings
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        if self.embedding_strategy == "cls":
            bert_embeddings = bert_outputs.last_hidden_state[:, 0, :]
        elif self.embedding_strategy == "mean_pooling":
            bert_embeddings = mean_pooling(bert_outputs, attention_mask)
            bert_embeddings = F.normalize(bert_embeddings, p=2, dim=1)
        # Dimension reduction
        if self.dimension_reduction == "linear":
            tabpfn_input = self.linear_translator(bert_embeddings)
        elif self.dimension_reduction == "subset":
            tabpfn_input = bert_embeddings[:, :self.dim_tabpfn]
        elif self.dimension_reduction == "pca":
            tabpfn_input = torchPCA(bert_embeddings, n_components=self.dim_tabpfn)
        elif self.dimension_reduction == "pca_fixed":
            if self.Vt is None:
                with torch.no_grad():
                    self.Vt = torchPCA(bert_embeddings, n_components=self.dim_tabpfn, return_Vt=True)
            tabpfn_input = bert_embeddings @ self.Vt.t()[:, :self.dim_tabpfn]
        if return_tabpfn_input:
            return tabpfn_input
        if tabular_data is not None:
            print("Using additional tabular data of shape", tabular_data.shape)
            tabpfn_input = torch.cat([tabpfn_input, tabular_data], dim=1)
        tabpfn_input = tabpfn_input.reshape(tabpfn_input.shape[0], 1, tabpfn_input.shape[1])
        #TODO don't preprocess categorical features?
        if self.preprocess_before_tabpfn:
            tabpfn_input = preprocess_input(tabpfn_input, y, single_eval_pos, preprocess_transform="none", device=input_ids.device)
        # pad with 0 to 100
        tabpfn_input = torch.cat([tabpfn_input, torch.zeros(tabpfn_input.shape[0], tabpfn_input.shape[1], 100 - tabpfn_input.shape[2], device=input_ids.device)], dim=-1)
        y = y.reshape(y.shape[0], 1)
        tabpfn_outputs = self.tabpfn((tabpfn_input, y), single_eval_pos=single_eval_pos)
        # find number of classes in train
        n_classes = y[:single_eval_pos].unique().shape[0]
        # restrict to class
        tabpfn_outputs = tabpfn_outputs[:, :n_classes]
        return tabpfn_outputs
    

# class BertEmbedder(nn.Module):
#     def __init__(self, transformer_name):
#         super().__init__()
#             # Load model from HuggingFace Hub
#         self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
#         self.model = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1')
    
#     def encode(self, sentences):
#         # get the embeddings in one batch
#         encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
#         with torch.no_grad():
#             model_output = self.model(**encoded_input)
#         return bert_embeddings
    

