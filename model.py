import os
import torch
from transformers import BertConfig, BertModel


def load_bert(bert_path, device):
    bert_config_path = os.path.join(bert_path, 'bert_config.json')
    bert = BertModel(BertConfig(bert_config_path)).to(device)
    bert_model_path = os.path.join(bert_path, 'model.bin')
    bert.load_state_dict(torch.load(bert_model_path))
    return bert
