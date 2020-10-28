import os
import math
import torch
import torch.nn as nn
from fairseq.modules import SinusoidalPositionalEmbedding
from abc import ABC
from utils import load_json
from transformers import BertConfig, BertModel


def clean_state_dict(state_dict):
    new = {}
    for key, value in state_dict.items():
        if key in ['fc.weight', 'fc.bias']:
            continue
        new[key.replace('bert.', '')] = value
    return new


def load_bert(bert_path, device):
    bert_config_path = os.path.join(bert_path, 'config.json')
    bert = BertModel(BertConfig(**load_json(bert_config_path))).to(device)
    bert_model_path = os.path.join(bert_path, 'model.bin')
    bert.load_state_dict(clean_state_dict(torch.load(bert_model_path)))
    return bert


class CrossmodalTransformer(nn.Module, ABC):
    def __init__(self,
                 n_layers,
                 n_heads,
                 d_model,
                 attn_dropout,
                 relu_dropout,
                 emb_dropout,
                 res_dropout,
                 attn_mask,
                 scale_embedding=True):
        super(CrossmodalTransformer, self).__init__()
        self.attn_mask = attn_mask
        self.emb_scale = math.sqrt(d_model) if scale_embedding else 1.0
        # 0: padding
        self.pos = SinusoidalPositionalEmbedding(d_model, 0, init_size=128)
        self.emb_dropout = emb_dropout
        self.layers = nn.ModuleList([])
        for layer in range(n_layer):
            new_layer = TransformerEncoderBlock(
                d_model, nhead, d_model * 4, attn_dropout, res_dropout, relu_dropout
            )
            self.layers.append(new_layer)

    def forward(self, x_query, x_key=None, x_key_padding_mask=None):
        # Positional Encoder for Inputs -> (B, S) => (B, S, D)
        x_query_pos = self.pos(x_query[:, :, 0])

        # (B, S, D) => (S, B, D)
        x_query = F.dropout(
            (self.emb_scale * x_query + x_query_pos), self.emb_dropout, self.training
        ).transpose(0, 1)

        if x_key is not None:
            # in the same way
            x_key_pos = self.pos(x_key[:, :, 0])
            x_key = F.dropout(
                (self.emb_scale * x_key + x_key_pos), self.emb_dropout, self.training
            ).transpose(0, 1)
        for layer in self.layers:
            x_query = layer(x_query, x_key, attn_mask=self.attn_mask)
        return x_query


class MultimodalTransformer(nn.Module, ABC):
    def __init__(self,
                 n_layers=4,
                 n_heads=8,
                 n_classes=7,
                 only_audio=False,
                 only_text=False,
                 d_audio_orig=40,
                 d_text_orig=768,
                 d_model=64,
                 attn_dropout=.2,
                 relu_dropout=.1,
                 emb_dropout=.2,
                 res_dropout=.1,
                 out_dropout=.1,
                 attn_mask=True):
        super(MultimodalTransformer, self).__init__()
        combined_dim = d_model * 2

        # temporal convolutional layers
        # (B, d_orig, L) => (B, d_model, L)
        self.audio_encoder = nn.Conv1d(d_audio_orig, d_model, 3, bias=False)
        self.text_encoder = nn.Conv1d(d_text_orig, d_model, 3, bias=False)

        # kwargs for crossmodal transformers
        kwargs = {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'd_model': d_model,
            'attn_dropout': attn_dropout,
            'relu_dropout': relu_dropout,
            'emb_dropout': emb_dropout,
            'res_dropout': res_dropout,
            'attn_mask': attn_mask
        }

        # crossmodal transformers
        self.audio_with_text = self.get_network(**kwargs)
        self.text_with_audio = self.get_network(**kwargs)

        # self-attention layers
        self.audio_layers = self.get_network(**kwargs)
        self.text_layers = self.get_network(**kwargs)

        # Projection layers
        self.fc1 = nn.Linear(combined_dim, combined_dim)
        self.fc2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, n_classes)

    def forward(self, x_audio, x_text=None,
                a_mask=None,
                t_mask=None,
                ):
        """
            Args:
        x_vision, x_audio, x_text : input tensor -> (B, L, d)
        """
        if self.only_audio:
            #  (B, L, d) -> (L, B, d)
            x_audio = self.audio_layers(x_audio,
                                        x_key_padding_mask=None)

            if self.merge_how == 'average':
                # (bat, dim)

                features = x_audio.mean(dim=0)
            else:
                # (bat, dim)
                features = x_audio[-1]
        else:
            # for conv, (B, L, D) => (B, D, L)
            x_audio = x_audio.transpose(1, 2)
            x_text = F.dropout(x_text.transpose(1, 2), self.emb_dropout, self.training)

            # (B, D, L) => (B, L, D)
            x_audio = self.audio_encoder(x_audio).transpose(1, 2)
            x_text = self.text_encoder(x_text).transpose(1, 2)

            # Crossmodal Attention
            # out: (seq, bat, dim)
            # key masking was already applied to BERT model
            x_audio_with_text = self.audio_layers_with_text(x_audio,
                                                            x_text)
            # out: (seq, bat, dim)
            x_text_with_audio = self.text_layers_with_audio(x_text,
                                                            x_audio,
                                                            x_key_padding_mask=a_mask)

            # bat, seq, dim -> seq, bat, dim
            x_audio2 = x_audio_with_text.transpose(0, 1)
            x_text2 = x_text_with_audio.transpose(0, 1)

            x_audio2 = self.audio_layers(x_audio2)
            x_text2 = self.text_layers(x_text2)

            if self.merge_how == 'average':

                # (bat, 2*dim)
                features = torch.cat([x_audio2.mean(dim=0), x_text2.mean(dim=0)], dim=1)
            else:
                # (bat, 2*dim)
                features = torch.cat([x_audio2[-1], x_text2[-1]], dim=1)

        # --------------------

        out = F.relu(self.fc_layer1(features))
        out = self.fc_layer2(F.dropout(out, p=self.out_dropout, training=self.training))
        out = out + features

        out = self.out_layer(out)

        return out, features

    @staticmethod
    def get_network(**kwargs):
        return CrossmodalTransformer(
            n_layers=kwargs['n_layers'],
            n_heads=kwargs['n_heads'],
            d_model=kwargs['d_model'],
            attn_dropout=kwargs['attn_dropout'],
            relu_dropout=kwargs['relu_dropout'],
            emb_dropout=kwargs['emb_dropout'],
            res_dropout=kwargs['res_dropout'],
            attn_mask=kwargs['attn_mask'],
            scale_embedding=True
        )
