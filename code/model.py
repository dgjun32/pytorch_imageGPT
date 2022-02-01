import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, List, Union, bool


def future_mask(seq_len: int) -> torch.BoolTensor:
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
    mask = mask.triu(1)
    return mask


class PixelEmbedding(nn.Module):
    def __init__(self,
                n_vocabs: int,
                h_dim: int):
        super(self, PixelEmbedding).__init__()
        self.embedding = nn.Embedding(n_vocabs, h_dim)

    def forward(self, x : torch.LongTensor) -> torch.FloatTensor:
        return self.embedding(x)


class PositionwiseFFN(nn.Module):
    def __init__(self,
                h_dim: int,
                rate: int = 4,
                dropout: float = 0.1):
        self.mlp_1 = nn.Linear(h_dim, h_dim*rate)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.mlp_2 = nn.Linear(h_dim*rate, h_dim)
    
    def forward(self, x):
        x = self.mlp_1(x)
        x = self.sigmoid(x)*x
        x = self.dropout(x)
        x = self.mlp_2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,
                seq_len: int,
                n_heads: int,
                h_dim: int,
                dropout: float = 0.1,
                future_mask: bool = True):
        super(self, TransformerBlock).__init__()
        if future_mask:
            self.mask = future_mask(seq_len)
        else:
            self.mask = None
        self.ln_1 = nn.LayerNorm()
        self.msa = nn.MultiheadAttention(h_dim, n_heads, dropout=0.1)
        self.ln_2 = nn.LayerNorm()
        self.mlp = PositionwiseFFN(h_dim, rate=4)
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x1 = self.ln_1(x)
        x1 = self.msa(x1, x1, x1, attn_mask = self.mask) + x
        x2 = self.ln_2(x1)
        x2 = self.mlp(x2) + x1
        return x2


class ImageGPT(nn.Module):
    def __init__(self,
                type: str,
                name: str):
        super(self, ImageGPT).__init__()
        if name == 'igpt_s':
            n_layers, h_dim, n_heads, seq_len = 24, 512, 12, 32*32
        elif name == 'igpt_m':
            n_layers, h_dim, n_heads, seq_len = 36, 1024, 12, 32*32
        elif name == 'igpt_l':
            n_layers, h_dim, n_heads, seq_len = 48, 1536, 12, 32*32
        elif name == 'igpt_xl':
            n_layers, h_dim, n_heads, seq_len = 60, 3072, 12, 64*64
        
        self.pixel_embedding = PixelEmbedding(512, seq_len)
        if type == 'gpt':
            self.transformers = nn.Sequential(*[
                TransformerBlock(seq_len, n_heads, h_dim, future_mask=True) for _ in n_layers
            ])
        elif type == 'bert':
            self.transformers = nn.Sequential(*[
                TransformerBlock(seq_len, n_heads, h_dim, future_mask=False) for _ in n_layers
            ])
        self.classifier = nn.Linear(h_dim, 512)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.pixel_embedding(x)
        last_hidden_states = self.transformers(x)
        output = self.softmax(self.classifier(last_hidden_states))
        return {'logits':output,
                'last_hidden_states':last_hidden_states}

        

    



