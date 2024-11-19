from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        # Key, Query, Value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        # Regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=False)
        # AIAYN approximates GELU, although this is not longer standard practice
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

# Transformer block
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    # In AIAYN the residuals are normalized. We want gradients to flow unchanged
    # to the bottom by using unnormalized residuals. It makes optimization easier
    def forward(self, x):
        # Tranformer is just a repeated application of MapReduce
        # Attention is `reduce` and MLP is `map`
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x





@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_heads: int = 6
    n_embed: int = 384

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize the transformer module to index inside it like a dict
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            # Index with  list of layers inside the block
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)



