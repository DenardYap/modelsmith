import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0 

        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):

        B, T, C = x.shape # Batch, Sequence Length, Embedding Dim (e.g. 768 for BERT)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 4 x 5 
        # 4 x 5 
        # Reshape it for multi-head structure
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # ensuring the last 2 are sentence length and head_dim
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).traponse(1, 2)

        # q.shape = (B, self.num_heads, T, self.head_dim)
        # k.shape = (B, self.num_heads, T, self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        attn_output = weights @ v 

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(attn_output)

        return output 
    

class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout_prob=0.1):
        super().__init__() 
        self.attn = SelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, mask=None):
        """
        Here x is batch of embeddings 
        We assume it's already embed before passing in 
        """

        attn_out = self.attn(x, mask)
        attn_out = self.dropout(attn_out)
        x = x + attn_out
        x = self.norm1(x) 
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x 


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)
        ])
        self.output = nn.Linear(embed_dim, vocab_size)

    # def forward(self, x, mask=None):
    #     B, T = x.shape
    #     token_embeddings = self.token_emb(x)
    #     positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
    #     pos_embeddings = self.pos_emb(positions)
    #     x = token_embeddings + pos_embeddings

    #     for layer in self.layers:
    #         x = layer(x, mask)

    #     logits = self.output(x)
    #     return logits


    def forward(self, x, mask=None):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(x) 
        x = token_embeddings + pos_embeddings 

        for layer in self.layers:
            x = layer(x, mask)

        logits = self.output(x)
        return logits