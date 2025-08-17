import torch 
import torch.nn as nn 
import torch.nn.functional as F 


embed_dim = 768 
q_proj = nn.Linear(embed_dim, embed_dim)
k_proj = nn.Linear(embed_dim, embed_dim)
v_proj = nn.Linear(embed_dim, embed_dim)

BATCH_SIZE = 64
SENTENCE_LENGTH = 6 # "My cat sat at a mat" 
NUM_HEADS = 8
head_dim = embed_dim // NUM_HEADS

x = torch.rand(BATCH_SIZE, SENTENCE_LENGTH, embed_dim)
q = q_proj(x)
k = k_proj(x)
v = v_proj(x)
# q = q.view(BATCH_SIZE, SENTENCE_LENGTH, NUM_HEADS, head_dim).transpose(1, 2)
q = q.view(BATCH_SIZE, NUM_HEADS, SENTENCE_LENGTH, head_dim)
k = k.view(BATCH_SIZE, NUM_HEADS, SENTENCE_LENGTH, head_dim)
v = v.view(BATCH_SIZE, NUM_HEADS, SENTENCE_LENGTH, head_dim)
scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
scores = F.softmax(scores, dim=-2)
print(scores.shape, v.shape)
