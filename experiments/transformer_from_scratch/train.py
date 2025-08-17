import torch.nn as nn 
import torch 
from transformer import * 

num_epochs = 5
model = MiniTransformer(vocab_size=5000, embed_dim=128, num_heads=4, ff_hidden_dim=512, num_layers=4, max_seq_len=100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch  # e.g., (B, T)
        logits = model(inputs)  # (B, T, vocab_size)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
