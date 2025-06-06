"""
see papers/blogs:
- Attention is all you need
- Deep Residual Learning for Image Recognition
- https://medium.com/data-science/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters - global variables
batch_size = 32  # number of independent sequences to process in parallel
block_size = 8  #  the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3  # self-attention doesn't tolerate high learning rates
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # ability to run on a gpu if present, and it'll be faster
eval_iters = 200
# -------------

# reproducibility
torch.manual_seed(1337)
# read data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: takes a list of integers, output a string
# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)  # important for device cuda
    return x, y

# estimate_loss() function used to average the loss over multiple batches
# torch.no_grad is a context manager, indicates to pytorch that nothing in this function will call backward() on, aka no back propagation
# this makes pytorch more efficient with memory since not expecting to call intermediate values or backward()
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # sets model to evaluation phase
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # resetting model back to training phase
    return out

# simple bigram model 
# model works the same whether in eval or train modes
# there is no dropout layer nor batch norm layer
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        # token_embedding_table has a .weight inside that stores the lookup table
        # so that when move model to device, that would move to the gpu all the calculations in the training loop
        print(f'{self.token_embedding_table.num_embeddings=}')
        print(f'at model init: {self.token_embedding_table.weight.shape=}')
        print(f'at model init: {self.token_embedding_table.weight=}')

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)  # stretches out the array to 2D
            # targets is of shape (B, T) and we want one dimension B*T, can also just use -1 in view
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictons
            logits, loss = self(idx)  # self(idx) calls forward() function
            # focus only on the last step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1), dim=1 is the time dimension
        return idx
    
model = BigramLanguageModel()
m = model.to(device)  # moves model parameters to device, important for device cuda

# create a PyTorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):
    # every once in a while, evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.5f}, vall loss {losses['val']:.4f}")
    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)  # make sure to create context on device
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
