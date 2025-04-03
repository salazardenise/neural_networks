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
n_embd = 32
n_head = 6
n_layer = 6
dropout = 0.2  # 20% disabled and drop to zero
# -------------
# # hyperparameters - to use on a good GPU, not on a CPU or macbook
# # 
# batch_size = 64
# block_size = 256  # take 256 characters to predict the 257th character
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4  # brought down the leanring rate since network is bigger
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 384  # 384 / 6 means every head is 64 dimensions
# n_head = 6
# n_layer = 6
# dropout = 0.2
# # -------------


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

def print_cuda_info():
    print(f'{torch.cuda.is_available()=}, using device {device}')
    if device == 'cuda':
        print(f'{torch.cuda.device_count()=}') 
        print(f'{torch.cuda.current_device()=}')
        print(f'{torch.cuda.get_device_name(torch.cuda.current_device())=}')
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(torch.cuda.current_device())/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(torch.cuda.current_device())/1024**3,1), 'GB')
        print(torch.cuda.memory_summary())

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

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # here we create a tril variable, tril is not a parameter of the module
        # therefore it is a buffer, and this assings it to the module
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)  # randonly prevents some nodes from communicating
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)  # projection layer going back into residual pathway
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # concat over channel dimension
        # projection is a linear transformation of the outcome of prev line (layer)
        # aka this is the projection back into the residual pathway
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity, aka a simple multi-layer perceptron """
    # this is on a per token level, all the tokens do this independently

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # projection layer going back into residual pathway
            nn.Dropout(dropout),
            # dropout can be added just before conneciton into residual pathway
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension; n_head: the number of heads we'd like, like a group size
        super().__init__()
        # if n_embd is 32, head size should be 8, so everything works out channel wise
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # communication
        self.ffwd = FeedForward(n_embd)  # computation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # residual connections (the plus)
        # layer norm are applied directly on x first, per token transformation, normalizing initially, maybe not unit gaussian later on, optimization will determine that
        x = x + self.sa(self.ln1(x))
        x =  x + self.ffwd(self.ln2(x))
        return x

# decoder transformer
class DecoderTransformerModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # idx encodes identity of token, let's also encode token position
        # so each block position, say from 0 to 0, will also get it's own n embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # blocks of self attention head (communication) and feed forward layer (compuation)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # there should be a final LayerNorm at the end of the transformer and before the final linear layer 

        # to go from token embedding tabel to logits, need a linear language modeling layer
        self.lm_head = nn.Linear(n_embd, vocab_size)

        
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C_n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C), where torch.arange is ints from 0 to T-1, embedded thru the table to get (T,  C)
        x = tok_emb + pos_emb  # (B, T, C) + (T, C) --> right align and broadcast (B, T, C) + (1, T, C) --> (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, C_vocab_size)

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
            # crop idx to the last block_size tokens
            # we can never have more than block_size coming in because positon embedding table is not any bigger than block_size
            idx_cond = idx[:, -block_size:]
            # get the predictons
            logits, loss = self(idx_cond)  # self(idx) calls forward() function
            # focus only on the last step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1), dim=1 is the time dimension
        return idx


print_cuda_info()
model = DecoderTransformerModel()
m = model.to(device)  # moves model parameters to device, important for device cuda

# create a PyTorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):
    # every once in a while, evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.5f}, vall loss {losses['val']:.4f}")
    if device == 'cuda' and iter % 1000 == 0:
        print_cuda_info()
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
