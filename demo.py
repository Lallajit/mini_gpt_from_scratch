import torch
import torch.nn as nn
import torch.nn.functional as F
import platform

from transformer_block import Block


print("Torch version:", torch.__version__)
print("CPU available:", torch.cpu.is_available())
print("CPU name:", platform.processor())

import sentencepiece as spm
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()
spm.SentencePieceTrainer.Train(
    input = "corpus.txt",
    model_prefix = "tokenizer",
    vocab_size = 2000,
    model_type="bpe"
)
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

ids = sp.encode(text,out_type=int)
data = torch.tensor(ids, dtype=torch.long)
print(data)

vocab_size = sp.get_piece_size()
print(vocab_size)


block_size = 128  
embedding_dim = 128
n_head = 4
n_layer = 4
lr = 1e-3
epochs = 1400

def get_batch(batch_size = 32):
    ix = torch.randint(len(data)- block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
  
    y = torch.stack([data[i+1:i+block_size+1]for i in ix])

    return x, y

'''
data: tensor([11, 32, 33, 28, 37, 14,  0, 21,  5, 16, 31, 20, 21,  6, 15, 21, 39, 26,
        34,  4, 21, 10,  7, 17, 27, 19, 21, 35, 14,  8, 24, 36, 40, 30, 12, 41,
        23, 38, 21, 31, 13,  1,  2, 29,  9, 12, 18, 42, 22, 14,  3, 25])

           11 = 'brings' [32 floating point values] = [0.66, 0.03, 0.005.......32(like numbers)]
           each an every token will have  32 floating points values.
           it is 1 dimentional vector value
'''


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape  
        tok_emb = self.token_embedding(idx) 
        
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb  
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x) 
        loss = None
        if targets is not None:
            B, T, C = logits.shape 
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) 
        return logits, loss
    
    def generate(self, idx, max_new_token):
        for _ in range(max_new_token):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim= -1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx,next_idx),dim=1)
        return idx

model = MiniGPT()
optmizer = torch.optim.AdamW(model.parameters(), lr = lr)
for step in range(epochs):
    xb,yb = get_batch()
    logits, loss = model(xb,yb)
    optmizer.zero_grad()
    loss.backward()
    optmizer.step()
    if step % 300 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")

torch.save(model.state_dict(), "mini_gpt.pth")

print("Training completed!")
print("Model saved as mini_gpt.pth")