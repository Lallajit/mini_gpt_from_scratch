import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from transformer_block import Block

# Tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

vocab_size = sp.get_piece_size()

# Model configuration
block_size = 128
embedding_dim = 128
n_head = 4
n_layer = 4

# MiniGPT
class MiniGPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.position_embedding = nn.Embedding(block_size, embedding_dim)

        self.blocks = nn.Sequential(
            *[Block(embedding_dim, block_size, n_head) for _ in range(n_layer)]
        )

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

            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_token):

        for _ in range(max_new_token):

            idx_cond = idx[:, -block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, 1)

            idx = torch.cat((idx, next_idx), dim=1)

        return idx
    
# Load trained model
model = MiniGPT()

model.load_state_dict(torch.load("mini_gpt.pth", map_location="cpu"))

model.eval()

print("Model loaded successfully!")
print("Type 'exit' to stop.")

# Interactive generation
while True:

    prompt = input("\nEnter prompt: ")

    if prompt.lower() == "exit":
        break

    ids = sp.encode(prompt, out_type=int)

    context = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():

        out = model.generate(context, max_new_token=30)

    generated_text = sp.decode(out[0].tolist())

    print("Generated:", generated_text)
