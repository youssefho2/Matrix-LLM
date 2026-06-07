import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, n_head, n_embd, dropout, bias, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        assert n_embd % n_head == 0
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.out = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out(out)
        out = self.resid_drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, n_embd, ffn_dim, dropout, bias):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, ffn_dim, bias=bias)
        self.fc2 = nn.Linear(ffn_dim, n_embd, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(nn.functional.relu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias, block_size, ffn_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, bias=bias)
        self.ln2 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_head, n_embd, dropout, bias, block_size)
        self.mlp = MLP(n_embd, ffn_dim, dropout, bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.pos_emb = nn.Embedding(config["block_size"], config["n_embd"])
        self.drop = nn.Dropout(config["dropout"])
        self.blocks = nn.Sequential(
            *[
                Block(
                    config["n_embd"],
                    config["n_head"],
                    config["dropout"],
                    config["bias"],
                    config["block_size"],
                    config["ffn_dim"],
                )
                for _ in range(config["n_layer"])
            ]
        )
        self.ln_f = nn.LayerNorm(config["n_embd"], bias=config["bias"])
        self.head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
        self.block_size = config["block_size"]
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
