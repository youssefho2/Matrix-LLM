import os

import torch

from config import DEVICE, MODEL
from tokenizer import BPETokenizer
from model import GPT


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=200, temp=1.0, top_k=None):
    model.eval()
    device = torch.device(DEVICE)
    ids = [tokenizer.special["<BOS>"]] + tokenizer.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        logits, _ = model(ids[:, -MODEL["block_size"] :])
        logits = logits[:, -1, :] / temp
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == tokenizer.special["<EOS>"]:
            break

    return tokenizer.decode(ids[0].tolist())


def main():
    device = torch.device(DEVICE)
    tokenizer = BPETokenizer.load("checkpoints/tokenizer.pt")
    model = GPT(MODEL).to(device)

    ckpt = torch.load("checkpoints/best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    prompt = "Once upon a time, "
    out = generate(model, tokenizer, prompt)
    print(out)


if __name__ == "__main__":
    main()
