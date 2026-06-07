import os

import torch

from config import DEVICE, MODEL
from model import GPT
from tokenizer import BPETokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=150, temp=0.8, top_k=50):
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
    tokenizer_path = "checkpoints/tokenizer.pt"
    ckpt_path = "checkpoints/best.pt"

    if not os.path.exists(tokenizer_path):
        print("Tokenizer checkpoint not found. Please run training first.")
        return
    if not os.path.exists(ckpt_path):
        print("Model checkpoint not found. Please run training first.")
        return

    tokenizer = BPETokenizer.load(tokenizer_path)
    model = GPT(MODEL).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Model loaded. Type your prompt. Press Ctrl+C or 'exit' to quit.\n")

    try:
        while True:
            prompt = input("You: ").strip()
            if prompt.lower() in ("exit", "quit"):
                break
            if not prompt:
                continue
            response = generate_response(model, tokenizer, prompt)
            print(f"Matrix: {response}\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
