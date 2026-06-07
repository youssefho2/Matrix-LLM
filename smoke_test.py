import torch

from config import MODEL
from dataloader import TinyStoriesDataset, get_dataloader
from model import GPT
from tokenizer import BPETokenizer


def train_smoke_test():
    tokenizer = BPETokenizer(vocab_size=256)
    dummy_specials = {"<PAD>":0,"<UNK>":1,"<BOS>":2,"<EOS>":3,"a":4,"b":5,"c":6,"d":7,"e":8,"f":9}
    tokenizer.vocab = dict(dummy_specials)
    tokenizer.inv_vocab = {v:k for k,v in dummy_specials.items()}
    tokenizer.special = {k:v for k,v in dummy_specials.items() if k in {"<PAD>","<UNK>","<BOS>","<EOS>"}}
    tokenizer.merges = {}

    class DummyTok:
        special = tokenizer.special
        def encode(self, text):
            return [dummy_specials.get(ch, 1) for ch in text.lower() if ch in "abcdef"]

        def save(self, path):
            return None

    dtok = DummyTok()

    texts = ["abc", "def", "abcd", "bcdea", "aaa", "bbb", "ccc", "ddd"]
    ds = TinyStoriesDataset(texts, dtok, max_seq_len=8)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)

    model = GPT(MODEL).cpu()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step, (x, y) in enumerate(loader):
        if step >= 3:
            break
        opt.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        opt.step()
        print(f"step {step} loss {loss.item():.4f}")

    torch.save({"model_state": model.state_dict()}, "checkpoints/best.pt")
    print("Smoke test passed.")


if __name__ == "__main__":
    train_smoke_test()

