mport os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from tokenizer import BPETokenizer


class TinyStoriesDataset:
    def __init__(self, texts, tokenizer, max_seq_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []
        for text in texts:
            ids = [tokenizer.special["<BOS>"]] + tokenizer.encode(text)
            ids = ids[:max_seq_len]
            while len(ids) < max_seq_len:
                ids.append(tokenizer.special["<PAD>"])
            ids = ids[:max_seq_len]
            self.data.append(ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def get_dataloader(split="train"):
    from datasets import load_dataset

    print(f"[INFO] Loading datasets (split '{split}')...")
    all_texts = []

    # Base dataset
    ds = load_dataset(cfg.DATA["dataset"], split=split)
    col = "story" if "story" in ds.column_names else next((name for name, feat in ds.features.items() if getattr(feat, "dtype", None) == "string"), None)
    if col is None:
        raise ValueError(f"Base dataset has no string column. Columns: {ds.column_names}")
    if "dataset_size" in cfg.DATA and cfg.DATA["dataset_size"] is not None:
        ds = ds.select(range(min(cfg.DATA["dataset_size"], len(ds))))
    texts = list(ds[col])
    all_texts.extend(texts)
    print(f"[INFO] Base dataset: {len(texts)} samples from '{cfg.DATA['dataset']}'")

    # Assist dataset (if configured)
    assist_name = cfg.DATA.get("assist_dataset")
    if assist_name:
        assist_split = cfg.DATA.get("assist_splits", [split])[0]
        ds_assist = load_dataset(assist_name, split=assist_split)
        assist_col = cfg.DATA.get("assist_column")
        if assist_col not in ds_assist.column_names:
            assist_col = next((name for name, feat in ds_assist.features.items() if getattr(feat, "dtype", None) == "string"), None)
        if assist_col is None:
            raise ValueError(f"Assist dataset has no string column. Columns: {ds_assist.column_names}")
        assist_limit = cfg.DATA.get("assist_size")
        if assist_limit is not None:
            ds_assist = ds_assist.select(range(min(assist_limit, len(ds_assist))))
        raw_assist = list(ds_assist[assist_col])
        assist_texts = []
        for item in raw_assist:
            if isinstance(item, list):
                assist_texts.append(" ".join(str(u) for u in item))
            else:
                assist_texts.append(str(item))
        all_texts.extend(assist_texts)
        print(f"[INFO] Assist dataset: {len(assist_texts)} samples from '{assist_name}'")

    print(f"[INFO] Total texts for tokenizer: {len(all_texts)}")

    tokenizer_path = os.path.join(cfg.OUTPUT_DIR, "tokenizer.pt")
    if os.path.exists(tokenizer_path):
        print(f"[INFO] Loading tokenizer from {tokenizer_path}...")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print(f"[INFO] Training new tokenizer on {len(all_texts)} texts (this can take a while)...")
        tokenizer = BPETokenizer(vocab_size=cfg.MODEL["vocab_size"])
        tokenizer.train(all_texts)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        tokenizer.save(tokenizer_path)
        print("[INFO] Tokenizer training complete and saved.")

    print("[INFO] Creating dataset and DataLoader...")
    dataset = TinyStoriesDataset(all_texts, tokenizer, cfg.DATA["max_seq_len"])
    loader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN["batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg.DATA["num_workers"],
        drop_last=True,
    )
    return loader, tokenizer


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_steps=50):
    model.eval()
    losses = {"train": [], "val": []}
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        for step, (x, y) in enumerate(loader):
            if step >= eval_steps:
                break
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            _, loss = model(x, y)
            losses[split].append(loss.item())
    model.train()
    return {k: sum(v) / len(v) for k, v in losses.items()}


def train():
    # Check if running in Google Colab
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False

    if in_colab:
        if os.path.exists('/content/drive'):
            cfg.OUTPUT_DIR = "/content/drive/MyDrive/matrix_llm_checkpoints"
            print(f"[INFO] Google Drive mounted. Overriding checkpoint directory to: {cfg.OUTPUT_DIR}")
        else:
            print("[WARNING] Google Drive is not mounted. Checkpoints will be saved locally inside Colab and will be deleted when the session ends.")
            print("[HELP] Recommendation: Run 'from google.colab import drive; drive.mount(\"/content/drive\")' in a Colab notebook cell before starting training.")

    print(f"[INFO] Using device: {cfg.DEVICE}")
    print(f"[INFO] Config: vocab_size={cfg.MODEL['vocab_size']}, n_embd={cfg.MODEL['n_embd']}, n_layer={cfg.MODEL['n_layer']}, n_head={cfg.MODEL['n_head']}")

    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Build dataloaders
    train_loader, tokenizer = get_dataloader("train")
    val_loader, _ = get_dataloader("validation")

    # Build model
    from model import GPT
    model = GPT(cfg.MODEL).to(cfg.DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.TRAIN["lr"],
        weight_decay=cfg.TRAIN["weight_decay"],
    )
    total_steps = len(train_loader) * cfg.TRAIN["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.TRAIN["lr"],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # Check for latest checkpoint (auto-resume)
    ckpt_path = os.path.join(cfg.OUTPUT_DIR, "latest.pt")
    start_epoch = 0
    start_batch_idx = 1
    best_val_loss = float("inf")
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0)
        start_batch_idx = ckpt.get("batch_idx", 1)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, batch {start_batch_idx}")

    # Training loop
    for epoch in range(start_epoch, cfg.TRAIN["epochs"]):
        model.train()
        pbar = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{cfg.TRAIN['epochs']}",
        )
        for batch_idx, (x, y) in pbar:
            # If resuming, skip past batches until we reach start_batch_idx in the first resumed epoch
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue

            # Move batch to device
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)

            # Forward + backward + optimize
            logits, loss = model(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Periodic evaluation and checkpointing
            if batch_idx % cfg.TRAIN["eval_interval"] == 0 and batch_idx > 0:
                losses = estimate_loss(model, train_loader, val_loader, eval_steps=50)
                print(f"Step {batch_idx}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                ckpt = {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_loss": losses["val"],
                    "best_val_loss": best_val_loss,
                }
                torch.save(ckpt, os.path.join(cfg.OUTPUT_DIR, "latest.pt"))
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    torch.save(ckpt, os.path.join(cfg.OUTPUT_DIR, "best.pt"))
                    tokenizer.save(os.path.join(cfg.OUTPUT_DIR, "tokenizer.pt"))

    print("Training complete.")


if __name__ == "__main__":
    train()
