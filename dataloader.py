import os
from torch.utils.data import DataLoader

import config as cfg
from model import GPT
from tokenizer import CharTokenizer, BPETokenizer
from train import train


def build_dataloaders(tokenizer, split="train", streaming=False):
    if streaming:
        from datasets import load_dataset

        ds = load_dataset(cfg.DATA["dataset"], split=split, streaming=streaming)
        texts = ds["story"].take(cfg.DATA.get("dataset_size", 50000))
        tokenizer = BPETokenizer(vocab_size=cfg.MODEL["vocab_size"])
        tokenizer.train(list(texts), target_vocab=cfg.MODEL["vocab_size"])
    else:
        from datasets import load_dataset

        ds = load_dataset(cfg.DATA["dataset"], split=split)
        texts = ds["story"]
        extra_size = cfg.DATA.get("dataset_size", None)

        if extra_size:
            texts = texts[:extra_size]
        if split == "train" and tokenizer.vocab_size < 500:
            tokenizer = BPETokenizer(vocab_size=cfg.MODEL["vocab_size"])
            tokenizer.train(texts, target_vocab=cfg.MODEL["vocab_size"])
            tokenizer.save(cfg.OUTPUT_DIR + "/tokenizer.pt")
    texts_list = list(texts) if hasattr(texts, "__len__") else list(texts)

    dataset_obj = TinyStoriesDataset(texts_list, tokenizer, cfg.DATA["max_seq_len"])

    loader = DataLoader(
        dataset_obj,
        batch_size=cfg.TRAIN["batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg.DATA["num_workers"],
        drop_last=True,
    )
    return loader, tokenizer


def build_best_dataloaders(tokenizer: BPETokenizer, split="train", streaming=False):
    if streaming:
        from datasets import load_dataset

        ds = load_dataset(cfg.DATA["dataset"], split=split, streaming=streaming)
        texts = ds["story"].take(cfg.DATA.get("dataset_size", 50000))
        tokenizer = BPETokenizer(vocab_size=cfg.MODEL["vocab_size"])
        tokenizer.train(list(texts), target_vocab=cfg.MODEL["vocab_size"])
    else:
        from datasets import load_dataset

        ds = load_dataset(cfg.DATA["dataset"], split=split)
        texts = ds["story"]
        extra_size = cfg.DATA.get("dataset_size", None)

        if extra_size:
            texts = texts[:extra_size]
        if split == "train" and tokenizer.vocab_size < 500:
            tokenizer = BPETokenizer(vocab_size=cfg.MODEL["vocab_size"])
            tokenizer.train(texts, target_vocab=cfg.MODEL["vocab_size"])
            tokenizer.save(cfg.OUTPUT_DIR + "/tokenizer.pt")
    texts_list = list(texts) if hasattr(texts, "__len__") else list(texts)

    dataset_obj = BestTinyStoriesDataset(texts_list, tokenizer, cfg.DATA["max_seq_len"])

    loader = DataLoader(
        dataset_obj,
        batch_size=cfg.TRAIN["batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg.DATA["num_workers"],
        drop_last=True,
    )
    return loader, tokenizer


class TinyStoriesDataset:
    def __init__(self, texts, tokenizer, max_seq_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []

        for text in texts:
            ids = [self.tokenizer.special["<BOS>"]] + self.tokenizer.encode(text)
            ids = ids[: self.max_seq_len]
            pad = [self.tokenizer.special["<PAD>"]]
            while len(ids) < self.max_seq_len:
                ids.extend(pad)
            ids = ids[: self.max_seq_len]
            self.data.append(ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        x = ids[:-1]
        y = ids[1:]
        from torch import tensor
        return tensor(x, dtype=torch.long), tensor(y, dtype=torch.long)

class BestTinyStoriesDataset:
    def __init__(self, texts, tokenizer, max_seq_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []

        for text in texts:
            ids = [self.tokenizer.special["<BOS>"]] + self.tokenizer.encode(text)
            ids = ids[: self.max_seq_len]
            pad = [self.tokenizer.special["<PAD>"]]
            while len(ids) < self.max_seq_len:
                ids.extend(pad)
            ids = ids[: self.max_seq_len]
            self.data.append(ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        x = ids[:-1]
        y = ids[1:]
        from torch import tensor
        return tensor(x, dtype=torch.long), tensor(y, dtype=torch.long)

