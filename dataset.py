from torch.utils.data import Dataset


class TinyStoriesDataset:
    def __init__(self, texts, tokenizer, max_seq_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []

        for text in texts:
            ids = [self.tokenizer.special["<BOS>"]] + self.tokenizer.encode(text)
            # Truncate and ensure fixed length
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
        x = ids[:-1]   # input
        y = ids[1:]    # target
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

