import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


class BPETokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.special = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self._tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self._tokenizer.pre_tokenizer = ByteLevel()
        self._tokenizer.decoder = ByteLevelDecoder()
        self._trained = False

    def train(self, texts, target_vocab=None):
        if target_vocab is None:
            target_vocab = self.vocab_size
        trainer = BpeTrainer(
            vocab_size=target_vocab,
            special_tokens=list(self.special.keys()),
            show_progress=True,
        )

        def _iter():
            for text in texts:
                yield text

        self._tokenizer.train_from_iterator(_iter(), trainer)
        self._trained = True

    def encode(self, text):
        if not self._trained:
            raise RuntimeError("Tokenizer not trained")
        encoded = self._tokenizer.encode(text, add_special_tokens=False)
        return encoded.ids

    def decode(self, ids):
        filtered = [i for i in ids if i not in self.special.values()]
        return self._tokenizer.decode(filtered)

    def save(self, path):
        self._tokenizer.save(path)

    @classmethod
    def load(cls, path):
        tok = cls.__new__(cls)
        tok.vocab_size = 32000
        tok.special = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        tok._tokenizer = Tokenizer.from_file(path)
        tok._trained = True
        return tok
