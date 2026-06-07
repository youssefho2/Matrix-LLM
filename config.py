import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = {
    "vocab_size": 32000,
    "n_embd": 640,
    "n_head": 8,
    "n_layer": 9,
    "block_size": 256,
    "dropout": 0.1,
    "ffn_dim": 2560,
    "bias": True,
}

TRAIN = {
    "batch_size": 8,
    "lr": 3e-4,
    "weight_decay": 0.1,
    "epochs": 5,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 500,
    "eval_interval": 1000,
    "save_interval": 5000,
    "grad_clip": 1.0,
}

DATA = {
    "dataset": "roneneldan/TinyStories",
    "assist_dataset": "HuggingFaceH4/ultrachat_200k",
    "assist_splits": ["train_sft"],
    "assist_column": "text",
    "max_seq_len": 256,
    "num_workers": 0,
    "dataset_size": 20000,
    "assist_size": 20000,
}

OUTPUT_DIR = "checkpoints"
LOG_DIR = "logs"
