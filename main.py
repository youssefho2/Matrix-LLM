import argparse

from train import train as run_main


def main():
    parser = argparse.ArgumentParser(description="Train or generate with TinyStories GPT-2")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "generate"])
    parser.add_argument("--prompt", type=str, default="Once upon a time,")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer.pt")
    args = parser.parse_args()

    if args.mode == "train":
        run_main()
    elif args.mode == "generate":
        import torch
        from config import DEVICE, MODEL
        from model import GPT
        from generate import generate
        
        device = torch.device(DEVICE)
        tokenizer = __import__("tokenizer").BPETokenizer.load(args.tokenizer_path)
        model = GPT(MODEL).to(device)
        
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        
        out = generate(model, tokenizer, args.prompt)
        print(out)


if __name__ == "__main__":
    main()
