"""Chunked GRPO training entry point (resumable across runs).

Usage:
  python train_grpo.py --updates 3 --seed-start 0            # fresh
  python train_grpo.py --updates 3 --seed-start 3 --resume   # continue
State lives in results/adapter (LoRA) and results/train_state.json.
"""
import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import grpo

RESULTS = Path(__file__).parent / "results"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, required=True)
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(base, lcfg)
    if args.resume and (RESULTS / "adapter").exists():
        model.load_adapter(RESULTS / "adapter", adapter_name="default")
        print("resumed adapter")
    model.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    st_path = RESULTS / "train_state.json"
    if args.resume and (st_path / "x").exists() is False and (st_path.with_suffix(".pt")).exists():
        pass
    opt_path = RESULTS / "optimizer.pt"
    if args.resume and opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path, weights_only=True))
        print("resumed optimizer")

    grpo.train_chunk(tok, model, optimizer, args.seed_start, args.updates)
    model.save_pretrained(RESULTS / "adapter")
    torch.save(optimizer.state_dict(), opt_path)
    json.dump({"seed_start": args.seed_start, "updates": args.updates, "lr": args.lr},
              open(RESULTS / "train_state.json", "w"))
    print("saved adapter + optimizer")


if __name__ == "__main__":
    main()
