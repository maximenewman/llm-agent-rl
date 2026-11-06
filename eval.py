"""Greedy evaluation on held-out problems: base model vs RL-trained adapter.

Usage: python eval.py --n 20 --offset 0 --which base|trained --out results/eval_base.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import agent
from problems import eval_problems

RESULTS = Path(__file__).parent / "results"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--which", choices=["base", "trained"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    if args.which == "trained":
        model.load_adapter(RESULTS / "adapter", adapter_name="default")
    model.eval()

    probs = eval_problems(200)[args.offset : args.offset + args.n]
    rows = []
    for i, p in enumerate(probs):
        if args.which == "base":
            with model.disable_adapter():
                r = agent.rollout_one(tok, model, p, greedy=True)
        else:
            r = agent.rollout_one(tok, model, p, greedy=True)
        row = {"idx": args.offset + i, "kind": p["kind"], "correct": r["correct"],
               "used_tool": r["used_tool"], "answer_given": r["answer_given"], "truth": r["truth"],
               "n_tokens": r.get("n_tokens")}
        rows.append(row)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(i, p["kind"], "correct" if r["correct"] else "wrong", "tool" if r["used_tool"] else "", flush=True)
    acc = sum(r["correct"] for r in rows) / max(1, len(rows))
    print(f"chunk acc {acc:.2f} ({sum(r['correct'] for r in rows)}/{len(rows)})")


if __name__ == "__main__":
    main()
