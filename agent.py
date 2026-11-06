"""Rollout machinery: the LLM acts as a tool-using agent.

Turn 1: the model answers a word problem; it may call the calculator with
CALC[expression]. If it does, the environment replies RESULT[value] and the
model gets turn 2. The episode ends with ANSWER[number] (or a format failure).

Reward (per episode):
  +1.0  final answer correct
  +0.2  used CALC with a syntactically valid expression
  -0.2  no ANSWER tag produced
"""
import re

import torch

from problems import safe_calc

SYSTEM = (
    "You are an agent with a calculator tool. To use it, write CALC[expression] "
    "with a plain arithmetic expression. The environment then replies "
    "RESULT[value], and you finish with ANSWER[number]. If you do not need the "
    "tool, just write ANSWER[number].\n\n"
    "Example:\n"
    "Question: A fund holds 5 contracts and buys 3 more. How many now?\n"
    "You: CALC[5 + 3]\n"
    "Environment: RESULT[8]\n"
    "You: ANSWER[8]"
)


CALC_RE = re.compile(r"CALC\[([^\]]+)\]")
ANSWER_RE = re.compile(r"ANSWER\[(-?\d+(?:\.\d+)?)\]")


def build_prompt(tok, problem: str, budget: int | None = None) -> str:
    sys = SYSTEM if budget is None else SYSTEM + f"\n\nAnswer within {budget} tokens in total."
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": problem}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def generate(tok, model, prompts, max_new_tokens=64, temperature=1.0, top_p=0.95, greedy=False):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=None if greedy else temperature,
            top_p=None if greedy else top_p,
            pad_token_id=tok.eos_token_id,
        )
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]


def rollout_one(tok, model, prob, greedy=False, max_new_tokens=64, budget=None):
    """One episode. Returns dict with text segments and reward."""
    prompt = build_prompt(tok, prob["problem"], budget=budget)
    gen1 = generate(tok, model, [prompt], max_new_tokens=max_new_tokens, greedy=greedy)[0]
    segments = [("prompt", prompt), ("gen", gen1)]
    used_tool = False
    m = CALC_RE.search(gen1)
    if m:
        val = safe_calc(m.group(1))
        if val is not None:
            used_tool = True
            tool_msg = f"<|im_end|>\n<|im_start|>user\nRESULT[{val}]<|im_end|>\n<|im_start|>assistant\n"
            gen2 = generate(tok, model, [prompt + gen1 + tool_msg], max_new_tokens=max_new_tokens, greedy=greedy)[0]
            segments.append(("tool", tool_msg))
            segments.append(("gen", gen2))
            full_gen = gen1 + gen2
        else:
            full_gen = gen1
    else:
        full_gen = gen1

    am = ANSWER_RE.search(full_gen)
    correct = am is not None and int(float(am.group(1))) == prob["answer"]
    reward = (1.0 if correct else 0.0) + (0.2 if used_tool else 0.0) + (0.0 if am else -0.2)
    n_tokens = sum(len(tok.encode(t)) for k, t in segments if k == "gen")
    return {
        "segments": segments, "reward": reward, "correct": correct,
        "used_tool": used_tool, "answer_given": am.group(1) if am else None,
        "truth": prob["answer"], "problem": prob["problem"], "n_tokens": n_tokens,
    }
