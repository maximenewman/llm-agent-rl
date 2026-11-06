"""GRPO-style training (group-relative policy gradient, LoRA adapters only).

For each problem we sample G rollouts, score each with the environment reward,
and use the group mean/std to form advantages (no value network). One policy
gradient step per batch (= GRPO with a single inner epoch). The base model
stays frozen in bf16; only the LoRA adapters are trained in fp32.
"""
import json
import os
import random
import re
import time
from pathlib import Path

import torch

import agent
from problems import safe_calc, train_problem

RESULTS = Path(__file__).parent / "results"
LOG_PATH = RESULTS / "train_log.jsonl"


ALPHA = 0.03   # LCPO-Max length-penalty scale (L1 used 0.0003 for ~4k-token outputs; ours are ~48)
DELTA = 0.5    # LCPO-Max offset so correct-but-slightly-over still beats incorrect


def lcpo_max_reward(correct: bool, n_tokens: int, budget: int) -> float:
    """LCPO-Max reward, Aggarwal & Welleck 2025 (arXiv:2503.04697), Eq. 2."""
    if not correct:
        return 0.0
    return min(max(ALPHA * (budget - n_tokens) + DELTA, 0.0), 1.0)


def group_rollouts(tok, model, prob, G=4, max_new_tokens=48, budget=None):
    """Sample G episodes for one problem, batched. Returns list of episode dicts with token ids."""
    prompt = agent.build_prompt(tok, prob["problem"], budget=budget)
    p_ids = tok(prompt, return_tensors="pt").input_ids
    # turn 1: batched sampling
    enc = {"input_ids": p_ids.repeat(G, 1)}
    enc["attention_mask"] = torch.ones_like(enc["input_ids"])
    with torch.no_grad():
        out1 = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                              temperature=1.15, top_p=0.98, pad_token_id=tok.eos_token_id)
    gen1_ids = [o[p_ids.shape[1]:].tolist() for o in out1]
    gen1_txt = [tok.decode(g, skip_special_tokens=True) for g in gen1_ids]

    episodes = []
    # figure out which rollouts made a valid CALC call
    calc = []
    for i, txt in enumerate(gen1_txt):
        m = agent.CALC_RE.search(txt)
        val = safe_calc(m.group(1)) if m else None
        calc.append(val)
    turn2_idx = [i for i, v in enumerate(calc) if v is not None]
    gen2_ids, gen2_txt = {}, {}
    if turn2_idx:
        prompts2 = [prompt + gen1_txt[i] + f"<|im_end|>\n<|im_start|>user\nRESULT[{calc[i]}]<|im_end|>\n<|im_start|>assistant\n" for i in turn2_idx]
        enc2 = tok(prompts2, return_tensors="pt", padding=True, padding_side="left")
        with torch.no_grad():
            out2 = model.generate(**enc2, max_new_tokens=max_new_tokens, do_sample=True,
                                  temperature=1.15, top_p=0.98, pad_token_id=tok.eos_token_id)
        for j, i in enumerate(turn2_idx):
            g2 = out2[j][enc2["input_ids"].shape[1]:].tolist()
            gen2_ids[i] = g2
            gen2_txt[i] = tok.decode(g2, skip_special_tokens=True)

    for i in range(G):
        segs_ids = [("prompt", p_ids[0].tolist()), ("gen", gen1_ids[i])]
        full_gen = gen1_txt[i]
        used_tool = False
        ntok = len(gen1_ids[i])
        if calc[i] is not None:
            used_tool = True
            t_ids = tok(f"<|im_end|>\n<|im_start|>user\nRESULT[{calc[i]}]<|im_end|>\n<|im_start|>assistant\n",
                        add_special_tokens=False).input_ids
            segs_ids.append(("tool", t_ids))
            segs_ids.append(("gen", gen2_ids[i]))
            full_gen = full_gen + gen2_txt[i]
            ntok += len(gen2_ids[i])
        am = agent.ANSWER_RE.search(full_gen)
        correct = am is not None and int(float(am.group(1))) == prob["answer"]
        reward = lcpo_max_reward(correct, ntok, budget)
        # small additive bonuses (documented in README): correctness bonus keeps
        # the correctness signal alive when every rollout is over budget and the
        # multiplicative LCPO term clips to 0; format shaping keeps early learning alive
        reward += (0.15 if correct else 0.0) + (0.1 if used_tool else 0.0) + (0.0 if am else -0.1)
        episodes.append({"segs": segs_ids, "reward": reward, "correct": correct,
                         "used_tool": used_tool, "full_gen": full_gen, "n_tokens": ntok})
    return episodes


def seq_logprob_one(model, ids, m):
    """Mean logprob over assistant tokens for ONE sequence."""
    lm_head = model.get_output_embeddings()
    batch = torch.tensor([ids], dtype=torch.long)
    tmask = torch.tensor([m], dtype=torch.bool)
    out = model(input_ids=batch, output_hidden_states=True)
    hidden = out.hidden_states[-1][0]
    pos = torch.nonzero(tmask[0, 1:]).squeeze(-1)
    if len(pos) == 0:
        return torch.zeros((), requires_grad=True)
    h = hidden[pos]
    logits = lm_head(h).float()
    tgt = batch[0, pos + 1]
    return torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()


def seq_logprobs(model, ids_list, mask_list):
    """Per-sequence mean logprob over assistant tokens, memory-bounded.

    One sequence at a time (2GB sandbox): forward for hidden states, lm_head
    only at positions whose next token is an assistant token.
    """
    lm_head = model.get_output_embeddings()
    per_seq = []
    for ids, m in zip(ids_list, mask_list):
        batch = torch.tensor([ids], dtype=torch.long)
        tmask = torch.tensor([m], dtype=torch.bool)
        out = model(input_ids=batch, output_hidden_states=True)
        hidden = out.hidden_states[-1][0]
        pos = torch.nonzero(tmask[0, 1:]).squeeze(-1)  # logits at i predict token i+1
        if len(pos) == 0:
            per_seq.append(torch.zeros((), requires_grad=True))
            continue
        h = hidden[pos]                    # (n, d)
        logits = lm_head(h).float()        # (n, vocab)
        tgt = batch[0, pos + 1]
        logp = torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        per_seq.append(logp.mean())
    return torch.stack(per_seq)


def _save(model, optimizer):
    model.save_pretrained(RESULTS / "adapter")
    torch.save(optimizer.state_dict(), RESULTS / "optimizer.pt")


def train_chunk(tok, model, optimizer, seed_start, n_updates, G=4, lr_tag=""):
    log_rows = []
    rng = random.Random(1234 + seed_start)
    for u in range(n_updates):
        t0 = time.time()
        prob = train_problem(seed_start + u)
        if os.environ.get("HARD_ONLY") == "1":  # curriculum: concentrate on unsolved kinds
            tries = 0
            while not prob["kind"].startswith("hard") and tries < 50:
                tries += 1
                prob = train_problem(seed_start + u + 1000 * tries)
        budget = rng.randint(32, 88)
        episodes = group_rollouts(tok, model, prob, G=G, budget=budget)
        rewards = torch.tensor([e["reward"] for e in episodes])
        mean, std = rewards.mean(), rewards.std()
        row = {
            "update": seed_start + u, "kind": prob["kind"],
            "reward_mean": float(mean), "reward_max": float(rewards.max()),
            "correct_frac": float(torch.tensor([float(e["correct"]) for e in episodes]).mean()),
            "tool_frac": float(torch.tensor([float(e["used_tool"]) for e in episodes]).mean()),
            "budget": budget,
            "tok_mean": float(torch.tensor([float(e["n_tokens"]) for e in episodes]).mean()),
        }
        if std < 1e-6:
            row["skipped"] = True
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"u{u} skip (no reward variance, mean={float(mean):.2f}) [{time.time()-t0:.0f}s]", flush=True)
            continue
        advs = (rewards - mean) / (std + 1e-6)
        ids_list, mask_list = [], []
        for e in episodes:
            ids, mask = [], []
            for kind, t in e["segs"]:
                ids += t
                mask += [kind == "gen"] * len(t)
            ids_list.append(ids)
            mask_list.append(mask)
        model.train()
        optimizer.zero_grad()
        loss = 0.0
        Gn = float(len(ids_list))
        for k in range(len(ids_list)):
            lp = seq_logprob_one(model, ids_list[k], mask_list[k])
            (-(advs[k] * lp) / Gn).backward()   # graph freed each iter: 2GB RAM safe
            loss = loss + float(-(advs[k] * lp) / Gn)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        model.eval()
        row.update({"loss": float(loss), "s": round(time.time() - t0, 1)})
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        _save(model, optimizer)
        print(f"u{u} loss={float(loss):.3f} r_mean={float(mean):.2f} correct={row['correct_frac']:.2f} tool={row['tool_frac']:.2f} [{row['s']}s]", flush=True)
    return log_rows
