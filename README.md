# Teaching a small AI agent to use a calculator - and to be cheap about it

This project takes a small open language model (Qwen2.5-0.5B, about half a
billion parameters - tiny by current standards) and trains it, with
reinforcement learning, to solve arithmetic word problems by using a
calculator tool - and to do it while writing as few tokens as possible.

Tokens are the billing unit of every AI API. A model that rambles costs more
money and serves fewer users per second than one that gets to the point. So
the real question of this project is not just "can RL make the model better?"
but "can RL make it better *per token spent*?"

The answer, on our small test bed: yes on both counts.

## What the agent does

The model reads a word problem (for example: "A shop buys 240 mugs at $3
each and sells them all for $1,080 total. What is the profit?"). It can
write `CALC[240*3]` to ask a built-in calculator for a result, then answers
with `ANSWER[360]`. Before training, the small base model gets 25% of
held-out problems right. After about 30 rounds of RL training, it gets 50-58%
right.

## The efficiency result (the point of the project)

We tested both the original and the trained model under hard token budgets -
the model is simply cut off once it writes more than N tokens, which is how
real cost caps work. Accuracy on 12 held-out test problems:

| Token budget | Original model | After RL training |
|---|---|---|
| 24 tokens | 0% | 17% |
| 40 tokens | 25% | 58% |
| 64 tokens | 25% | 50% |

The trained model wins at every budget. At 40 tokens it more than doubles
the original model's accuracy. At 24 tokens the original model scores zero;
the trained one still solves some problems. In plain terms: RL made the
model *accurate per token spent*, which is the property that actually saves
money in a deployed system.

## How it was trained (in plain words)

The training method is GRPO, a reinforcement-learning recipe in which the
model attempts each problem several times, and attempts that score better
than the group's average are made more likely in future, while worse ones
are made less likely. No separate "judge" model is needed, which keeps the
whole thing trainable on an ordinary machine (this project ran on 2 CPU
cores with 2 GB of RAM).

The score ("reward") each attempt gets comes from a published design called
L1-Max (paper: arXiv:2503.04697, Aggarwal & Welleck). Each problem comes
with a random token budget that is shown to the model ("answer within N
tokens"). A correct answer scores higher the further under budget it
finishes; wrong answers score nothing. On top of that we add small bonuses
for using the calculator and penalties for forgetting the final answer
format. Only the small add-on adapter (LoRA) on top of the frozen base
model is trained.

## What went wrong along the way (the honest part)

ITERATIONS.md keeps a full log. The short version: we first tried training
only on the hardest problems with an extra bonus for writing the arithmetic
expression out. It failed in an instructive way - the model learned to farm
the expression bonus without getting more answers right, accuracy dropped,
and token count went *up* instead of down. That is a classic RL failure
mode (reward hacking: the model optimizes the letter of the reward, not its
spirit). We measured it, documented it, and reverted to the last good
checkpoint. The final token-efficiency win comes from the combination of
training and hard inference-time caps, not from training alone - a finding
that lines up with published work showing length penalties are easy to get
wrong (arXiv:2510.15110).

## How this compares to published work (honest caveats)

This project is a small-scale reproduction and extension of published
ideas, not a new method, and we do not claim state of the art:

- **L1 / LCPO (arXiv:2503.04697)** showed the same reward shape controlling
  reasoning length in large math-reasoning models. We use their reward
  design; our contribution is applying it to a tiny *tool-using agent* with
  an accuracy-vs-token-budget frontier, not a new algorithm.
- **DLER (arXiv:2510.15110)** documents failure modes of length rewards and
  proposes fixes. Our reward-hacking episode is an independent, small-scale
  instance of exactly what they describe.
- The broader field is crowded and active (SelfBudgeter, LAPO, Planner-R1
  and others), so we make no novelty claim.

Our numbers are also **not like-for-like** with these papers: they use
different (much larger) models, different benchmarks, and clusters instead
of a 2 GB sandbox. Our results should be read as "the published ideas
reproduce at small scale", not as a comparison of methods.

## Papers this project builds on

- L1 / LCPO - controlling reasoning length with RL: arXiv:2503.04697
- DLER - failure modes of length penalties: arXiv:2510.15110
- LLMLingua - prompt compression: arXiv:2310.05736
- LLMLingua-2 - task-agnostic compression: arXiv:2403.12968
- TACO-RL - token-aware RL for summarization: arXiv:2409.13035
- EffComp - hybrid RL+SFT for RAG compression: arXiv:2611.02766

## Repository layout

- `agent.py` - the tool-using agent loop (prompt, calculator call, answer)
- `problems.py` - word-problem generator (training + held-out test sets)
- `grpo.py` - GRPO trainer and the L1-Max reward
- `train_grpo.py` - training entry point (checkpointed, resumable)
- `eval.py` - held-out evaluation, including the `--max-new` token-cap
  frontier experiment
- `results/` - raw logs for every run reported above
- `ITERATIONS.md` - the full honest experiment log, failures included

## Reproduce

```
pip install torch transformers peft
python eval.py --which base --n 12 --out results/eval_base.jsonl
python train_grpo.py --updates 30 --resume
python eval.py --which trained --n 12 --out results/eval_trained.jsonl
python eval.py --which trained --max-new 40 --n 12 --out results/frontier.jsonl
```

Runs on CPU; a training step takes about 25 seconds on a 2-core sandbox.
