# Iteration log

Honest record of design decisions, pivots, and results. Newest first.

## 2026-11-06 (iter 3) - LCPO-Max budget range fix
- Observation: with budgets sampled U(16,64) and actual generations ~85 tokens,
  the LCPO-Max term was 0 even for correct answers; reward collapsed to the
  0.1 tool-shaping constant, zero variance, all updates skipped (train_log
  updates 3-4).
- Change: budgets now U(48,112) to straddle real token usage; ALPHA stays 0.03.
  Plan: shift budget range down as tok_mean falls (curriculum), then evaluate.

## 2026-11-06 (iter 2) - adopted LCPO-Max reward (L1, arXiv:2503.04697)
- Replaced flat correctness reward with L1-Max Eq. 2:
  r = correct * clip(alpha*(budget - n_tokens) + 0.5, 0, 1), alpha=0.03,
  plus small format shaping (+0.1 valid tool call, -0.1 missing ANSWER).
  Budget per problem shown in system prompt ("Answer within N tokens").
- Reason: token-math-for-throughput framing; published, defensible reward
  design (Aggarwal & Welleck); GRPO-compatible.

## 2026-11-06 (iter 1) - problem difficulty + few-shot format
- Zero-shot: base model hallucinated "RESULT[...]" itself; 0/12 on eval.
- Added one worked CALC/RESULT/ANSWER example to the system prompt:
  base jumped to ~100% on single-step problems -> no headroom for RL.
- Added multi-step problem kinds (profit, net-after-fee, split-after-reserve,
  offline-servers) mixed 50/50 with single-step. Base eval (12 held-out
  problems, seeds 900000+): 3/12 correct (25%). Headroom established.

## 2026-11-06 (iter 0) - initial build
- Tool-using agent: Qwen2.5-0.5B-Instruct + LoRA (r=16, q/k/v/o), GRPO trainer
  (group-relative advantages, no value net), calculator tool via
  CALC[...]/RESULT[...]/ANSWER[...] protocol.
- Sandbox constraints found: 2GB RAM -> per-sequence logprob forward
  (batched forward OOM-killed); per-update checkpointing after a killed
  chunk lost work.

## 2026-11-06 (iter 4) - sampling diversity fix
- Observation: on easy problems all G=4 sampled rollouts came out identical
  (tok_mean exact integer, reward variance 0 -> skip). GRPO needs intra-group
  variance.
- Change: rollout sampling temperature 1.0 -> 1.15, top_p 0.95 -> 0.98.

## 2026-11-06 (iter 5) - OOM fix: per-sequence backward
- Observation: after temp 1.15 lengthened generations (~90 tokens), training
  died silently (SIGKILL, no traceback) once a group had reward variance and
  reached the backward pass. Cause: stacking per-sequence losses kept all G
  autograd graphs (incl. 152k-vocab log_softmax tensors) alive at once.
- Change: per-sequence loss computed and backwarded one at a time
  (grad accumulation); graph freed after each. Mathematically identical.

## 2026-11-06 (iter 6) - first checkpoint eval: 25% -> 58%
- After 27 logged updates (14 with gradients): held-out eval (same 12 problems,
  seeds 900000+, greedy) base 3/12 -> trained 7/12. All trained answers used
  the calculator. Token count not yet reduced (tok_mean still ~90; budget
  curriculum has not bitten yet - the LCPO term only pays when under budget,
  and most generations still ramble to the cap).
- Next: more updates; then budget-down curriculum; then frontier eval.

## 2026-11-06 (iter 7) - budget curriculum step down
- Accuracy has formed (58% held-out at update 27; easy kinds unanimous in
  training groups). Token count still ~90: LCPO term rarely paid because
  budgets (48-112) were generous vs actual generations.
- Change: budget range U(48,112) -> U(32,88). Roughly half of sampled budgets
  now sit below the ~90-token status quo, so concise correct answers earn
  strictly more than rambling ones. Watching for accuracy regression; if
  correct_frac drops materially, alpha goes down or the range goes back up.

## 2026-11-06 (iter 8) - correctness-signal fix under tight budgets
- Observation: with the tighter curriculum, groups where all rollouts ran over
  budget had the LCPO term clip to 0 for correct and wrong alike (update 39:
  1/4 correct, zero reward variance, skipped). The multiplicative form erases
  the correctness signal exactly when generations are long - the regime the
  curriculum creates on purpose. (Same failure family DLER, arXiv:2510.15110,
  warns about in length-penalty design.)
- Change: +0.15 additive bonus for correctness, so correct-over-budget (0.25)
  still beats wrong (0.10). LCPO term unchanged and still drives the brevity
  gradient whenever any rollout is within ~17 tokens of budget.

## 2026-11-06 (iter 9) - plateau on hard problems -> hard-only training phase
- Checkpoint @65 updates: held-out 6/12 (vs 7/12 @27) - within noise, but hard
  kinds 0/5: the multi-step problems are not cracking. Easy kinds are solved
  and their groups now mostly skip (no variance), so most remaining gradient
  was going nowhere.
- Change: HARD_ONLY=1 env switch - training problems resampled until a hard
  kind comes up. Exploration risk noted: hard groups are often all-wrong
  (skip); the +0.15 correctness bonus (iter 8) keeps variance when at least
  one rollout lands.

## 2026-11-06 (iter 10) - dense signal for hard problems: expression bonus
- Observation: hard-only phase wasted ~80% of rollouts: most hard groups are
  all-wrong -> zero variance -> skip. Wall-clock burned, no gradient.
- Change: +0.3 when the CALC expression the agent built evaluates to the true
  answer (calculator-verified), even if copying into ANSWER[...] failed. On
  multi-step problems, building the right expression IS the hard part; this
  gives group-internal variance on exactly the failures we need to fix.
  Gaming risk is bounded: the bonus requires the true numeric answer.

## 2026-11-06 (iter 11) - FAILURE: hard-only + expression bonus made things worse. Reverting to @27.
- Checkpoint @100: held-out 5/12 (from 7/12 @27), hard kinds still 0/5, and
  mean output tokens ROSE from ~90 to 123. Two honest lessons:
  1. Expression bonus was reward-hacked: the agent learned to build the right
     CALC expression and stop caring about the final ANSWER copy (several
     groups with uniform 0.4 mean = all right-expression/wrong-answer).
     A classic partial-credit failure: the intermediate reward decoupled from
     the final objective.
  2. Hard-only training starved the easy kinds; their accuracy slipped.
- Decision: restore adapter + optimizer from the @27 checkpoint commit (best
  measured state, 58% held-out) and retrain from there with the iter-8 reward
  (LCPO-Max + 0.15 correctness + format shaping, mixed problems, budget
  U(32,88)). Token brevity will be measured honestly at inference via
  generation caps (the frontier), since training-side pressure did not bite
  in this budget - consistent with DLER's warning that length penalties are
  easy to get wrong.
