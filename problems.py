"""Synthetic arithmetic word problems with ground-truth answers.

The agent sees a short scenario (trading, inventory, pricing, ...) and must
compute the answer, optionally calling the calculator tool. Answers are
integers (division problems are generated from divisible pairs).
"""
import random

_CONTEXTS = [
    ("A trader buys {a} shares and sells {b}. How many shares remain?", lambda a, b: a - b, "sub"),
    ("A fund holds {a} contracts and buys {b} more. How many contracts does it hold now?", lambda a, b: a + b, "add"),
    ("One option contract costs {a} dollars. What do {b} contracts cost?", lambda a, b: a * b, "mul"),
    ("A portfolio gained {a} dollars over {b} equal monthly installments. How much per month?", lambda a, b: a // b, "div"),
    ("A warehouse has {a} boxes and ships out {b}. How many boxes are left?", lambda a, b: a - b, "sub"),
    ("A shop sells {a} items on Monday and {b} on Tuesday. How many items in total?", lambda a, b: a + b, "add"),
    ("Each server processes {a} requests per hour. How many requests do {b} servers process in an hour?", lambda a, b: a * b, "mul"),
    ("A team of {b} analysts splits {a} reports equally. How many reports each?", lambda a, b: a // b, "div"),
]


def make_problem(rng: random.Random):
    text_fn, fn, kind = rng.choice(_CONTEXTS)
    if kind == "sub":
        a = rng.randint(150, 999); b = rng.randint(20, a - 1)
    elif kind == "add":
        a = rng.randint(120, 899); b = rng.randint(45, 899)
    elif kind == "mul":
        a = rng.randint(12, 99); b = rng.randint(3, 12)
    else:  # div
        b = rng.randint(3, 12); q = rng.randint(12, 99); a = b * q
    return {"problem": text_fn.format(a=a, b=b), "answer": fn(a, b), "kind": kind}


_HARD = [
    # multi-step: the agent must build a two-operation expression
    ("A trader buys {a} shares at {b} dollars each and sells them all at {c} dollars each. What is the total profit?",
     lambda r: (r.randint(20, 90), r.randint(5, 40)), lambda a, b, c: a * (c - b), "profit", "a*(c-b)"),
    ("A shop buys {a} boxes at {b} dollars per box and sells each box for {c} dollars. What is the total profit?",
     lambda r: (r.randint(15, 80), r.randint(4, 30)), lambda a, b, c: a * (c - b), "profit", "a*(c-b)"),
    ("A portfolio of {a} shares gains {b} dollars per share, then pays a flat {c} dollar fee. What is the net gain?",
     lambda r: (r.randint(20, 95), r.randint(3, 25)), lambda a, b, c: a * b - c, "net", "a*b-c"),
    ("A team processes {a} orders per hour for {b} hours, and then {c} orders are cancelled. How many orders remain?",
     lambda r: (r.randint(12, 60), r.randint(3, 9)), lambda a, b, c: a * b - c, "net", "a*b-c"),
    ("A fund earns {a} dollars in January and {b} dollars in February, then pays {c} dollars in fees. What is the net?",
     lambda r: (r.randint(200, 900), r.randint(150, 850)), lambda a, b, c: a + b - c, "net", "a+b-c"),
    ("A budget of {a} dollars is split equally among {b} teams after {c} dollars are reserved. How much does each team get?",
     lambda r: (r.randint(4, 12),), None, "splitfee", "(a-c)/b"),
    ("A trader makes {a} winning trades of {b} dollars each, then loses {c} dollars on one bad trade. What is the total?",
     lambda r: (r.randint(5, 30), r.randint(10, 60)), lambda a, b, c: a * b - c, "net", "a*b-c"),
    ("Each of {a} servers handles {b} requests per hour. After {c} servers go offline, how many requests per hour remain?",
     lambda r: (r.randint(6, 30), r.randint(20, 90)), lambda a, b, c: (a - c) * b, "net", "(a-c)*b"),
]


def make_hard(rng):
    text, argfn, fn, kind, _expr = rng.choice(_HARD)
    if kind == "profit":
        a, b = argfn(rng); c = b + rng.randint(1, 30)
    elif kind == "net":
        a, b = argfn(rng); c = rng.randint(1, max(2, (a * b) // 3 if "gains" in text or "winning" in text or "processes" in text else a + b - 1))
        if "offline" in text: c = rng.randint(1, a - 1)
        if "earns" in text: c = rng.randint(1, (a + b) // 2)
    else:  # splitfee
        (b,) = argfn(rng); q = rng.randint(20, 120); c = rng.randint(1, 200); a = b * q + c
    ans = {"profit": a * (c - b), "splitfee": (a - c) // b}.get(kind)
    if ans is None:
        if "offline" in text: ans = (a - c) * b
        elif "earns" in text: ans = a + b - c
        elif "gains" in text or "winning" in text or "processes" in text or "orders per hour" in text: ans = a * b - c
    return {"problem": text.format(a=a, b=b, c=c), "answer": ans, "kind": "hard_" + kind}


_orig_make = make_problem

def make_problem(rng, mix=0.5):
    if rng.random() < mix:
        return make_hard(rng)
    return _orig_make(rng)


def train_problem(seed: int):
    return make_problem(random.Random(seed))


def eval_problems(n: int, base_seed: int = 900_000):
    return [make_problem(random.Random(base_seed + i)) for i in range(n)]


def safe_calc(expr: str):
    """Evaluate a +-*/ expression safely; return None on anything invalid."""
    import ast
    import operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.floordiv}
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            l, r = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Div) and r == 0:
                return None
            return ops[type(node.op)](l, r)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        return None
    try:
        return ev(ast.parse(expr.strip(), mode="eval"))
    except Exception:
        return None
