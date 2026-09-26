def edges(seq):
    s = ["START"] + list(seq) + ["END"]
    return set(zip(s[:-1], s[1:]))


def f1(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    tp = len(a & b)
    if tp == 0:
        return 0.0
    p, r = tp / len(a), tp / len(b)
    return 2 * p * r / (p + r)


def score_one(pred, gold):
    return 0.45 * f1(pred, gold) + 0.45 * f1(edges(pred), edges(gold)) + 0.10 * float(list(pred) == list(gold))
