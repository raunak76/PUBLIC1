import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

W_POS = 0.70
W_ADJ = 0.30
SEED = 20240611


def load(path, has_target):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            cj = json.loads(r["case_json"])
            slots = sorted(cj["slots"], key=lambda s: s["position"])
            item = {
                "id": r["id"],
                "inv": list(cj["inventory"]),
                "slots": [(bool(s["compound"]), bool(s["very_short"])) for s in slots],
            }
            if has_target:
                item["seq"] = json.loads(r["sequence_json"])
            rows.append(item)
    return rows


def slot_code(s):
    return ("C" if s[0] else "c") + ("S" if s[1] else "s")


def emit_feats(case, k, b):
    sl = case["slots"]
    cur = slot_code(sl[k])
    prv = slot_code(sl[k - 1]) if k > 0 else "BOS"
    nxt = slot_code(sl[k + 1]) if k < 7 else "EOS"
    cnt = case["cnt"][b]
    nvs = case["nvs"]
    edge = "F" if k == 0 else ("L" if k == 7 else "M")
    return [
        ("b", b),
        ("bs", b, cur),
        ("bk", b, k),
        ("bks", b, k, cur),
        ("bp", b, prv),
        ("bn", b, nxt),
        ("bpcn", b, prv, cur, nxt),
        ("bec", b, edge, cur),
        ("bcnt", b, min(cnt, 5)),
        ("bkcnt", b, edge, min(cnt, 5)),
        ("bnvs", b, cur, min(nvs, 4)),
    ]


def trans_feats(case, k, a, b):
    sl = case["slots"]
    cur = slot_code(sl[k])
    prv = slot_code(sl[k - 1])
    return [
        ("t", a, b),
        ("ts", a, b, prv, cur),
        ("tk", a, b, "F" if k == 1 else ("L" if k == 7 else "M")),
    ]


def hist_feats(case, k, hist, b):
    bl = case["bl"]
    sl = case["slots"]
    out = []
    if len(hist) >= 2:
        z, a = bl[hist[-2]], bl[hist[-1]]
        out.append(("u", z, a, b))
        out.append(("us", z, a, b, slot_code(sl[k])))
    if len(hist) >= 3:
        y, z, a = bl[hist[-3]], bl[hist[-2]], bl[hist[-1]]
        out.append(("q", y, z, a, b))
    return out


PREC = 0


def prec_feats(case, used, j):
    if not PREC:
        return []
    bl = case["bl"]
    need = case["need"]
    b = bl[j]
    out = []
    for x in range(len(bl)):
        u = used[x]
        r = need[x] - u - (1 if x == j else 0)
        out.append(("pu", b, bl[x], min(u, 2)))
        out.append(("pr", b, bl[x], min(r, 2)))
        if PREC > 1:
            out.append(("pur", b, bl[x], min(u, 2), min(r, 2)))
    return out


def prep(case, order):
    case["cnt"] = Counter(case["inv"])
    case["nvs"] = sum(1 for s in case["slots"] if s[1])
    bl = sorted(case["cnt"])
    case["bl"] = bl
    case["need"] = tuple(case["cnt"][b] for b in bl)
    case["ef"] = [[emit_feats(case, k, b) for b in bl] for k in range(8)]
    case["tf"] = [None] + [[[trans_feats(case, k, a, b) for b in bl] for a in bl] for k in range(1, 8)]
    case["sf"] = [[("st", b)] for b in bl]
    case["nf"] = [[("en", b)] for b in bl]
    case["order"] = order
    build_graph(case, order)
    if "seq" in case:
        case["gf"] = gold_feats(case)


def build_graph(case, order):
    need = case["need"]
    m = len(need)
    levels = [dict() for _ in range(9)]
    levels[0][(tuple([0] * m), ())] = 0
    edges = [[] for _ in range(8)]
    efs = [[] for _ in range(8)]
    for k in range(8):
        lst = list(levels[k].keys())
        for si, st in enumerate(lst):
            used, hist = st
            for j in range(m):
                if used[j] < need[j]:
                    nu = list(used)
                    nu[j] += 1
                    ns = (tuple(nu), (hist + (j,))[-order:])
                    if ns not in levels[k + 1]:
                        levels[k + 1][ns] = len(levels[k + 1])
                    a = hist[-1] if hist else -1
                    edges[k].append((si, levels[k + 1][ns], a, j))
                    fs = list(case["ef"][k][j])
                    if k == 0:
                        fs += case["sf"][j]
                    else:
                        fs += case["tf"][k][a][j]
                        fs += hist_feats(case, k, hist, case["bl"][j])
                    if k == 7:
                        fs += case["nf"][j]
                    fs += prec_feats(case, used, j)
                    efs[k].append(filt(fs))
    case["edges"] = edges
    case["efs"] = efs
    case["nlev"] = [len(l) for l in levels]


def logsumexp(xs):
    mx = max(xs)
    return mx + math.log(sum(math.exp(x - mx) for x in xs))


DROP = set()


def filt(fs):
    return [f for f in fs if f[0] not in DROP]


def score_fs(w, fs):
    s = 0.0
    for f in fs:
        v = w.get(f)
        if v is not None:
            s += v
    return s


def forward_backward(case, w):
    es = [[score_fs(w, fs) for fs in case["efs"][k]] for k in range(8)]
    lv = case["nlev"]
    alpha = [[-1e30] * lv[k] for k in range(9)]
    alpha[0][0] = 0.0
    for k in range(8):
        acc = defaultdict(list)
        for e, (si, ti, a, j) in enumerate(case["edges"][k]):
            acc[ti].append(alpha[k][si] + es[k][e])
        for ti, xs in acc.items():
            alpha[k + 1][ti] = logsumexp(xs)
    beta = [[-1e30] * lv[k] for k in range(9)]
    for i in range(lv[8]):
        beta[8][i] = 0.0
    for k in range(7, -1, -1):
        acc = defaultdict(list)
        for e, (si, ti, a, j) in enumerate(case["edges"][k]):
            acc[si].append(beta[k + 1][ti] + es[k][e])
        for si, xs in acc.items():
            beta[k][si] = logsumexp(xs)
    logZ = beta[0][0]
    marg = []
    for k in range(8):
        mk = []
        for e, (si, ti, a, j) in enumerate(case["edges"][k]):
            mk.append(math.exp(alpha[k][si] + es[k][e] + beta[k + 1][ti] - logZ))
        marg.append(mk)
    return marg


def gold_feats(case):
    bl = case["bl"]
    idx = {b: i for i, b in enumerate(bl)}
    p = [idx[b] for b in case["seq"]]
    order = case["order"]
    fs = []
    for k in range(8):
        hist = tuple(p[max(0, k - order):k])
        a = p[k - 1] if k >= 1 else -1
        fs += case["ef"][k][p[k]]
        if k == 0:
            fs += case["sf"][p[k]]
        else:
            fs += case["tf"][k][a][p[k]]
            fs += hist_feats(case, k, hist, bl[p[k]])
        if k == 7:
            fs += case["nf"][p[k]]
        used = tuple(sum(1 for t in p[:k] if t == x) for x in range(len(bl)))
        fs += prec_feats(case, used, p[k])
    return filt(fs)


def train(cases, epochs, lr, l2, seed):
    rng = random.Random(seed)
    w = {}
    g2 = {}
    order = list(range(len(cases)))
    for ep in range(epochs):
        rng.shuffle(order)
        for ci in order:
            case = cases[ci]
            grad = defaultdict(float)
            for f in case["gf"]:
                grad[f] += 1.0
            marg = forward_backward(case, w)
            for k in range(8):
                efk = case["efs"][k]
                for e, pm in enumerate(marg[k]):
                    if pm < 1e-9:
                        continue
                    for f in efk[e]:
                        grad[f] -= pm
            for f, g in grad.items():
                v = w.get(f, 0.0)
                g = g - l2 * v
                h = g2.get(f, 1e-6) + g * g
                g2[f] = h
                w[f] = v + lr * g / math.sqrt(h)
    return w


def gen_weights(cases, a1=1.0, a2=2.0, a3=3.0):
    uni = Counter()
    bi = defaultdict(Counter)
    tri = defaultdict(Counter)
    st = Counter()
    em = {"bs": defaultdict(Counter), "bp": defaultdict(Counter), "bn": defaultdict(Counter)}
    for c in cases:
        q = c["seq"]
        sl = c["slots"]
        st[q[0]] += 1
        for k in range(8):
            uni[q[k]] += 1
            if k >= 1:
                bi[q[k - 1]][q[k]] += 1
            if k >= 2:
                tri[(q[k - 2], q[k - 1])][q[k]] += 1
            em["bs"][q[k]][slot_code(sl[k])] += 1
            em["bp"][q[k]][slot_code(sl[k - 1]) if k > 0 else "BOS"] += 1
            em["bn"][q[k]][slot_code(sl[k + 1]) if k < 7 else "EOS"] += 1
    V = sorted(uni)
    N = sum(uni.values())
    pu = {b: (uni[b] + a1) / (N + a1 * len(V)) for b in V}
    pb = {}
    w = {}
    for a in V:
        ta = sum(bi[a].values())
        for b in V:
            p = (bi[a][b] + a2 * pu[b]) / (ta + a2)
            pb[(a, b)] = p
            w[("t", a, b)] = math.log(p)
    for (z, a), cnt in tri.items():
        t = sum(cnt.values())
        for b in V:
            p = (cnt[b] + a3 * pb[(a, b)]) / (t + a3)
            w[("u", z, a, b)] = math.log(p) - math.log(pb[(a, b)])
    ts = sum(st.values())
    for b in V:
        w[("st", b)] = math.log((st[b] + a2 * pu[b]) / (ts + a2)) - math.log(pu[b])
    for tag, tab in em.items():
        for b in V:
            tot = sum(tab[b].values())
            keys = ["cs", "cS", "Cs", "CS"] + (["BOS"] if tag == "bp" else ["EOS"] if tag == "bn" else [])
            for kk in keys:
                w[(tag, b, kk)] = math.log((tab[b][kk] + 0.5) / (tot + 0.5 * len(keys)))
    for b in V:
        w[("b", b)] = math.log(pu[b])
    return w


def marginals_T(case, w, T):
    return marginals(case, {f: v * T for f, v in w.items()})


def marginals(case, w):
    marg = forward_backward(case, w)
    m = len(case["bl"])
    P = [[0.0] * m for _ in range(8)]
    Q = [[[0.0] * m for _ in range(m)] for _ in range(8)]
    for k in range(8):
        for e, (si, ti, a, j) in enumerate(case["edges"][k]):
            pm = marg[k][e]
            P[k][j] += pm
            if k >= 1:
                Q[k][a][j] += pm
    return P, Q


def mbr_decode(case, P, Q):
    need = case["need"]
    m = len(need)
    best = {(tuple([0] * m), -1): (0.0, [])}
    for k in range(8):
        nb = {}
        for (used, a), (sc, path) in best.items():
            for j in range(m):
                if used[j] < need[j]:
                    s = sc + W_POS / 8.0 * P[k][j]
                    if k >= 1:
                        s += W_ADJ / 7.0 * Q[k][a][j]
                    nu = list(used)
                    nu[j] += 1
                    key = (tuple(nu), j)
                    if key not in nb or s > nb[key][0] + 1e-15:
                        nb[key] = (s, path + [j])
        best = nb
    sc, path = max(best.values(), key=lambda x: (x[0], [-v for v in x[1]]))
    return [case["bl"][j] for j in path]


def average_marginals(ms):
    n = len(ms)
    P = [[sum(x[0][k][j] for x in ms) / n for j in range(len(ms[0][0][k]))] for k in range(8)]
    Q = [[[sum(x[1][k][a][j] for x in ms) / n for j in range(len(ms[0][1][k][a]))] for a in range(len(ms[0][1][k]))] for k in range(8)]
    return P, Q


def row_score(pred, gold):
    P = sum(1 for a, b in zip(pred, gold) if a == b) / 8.0
    A = sum(1 for i in range(7) if pred[i] == gold[i] and pred[i + 1] == gold[i + 1]) / 7.0
    return W_POS * P + W_ADJ * A


def row_chance(gold):
    c = Counter(gold)
    cp = sum(v * v for v in c.values()) / 64.0
    ca = 0.0
    for i in range(7):
        b, d = gold[i], gold[i + 1]
        ca += (c[b] * c[d] if b != d else c[b] * (c[b] - 1)) / 56.0
    ca /= 7.0
    return W_POS * cp + W_ADJ * ca


def metric(preds, golds):
    mr = sum(row_score(p, g) for p, g in zip(preds, golds)) / len(golds)
    mc = sum(row_chance(g) for g in golds) / len(golds)
    return max(0.0, min(1.0, (mr - mc) / (1 - mc)))


CONFIG = {"order": 2, "epochs": 10, "lr": 0.05, "l2": 0.01}
N_SEEDS = 3


def fit_predict(train_cases, test_cases, cfg):
    ms = [[] for _ in test_cases]
    for s in range(N_SEEDS):
        w = train(train_cases, cfg["epochs"], cfg["lr"], cfg["l2"], SEED + 97 * s)
        for i, c in enumerate(test_cases):
            ms[i].append(marginals(c, w))
    return [mbr_decode(c, *average_marginals(m)) for c, m in zip(test_cases, ms)]


def main():
    data_dir, out_path = sys.argv[1], sys.argv[2]
    # Inputs are limited to the released public train and test files.
    tr = load(os.path.join(data_dir, "train.csv"), True)
    te = load(os.path.join(data_dir, "test.csv"), False)
    for c in tr + te:
        prep(c, CONFIG["order"])

    idx = list(range(len(tr)))
    random.Random(SEED).shuffle(idx)
    preds = [None] * len(tr)
    for f in range(5):
        vset = set(idx[f::5])
        trn = [tr[i] for i in idx if i not in vset]
        val = [tr[i] for i in idx[f::5]]
        for i, p in zip(idx[f::5], fit_predict(trn, val, CONFIG)):
            preds[i] = p
    print(json.dumps({"cv_score": round(metric(preds, [c["seq"] for c in tr]), 5)}), file=sys.stderr, flush=True)

    preds = fit_predict(tr, te, CONFIG)
    d = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(d, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "sequence_json"])
        for c, p in zip(te, preds):
            assert Counter(p) == Counter(c["inv"]) and len(p) == 8
            wr.writerow([c["id"], json.dumps(p, separators=(",", ":"))])


if __name__ == "__main__":
    main()
