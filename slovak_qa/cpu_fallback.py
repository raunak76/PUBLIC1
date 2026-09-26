import os
import sys
import json
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import lightgbm as lgb

SEED = 42
TOK = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PUNCT_BREAK = set(",;:()[]\"'„“”–-")
SENT_END = set(".!?")
WH = ["kto", "koho", "komu", "kým", "čo", "čoho", "čím", "kedy", "kde", "kam", "odkiaľ", "koľko", "ako", "aký", "aká",
      "aké", "akú", "akého", "akej", "akom", "akým", "akých", "ktorý", "ktorá", "ktoré", "ktorú", "ktorého", "ktorej",
      "ktorom", "ktorým", "ktorých", "prečo", "v", "na", "s", "z", "do", "od", "po", "pod", "pri", "za", "o", "k"]
WH_ID = {w: i for i, w in enumerate(WH)}
STOP = set("a aj ako alebo ale do je jej jeho ich k ku na nie o od po pre pri s so sa si v vo z zo za že ktorý ktorá ktoré ktorú ktorého ktorej ktorom ktorým ktorých kto čo kedy kde koľko aký aká aké akú akého akej akom akým akých bol bola bolo boli bude sú".split())


def norm(s):
    return unicodedata.normalize("NFKC", s).casefold()


def stem(w):
    w = norm(w)
    return w[:5] if len(w) > 5 else w


def tokens(s):
    return [(m.group(), m.start(), m.end()) for m in TOK.finditer(s)]


def f1(pred, gold):
    p, g = TOK.findall(norm(pred)), TOK.findall(norm(gold))
    if not p or not g:
        return float(p == g)
    c = sum((Counter(p) & Counter(g)).values())
    if c == 0:
        return 0.0
    pr, rc = c / len(p), c / len(g)
    return 2 * pr * rc / (pr + rc)


def packet_score(passage, pred, gold):
    utils, quals = [], []
    for (ps, pe), (gs, ge) in zip(pred, gold):
        if ps == -1:
            utils.append(1.0 if gs == -1 else 0.0)
            continue
        q = 0.0 if gs == -1 else f1(passage[ps:pe], passage[gs:ge])
        utils.append(q)
        quals.append(q)
    return float(np.mean(utils)) * (min(quals) if quals else 1.0)


def grouped_score(df, preds):
    s = [packet_score(r.passage, p, json.loads(r.spans)) for r, p in zip(df.itertuples(), preds)]
    return float(pd.DataFrame({"g": df.group_id.values, "s": s}).groupby("g").s.mean().mean())


def sentences(ptoks):
    out, cur = [], []
    for i, t in enumerate(ptoks):
        cur.append(i)
        if t[0] in SENT_END and (i + 1 == len(ptoks) or ptoks[i + 1][0][:1].isupper() or not ptoks[i + 1][0][:1].isalpha()):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def question_info(q, heading, pstems_set):
    qt = [t[0] for t in tokens(q)]
    words = [w for w in qt if w[:1].isalnum()]
    content = [w for w in words if norm(w) not in STOP]
    cst = set(stem(w) for w in content)
    first = norm(words[0]) if words else ""
    second = norm(words[1]) if len(words) > 1 else ""
    nums = [w for w in words if any(c.isdigit() for c in w)]
    caps = [w for w in words[1:] if w[:1].isupper()]
    hst = set(stem(w) for w, _, _ in tokens(heading) if w[:1].isalnum())
    return {
        "cst": cst,
        "wh1": WH_ID.get(first, -1),
        "wh2": WH_ID.get(second, -1),
        "n_content": len(cst),
        "miss_content": sum(1 for s in cst if s not in pstems_set and s not in hst),
        "miss_num": sum(1 for w in nums if norm(w) not in pstems_set and stem(w) not in pstems_set),
        "miss_cap": sum(1 for w in caps if stem(w) not in pstems_set and stem(w) not in hst),
        "n_num": len(nums),
        "n_cap": len(caps),
        "qlen": len(words),
        "numset": set(norm(w) for w in nums),
    }


def gen_packet(passage, heading, qs, spans, with_gold):
    ptoks = tokens(passage)
    pst = [stem(t[0]) for t in ptoks]
    pset = set(pst)
    pset |= set(norm(t[0]) for t in ptoks)
    sents = sentences(ptoks)
    feats, meta, qfeats = [], [], []
    for k, q in enumerate(qs):
        qi = question_info(q, heading, pset)
        cst = qi["cst"]
        match = np.array([s in cst for s in pst], dtype=bool)
        sscore = []
        for si, sent in enumerate(sents):
            m = set(pst[i] for i in sent if match[i])
            sscore.append(len(m) / max(1, len(cst)))
        order = np.argsort(-np.array(sscore), kind="stable")
        top = order[:2]
        gs, ge = spans[k] if with_gold else (-1, -1)
        gold_text = passage[gs:ge] if gs >= 0 else None
        cands = []
        for rank, si in enumerate(top):
            sent = sents[si]
            if not sent:
                continue
            ss, se = sent[0], sent[-1] + 1
            starts, ends = set([ss]), set([se])
            for i in range(ss, se):
                t = ptoks[i][0]
                if t in PUNCT_BREAK or t in SENT_END:
                    starts.add(i + 1)
                    ends.add(i)
                    ends.add(i + 1)
                if match[i]:
                    starts.add(i + 1)
                    ends.add(i)
                elif i > ss and match[i - 1]:
                    starts.add(i)
                if i + 1 < se and match[i + 1]:
                    ends.add(i + 1)
            for a in range(ss, se):
                for b in range(a + 1, min(se, a + 60) + 1):
                    L = b - a
                    if not ((a in starts and b in ends) or (L <= 3) or (a in starts and L <= 8) or (b in ends and L <= 8)):
                        continue
                    if ptoks[a][0] in PUNCT_BREAK or ptoks[a][0] in SENT_END:
                        continue
                    cands.append((rank, si, a, b))
        best_f1 = 0.0
        rows_k = []
        for rank, si, a, b in cands:
            sent = sents[si]
            ss, se = sent[0], sent[-1] + 1
            inm = match[a:b]
            left = [i for i in range(ss, a) if match[i]]
            right = [i for i in range(b, se) if match[i]]
            dl = a - left[-1] if left else 99
            dr = right[0] - b + 1 if right else 99
            span_text = passage[ptoks[a][1]:ptoks[b - 1][2]]
            wcount = sum(1 for i in range(a, b) if ptoks[i][0][:1].isalnum())
            f = [
                qi["wh1"], qi["wh2"], qi["n_content"], qi["qlen"], qi["n_num"], qi["n_cap"],
                qi["miss_content"], qi["miss_num"], qi["miss_cap"],
                rank, sscore[si], sscore[order[1]] if len(order) > 1 else 0.0, len(sent),
                b - a, len(span_text), wcount,
                inm.mean(), inm.sum(), len(left), len(right), dl, dr,
                int(a == ss), int(b == se),
                int(a > ss and (ptoks[a - 1][0] in PUNCT_BREAK)), int(b < se and ptoks[b][0] in PUNCT_BREAK or b < se and ptoks[b][0] in SENT_END),
                int(ptoks[b - 1][0] in PUNCT_BREAK or ptoks[b - 1][0] in SENT_END),
                sum(1 for i in range(a, b) if any(c.isdigit() for c in ptoks[i][0])),
                sum(1 for i in range(a, b) if ptoks[i][0][:1].isupper()) / max(1, b - a),
                (a - ss) / max(1, len(sent)), (se - b) / max(1, len(sent)),
                int(a > ss and match[a - 1]), int(b < se and match[b]),
                int(norm(ptoks[a][0]) in STOP), int(norm(ptoks[b - 1][0]) in STOP),
                sum(1 for i in range(a, b) if ptoks[i][0] in PUNCT_BREAK),
                int(any(norm(ptoks[i][0]) in qi["numset"] for i in range(a, b))),
            ]
            y = f1(span_text, gold_text) if gold_text is not None else 0.0
            best_f1 = max(best_f1, y)
            rows_k.append((f, (ptoks[a][1], ptoks[b - 1][2]), y))
        for f, sp, y in rows_k:
            feats.append(f)
            meta.append((k, sp, y))
        qfeats.append([qi["wh1"], qi["wh2"], qi["n_content"], qi["qlen"], qi["n_num"], qi["n_cap"], qi["miss_content"],
                       qi["miss_num"], qi["miss_cap"], sscore[order[0]] if len(order) else 0.0,
                       sscore[order[1]] if len(order) > 1 else 0.0, len(ptoks), len(cands),
                       qi["miss_content"] / max(1, qi["n_content"])])
    return feats, meta, qfeats


def build(df, with_gold):
    X, M, P, QX, QP = [], [], [], [], []
    for pi, r in enumerate(df.itertuples()):
        qs = json.loads(r.questions)
        sp = json.loads(r.spans) if with_gold else [[-1, -1]] * 4
        f, m, qf = gen_packet(r.passage, str(r.heading), qs, sp, with_gold)
        X += f
        M += m
        P += [pi] * len(f)
        QX += qf
        QP += [(pi, k) for k in range(4)]
    return np.array(X, dtype=np.float32), M, np.array(P), np.array(QX, dtype=np.float32), QP


LGB_SPAN = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, seed=SEED, verbose=-1, num_threads=4, deterministic=True)
LGB_ANS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, seed=SEED, verbose=-1, num_threads=4, deterministic=True)


def fit_span(X, y, P):
    keep = (y > 0) | (np.random.RandomState(SEED).rand(len(y)) < 0.25)
    return lgb.train(LGB_SPAN, lgb.Dataset(X[keep], y[keep]), 400)


def pick(M, P, pred, n_packets):
    best = {}
    for i, (k, sp, _) in enumerate(M):
        key = (P[i], k)
        if key not in best or pred[i] > best[key][0]:
            best[key] = (pred[i], sp)
    return best


def qfeat_aug(QX, QP, best):
    s = np.array([best.get(key, (0.0, None))[0] for key in QP], dtype=np.float32)
    return np.column_stack([QX, s])


def decide(QP, best, pans, n_packets, thr):
    out = [[[-1, -1] for _ in range(4)] for _ in range(n_packets)]
    for j, key in enumerate(QP):
        if key in best and best[key][1] is not None:
            conf = pans[j] * best[key][0]
            if conf > thr:
                out[key[0]][key[1]] = [int(best[key][1][0]), int(best[key][1][1])]
    return out


def main():
    data_dir, out_path = sys.argv[1], sys.argv[2]
    tr = pd.read_csv(os.path.join(data_dir, "train.csv"))
    te = pd.read_csv(os.path.join(data_dir, "test.csv"))
    X, M, P, QX, QP = build(tr, True)
    y = np.array([m[2] for m in M], dtype=np.float32)
    qy = np.array([json.loads(tr.spans.iloc[p])[k][0] >= 0 for p, k in QP], dtype=np.float32)
    print("train cands", X.shape, flush=True)
    groups = tr.group_id.values
    ug = np.array(sorted(set(groups)))
    rng = np.random.RandomState(SEED)
    fold_of_group = dict(zip(ug, rng.randint(0, 5, len(ug))))
    pf = np.array([fold_of_group[g] for g in groups])
    oof_span = np.zeros(len(y), dtype=np.float32)
    for f in range(5):
        tri, vai = pf[P] != f, pf[P] == f
        oof_span[vai] = fit_span(X[tri], y[tri], P[tri]).predict(X[vai])
    best_oof = pick(M, P, oof_span, len(tr))
    QXa = qfeat_aug(QX, QP, best_oof)
    qpf = np.array([pf[p] for p, _ in QP])
    oof_ans = np.zeros(len(qy), dtype=np.float32)
    for f in range(5):
        m = lgb.train(LGB_ANS, lgb.Dataset(QXa[qpf != f], qy[qpf != f]), 300)
        oof_ans[qpf == f] = m.predict(QXa[qpf == f])
    best_thr, best_sc = 0.0, -1.0
    for thr in np.arange(0.0, 1.0, 0.02):
        sc = grouped_score(tr, decide(QP, best_oof, oof_ans, len(tr), thr))
        if sc > best_sc:
            best_thr, best_sc = float(thr), sc
    print(f"grouped 5-fold OOF packet score {best_sc:.4f} thr {best_thr:.2f}", flush=True)
    print(f"all-abstain OOF {grouped_score(tr, [[[-1, -1]] * 4 for _ in range(len(tr))]):.4f}", flush=True)

    span_m = fit_span(X, y, P)
    ans_m = lgb.train(LGB_ANS, lgb.Dataset(QXa, qy), 300)
    Xt, Mt, Pt, QXt, QPt = build(te, False)
    best_t = pick(Mt, Pt, span_m.predict(Xt), len(te))
    pans_t = ans_m.predict(qfeat_aug(QXt, QPt, best_t))
    preds = decide(QPt, best_t, pans_t, len(te), best_thr)
    for r, p in zip(te.itertuples(), preds):
        for sp in p:
            if sp[0] != -1:
                assert 0 <= sp[0] < sp[1] <= min(len(r.passage), 4096)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pd.DataFrame({"id": te.id.values, "spans": [json.dumps(p, separators=(",", ":")) for p in preds]}).to_csv(out_path, index=False)
    print("wrote", out_path, len(preds), flush=True)


if __name__ == "__main__":
    main()
