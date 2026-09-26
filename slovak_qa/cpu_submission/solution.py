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
YEAR = re.compile(r"1\d{3}|20\d{2}")
MONTHS = set("januára februára marca apríla mája júna júla augusta septembra októbra novembra decembra január február marec apríl máj jún júl august september október november december".split())
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
        hst_all = set(stem(w) for w, _, _ in tokens(heading) if w[:1].isalnum())
        first_m, last_m = {}, {}
        for si, sent in enumerate(sents):
            ms = [i for i in sent if match[i]]
            first_m[si] = ms[0] if ms else -1
            last_m[si] = ms[-1] if ms else -1
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
                int(any(YEAR.fullmatch(ptoks[i][0]) for i in range(a, b))),
                int(any(norm(ptoks[i][0]) in MONTHS for i in range(a, b))),
                sum(1 for i in range(a, b) if pst[i] in hst_all) / max(1, b - a),
                (a - first_m[si]) if first_m[si] >= 0 else -99,
                (a - last_m[si]) if last_m[si] >= 0 else -99,
                qi["miss_content"] / max(1, qi["n_content"]),
                int(norm(ptoks[a][0]) in STOP and ptoks[a][0][:1].islower()),
                sum(1 for i in range(a, b) if not ptoks[i][0][:1].isalnum()) / max(1, b - a),
                len(sents), si / max(1, len(sents)),
                sum(1 for i in range(ss, se) if match[i]),
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


LGB_RANK = dict(objective="lambdarank", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, seed=SEED, verbose=-1, num_threads=4, deterministic=True,
                lambdarank_truncation_level=20, label_gain=list(range(11)), eval_at=[1])


def fit_span(X, y, P, Q):
    keep = (y > 0) | (np.random.RandomState(SEED).rand(len(y)) < 0.25)
    reg = lgb.train(LGB_SPAN, lgb.Dataset(X[keep], y[keep]), 400)
    Xk, yk, Qk = X[keep], y[keep], Q[keep]
    _, cnt = np.unique(Qk, return_counts=True)
    rk = lgb.train(LGB_RANK, lgb.Dataset(Xk, np.round(yk * 10).astype(int), group=cnt), 400)
    return reg, rk


def span_predict(models, X):
    reg, rk = models
    return reg.predict(X), rk.predict(X)


def pick(M, P, pred, n_packets):
    reg, rk = pred
    best = {}
    for i, (k, sp, _) in enumerate(M):
        key = (P[i], k)
        if key not in best or rk[i] > best[key][2]:
            best[key] = (reg[i], sp, rk[i])
    return {key: (v[0], v[1]) for key, v in best.items()}


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


def decide_subset(QP, best, pans, n_packets, alpha, floor):
    conf = {}
    for j, key in enumerate(QP):
        sp = best.get(key, (0.0, None))
        conf[key] = (float(pans[j]), float(np.clip(sp[0], 0.0, 1.0)), sp[1])
    out = []
    for p in range(n_packets):
        items = [conf[(p, k)] for k in range(4)]
        best_val, best_mask = -1.0, 0
        for mask in range(16):
            ok = True
            tot, mn = 0.0, 1.0
            for k in range(4):
                pa, q, sp = items[k]
                if mask >> k & 1:
                    c = pa * q
                    if sp is None or c < floor:
                        ok = False
                        break
                    tot += c
                    mn = min(mn, c)
                else:
                    tot += 1.0 - pa
            if not ok:
                continue
            val = tot / 4.0 * mn ** alpha
            if val > best_val:
                best_val, best_mask = val, mask
        out.append([[int(items[k][2][0]), int(items[k][2][1])] if best_mask >> k & 1 else [-1, -1] for k in range(4)])
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
    Q = P * 4 + np.array([m[0] for m in M])
    oof_reg = np.zeros(len(y))
    oof_rk = np.zeros(len(y))
    for f in range(5):
        tri, vai = pf[P] != f, pf[P] == f
        oof_reg[vai], oof_rk[vai] = span_predict(fit_span(X[tri], y[tri], P[tri], Q[tri]), X[vai])
    oof_span = (oof_reg, oof_rk)
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
    best_cfg, best_sub = None, -1.0
    for alpha in (0.5, 1.0, 1.5, 2.0, 3.0):
        for floor in (0.0, 0.2, 0.3, 0.4, 0.5):
            sc = grouped_score(tr, decide_subset(QP, best_oof, oof_ans, len(tr), alpha, floor))
            if sc > best_sub:
                best_cfg, best_sub = (alpha, floor), sc
    print(f"subset policy OOF {best_sub:.4f} cfg {best_cfg}", flush=True)
    ansq = [key for key, t in zip(QP, qy) if t > 0]
    print("oof picked F1 on answerable", np.mean([f1(tr.passage.iloc[p][best_oof[(p, k)][1][0]:best_oof[(p, k)][1][1]], tr.passage.iloc[p][json.loads(tr.spans.iloc[p])[k][0]:json.loads(tr.spans.iloc[p])[k][1]]) if (p, k) in best_oof else 0 for p, k in ansq]), flush=True)
    print(f"all-abstain OOF {grouped_score(tr, [[[-1, -1]] * 4 for _ in range(len(tr))]):.4f}", flush=True)

    span_m = fit_span(X, y, P, Q)
    ans_m = lgb.train(LGB_ANS, lgb.Dataset(QXa, qy), 300)
    Xt, Mt, Pt, QXt, QPt = build(te, False)
    best_t = pick(Mt, Pt, span_predict(span_m, Xt), len(te))
    pans_t = ans_m.predict(qfeat_aug(QXt, QPt, best_t))
    if best_sub > best_sc:
        preds = decide_subset(QPt, best_t, pans_t, len(te), *best_cfg)
    else:
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
