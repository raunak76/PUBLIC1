import sys
import json
import math
import re
import time
import itertools
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

SEED = 42
N_FOLDS = 5
TOP_SENTENCES = 3
MAX_SPAN = 45
SHORT_SPAN = 6
CLAUSE_SPAN = 12
THREADS = 4
BREAKS = set(',;:()„“"–—-')
ENDS = set('.!?')
PREPS = ["v", "vo", "na", "z", "zo", "do", "od", "po", "pri", "s", "so", "k", "ku", "za", "pre", "o", "od", "medzi", "pod", "nad", "pred", "cez", "bez", "u"]
QWORDS = ["kedy", "kde", "kto", "koľko", "čo", "prečo", "ako", "aký", "aká", "aké", "akú", "akého", "akej",
          "akom", "akým", "akou", "akí", "ktorý", "ktorá", "ktoré", "ktorú", "ktorého", "ktorej", "ktorom",
          "ktorým", "ktorí", "odkiaľ", "kam", "čím", "čom", "čoho", "komu", "koho", "kým", "je", "bol",
          "bola", "má", "môže", "v", "na", "z", "do", "od", "s", "o", "za", "pre", "po", "pri"]
QMAP = {w: i + 1 for i, w in enumerate(QWORDS)}
PMAP = {w: i + 1 for i, w in enumerate(PREPS)}
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"\w+", re.UNICODE)
YEAR_RE = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")
START = time.time()


def log(msg):
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


def norm_tokens(text):
    return TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def token_f1(pred, gold):
    if not pred or not gold:
        return float(pred == gold)
    common = sum((Counter(pred) & Counter(gold)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pred), common / len(gold)
    return 2 * p * r / (p + r)


def official_score(pred_spans, gold_spans, passages, groups):
    scores = []
    for pred, gold, passage in zip(pred_spans, gold_spans, passages):
        utils, quals = [], []
        for (ps, pe), (gs, ge) in zip(pred, gold):
            if ps < 0:
                utils.append(1.0 if gs < 0 else 0.0)
                continue
            q = 0.0 if gs < 0 else token_f1(norm_tokens(passage[ps:pe]), norm_tokens(passage[gs:ge]))
            utils.append(q)
            quals.append(q)
        scores.append(float(np.mean(utils)) * (min(quals) if quals else 1.0))
    frame = pd.DataFrame({"g": groups, "s": scores})
    return float(frame.groupby("g")["s"].mean().mean())


def stem(word, k=5):
    return word.casefold()[:k]


def load_packets(path, with_labels):
    frame = pd.read_csv(path, dtype={"id": str, "group_id": str}, keep_default_na=False)
    packets = []
    for row in frame.itertuples(index=False):
        spans = json.loads(row.spans) if with_labels else [[-1, -1]] * 4
        clean = []
        for s, e in spans:
            s, e = int(s), int(e)
            if s >= 0:
                while s < e and row.passage[s].isspace():
                    s += 1
                while e > s and row.passage[e - 1].isspace():
                    e -= 1
            clean.append((s, e) if s < e else (-1, -1))
        packets.append({"id": row.id, "group": row.group_id, "heading": row.heading, "passage": row.passage,
                        "questions": json.loads(row.questions), "spans": clean})
    return packets


def analyse_passage(pk):
    passage = pk["passage"]
    toks = [(m.start(), m.end(), m.group()) for m in TOKEN_RE.finditer(passage)]
    n = len(toks)
    sent_start = [i == 0 or (toks[i - 1][2] in ENDS and toks[i][2][:1].isupper()) for i in range(n)]
    starts = [i for i in range(n) if sent_start[i]]
    bounds = list(zip(starts, starts[1:] + [n]))
    sent_of = np.zeros(n, dtype=np.int64)
    for si, (a, b) in enumerate(bounds):
        sent_of[a:b] = si
    is_word = np.array([t[2][0].isalnum() or t[2][0] == "_" for t in toks])
    stems5 = [stem(t[2]) if w else "" for t, w in zip(toks, is_word)]
    lower = [t[2].casefold() for t in toks]
    upper = np.array([t[2][:1].isupper() for t in toks])
    digit = np.array([any(c.isdigit() for c in t[2]) for t in toks])
    year = np.array([bool(YEAR_RE.match(t[2])) for t in toks])
    start_ok = np.array([sent_start[i] or (i > 0 and toks[i - 1][2] in BREAKS) for i in range(n)])
    end_ok = np.array([i == n - 1 or toks[i + 1][2] in BREAKS or toks[i + 1][2] in ENDS or toks[i][2] in ENDS
                       for i in range(n)])
    return {"toks": toks, "bounds": bounds, "sent_of": sent_of, "is_word": is_word, "stems": stems5,
            "lower": lower, "upper": upper, "digit": digit, "year": year, "start_ok": start_ok,
            "end_ok": end_ok, "stem_set": set(stems5) | {stem(w) for w in WORD_RE.findall(pk["heading"])},
            "stem4": {w.casefold()[:4] for w in WORD_RE.findall(passage + " " + pk["heading"])},
            "digits": {t[2] for t in toks if any(c.isdigit() for c in t[2])}}


def question_info(q, idf):
    words = WORD_RE.findall(q)
    low = [w.casefold() for w in words]
    stems = {stem(w) for w in words if len(w) > 2}
    return {"words": words, "stems": stems, "idf": {s: idf(s) for s in stems},
            "q1": QMAP.get(low[0], 0) if low else 0, "q2": QMAP.get(low[1], 0) if len(low) > 1 else 0,
            "caps": [w for w in words[1:] if w[:1].isupper()], "nums": [w for w in words if w.isdigit()],
            "len": len(words)}


def span_rows(pk, pa, qi):
    toks, n = pa["toks"], len(pa["toks"])
    match = np.array([bool(s) and s in qi["stems"] for s in pa["stems"]])
    weight = np.array([qi["idf"].get(s, 0.0) if m else 0.0 for s, m in zip(pa["stems"], match)])
    sent_scores = np.array([weight[a:b].sum() for a, b in pa["bounds"]])
    q_total = sum(qi["idf"].values()) + 1e-6
    order = np.argsort(-sent_scores, kind="stable")
    rank_of = np.empty(len(order), dtype=np.int64)
    rank_of[order] = np.arange(len(order))
    feats, spans = [], []
    match_pos = np.flatnonzero(match)
    for si in order[:TOP_SENTENCES]:
        a, b = pa["bounds"][si]
        sent_len = b - a
        in_sent = match_pos[(match_pos >= a) & (match_pos < b)]
        last_match = in_sent.max() if len(in_sent) else -1
        first_match = in_sent.min() if len(in_sent) else -1
        for i in range(a, b):
            if toks[i][2] in BREAKS or toks[i][2] in ENDS:
                continue
            for j in range(i, min(b, i + MAX_SPAN)):
                ln = j - i + 1
                if not ((pa["start_ok"][i] and pa["end_ok"][j]) or ln <= SHORT_SPAN
                        or (pa["start_ok"][i] and ln <= CLAUSE_SPAN) or (pa["end_ok"][j] and ln <= CLAUSE_SPAN)):
                    continue
                words = pa["is_word"][i:j + 1]
                n_words = int(words.sum())
                n_match = int(match[i:j + 1].sum())
                inside = ((in_sent >= i) & (in_sent <= j)).sum() if len(in_sent) else 0
                before = (in_sent < i).sum() if len(in_sent) else 0
                after = (in_sent > j).sum() if len(in_sent) else 0
                dist = 0
                if len(in_sent) and inside == 0:
                    dist = min(abs(in_sent - i).min(), abs(in_sent - j).min())
                prev = toks[i - 1][2] if i > 0 else "^"
                nxt = toks[j + 1][2] if j + 1 < n else "$"
                feats.append([
                    rank_of[si], sent_scores[si], sent_scores[si] / q_total, sent_scores[si] - sent_scores[order[0]],
                    len(pa["bounds"]), si, sent_len, ln, toks[j][1] - toks[i][0], ln / sent_len,
                    int(i == a), int(j >= b - 2), int(i == a and j >= b - 2), int(toks[j][2] in ENDS),
                    int(pa["start_ok"][i]), int(pa["end_ok"][j]),
                    0 if prev == "^" else 1 if prev in ENDS else 2 if prev == "," else 3 if prev in "(„\"" else 4 if prev in BREAKS else 5,
                    0 if nxt == "$" else 1 if nxt in ENDS else 2 if nxt == "," else 3 if nxt in ")“\"" else 4 if nxt in BREAKS else 5,
                    n_words, n_match, n_match / max(1, n_words), n_words - n_match,
                    inside, before, after, dist, i - first_match if first_match >= 0 else -99,
                    i - last_match if last_match >= 0 else -99,
                    int(pa["digit"][i:j + 1].sum()), int(pa["year"][i:j + 1].sum()), int(pa["upper"][i:j + 1].sum()),
                    int(pa["upper"][i]), PMAP.get(pa["lower"][i], 0), PMAP.get(pa["lower"][j], 0),
                    qi["q1"], qi["q2"], qi["len"], len(in_sent),
                ])
                spans.append((toks[i][0], toks[j][1]))
    return np.array(feats, dtype=np.float32).reshape(-1, 38), spans, sent_scores, q_total


def null_row(pk, pa, qi, sent_scores, q_total):
    stems = qi["stems"]
    found = sum(1 for s in stems if s in pa["stem_set"])
    long_words = [w for w in qi["words"] if len(w) > 3]
    miss4 = sum(1 for w in long_words if w.casefold()[:4] not in pa["stem4"])
    caps_miss = sum(1 for w in qi["caps"] if w.casefold()[:4] not in pa["stem4"])
    nums_miss = sum(1 for w in qi["nums"] if w not in pa["digits"])
    head = {stem(w) for w in WORD_RE.findall(pk["heading"])}
    best = sent_scores.max() if len(sent_scores) else 0.0
    second = np.sort(sent_scores)[-2] if len(sent_scores) > 1 else 0.0
    return [found / max(1, len(stems)), len(stems) - found, miss4, caps_miss, len(qi["caps"]), nums_miss,
            len(qi["nums"]), len(head & stems), best / q_total, best, best - second, qi["q1"], qi["q2"], qi["len"]]


def featurize(packets, idf):
    span_x, span_meta, null_x = [], [], []
    for pi, pk in enumerate(packets):
        pa = analyse_passage(pk)
        for slot, q in enumerate(pk["questions"]):
            qi = question_info(q, idf)
            x, spans, sent_scores, q_total = span_rows(pk, pa, qi)
            span_x.append(x)
            span_meta.append(spans)
            null_x.append(null_row(pk, pa, qi, sent_scores, q_total))
    return span_x, span_meta, np.array(null_x, dtype=np.float64)


def span_targets(packets, span_meta):
    ys = []
    for qi, spans in enumerate(span_meta):
        pk = packets[qi // 4]
        gs, ge = pk["spans"][qi % 4]
        if gs < 0:
            ys.append(None)
            continue
        gold = norm_tokens(pk["passage"][gs:ge])
        ys.append(np.array([token_f1(norm_tokens(pk["passage"][a:b]), gold) for a, b in spans], dtype=np.float32))
    return ys


SPAN_PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
               "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
               "seed": SEED, "deterministic": True, "force_row_wise": True, "num_threads": THREADS,
               "verbose": -1}
NULL_PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 40,
               "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
               "seed": SEED, "deterministic": True, "force_row_wise": True, "num_threads": THREADS,
               "verbose": -1}
SPAN_ROUNDS = 400
NULL_ROUNDS = 300


def fit_span(qidx, span_x, span_y):
    rows = [i for i in qidx if span_y[i] is not None and len(span_x[i])]
    x = np.concatenate([span_x[i] for i in rows])
    y = np.concatenate([span_y[i] for i in rows])
    return lgb.train(SPAN_PARAMS, lgb.Dataset(x, y, categorical_feature=[32, 33, 34, 35]), SPAN_ROUNDS)


def best_spans(model, qidx, span_x, span_meta):
    sizes = [len(span_x[i]) for i in qidx]
    stacked = [span_x[i] for i in qidx if len(span_x[i])]
    preds = model.predict(np.concatenate(stacked), num_threads=THREADS) if stacked else np.zeros(0)
    out, pos = {}, 0
    for i, n in zip(qidx, sizes):
        if n == 0:
            out[i] = (None, 0.0, 0.0)
            continue
        p = preds[pos:pos + n]
        pos += n
        k = int(np.argmax(p))
        second = float(np.sort(p)[-2]) if n > 1 else 0.0
        out[i] = (span_meta[i][k], float(np.clip(p[k], 0.0, 1.0)), float(p[k] - second))
    return out


def null_matrix(null_x, chosen, qidx):
    base = np.array([list(null_x[i]) + [chosen[i][1], chosen[i][2]] for i in qidx])
    packet = base.reshape(-1, 4, base.shape[1])
    others = (packet.sum(1, keepdims=True) - packet) / 3.0
    return np.concatenate([packet, others[:, :, [0, 2, 3, 5, 14]]], axis=2).reshape(len(qidx), -1)


GRID = np.linspace(0.0, 1.0, 11)
SUBSETS = sorted([c for k in range(1, 5) for c in itertools.combinations(range(4), k)], key=len)
SUM_MIN = {}
for _k in range(1, 5):
    _vals = GRID[np.indices((11,) * _k).reshape(_k, -1)]
    SUM_MIN[_k] = (_vals.sum(0) * _vals.min(0), _vals.min(0))


def packet_decision(qdists, p_nulls):
    best_val, best_set = float(np.sum(p_nulls)) / 4.0, ()
    for subset in SUBSETS:
        if any(qdists[i] is None for i in subset):
            continue
        joint = qdists[subset[0]]
        for i in subset[1:]:
            joint = np.multiply.outer(joint, qdists[i]).ravel()
        sm, mn = SUM_MIN[len(subset)]
        rest = sum(p_nulls[j] for j in range(4) if j not in subset)
        val = (joint @ sm + rest * (joint @ mn)) / 4.0
        if val > best_val + 1e-12:
            best_val, best_set = val, subset
    return best_set


def decide(n_packets, chosen, p_null, spread):
    preds = []
    for pi in range(n_packets):
        dists, pns = [], []
        for s in range(4):
            i = 4 * pi + s
            span, g, _ = chosen[i]
            pn = float(p_null[i])
            pns.append(pn)
            if span is None:
                dists.append(None)
                continue
            dist = np.zeros(11)
            lo = max(0.0, g - spread)
            hi = min(1.0, g + spread)
            w = 0.5 if hi > lo else 1.0
            dist[int(round(lo * 10))] += (1 - pn) * w
            dist[int(round(hi * 10))] += (1 - pn) * (1 - w)
            dist[0] += pn
            dists.append(dist)
        subset = packet_decision(dists, pns)
        preds.append([list(chosen[4 * pi + s][0]) if s in subset else [-1, -1] for s in range(4)])
    return preds


def main():
    public_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    np.random.seed(SEED)

    # Inputs are limited to the released public train and test files.
    train = load_packets(public_dir / "train.csv", True)
    test = load_packets(public_dir / "test.csv", False)
    log(f"train packets {len(train)}, test packets {len(test)}")

    df = Counter()
    for pk in train:
        df.update({stem(w) for w in WORD_RE.findall(pk["passage"])})
    n_docs = len(train)
    idf = lambda s: math.log((n_docs + 1) / (df.get(s, 0) + 1))

    tr_x, tr_meta, tr_null = featurize(train, idf)
    te_x, te_meta, te_null = featurize(test, idf)
    tr_y = span_targets(train, tr_meta)
    log(f"span candidates: train {sum(len(x) for x in tr_x)}, test {sum(len(x) for x in te_x)}")

    y_null = np.array([int(pk["spans"][s][0] < 0) for pk in train for s in range(4)])
    q_groups = np.array([pk["group"] for pk in train for _ in range(4)])
    p_groups = np.array([pk["group"] for pk in train])
    folds = list(GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(train)), groups=p_groups))

    oof_chosen = {}
    for k, (tr_p, va_p) in enumerate(folds):
        tr_q = [4 * p + s for p in tr_p for s in range(4)]
        va_q = [4 * p + s for p in va_p for s in range(4)]
        oof_chosen.update(best_spans(fit_span(tr_q, tr_x, tr_y), va_q, tr_x, tr_meta))
        log(f"span fold {k} done")
    all_q = list(range(4 * len(train)))
    oof_null_x = null_matrix(tr_null, oof_chosen, all_q)
    oof_p = np.zeros(len(all_q))
    for tr_p, va_p in folds:
        tr_q = [4 * p + s for p in tr_p for s in range(4)]
        va_q = [4 * p + s for p in va_p for s in range(4)]
        model = lgb.train(NULL_PARAMS, lgb.Dataset(oof_null_x[tr_q], y_null[tr_q]), NULL_ROUNDS)
        oof_p[va_q] = model.predict(oof_null_x[va_q], num_threads=THREADS)

    gold = [pk["spans"] for pk in train]
    passages = [pk["passage"] for pk in train]
    answerable = [i for i in all_q if y_null[i] == 0 and oof_chosen[i][0] is not None]
    f1s = [token_f1(norm_tokens(passages[i // 4][slice(*oof_chosen[i][0])]),
                    norm_tokens(passages[i // 4][slice(*gold[i // 4][i % 4])])) for i in answerable]
    log(f"OOF top-1 token F1 on answerable questions {np.mean(f1s):.4f}")
    abstain = official_score([[[-1, -1]] * 4] * len(train), gold, passages, p_groups)
    best = (abstain, None, None)
    for spread in (0.0, 0.2, 0.4):
        for shift in (0.0, -0.5, 0.5, 1.0, 1.5):
            logit = np.log(np.clip(oof_p, 1e-6, 1 - 1e-6) / np.clip(1 - oof_p, 1e-6, 1)) + shift
            preds = decide(len(train), oof_chosen, 1 / (1 + np.exp(-logit)), spread)
            score = official_score(preds, gold, passages, p_groups)
            if score > best[0] + 1e-9:
                best = (score, spread, shift)
    log(f"OOF all-abstain {abstain:.4f}; selected decision {best[0]:.4f} (spread {best[1]}, shift {best[2]})")

    te_q = list(range(4 * len(test)))
    if best[1] is None:
        preds = [[[-1, -1]] * 4 for _ in test]
    else:
        span_model = fit_span(all_q, tr_x, tr_y)
        te_chosen = best_spans(span_model, te_q, te_x, te_meta)
        full_null_x = null_matrix(tr_null, oof_chosen, all_q)
        null_model = lgb.train(NULL_PARAMS, lgb.Dataset(full_null_x, y_null), NULL_ROUNDS)
        p_te = null_model.predict(null_matrix(te_null, te_chosen, te_q), num_threads=THREADS)
        logit = np.log(np.clip(p_te, 1e-6, 1 - 1e-6) / np.clip(1 - p_te, 1e-6, 1)) + best[2]
        preds = decide(len(test), te_chosen, 1 / (1 + np.exp(-logit)), best[1])

    sub = pd.DataFrame({"id": [pk["id"] for pk in test],
                        "spans": [json.dumps(p, separators=(",", ":")) for p in preds]})
    assert len(sub) == len(test) and sub["id"].is_unique
    for pk, pr in zip(test, preds):
        assert len(pr) == 4 and all((s, e) == (-1, -1) or 0 <= s < e <= len(pk["passage"]) for s, e in pr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    answered = sum(s >= 0 for pr in preds for s, _ in pr) / (4 * len(preds))
    log(f"wrote {out_path} ({len(sub)} rows, answered fraction {answered:.3f})")


if __name__ == "__main__":
    main()
