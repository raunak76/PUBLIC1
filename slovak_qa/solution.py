import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import math
import random
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, get_linear_schedule_with_warmup

SEED = 42
MODEL_NAME = "xlm-roberta-large"
MAX_LEN = 384
STRIDE = 128
MAX_ANS_TOK = 120
EPOCHS = 3
LR = 1.5e-5
BS = 8
ACCUM = 2
HOLDOUT_FRAC = 0.15


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def toks(s):
    return TOK_RE.findall(unicodedata.normalize("NFKC", s).casefold())


def f1(pred, gold):
    p, g = toks(pred), toks(gold)
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


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    return tok, mdl


def explode(df, with_labels):
    rows = []
    for r in df.itertuples():
        qs = json.loads(r.questions)
        sp = json.loads(r.spans) if with_labels else [[-1, -1]] * 4
        for k in range(4):
            q = f"{qs[k].strip()} ({str(r.heading).strip()})"
            rows.append((r.id, k, q, r.passage, sp[k][0], sp[k][1]))
    return rows


def featurize(tok, rows, train):
    enc = tok([r[2] for r in rows], [r[3] for r in rows], truncation="only_second", max_length=MAX_LEN,
              stride=STRIDE, return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
    smap = enc["overflow_to_sample_mapping"]
    feats = {"input_ids": [], "attention_mask": [], "start": [], "end": [], "sample": [], "offsets": []}
    for i in range(len(enc["input_ids"])):
        seq = enc.sequence_ids(i)
        offs = enc["offset_mapping"][i]
        si = smap[i]
        cs, ce = rows[si][4], rows[si][5]
        sp, ep = 0, 0
        if cs >= 0:
            c0 = seq.index(1)
            c1 = len(seq) - 1 - seq[::-1].index(1)
            if offs[c0][0] <= cs and offs[c1][1] >= ce:
                a = c0
                while a <= c1 and offs[a][1] <= cs:
                    a += 1
                b = c1
                while b >= c0 and offs[b][0] >= ce:
                    b -= 1
                if a <= b:
                    sp, ep = a, b
        feats["input_ids"].append(enc["input_ids"][i])
        feats["attention_mask"].append(enc["attention_mask"][i])
        feats["start"].append(sp)
        feats["end"].append(ep)
        feats["sample"].append(si)
        feats["offsets"].append([o if s == 1 else None for o, s in zip(offs, seq)])
    return feats


def train(model, feats, dev):
    n = len(feats["input_ids"])
    ids = torch.tensor(feats["input_ids"])
    am = torch.tensor(feats["attention_mask"])
    st = torch.tensor(feats["start"])
    en = torch.tensor(feats["end"])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * math.ceil(n / (BS * ACCUM))
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    model.train()
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        opt.zero_grad()
        for bi, i in enumerate(range(0, n, BS)):
            b = perm[i:i + BS]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids[b].to(dev), attention_mask=am[b].to(dev),
                            start_positions=st[b].to(dev), end_positions=en[b].to(dev))
            (out.loss / ACCUM).backward()
            if (bi + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
            if bi % 200 == 0:
                print(f"ep{ep} b{bi} loss {out.loss.item():.4f}", flush=True)
    model.eval()


@torch.no_grad()
def predict(model, feats, rows, dev):
    n = len(feats["input_ids"])
    ids = torch.tensor(feats["input_ids"])
    am = torch.tensor(feats["attention_mask"])
    best = {}
    for i in range(0, n, 64):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids[i:i + 64].to(dev), attention_mask=am[i:i + 64].to(dev))
        sl = out.start_logits.float().cpu().numpy()
        el = out.end_logits.float().cpu().numpy()
        for j in range(sl.shape[0]):
            f = i + j
            si = feats["sample"][f]
            offs = feats["offsets"][f]
            s, e = sl[j], el[j]
            null = s[0] + e[0]
            valid = np.array([o is not None for o in offs])
            s2 = np.where(valid, s, -1e9)
            e2 = np.where(valid, e, -1e9)
            top_s = np.argsort(-s2)[:20]
            top_e = np.argsort(-e2)[:20]
            bs, bspan = -1e9, None
            for a in top_s:
                for b in top_e:
                    if b < a or b - a + 1 > MAX_ANS_TOK or not valid[a] or not valid[b]:
                        continue
                    sc = s2[a] + e2[b]
                    if sc > bs:
                        bs, bspan = sc, (offs[a][0], offs[b][1])
            cur = best.get(si)
            if cur is None:
                best[si] = [bs, bspan, null]
            else:
                if bs > cur[0]:
                    cur[0], cur[1] = bs, bspan
                cur[2] = min(cur[2], null)
    res = []
    for si in range(len(rows)):
        bs, span, null = best[si]
        p = rows[si][3]
        if span is not None:
            a, b = span
            while a < b and p[a].isspace():
                a += 1
            while b > a and p[b - 1].isspace():
                b -= 1
            span = (a, b) if a < b else None
        res.append((bs - null if span is not None else -1e9, span))
    return res


def decide(res, thr):
    out = []
    for k in range(0, len(res), 4):
        packet = []
        for d, sp in res[k:k + 4]:
            packet.append([int(sp[0]), int(sp[1])] if (sp is not None and d > thr) else [-1, -1])
        out.append(packet)
    return out


def main():
    data_dir, out_path = sys.argv[1], sys.argv[2]
    seed_all(SEED)
    dev = torch.device("cuda")
    tr = pd.read_csv(os.path.join(data_dir, "train.csv"))
    te = pd.read_csv(os.path.join(data_dir, "test.csv"))
    groups = np.array(sorted(tr.group_id.unique()))
    rng = np.random.RandomState(SEED)
    ho_groups = set(rng.choice(groups, int(len(groups) * HOLDOUT_FRAC), replace=False))
    ho_mask = tr.group_id.isin(ho_groups).values
    fit_df, ho_df = tr[~ho_mask].reset_index(drop=True), tr[ho_mask].reset_index(drop=True)
    print("fit", len(fit_df), "holdout", len(ho_df), flush=True)

    tok, model = load_model()
    model.to(dev)
    fit_rows = explode(fit_df, True)
    fit_feats = featurize(tok, fit_rows, True)
    print("train feats", len(fit_feats["input_ids"]), flush=True)
    train(model, fit_feats, dev)

    ho_rows = explode(ho_df, True)
    ho_res = predict(model, featurize(tok, ho_rows, False), ho_rows, dev)
    best_thr, best_sc = 0.0, -1.0
    for thr in np.arange(-8, 10.01, 0.25):
        sc = grouped_score(ho_df, decide(ho_res, thr))
        if sc > best_sc:
            best_thr, best_sc = float(thr), sc
    print(f"holdout grouped packet score {best_sc:.4f} at thr {best_thr}", flush=True)
    print(f"holdout score at thr 0: {grouped_score(ho_df, decide(ho_res, 0.0)):.4f}", flush=True)

    te_rows = explode(te, False)
    te_res = predict(model, featurize(tok, te_rows, False), te_rows, dev)
    preds = decide(te_res, best_thr)
    for r, p in zip(te.itertuples(), preds):
        for sp in p:
            if sp[0] != -1:
                assert 0 <= sp[0] < sp[1] <= min(len(r.passage), 4096)
    sub = pd.DataFrame({"id": te.id.values, "spans": [json.dumps(p, separators=(",", ":")) for p in preds]})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sub.to_csv(out_path, index=False)
    print("wrote", out_path, len(sub), flush=True)


if __name__ == "__main__":
    main()
