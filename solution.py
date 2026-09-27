import os
import re
import sys
import math
import time
import random
import difflib
import unicodedata

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

SEED = 1234
DEV = os.environ.get("DEV", "0") == "1"
EPOCHS = int(os.environ.get("EPOCHS", "36"))
N_MODELS = int(os.environ.get("N_MODELS", "1"))
BEAM = 5
T0 = time.time()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(10, os.cpu_count() or 1)))
DEVICE = torch.device("cpu")


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def canonical(value):
    s = unicodedata.normalize("NFC", "" if value is None else str(value))
    s = "".join(c for c in s if unicodedata.category(c) != "Cf")
    return re.sub(r"[\s।॥.,;+-]+", "", s)


def similarity(a, b):
    a, b = canonical(a), canonical(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def row_credit(po, pe, to, te):
    if po != to:
        return 0.0
    if to == "desi":
        return 1.0
    t = canonical(te)
    ex = 1.0 if t and canonical(pe) == t else 0.0
    return 0.7 * ex + 0.3 * similarity(pe, te)


def score(po, pe, to, te):
    cs = [row_credit(a, b, c, d) for a, b, c, d in zip(po, pe, to, te)]
    cs = np.array(cs)
    to = np.array(to)
    s = cs[to == "sanskrit"].mean() if (to == "sanskrit").any() else 0.0
    d = cs[to == "desi"].mean() if (to == "desi").any() else 0.0
    return 0.85 * s + 0.15 * d


DEV_RE = re.compile(r"[ऀ-ॿ]+")
CIT_RE = re.compile(r"\(दे\s*[०-९]|;\s*दे\s*[०-९]|\(दे\)|दे\s*[०-९]+\s*,")


def gloss_head(g):
    g = re.split(r"[(।;‘]", g)[0]
    g = re.sub(r"[०-९]+\.", " ", g)
    toks = DEV_RE.findall(g)
    return " ".join(toks)[:20]


# Inputs are limited to the released public train and test files.
pub = sys.argv[1]
out_path = sys.argv[2]
train = pd.read_csv(os.path.join(pub, "train.csv"), dtype=str, keep_default_na=False)
test = pd.read_csv(os.path.join(pub, "test.csv"), dtype=str, keep_default_na=False)
for df in (train, test):
    df["hw"] = df["headword_dev"].map(lambda x: unicodedata.normalize("NFC", x))
    df["gh"] = df["gloss"].map(gloss_head)
    df["grp"] = df["hw"].str[:3]

if DEV:
    grp_key = train["grp"].map(lambda k: sum(ord(c) * (i + 7) for i, c in enumerate(k)) % 5)
    tr = train[grp_key != 0].reset_index(drop=True)
    te = train[grp_key == 0].reset_index(drop=True)
else:
    tr = train
    te = test
log("train", len(tr), "eval", len(te))

# ---------------- origin classifier ----------------


def origin_feats(fit_df, dfs):
    v1 = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), sublinear_tf=True, min_df=2)
    v2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3, max_features=300000)
    v3 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, token_pattern=r"\S+")
    hw = lambda d: ("^" + d["hw"] + "$ " + d["pos"]).tolist()
    v1.fit(hw(fit_df))
    v2.fit(fit_df["gloss"].tolist())
    v3.fit((fit_df["gloss"] + " POS_" + fit_df["pos"]).tolist())
    res = []
    for d in dfs:
        cit = d["gloss"].map(lambda g: len(CIT_RE.findall(g))).values.astype(float)
        extra = np.stack([cit > 0, np.minimum(cit, 3) / 3, d["gloss"].str.len().values / 1000.0], 1)
        res.append(hstack([v1.transform(hw(d)), v2.transform(d["gloss"].tolist()),
                           v3.transform((d["gloss"] + " POS_" + d["pos"]).tolist()),
                           csr_matrix(extra * 3.0)]).tocsr())
    return res


def fit_origin(fit_df, pred_df):
    Xa, Xb = origin_feats(fit_df, [fit_df, pred_df])
    y = (fit_df["origin"] == "desi").astype(int).values
    clf = LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced")
    clf.fit(Xa, y)
    return clf.predict_proba(Xb)[:, 1]


p_desi = fit_origin(tr, te)
oof = np.zeros(len(tr))
for a, b in GroupKFold(n_splits=4).split(tr, groups=tr["grp"]):
    oof[b] = fit_origin(tr.iloc[a].reset_index(drop=True), tr.iloc[b].reset_index(drop=True))
log("origin done")

# ---------------- seq2seq ----------------
POS_LIST = sorted(set(train["pos"]) | set(test["pos"]))
PAD, BOS, EOS, SEP, UNK = 0, 1, 2, 3, 4
src_chars = set()
for d in (train, test):
    for s in d["hw"].tolist() + d["gh"].tolist():
        src_chars.update(s)
tgt_train = train.loc[train.origin == "sanskrit", "etymon"].map(canonical)
tgt_chars = set("".join(tgt_train.tolist()))
src_vocab = {c: i + 5 + len(POS_LIST) for i, c in enumerate(sorted(src_chars))}
for i, p in enumerate(POS_LIST):
    src_vocab["<pos:" + p + ">"] = 5 + i
tgt_itos = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"] + sorted(tgt_chars)
tgt_vocab = {c: i for i, c in enumerate(tgt_itos)}
MAXS, MAXT = 48, 24


def enc_src(hw, pos, gh):
    ids = [src_vocab["<pos:" + pos + ">"]] + [src_vocab.get(c, UNK) for c in hw] + [SEP] + [src_vocab.get(c, UNK) for c in gh]
    return ids[:MAXS]


def enc_tgt(t):
    return [BOS] + [tgt_vocab.get(c, UNK) for c in t][: MAXT - 2] + [EOS]


class PosEnc(nn.Module):
    def __init__(self, d, n=128):
        super().__init__()
        pe = torch.zeros(n, d)
        pos = torch.arange(n).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class Seq2Seq(nn.Module):
    def __init__(self, ns, nt, d=192, h=4, L=3, ff=512, drop=0.15):
        super().__init__()
        self.d = d
        self.se = nn.Embedding(ns, d, padding_idx=PAD)
        self.seg = nn.Embedding(2, d)
        self.te = nn.Embedding(nt, d, padding_idx=PAD)
        self.pe = PosEnc(d)
        self.tf = nn.Transformer(d, h, L, L, ff, drop, batch_first=True, norm_first=True)
        self.out = nn.Linear(d, nt)
        self.drop = nn.Dropout(drop)

    def encode(self, src, seg):
        x = self.drop(self.pe(self.se(src) * math.sqrt(self.d) + self.seg(seg)))
        m = src == PAD
        return self.tf.encoder(x, src_key_padding_mask=m), m

    def decode(self, mem, mm, tgt):
        y = self.drop(self.pe(self.te(tgt) * math.sqrt(self.d)))
        T = tgt.size(1)
        cm = torch.triu(torch.full((T, T), float("-inf")), 1)
        o = self.tf.decoder(y, mem, tgt_mask=cm, tgt_key_padding_mask=tgt == PAD, memory_key_padding_mask=mm)
        return self.out(o)


def make_seg(src):
    seg = torch.zeros_like(src)
    for i in range(src.size(0)):
        w = (src[i] == SEP).nonzero()
        if len(w):
            seg[i, w[0, 0]:] = 1
    return seg


def pad(seqs, L=None):
    L = L or max(len(s) for s in seqs)
    a = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        a[i, : len(s)] = torch.tensor(s[:L])
    return a


def train_model(src, tgt, seed, epochs):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    m = Seq2Seq(len(src_vocab) + 5 + len(POS_LIST), len(tgt_itos))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0.01)
    bs = 128
    n = len(src)
    steps = epochs * ((n + bs - 1) // bs)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.02, 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    for ep in range(epochs):
        m.train()
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i: i + bs]
            s = pad([src[j] for j in idx])
            t = pad([tgt[j] for j in idx])
            mem, mm = m.encode(s, make_seg(s))
            lo = m.decode(mem, mm, t[:, :-1])
            loss = F.cross_entropy(lo.reshape(-1, lo.size(-1)), t[:, 1:].reshape(-1), ignore_index=PAD, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"seed {seed} ep {ep} loss {tot / n:.4f}")
    m.eval()
    return m


@torch.no_grad()
def beam_search(models, src, beam=BEAM, bs=128):
    results = []
    for i in range(0, len(src), bs):
        s = pad(src[i: i + bs])
        B = s.size(0)
        seg = make_seg(s)
        mems = [mdl.encode(s, seg) for mdl in models]
        mems = [(mem.repeat_interleave(beam, 0), mm.repeat_interleave(beam, 0)) for mem, mm in mems]
        seqs = torch.full((B * beam, 1), BOS, dtype=torch.long)
        scores = torch.full((B, beam), float("-inf"))
        scores[:, 0] = 0.0
        scores = scores.view(-1)
        done = torch.zeros(B * beam, dtype=torch.bool)
        for _ in range(MAXT - 1):
            lp = 0
            for mdl, (mem, mm) in zip(models, mems):
                lp = lp + F.log_softmax(mdl.decode(mem, mm, seqs)[:, -1], -1)
            lp = lp / len(models)
            lp[:, PAD] = float("-inf")
            lp[:, BOS] = float("-inf")
            lp[done] = float("-inf")
            lp[done, PAD] = 0.0
            cand = (scores.unsqueeze(1) + lp).view(B, -1)
            top, ti = cand.topk(beam, -1)
            V = lp.size(1)
            bi = ti // V
            tok = ti % V
            base = (torch.arange(B) * beam).unsqueeze(1)
            sel = (base + bi).view(-1)
            seqs = torch.cat([seqs[sel], tok.view(-1, 1)], 1)
            done = done[sel] | (tok.view(-1) == EOS) | (tok.view(-1) == PAD)
            scores = top.view(-1)
            if done.all():
                break
        seqs = seqs.view(B, beam, -1)
        sc = scores.view(B, beam)
        for b in range(B):
            hyps = []
            for k in range(beam):
                toks = []
                for t in seqs[b, k, 1:].tolist():
                    if t in (EOS, PAD):
                        break
                    toks.append(tgt_itos[t] if t > 4 else "")
                L = max(1, len(toks) + 1)
                hyps.append(("".join(toks), sc[b, k].item() / (L ** 0.6)))
            results.append(hyps)
    return results


sk = tr[tr.origin == "sanskrit"].reset_index(drop=True)
S = [enc_src(h, p, g) for h, p, g in zip(sk.hw, sk.pos, sk.gh)]
Tg = [enc_tgt(canonical(e)) for e in sk.etymon]
models = []
for k in range(N_MODELS):
    models.append(train_model(S, Tg, SEED + k, EPOCHS))
log("models trained")

te_src = [enc_src(h, p, g) for h, p, g in zip(te.hw, te.pos, te.gh)]
hyps = beam_search(models, te_src)


def pick(hs):
    # choose hypothesis maximizing expected credit under beam posterior (MBR)
    sc = np.array([h[1] for h in hs])
    w = np.exp(sc - sc.max())
    w /= w.sum()
    best, bv = hs[0][0], -1
    for i, (a, _) in enumerate(hs):
        v = sum(w[j] * (0.7 * (a == b) + 0.3 * difflib.SequenceMatcher(None, a, b, autojunk=False).ratio())
                for j, (b, _) in enumerate(hs))
        if v > bv:
            bv, best = v, a
    return best, bv


picked = [pick(h) for h in hyps]
pred_ety = [p[0] if p[0] else h for p, h in zip(picked, te.hw)]
exp_credit = np.array([p[1] for p in picked])

# decision threshold: choose desi when expected desi gain beats expected sanskrit gain
n_s = (tr.origin == "sanskrit").sum()
n_d = (tr.origin == "desi").sum()
alpha = 0.15 / n_d
beta = 0.85 / n_s
tau_grid = np.linspace(0.05, 0.95, 91)
yo = (tr.origin == "desi").values
# choose a threshold multiplier on OOF using an assumed average sanskrit credit
avg_credit = float(os.environ.get("AVG_CREDIT", "0.5"))
best_t, best_v = 0.5, -1
for t in tau_grid:
    pd_ = oof > t
    v = 0.85 * avg_credit * ((~pd_) & (~yo)).sum() / (~yo).sum() + 0.15 * (pd_ & yo).sum() / yo.sum()
    if v > best_v:
        best_v, best_t = v, t
log("threshold", best_t)
pred_org = np.where(p_desi > best_t, "desi", "sanskrit")
pred_ety = [e if o == "sanskrit" else "-" for e, o in zip(pred_ety, pred_org)]

if DEV:
    to = te.origin.tolist()
    tet = te.etymon.tolist()
    log("score", round(score(list(pred_org), pred_ety, to, tet), 4))
    log("score true origin", round(score(to, [p[0] for p in picked], to, tet), 4))
    top1 = [h[0][0] for h in hyps]
    log("score true origin top1", round(score(to, top1, to, tet), 4))
    log("origin acc", (pred_org == np.array(to)).mean())
    sm = te.origin.values == "sanskrit"
    log("exact", np.mean([canonical(a) == canonical(b) for a, b, s in zip([p[0] for p in picked], tet, sm) if s]))

sub = pd.DataFrame({"entry_id": te["entry_id"], "origin": pred_org, "etymon": pred_ety})
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
sub.to_csv(out_path, index=False, encoding="utf-8")
log("wrote", out_path, len(sub))
