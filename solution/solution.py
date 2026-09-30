#!/usr/bin/env python
"""SubjectWeave: candidate-conditioned scorer for unseen-taxonomy subject selection.

Usage: python solution.py <data_dir> <submission_csv>

Pipeline (deterministic):
 1. Encode every bill in 128-token windows (beginning/middle/end parts kept apart) and every
    candidate subject with the pretrained all-MiniLM-L6-v2 sentence encoder (ONNX).
 2. Train a small listwise pair scorer on train.csv: learned text/candidate projections
    (identity-initialised, regularised toward identity), part embeddings, soft-max pooling
    over windows and a tiny MLP on similarity statistics. Softmax over the 8 candidates.
    No parameters are tied to a subject name, so it transfers to the unseen test vocabulary.
 3. Average several seeds and predict argmax slot for test.csv.

Model weights: set MINILM_ONNX_DIR to a folder with model.onnx + tokenizer.json, otherwise the
public all-MiniLM-L6-v2 ONNX export is downloaded once from Chroma's S3 bucket.
"""
import os, sys, tarfile, urllib.request, random
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import onnxruntime as ort
from tokenizers import Tokenizer

OC = [f'option_{k}' for k in range(8)]
WIN, MAXW, DIM = 128, 12, 384
SEEDS, EPOCHS = [0, 1, 2], 8
URL = 'https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz'


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def model_dir(cache):
    d = os.environ.get('MINILM_ONNX_DIR')
    if d:
        return d
    d = os.path.join(cache, 'onnx')
    if not os.path.exists(os.path.join(d, 'model.onnx')):
        os.makedirs(cache, exist_ok=True)
        tgz = os.path.join(cache, 'onnx.tar.gz')
        urllib.request.urlretrieve(URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(cache)
    return d


class Encoder:
    def __init__(self, d):
        o = ort.SessionOptions(); o.intra_op_num_threads = os.cpu_count() or 4
        prov = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(os.path.join(d, 'model.onnx'), o, providers=prov)
        self.tok = Tokenizer.from_file(os.path.join(d, 'tokenizer.json'))
        self.tok.no_padding(); self.tok.no_truncation()

    def ids(self, texts):
        return [e.ids[1:-1] for e in self.tok.encode_batch(texts)]

    def run(self, seqs, bs=128):
        out = np.zeros((len(seqs), DIM), np.float32)
        order = np.argsort([len(x) for x in seqs], kind='stable')
        for i in range(0, len(seqs), bs):
            idx = order[i:i + bs]; L = max(len(seqs[j]) for j in idx) + 2
            ii = np.zeros((len(idx), L), np.int64); am = np.zeros_like(ii)
            for r, j in enumerate(idx):
                t = [101] + seqs[j] + [102]; ii[r, :len(t)] = t; am[r, :len(t)] = 1
            h = self.sess.run(None, {'input_ids': ii, 'attention_mask': am, 'token_type_ids': np.zeros_like(ii)})[0]
            m = (h * am[:, :, None]).sum(1) / am.sum(1, keepdims=True)
            out[idx] = m / np.linalg.norm(m, axis=1, keepdims=True)
        return out


def parts(t):
    t = t.replace('[TOPIC]', '').replace('[END]', '[MIDDLE]')
    return [p.strip() for p in t.split('[MIDDLE]') if p.strip()]


def encode_bills(E, texts):
    segs, rid, pid = [], [], []
    for i, t in enumerate(texts):
        ps = parts(t)
        for pi, ids in enumerate(E.ids(ps)):
            for s in range(0, len(ids), WIN):
                segs.append(ids[s:s + WIN]); rid.append(i); pid.append(pi)
    X = E.run(segs)
    n = len(texts)
    T = np.zeros((n, MAXW, DIM), np.float32); M = np.zeros((n, MAXW), np.float32); P = np.zeros((n, MAXW), np.int64)
    cnt = np.zeros(n, int)
    for j, r in enumerate(rid):
        c = cnt[r]
        if c < MAXW:
            T[r, c] = X[j]; M[r, c] = 1; P[r, c] = min(pid[j], 2)
        cnt[r] += 1
    return T, M, P


class Scorer(nn.Module):
    def __init__(s, d=DIM, h=64):
        super().__init__()
        s.A = nn.Linear(d, d, bias=False); s.B = nn.Linear(d, d, bias=False)
        nn.init.eye_(s.A.weight); nn.init.eye_(s.B.weight)
        s.temp = nn.Parameter(torch.tensor(10.)); s.tau = nn.Parameter(torch.tensor(5.))
        s.pe = nn.Parameter(torch.zeros(3, d))
        s.mlp = nn.Sequential(nn.Linear(8, h), nn.GELU(), nn.Linear(h, 1))
        s.drop = nn.Dropout(0.1)

    def forward(s, T, M, P, O):
        t = F.normalize(s.A(s.drop(T + s.pe[P])), dim=-1); o = F.normalize(s.B(s.drop(O)), dim=-1)
        S = torch.einsum('bwd,bkd->bkw', t, o)
        neg = (1 - M[:, None, :]) * -1e4
        mx = (S + neg).max(-1).values
        lse = torch.logsumexp(s.tau * S + neg, -1) / s.tau
        mean = (S * M[:, None, :]).sum(-1) / M.sum(-1, keepdim=True)
        tm = F.normalize((T * M[..., None]).sum(1), dim=-1); om = F.normalize(O, dim=-1)
        raw = torch.einsum('bd,bkd->bk', tm, om)
        f = torch.stack([mx, lse, mean, raw], -1)
        f = torch.cat([f, f - f.mean(1, keepdim=True)], -1)
        return s.temp * lse + s.mlp(f).squeeze(-1)


def train(T, M, P, O, y, seed, epochs=EPOCHS, lr=2e-3, wd_eye=1e-2, bs=256):
    seed_all(seed)
    m = Scorer(); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0)
    n = len(y); steps = epochs * ((n + bs - 1) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    T, M, P, O = map(torch.from_numpy, (T, M, P, O)); y = torch.from_numpy(y)
    I = torch.eye(DIM)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = F.cross_entropy(m(T[b], M[b], P[b], O[b]), y[b]) + \
                wd_eye * (((m.A.weight - I) ** 2).sum() + ((m.B.weight - I) ** 2).sum())
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m


@torch.no_grad()
def predict(m, T, M, P, O):
    m.eval()
    return np.concatenate([m(*[torch.from_numpy(a[i:i + 1024]) for a in (T, M, P, O)]).numpy()
                           for i in range(0, len(T), 1024)])


def main(data_dir, out_path):
    torch.set_num_threads(os.cpu_count() or 4)
    cache = os.environ.get('SW_CACHE', os.path.join(os.path.dirname(os.path.abspath(out_path)), 'sw_cache'))
    tr = pd.read_csv(os.path.join(data_dir, 'train.csv')); te = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    E = Encoder(model_dir(cache))
    names = sorted(set(tr[OC].values.ravel()) | set(te[OC].values.ravel()))
    od = dict(zip(names, E.run(E.ids(names))))
    optemb = lambda df: np.stack([[od[x] for x in r] for r in df[OC].values]).astype(np.float32)
    Ttr, Mtr, Ptr = encode_bills(E, tr.bill_text.tolist()); print('train encoded', flush=True)
    Tte, Mte, Pte = encode_bills(E, te.bill_text.tolist()); print('test encoded', flush=True)
    Otr, Ote = optemb(tr), optemb(te); y = tr.target.values.astype(np.int64)
    Z = np.zeros((len(te), 8))
    for s in SEEDS:
        m = train(Ttr, Mtr, Ptr, Otr, y, seed=s)
        z = predict(m, Tte, Mte, Pte, Ote); Z += torch.log_softmax(torch.from_numpy(z), 1).numpy()
        print('seed', s, 'done', flush=True)
    sub = pd.DataFrame({'id': te.id, 'prediction': Z.argmax(1).astype(int)})
    sub.to_csv(out_path, index=False); print('wrote', out_path, sub.prediction.value_counts().sort_index().tolist())


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
