#!/usr/bin/env python
"""SubjectWeave: candidate-conditioned pair scorer for unseen-taxonomy subject selection.

Usage: python solution.py <data_dir> <submission_csv>

Fixed deterministic plan, one NVIDIA A10G (CUDA required, no fallback path):
 1. Encode every bill with the pretrained sentence encoder sentence-transformers/all-MiniLM-L6-v2
    (pinned revision, frozen, fp32). The beginning/middle/end excerpts are kept apart and cut into
    128-token windows; each window and each candidate subject is mean-pooled and L2-normalised.
    [TOPIC] masks and excerpt markers are removed from the text before encoding.
 2. Train a small listwise pair scorer on train.csv only: identity-initialised text/candidate
    projections (regularised toward identity), excerpt-position embeddings, soft-max pooling over
    windows and a tiny MLP on similarity statistics; softmax cross-entropy over the 8 candidates.
    No parameter is tied to a subject name or a slot, so it transfers to the unseen test vocabulary.
 3. Average log-probabilities of 3 fixed seeds and write the argmax slot for every test row.

Only train.csv / test.csv and the public pretrained encoder are used. No IDs, row order, slot
position, label frequencies or test answers are used.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys, random
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '7dbbc90392e2f80f3d3c277d6e90027e55de9125'
DEVICE = torch.device('cuda')
OC = [f'option_{k}' for k in range(8)]
WIN, MAXW, DIM = 128, 12, 384
SEEDS, EPOCHS, BATCH, LR = [0, 1, 2], 8, 256, 2e-3


def set_determinism(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class Encoder:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.model = AutoModel.from_pretrained(MODEL, revision=REVISION).to(DEVICE).eval()
        self.cls, self.sep = self.tok.cls_token_id, self.tok.sep_token_id

    def ids(self, texts):
        return self.tok(texts, add_special_tokens=False)['input_ids']

    @torch.no_grad()
    def run(self, seqs, bs=128):
        out = np.zeros((len(seqs), DIM), np.float32)
        order = np.argsort([len(x) for x in seqs], kind='stable')
        for i in range(0, len(seqs), bs):
            idx = order[i:i + bs]; L = max(len(seqs[j]) for j in idx) + 2
            ii = np.zeros((len(idx), L), np.int64); am = np.zeros_like(ii)
            for r, j in enumerate(idx):
                t = [self.cls] + list(seqs[j]) + [self.sep]; ii[r, :len(t)] = t; am[r, :len(t)] = 1
            ii, am = torch.from_numpy(ii).to(DEVICE), torch.from_numpy(am).to(DEVICE)
            h = self.model(input_ids=ii, attention_mask=am, token_type_ids=torch.zeros_like(ii)).last_hidden_state
            m = (h * am[..., None]).sum(1) / am.sum(1, keepdim=True)
            out[idx] = F.normalize(m, dim=-1).cpu().numpy()
        return out


def parts(t):
    t = t.replace('[TOPIC]', '').replace('[END]', '[MIDDLE]')
    return [p.strip() for p in t.split('[MIDDLE]') if p.strip()]


def encode_bills(E, texts):
    segs, rid, pid = [], [], []
    for i, t in enumerate(texts):
        for pi, ids in enumerate(E.ids(parts(t))):
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
        t = F.normalize(s.A(s.drop(T + F.embedding(P, s.pe))), dim=-1); o = F.normalize(s.B(s.drop(O)), dim=-1)
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


def train(T, M, P, O, y, seed, wd_eye=1e-2):
    set_determinism(seed)
    m = Scorer().to(DEVICE); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0)
    n = len(y); steps = EPOCHS * ((n + BATCH - 1) // BATCH)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
    I = torch.eye(DIM, device=DEVICE); g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH].to(DEVICE)
            loss = F.cross_entropy(m(T[b], M[b], P[b], O[b]), y[b]) + \
                wd_eye * (((m.A.weight - I) ** 2).sum() + ((m.B.weight - I) ** 2).sum())
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m


@torch.no_grad()
def predict(m, T, M, P, O):
    m.eval()
    return torch.cat([torch.log_softmax(m(T[i:i + 1024], M[i:i + 1024], P[i:i + 1024], O[i:i + 1024]), 1)
                      for i in range(0, len(T), 1024)]).cpu().numpy().astype(np.float64)


def main(data_dir, out_path):
    if DEVICE.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('This solution is specified for one NVIDIA A10G GPU; CUDA is not available.')
    set_determinism(0)
    tr = pd.read_csv(os.path.join(data_dir, 'train.csv')); te = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    E = Encoder()
    names = sorted(set(tr[OC].values.ravel()) | set(te[OC].values.ravel()))
    od = dict(zip(names, E.run(E.ids(names))))
    optemb = lambda df: np.stack([[od[x] for x in r] for r in df[OC].values]).astype(np.float32)
    tr_arr = encode_bills(E, tr.bill_text.tolist()) + (optemb(tr),); print('train encoded', flush=True)
    te_arr = encode_bills(E, te.bill_text.tolist()) + (optemb(te),); print('test encoded', flush=True)
    del E; torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None
    Ttr, Mtr, Ptr, Otr = [torch.from_numpy(a).to(DEVICE) for a in tr_arr]
    Tte, Mte, Pte, Ote = [torch.from_numpy(a).to(DEVICE) for a in te_arr]
    y = torch.from_numpy(tr.target.values.astype(np.int64)).to(DEVICE)
    Z = np.zeros((len(te), 8))
    for s in SEEDS:
        m = train(Ttr, Mtr, Ptr, Otr, y, seed=s)
        Z += predict(m, Tte, Mte, Pte, Ote); print('seed', s, 'done', flush=True)
    sub = pd.DataFrame({'id': te.id, 'prediction': Z.argmax(1).astype(int)})
    sub.to_csv(out_path, index=False)
    print('wrote', out_path, sub.prediction.value_counts().sort_index().tolist())


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
