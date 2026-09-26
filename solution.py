import sys
import os
import csv
import json
import multiprocessing as mp

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import hilbert

FS = 160
LAMBDA = 1000.0
EPS = 0.0025
SA_ITERS = 500000
SA_RESTARTS = 10
SA_TEMP = 1.0
SEED = 20260926


def nrm(x):
    x = x - x.mean(-1, keepdims=True)
    return x / (np.sqrt((x ** 2).mean(-1, keepdims=True)) + 1e-9)


def band(x, lo, hi):
    f = np.fft.rfft(x, axis=-1)
    fr = np.fft.rfftfreq(x.shape[-1], 1.0 / FS)
    f[..., (fr < lo) | (fr > hi)] = 0
    return nrm(np.fft.irfft(f, n=x.shape[-1], axis=-1))


def car(x):
    return nrm(x - x.mean(-2, keepdims=True))


def remove_pcs(x, k):
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return nrm(x - (u[:, :k] * s[:k]) @ vt[:k])


def corr(x):
    return x @ x.T / x.shape[-1]


def fz(c):
    return np.arctanh(np.clip(c, -0.995, 0.995))


def lagc(x, k):
    a = nrm(x[:, :-k])
    b = nrm(x[:, k:])
    return a @ b.T / a.shape[-1]


def envelope(x, lo, hi):
    return nrm(np.abs(hilbert(band(x, lo, hi), axis=-1)))


def pair_features(x):
    d1 = nrm(np.diff(x, axis=-1))
    d2 = nrm(np.diff(x, 2, axis=-1))
    cx = car(x)
    l1 = lagc(x, 1)
    l2 = lagc(x, 2)
    feats = [
        fz(corr(x)),
        fz(corr(d1)),
        fz(corr(d2)),
        fz(corr(cx)),
        fz(corr(nrm(np.diff(cx, axis=-1)))),
        fz(corr(remove_pcs(x, 1))),
        fz(corr(band(x, 0, 4))),
        fz(corr(band(x, 4, 8))),
        fz(corr(band(x, 8, 13))),
        fz(corr(band(x, 13, 30))),
        fz(corr(band(x, 30, 80))),
        fz((l1 + l1.T) / 2),
        (l1 - l1.T) * 5,
        (l2 - l2.T) * 5,
        fz(corr(envelope(x, 8, 13))),
        fz(corr(envelope(x, 13, 30))),
        fz(corr(band(x, 30, 50))),
        fz(corr(band(x, 50, 80))),
        fz(corr(band(cx, 8, 13))),
        fz(corr(band(cx, 13, 30))),
        fz(corr(band(cx, 0, 4))),
        fz(corr(remove_pcs(x, 2))),
        fz(corr(remove_pcs(x, 3))),
    ]
    return np.stack(feats, -1)


def fit_pair_gaussians(zs, ys):
    n, _, _, k = zs.shape
    zc = np.zeros_like(zs)
    for m in range(n):
        zc[m][np.ix_(ys[m], ys[m])] = zs[m]
    mu = zc.mean(0)
    dev = zc - mu
    cov = np.einsum('nabk,nabl->abkl', dev, dev) / n
    off = ~np.eye(64, dtype=bool)
    pooled = cov[off].mean(0)
    cov = (n * cov + LAMBDA * pooled) / (n + LAMBDA) + EPS * np.eye(k)
    prec = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(cov)
    th2 = -0.5 * prec.reshape(64, 64, k * k)
    th1 = np.einsum('abkl,abl->abk', prec, mu)
    th0 = -0.5 * np.einsum('abk,abk->ab', th1, mu) - 0.5 * logdet
    theta = np.concatenate([th2, th1, th0[..., None]], -1)
    idx = np.arange(64)
    theta[idx, idx] = 0
    return theta


def pair_scores(z, theta):
    k = z.shape[-1]
    phi = np.concatenate([(z[..., :, None] * z[..., None, :]).reshape(64, 64, k * k), z, np.ones((64, 64, 1))], -1)
    d = phi.shape[-1]
    w = (phi.reshape(-1, d) @ theta.reshape(-1, d).T).reshape(64, 64, 64, 64).transpose(0, 2, 1, 3)
    w = np.ascontiguousarray(w, dtype=np.float32)
    idx = np.arange(64)
    w[idx, :, idx, :] = 0
    return w


def total_score(w, p):
    r = np.arange(64)
    return float(w[r[:, None], p[:, None], r[None, :], p[None, :]].sum())


def anneal(w, p, free, rng):
    nf = len(free)
    best_all, best_p = -np.inf, p.copy()
    for r in range(SA_RESTARTS):
        p = p.copy()
        if r > 0:
            p[free] = p[free][rng.permutation(nf)]
        e = w[np.arange(64), p].sum(0)
        ii = free[rng.integers(0, nf, 2000)]
        jj = free[rng.integers(0, nf, 2000)]
        a, b = p[ii], p[jj]
        dd = 2 * (e[ii, b] - e[ii, a] + e[jj, a] - e[jj, b] - w[ii, b, jj, b] - w[jj, a, ii, a] + w[ii, a, jj, b] + w[ii, b, jj, a])
        t0 = np.std(dd[ii != jj]) * SA_TEMP
        t1 = t0 * 1e-3
        cur = total_score(w, p)
        best, bp = cur, p.copy()
        ia = rng.integers(0, nf, SA_ITERS)
        ja = rng.integers(0, nf - 1, SA_ITERS)
        ja = np.where(ja >= ia, ja + 1, ja)
        thr = np.log(rng.random(SA_ITERS)) * (t0 * (t1 / t0) ** (np.arange(SA_ITERS) / SA_ITERS))
        fi, fj = free[ia], free[ja]
        for t in range(SA_ITERS):
            i, j = fi[t], fj[t]
            pi, pj = p[i], p[j]
            d = 2 * (e[i, pj] - e[i, pi] + e[j, pi] - e[j, pj] - w[i, pj, j, pj] - w[j, pi, i, pi] + w[i, pi, j, pj] + w[i, pj, j, pi])
            if d > thr[t]:
                e += w[i, pj] - w[i, pi] + w[j, pi] - w[j, pj]
                p[i], p[j] = pj, pi
                cur += d
                if cur > best:
                    best, bp = cur, p.copy()
        best = total_score(w, bp)
        if best > best_all:
            best_all, best_p = best, bp
    return best_p


def initial_assignment(anchors):
    m = np.zeros((64, 64))
    for i, a in anchors.items():
        m[i, :] = -1e6
        m[:, a] = -1e6
        m[i, a] = 1e6
    r, c = linear_sum_assignment(-m)
    p = np.empty(64, int)
    p[r] = c
    return p


STATE = {}


def solve_packet(task):
    idx, z, anchors = task
    w = pair_scores(z, STATE['theta'])
    free = np.array([i for i in range(64) if i not in anchors])
    rng = np.random.default_rng(SEED + idx)
    p = anneal(w, initial_assignment(anchors), free, rng)
    return idx, p


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    data_dir, out_path = sys.argv[1], sys.argv[2]
    # Inputs are limited to the released config, train/validation/test manifests and their referenced signal files.
    with open(os.path.join(data_dir, 'config.json')) as f:
        cfg = json.load(f)
    labels = cfg['labels']
    lab_idx = {l: i for i, l in enumerate(labels)}
    labeled = read_jsonl(os.path.join(data_dir, 'train.jsonl')) + read_jsonl(os.path.join(data_dir, 'validation.jsonl'))
    test = read_jsonl(os.path.join(data_dir, 'test.jsonl'))

    def load(rec):
        return np.load(os.path.join(data_dir, rec['path']), allow_pickle=False).astype(np.float64)

    zs = np.stack([pair_features(load(r)) for r in labeled])
    ys = np.array([[lab_idx[l] for l in r['labels']] for r in labeled])
    STATE['theta'] = fit_pair_gaussians(zs, ys)
    del zs

    tasks = []
    for n, r in enumerate(test):
        anchors = {int(k): lab_idx[v] for k, v in r['anchors'].items()}
        tasks.append((n, pair_features(load(r)), anchors))

    workers = max(1, min(10, os.cpu_count() or 1))
    ctx = mp.get_context('fork')
    with ctx.Pool(workers) as pool:
        results = dict(pool.imap_unordered(solve_packet, tasks, chunksize=1))

    rows = []
    for n, r in enumerate(test):
        p = results[n]
        pred = [labels[a] for a in p]
        for k, v in r['anchors'].items():
            assert pred[int(k)] == v
        assert sorted(pred) == sorted(labels)
        rows.append((r['id'], json.dumps(pred)))

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['id', 'labels_json'])
        wr.writerows(rows)


if __name__ == '__main__':
    main()
