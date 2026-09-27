import os
import re
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import lightgbm as lgb
from multiprocessing import Pool
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import normalize

SEED = 0
NF = 5
ROUNDS = 2
NT = os.cpu_count() or 4
PAR = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=NT, seed=SEED,
           deterministic=True, force_row_wise=True)
NR = {'SP': 400, 'PS': 400, 'US': 300, 'UP': 300}
POSK = ['first', 'lastsk', 'firstsp', 'last', 'rel', 'idx']
TOK = re.compile(r"[a-z']+|[?!.,—…]")


def build_pools(prof_df, turn_df, sol_df=None):
    pools = {}
    for pid, g in turn_df.groupby('pool_id', sort=True):
        pools[pid] = dict(pid=pid, tids=g.turn_id.tolist(), spk=g.speaker.tolist(), text=g.text.tolist())
    for pid, g in prof_df.groupby('pool_id', sort=True):
        pools[pid]['uids'] = g.profile_id.tolist()
        pools[pid]['prof'] = g.profile.tolist()
    if sol_df is not None:
        for r in sol_df.itertuples():
            pid, uid = r.id.split('-')
            pools[pid].setdefault('gold', {})[uid] = r.prediction.split()
            pools[pid].setdefault('mid', {})[uid] = r.member_id
    return pools


def pool_score(pool, pred):
    gold, uids = pool['gold'], pool['uids']
    M, N = len(uids), len(pool['tids'])
    gl, pl = set(), set()
    for u in uids:
        g = gold[u]; gl |= set(zip(g[:-1], g[1:]))
        q = pred[u]; pl |= set(zip(q[:-1], q[1:]))
    tp = len(gl & pl)
    f1 = 0.0 if tp == 0 else 2 * tp / (len(gl) + len(pl))
    owner = {t: u for u in uids for t in gold[u]}
    acc = sum(1 for u in uids for t in pred[u] if owner[t] == u) / N
    attr = max(0.0, (acc - 1 / M) / (1 - 1 / M))
    ex = sum(1 for u in uids if pred[u] == gold[u]) / M
    return 0.5 * f1 + 0.3 * attr + 0.2 * ex, f1, attr, ex


class TextRep:
    def __init__(self, corpus):
        self.wv = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2,
                                  token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")
        self.cv = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(3, 5), min_df=3,
                                  max_features=300000)
        W = self.wv.fit_transform(corpus)
        self.cv.fit(corpus)
        self.svd = TruncatedSVD(200, random_state=SEED).fit(W)

    def enc(self, texts):
        W = self.wv.transform(texts)
        return W, self.cv.transform(texts), normalize(self.svd.transform(W))


def prefix_doc(t):
    toks = TOK.findall(t.lower().replace('’', "'"))
    pre = ' '.join('B%d_%s' % (i, w.replace("'", "Q")) for i, w in enumerate(toks[:4]))
    suf = ' '.join('E%d_%s' % (i, w.replace("'", "Q")) for i, w in enumerate(toks[::-1][:3]))
    return pre + ' ' + suf


class PosModel:
    def fit(self, df):
        L = df.groupby('session_id').turn_index.transform('max').values + 1
        ti = df.turn_index.values
        self.v1 = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_features=200000,
                                  token_pattern=r"(?u)\b\w[\w']*\b|[?!]")
        self.v2 = TfidfVectorizer(token_pattern=r"\S+", min_df=3, lowercase=False)
        X = self._X(df.text.tolist(), fit=True)
        seek = df.speaker.values == 'seeker'
        self.m = {}
        targets = {'first': (seek, ti == 0), 'lastsk': (seek, ti == L - 2),
                   'firstsp': (~seek, ti == 1), 'last': (~seek, ti == L - 1)}
        for k, (mask, y) in targets.items():
            self.m[k] = LogisticRegression(C=2.0, max_iter=2000).fit(X[mask], y[mask])
        self.m['rel'] = Ridge(alpha=3.0).fit(X, ti / (L - 1))
        self.m['idx'] = Ridge(alpha=3.0).fit(X, np.log1p(ti))
        return self

    def _X(self, texts, fit=False):
        pre = [prefix_doc(t) for t in texts]
        ln = np.array([[np.log1p(len(t.split())), t.count('?') > 0] for t in texts], dtype=np.float64)
        if fit:
            A = self.v1.fit_transform(texts); B = self.v2.fit_transform(pre)
        else:
            A = self.v1.transform(texts); B = self.v2.transform(pre)
        return sp.hstack([A, B, sp.csr_matrix(ln)]).tocsr()

    def predict(self, texts):
        X = self._X(texts)
        out = {k: self.m[k].predict_proba(X)[:, 1] for k in ['first', 'lastsk', 'firstsp', 'last']}
        out['rel'] = self.m['rel'].predict(X)
        out['idx'] = self.m['idx'].predict(X)
        return out


def rel_feats(S):
    rr = (-S).argsort(1).argsort(1).astype(np.float32)
    cr = (-S).argsort(0).argsort(0).astype(np.float32)
    f = [rr, cr, S - S.max(1, keepdims=True), S - S.max(0, keepdims=True)]
    Z = S / (S.std() + 1e-6)
    e = np.exp(Z - Z.max(1, keepdims=True)); f.append(e / e.sum(1, keepdims=True))
    e = np.exp(Z - Z.max(0, keepdims=True)); f.append(e / e.sum(0, keepdims=True))
    return f


def sim_mats(W, C, L, W2, C2, L2):
    Wb = (W > 0).astype(np.float32); Wb2 = (W2 > 0).astype(np.float32)
    return [(W @ W2.T).toarray(), (C @ C2.T).toarray(), L @ L2.T, (Wb @ Wb2.T).toarray()]


def pool_feats(p, enc, penc):
    spk = np.array(p['spk'])
    S = np.where(spk == 'seeker')[0]; P = np.where(spk == 'supporter')[0]
    W, C, L = enc
    sims = sim_mats(W, C, L, W, C, L)
    tb = np.array([[len(t.split()), t.count('?'), t.count('!'), t.strip().endswith('?')] for t in p['text']],
                  dtype=np.float32)
    PT = np.stack([p['pos'][k] for k in POSK], 1)
    n, M = len(S), len(p['uids'])

    def block(A, B):
        feats = []
        for s in sims:
            m = s[np.ix_(A, B)]
            feats.append(m); feats += rel_feats(m)
        na, nb = len(A), len(B)
        for k in range(PT.shape[1]):
            feats.append(np.repeat(PT[A, k][:, None], nb, 1)); feats.append(np.repeat(PT[B, k][None, :], na, 0))
        feats.append(np.repeat(PT[A, 4][:, None], nb, 1) - np.repeat(PT[B, 4][None, :], na, 0))
        feats.append(np.repeat(PT[A, 5][:, None], nb, 1) - np.repeat(PT[B, 5][None, :], na, 0))
        for k in range(tb.shape[1]):
            feats.append(np.repeat(tb[A, k][:, None], nb, 1)); feats.append(np.repeat(tb[B, k][None, :], na, 0))
        feats.append(np.full((na, nb), n, np.float32)); feats.append(np.full((na, nb), M, np.float32))
        return np.stack(feats, -1).reshape(na * nb, -1).astype(np.float32)

    out = {'SP': block(S, P), 'PS': block(P, S), 'S': S, 'P': P}
    Wp, Cp, Lp = penc
    feats = []
    for m in sim_mats(Wp, Cp, Lp, W[S], C[S], L[S]):
        feats.append(m); feats += rel_feats(m)
    feats.append(np.repeat(PT[S, 4][None, :], M, 0)); feats.append(np.full((M, n), M, np.float32))
    out['US'] = np.stack(feats, -1).reshape(M * n, -1).astype(np.float32)
    return out


def pool_labels(p, S, P):
    nxt, owner = {}, {}
    for u in p['uids']:
        g = p['gold'][u]
        for i, t in enumerate(g):
            owner[t] = u
            if i + 1 < len(g):
                nxt[t] = g[i + 1]
    t = p['tids']
    return {'SP': np.array([nxt.get(t[a]) == t[b] for a in S for b in P], np.int8),
            'PS': np.array([nxt.get(t[a]) == t[b] for a in P for b in S], np.int8),
            'US': np.array([owner[t[b]] == u for u in p['uids'] for b in S], np.int8),
            'UP': np.array([owner[t[b]] == u for u in p['uids'] for b in P], np.int8)}


def member_aff(pools, wts, encs):
    agg = {}
    for pid, p in pools.items():
        X = encs[pid]
        for k in range(len(wts[pid])):
            Wk = sp.csr_matrix(wts[pid][k])
            for e in range(len(X)):
                Sm = (Wk @ X[e]).tocsr()
                for ui, prof in enumerate(p['prof']):
                    agg.setdefault((prof, k, e), {})[pid] = Sm[ui]
    tot = {key: sum(v.values()) for key, v in agg.items()}
    out = {}
    for pid, p in pools.items():
        X = encs[pid]
        mats = []
        for k in range(len(wts[pid])):
            for e in range(len(X)):
                cen = normalize(sp.vstack([tot[(prof, k, e)] - agg[(prof, k, e)][pid] for prof in p['prof']]).tocsr())
                mats.append((X[e] @ cen.T).toarray().T)
        out[pid] = mats
    return out


def round_feats(A, spk, S, P):
    sk = np.array(spk) == 'seeker'
    zs = A[0] + A[1]; zp = A[2] + A[3]
    z = np.where(sk[None], zs, zp + 0.5 * zs)
    zz = 20.0 * (z - z.max(0, keepdims=True)); q = np.exp(zz); q = q / q.sum(0, keepdims=True)
    M = z.shape[0]
    out = {}

    def turnf(idx):
        f = []
        for m in A + [z]:
            mm = m[:, idx]
            f += [mm, mm - mm.mean(0, keepdims=True), mm - mm.max(0, keepdims=True)] + rel_feats(mm)
        f.append(q[:, idx])
        return np.stack(f, -1).reshape(M * len(idx), -1).astype(np.float32)

    def pairf(a, b):
        qa, qb = q[:, a], q[:, b]
        f = [qa.T @ qb, qa.max(0)[:, None] * qb.max(0)[None], (qa.argmax(0)[:, None] == qb.argmax(0)[None]).astype(float)]
        za = z[:, a] - z[:, a].mean(0); zb = z[:, b] - z[:, b].mean(0)
        f.append((za.T @ zb) / (np.linalg.norm(za, axis=0)[:, None] * np.linalg.norm(zb, axis=0)[None] + 1e-9))
        return np.stack(f, -1).reshape(len(a) * len(b), -1).astype(np.float32)

    out['US'] = turnf(S); out['UP'] = turnf(P)
    out['SP'] = pairf(S, P); out['PS'] = pairf(P, S)
    return out


def nlog(p):
    return -np.log(np.clip(p, 1e-9, 1))


def logodds(x):
    return np.log(np.clip(x, 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - x, 1e-6, 1))


def chain(cost, cfirst, clast, M):
    n = len(cfirst)
    cost = cost.astype(np.float64).copy()
    chains = []
    for it in range(50):
        A = np.full((n + M, n + M), 1e6)
        A[:n, :n] = cost
        np.fill_diagonal(A[:n, :n], 1e6)
        A[:n, n:] = clast[:, None]
        A[n:, :n] = cfirst[None, :]
        r, c = linear_sum_assignment(A)
        succ = {a: b for a, b in zip(r, c) if a < n and b < n}
        heads = [b for a, b in zip(r, c) if a >= n and b < n]
        chains, seen = [], set()
        for h in heads:
            ch = [h]; seen.add(h)
            while ch[-1] in succ:
                ch.append(succ[ch[-1]]); seen.add(ch[-1])
            chains.append(ch)
        rest = [k for k in range(n) if k not in seen]
        if not rest:
            return chains
        vis = set()
        for k in rest:
            if k in vis:
                continue
            cyc = [k]; vis.add(k)
            while succ[cyc[-1]] not in vis:
                cyc.append(succ[cyc[-1]]); vis.add(cyc[-1])
            a, b = max([(x, succ[x]) for x in cyc], key=lambda e: cost[e[0], e[1]])
            cost[a, b] += 50
    for k in rest:
        chains[0].append(k)
    return chains


def decode(p, S, P, pr):
    n, M = len(S), len(p['uids'])
    c = linear_sum_assignment(nlog(pr['SP'].reshape(n, n)))[1]
    chains = chain(nlog(pr['PS'].reshape(n, n)[c, :]), nlog(p['pos']['first'][S]), nlog(p['pos']['last'][P[c]]), M)
    ls = logodds(pr['US'].reshape(M, n))
    lp = logodds(pr['UP'].reshape(M, n)) if 'UP' in pr else np.zeros((M, n))
    A = np.array([[ls[u, ch].sum() + lp[u, c[ch]].sum() for u in range(M)] for ch in chains])
    r, cc = linear_sum_assignment(-A)
    pred = {}
    for a, b in zip(r, cc):
        pred[p['uids'][b]] = [x for k in chains[a] for x in (p['tids'][S[k]], p['tids'][P[c[k]]])]
    return pred


class LocalSearch:
    def __init__(self, sp_, ps, fi, la, au, ap):
        self.sp, self.ps, self.fi, self.la, self.au, self.ap = sp_, ps, fi, la, au, ap

    def cs(self, u, ch):
        if not ch:
            return -1e9
        I = [e[0] for e in ch]; J = [e[1] for e in ch]
        s = self.fi[I[0]] + self.la[J[-1]] + self.sp[I, J].sum() + self.au[u, I].sum() + self.ap[u, J].sum()
        if len(ch) > 1:
            s += self.ps[J[:-1], I[1:]].sum()
        return s

    def run(self, chains, max_it=100):
        M = len(chains)
        sc = [self.cs(u, chains[u]) for u in range(M)]
        for it in range(max_it):
            best = (1e-9, None)
            for u in range(M):
                for v in range(u + 1, M):
                    U, V = chains[u], chains[v]
                    base = sc[u] + sc[v]
                    for a in range(len(U) + 1):
                        for b in range(len(V) + 1):
                            if a == len(U) and b == len(V):
                                continue
                            nu = U[:a] + V[b:]; nv = V[:b] + U[a:]
                            if not nu or not nv:
                                continue
                            d = self.cs(u, nu) + self.cs(v, nv) - base
                            if d > best[0]:
                                best = (d, ((u, nu), (v, nv)))
            for u in range(M):
                U = chains[u]
                for a in range(len(U)):
                    e = U[a]; Ur = U[:a] + U[a + 1:]
                    if not Ur:
                        continue
                    su = self.cs(u, Ur)
                    for v in range(M):
                        if v == u:
                            for b in range(len(Ur) + 1):
                                if b == a:
                                    continue
                                nu = Ur[:b] + [e] + Ur[b:]
                                d = self.cs(u, nu) - sc[u]
                                if d > best[0]:
                                    best = (d, ((u, nu),))
                        else:
                            V = chains[v]
                            for b in range(len(V) + 1):
                                nv = V[:b] + [e] + V[b:]
                                d = su + self.cs(v, nv) - sc[u] - sc[v]
                                if d > best[0]:
                                    best = (d, ((u, Ur), (v, nv)))
            loc = [(u, a) for u in range(M) for a in range(len(chains[u]))]
            for x in range(len(loc)):
                for y in range(x + 1, len(loc)):
                    (u, a), (v, b) = loc[x], loc[y]
                    if u == v:
                        U = list(chains[u]); e1, e2 = U[a], U[b]
                        U[a] = (e1[0], e2[1]); U[b] = (e2[0], e1[1])
                        d = self.cs(u, U) - sc[u]
                        if d > best[0]:
                            best = (d, ((u, U),))
                    else:
                        U = list(chains[u]); V = list(chains[v]); e1, e2 = U[a], V[b]
                        U[a] = (e1[0], e2[1]); V[b] = (e2[0], e1[1])
                        d = self.cs(u, U) + self.cs(v, V) - sc[u] - sc[v]
                        if d > best[0]:
                            best = (d, ((u, U), (v, V)))
            if best[1] is None:
                break
            for u, ch in best[1]:
                chains[u] = ch; sc[u] = self.cs(u, ch)
        return chains


def lsm(L):
    L = L - L.max(0, keepdims=True)
    return L - np.log(np.exp(L).sum(0, keepdims=True))


def refine(args):
    p, S, P, pr, pred = args
    n, M = len(S), len(p['uids'])
    spm = np.log(np.clip(pr['SP'].reshape(n, n), 1e-6, 1))
    psm = np.log(np.clip(pr['PS'].reshape(n, n), 1e-6, 1))
    fi = 0.5 * np.log(np.clip(p['pos']['first'][S], 1e-6, 1))
    la = 0.5 * np.log(np.clip(p['pos']['last'][P], 1e-6, 1))
    au = lsm(logodds(pr['US'].reshape(M, n))); ap = lsm(logodds(pr['UP'].reshape(M, n)))
    ix = {t: i for i, t in enumerate(p['tids'])}
    sx = {s: k for k, s in enumerate(S)}; px = {s: k for k, s in enumerate(P)}
    chains = []
    for u in p['uids']:
        seq = [ix[t] for t in pred[u]]
        chains.append([(sx[seq[k]], px[seq[k + 1]]) for k in range(0, len(seq), 2)])
    chains = LocalSearch(spm, psm, fi, la, au, ap).run(chains)
    return {u: [x for (i, j) in chains[ui] for x in (p['tids'][S[i]], p['tids'][P[j]])] for ui, u in enumerate(p['uids'])}


def post_from_pred(p, pred):
    Pm = np.zeros((len(p['uids']), len(p['tids'])))
    ix = {t: i for i, t in enumerate(p['tids'])}
    for ui, u in enumerate(p['uids']):
        for t in pred.get(u, []):
            Pm[ui, ix[t]] = 1
    return Pm


def fit_predict(key, X, Y, tr_ids, fold_of, te_ids):
    out = {}
    for f in range(NF):
        tr = [q for q in tr_ids if fold_of[q] != f]
        b = lgb.train(PAR, lgb.Dataset(np.concatenate([X[q][key] for q in tr]), np.concatenate([Y[q][key] for q in tr])), NR[key])
        for q in tr_ids:
            if fold_of[q] == f:
                out[q] = b.predict(X[q][key])
    b = lgb.train(PAR, lgb.Dataset(np.concatenate([X[q][key] for q in tr_ids]), np.concatenate([Y[q][key] for q in tr_ids])), NR[key])
    for q in te_ids:
        out[q] = b.predict(X[q][key])
    return out


def main():
    data_dir, out_path = sys.argv[1], sys.argv[2]
    np.random.seed(SEED)
    # Inputs are limited to the released public train and test files.
    rd = lambda f: pd.read_csv(os.path.join(data_dir, f), keep_default_na=False)
    members, ses = rd('train_members.csv'), rd('train_sessions.csv')
    trp, trt, trs = rd('train_pools.csv'), rd('train_pool_turns.csv'), rd('train_pools_solution.csv')
    tep, tet = rd('test_pools.csv'), rd('test_turns.csv')
    sub_ids = rd('sample_submission.csv').id.tolist()

    topic_of = dict(zip(members.member_id, members.topic))
    trpools = build_pools(trp, trt, trs)
    tepools = build_pools(tep, tet)
    topics = sorted(members.topic.unique())
    perm = np.random.RandomState(SEED).permutation(topics)
    ftopic = {t: i % NF for i, t in enumerate(perm)}
    fold_of = {q: ftopic[topic_of[next(iter(p['mid'].values()))]] for q, p in trpools.items()}
    tr_ids, te_ids = sorted(trpools), sorted(tepools)
    pools = {**trpools, **tepools}

    corpus = ses.text.tolist() + tet.text.tolist() + members.profile.tolist() + tep.profile.tolist()
    rep = TextRep(corpus)
    print('text representation ready', flush=True)
    ses = ses.assign(fold=ses.member_id.map(topic_of).map(ftopic))
    for f in range(NF):
        pm = PosModel().fit(ses[ses.fold != f])
        for q in tr_ids:
            if fold_of[q] == f:
                trpools[q]['pos'] = pm.predict(trpools[q]['text'])
    pm = PosModel().fit(ses)
    for q in te_ids:
        tepools[q]['pos'] = pm.predict(tepools[q]['text'])
    print('position models ready', flush=True)

    X, Y, encs, SP_ = {}, {}, {}, {}
    for q, p in pools.items():
        enc = rep.enc(p['text'])
        encs[q] = [enc[0], enc[1]]
        fe = pool_feats(p, enc, rep.enc(p['prof']))
        SP_[q] = (fe['S'], fe['P'])
        X[q] = {k: fe[k] for k in ['SP', 'PS', 'US']}
        if q in trpools:
            Y[q] = pool_labels(p, fe['S'], fe['P'])
    base = {q: dict(X[q]) for q in pools}
    print('features ready', flush=True)

    PR = {q: {} for q in pools}
    for key in ['SP', 'PS', 'US']:
        for q, v in fit_predict(key, X, Y, tr_ids, fold_of, te_ids).items():
            PR[q][key] = v
    pred = {q: decode(pools[q], *SP_[q], PR[q]) for q in pools}
    print('base round done', flush=True)

    for r in range(ROUNDS):
        post = {q: post_from_pred(pools[q], pred[q]) for q in pools}
        A = {}
        for ids in (tr_ids, te_ids):
            sub = {q: pools[q] for q in ids}
            wts = {}
            for q in ids:
                sk = (np.array(pools[q]['spk']) == 'seeker')[None]
                wts[q] = [post[q] * sk, post[q] * (~sk)]
            A.update(member_aff(sub, wts, {q: encs[q] for q in ids}))
        for q in pools:
            rf = round_feats(A[q], pools[q]['spk'], *SP_[q])
            X[q] = {k: np.hstack([base[q][k], rf[k]]) for k in ['SP', 'PS', 'US']}
            X[q]['UP'] = rf['UP']
        PR = {q: {} for q in pools}
        for key in ['SP', 'PS', 'US', 'UP']:
            for q, v in fit_predict(key, X, Y, tr_ids, fold_of, te_ids).items():
                PR[q][key] = v
        pred = {q: decode(pools[q], *SP_[q], PR[q]) for q in pools}
        res = np.array([pool_score(pools[q], pred[q]) for q in tr_ids])
        print('round %d validation score %.4f link_f1 %.4f attribution %.4f exact %.4f' % ((r,) + tuple(res.mean(0))), flush=True)

    with Pool(NT) as pl:
        out = pl.map(refine, [(pools[q], *SP_[q], PR[q], pred[q]) for q in pools], chunksize=4)
    pred = dict(zip(list(pools), out))
    res = np.array([pool_score(pools[q], pred[q]) for q in tr_ids])
    print('final out-of-fold validation score %.4f link_f1 %.4f attribution %.4f exact %.4f' % tuple(res.mean(0)), flush=True)

    rows = []
    for i in sub_ids:
        q, u = i.split('-')
        rows.append((i, ' '.join(pred[q][u])))
    d = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows, columns=['id', 'prediction']).to_csv(out_path, index=False)
    print('wrote', out_path, len(rows), flush=True)


if __name__ == '__main__':
    main()
