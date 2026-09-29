"""Dispatch chain restoration.

Usage: python solution.py <public_dir> <submission_csv>

A pretrained ELECTRA-small encoder is fine-tuned on the training editions with three learned heads:
  * membership: an embedding trained with a within-edition multi-positive contrastive loss, so the
    five excerpts of one report are closer to each other than to the other seven related reports;
  * order: an asymmetric "leads-into" scorer plus a within-pool position logit, trained to pick the
    true reading order among the four orderings of a report's two middle and two closing excerpts;
  * the same encoder reads each opening together with its place claims.
At inference, the learned pairwise membership scores of each test edition are decoded with an exact
capacity-constrained assignment (integer program: every opening receives two middles and two
closings, every excerpt used once), alternating between the pools so that middle-closing
relationships inform both.  The learned order scores then pick the reading order of each chain.
"""
import os
import sys
import json
import math
import time
import random
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import milp, LinearConstraint, Bounds

MODEL_NAME = "google/electra-small-discriminator"
# Public copy of the same Hugging Face checkpoint, used only if the Hub itself is unreachable.
MIRROR = "https://s3.amazonaws.com/models.huggingface.co/bert/" + MODEL_NAME + "/"
MAX_LEN_OPEN, MAX_LEN_EXC = 160, 128
EPOCHS = 2
LR_ENC, LR_HEAD, WD, WARMUP = 1e-4, 1e-3, 0.01, 0.06
EMB_DIM, ORD_DIM = 256, 128
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PAIRS16 = list(itertools.combinations(range(16), 2))


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- model loading

def load_backbone():
    from transformers import AutoModel, AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME), AutoModel.from_pretrained(MODEL_NAME)
    except Exception:
        import urllib.request
        cache = os.path.join(os.path.expanduser("~"), ".cache", "dispatch_chain", MODEL_NAME.replace("/", "__"))
        os.makedirs(cache, exist_ok=True)
        for f in ("config.json", "vocab.txt", "pytorch_model.bin"):
            p = os.path.join(cache, f)
            if not os.path.exists(p):
                urllib.request.urlretrieve(MIRROR + f, p + ".tmp"); os.replace(p + ".tmp", p)
        return AutoTokenizer.from_pretrained(cache), AutoModel.from_pretrained(cache)


class ChainNet(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        d = enc.config.hidden_size
        self.z = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, EMB_DIM))
        self.a = nn.Linear(d, ORD_DIM)   # what a text leads into
        self.b = nn.Linear(d, ORD_DIM)   # what a text follows from
        self.pos = nn.Linear(d, 1)       # earlier-in-pool logit
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))

    def forward(self, ids, mask, tt):
        h = self.enc(input_ids=ids, attention_mask=mask, token_type_ids=tt).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        p = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return {"z": F.normalize(self.z(p).float(), dim=-1), "a": self.a(p).float(),
                "b": self.b(p).float(), "pos": self.pos(p).float().squeeze(-1), "log_scale": self.log_scale}


# ---------------------------------------------------------------- data

def load_editions(public, needed):
    ed = {}
    with open(os.path.join(public, "dispatches.jsonl"), "rb") as fh:
        for line in fh:
            e = json.loads(line)
            if e["edition_id"] in needed:
                ed[e["edition_id"]] = e
    return ed


def encode_edition(tok, e):
    """Tokenise one edition: 8 openings (text + place claims), 16 middles, 16 closings (in that order)."""
    op = tok([o["text"] for o in e["openings"]], ["; ".join(o["place_claims"]) or "none" for o in e["openings"]],
             truncation="only_first", max_length=MAX_LEN_OPEN)
    ex = tok([x["text"] for x in e["middle_excerpts"] + e["closing_excerpts"]], truncation=True, max_length=MAX_LEN_EXC)
    seqs = [(op["input_ids"][i], op["token_type_ids"][i]) for i in range(8)]
    seqs += [(ex["input_ids"][i], ex["token_type_ids"][i]) for i in range(32)]
    L = max(len(s[0]) for s in seqs)
    ids = torch.zeros(40, L, dtype=torch.long); tt = torch.zeros(40, L, dtype=torch.long); mask = torch.zeros(40, L, dtype=torch.long)
    for i, (a, t) in enumerate(seqs):
        ids[i, :len(a)] = torch.tensor(a); tt[i, :len(a)] = torch.tensor(t); mask[i, :len(a)] = 1
    return ids, mask, tt


def edition_truth(e, rows):
    """rows: list of (opening_id, prediction dict). Returns chains as index tuples into the 40 texts."""
    oi = {o["opening_id"]: i for i, o in enumerate(e["openings"])}
    mi = {x["excerpt_id"]: 8 + i for i, x in enumerate(e["middle_excerpts"])}
    ci = {x["excerpt_id"]: 24 + i for i, x in enumerate(e["closing_excerpts"])}
    return [(oi[o], mi[p["middle_ids"][0]], mi[p["middle_ids"][1]], ci[p["closing_ids"][0]], ci[p["closing_ids"][1]])
            for o, p in rows]


# ---------------------------------------------------------------- losses

def order_scores(out, chains):
    """chains: LongTensor (n, 5) of index sequences o, m1, m2, c1, c2 -> learned order score per chain."""
    a, b, pos = out["a"], out["b"], out["pos"]
    s = 0
    for k in range(4):
        s = s + (a[chains[:, k]] * b[chains[:, k + 1]]).sum(-1) / math.sqrt(ORD_DIM)
    return s + pos[chains[:, 1]] - pos[chains[:, 2]] + pos[chains[:, 3]] - pos[chains[:, 4]]


PERMS = [(0, 1, 2, 3, 4), (0, 2, 1, 3, 4), (0, 1, 2, 4, 3), (0, 2, 1, 4, 3)]


def edition_loss(out, chains):
    n = out["z"].shape[0]
    grp = torch.full((n,), -1, dtype=torch.long, device=out["z"].device)
    for g, ch in enumerate(chains):
        grp[list(ch)] = g
    sim = out["z"] @ out["z"].T * out["log_scale"].exp()
    eye = torch.eye(n, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(eye, -1e4)
    pos = (grp[:, None] == grp[None, :]) & ~eye
    lse_all = torch.logsumexp(sim, 1)
    lse_pos = torch.logsumexp(sim.masked_fill(~pos, -1e4), 1)
    l_mem = (lse_all - lse_pos).mean()
    ch = torch.tensor(chains, device=sim.device)
    sc = torch.stack([order_scores(out, ch[:, list(p)]) for p in PERMS], 1)
    l_ord = F.cross_entropy(sc, torch.zeros(len(chains), dtype=torch.long, device=sim.device))
    return l_mem + l_ord, l_mem.item(), l_ord.item()


# ---------------------------------------------------------------- training / inference

def run_model(net, batch, train):
    ids, mask, tt = (t.to(DEV) for t in batch)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
        return net(ids, mask, tt)


def train_model(tok, train_eds, train_rows, epochs=EPOCHS, log=print):
    seed_all(SEED)
    tok_, enc = tok
    net = ChainNet(enc).to(DEV)
    data = [(encode_edition(tok_, train_eds[eid]), edition_truth(train_eds[eid], rows)) for eid, rows in train_rows.items()]
    heads = [p for n, p in net.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": LR_ENC}, {"params": heads, "lr": LR_HEAD}], weight_decay=WD)
    total = epochs * len(data); warm = int(WARMUP * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (total - s) / max(1, total - warm)))
    step, t0 = 0, time.time()
    for ep in range(epochs):
        net.train(); order = np.random.permutation(len(data)); agg = []
        for i in order:
            batch, chains = data[i]
            out = run_model(net, batch, True)
            loss, lm, lo = edition_loss(out, chains)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1
            agg.append((lm, lo))
            if step % 100 == 0:
                a = np.mean(agg[-100:], 0)
                log(f"ep {ep} step {step}/{total} mem {a[0]:.3f} ord {a[1]:.3f} {time.time() - t0:.0f}s")
    return net


@torch.no_grad()
def infer_edition(net, tok_, e):
    net.eval()
    out = run_model(net, encode_edition(tok_, e), False)
    return {k: v.float().cpu() for k, v in out.items()}


# ---------------------------------------------------------------- decoding

def solve_pairs(score_o_pair):
    """score_o_pair: (8, 120) array. Exact assignment of one disjoint pair per opening."""
    n = 8 * 120
    c = -score_o_pair.reshape(-1)
    A = np.zeros((8 + 16, n))
    for o in range(8):
        A[o, o * 120:(o + 1) * 120] = 1
        for p, (a, b) in enumerate(PAIRS16):
            A[8 + a, o * 120 + p] = 1; A[8 + b, o * 120 + p] = 1
    res = milp(c, constraints=LinearConstraint(A, 1, 1), integrality=np.ones(n), bounds=Bounds(0, 1))
    x = res.x.reshape(8, 120)
    return [PAIRS16[int(np.argmax(x[o]))] for o in range(8)]


@torch.no_grad()
def decode_edition(out, iters=4):
    S = (out["z"] @ out["z"].T * out["log_scale"].exp()).numpy().astype(np.float64)
    O, Mi, Ci = np.arange(8), np.arange(8, 24), np.arange(24, 40)
    pa = np.array([p[0] for p in PAIRS16]); pb = np.array([p[1] for p in PAIRS16])

    def pair_scores(items, other):
        base = S[np.ix_(O, items)]
        s = base[:, pa] + base[:, pb] + S[items[pa], items[pb]][None, :]
        if other is not None:
            for o in range(8):
                oth = other[o]
                s[o] += S[np.ix_(items[pa], oth)].sum(1) + S[np.ix_(items[pb], oth)].sum(1)
        return s

    mid = solve_pairs(pair_scores(Mi, None)); clo = solve_pairs(pair_scores(Ci, None))
    for _ in range(iters):
        mid_new = solve_pairs(pair_scores(Mi, [Ci[list(clo[o])] for o in range(8)]))
        clo_new = solve_pairs(pair_scores(Ci, [Mi[list(mid_new[o])] for o in range(8)]))
        if mid_new == mid and clo_new == clo:
            break
        mid, clo = mid_new, clo_new
    chains = []
    for o in range(8):
        m, c = Mi[list(mid[o])], Ci[list(clo[o])]
        cands = torch.tensor([(o, m[i], m[1 - i], c[j], c[1 - j]) for i in (0, 1) for j in (0, 1)])
        chains.append(tuple(cands[int(order_scores(out, cands).argmax())].tolist()))
    return chains


def chains_to_rows(e, chains):
    mids = [x["excerpt_id"] for x in e["middle_excerpts"]]; clos = [x["excerpt_id"] for x in e["closing_excerpts"]]
    res = {}
    for o, m1, m2, c1, c2 in chains:
        res[e["openings"][o]["opening_id"]] = {"middle_ids": [mids[m1 - 8], mids[m2 - 8]],
                                               "closing_ids": [clos[c1 - 24], clos[c2 - 24]]}
    return res


def target_score(pred, truth):
    A = (len(set(pred["middle_ids"]) & set(truth["middle_ids"])) + len(set(pred["closing_ids"]) & set(truth["closing_ids"]))) / 4
    M = float(pred["middle_ids"] == truth["middle_ids"]); C = float(pred["closing_ids"] == truth["closing_ids"])
    return 0.2 * A + 0.2 * M + 0.2 * C + 0.4 * M * C


# ---------------------------------------------------------------- main

def main():
    public, out_path = sys.argv[1], sys.argv[2]
    t0 = time.time()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    # Inputs are limited to the released public dispatches, train, train_targets and test files.
    train = pd.read_csv(os.path.join(public, "train.csv")).merge(pd.read_csv(os.path.join(public, "train_targets.csv")), on="target_id")
    test = pd.read_csv(os.path.join(public, "test.csv"))
    eds = load_editions(public, set(train.edition_id) | set(test.edition_id))
    train_rows = {}
    for r in train.itertuples():
        train_rows.setdefault(r.edition_id, []).append((r.opening_id, json.loads(r.prediction)))
    tok = load_backbone()
    net = train_model(tok, eds, train_rows)
    print(f"trained in {time.time() - t0:.0f}s", flush=True)
    preds = {}
    for eid in sorted(set(test.edition_id)):
        e = eds[eid]
        preds[eid] = chains_to_rows(e, decode_edition(infer_edition(net, tok[0], e)))
    rows = [(r.target_id, json.dumps(preds[r.edition_id][r.opening_id], separators=(",", ":"))) for r in test.itertuples()]
    sub = pd.DataFrame(rows, columns=["target_id", "prediction"])
    assert len(sub) == len(test) and sub.target_id.is_unique
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"wrote {len(sub)} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
