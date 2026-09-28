import os
import sys
import json
import math
import re
import time
import random
import itertools
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_NAME = "FacebookAI/xlm-roberta-large"
MAX_LEN = 512
STRIDE = 128
MAX_ANSWER_TOKENS = 128
TOP_N = 20
TOP_K = 20
LR = 1.5e-5
LLRD = 0.95
WEIGHT_DECAY = 0.01
WARMUP = 0.1
BATCH = 16
MICRO = 8
EPOCH_OPTIONS = (3, 2)
HOLDOUT_FRAC = 0.12
SEEDS = (42, 1337)
TIME_LIMIT = 90 * 60
SAFETY = 14 * 60
SPLIT_SEED = 2024
T_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
D_GRID = (0.0, -0.5, 0.5, -1.0, 1.0, 2.0)

START = time.time()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"\w+", re.UNICODE)


def log(msg):
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_tokens(text):
    return TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def token_f1(pred, gold):
    if not pred or not gold:
        return float(pred == gold)
    common = sum((Counter(pred) & Counter(gold)).values())
    if common == 0:
        return 0.0
    p = common / len(pred)
    r = common / len(gold)
    return 2 * p * r / (p + r)


def official_score(pred_spans, gold_spans, passages, groups):
    packet_scores = []
    for pred, gold, passage in zip(pred_spans, gold_spans, passages):
        utils, quals = [], []
        for (ps, pe), (gs, ge) in zip(pred, gold):
            if ps < 0:
                utils.append(1.0 if gs < 0 else 0.0)
                continue
            if gs < 0 or pe > len(passage):
                q = 0.0
            else:
                q = token_f1(norm_tokens(passage[ps:pe]), norm_tokens(passage[gs:ge]))
            utils.append(q)
            quals.append(q)
        packet_scores.append(float(np.mean(utils)) * (min(quals) if quals else 1.0))
    frame = pd.DataFrame({"g": groups, "s": packet_scores})
    return float(frame.groupby("g")["s"].mean().mean())


def lexical_miss(question, passage, heading):
    known = {w[:4] for w in WORD_RE.findall((passage + " " + heading).casefold())}
    words = [w for w in WORD_RE.findall(question.casefold()) if len(w) > 3]
    return min(4, sum(1 for w in words if w[:4] not in known))


def load_packets(path, with_labels):
    frame = pd.read_csv(path, dtype={"id": str, "group_id": str}, keep_default_na=False)
    packets = []
    for row in frame.itertuples(index=False):
        questions = json.loads(row.questions)
        spans = json.loads(row.spans) if with_labels else [[-1, -1]] * 4
        clean = []
        for s, e in spans:
            s, e = int(s), int(e)
            if s >= 0:
                while s < e and row.passage[s].isspace():
                    s += 1
                while e > s and row.passage[e - 1].isspace():
                    e -= 1
                if s >= e:
                    s, e = -1, -1
            clean.append((s, e))
        packets.append({"id": row.id, "group": row.group_id, "heading": row.heading,
                        "passage": row.passage, "questions": questions, "spans": clean})
    return packets


def build_features(tokenizer, packets, with_targets):
    first, second, owners = [], [], []
    for pi, pk in enumerate(packets):
        for slot, q in enumerate(pk["questions"]):
            first.append(f"{pk['heading']}: {q}")
            second.append(pk["passage"])
            owners.append((pi, slot))
    enc = tokenizer(first, second, truncation="only_second", max_length=MAX_LEN, stride=STRIDE,
                    return_overflowing_tokens=True, return_offsets_mapping=True, padding=False)
    feats = []
    for i, ids in enumerate(enc["input_ids"]):
        pi, slot = owners[enc["overflow_to_sample_mapping"][i]]
        seq = enc.sequence_ids(i)
        ctx = np.array([t for t, s in enumerate(seq) if s == 1], dtype=np.int64)
        offsets = enc["offset_mapping"][i]
        feat = {"ids": ids, "packet": pi, "slot": slot, "ctx": ctx, "offsets": offsets}
        if with_targets:
            gs, ge = packets[pi]["spans"][slot]
            sp, ep = 0, 0
            if gs >= 0 and len(ctx) and offsets[ctx[0]][0] <= gs and offsets[ctx[-1]][1] >= ge:
                starts = [t for t in ctx if offsets[t][1] > gs]
                ends = [t for t in ctx if offsets[t][0] < ge]
                if starts and ends and starts[0] <= ends[-1]:
                    sp, ep = int(starts[0]), int(ends[-1])
            feat["start"], feat["end"] = sp, ep
        feats.append(feat)
    return feats


def collate(feats, pad_id):
    width = max(len(f["ids"]) for f in feats)
    ids = torch.full((len(feats), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(feats), width), dtype=torch.long)
    for i, f in enumerate(feats):
        ids[i, :len(f["ids"])] = torch.tensor(f["ids"], dtype=torch.long)
        mask[i, :len(f["ids"])] = 1
    return ids, mask


def length_batches(feats, size, rng):
    order = rng.permutation(len(feats))
    chunk = size * 64
    batches = []
    for c in range(0, len(order), chunk):
        part = sorted(order[c:c + chunk], key=lambda i: len(feats[i]["ids"]))
        batches.extend(part[b:b + size] for b in range(0, len(part), size))
    rng.shuffle(batches)
    return batches


def autocast():
    if DEVICE.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", enabled=False)


def load_model():
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, attn_implementation="sdpa")
    except (TypeError, ValueError, ImportError):
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    return model.to(DEVICE)


def param_groups(model):
    n_layers = model.config.num_hidden_layers
    groups = defaultdict(list)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(r"encoder\.layer\.(\d+)\.", name)
        if m:
            depth = int(m.group(1)) + 1
        elif "embeddings" in name:
            depth = 0
        else:
            depth = n_layers + 1
        decay = 0.0 if (name.endswith("bias") or "LayerNorm" in name or "layer_norm" in name) else WEIGHT_DECAY
        groups[(depth, decay)].append(p)
    return [{"params": ps, "lr": LR * LLRD ** (n_layers + 1 - depth), "weight_decay": wd}
            for (depth, wd), ps in groups.items()]


def train_model(seed, feats, pad_id, plan):
    set_seed(seed)
    model = load_model()
    model.train()
    opt_kwargs = {"betas": (0.9, 0.98), "eps": 1e-6}
    if DEVICE.type == "cuda":
        opt_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(param_groups(model), **opt_kwargs)
    except (TypeError, RuntimeError):
        opt_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(param_groups(model), **opt_kwargs)
    steps_per_epoch = math.ceil(len(feats) / BATCH)
    plan["total"] = plan["epochs"] * steps_per_epoch

    def lr_lambda(step):
        warm = max(1, int(WARMUP * plan["total"]))
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (plan["total"] - step) / max(1, plan["total"] - warm))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_scaler = DEVICE.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    rng = np.random.RandomState(seed)
    step, t_probe = 0, None
    done = False
    for epoch in range(max(EPOCH_OPTIONS)):
        for batch in length_batches(feats, BATCH, rng):
            if step >= plan["total"] or time.time() > plan["hard_stop"]:
                done = True
                break
            optimizer.zero_grad(set_to_none=True)
            for m0 in range(0, len(batch), MICRO):
                sub = [feats[i] for i in batch[m0:m0 + MICRO]]
                ids, mask = collate(sub, pad_id)
                ids, mask = ids.to(DEVICE), mask.to(DEVICE)
                sp = torch.tensor([f["start"] for f in sub], device=DEVICE)
                ep = torch.tensor([f["end"] for f in sub], device=DEVICE)
                with autocast():
                    out = model(input_ids=ids, attention_mask=mask)
                neg = torch.finfo(torch.float32).min
                s_log = out.start_logits.float().masked_fill(mask == 0, neg)
                e_log = out.end_logits.float().masked_fill(mask == 0, neg)
                loss = (F.cross_entropy(s_log, sp) + F.cross_entropy(e_log, ep)) / 2
                scaler.scale(loss * len(sub) / len(batch)).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if step == 5:
                t_probe = time.time()
            if step == 30 and plan.get("on_probe") is not None:
                plan["on_probe"]((time.time() - t_probe) / 25, steps_per_epoch)
            if step % 100 == 0:
                log(f"seed {seed} step {step}/{plan['total']} loss {loss.item():.4f}")
        if done or step >= plan["total"]:
            break
    log(f"seed {seed} finished after {step} steps")
    model.eval()
    return model


@torch.no_grad()
def predict_logits(model, feats, pad_id):
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["ids"]))
    result = [None] * len(feats)
    for b in range(0, len(order), 32):
        idx = order[b:b + 32]
        sub = [feats[i] for i in idx]
        ids, mask = collate(sub, pad_id)
        with autocast():
            out = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE))
        s_all = out.start_logits.float().cpu().numpy()
        e_all = out.end_logits.float().cpu().numpy()
        for j, i in enumerate(idx):
            n = len(feats[i]["ids"])
            result[i] = (s_all[j, :n], e_all[j, :n])
    return result


def question_pools(feats, logits, packets):
    pools = {}
    for f, (s_log, e_log) in zip(feats, logits):
        key = (f["packet"], f["slot"])
        passage = packets[f["packet"]]["passage"]
        pool = pools.setdefault(key, {"cands": {}, "null": float("inf")})
        pool["null"] = min(pool["null"], float(s_log[0] + e_log[0]))
        ctx = f["ctx"]
        if len(ctx) == 0:
            continue
        s_ctx, e_ctx = s_log[ctx], e_log[ctx]
        top_s = np.argsort(-s_ctx, kind="stable")[:TOP_N]
        top_e = np.argsort(-e_ctx, kind="stable")[:TOP_N]
        offs = f["offsets"]
        for a in top_s:
            for b in top_e:
                if b < a or b - a + 1 > MAX_ANSWER_TOKENS:
                    continue
                cs, ce = offs[ctx[a]][0], offs[ctx[b]][1]
                while cs < ce and passage[cs].isspace():
                    cs += 1
                while ce > cs and passage[ce - 1].isspace():
                    ce -= 1
                if cs >= ce:
                    continue
                score = float(s_ctx[a] + e_ctx[b])
                if score > pool["cands"].get((cs, ce), -float("inf")):
                    pool["cands"][(cs, ce)] = score
    out = {}
    for key, pool in pools.items():
        items = sorted(pool["cands"].items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K]
        spans = [k for k, _ in items]
        scores = np.array([v for _, v in items], dtype=np.float64)
        best = scores.max() if len(scores) else pool["null"] - 50.0
        out[key] = {"spans": spans, "scores": scores, "margin": pool["null"] - best}
    return out


def null_features(packets, pools, pidx):
    rows = []
    for pi in pidx:
        pk = packets[pi]
        margins = [pools[(pi, s)]["margin"] for s in range(4)]
        for s in range(4):
            others = np.mean([margins[t] for t in range(4) if t != s])
            rows.append([margins[s], others, lexical_miss(pk["questions"][s], pk["passage"], pk["heading"])])
    return np.array(rows, dtype=np.float64)


def fit_null_model(x, y):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit((x - mu) / sd, y)
    return lambda z: clf.decision_function((z - mu) / sd)


GRID = np.linspace(0.0, 1.0, 11)
SUBSETS = sorted([c for k in range(1, 5) for c in itertools.combinations(range(4), k)], key=len)
SUM_MIN = {}
for _k in range(1, 5):
    _idx = np.indices((11,) * _k).reshape(_k, -1)
    _vals = GRID[_idx]
    SUM_MIN[_k] = (_vals.sum(0) * _vals.min(0), _vals.min(0))


class TokenCache:
    def __init__(self):
        self.tokens = {}
        self.matrices = {}

    def get(self, passage_id, passage, span):
        key = (passage_id, span)
        if key not in self.tokens:
            self.tokens[key] = norm_tokens(passage[span[0]:span[1]])
        return self.tokens[key]

    def f1_matrix(self, passage_id, passage, spans):
        key = (passage_id, tuple(spans))
        if key not in self.matrices:
            toks = [self.get(passage_id, passage, s) for s in spans]
            self.matrices[key] = np.array([[token_f1(a, b) for b in toks] for a in toks])
        return self.matrices[key]


def question_choice(dists, p_null, passage_id, passage, cache):
    merged = defaultdict(float)
    for spans, probs in dists:
        for sp, pr in zip(spans, probs):
            merged[sp] += pr / len(dists)
    if not merged:
        return None, None
    spans = sorted(merged.keys())
    probs = np.array([merged[s] for s in spans])
    probs = probs / probs.sum()
    mat = cache.f1_matrix(passage_id, passage, spans)
    choice = int(np.argmax(mat @ probs))
    dist = np.zeros(11)
    np.add.at(dist, np.rint(mat[choice] * 10).astype(int), probs * (1.0 - p_null))
    dist[0] += p_null
    return spans[choice], dist


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


def decide(packets, pidx, model_outputs, temp, cache):
    preds = []
    for pi in pidx:
        pk = packets[pi]
        choices, qdists, p_nulls = [], [], []
        for s in range(4):
            dists, pn = [], []
            for pools, null_logit in model_outputs:
                pool = pools[(pi, s)]
                if len(pool["spans"]):
                    z = pool["scores"] / temp
                    z = np.exp(z - z.max())
                    dists.append((pool["spans"], z / z.sum()))
                pn.append(1.0 / (1.0 + math.exp(-null_logit[(pi, s)])))
            p_null = float(np.mean(pn))
            span, dist = question_choice(dists, p_null, pk["id"], pk["passage"], cache)
            choices.append(span)
            qdists.append(dist)
            p_nulls.append(p_null)
        chosen = packet_decision(qdists, p_nulls)
        preds.append([list(choices[s]) if s in chosen else [-1, -1] for s in range(4)])
    return preds


def main():
    public_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_seed(SEEDS[0])

    # Inputs are limited to the released public train and test files.
    train = load_packets(public_dir / "train.csv", True)
    test = load_packets(public_dir / "test.csv", False)
    log(f"device {DEVICE}; train packets {len(train)}, test packets {len(test)}")

    groups = sorted({p["group"] for p in train})
    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(groups)
    n_hold = int(round(HOLDOUT_FRAC * len(groups)))
    hold_sets = [set(groups[:n_hold]), set(groups[n_hold:2 * n_hold])]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    pad_id = tokenizer.pad_token_id
    all_train_feats = build_features(tokenizer, train, True)
    test_feats = build_features(tokenizer, test, False)
    log(f"train features {len(all_train_feats)}, test features {len(test_feats)}")

    deadline = START + TIME_LIMIT - SAFETY
    plan_state = {"n_models": len(SEEDS), "epochs": EPOCH_OPTIONS[0], "sec_per_step": None, "infer": 0.0}

    def planner(sec_per_step, steps_per_epoch):
        remaining = deadline - time.time()
        infer = (len(test_feats) + len(all_train_feats) * HOLDOUT_FRAC) / BATCH * sec_per_step / 3
        second = lambda ep: ep * steps_per_epoch * sec_per_step + infer + 120
        chosen = None
        for n_models in (len(SEEDS), 1):
            for ep in EPOCH_OPTIONS:
                cost = (ep * steps_per_epoch - 30) * sec_per_step + infer + (n_models - 1) * second(ep)
                if cost < remaining:
                    chosen = (n_models, ep)
                    break
            if chosen:
                break
        if chosen is None:
            affordable = int((remaining - infer) / sec_per_step) + 30
            chosen = (1, max(31, affordable) / steps_per_epoch)
        plan_state.update(n_models=chosen[0], epochs=chosen[1], sec_per_step=sec_per_step, infer=infer)
        current["epochs"] = chosen[1]
        current["total"] = max(31, int(chosen[1] * steps_per_epoch))
        current["hard_stop"] = deadline - infer - (chosen[0] - 1) * second(chosen[1])
        log(f"{sec_per_step:.3f}s/step -> {chosen[0]} model(s), {chosen[1]} epoch(s)")

    model_test_pools, hold_records = [], []
    for m, seed in enumerate(SEEDS):
        if m >= plan_state["n_models"]:
            break
        hold = hold_sets[m]
        fit_feats = [f for f in all_train_feats if train[f["packet"]]["group"] not in hold]
        hold_pidx = [i for i, p in enumerate(train) if p["group"] in hold]
        hold_feats = [f for f in all_train_feats if train[f["packet"]]["group"] in hold]
        if m == 0:
            current = {"epochs": plan_state["epochs"], "hard_stop": deadline, "on_probe": planner}
        else:
            need = plan_state["epochs"] * math.ceil(len(fit_feats) / BATCH) * plan_state["sec_per_step"]
            if time.time() + 0.8 * (need + plan_state["infer"]) > deadline:
                log(f"skipping model {m}: not enough time left")
                break
            current = {"epochs": plan_state["epochs"], "hard_stop": deadline - plan_state["infer"],
                       "on_probe": None}
        model = train_model(seed, fit_feats, pad_id, current)
        hold_pools = question_pools(hold_feats, predict_logits(model, hold_feats, pad_id), train)
        test_pools = question_pools(test_feats, predict_logits(model, test_feats, pad_id), test)
        hold_records.append((hold_pidx, hold_pools))
        model_test_pools.append(test_pools)
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        log(f"model {m} done")

    hold_pidx = [pi for pidx, _ in hold_records for pi in pidx]
    hold_pools = {k: v for _, pools in hold_records for k, v in pools.items()}
    x_hold = null_features(train, hold_pools, hold_pidx)
    y_hold = np.array([int(train[pi]["spans"][s][0] < 0) for pi in hold_pidx for s in range(4)])
    q_groups = np.array([train[pi]["group"] for pi in hold_pidx for _ in range(4)])

    oof_logit = np.zeros(len(y_hold))
    for tr_i, va_i in GroupKFold(n_splits=5).split(x_hold, y_hold, q_groups):
        oof_logit[va_i] = fit_null_model(x_hold[tr_i], y_hold[tr_i])(x_hold[va_i])
    keys = [(pi, s) for pi in hold_pidx for s in range(4)]
    gold = [train[pi]["spans"] for pi in hold_pidx]
    passages = [train[pi]["passage"] for pi in hold_pidx]
    pgroups = [train[pi]["group"] for pi in hold_pidx]
    cache = TokenCache()

    naive = []
    for pi in hold_pidx:
        row = []
        for s in range(4):
            pool = hold_pools[(pi, s)]
            row.append(list(pool["spans"][0]) if pool["spans"] and pool["margin"] < 0 else [-1, -1])
        naive.append(row)
    log(f"holdout naive null-threshold score {official_score(naive, gold, passages, pgroups):.4f}")

    top1 = [token_f1(norm_tokens(train[pi]["passage"][slice(*hold_pools[(pi, s)]["spans"][0])]),
                     norm_tokens(train[pi]["passage"][slice(*train[pi]["spans"][s])]))
            for pi in hold_pidx for s in range(4)
            if train[pi]["spans"][s][0] >= 0 and hold_pools[(pi, s)]["spans"]]
    log(f"holdout top-1 token F1 on answerable questions {np.mean(top1):.4f}")

    abstain_score = official_score([[[-1, -1]] * 4] * len(hold_pidx), gold, passages, pgroups)
    best = (abstain_score, None, None)
    for temp in T_GRID:
        for shift in D_GRID:
            nl = {k: v + shift for k, v in zip(keys, oof_logit)}
            preds = decide(train, hold_pidx, [(hold_pools, nl)], temp, cache)
            score = official_score(preds, gold, passages, pgroups)
            if score > best[0] + 1e-9:
                best = (score, temp, shift)
    log(f"holdout all-abstain {abstain_score:.4f}; selected decision {best[0]:.4f} "
        f"with temperature {best[1]} shift {best[2]}")

    test_pidx = list(range(len(test)))
    if best[1] is None:
        preds = [[[-1, -1]] * 4 for _ in test_pidx]
    else:
        null_model = fit_null_model(x_hold, y_hold)
        outputs = []
        for pools in model_test_pools:
            z = null_model(null_features(test, pools, test_pidx)) + best[2]
            outputs.append((pools, {(pi, s): z[4 * n + s] for n, pi in enumerate(test_pidx) for s in range(4)}))
        preds = decide(test, test_pidx, outputs, best[1], cache)

    sub = pd.DataFrame({"id": [p["id"] for p in test],
                        "spans": [json.dumps(p, separators=(",", ":")) for p in preds]})
    assert len(sub) == len(test) and sub["id"].is_unique
    for pk, pr in zip(test, preds):
        assert len(pr) == 4
        for s, e in pr:
            assert (s, e) == (-1, -1) or 0 <= s < e <= len(pk["passage"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    answered = sum(s >= 0 for pr in preds for s, _ in pr) / (4 * len(preds))
    log(f"wrote {out_path} ({len(sub)} rows, answered fraction {answered:.3f})")


if __name__ == "__main__":
    main()
