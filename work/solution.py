"""Crash event-program reconstruction.

Pipeline
  1. Card model   : LightGBM scores each of the ten cards for membership, using card-vs-board
                    consistency features (impact points, most/first harmful event, fire, rollover,
                    roadway departure, within-bank duplication counts).
  2. Order models : LightGBM pairwise-precedence and immediate-succession (with START/END) models
                    trained on gold programs; applied to every card pair of a case.
  3. Set ranker   : every card subset of the required size is enumerated and ranked by a
                    LambdaRank model on set-level coverage features, card scores, and the best
                    achievable ordering score of the subset.
  4. Decoding     : the top-ranked subset is ordered by exhaustive permutation search.

Usage: python solution.py <public_data_dir> <submission_csv_path>
"""
import itertools
import json
import os
import re
import sys
from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SEED = 0
N_FOLDS = 5
VFIELDS = ["body", "configuration", "precrash_movement", "critical_event", "avoidance", "stability",
           "location_after_critical_event", "trafficway", "alignment", "profile", "surface",
           "initial_impact", "damage", "rollover", "rollover_location", "most_harmful_event",
           "tow_status", "fire", "accident_type"]
PFIELDS = ["initial_impact", "most_harmful_event", "critical_event", "precrash_movement", "rollover", "fire",
           "location_after_critical_event", "accident_type"]
SFIELDS = ["collision_configuration", "first_harmful_event", "road_system", "road_relation",
           "junction_relation", "intersection_type", "urbanicity", "light", "weather"]
FIRE = "Fire/Explosion"
ROLL = "Rollover/Overturn"
MV = "Motor Vehicle In-Transport"
RAN = ("Ran Off Roadway - Right", "Ran Off Roadway - Left", "Ran off Roadway - Direction Unknown")


# ----------------------------------------------------------------------------- parsing
def parse(df):
    cases = []
    for r in df.itertuples(index=False):
        cases.append(dict(case_id=r.case_id, scene=json.loads(r.scene_context),
                          board={v["vehicle"]: v for v in json.loads(r.vehicle_board)},
                          bank=json.loads(r.event_bank), L=int(r.program_length)))
    return cases


def clock(s):
    m = re.match(r"(\d+) Clock Point", s or "")
    return int(m.group(1)) % 12 if m else None


def side(s):
    c = clock(s)
    if c is not None:
        if c in (11, 0, 1):
            return "F"
        if c in (5, 6, 7):
            return "B"
        return "L" if c in (8, 9, 10) else "R"
    s = s or ""
    if s.startswith("Left"):
        return "L"
    if s.startswith("Right"):
        return "R"
    return s


def clockdist(a, b):
    ca, cb = clock(a), clock(b)
    if ca is None or cb is None:
        return -1
    return min((ca - cb) % 12, (cb - ca) % 12)


def speed(s):
    m = re.match(r"(\d+) MPH", s or "")
    return float(m.group(1)) if m else np.nan


def vehicles_of(e):
    return set([e["actor"]] + ([e["partner"]] if e["partner_kind"] == "vehicle" else []))


def explained(e, bd):
    """Board vehicles whose recorded initial impact point this card reproduces."""
    s = set()
    if e["actor"] in bd and e["actor_impact"] == bd[e["actor"]]["initial_impact"]:
        s.add(e["actor"])
    if e["partner_kind"] == "vehicle" and e["partner"] in bd and e["partner_impact"] == bd[e["partner"]]["initial_impact"]:
        s.add(e["partner"])
    return s


# ----------------------------------------------------------------------------- card features
def card_rows(c):
    sc, bd, bank, L = c["scene"], c["board"], c["bank"], c["L"]
    fhe = sc["first_harmful_event"]
    mhes = [v["most_harmful_event"] for v in bd.values()]
    cnt_type = Counter(e["event_type"] for e in bank)
    cnt_atype = Counter((e["actor"], e["event_type"]) for e in bank)
    cnt_pair = Counter((frozenset([e["actor"], e["partner"]]), e["event_type"]) for e in bank)
    cnt_actor = Counter(e["actor"] for e in bank)
    cnt_kind = Counter(e["partner_kind"] for e in bank)
    expl = Counter()
    for e in bank:
        for v in explained(e, bd):
            expl[v] += 1
    exact = Counter((e["actor"], e["partner"], e["event_type"], e["actor_impact"], e["partner_impact"]) for e in bank)
    rows = []
    for e in bank:
        a = bd.get(e["actor"], {})
        p = bd.get(e["partner"], {}) if e["partner_kind"] == "vehicle" else {}
        r = {"event_type": e["event_type"], "partner_kind": e["partner_kind"],
             "actor_impact": e["actor_impact"], "partner_impact": e["partner_impact"],
             "ai_side": side(e["actor_impact"]), "pi_side": side(e["partner_impact"]),
             "actor_i": int(e["actor"][1:]),
             "partner_i": int(e["partner"][1:]) if e["partner_kind"] == "vehicle" else 0,
             "nv": len(bd), "L": L}
        r["actor_lt_partner"] = float(r["partner_i"] > r["actor_i"]) if r["partner_i"] else -1
        for f in VFIELDS:
            r["a_" + f] = a.get(f, "NA")
        for f in PFIELDS:
            r["p_" + f] = p.get(f, "NA")
        for f in SFIELDS:
            r["s_" + f] = sc.get(f, "NA")
        r["a_acc_letter"] = a.get("accident_type", "NA")[:1]
        r["a_speed"] = speed(a.get("travel_speed"))
        r["a_limit"] = speed(a.get("speed_limit"))
        r["a_over"] = r["a_speed"] - r["a_limit"]
        r["ai_match"] = float(e["actor_impact"] == a.get("initial_impact"))
        r["ai_side_match"] = float(side(e["actor_impact"]) == side(a.get("initial_impact")))
        r["ai_clockdist"] = clockdist(e["actor_impact"], a.get("initial_impact"))
        if p:
            r["pi_match"] = float(e["partner_impact"] == p.get("initial_impact"))
            r["pi_side_match"] = float(side(e["partner_impact"]) == side(p.get("initial_impact")))
            r["pi_clockdist"] = clockdist(e["partner_impact"], p.get("initial_impact"))
        else:
            r["pi_match"] = r["pi_side_match"] = r["pi_clockdist"] = -1
        r["et_fhe"] = float(e["event_type"] == fhe)
        r["et_amhe"] = float(e["event_type"] == a.get("most_harmful_event"))
        r["et_pmhe"] = float(e["event_type"] == p.get("most_harmful_event")) if p else -1
        r["et_anymhe"] = float(e["event_type"] in mhes)
        r["fire_ok"] = float(a.get("fire") == "Yes") if e["event_type"] == FIRE else -1
        r["roll_ok"] = float(a.get("rollover") == "Rollover") if e["event_type"] == ROLL else -1
        r["n_type"] = cnt_type[e["event_type"]]
        r["n_atype"] = cnt_atype[(e["actor"], e["event_type"])]
        r["n_pair"] = cnt_pair[(frozenset([e["actor"], e["partner"]]), e["event_type"])]
        r["n_actor"] = cnt_actor[e["actor"]]
        r["n_kind"] = cnt_kind[e["partner_kind"]]
        r["mirror"] = float(exact[(e["partner"], e["actor"], e["event_type"], e["partner_impact"],
                                   e["actor_impact"])] > 0) if p else -1
        r["a_expl"] = expl[e["actor"]]
        r["p_expl"] = expl[e["partner"]] if p else -1
        r["a_uniq_expl"] = float(r["ai_match"] == 1 and expl[e["actor"]] == 1)
        r["p_uniq_expl"] = float(r["pi_match"] == 1 and expl[e["partner"]] == 1) if p else -1
        r["n_fhe_cards"] = cnt_type[fhe]
        rows.append(r)
    return rows


def build_card_df(cases):
    rows, ci, cj = [], [], []
    for i, c in enumerate(cases):
        for j, r in enumerate(card_rows(c)):
            rows.append(r)
            ci.append(i)
            cj.append(j)
    df = pd.DataFrame(rows)
    df["_case"] = ci
    df["_j"] = cj
    return df


def fit_cats(df):
    return {c: pd.Index(sorted(df[c].astype(str).unique())) for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])}


def apply_cats(df, cats):
    out = df.copy()
    for c, idx in cats.items():
        out[c] = pd.Categorical(out[c].astype(str), categories=idx)
    return out


# ----------------------------------------------------------------------------- order features
OCOLS = ["event_type", "partner_kind", "actor_impact", "partner_impact", "ai_side", "actor_i", "partner_i",
         "et_fhe", "et_amhe", "et_pmhe", "ai_match", "pi_match", "fire_ok", "roll_ok",
         "a_location_after_critical_event", "a_critical_event", "a_precrash_movement", "a_rollover_location",
         "a_most_harmful_event", "a_initial_impact", "a_acc_letter", "a_avoidance", "a_stability"]
SHARED = ["s_collision_configuration", "s_first_harmful_event", "s_junction_relation", "s_road_relation", "nv", "L"]
REL = ["same_actor", "ip_ja", "jp_ia", "share_v", "same_et", "same_vset"]


def rel(bank, i, j):
    if i < 0 or j < 0:
        return [-1] * len(REL)
    a, b = bank[i], bank[j]
    va, vb = vehicles_of(a), vehicles_of(b)
    return [float(a["actor"] == b["actor"]), float(a["partner"] == b["actor"]), float(b["partner"] == a["actor"]),
            float(len(va & vb) > 0), float(a["event_type"] == b["event_type"]), float(va == vb)]


def pair_frame(cdf, cases, items):
    """items: (case, i, j); i=-1 is START, j=-1 is END."""
    ci = cdf["_case"].values * 10 + cdf["_j"].values
    pos = np.empty(ci.max() + 1, int)
    pos[ci] = np.arange(len(cdf))
    it = np.array(items)
    ri = np.where(it[:, 1] >= 0, pos[it[:, 0] * 10 + np.maximum(it[:, 1], 0)], -1)
    rj = np.where(it[:, 2] >= 0, pos[it[:, 0] * 10 + np.maximum(it[:, 2], 0)], -1)
    rs = pos[it[:, 0] * 10]

    def take(rows, pre):
        t = cdf[OCOLS].iloc[np.maximum(rows, 0)].reset_index(drop=True)
        miss = rows < 0
        for col in OCOLS:
            if isinstance(t[col].dtype, pd.CategoricalDtype):
                t[col] = t[col].where(~miss)
            else:
                t[col] = np.where(miss, -9, t[col])
        t.columns = [pre + x for x in OCOLS]
        return t

    out = pd.concat([take(ri, "i_"), take(rj, "j_"), cdf[SHARED].iloc[rs].reset_index(drop=True),
                     pd.DataFrame([rel(cases[c]["bank"], i, j) for c, i, j in items], columns=REL)], axis=1)
    out["i_start"] = (it[:, 1] < 0).astype(float)
    out["j_end"] = (it[:, 2] < 0).astype(float)
    return out


def gold_order_items(gold, idxs):
    pw, pwy, ad, ady = [], [], [], []
    for c in idxs:
        g = gold[c]
        for a in range(len(g)):
            for b in range(len(g)):
                if a != b:
                    pw.append((c, g[a], g[b]))
                    pwy.append(int(a < b))
        ext = [-1] + g + [-1]
        for a in range(len(ext) - 1):
            for b in range(1, len(ext)):
                if a != b and not (a == 0 and b == len(ext) - 1):
                    ad.append((c, ext[a], ext[b]))
                    ady.append(int(b == a + 1))
    return pw, np.array(pwy), ad, np.array(ady)


def all_order_items(idxs):
    pw, ad = [], []
    for c in idxs:
        for i in range(10):
            for j in range(10):
                if i != j:
                    pw.append((c, i, j))
                    ad.append((c, i, j))
            ad.append((c, -1, i))
            ad.append((c, i, -1))
    return pw, ad


def order_tables(pw, ppw, ad, pad, n_cases_map):
    """Log-probability tables per case: LP[c][i,j] (i before j), LA[c][i+1,j+1] (j follows i; index 0=START/END)."""
    LP = {c: np.zeros((10, 10)) for c in n_cases_map}
    LA = {c: np.zeros((11, 11)) for c in n_cases_map}
    lp = np.log(np.clip(ppw, 1e-4, 1 - 1e-4))
    la = np.log(np.clip(pad, 1e-4, 1 - 1e-4))
    for (c, i, j), v in zip(pw, lp):
        LP[c][i, j] = v
    for (c, i, j), v in zip(ad, la):
        LA[c][i + 1, j + 1] = v
    return LP, LA


PERMS = {3: list(itertools.permutations(range(3))), 4: list(itertools.permutations(range(4)))}
W_ADJ = 2.0


def best_order(S, LP, LA):
    best, bs = None, -1e18
    for pm in PERMS[len(S)]:
        seq = [S[k] for k in pm]
        s = 0.0
        for a in range(len(seq)):
            for b in range(a + 1, len(seq)):
                s += LP[seq[a], seq[b]]
        ext = [-1] + seq + [-1]
        s += W_ADJ * sum(LA[ext[t] + 1, ext[t + 1] + 1] for t in range(len(ext) - 1))
        if s > bs:
            bs, best = s, seq
    return best, bs


# ----------------------------------------------------------------------------- set features
SET_NAMES = ["sumlog", "pmean", "pmin", "pmax", "p2", "ranksum", "ntopL", "k_veh", "k_nonh", "k_obj",
             "n_et", "max_et", "n_fhe", "n_mv", "n_fire", "n_roll", "n_ran", "max_actor_et",
             "n_inv", "n_notinv", "n_act", "n_expl", "n_notexpl", "n_inv_notexpl", "n_mhe", "n_notmhe",
             "fire_cov", "fire_miss", "fire_bad", "roll_cov", "roll_miss", "roll_bad",
             "max_ran_v", "ran_dep", "ran_nodep", "n_objv", "objv_noran", "max_pair", "n_pairs", "n_mv_noexpl",
             "ord_best", "ord_margin", "nv", "L"]


def set_features(c, p, LP, LA):
    bank, bd, L = c["bank"], c["board"], c["L"]
    fhe = c["scene"]["first_harmful_event"]
    lg = np.log(np.clip(p, 1e-4, 1 - 1e-4))
    l1 = np.log(np.clip(1 - p, 1e-4, 1))
    tot = l1.sum()
    rank = np.argsort(np.argsort(-p))
    vs = list(bd)
    et = [e["event_type"] for e in bank]
    kinds = [e["partner_kind"] for e in bank]
    inv = [vehicles_of(e) for e in bank]
    expl = [explained(e, bd) for e in bank]
    mheh = [set(v for v in inv[j] if v in bd and bd[v]["most_harmful_event"] == et[j]) for j in range(10)]
    fire_v = {v for v in vs if bd[v]["fire"] == "Yes"}
    roll_v = {v for v in vs if bd[v]["rollover"] == "Rollover"}
    dep_v = {v for v in vs if bd[v]["location_after_critical_event"] in ("Departed roadway", "Returned to roadway")}
    sets, F, orders = [], [], []
    for S in itertools.combinations(range(10), L):
        S = list(S)
        ps = p[S]
        f = [lg[S].sum() + tot - l1[S].sum(), ps.mean(), ps.min(), ps.max(), np.sort(ps)[1], rank[S].sum(),
             (rank[S] < L).sum()]
        ks = Counter(kinds[j] for j in S)
        f += [ks["vehicle"], ks["non_harmful"], ks["object_or_person"]]
        ce = Counter(et[j] for j in S)
        f += [len(ce), max(ce.values()), ce[fhe], ce[MV], ce[FIRE], ce[ROLL], sum(ce[r] for r in RAN)]
        f += [max(Counter((bank[j]["actor"], et[j]) for j in S).values())]
        invd = set().union(*[inv[j] for j in S])
        acts = set(bank[j]["actor"] for j in S)
        ex = set().union(*[expl[j] for j in S])
        mh = set().union(*[mheh[j] for j in S])
        f += [len(invd), len(vs) - len(invd), len(acts), len(ex), len(vs) - len(ex), len(invd - ex), len(mh),
              len(vs) - len(mh)]
        fc = set(bank[j]["actor"] for j in S if et[j] == FIRE)
        rc = set(bank[j]["actor"] for j in S if et[j] == ROLL)
        ranv = Counter(bank[j]["actor"] for j in S if et[j] in RAN)
        f += [len(fire_v & fc), len(fire_v - fc), len(fc - fire_v), len(roll_v & rc), len(roll_v - rc),
              len(rc - roll_v)]
        f += [max(ranv.values()) if ranv else 0, len(dep_v & set(ranv)), len(set(ranv) - dep_v)]
        objv = set(bank[j]["actor"] for j in S if kinds[j] == "object_or_person" and et[j] not in (FIRE, ROLL))
        f += [len(objv), len(objv - set(ranv))]
        pairs = Counter(frozenset(inv[j]) for j in S if kinds[j] == "vehicle")
        f += [max(pairs.values()) if pairs else 0, len(pairs)]
        f += [sum(1 for j in S if kinds[j] == "vehicle" and len(expl[j]) == 0)]
        seq, osc = best_order(S, LP, LA)
        f += [osc, 0.0, len(bd), L]
        sets.append(S)
        orders.append(seq)
        F.append(f)
    F = np.array(F, dtype=np.float32)
    F[:, SET_NAMES.index("ord_margin")] = F[:, SET_NAMES.index("ord_best")] - F[:, SET_NAMES.index("ord_best")].max()
    return sets, orders, F


# ----------------------------------------------------------------------------- models
P_CARD = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=20, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=10, cat_l2=10, verbose=-1,
              seed=SEED, num_threads=4, deterministic=True, force_row_wise=True)
N_CARD = 1000
P_ORD = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=20, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, cat_smooth=10, verbose=-1, seed=SEED, num_threads=4,
             deterministic=True, force_row_wise=True)
N_ORD = 500
P_SET = dict(objective="lambdarank", learning_rate=0.05, num_leaves=31, min_child_samples=50, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, seed=SEED,
             num_threads=4, deterministic=True, force_row_wise=True)
N_SET = 400


def card_X(cdf):
    return cdf.drop(columns=["_case", "_j"])


def fit_card(Xc, y):
    return lgb.train(P_CARD, lgb.Dataset(Xc, y), N_CARD)


def fit_order(cdf, cases, gold, idxs):
    pw, pwy, ad, ady = gold_order_items(gold, idxs)
    mp = lgb.train(P_ORD, lgb.Dataset(pair_frame(cdf, cases, pw), pwy), N_ORD)
    ma = lgb.train(P_ORD, lgb.Dataset(pair_frame(cdf, cases, ad), ady), N_ORD)
    return mp, ma


def predict_order(models, cdf, cases, idxs):
    mp, ma = models
    pw, ad = all_order_items(idxs)
    return order_tables(pw, mp.predict(pair_frame(cdf, cases, pw)), ad, ma.predict(pair_frame(cdf, cases, ad)), idxs)


def build_sets(cases, idxs, p10, LP, LA):
    out = {}
    for c in idxs:
        out[c] = set_features(cases[c], p10[c], LP[c], LA[c])
    return out


def set_xy(SF, idxs, gold):
    X, y, g = [], [], []
    for c in idxs:
        sets, _, F = SF[c]
        gs = set(gold[c])
        X.append(F)
        y.append(np.array([set(s) == gs for s in sets], int))
        g.append(len(sets))
    return np.vstack(X), np.concatenate(y), g


def oof_stage1(cases, gold, cdf, Xc, y, idxs, seed_fold=SEED):
    """Out-of-fold card probabilities and order tables for the given training cases."""
    p10 = np.zeros((len(cases), 10))
    LP, LA = {}, {}
    idxs = np.asarray(idxs)
    card_case = cdf["_case"].values
    kf = KFold(N_FOLDS, shuffle=True, random_state=seed_fold)
    for tr_i, va_i in kf.split(idxs):
        tr_c, va_c = idxs[tr_i], idxs[va_i]
        mtr = np.isin(card_case, tr_c)
        mva = np.isin(card_case, va_c)
        m = fit_card(Xc[mtr], y[mtr])
        p10[card_case[mva], cdf["_j"].values[mva]] = m.predict(Xc[mva])
        om = fit_order(cdf, cases, gold, list(tr_c))
        lp, la = predict_order(om, cdf, cases, list(va_c))
        LP.update(lp)
        LA.update(la)
    return p10, LP, LA


def decode(SF, idxs, set_model):
    preds = {}
    for c in idxs:
        sets, orders, F = SF[c]
        s = set_model.predict(F)
        preds[c] = orders[int(np.argmax(s))]
    return preds


# ----------------------------------------------------------------------------- main
def main(data_dir, out_path):
    # Inputs are limited to the released public train, label, and test files.
    tr = pd.read_csv(os.path.join(data_dir, "train.csv"), dtype={"case_id": str})
    lab = pd.read_csv(os.path.join(data_dir, "train_labels.csv"), dtype={"case_id": str})
    te = pd.read_csv(os.path.join(data_dir, "test.csv"), dtype={"case_id": str})
    tr = tr.merge(lab, on="case_id", how="inner")
    cases = parse(tr) + parse(te)
    ntr = len(tr)
    gold = [[int(x[1:]) - 1 for x in json.loads(g)] for g in tr["event_program"]] + [None] * len(te)
    tr_idx, te_idx = list(range(ntr)), list(range(ntr, len(cases)))

    cdf_raw = build_card_df(cases)
    cats = fit_cats(cdf_raw[cdf_raw["_case"] < ntr])
    cdf = apply_cats(cdf_raw, cats)
    Xc = card_X(cdf)
    y = np.array([int(gold[c] is not None and j in gold[c]) for c, j in zip(cdf["_case"], cdf["_j"])])
    is_tr = cdf["_case"].values < ntr

    # stage 1 out-of-fold on train (to train the set ranker), full fit for test
    p10, LP, LA = oof_stage1(cases, gold, cdf, Xc, y, tr_idx)
    mc = fit_card(Xc[is_tr], y[is_tr])
    p10[cdf["_case"].values[~is_tr], cdf["_j"].values[~is_tr]] = mc.predict(Xc[~is_tr])
    om = fit_order(cdf, cases, gold, tr_idx)
    lp, la = predict_order(om, cdf, cases, te_idx)
    LP.update(lp)
    LA.update(la)

    SF = build_sets(cases, tr_idx + te_idx, p10, LP, LA)
    Xs, ys, gs = set_xy(SF, tr_idx, gold)
    ms = lgb.train(P_SET, lgb.Dataset(Xs, ys, group=gs), N_SET)
    preds = decode(SF, te_idx, ms)

    rows = []
    for c in te_idx:
        seq = ["e%02d" % (j + 1) for j in preds[c]]
        assert len(seq) == cases[c]["L"] and len(set(seq)) == len(seq)
        rows.append((cases[c]["case_id"], json.dumps(seq, separators=(",", ":"))))
    sub = pd.DataFrame(rows, columns=["case_id", "event_program"])
    d = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(d, exist_ok=True)
    sub.to_csv(out_path, index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
