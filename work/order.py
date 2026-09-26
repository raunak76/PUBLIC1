import numpy as np, pandas as pd

OCOLS = ["event_type", "partner_kind", "actor_impact", "partner_impact", "ai_side", "actor_i", "partner_i",
         "et_fhe", "et_amhe", "et_pmhe", "ai_match", "pi_match", "fire_ok", "roll_ok",
         "a_location_after_critical_event", "a_critical_event", "a_precrash_movement", "a_rollover_location",
         "a_most_harmful_event", "a_initial_impact", "a_acc_letter", "a_avoidance", "a_stability"]
SHARED = ["s_collision_configuration", "s_first_harmful_event", "s_junction_relation", "s_road_relation", "nv", "L"]


def _veh(e):
    return set([e["actor"]] + ([e["partner"]] if e["partner_kind"] == "vehicle" else []))


def rel(bank, i, j):
    """relational features between card i and card j (-1 = START/END)"""
    if i < 0 or j < 0:
        return [-1] * 6
    a, b = bank[i], bank[j]
    va, vb = _veh(a), _veh(b)
    return [float(a["actor"] == b["actor"]), float(a["partner"] == b["actor"]), float(b["partner"] == a["actor"]),
            float(len(va & vb) > 0), float(a["event_type"] == b["event_type"]), float(va == vb)]


def pair_frame(cdf, cases, items):
    """items: list of (case_idx, i, j); i/j = -1 means START (as i) or END (as j)."""
    base = cdf.set_index(["_case", "_j"])
    A = cdf[OCOLS].iloc[:0]
    rows_i, rows_j, sh, rl = [], [], [], []
    idx = {(c, j): k for k, (c, j) in enumerate(zip(cdf["_case"].values, cdf["_j"].values))}
    for c, i, j in items:
        rows_i.append(idx[(c, i)] if i >= 0 else -1)
        rows_j.append(idx[(c, j)] if j >= 0 else -1)
        sh.append(idx[(c, 0)])
        rl.append(rel(cases[c]["bank"], i, j))
    def take(rows, pre):
        rows = np.array(rows)
        t = cdf[OCOLS].iloc[np.where(rows < 0, 0, rows)].reset_index(drop=True).copy()
        for col in OCOLS:
            if isinstance(t[col].dtype, pd.CategoricalDtype):
                t.loc[rows < 0, col] = np.nan
            else:
                t.loc[rows < 0, col] = -9
        t.columns = [pre + x for x in OCOLS]
        return t
    out = pd.concat([take(rows_i, "i_"), take(rows_j, "j_"),
                     cdf[SHARED].iloc[sh].reset_index(drop=True),
                     pd.DataFrame(rl, columns=["same_actor", "ip_ja", "jp_ia", "share_v", "same_et", "same_vset"])], axis=1)
    out["i_start"] = [float(i < 0) for _, i, _ in items]
    out["j_end"] = [float(j < 0) for _, _, j in items]
    return out
