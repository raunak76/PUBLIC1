import json, re
import numpy as np
import pandas as pd

VFIELDS = ["body", "configuration", "precrash_movement", "critical_event", "avoidance", "stability",
           "location_after_critical_event", "trafficway", "alignment", "profile", "surface",
           "initial_impact", "damage", "rollover", "rollover_location", "most_harmful_event",
           "tow_status", "fire", "accident_type"]
SFIELDS = ["collision_configuration", "first_harmful_event", "road_system", "road_relation",
           "junction_relation", "intersection_type", "urbanicity", "light", "weather"]


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


def speed(s):
    m = re.match(r"(\d+) MPH", s or "")
    return float(m.group(1)) if m else np.nan


def parse(df):
    cases = []
    for r in df.itertuples(index=False):
        cases.append(dict(case_id=r.case_id, scene=json.loads(r.scene_context),
                          board={v["vehicle"]: v for v in json.loads(r.vehicle_board)},
                          bank=json.loads(r.event_bank), L=int(r.program_length)))
    return cases


class Vocab:
    def __init__(self):
        self.m = {}

    def fit(self, key, vals):
        d = self.m.setdefault(key, {})
        for v in vals:
            if v not in d:
                d[v] = len(d)

    def get(self, key, v):
        return self.m.get(key, {}).get(v, -1)


def vehicle_impacts(e):
    """(vehicle, impact) pairs a card asserts about board vehicles."""
    out = [(e["actor"], e["actor_impact"])]
    if e["partner_kind"] == "vehicle":
        out.append((e["partner"], e["partner_impact"]))
    return out


FIRE = "Fire/Explosion"
ROLL = "Rollover/Overturn"


def card_rows(c):
    sc, bd, bank, L = c["scene"], c["board"], c["bank"], c["L"]
    nv = len(bd)
    fhe = sc["first_harmful_event"]
    mhes = [v["most_harmful_event"] for v in bd.values()]
    # within-case counters
    from collections import Counter
    cnt_type = Counter(e["event_type"] for e in bank)
    cnt_atype = Counter((e["actor"], e["event_type"]) for e in bank)
    cnt_pair = Counter((frozenset([e["actor"], e["partner"]]), e["event_type"]) for e in bank)
    cnt_actor = Counter(e["actor"] for e in bank)
    cnt_kind = Counter(e["partner_kind"] for e in bank)
    # which cards explain each vehicle's initial impact
    expl = Counter()
    for e in bank:
        for v, imp in vehicle_impacts(e):
            if v in bd and imp == bd[v]["initial_impact"]:
                expl[v] += 1
    exact = Counter((e["actor"], e["partner"], e["event_type"], e["actor_impact"], e["partner_impact"]) for e in bank)
    rows = []
    for e in bank:
        a = bd.get(e["actor"], {})
        p = bd.get(e["partner"], {}) if e["partner_kind"] == "vehicle" else {}
        r = {}
        r["event_type"] = e["event_type"]
        r["partner_kind"] = e["partner_kind"]
        r["actor_impact"] = e["actor_impact"]
        r["partner_impact"] = e["partner_impact"]
        r["ai_side"] = side(e["actor_impact"])
        r["pi_side"] = side(e["partner_impact"])
        r["actor_i"] = int(e["actor"][1:])
        r["partner_i"] = int(e["partner"][1:]) if e["partner_kind"] == "vehicle" else 0
        r["actor_lt_partner"] = float(r["partner_i"] > r["actor_i"]) if r["partner_i"] else -1
        r["nv"] = nv
        r["L"] = L
        for f in VFIELDS:
            r["a_" + f] = a.get(f, "NA")
        for f in ["initial_impact", "most_harmful_event", "critical_event", "precrash_movement", "rollover", "fire",
                  "location_after_critical_event", "accident_type"]:
            r["p_" + f] = p.get(f, "NA")
        for f in SFIELDS:
            r["s_" + f] = sc.get(f, "NA")
        r["a_acc_letter"] = a.get("accident_type", "NA")[:1]
        r["a_speed"] = speed(a.get("travel_speed"))
        r["a_limit"] = speed(a.get("speed_limit"))
        r["a_over"] = r["a_speed"] - r["a_limit"]
        ai_init = a.get("initial_impact")
        r["ai_match"] = float(e["actor_impact"] == ai_init)
        r["ai_side_match"] = float(side(e["actor_impact"]) == side(ai_init))
        ca, cb = clock(e["actor_impact"]), clock(ai_init)
        r["ai_clockdist"] = min((ca - cb) % 12, (cb - ca) % 12) if ca is not None and cb is not None else -1
        if p:
            r["pi_match"] = float(e["partner_impact"] == p.get("initial_impact"))
            r["pi_side_match"] = float(side(e["partner_impact"]) == side(p.get("initial_impact")))
            ca, cb = clock(e["partner_impact"]), clock(p.get("initial_impact"))
            r["pi_clockdist"] = min((ca - cb) % 12, (cb - ca) % 12) if ca is not None and cb is not None else -1
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
        r["mirror"] = float(exact[(e["partner"], e["actor"], e["event_type"], e["partner_impact"], e["actor_impact"])] > 0) if p else -1
        r["a_expl"] = expl[e["actor"]]
        r["p_expl"] = expl[e["partner"]] if p else -1
        r["a_uniq_expl"] = float(r["ai_match"] == 1 and expl[e["actor"]] == 1)
        r["p_uniq_expl"] = float(r["pi_match"] == 1 and expl[e["partner"]] == 1) if p else -1
        r["n_fhe_cards"] = cnt_type[fhe]
        r["ai_nonharm"] = float(e["actor_impact"] == "Non-Harmful Event")
        rows.append(r)
    return rows


def build_card_df(cases):
    rows, keys = [], []
    for ci, c in enumerate(cases):
        rr = card_rows(c)
        for j, r in enumerate(rr):
            rows.append(r)
            keys.append((ci, j))
    df = pd.DataFrame(rows)
    df["_case"] = [k[0] for k in keys]
    df["_j"] = [k[1] for k in keys]
    return df


def to_cat(df, cats=None):
    obj = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if cats is None:
        cats = {c: pd.Index(sorted(df[c].unique())) for c in obj}
    out = df.copy()
    for c in obj:
        out[c] = pd.Categorical(out[c], categories=cats[c])
    return out, cats
