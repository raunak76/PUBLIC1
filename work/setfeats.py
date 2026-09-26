import itertools
import numpy as np
from collections import Counter

NONH = "non_harmful"
RAN = ("Ran Off Roadway - Right", "Ran Off Roadway - Left", "Ran off Roadway - Direction Unknown")
FIRE = "Fire/Explosion"
ROLL = "Rollover/Overturn"
MV = "Motor Vehicle In-Transport"


def set_features(c, p):
    """Enumerate every candidate card set of size L; return (list_of_sets, feature matrix)."""
    bank, bd, L = c["bank"], c["board"], c["L"]
    fhe = c["scene"]["first_harmful_event"]
    lg = np.log(np.clip(p, 1e-4, 1 - 1e-4)); l1 = np.log(np.clip(1 - p, 1e-4, 1))
    rank = np.argsort(np.argsort(-p))
    vs = list(bd)
    et = [e["event_type"] for e in bank]
    kinds = [e["partner_kind"] for e in bank]
    inv = [set([e["actor"]] + ([e["partner"]] if e["partner_kind"] == "vehicle" else [])) for e in bank]
    expl = []
    for e in bank:
        s = set()
        if e["actor_impact"] == bd[e["actor"]]["initial_impact"]:
            s.add(e["actor"])
        if e["partner_kind"] == "vehicle" and e["partner"] in bd and e["partner_impact"] == bd[e["partner"]]["initial_impact"]:
            s.add(e["partner"])
        expl.append(s)
    # mhe hit: card event equals most harmful event of an involved vehicle
    mheh = [set(v for v in inv[j] if bd[v]["most_harmful_event"] == et[j]) for j in range(10)]
    fire_v = {v for v in vs if bd[v]["fire"] == "Yes"}
    roll_v = {v for v in vs if bd[v]["rollover"] == "Rollover"}
    dep_v = {v for v in vs if bd[v]["location_after_critical_event"] in ("Departed roadway", "Returned to roadway")}
    tot = lg.sum() if False else l1.sum()
    sets, F = [], []
    for S in itertools.combinations(range(10), L):
        S = list(S)
        f = []
        ps = p[S]
        f += [lg[S].sum() + tot - l1[S].sum(), ps.mean(), ps.min(), ps.max(), np.sort(ps)[1], rank[S].sum(), (rank[S] < L).sum()]
        ets = [et[j] for j in S]
        ks = Counter(kinds[j] for j in S)
        f += [ks["vehicle"], ks[NONH], ks["object_or_person"]]
        ce = Counter(ets)
        f += [len(ce), max(ce.values()), ce[fhe], ce[MV], ce[FIRE], ce[ROLL], sum(ce[r] for r in RAN)]
        cat = Counter((bank[j]["actor"], et[j]) for j in S)
        f += [max(cat.values())]
        invd = set().union(*[inv[j] for j in S])
        acts = set(bank[j]["actor"] for j in S)
        ex = set().union(*[expl[j] for j in S])
        mh = set().union(*[mheh[j] for j in S])
        f += [len(invd), len(vs) - len(invd), len(acts), len(ex), len(vs) - len(ex), len(invd - ex), len(mh), len(vs) - len(mh)]
        fc = set(bank[j]["actor"] for j in S if et[j] == FIRE)
        rc = set(bank[j]["actor"] for j in S if et[j] == ROLL)
        ranv = Counter(bank[j]["actor"] for j in S if et[j] in RAN)
        f += [len(fire_v & fc), len(fire_v - fc), len(fc - fire_v), len(roll_v & rc), len(roll_v - rc), len(rc - roll_v)]
        f += [max(ranv.values()) if ranv else 0, len(dep_v & set(ranv)), len(set(ranv) - dep_v)]
        objv = set(bank[j]["actor"] for j in S if kinds[j] == "object_or_person" and et[j] not in (FIRE, ROLL))
        f += [len(objv), len(objv - set(ranv))]
        # mirrored duplicate MV cards within set
        pairs = Counter(frozenset(inv[j]) for j in S if kinds[j] == "vehicle")
        f += [max(pairs.values()) if pairs else 0, len(pairs)]
        nexpl_cards = sum(1 for j in S if kinds[j] == "vehicle" and len(expl[j]) == 0)
        f += [nexpl_cards]
        sets.append(S)
        F.append(f)
    return sets, np.array(F, dtype=np.float32)


SET_NAMES = ["sumlog", "pmean", "pmin", "pmax", "p2", "ranksum", "ntopL", "k_veh", "k_nonh", "k_obj",
             "n_et", "max_et", "n_fhe", "n_mv", "n_fire", "n_roll", "n_ran", "max_actor_et",
             "n_inv", "n_notinv", "n_act", "n_expl", "n_notexpl", "n_inv_notexpl", "n_mhe", "n_notmhe",
             "fire_cov", "fire_miss", "fire_bad", "roll_cov", "roll_miss", "roll_bad",
             "max_ran_v", "ran_dep", "ran_nodep", "n_objv", "objv_noran", "max_pair", "n_pairs", "n_mv_noexpl"]
