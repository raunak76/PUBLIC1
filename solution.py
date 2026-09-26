import sys, os, re
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score

DATA, OUT = sys.argv[1], sys.argv[2]
VALIDATE = os.environ.get("VALIDATE") == "1"
tr = pd.read_csv(os.path.join(DATA, "train.csv"))
te = pd.read_csv(os.path.join(DATA, "test.csv"))
CLS = ["fits", "wraps", "overflows"]
NOSPACE = {"Han", "Thai"}


def text_feats(df):
    s = df.source_text.fillna("")
    f = pd.DataFrame(index=df.index)
    f["n_chars"] = s.str.len()
    f["n_words"] = s.str.split().str.len().fillna(0)
    f["avg_wlen"] = f.n_chars / f.n_words.clip(lower=1)
    f["max_wlen"] = s.apply(lambda x: max([len(w) for w in x.split()] or [0]))
    f["n_upper"] = s.str.count(r"[A-Z]")
    f["n_digit"] = s.str.count(r"\d")
    f["n_punct"] = s.str.count(r"[^\w\s]")
    f["ellipsis"] = s.str.contains("…").astype(int)
    f["colon"] = s.str.contains(":").astype(int)
    f["q"] = s.str.contains(r"\?").astype(int)
    f["title_case"] = s.apply(lambda x: np.mean([w[:1].isupper() for w in x.split()] or [0]))
    f["url_like"] = s.str.contains(r"[/._]\w").astype(int)
    f["log_sw"] = np.log1p(df.source_width_px)
    f["log_smin"] = np.log1p(df.source_min_width_px)
    f["log_seg"] = np.log1p(df.source_widest_segment_px)
    f["seg_ratio"] = df.source_widest_segment_px / df.source_min_width_px
    f["sw_ratio"] = df.source_width_px / df.source_min_width_px
    f["box_ratio"] = df.box_width_px / df.source_min_width_px
    f["box_sw"] = df.box_width_px / df.source_width_px
    f["box_seg"] = df.box_width_px / df.source_widest_segment_px
    f["max_lines"] = df.max_lines
    f["nospace"] = df.script.isin(NOSPACE).astype(int)
    f["widget"] = df.widget.astype("category")
    return f


def locale_table(train, cold):
    """Per-locale expansion stats. Pilot = the 12 shared menus strings, available for every locale;
    full stats for warm locales, transferred to cold ones via pilot + mean (full - pilot) offset."""
    t = train.copy()
    t["lr"] = np.log(t.reserve_px / t.source_min_width_px)
    t["lw"] = np.log(t.reserve_px / t.source_width_px)
    t["ml1"] = t.max_lines == 1
    pilot_ids = set(t[t.locale.isin(cold)].string_id)
    p = t[t.string_id.isin(pilot_ids)]
    P = p.groupby("locale").agg(p_lr=("lr", "mean"), p_lw=("lw", "mean"), p_q=("lr", lambda x: x.quantile(.8)))
    full = t[~t.locale.isin(cold)]
    F = full.groupby("locale").agg(f_lr=("lr", "mean"), f_lw=("lw", "mean"), f_sd=("lr", "std"),
                                   f_q=("lr", lambda x: x.quantile(.8)))
    F["f_lr1"] = full[full.ml1].groupby("locale").lr.mean()
    F["f_lrm"] = full[~full.ml1].groupby("locale").lr.mean()
    F["f_ovf"] = full.groupby("locale").fit_class.apply(lambda x: (x == "overflows").mean())
    L = P.join(F, how="outer")
    L["script"] = t.groupby("locale").script.first()
    warm = L[L.f_lr.notna() & L.p_lr.notna()]
    for c in ["f_lr", "f_lw", "f_sd", "f_q", "f_lr1", "f_lrm", "f_ovf"]:
        base = "p_lw" if c == "f_lw" else "p_lr"
        if c in ("f_sd", "f_ovf"):
            # regress on pilot mean
            a, b = np.polyfit(warm.p_lr, warm[c], 1)
            L.loc[L[c].isna(), c] = a * L.p_lr + b
        else:
            off = (warm[c] - warm[base]).mean()
            L.loc[L[c].isna(), c] = L[base] + off
    return L.drop(columns="script")


def build(df, L):
    f = text_feats(df)
    f = f.join(df[["locale"]].join(L, on="locale").drop(columns="locale"))
    f["exp_box"] = f.box_ratio / np.exp(f.f_lr)
    f["exp_box_p"] = f.box_ratio / np.exp(f.p_lr)
    return f


def score(y, pc, T, R, loc):
    def cc_macro(a, b):
        return max(0, (f1_score(a, b, average="macro", labels=CLS) - 1 / 3) / (2 / 3))
    s_fit = cc_macro(y, pc)
    over = np.clip((R - T) / T, 0, None); under = np.clip((T - R) / T, 0, None)
    s_res = np.clip(1 - over - 3 * under, 0, 1).mean()
    ws = [cc_macro(y[loc == l], pc[loc == l]) for l in np.unique(loc) if (loc == l).sum() >= 40]
    share = (y == "overflows").mean()
    fo = f1_score(y == "overflows", pc == "overflows")
    s_ov = max(0, (fo - share) / (1 - share))
    tot = .3 * s_fit + .25 * s_res + .2 * min(ws) + .25 * s_ov
    print(f"fit {s_fit:.4f} res {s_res:.4f} worst {min(ws):.4f} ovf {s_ov:.4f} TOTAL {tot:.4f}")
    return tot


def fit_predict(train, test, cold):
    L = locale_table(train, cold)
    Xtr, Xte = build(train, L), build(test, L)
    ytr = np.log(train.reserve_px / train.source_min_width_px)
    # warm locales are weighted down a bit so cold transfer isn't dominated by memorised stats
    reg = lgb.LGBMRegressor(objective="quantile", alpha=0.72, n_estimators=900, learning_rate=0.04,
                            num_leaves=63, min_child_samples=30, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, verbose=-1)
    reg.fit(Xtr, ytr)
    R = np.exp(reg.predict(Xte)) * test.source_min_width_px.values
    yc = train.fit_class.map({c: i for i, c in enumerate(CLS)})
    clf = lgb.LGBMClassifier(n_estimators=700, learning_rate=0.04, num_leaves=63, min_child_samples=30,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             class_weight="balanced", verbose=-1)
    clf.fit(Xtr, yc)
    P = clf.predict_proba(Xte)
    pc = np.array(CLS)[P.argmax(1)]
    # 1-line components cannot wrap
    one = test.max_lines.values == 1
    pc[one] = np.where(P[one, 2] > P[one, 0], "overflows", "fits")
    return pc, R


cold = set(tr[tr.groupby("locale").locale.transform("size") <= 12].locale)
if VALIDATE:
    rng = np.random.RandomState(0)
    warm = sorted(set(tr.locale) - cold)
    sim_cold = set(rng.choice(warm, 8, replace=False))
    pilot_ids = set(tr[tr.locale.isin(cold)].string_id)
    surf = ["support_diag", "dom_errors"]
    strs = tr.string_id.unique(); rng.shuffle(strs)
    val_str = set(strs[: len(strs) // 5]) - pilot_ids
    is_val = tr.string_id.isin(val_str) | tr.surface.isin(surf)
    trn = tr[~is_val & (~tr.locale.isin(sim_cold) | tr.string_id.isin(pilot_ids))]
    val = tr[is_val & ~tr.string_id.isin(pilot_ids)]
    pc, R = fit_predict(trn, val, cold | sim_cold)
    score(val.fit_class.values, pc, val.reserve_px.values, R, val.locale.values)
    vc = val.locale.isin(sim_cold).values
    score(val.fit_class.values[vc], pc[vc], val.reserve_px.values[vc], R[vc], val.locale.values[vc])

pc, R = fit_predict(tr, te, cold)
pd.DataFrame({"item_id": te.item_id, "fit_class": pc, "reserve_px": np.round(np.maximum(R, 1.0), 2)}).to_csv(OUT, index=False)
print("wrote", OUT, len(te))
