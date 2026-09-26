"""Localisation fit/reserve predictor.

Usage: python solution.py <public_data_dir> <submission_csv_path>

Compliance: every model, statistic, encoding and threshold is fitted on train.csv only (thresholds were chosen on
a held-out split of train.csv). test.csv is only featurised and scored by the fitted models: no test-time
augmentation or normalisation, no pseudo-labelling, no domain adaptation. No language models, translation memories,
parallel corpora or external data are used; the only inputs are the files in the public directory. LightGBM runs
with fixed seeds and deterministic=True on CPU.
"""
import sys, os, re
import numpy as np, pandas as pd, lightgbm as lgb

DATA, OUT = sys.argv[1], sys.argv[2]
tr = pd.read_csv(os.path.join(DATA, "train.csv"))
te = pd.read_csv(os.path.join(DATA, "test.csv"))
T_O, T_W = 0.46, 0.55
# cold locales: the pilot-only size estimate is less certain, so call overflow earlier and reserve a bit more
# (both chosen on a held-out split of train.csv with simulated cold locales)
T_O_COLD, COLD_RESERVE = 0.30, 1.06
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
    # string-adjusted pilot offsets: some cold locales have fewer than 12 pilot strings (mr: 2, te: 5), so the
    # offset is measured against each string's warm-locale mean and shrunk toward the script's warm offset.
    ms = p[~p.locale.isin(cold)].groupby("string_id")[["lr", "lw"]].mean()
    p = p.join(ms, on="string_id", rsuffix="_s")
    p = p.assign(dlr=p.lr - p.lr_s, dlw=p.lw - p.lw_s)
    g = p.groupby("locale").agg(n=("dlr", "size"), dlr=("dlr", "mean"), dlw=("dlw", "mean"),
                                script=("script", "first"))
    wg = g[~g.index.isin(cold)]
    prior = g.script.map(wg.groupby("script").dlr.mean()).fillna(0.0)
    prior_w = g.script.map(wg.groupby("script").dlw.mean()).fillna(0.0)
    K = 1.0
    P = pd.DataFrame(index=g.index)
    P["p_lr"] = ms.lr.mean() + (g.n * g.dlr + K * prior) / (g.n + K)
    P["p_lw"] = ms.lw.mean() + (g.n * g.dlw + K * prior_w) / (g.n + K)
    P["p_n"] = g.n
    full = t[~t.locale.isin(cold)]
    F = full.groupby("locale").agg(f_lr=("lr", "mean"), f_lw=("lw", "mean"), f_sd=("lr", "std"),
                                   f_q=("lr", lambda x: x.quantile(.8)))
    F["f_lr1"] = full[full.ml1].groupby("locale").lr.mean()
    F["f_lrm"] = full[~full.ml1].groupby("locale").lr.mean()
    F["f_ovf"] = full.groupby("locale").fit_class.apply(lambda x: (x == "overflows").mean())
    L = P.join(F, how="outer")
    L["script"] = t.groupby("locale").script.first()
    warm = L[L.f_lr.notna() & L.p_lr.notna()]
    Lc = L.copy()  # "cold view": every locale's full stats estimated from its pilot only
    for c in ["f_lr", "f_lw", "f_sd", "f_q", "f_lr1", "f_lrm", "f_ovf"]:
        base = "p_lw" if c == "f_lw" else "p_lr"
        if c in ("f_sd", "f_ovf"):
            # regress on pilot mean
            a, b = np.polyfit(warm.p_lr, warm[c], 1)
            est = a * L.p_lr + b
        else:
            est = L[base] + (warm[c] - warm[base]).mean()
        L.loc[L[c].isna(), c] = est
        Lc[c] = est
    L["is_cold"] = L.index.isin(cold).astype(int)
    Lc["is_cold"] = 1
    return L.drop(columns="script"), Lc.drop(columns="script")


def build(df, L):
    f = text_feats(df)
    f = f.join(df[["locale"]].join(L, on="locale").drop(columns="locale"))
    f["exp_box"] = f.box_ratio / np.exp(f.f_lr)
    f["exp_box_p"] = f.box_ratio / np.exp(f.p_lr)
    return f


TOK0 = re.compile(r"[a-z]+")


class _Tok:
    @staticmethod
    def findall(s):
        w = TOK0.findall(s)
        return w + [a + "_" + b for a, b in zip(w, w[1:])]


TOK = _Tok()


def word_stats(df, L):
    t = df[["string_id", "locale", "script", "source_text"]].copy()
    t["res"] = np.log(df.reserve_px / df.source_min_width_px).values - df.locale.map(L.f_lr).values
    sres = t.groupby("string_id").agg(res=("res", "mean"), txt=("source_text", "first"))
    W = {}
    for txt, r in zip(sres.txt, sres.res):
        for w in set(TOK.findall(str(txt).lower())):
            a = W.setdefault(w, [0.0, 0]); a[0] += r; a[1] += 1
    WL = {}
    for txt, loc, r in zip(t.source_text, t.locale, t.res):
        for w in set(TOK.findall(str(txt).lower())):
            a = WL.setdefault((w, loc), [0.0, 0]); a[0] += r; a[1] += 1
    WS = {}
    for txt, sc, r in zip(t.source_text, t.script, t.res):
        for w in set(TOK.findall(str(txt).lower())):
            a = WS.setdefault((w, sc), [0.0, 0]); a[0] += r; a[1] += 1
    return W, WL, WS


def word_feats(df, W, WL, WS, k=5.0, kl=3.0):
    out = np.full((len(df), 8), np.nan)
    for i, (txt, loc, sc) in enumerate(zip(df.source_text, df.locale, df.script)):
        ws = TOK.findall(str(txt).lower())
        if not ws:
            continue
        e = [W[w][0] / (W[w][1] + k) if w in W else 0.0 for w in ws]
        n = [W[w][1] if w in W else 0 for w in ws]
        el = [WL[(w, loc)][0] / (WL[(w, loc)][1] + kl) for w in ws if (w, loc) in WL]
        es = [WS[(w, sc)][0] / (WS[(w, sc)][1] + k) for w in ws if (w, sc) in WS]
        out[i] = [np.mean(e), np.min(e), np.max(e), np.mean(np.log1p(n)),
                  np.mean(el) if el else np.nan, len(el) / len(ws),
                  np.mean(es) if es else np.nan, len(es) / len(ws)]
    return pd.DataFrame(out, index=df.index, columns=["we_mean", "we_min", "we_max", "we_cnt", "wl_mean", "wl_frac",
                                                      "ws_mean", "ws_frac"])


def add_word(train, test, L, Xtr, Xte, nf=5):
    sids = train.string_id.unique()
    fold = dict(zip(sids, np.random.RandomState(1).randint(0, nf, len(sids))))
    fo = train.string_id.map(fold).values
    parts = []
    for k in range(nf):
        W, WL, WS = word_stats(train[fo != k], L)
        parts.append(word_feats(train[fo == k], W, WL, WS))
    Xtr = Xtr.join(pd.concat(parts))
    W, WL, WS = word_stats(train, L)
    Xte = Xte.join(word_feats(test, W, WL, WS))
    return Xtr, Xte


H = np.array([0.95, 1.10, 1.25, 1.45, 1.75])
BOXF = ["box_ratio", "box_sw", "box_seg", "exp_box", "exp_box_p"]
GB = dict(learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.8, verbose=-1, random_state=0, deterministic=True, n_jobs=4)


def set_box(X, df, box):
    X = X.copy()
    X["box_ratio"] = box / df.source_min_width_px.values
    X["box_sw"] = box / df.source_width_px.values
    X["box_seg"] = box / df.source_widest_segment_px.values
    X["exp_box"] = X.box_ratio / np.exp(X.f_lr)
    X["exp_box_p"] = X.box_ratio / np.exp(X.p_lr)
    return X


def augment(X, df):
    """The translation's reserve does not depend on the box, so every labelled row tells us whether
    it would overflow at each of the 5 published headroom steps."""
    Xs, ys = [], []
    for h in H:
        box = np.round(h * df.source_min_width_px.values)
        Xs.append(set_box(X, df, box)); ys.append((df.reserve_px.values > box).astype(int))
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def fit_predict(train, test, cold, alpha=0.72):
    L, Lc = locale_table(train, cold)
    Xtr, Xte = build(train, L), build(test, L)
    Xtr, Xte = add_word(train, test, L, Xtr, Xte)
    # cold-view copies of the warm rows: the model learns to size a language from its pilot alone,
    # which is all it has for the cold locales at prediction time.
    w = ~train.locale.isin(cold).values
    Xc = build(train[w], Lc).join(Xtr.loc[w, [c for c in Xtr.columns if c.startswith(("we_", "wl_", "ws_"))]])
    Xc[["wl_mean", "wl_frac"]] = np.nan
    Xtr = pd.concat([Xtr, Xc[Xtr.columns]], ignore_index=True)
    train = pd.concat([train, train[w]], ignore_index=True)
    base = [c for c in Xtr.columns if c not in BOXF]
    ytr = np.log(train.reserve_px / train.source_min_width_px)
    reg = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=800, **GB)
    reg.fit(Xtr[base], ytr)
    R = np.exp(reg.predict(Xte[base])) * test.source_min_width_px.values
    # overflow: augmented binary model over the headroom grid
    Xa, ya = augment(Xtr, train)
    ovf = lgb.LGBMClassifier(n_estimators=800, **GB).fit(Xa, ya)
    p_ovf = ovf.predict_proba(Xte)[:, 1]
    # single-line width > box ("does not fit on one line"): 1-line rows augmented + multi-line rows as labelled
    m1 = (train.max_lines == 1).values
    Xa1, ya1 = augment(Xtr[m1], train[m1])
    Xw = pd.concat([Xa1, Xtr[~m1]], ignore_index=True)
    yw = np.concatenate([ya1, (train.fit_class.values[~m1] != "fits").astype(int)])
    wm = lgb.LGBMClassifier(n_estimators=800, **GB).fit(Xw, yw)
    p_w = wm.predict_proba(Xte)[:, 1]
    return p_ovf, p_w, R


def decide(p_ovf, p_w, ml, t_o=0.5, t_w=0.5):
    pc = np.where(p_w > t_w, "wraps", "fits").astype(object)
    pc[ml == 1] = "fits"
    pc[p_ovf > t_o] = "overflows"
    return pc.astype(str)


cold = set(tr[tr.groupby("locale").locale.transform("size") <= 12].locale)
p_ovf, p_w, R = fit_predict(tr, te, cold)
pc = decide(p_ovf, p_w, te.max_lines.values, T_O, T_W)
is_c = te.locale.isin(cold).values
pc[is_c] = decide(p_ovf, p_w, te.max_lines.values, T_O_COLD, T_W)[is_c]
R = R * np.where(is_c, COLD_RESERVE, 1.0)
pd.DataFrame({"item_id": te.item_id, "fit_class": pc, "reserve_px": np.round(np.maximum(R, 1.0), 2)}).to_csv(OUT, index=False)
print("wrote", OUT, len(te))
