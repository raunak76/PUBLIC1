import sys, os, re, json, math, itertools, bisect
import numpy as np
import pandas as pd
import lightgbm as lgb

SEED = 0
RANK = 1
np.random.seed(SEED)

CALC_INPUTS = {
    'map': ['sbp', 'dbp'],
    'bmi': ['weight', 'height'],
    'bsa': ['weight', 'height'],
    'ibw': ['height'],
    'abw': ['weight', 'height'],
    'serum_osmolality': ['sodium', 'bun', 'glucose'],
    'anion_gap': ['sodium', 'chloride', 'bicarbonate'],
    'delta_gap': ['sodium', 'chloride', 'bicarbonate'],
    'delta_ratio': ['sodium', 'chloride', 'bicarbonate'],
    'albumin_corrected_anion_gap': ['sodium', 'chloride', 'bicarbonate', 'albumin'],
    'albumin_corrected_delta_gap': ['sodium', 'chloride', 'bicarbonate', 'albumin'],
    'albumin_corrected_delta_ratio': ['sodium', 'chloride', 'bicarbonate', 'albumin'],
    'calcium_correction': ['calcium', 'albumin'],
    'sodium_correction': ['sodium', 'glucose'],
    'ldl': ['total_cholesterol', 'hdl', 'triglycerides'],
    'fena': ['sodium', 'urine_sodium', 'creatinine', 'urine_creatinine'],
    'fib4': ['age', 'ast', 'alt', 'platelets'],
    'homa_ir': ['insulin', 'glucose'],
    'free_water_deficit': ['weight', 'sodium', 'age'],
    'mdrd': ['creatinine', 'age'],
}
NEEDS_SEX = {'ibw', 'abw', 'free_water_deficit', 'mdrd'}
NEEDS_RACE = {'mdrd'}

UNITS = {
    'weight': [('kg', 1.0), ('lbs', 0.453592), ('g', 0.001)],
    'height': [('cm', 1.0), ('m', 100.0), ('in', 2.54), ('ft', 30.48)],
    'sodium': [('mmol/l', 1.0)], 'chloride': [('mmol/l', 1.0)],
    'bicarbonate': [('mmol/l', 1.0)], 'urine_sodium': [('mmol/l', 1.0)],
    'albumin': [('g/dl', 1.0), ('g/l', 0.1), ('mg/dl', 0.001)],
    'calcium': [('mg/dl', 1.0), ('mmol/l', 4.008)],
    'glucose': [('mg/dl', 1.0), ('mmol/l', 18.016)],
    'bun': [('mg/dl', 1.0), ('mmol/l', 2.801)],
    'creatinine': [('mg/dl', 1.0), ('umol/l', 1 / 88.4)],
    'urine_creatinine': [('mg/dl', 1.0), ('umol/l', 1 / 88.4)],
    'total_cholesterol': [('mg/dl', 1.0), ('mmol/l', 38.67)],
    'hdl': [('mg/dl', 1.0), ('mmol/l', 38.67)],
    'triglycerides': [('mg/dl', 1.0), ('mmol/l', 88.57)],
    'insulin': [('uiu/ml', 1.0), ('pmol/l', 1 / 6)],
    'platelets': [('10^9/l', 1.0), ('/ul', 0.001)],
    'age': [('years', 1.0), ('months', 1 / 12), ('weeks', 1 / 52), ('days', 1 / 365)],
    'ast': [('u/l', 1.0)], 'alt': [('u/l', 1.0)],
    'sbp': [('mmhg', 1.0)], 'dbp': [('mmhg', 1.0)],
}
UNIT_ALIASES = {'meq/l': 'mmol/l'}
SLOTS = sorted(UNITS)
SLOT_ID = {s: i for i, s in enumerate(SLOTS)}

KEYWORDS = {
    'sbp': r'systolic|blood pressure|\bbp\b|\bsbp\b|mm\s*hg|arterial pressure',
    'dbp': r'diastolic|blood pressure|\bbp\b|\bdbp\b|mm\s*hg|arterial pressure',
    'weight': r'weigh|body mass\b|\bwt\b|\bkg\b|\blbs?\b|pounds',
    'height': r'height|\btall\b|length|stature|\bht\b|\bcm\b',
    'sodium': r'sodium|\bna\b|\bna\+|natremia',
    'chloride': r'chloride|\bcl\b|\bcl-',
    'bicarbonate': r'bicarbonate|hco3|hco 3|bicarb|\bco2\b|\btco2|total co',
    'urine_sodium': r'urin[a-z]* sodium|urin[a-z]* na\b|\buna\b|sodium[^.]{0,20}urin',
    'albumin': r'albumin|\balb\b',
    'calcium': r'calcium|\bca\b|\bca2|calcemia',
    'glucose': r'glucose|sugar|glyc[ae]mia|\bbg\b|\bbs\b|\bglu\b',
    'bun': r'\bbun\b|urea',
    'creatinine': r'creatinine|\bcr\b|\bscr\b|\bcre\b|\bcreat\b',
    'urine_creatinine': r'urin[a-z]* creatinine|urin[a-z]* cr\b|creatinine[^.]{0,20}urin',
    'total_cholesterol': r'cholesterol|\btc\b|\bt-chol',
    'hdl': r'\bhdl|high[- ]density',
    'triglycerides': r'triglyceride|\btg\b|\btgs\b',
    'insulin': r'insulin',
    'platelets': r'platelet|\bplt\b|\bplts\b',
    'ast': r'\bast\b|aspartate|\bgot\b|\bsgot\b',
    'alt': r'\balt\b|alanine|\bgpt\b|\bsgpt\b',
    'age': r'year[- ]old|years? old|\baged?\b|month[- ]old|day[- ]old|week[- ]old|\byo\b|y/o',
}
KW_RE = {s: re.compile(p, re.I) for s, p in KEYWORDS.items()}

CUES = ['admi', 'present', 'arriv', 'emergency', 'initial', 'examination', 'baseline', 'prior', 'previous',
        'ago', 'history', 'later', 'after', 'follow', 'discharge', 'post', 'repeat', 'subsequent', 'improv',
        'treat', 'day', 'month', 'year', 'week', 'hour', 'normal', 'range', 'refer', 'urine', 'urinary',
        'serum', 'plasma', 'blood', 'birth', 'current', 'decreas', 'increas', 'rose', 'fell', 'drop',
        'peak', 'nadir', 'target', 'goal', 'when', 'at the time', 'upon', 'on ', 'visit', 'outpatient',
        'hospital', 'transfer', 'dose', 'mg/kg', 'ml', 'percent', 'gestation', 'weeks', 'birth weight',
        'mother', 'father', 'sibling', 'sister', 'brother', 'reveal', 'showed', 'laborator', 'lab',
        'ventilat', 'gas', 'abg', 'venous', 'arterial', 'corrected', 'calculated', 'estimated', 'fluid',
        'infus', 'bolus', 'saline', 'within', 'range', 'less than', 'greater', 'above', 'below', 'up to',
        'first', 'second', 'third', 'next', 'same', 'finally', 'eventually', 'weeks later', 'days later']
CUES = list(dict.fromkeys(CUES))

MALE_RE = re.compile(r'\b(he|his|him|himself|man|male|boy|gentleman|mr|husband|father|son|men)\b', re.I)
FEMALE_RE = re.compile(r'\b(she|her|hers|herself|woman|female|girl|lady|mrs|ms|wife|mother|daughter|women|pregnan\w*|gravida|menstrua\w*)\b', re.I)
BLACK_RE = re.compile(r'african[- ]american|\bblack\b|african|afro', re.I)
OTHER_RACE_RE = re.compile(r'caucasian|\bwhite\b|asian|hispanic|latino|chinese|japanese|indian|korean|arab', re.I)

NUM_RE = re.compile(r'(?<![^\W\d])(?<![\d.,])\d+(?:,\d+)*(?:\.\d+)?')
LOOSE_RE = re.compile(r'\d+(?:,\d+)*(?:\.\d+)?')


def fnum(s):
    return float(s.replace(',', ''))


def tokenize(text, expected):
    ms = [(m.start(), m.end()) for m in NUM_RE.finditer(text)]
    if len(ms) == len(expected) and all(abs(fnum(text[a:b]) - v) < 1e-9 for (a, b), v in zip(ms, expected)):
        return ms
    out = []
    pos = 0
    for v in expected:
        found = None
        p = pos
        while p < len(text):
            m = LOOSE_RE.match(text, p)
            if m:
                for e in range(m.end(), m.start(), -1):
                    seg = text[m.start():e].rstrip(',.')
                    if seg and abs(fnum(seg) - v) < 1e-9:
                        found = (m.start(), m.start() + len(seg))
                        break
                if found:
                    break
            p += 1
        if found is None:
            found = (min(pos, len(text)), min(pos, len(text)))
        out.append(found)
        pos = found[1]
    return out


def unit_str(text, end):
    m = re.match(r'\s*([^\s\d,;()]{0,14})', text[end:end + 20])
    return m.group(1).lower().rstrip('.') if m else ''


def calc(c, v, sex='male', race='other'):
    g = lambda k: v[k]
    if c == 'map':
        return (2 * g('dbp') + g('sbp')) / 3
    if c == 'bmi':
        return g('weight') / (g('height') / 100) ** 2
    if c == 'bsa':
        return np.sqrt(np.maximum(g('weight') * g('height'), 0) / 3600)
    if c in ('ibw', 'abw'):
        ibw = (50 if sex == 'male' else 45.5) + 2.3 * (g('height') / 2.54 - 60)
        if c == 'ibw':
            return ibw
        return ibw + 0.4 * (g('weight') - ibw)
    if c == 'serum_osmolality':
        return 2 * g('sodium') + g('bun') / 2.8 + g('glucose') / 18
    if c in ('anion_gap', 'delta_gap', 'delta_ratio', 'albumin_corrected_anion_gap',
             'albumin_corrected_delta_gap', 'albumin_corrected_delta_ratio'):
        ag = g('sodium') - (g('chloride') + g('bicarbonate'))
        if c.startswith('albumin'):
            ag = ag + 2.5 * (4 - g('albumin'))
        if c.endswith('anion_gap'):
            return ag
        if c.endswith('delta_gap'):
            return ag - 12
        with np.errstate(divide='ignore', invalid='ignore'):
            return (ag - 12) / (24 - g('bicarbonate'))
    if c == 'calcium_correction':
        return g('calcium') + 0.8 * (4 - g('albumin'))
    if c == 'sodium_correction':
        return g('sodium') + 0.024 * (g('glucose') - 100)
    if c == 'ldl':
        return g('total_cholesterol') - g('hdl') - g('triglycerides') / 5
    if c == 'fena':
        with np.errstate(divide='ignore', invalid='ignore'):
            return g('urine_sodium') * g('creatinine') / (g('sodium') * g('urine_creatinine')) * 100
    if c == 'fib4':
        with np.errstate(divide='ignore', invalid='ignore'):
            return g('age') * g('ast') / (g('platelets') * np.sqrt(g('alt')))
    if c == 'homa_ir':
        return g('insulin') * g('glucose') / 405
    if c == 'free_water_deficit':
        a = g('age')
        if sex == 'male':
            f = np.where(a < 18, 0.6, np.where(a < 65, 0.6, 0.5))
        else:
            f = np.where(a < 18, 0.6, np.where(a < 65, 0.5, 0.45))
        return f * g('weight') * (g('sodium') / 140 - 1)
    if c == 'mdrd':
        with np.errstate(divide='ignore', invalid='ignore'):
            r = 175 * np.power(g('creatinine'), -1.154) * np.power(g('age'), -0.203)
        if sex == 'female':
            r = r * 0.742
        if race == 'black':
            r = r * 1.212
        return r
    raise ValueError(c)


def close(a, b, rel=1e-3, ab=1e-4):
    return np.isfinite(a) & (np.abs(a - b) <= np.maximum(rel * np.abs(b), ab))


class Case:
    pass


def prep_case(row):
    c = Case()
    c.id = row.case_id
    c.calc = row.calculator
    c.note = row.note
    nums = json.loads(row.numbers_json)
    c.refs = [r for r, v in nums if r[0] == 'x']
    c.vals = np.array([float(v) for r, v in nums if r[0] == 'x'])
    c.spans = tokenize(c.note, list(c.vals))
    return c


def note_context(c, unit_vocab):
    text = c.note
    low = text.lower()
    n = len(c.vals)
    L = max(len(text), 1)
    kw_pos = {}
    for s, rx in KW_RE.items():
        kw_pos[s] = [(m.start(), m.end()) for m in rx.finditer(text)]
    sent_bounds = [0] + [m.end() for m in re.finditer(r'\.\s+(?=[A-Z])|;\s', text)] + [L + 1]
    starts = [a for a, b in c.spans]
    base = []
    for k, (a, b) in enumerate(c.spans):
        v = c.vals[k]
        u = unit_str(text, b)
        prev = text[max(0, a - 3):a]
        nxt = text[b:b + 3]
        si = bisect.bisect_right(sent_bounds, a) - 1
        s0, s1 = sent_bounds[si], sent_bounds[si + 1]
        win = low[max(0, a - 120):a]
        sent = low[s0:min(s1, L)]
        cue_l = [1.0 if q in win else 0.0 for q in CUES]
        cue_s = [1.0 if q in sent else 0.0 for q in CUES]
        f = [
            math.log10(v + 1e-3), v, float(v == int(v)), k / max(n, 1), a / L, float(k), float(n - k),
            float(bool(re.match(r'\s*/\s*\d', text[b:b + 4]))),
            float(bool(re.search(r'\d\s*/\s*$', text[max(0, a - 4):a]))),
            float('-' in prev or '–' in prev), float(nxt.startswith('-') or nxt.startswith('–')),
            float('(' in prev), float(')' in nxt), float('±' in text[b:b + 3] or '±' in prev),
            float(unit_vocab.get(u, -1)), si / max(len(sent_bounds), 1), float(np.sum(c.vals == v)),
            float(np.sum(c.vals[:k] == v)),
        ] + cue_l + cue_s
        base.append(f)
    return np.array(base, dtype=float), kw_pos, starts


UNIT_PAT = {
    'kg': r'^kg', 'lbs': r'^(lb|pound)', 'g': r'^(g|gram|gm)$', 'cm': r'^cm', 'm': r'^(m|meter|metre)$',
    'in': r'^(in|inch|")', 'ft': r'^(ft|feet|foot|\')', 'mmol/l': r'^(mmol|meq|mm/l|mm$)', 'g/dl': r'^(g/dl|gm/dl|g/100)',
    'g/l': r'^g/l', 'mg/dl': r'^mg/dl|^mg%', 'umol/l': r'^(μmol|µmol|umol|micromol)', 'uiu/ml': r'^(μiu|µiu|uiu|miu/l|μu|µu|uu|mu/l|iu/ml|mu/ml)',
    'pmol/l': r'^pmol', '10^9/l': r'^(×|x|\*|10)', '/ul': r'^(/μl|/µl|/ul|/mm|/cumm|cells)', 'years': r'^(-?year|y|yr)',
    'months': r'^(-?month|mo)', 'weeks': r'^(-?week|wk)', 'days': r'^(-?day|d$)', 'u/l': r'^(u/l|iu/l|iu|u$)', 'mmhg': r'^(mm ?hg|/)',
}
UNIT_PAT = {k: re.compile(p) for k, p in UNIT_PAT.items()}


def slot_feats(c, base, kw_pos, starts, slot, unit_vocab, slot_stats):
    rows = []
    meta = []
    text = c.note
    kws = kw_pos[slot]
    kw_ends = sorted(e for s_, e in kws)
    kw_starts = sorted(s_ for s_, e in kws)
    all_kw = []
    for s2, lst in kw_pos.items():
        for s_, e in lst:
            all_kw.append((e, s2))
    all_kw.sort()
    all_kw_e = [e for e, s2 in all_kw]
    nkw = len(kws)
    mu, sd = slot_stats.get(slot, (0.0, 3.0))
    for k, (a, b) in enumerate(c.spans):
        i = bisect.bisect_right(kw_ends, a) - 1
        dl = a - kw_ends[i] if i >= 0 else 5000
        nl = (bisect.bisect_left(starts, a) - bisect.bisect_left(starts, kw_ends[i])) if i >= 0 else 99
        j = bisect.bisect_left(kw_starts, b)
        dr = kw_starts[j] - b if j < len(kw_starts) else 5000
        nr = (bisect.bisect_left(starts, kw_starts[j]) - k - 1) if j < len(kw_starts) else 99
        ii = bisect.bisect_right(all_kw_e, a) - 1
        near_is = 0.0
        near_d = 5000
        if ii >= 0:
            near_d = a - all_kw[ii][0]
            e0 = all_kw[ii][0]
            near_is = float(any(e == e0 and s2 == slot for e, s2 in all_kw[max(0, ii - 5):ii + 5]))
        occ_idx = i
        u = unit_str(text, b)
        for ui, (un, fac) in enumerate(UNITS[slot]):
            cv = c.vals[k] * fac
            z = (math.log10(cv + 1e-6) - mu) / sd
            pm = UNIT_PAT.get(un)
            f = list(base[k]) + [
                SLOT_ID[slot], ui, fac, math.log10(cv + 1e-6), z, abs(z),
                float(bool(pm.search(u))) if pm else -1.0,
                dl, nl, dr, nr, near_d, near_is, float(occ_idx), float(nkw),
                float(occ_idx) / max(nkw, 1),
                float(len(UNITS[slot])),
            ]
            rows.append(f)
            meta.append((k, ui))
    R = np.array(rows, dtype=float)
    nb = len(base[0])
    col_nl = nb + 8
    col_dl = nb + 7
    col_pm = nb + 6
    adj = (R[:, col_nl] == 0) & (R[:, col_dl] < 60)
    adj_u = adj & (R[:, col_pm] == 1)
    kpos = R[:, 3]
    extra = []
    for mask in (adj, adj_u):
        cnt = mask.sum()
        ks = np.unique(np.array([m[0] for m, a_ in zip(meta, mask) if a_], dtype=int))
        rk = np.array([np.searchsorted(ks, m[0]) if a_ else -1 for m, a_ in zip(meta, mask)], dtype=float)
        vv = np.unique(np.array([c.vals[m[0]] for m, a_ in zip(meta, mask) if a_]))
        extra += [np.full(len(R), cnt), rk, np.where(rk >= 0, len(ks) - 1 - rk, -1), np.full(len(R), len(vv))]
    order_dl = np.argsort(np.argsort(R[:, col_dl] + 1000 * R[:, col_nl]))
    extra.append(order_dl.astype(float))
    R = np.hstack([R, np.array(extra).T])
    return R, meta


def build_unit_vocab(cases, top=200):
    from collections import Counter
    cnt = Counter()
    for c in cases:
        for a, b in c.spans:
            cnt[unit_str(c.note, b)] += 1
    return {u: i for i, (u, _) in enumerate(cnt.most_common(top))}


def parse_binding(s):
    out = {}
    for p in s.split(';'):
        p = p.strip()
        if not p:
            continue
        k, v = p.split('=')
        k = k.strip().lower()
        v = v.strip()
        if k in ('sex', 'race'):
            out[k] = v.lower()
        else:
            parts = v.split(None, 1)
            u = parts[1].lower() if len(parts) > 1 else None
            out[k] = (parts[0], UNIT_ALIASES.get(u, u))
    return out


def unit_index(slot, u):
    if u is None:
        return 0
    for i, (un, f) in enumerate(UNITS[slot]):
        if un == u:
            return i
    return 0


def sex_feats(c):
    t = c.note
    m = MALE_RE.findall(t)
    f = FEMALE_RE.findall(t)
    fm = MALE_RE.search(t)
    ff = FEMALE_RE.search(t)
    head = t[:300]
    return [len(m), len(f), fm.start() if fm else 1e5, ff.start() if ff else 1e5,
            len(MALE_RE.findall(head)), len(FEMALE_RE.findall(head)), len(t)]


def race_feats(c):
    t = c.note
    b = BLACK_RE.findall(t)
    o = OTHER_RACE_RE.findall(t)
    return [len(b), len(o), float(bool(re.search(r'african[- ]american', t, re.I))), len(t)]


def main():
    pub, out_path = sys.argv[1], sys.argv[2]
    # Inputs are limited to the released public train/test files.
    tr = pd.read_csv(os.path.join(pub, 'train.csv'))
    te = pd.read_csv(os.path.join(pub, 'test.csv'))
    lab = pd.read_csv(os.path.join(pub, 'train_labels.csv'))
    seed = pd.read_csv(os.path.join(pub, 'train_seed_bindings.csv'))
    tr = tr.merge(lab, on='case_id', how='left')
    seed_map = dict(zip(seed.case_id, seed.binding))
    tr_cases = [prep_case(r) for r in tr.itertuples()]
    for c, a in zip(tr_cases, tr.answer):
        c.answer = a
    te_cases = [prep_case(r) for r in te.itertuples()]
    unit_vocab = build_unit_vocab(tr_cases)
    model = Pipeline(unit_vocab)
    model.fit(tr_cases, seed_map)
    preds = [model.predict_binding(c) for c in te_cases]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pd.DataFrame({'case_id': [c.id for c in te_cases], 'binding': preds}).to_csv(out_path, index=False)


class Pipeline:
    def __init__(self, unit_vocab, n_iter=3, topn=10):
        self.unit_vocab = unit_vocab
        self.n_iter = n_iter
        self.topn = topn
        self.slot_stats = {}

    def case_feats(self, c):
        if getattr(c, '_fcache', None) is not None:
            return c._fcache
        base, kw_pos, starts = note_context(c, self.unit_vocab)
        res = {}
        for s in CALC_INPUTS[c.calc]:
            rows, meta = slot_feats(c, base, kw_pos, starts, s, self.unit_vocab, self.slot_stats)
            res[s] = (rows, meta)
        c._fcache = res
        return res

    def seed_targets(self, c, b):
        tg = {}
        for s in CALC_INPUTS[c.calc]:
            if s not in b:
                return None
            ref, u = b[s]
            k = c.refs.index(ref) if ref in c.refs else None
            if k is None:
                return None
            tg[s] = (k, unit_index(s, u))
        return tg

    def fit_slot_stats(self, labelled):
        vals = {}
        for c, tg in labelled:
            for s, (k, ui) in tg.items():
                vals.setdefault(s, []).append(math.log10(c.vals[k] * UNITS[s][ui][1] + 1e-6))
        for s, v in vals.items():
            v = np.array(v)
            self.slot_stats[s] = (float(np.median(v)), float(max(np.std(v), 0.05)))

    def train_ranker(self, labelled):
        X, y, G = [], [], []
        for c, tg in labelled:
            F = self.case_feats(c)
            for s, (k, ui) in tg.items():
                rows, meta = F[s]
                v = c.vals[k]
                g0 = len(y)
                for r, (kk, uu) in zip(rows, meta):
                    if kk == k and uu == ui:
                        X.append(r); y.append(1)
                    elif c.vals[kk] == v and uu == ui:
                        continue
                    else:
                        X.append(r); y.append(0)
                G.append(len(y) - g0)
        X = np.array(X); y = np.array(y)
        if RANK:
            m = lgb.LGBMRanker(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=10,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                               random_state=SEED, verbose=-1, n_jobs=4)
            m.fit(X, y, group=G)
            self.ranker = m
            return
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=10,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                               random_state=SEED, verbose=-1, n_jobs=4)
        m.fit(X, y)
        self.ranker = m

    def slot_scores(self, c):
        F = self.case_feats(c)
        out = {}
        for s, (rows, meta) in F.items():
            if RANK:
                z = self.ranker.predict(rows)
                z = np.exp(z - z.max())
                p = z / z.sum()
            else:
                p = self.ranker.predict_proba(rows)[:, 1]
            out[s] = (p, meta)
        return out

    def search(self, c, scores):
        slots = CALC_INPUTS[c.calc]
        cand = []
        for s in slots:
            p, meta = scores[s]
            order = np.argsort(-p)
            seen = set()
            lst = []
            for o in order:
                k, ui = meta[o]
                key = (c.vals[k], ui)
                if key in seen:
                    continue
                seen.add(key)
                lst.append((k, ui, c.vals[k] * UNITS[s][ui][1], math.log(p[o] + 1e-9)))
                if len(lst) >= self.topn:
                    break
            cand.append(lst)
        sexes = ['male', 'female'] if c.calc in NEEDS_SEX else ['male']
        races = ['other', 'black'] if c.calc in NEEDS_RACE else ['other']
        grids = np.meshgrid(*[np.arange(len(l)) for l in cand], indexing='ij')
        idx = [g.ravel() for g in grids]
        vals = {s: np.array([x[2] for x in cand[i]])[idx[i]] for i, s in enumerate(slots)}
        lp = sum(np.array([x[3] for x in cand[i]])[idx[i]] for i in range(len(slots)))
        best = None
        for sx in sexes:
            for rc in races:
                with np.errstate(all='ignore'):
                    r = calc(c.calc, vals, sx, rc)
                ok = close(np.asarray(r, dtype=float) * np.ones_like(lp), c.answer)
                if ok.any():
                    j = np.where(ok)[0]
                    jb = j[np.argmax(lp[j])]
                    if best is None or lp[jb] > best[0]:
                        best = (lp[jb], {s: (cand[i][idx[i][jb]][0], cand[i][idx[i][jb]][1]) for i, s in enumerate(slots)}, sx, rc, len(j))
        return best

    def fit(self, cases, seed_map):
        self.cases = cases
        seeds = []
        seed_sex, seed_race = [], []
        for c in cases:
            if c.id in seed_map:
                b = parse_binding(seed_map[c.id])
                tg = self.seed_targets(c, b)
                if tg is not None:
                    seeds.append((c, tg))
                if 'sex' in b:
                    seed_sex.append((c, b['sex']))
                if 'race' in b:
                    seed_race.append((c, b['race']))
        self.fit_slot_stats(seeds)
        for c in cases:
            c._fcache = None
        labelled = seeds
        seed_ids = {c.id for c, _ in seeds}
        for it in range(self.n_iter):
            self.train_ranker(labelled)
            found = []
            sex_l, race_l = list(seed_sex), list(seed_race)
            for c in cases:
                if c.id in seed_ids or not np.isfinite(c.answer):
                    continue
                best = self.search(c, self.slot_scores(c))
                if best is None:
                    continue
                found.append((c, best[1]))
                if c.calc in NEEDS_SEX:
                    alt = self._sex_decisive(c, best)
                    if alt:
                        sex_l.append((c, best[2]))
                if c.calc in NEEDS_RACE:
                    if self._race_decisive(c, best):
                        race_l.append((c, best[3]))
            labelled = seeds + found
            self.n_found = len(found)
            if it == 0:
                self.fit_slot_stats(labelled)
                for c in cases:
                    c._fcache = None
        self.train_ranker(labelled)
        self.train_sex(sex_l, race_l)

    def _sex_decisive(self, c, best):
        v = {s: np.array([c.vals[k] * UNITS[s][ui][1]]) for s, (k, ui) in best[1].items()}
        r1 = calc(c.calc, v, 'male', best[3])
        r2 = calc(c.calc, v, 'female', best[3])
        return not np.allclose(r1, r2)

    def _race_decisive(self, c, best):
        return True

    def train_sex(self, sex_l, race_l):
        X = np.array([sex_feats(c) for c, s in sex_l]); y = np.array([s == 'female' for c, s in sex_l], dtype=int)
        self.sex_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                                            random_state=SEED, verbose=-1).fit(X, y)
        X = np.array([race_feats(c) for c, s in race_l]); y = np.array([s == 'black' for c, s in race_l], dtype=int)
        self.race_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=7, min_child_samples=3,
                                             random_state=SEED, verbose=-1).fit(X, y)

    def predict_struct(self, c):
        sc = self.slot_scores(c)
        res = {}
        for s, (p, meta) in sc.items():
            j = int(np.argmax(p))
            res[s] = meta[j]
        sex = 'female' if self.sex_model.predict_proba(np.array([sex_feats(c)]))[0, 1] > 0.5 else 'male'
        race = 'black' if self.race_model.predict_proba(np.array([race_feats(c)]))[0, 1] > 0.5 else 'other'
        return res, sex, race

    def predict_binding(self, c):
        res, sex, race = self.predict_struct(c)
        parts = ['%s=%s %s' % (s, c.refs[k], UNITS[s][ui][0]) for s, (k, ui) in res.items()]
        if c.calc in NEEDS_SEX:
            parts.append('sex=' + sex)
        if c.calc in NEEDS_RACE:
            parts.append('race=' + race)
        return '; '.join(parts)


if __name__ == '__main__':
    main()
