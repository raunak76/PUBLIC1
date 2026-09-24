"""Unseen-Pair Shell Pipeline Synthesis - compositional retrieval + MBR solver.

Pure Python (stdlib only), CPU, trained from scratch on train.csv; no pretrained models or external data.
Pipeline:
  1. TF-IDF (unigram+bigram) retrieval over training descriptions.
  2. First stage: head voted by nearest neighbours. For `find`, neighbour commands are
     abstracted into skeletons (paths/patterns/numbers -> typed slots); the skeleton is chosen by
     minimum-Bayes-risk (expected token-Levenshtein similarity), then slots are filled with typed
     literals copied from the description (or voted implicit defaults such as `.`).
  3. Tail stage: head voted from neighbours' downstream stages, restricted to heads that never follow
     the first head in training (the benchmark's held-out-composition premise). `xargs` bodies choose
     their inner command by neighbour vote. The fragment is chosen by MBR among training stages
     with that head, with literal substitution (literals used by stage 1 are not reused).
Usage: python solution.py <data_dir> <submission.csv>
"""
import csv, re, math, collections, os, sys, shlex


def load(p):
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

MT = re.compile(r'\w+|[^\w\s]')
def mtok(s): return MT.findall(s)

def lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def case_score(p, t):
    P, T = mtok(p), mtok(t)
    sim = 1 - lev(P, T) / max(len(P), len(T), 1)
    return 0.5 * sim + 0.5 * (P == T)

def stages(c):
    out = []; cur = ''; q = None; i = 0
    while i < len(c):
        ch = c[i]
        if q:
            cur += ch
            if ch == '\\' and q == '"' and i + 1 < len(c): cur += c[i + 1]; i += 1
            elif ch == q: q = None
        else:
            if ch in '"\'': q = ch; cur += ch
            elif ch == '\\' and i + 1 < len(c): cur += ch + c[i + 1]; i += 1
            elif ch == '|' and not (i + 1 < len(c) and c[i + 1] == '|') and not (i > 0 and c[i - 1] in '|>'):
                out.append(cur.strip()); cur = ''
            else: cur += ch
        i += 1
    out.append(cur.strip())
    return out

def head(s):
    w = s.split()
    return w[0] if w else ''

def heads(c): return [head(s) for s in stages(c)]
def pairs(c):
    h = heads(c); return set(zip(h, h[1:]))

# ---------- description words ----------
WR = re.compile(r"[a-z0-9]+")
STOP = set('the a an of to in and all that with for from its their them it is are be on by as or this these those'.split())
def dwords(s):
    w = [x for x in WR.findall(s.lower())]
    w = [x[:-1] if len(x) > 3 and x.endswith('s') and not x.endswith('ss') else x for x in w]
    return w

def feats(s):
    w = dwords(s)
    f = [x for x in w if x not in STOP]
    f += [a + '_' + b for a, b in zip(w, w[1:])]
    return f

class TfIdf:
    def __init__(self, docs):
        self.df = collections.Counter()
        for d in docs: self.df.update(set(d))
        self.n = len(docs)
        self.vecs = [self.vec(d) for d in docs]
        self.inv = collections.defaultdict(list)
        for i, v in enumerate(self.vecs):
            for k, x in v.items(): self.inv[k].append((i, x))
    def idf(self, k): return math.log((self.n + 1) / (self.df.get(k, 0) + 1)) + 1
    def vec(self, d):
        c = collections.Counter(d)
        v = {k: (1 + math.log(x)) * self.idf(k) for k, x in c.items()}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1
        return {k: x / nrm for k, x in v.items()}
    def sims(self, d, allowed=None):
        q = self.vec(d); s = collections.defaultdict(float)
        for k, x in q.items():
            for i, y in self.inv.get(k, ()):
                s[i] += x * y
        if allowed is not None:
            s = {i: v for i, v in s.items() if i in allowed}
        return s

# ---------- literals ----------
QR = re.compile(r'(?<!\w)"([^"]*)"(?!\w)|(?<!\w)\'([^\']*)\'(?!\w)|`([^`]*)`|‘([^’]*)’|“([^”]*)”')
def dlits(s):
    """ordered literal strings found in a description"""
    out = []
    spans = []
    for m in QR.finditer(s):
        v = next(g for g in m.groups() if g is not None)
        if v.strip():
            out.append((m.start(), v)); spans.append((m.start(), m.end()))
    for m in re.finditer(r'\S+', s):
        if any(a <= m.start() < b for a, b in spans): continue
        t = m.group().strip('.,;:()[]')
        t2 = t
        if not t: continue
        if re.match(r'^[A-Za-z]+(/[A-Za-z]+)+$', t): continue
        if re.search(r'[/\*\$~\\\?]|\d', t) or re.match(r'^\.[\w]+$', t) or re.search(r'\w\.\w', t) or re.match(r'^-\w', t):
            out.append((m.start(), t2))
    out.sort()
    return [v for _, v in out]

def unq(t):
    if len(t) >= 2 and t[0] == t[-1] and t[0] in '"\'': return t[1:-1]
    return t

def ctoks(stage):
    """whitespace tokens of a stage respecting quotes (keeps quote chars)"""
    toks = []; cur = ''; q = None; i = 0
    while i < len(stage):
        ch = stage[i]
        if q:
            cur += ch
            if ch == '\\' and q == '"' and i + 1 < len(stage): cur += stage[i + 1]; i += 1
            elif ch == q: q = None
        elif ch in '"\'': q = ch; cur += ch
        elif ch == '\\' and i + 1 < len(stage): cur += ch + stage[i + 1]; i += 1
        elif ch.isspace():
            if cur: toks.append(cur); cur = ''
        else: cur += ch
        i += 1
    if cur: toks.append(cur)
    return toks

def norm_lit(x):
    return x.strip().strip('"\'').lower()

def ltype(x):
    if re.match(r'^\$\w', x) or re.match(r'^\$\{', x): return 'var'
    if re.match(r'^\.\w+$', x): return 'ext'
    if x in ('.', '..') or x.startswith(('/', './', '~', '../')) or x.endswith('/'): return 'path'
    if re.search(r'[\*\?\[]', x): return 'glob'
    if re.match(r'^[+-]?\d+[a-zA-Z]?$', x): return 'num'
    return 'str'
COMPAT = {'var': ('var', 'path', 'str'), 'path': ('path', 'var'), 'ext': ('ext', 'glob'), 'glob': ('glob', 'ext', 'str'),
          'num': ('num',), 'str': ('str', 'glob', 'var')}

def quote_like(t, u, new):
    q = t[0] if t != u else ''
    if not q and re.search(r'[\s\*\?\[\]\(\)\|&;<>]', new) and not new.startswith('$'):
        q = "'"
    if q == "'" and "'" in new: q = '"'
    if q == '"' and '"' in new: q = "'"
    return q + new + q if q else new

def substitute2(cmd, src_desc, tgt_desc, exclude=()):
    sl = dlits(src_desc); tl = [x for x in dlits(tgt_desc) if norm_lit(x) not in exclude and '*' + norm_lit(x) not in exclude]
    if not sl or not tl: return cmd
    sn = {}
    for x in sl:
        k = norm_lit(x); sn.setdefault(k, ltype(x))
        if ltype(x) == 'ext': sn.setdefault('*' + k, 'glob_ext')
    tlt = [(x, ltype(x)) for x in tl]
    used = set(); mapping = {}
    out = []
    for st in stages(cmd):
        toks = ctoks(st); nt = toks[:1]
        for t in toks[1:]:
            u = unq(t); key = u.lower()
            if key in mapping:
                nt.append(quote_like(t, u, mapping[key])); continue
            if key in sn and not re.match(r'^-', u) or (key in sn and len(u) > 2):
                ty = sn[key]
                cty = ltype(u)
                want = COMPAT['glob'] if ty == 'glob_ext' else COMPAT[ty]
                pick = None
                for w in want:
                    for j, (x, tt) in enumerate(tlt):
                        if j not in used and tt == w and (tt in COMPAT.get(cty, ()) or (cty == 'glob' and tt == 'ext')): pick = j; break
                    if pick is not None: break
                if pick is not None:
                    x, tt = tlt[pick]; used.add(pick)
                    if tt == 'ext' and ty in ('glob_ext', 'glob'): x = '*' + x
                    mapping[key] = x
                    nt.append(quote_like(t, u, x)); continue
            nt.append(t)
        out.append(' '.join(nt))
    return ' | '.join(out)

def tsim(a, b):
    P, T = mtok(a), mtok(b); return 1 - lev(P, T) / max(len(P), len(T), 1)

def mbr(cands, ws):
    """minimum-Bayes-risk choice: candidate with max expected token similarity"""
    best = None; bs = -1
    for c in cands:
        sc = sum(w * tsim(c, o) for o, w in zip(cands, ws))
        if sc > bs: bs = sc; best = c
    return best


PATF = {'-name', '-iname', '-path', '-ipath', '-regex', '-iregex', '-wholename', '-iwholename', '-lname', '-ilname'}
NUMF = {'-perm', '-mtime', '-mmin', '-atime', '-amin', '-ctime', '-cmin', '-size', '-maxdepth', '-mindepth', '-links', '-inum', '-uid', '-gid', '-used'}
STRF = {'-user', '-group', '-newer', '-samefile', '-anewer', '-cnewer', '-fstype', '-newermt'}

def skel(stage):
    """-> (skeleton tokens, slots[(kind, raw_token, value)])"""
    toks = ctoks(stage)
    if not toks or toks[0] != 'find': return None
    sk = ['find']; slots = []
    k = 1
    while k < len(toks) and not (toks[k].startswith('-') or toks[k] in ('(', '\\(', '!')):
        t = toks[k]; u = unq(t); q = t[0] if t != u else ''
        sk.append(q + '@P' + q); slots.append(('P', u)); k += 1
    prev = None
    for t in toks[k:]:
        u = unq(t); q = t[0] if t != u else ''
        if prev in PATF:
            sk.append(q + '@G' + q); slots.append(('G', u))
        elif prev in NUMF and re.match(r'^[+-]?\d', u):
            sign = u[0] if u[0] in '+-' else ''
            m = re.match(r'^[+-]?(\d+)(\w*)$', u)
            if m:
                sk.append(q + sign + '@N' + m.group(2) + q); slots.append(('N', m.group(1)))
            else: sk.append(t)
        elif prev in STRF:
            sk.append(q + '@S' + q); slots.append(('S', u))
        else:
            sk.append(t)
        prev = t
    return sk, slots

def fill(sk, vals):
    it = iter(vals); out = []
    for t in sk:
        m = re.search(r'@([PGNS])', t)
        if m:
            v = next(it)
            if v is None: v = ''
            s = t.replace('@' + m.group(1), v, 1)
            if s == v and m.group(1) == 'G' and re.search(r'[\*\?\[]', v) and not re.search(r'["\'\\]', v): s = '"' + v + '"'
            if s.startswith("'") and "'" in v[:] and s.count("'") > 2: s = '"' + s[1:-1] + '"'
            out.append(s)
        else: out.append(t)
    return ' '.join(x for x in out if x != '')

def dtyped(desc):
    ls = dlits(desc); out = {'P': [], 'G': [], 'N': [], 'S': []}
    for x in ls:
        t = ltype(x)
        if t in ('path', 'var'): out['P'].append(x); out['S'].append(x)
        if t == 'glob': out['G'].append(x)
        if t == 'ext': out['G'].append('*' + x)
        if t == 'str': out['G'].append(x); out['S'].append(x)
        if t == 'num': out['N'].append(x.lstrip('+-'))
    return out

class FindGen:
    def __init__(s, trn, N=25, P=3.0, T=None, mode='skel', extw=True, pen=0.8, penP=1.0):
        s.pen = pen; s.penP = penP
        s.trn = trn; s.N = N; s.P = P; s.mode = mode; s.extw = extw
        s.T = T or TfIdf([feats(r['input']) for r in trn])
        s.ST = [stages(r['output']) for r in trn]
        s.SK = [skel(st[0]) for st in s.ST]
        s.isfind = [x is not None for x in s.SK]
        # which slot values were copied from the description (literal) vs implicit
        s.lit = []
        for r, sk in zip(trn, s.SK):
            if sk is None: s.lit.append(None); continue
            ls = set(norm_lit(x) for x in dlits(r['input'])) | set('*' + norm_lit(x) for x in dlits(r['input']))
            s.lit.append([v.lower() in ls for _, v in sk[1]])
        s.gP = collections.Counter(v for i, sk in enumerate(s.SK) if sk for j, (k, v) in enumerate(sk[1]) if k == 'P' and not s.lit[i][j])
        # learned: word preceding 'file(s)' -> implicit pattern
        wm = collections.defaultdict(collections.Counter); wc = collections.Counter()
        for i, r in enumerate(trn):
            if not s.isfind[i]: continue
            ws = set(m.group(1) for m in re.finditer(r'\.?([A-Za-z0-9]+) (?:files?|scripts?|images?|documents?)\b', r['input']))
            gs = [v for j, (k, v) in enumerate(s.SK[i][1]) if k == 'G' and not s.lit[i][j]]
            for w in ws:
                wc[w.lower()] += 1
                for v in gs: wm[w.lower()][v] += 1
        s.exts = set(v[2:].lower() for sk in s.SK if sk for k, v in sk[1] if k == 'G' and re.match(r'^\*\.\w+$', v))
        s.wmap = {w: c.most_common(1)[0][0] for w, c in wm.items() if c.most_common(1)[0][1] >= 2 and c.most_common(1)[0][1] / wc[w] >= 0.5}

    def __call__(s, d, sim=None):
        if sim is None: sim = s.T.sims(feats(d))
        top = [i for i in sorted(sim, key=sim.get, reverse=True) if s.isfind[i]][:s.N]
        if not top: return 'find .'
        W = [max(sim[i], 1e-3) ** s.P for i in top]
        sks = [' '.join(s.SK[i][0]) for i in top]
        # slot values
        tl = dtyped(d); used = collections.Counter()
        if s.extw and not tl['G']:
            for m in re.finditer(r'(\.?)([A-Za-z0-9]+) (?:files?|scripts?|images?|documents?)\b', d):
                w = m.group(2)
                if w.lower() in s.wmap: tl['G'].append(s.wmap[w.lower()]); break
                if w.lower() in s.exts and w.lower() not in ('all', 'the', 'regular', 'normal', 'empty', 'hidden', 'my', 'those', 'these', 'other', 'log') or m.group(1):
                    tl['G'].append('*.' + w); break
        uniq = list(dict.fromkeys(sks)); best = None; bs = -1
        for u in uniq:
            sc = sum(w * tsim(u, o) for o, w in zip(sks, W))
            ng = u.count('@G'); npth = u.count('@P')
            if ng > len(tl['G']): sc *= s.pen ** (ng - len(tl['G']))
            if npth > max(len(tl['P']), 1): sc *= s.penP ** (npth - max(len(tl['P']), 1))
            if sc > bs: bs = sc; best = u
        same = [(i, w) for i, w in zip(top, W) if ' '.join(s.SK[i][0]) == best]
        vals = []
        for j, (kind, _) in enumerate(same[0] and s.SK[same[0][0]][1]):
            if used[kind] < len(tl[kind]):
                vals.append(tl[kind][used[kind]]); used[kind] += 1; continue
            # implicit: weighted vote among same-skeleton neighbours where value was not a literal
            v = collections.Counter()
            if kind == 'P':
                for i, w in zip(top, W):
                    for jj, (kk, vv) in enumerate(s.SK[i][1]):
                        if kk == 'P' and not s.lit[i][jj] and s.gP[vv] >= 3: v[vv] += w
            for i, w in same:
                if not v and not s.lit[i][j]: v[s.SK[i][1][j][1]] += w
            if not v:
                for i, w in same: v[s.SK[i][1][j][1]] += w
            vals.append(v.most_common(1)[0][0])
        s.last_used = set(norm_lit(v) for v in vals if v)
        return fill(s.SK[same[0][0]][0], vals)

XOPT2 = ('-I', '-n', '-L', '-P', '-d', '-s', '-i', '-E', '-a')
def xinner(s):
    w = ctoks(s)[1:]; i = 0
    while i < len(w) and w[i].startswith('-'):
        i += 2 if w[i] in XOPT2 else 1
    return unq(w[i]) if i < len(w) else None

def strip_files(stage, desc):
    """drop path/glob operands (not flag values) to turn a file command into an xargs body"""
    toks = ctoks(stage); out = [toks[0]]
    lits = set(norm_lit(x) for x in dlits(desc) if ltype(x) in ('path', 'glob', 'ext', 'var'))
    for k, t in enumerate(toks[1:], 1):
        u = unq(t)
        prev = toks[k - 1]
        isop = not u.startswith('-') and (ltype(u) in ('path', 'glob') or u.lower() in lits or u.startswith('$'))
        if isop and not (prev.startswith('-') and len(prev) == 2 and prev not in ('-r', '-R', '-f', '-l', '-i', '-v', '-L', '-n', '-c', '-w', '-H', '-s', '-q')):
            continue
        out.append(t)
    return ' '.join(out)


class Model:
    def __init__(s, trn, K=15, N=25, P=1.0, KX=30, xmode='vote', share=True, lam=0.3, minh=20):
        s.minh=minh; s.lam=lam; s.trn=trn; s.K=K; s.N=N; s.P=P; s.KX=KX; s.xmode=xmode; s.share=share
        D=[feats(r['input']) for r in trn]; s.T=TfIdf(D)
        s.ST=[stages(r['output']) for r in trn]; s.H=[[head(x) for x in st] for st in s.ST]
        s.seen=set(p for h in s.H for p in zip(h,h[1:]))
        s.FG=FindGen(trn,T=s.T)
        s.tailheads=collections.Counter(h for hs in s.H for h in hs[1:])
        s.nf=collections.Counter(); s.tot=collections.Counter()
        s.by_head=collections.defaultdict(list); s.any_by_head=collections.defaultdict(list)
        s.x_by_inner=collections.defaultdict(list); s.XI=[set() for _ in trn]
        for i,st in enumerate(s.ST):
            for k,x in enumerate(st):
                h=s.H[i][k]; s.tot[h]+=1; s.any_by_head[h].append((i,k))
                if k>0: s.nf[h]+=1; s.by_head[h].append((i,k))
                if h=='xargs':
                    inn=xinner(x)
                    if inn: s.XI[i].add(inn); s.x_by_inner[inn].append((i,k))
        s.hprog=collections.Counter(h for hs in s.H for h in set(hs))
    def frag(s, cands, d, sim, excl, strip=False, prefix=''):
        cands=sorted(cands,key=lambda ik:-sim.get(ik[0],0))[:s.N]
        txt=[];ws=[]
        for i,k in cands:
            st=s.ST[i][k]
            if strip: st=strip_files(st,s.trn[i]['input'])
            txt.append(prefix+substitute2(st,s.trn[i]['input'],d,excl)); ws.append(max(sim.get(i,0),1e-3)**s.P)
        return mbr(txt,ws) if txt else None
    def __call__(s, d):
        q=feats(d); sim=s.T.sims(q); top=sorted(sim,key=sim.get,reverse=True)
        v=collections.Counter()
        for i in top[:s.K]: v[s.H[i][0]]+=sim[i]
        h0=v.most_common(1)[0][0] if v else 'find'
        if h0=='find':
            first=s.FG(d,sim); excl=s.FG.last_used if s.share else set()
        else:
            first=s.frag([(i,0) for i in top if s.H[i][0]==h0],d,sim,set()); excl=set()
        cand=set(h for h in s.tailheads if (h0,h) not in s.seen and s.hprog[h]>=s.minh)
        tail=s.tail(d,h0,sim,top,excl,cand)
        return first+' | '+tail if tail else first
    def tail(s,d,h0,sim,top,excl,cand):
        tv=collections.Counter(); n=0
        for i in top:
            hs=[h for h in set(s.H[i][1:]) if h in cand]
            if not hs: continue
            for h in hs: tv[h]+=sim[i]
            n+=1
            if n>=s.K: break
        if not tv: return None
        th=tv.most_common(1)[0][0]
        tail=None
        if th=='xargs' and s.xmode=='vote':
            xv=collections.Counter()
            for i in top[:s.KX]:
                for h in s.XI[i]: xv[h]+=sim[i]
                for h in set(s.H[i]):
                    if h in (h0,'xargs') or h in s.XI[i]: continue
                    if h in s.x_by_inner or (s.nf[h]/max(s.tot[h],1)<0.3 and s.hprog[h]>=5): xv[h]+=s.lam*sim[i]
            if xv:
                h=xv.most_common(1)[0][0]
                if s.x_by_inner.get(h): tail=s.frag(s.x_by_inner[h],d,sim,excl)
                else: tail=s.frag(s.any_by_head[h],d,sim,excl,strip=True,prefix='xargs ')
        if tail is None: tail=s.frag(s.by_head[th],d,sim,excl)
        return tail


def safe(cmd, fallback):
    cmd = cmd.replace('\x00', '').strip()
    try:
        shlex.split(cmd, posix=True)
        ok = True
    except ValueError:
        ok = False
    if not ok or not cmd or len(cmd) > 4096:
        cmd = fallback
        try: shlex.split(cmd, posix=True)
        except ValueError: cmd = 'find .'
    return cmd


def main():
    data_dir, out_path = sys.argv[1], sys.argv[2]
    trn = load(os.path.join(data_dir, 'train.csv'))
    test = load(os.path.join(data_dir, 'test.csv'))
    model = Model(trn)
    rows = []
    for r in test:
        p = model(r['input'])
        rows.append((r['id'], safe(p, stages(p)[0])))
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        w.writerow(['id', 'output'])
        w.writerows(rows)
    print('wrote', len(rows), 'rows to', out_path)


if __name__ == '__main__':
    main()
