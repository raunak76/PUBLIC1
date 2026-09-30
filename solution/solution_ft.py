#!/usr/bin/env python
"""SubjectWeave: fine-tuned cross-encoder for unseen-taxonomy subject selection.

Usage: python solution_ft.py <data_dir> <submission_csv>

Fixed deterministic plan, one NVIDIA A10G (CUDA required, no fallback path):
 1. Each (candidate subject, bill excerpts) pair is read jointly by a pretrained cross-encoder
    (cross-encoder/ms-marco-MiniLM-L-12-v2), input "[CLS] subject [SEP] bill [SEP]", max 512 tokens.
    [TOPIC] masks are shown to the model as [MASK]; [MIDDLE]/[END] markers become [SEP].
 2. Listwise fine-tuning on train.csv: the 8 candidates of one proposal are scored together and
    trained with softmax cross-entropy over the 8 slots (bf16 autocast, AdamW, linear warmup/decay).
    Nothing is tied to a subject name or slot, so it transfers to the unseen test vocabulary.
 3. Test rows are scored the same way; the argmax slot is written for every test id.

Only train.csv / test.csv and the public pretrained model are used. No IDs, row order, slot
position, label frequencies or test answers are used.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys, random, math, time
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
DEVICE = torch.device('cuda')
OC = [f'option_{k}' for k in range(8)]
MAXLEN, EPOCHS, ROWS_PER_BATCH, LR, WARMUP, SEED = 512, 2, 4, 3e-5, 0.06, 0
EVAL_ROWS = 16


def set_determinism(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def clean(t, mask):
    t = t.replace('[TOPIC]', mask).replace('[MIDDLE]', ' [SEP] ').replace('[END]', ' [SEP] ')
    return ' '.join(t.split())


def tokenize(tok, df):
    """Pre-tokenise: bill ids truncated so option + bill fit MAXLEN."""
    bills = tok([clean(t, tok.mask_token) for t in df.bill_text], add_special_tokens=False)['input_ids']
    names = sorted(set(df[OC].values.ravel()))
    nid = dict(zip(names, tok(names, add_special_tokens=False)['input_ids']))
    opts = [[nid[x] for x in r] for r in df[OC].values]
    return bills, opts


def make_batch(tok, bills, opts, rows):
    seqs = []
    for r in rows:
        for o in opts[r]:
            b = bills[r][:MAXLEN - len(o) - 3]
            seqs.append(([tok.cls_token_id] + o + [tok.sep_token_id], b + [tok.sep_token_id]))
    L = max(len(a) + len(b) for a, b in seqs)
    ii = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    am = torch.zeros_like(ii); tt = torch.zeros_like(ii)
    for i, (a, b) in enumerate(seqs):
        n = len(a) + len(b)
        ii[i, :n] = torch.tensor(a + b); am[i, :n] = 1; tt[i, len(a):n] = 1
    return ii.to(DEVICE), am.to(DEVICE), tt.to(DEVICE)


def scores(model, tok, bills, opts, rows):
    ii, am, tt = make_batch(tok, bills, opts, rows)
    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        out = model(input_ids=ii, attention_mask=am, token_type_ids=tt).logits[:, 0]
    return out.float().view(len(rows), 8)


def main(data_dir, out_path, max_train_rows=None):
    if DEVICE.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('This solution is specified for one NVIDIA A10G GPU; CUDA is not available.')
    set_determinism(SEED)
    tr = pd.read_csv(os.path.join(data_dir, 'train.csv')); te = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    if max_train_rows:
        tr = tr.iloc[:max_train_rows]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, attn_implementation='eager').to(DEVICE)
    trb, tro = tokenize(tok, tr); teb, teo = tokenize(tok, te)
    y = torch.from_numpy(tr.target.values.astype(np.int64)).to(DEVICE)

    n = len(tr); steps = EPOCHS * math.ceil(n / ROWS_PER_BATCH); warm = int(WARMUP * steps)
    nd = ['bias', 'LayerNorm.weight']
    groups = [{'params': [p for k, p in model.named_parameters() if not any(x in k for x in nd)], 'weight_decay': 0.01},
              {'params': [p for k, p in model.named_parameters() if any(x in k for x in nd)], 'weight_decay': 0.0}]
    opt = torch.optim.AdamW(groups, lr=LR)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (steps - s) / max(1, steps - warm)))
    g = torch.Generator().manual_seed(SEED); step = 0; t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(n, generator=g).tolist(); run = 0.0
        for i in range(0, n, ROWS_PER_BATCH):
            rows = perm[i:i + ROWS_PER_BATCH]
            loss = F.cross_entropy(scores(model, tok, trb, tro, rows), y[rows])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            step += 1; run += loss.item()
            if step % 500 == 0:
                print(f'epoch {ep} step {step}/{steps} loss {run / 500:.4f} {time.time() - t0:.0f}s', flush=True); run = 0.0

    model.eval(); Z = []
    with torch.no_grad():
        for i in range(0, len(te), EVAL_ROWS):
            Z.append(scores(model, tok, teb, teo, list(range(i, min(i + EVAL_ROWS, len(te))))).cpu().numpy())
    Z = np.concatenate(Z)
    sub = pd.DataFrame({'id': te.id, 'prediction': Z.argmax(1).astype(int)})
    sub.to_csv(out_path, index=False)
    print('wrote', out_path, sub.prediction.value_counts().sort_index().tolist(), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
