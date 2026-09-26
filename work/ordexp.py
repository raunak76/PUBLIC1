import json, itertools
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
from feats import *
from order import *
from metric import *
D='/home/user/PUBLIC1/data/'
tr=pd.read_csv(D+'train.csv'); lab=pd.read_csv(D+'train_labels.csv')
cases=parse(tr); gold=[[int(x[1:])-1 for x in json.loads(g)] for g in lab.event_program]
cdf=build_card_df(cases); cdfc,cats=to_cat(cdf)
folds=np.zeros(len(cases),int)
for k,(a,b) in enumerate(KFold(5,shuffle=True,random_state=0).split(np.arange(len(cases)))): folds[b]=k
pw,pwy,ad,ady=[],[],[],[]
for c,g in enumerate(gold):
    for a in range(len(g)):
        for b in range(len(g)):
            if a!=b: pw.append((c,g[a],g[b])); pwy.append(int(a<b))
    ext=[-1]+g+[-1]
    for a in [-1]+g:
        for b in g+[-1]:
            if a==b or (a==-1 and b==-1): continue
            ad.append((c,a,b)); ia=0 if a==-1 else g.index(a)+1; ib=len(g)+1 if b==-1 else g.index(b)+1; ady.append(int(ib==ia+1))
PW=pair_frame(cdfc,cases,pw); AD=pair_frame(cdfc,cases,ad); pwy=np.array(pwy); ady=np.array(ady)
pwc=folds[[x[0] for x in pw]]; adc=folds[[x[0] for x in ad]]
P=dict(objective='binary',learning_rate=0.05,num_leaves=31,min_child_samples=20,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,seed=0,cat_smooth=10)
opw=np.zeros(len(pw)); oad=np.zeros(len(ad))
for k in range(5):
    m=lgb.train(P,lgb.Dataset(PW[pwc!=k],pwy[pwc!=k]),500); opw[pwc==k]=m.predict(PW[pwc==k])
    m=lgb.train(P,lgb.Dataset(AD[adc!=k],ady[adc!=k]),500); oad[adc==k]=m.predict(AD[adc==k])
dp={x:v for x,v in zip(pw,opw)}; da={x:v for x,v in zip(ad,oad)}
for w in [0,0.5,1,2,1e6]:
    sc=ex=0
    for c,g in enumerate(gold):
        best=None
        for perm in itertools.permutations(g):
            s=sum(np.log(dp[(c,perm[a],perm[b])]) for a in range(len(g)) for b in range(a+1,len(g)))*(1 if w<1e5 else 0)
            ext=[-1]+list(perm)+[-1]
            s+=w*sum(np.log(da[(c,ext[t],ext[t+1])]) for t in range(len(ext)-1)) if w<1e5 else sum(np.log(da[(c,ext[t],ext[t+1])]) for t in range(len(ext)-1))
            if best is None or s>best[0]: best=(s,perm)
        sc+=score_one(list(best[1]),g); ex+=list(best[1])==g
    print(w,'order-given-gold score',sc/len(gold),'exact',ex/len(gold))
