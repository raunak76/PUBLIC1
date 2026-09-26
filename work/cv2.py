import json, pickle, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
import solution as S
from metric import score_one
D='/home/user/PUBLIC1/data/'
tr=pd.read_csv(D+'train.csv',dtype={'case_id':str}).merge(pd.read_csv(D+'train_labels.csv',dtype={'case_id':str}),on='case_id')
gold=[[int(x[1:])-1 for x in json.loads(g)] for g in tr.event_program]
p10,LP,LA,SF=pickle.load(open('cv_cache.pkl','rb'))
idx=list(range(len(gold)))
folds=np.zeros(len(idx),int)
for k,(a,b) in enumerate(KFold(5,shuffle=True,random_state=1).split(idx)): folds[b]=k
sc={}
for k in range(5):
    tri=[i for i in idx if folds[i]!=k]; vai=[i for i in idx if folds[i]==k]
    Xs,ys,gs=S.set_xy(SF,tri,gold); ms=lgb.train(S.P_SET,lgb.Dataset(Xs,ys,group=gs),S.N_SET)
    for c in vai: sc[c]=ms.predict(SF[c][2])
pickle.dump(sc,open('setscores.pkl','wb'))
for T in [0,0.5,1,2]:
  for K in [5,15]:
    tot=0
    for c in idx:
        s=sc[c]; o=np.argsort(-s)[:K]; cand=[SF[c][1][i] for i in o]
        if T==0: pick=cand[0]
        else:
            q=np.exp((s[o]-s[o[0]])/T); q/=q.sum()
            pick=max(cand,key=lambda x: sum(qq*score_one(x,y) for qq,y in zip(q,cand)))
        tot+=score_one(pick,gold[c])
    print(T,K,tot/len(idx))
