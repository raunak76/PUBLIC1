import json, pickle, sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
import solution as S
from metric import score_one
D='/home/user/PUBLIC1/data/'
tr=pd.read_csv(D+'train.csv',dtype={'case_id':str}).merge(pd.read_csv(D+'train_labels.csv',dtype={'case_id':str}),on='case_id')
cases=S.parse(tr); gold=[[int(x[1:])-1 for x in json.loads(g)] for g in tr.event_program]
p10,LP,LA,_=pickle.load(open('cv_cache.pkl','rb'))
cb=np.load('oofCB.npy').reshape(-1,10)
idx=list(range(len(gold))); folds=np.zeros(len(idx),int)
for k,(a,b) in enumerate(KFold(5,shuffle=True,random_state=1).split(idx)): folds[b]=k
P=dict(S.P_SET); P.update(objective='rank_xendcg',learning_rate=0.03)
for name,p in [('cb',cb),('blend',0.5*cb+0.5*p10)]:
    SF=S.build_sets(cases,idx,p,LP,LA); t=0
    for k in range(5):
        tri=[i for i in idx if folds[i]!=k]; vai=[i for i in idx if folds[i]==k]
        Xs,ys,gs=S.set_xy(SF,tri,gold); m=lgb.train(P,lgb.Dataset(Xs,ys,group=gs),800)
        pr=S.decode(SF,vai,m); t+=sum(score_one(pr[c],gold[c]) for c in vai)
    print(name,t/len(idx),flush=True)
