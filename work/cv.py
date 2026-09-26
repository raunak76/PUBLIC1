import sys, json, time, pickle
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
import solution as S
from metric import score_one, f1, edges
D='/home/user/PUBLIC1/data/'
t0=time.time()
tr=pd.read_csv(D+'train.csv',dtype={'case_id':str}).merge(pd.read_csv(D+'train_labels.csv',dtype={'case_id':str}),on='case_id')
cases=S.parse(tr); gold=[[int(x[1:])-1 for x in json.loads(g)] for g in tr.event_program]
idx=list(range(len(cases)))
cdf_raw=S.build_card_df(cases); cdf=S.apply_cats(cdf_raw,S.fit_cats(cdf_raw)); Xc=S.card_X(cdf)
y=np.array([int(j in gold[c]) for c,j in zip(cdf._case,cdf._j)])
p10,LP,LA=S.oof_stage1(cases,gold,cdf,Xc,y,idx)
print('stage1',time.time()-t0)
SF=S.build_sets(cases,idx,p10,LP,LA); print('sets',time.time()-t0)
pickle.dump((p10,LP,LA,SF),open('cv_cache.pkl','wb'))
folds=np.zeros(len(cases),int)
for k,(a,b) in enumerate(KFold(5,shuffle=True,random_state=1).split(idx)): folds[b]=k
tot=[];comp=np.zeros(3)
for k in range(5):
    tri=[i for i in idx if folds[i]!=k]; vai=[i for i in idx if folds[i]==k]
    Xs,ys,gs=S.set_xy(SF,tri,gold); ms=lgb.train(S.P_SET,lgb.Dataset(Xs,ys,group=gs),S.N_SET)
    pr=S.decode(SF,vai,ms)
    for c in vai:
        tot.append(score_one(pr[c],gold[c])); comp+= [f1(pr[c],gold[c]),f1(edges(pr[c]),edges(gold[c])),pr[c]==gold[c]]
print('CV score',np.mean(tot),'components',comp/len(idx),time.time()-t0)
