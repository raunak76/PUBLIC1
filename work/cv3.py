import json, pickle, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
import solution as S
from metric import score_one
D='/home/user/PUBLIC1/data/'
gold=[[int(x[1:])-1 for x in json.loads(g)] for g in pd.read_csv(D+'train_labels.csv').event_program]
p10,LP,LA,SF=pickle.load(open('cv_cache.pkl','rb'))
idx=list(range(len(gold))); folds=np.zeros(len(idx),int)
for k,(a,b) in enumerate(KFold(5,shuffle=True,random_state=1).split(idx)): folds[b]=k
def run(P,N):
    tot=0
    for k in range(5):
        tri=[i for i in idx if folds[i]!=k]; vai=[i for i in idx if folds[i]==k]
        Xs,ys,gs=S.set_xy(SF,tri,gold); ms=lgb.train(P,lgb.Dataset(Xs,ys,group=gs),N)
        pr=S.decode(SF,vai,ms); tot+=sum(score_one(pr[c],gold[c]) for c in vai)
    return tot/len(idx)
B=dict(S.P_SET)
for name,upd,N in [('base',{},400),('lr.03 n800',dict(learning_rate=0.03),800),('leaves15',dict(num_leaves=15),600),('leaves63',dict(num_leaves=63,min_child_samples=100),400),('xendcg',dict(objective='rank_xendcg'),400),('binary',dict(objective='binary'),400)]:
    P=dict(B); P.update(upd); print(name,run(P,N),flush=True)
