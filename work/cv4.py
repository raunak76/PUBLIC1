import json, pickle, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
import solution as S
from metric import score_one
D='/home/user/PUBLIC1/data/'
gold=[[int(x[1:])-1 for x in json.loads(g)] for g in pd.read_csv(D+'train_labels.csv').event_program]
p10,LP,LA,SF=pickle.load(open('cv_cache.pkl','rb'))
idx=list(range(len(gold))); folds=np.zeros(len(idx),int)
for k,(a,b) in enumerate(KFold(5,shuffle=True,random_state=1).split(idx)): folds[b]=k
cfgs={'x1':(dict(objective='rank_xendcg',learning_rate=0.03,seed=0),800),'x2':(dict(objective='rank_xendcg',learning_rate=0.03,seed=1,feature_fraction=0.6),800),
 'x3':(dict(objective='rank_xendcg',learning_rate=0.03,seed=2,num_leaves=15),1000),'l1':(dict(learning_rate=0.03,seed=3),800)}
sc={k:{} for k in cfgs}
for k in range(5):
    tri=[i for i in idx if folds[i]!=k]; vai=[i for i in idx if folds[i]==k]
    Xs,ys,gs=S.set_xy(SF,tri,gold)
    for name,(u,N) in cfgs.items():
        P=dict(S.P_SET); P.update(u); m=lgb.train(P,lgb.Dataset(Xs,ys,group=gs),N)
        for c in vai: sc[name][c]=m.predict(SF[c][2])
pickle.dump(sc,open('ens_scores.pkl','wb'))
def ev(names):
    t=0
    for c in idx:
        s=sum((sc[n][c]-sc[n][c].mean())/(sc[n][c].std()+1e-9) for n in names)
        t+=score_one(SF[c][1][int(np.argmax(s))],gold[c])
    return t/len(idx)
for n in cfgs: print(n,ev([n]))
print('x123',ev(['x1','x2','x3'])); print('all',ev(list(cfgs)))
