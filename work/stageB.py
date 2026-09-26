import json, time
import numpy as np, pandas as pd, lightgbm as lgb
from feats import *
from setfeats import *
D='/home/user/PUBLIC1/data/'
tr=pd.read_csv(D+'train.csv'); lab=pd.read_csv(D+'train_labels.csv')
cases=parse(tr); gold=[json.loads(x) for x in lab.event_program]
oof=np.load('oofA.npy').reshape(-1,10)
t=time.time()
Xs,ys,gs,SS=[],[],[],[]
for ci,c in enumerate(cases):
    sets,F=set_features(c,oof[ci]); g=set(int(x[1:])-1 for x in gold[ci])
    extra=np.tile([len(c['board']),c['L']],(len(sets),1))
    Xs.append(np.hstack([F,extra])); ys.append(np.array([set(s)==g for s in sets],int)); gs.append(len(sets)); SS.append(sets)
print('setfeat',time.time()-t)
from sklearn.model_selection import KFold
folds=np.zeros(len(cases),int)
for k,(a,b) in enumerate(KFold(5,shuffle=True,random_state=0).split(np.arange(len(cases)))): folds[b]=k
X=np.vstack(Xs); y=np.concatenate(ys); cid=np.repeat(np.arange(len(cases)),gs)
P=dict(objective='lambdarank',learning_rate=0.05,num_leaves=31,min_child_samples=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,seed=0,lambdarank_truncation_level=10,eval_at=[1])
oofB=np.zeros(len(X))
for k in range(5):
    tri=np.where(folds!=k)[0]; m=folds[cid]!=k
    m2=lgb.train(P,lgb.Dataset(X[m],y[m],group=[gs[i] for i in tri]),400)
    oofB[~m]=m2.predict(X[~m])
ex=f1=0; off=0
for ci,c in enumerate(cases):
    s=oofB[off:off+gs[ci]]; S=set(SS[ci][int(np.argmax(s))]); off+=gs[ci]
    g=set(int(x[1:])-1 for x in gold[ci]); ex+=S==g; f1+=len(S&g)/c['L']
print('setexact',ex/len(cases),'setF1',f1/len(cases))
imp=pd.Series(m2.feature_importance('gain'),SET_NAMES+['nv','L']).sort_values(ascending=False); print(imp.head(20))
