import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
from feats import *
D='/home/user/PUBLIC1/data/'
tr=pd.read_csv(D+'train.csv'); lab=pd.read_csv(D+'train_labels.csv')
assert (tr.case_id==lab.case_id).all()
cases=parse(tr); gold=[json.loads(x) for x in lab.event_program]
t=time.time(); df=build_card_df(cases); print('feat',time.time()-t, df.shape)
y=np.array([cases[c]['bank'][j]['id'] in gold[c] for c,j in zip(df._case,df._j)]).astype(int)
X,cats=to_cat(df.drop(columns=['_case','_j']))
folds=np.zeros(len(cases),int)
for k,(a,b) in enumerate(KFold(5,shuffle=True,random_state=0).split(np.arange(len(cases)))): folds[b]=k
cf=folds[df._case.values]
oof=np.zeros(len(df))
P=dict(objective='binary',learning_rate=0.03,num_leaves=31,min_child_samples=20,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1,cat_smooth=10,cat_l2=10,verbose=-1,seed=0)
for k in range(5):
    m=lgb.train(P,lgb.Dataset(X[cf!=k],y[cf!=k]),1000)
    oof[cf==k]=m.predict(X[cf==k])
from sklearn.metrics import roc_auc_score
print('auc',roc_auc_score(y,oof))
# top-L set acc
ex=0;f1=0
for ci,c in enumerate(cases):
    p=oof[df._case.values==ci]; sel=set(np.argsort(-p)[:c['L']]); g={int(x[1:])-1 for x in gold[ci]}
    ex+=sel==g; f1+=len(sel&g)/c['L']
print('setexact',ex/len(cases),'setF1',f1/len(cases))
imp=pd.Series(m.feature_importance('gain'),X.columns).sort_values(ascending=False); print(imp.head(30))
np.save('oofA.npy',oof)
