import json,numpy as np,pandas as pd,sys,time
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import solution as S
D='/home/user/PUBLIC1/data/'
tr=pd.read_csv(D+'train.csv',dtype={'case_id':str}).merge(pd.read_csv(D+'train_labels.csv',dtype={'case_id':str}),on='case_id')
cases=S.parse(tr); gold=[[int(x[1:])-1 for x in json.loads(g)] for g in tr.event_program]
X=S.build_card_df(cases); cc=X._case.values; X=S.card_X(X)
catc=[c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
for c in catc: X[c]=X[c].astype(str)
y=np.array([int(j in gold[c]) for c,j in zip(cc,np.tile(np.arange(10),len(cases)))])
oof=np.zeros(len(y)); t=time.time()
for a,b in KFold(5,shuffle=True,random_state=0).split(np.arange(len(cases))):
    m=np.isin(cc,a)
    mdl=CatBoostClassifier(iterations=int(sys.argv[1]),learning_rate=0.05,depth=6,verbose=0,random_seed=0,thread_count=4)
    mdl.fit(X[m],y[m],cat_features=catc); oof[~m]=mdl.predict_proba(X[~m])[:,1]
p=oof.reshape(-1,10); ex=sum(set(np.argsort(-p[c])[:cases[c]['L']])==set(gold[c]) for c in range(len(cases)))/len(cases)
print('auc',roc_auc_score(y,oof),'topL exact',ex,time.time()-t)
np.save('oofCB.npy',oof)
