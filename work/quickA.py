import json,numpy as np,pandas as pd,lightgbm as lgb,sys
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import solution as S
D='/home/user/PUBLIC1/data/'
tr=pd.read_csv(D+'train.csv',dtype={'case_id':str}).merge(pd.read_csv(D+'train_labels.csv',dtype={'case_id':str}),on='case_id')
cases=S.parse(tr); gold=[[int(x[1:])-1 for x in json.loads(g)] for g in tr.event_program]
cdf_raw=S.build_card_df(cases); cdf=S.apply_cats(cdf_raw,S.fit_cats(cdf_raw)); Xc=S.card_X(cdf)
drop=sys.argv[1].split(',') if len(sys.argv)>1 and sys.argv[1] else []
Xc=Xc.drop(columns=drop)
y=np.array([int(j in gold[c]) for c,j in zip(cdf._case,cdf._j)])
oof=np.zeros(len(y)); cc=cdf._case.values
for a,b in KFold(5,shuffle=True,random_state=0).split(np.arange(len(cases))):
    m=np.isin(cc,a); oof[~m]=S.fit_card(Xc[m],y[m]).predict(Xc[~m])
p=oof.reshape(-1,10); ex=sum(set(np.argsort(-p[c])[:cases[c]['L']])==set(gold[c]) for c in range(len(cases)))/len(cases)
print('auc',roc_auc_score(y,oof),'topL exact',ex)
