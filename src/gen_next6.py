import os, re, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (train.csv/test.csv live here)
t0=time.time()
train=pd.read_csv(os.path.join(DIR,"train.csv")); test=pd.read_csv(os.path.join(DIR,"test.csv"))
y=train["Survived"].values.astype(int); gm=y.mean(); N=len(train)
def lastname(n): return str(n).split(",")[0].strip().lower()
def tp(t):
    m=re.match(r"\s*([A-Za-z]+)",str(t)); return m.group(1).upper() if m else "NONE"
def cl(c):
    c=str(c); return c[0].upper() if c not in ("nan","") else "U"
for df in (train,test):
    df["_ln"]=df["Name"].map(lastname)
    df["_tk"]=df["Ticket"].astype(str).str.replace(r"[\.\s]+","",regex=True).str.upper()
    df["_tp"]=df["Ticket"].map(tp); df["_cl"]=df["Cabin"].map(cl)
KEYS={'nm_ticket':train["_ln"]+"|"+train["_tk"],'ticket':train["_tk"],'lastname':train["_ln"],'ticket_prefix':train["_tp"],'cabin_letter':train["_cl"]}
TEST_KEYS={'nm_ticket':test["_ln"]+"|"+test["_tk"],'ticket':test["_tk"],'lastname':test["_ln"],'ticket_prefix':test["_tp"],'cabin_letter':test["_cl"]}
def build_features(df):
    d_={}
    age=df["Age"].fillna(df["Age"].median()); fare=df["Fare"].fillna(df["Fare"].median())
    d_['Pclass']=df['Pclass'].astype(int); d_['Sex']=(df['Sex']=='male').astype(int); d_['Age']=age.astype(float)
    d_['Fare_log']=np.log1p(fare); d_['Embarked']=df['Embarked'].fillna('S').map({'S':0,'C':1,'Q':2}).astype(int)
    d_['SibSp']=df['SibSp'].astype(int); d_['Parch']=df['Parch'].astype(int); fam=df['SibSp']+df['Parch']+1
    d_['Fam']=fam.astype(int); d_['IsAlone']=(fam==1).astype(int)
    title=df['Name'].str.extract(r",\s*([A-Za-z]+)\.")[0].str.lower()
    tm={'mr':0,'mrs':1,'miss':2,'master':3,'dr':4,'rev':4,'major':4,'col':4,'mlle':2,'ms':2,'lady':1,'countess':1,'sir':0,'capt':0,'jonkheer':0,'don':0,'mme':1,'dona':1}
    d_['Title']=title.map(lambda x:tm.get(x,4)).astype(int)
    deck=df['Cabin'].map(lambda c:str(c)[0].upper() if str(c) not in ('nan','') else 'U')
    d_['Deck']=deck.map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'U':7}).fillna(7).astype(int)
    return pd.DataFrame(d_)
FX=build_features(train); FXt=build_features(test)
FEATS={'base':['Pclass','Sex','Age','Fare_log','Embarked','SibSp','Parch','Fam','IsAlone'],
       'title':['Pclass','Sex','Age','Fare_log','Embarked','SibSp','Parch','Fam','IsAlone','Title'],
       'deck':['Pclass','Sex','Age','Fare_log','Embarked','SibSp','Parch','Fam','IsAlone','Deck','Title']}
def group_rate_idx(idx,karr,al):
    df=pd.DataFrame({'k':karr[idx],'y':y[idx]}); g=df.groupby('k')['y'].agg(['mean','size'])
    return (g['mean']*g['size']+al*gm)/(g['size']+al)
A=10
def fam_arrays(key):
    karr=KEYS[key].values; tarr=TEST_KEYS[key].values; oof=np.zeros(N)
    for tr,va in StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(N),y):
        rate=group_rate_idx(tr,karr,A); oof[va]=pd.Series(karr[va]).map(rate).fillna(gm).values
    rf=group_rate_idx(np.arange(N),karr,A); ft=pd.Series(tarr).map(rf).fillna(gm).values
    return oof,ft
def mkinst(model,leaf):
    if model=='ET': return ExtraTreesClassifier(n_estimators=800,max_depth=10,min_samples_leaf=leaf,n_jobs=-1,random_state=0)
    if model=='RF': return RandomForestClassifier(n_estimators=800,max_depth=10,min_samples_leaf=leaf,n_jobs=-1,random_state=0)
    return HistGradientBoostingClassifier(max_depth=10,learning_rate=0.1,max_iter=400,random_state=0,l2_regularization=0.0)
def build_one_key(key,model,leaf,feat):
    oof,ft=fam_arrays(key)
    X=np.column_stack([FX[FEATS[feat]].values,oof]); Xt=np.column_stack([FXt[FEATS[feat]].values,ft])
    skf=StratifiedKFold(5,shuffle=True,random_state=0); oofp=np.zeros(N)
    for tr,va in skf.split(X,y):
        m=mkinst(model,leaf); m.fit(X[tr],y[tr]); oofp[va]=m.predict_proba(X[va])[:,1]
    m=mkinst(model,leaf); m.fit(X,y); testp=m.predict_proba(Xt)[:,1]
    return oofp,testp
def write(outname, testp, target_rate=0.30):
    k=int(round(target_rate*len(testp)))
    thr=np.sort(testp)[::-1][k-1] if k>0 else 1.0
    s=(testp>=thr).astype(int)
    out=pd.DataFrame({'PassengerId':test['PassengerId'].values,'Survived':s})
    out.to_csv(os.path.join(DIR,outname),index=False)
    return s.mean(), thr

print("C1: HGB nm_ticket+ticket base l10", flush=True)
o1,t1=build_one_key('nm_ticket','HGB',10,'base'); o2,t2=build_one_key('ticket','HGB',10,'base')
r1,th1=write('cand_hgb2key.csv',(t1+t2)/2,0.30)
print(f"   rate={r1:.3f}")

print("C2: RF 3-key cvw (nm_ticket+ticket+lastname) base l10", flush=True)
oofs=[];tests=[]
for kk in ['nm_ticket','ticket','lastname']:
    op,tp_=build_one_key(kk,'RF',10,'base'); oofs.append(op); tests.append(tp_)
oofs=np.array(oofs);tests=np.array(tests)
w=np.zeros(3)
for j in range(3):
    best,ba=0.5,-1
    for tt in np.arange(0.30,0.701,0.02):
        ac=accuracy_score(y,(oofs[j]>=tt).astype(int))
        if ac>ba: ba=ac; best=tt
    w[j]=ba
w=w-w.max(); w=np.exp(w*10); w/=w.sum()
r2,th2=write('cand_3keycvw.csv',(tests.T*w).T.sum(0),0.30)
print(f"   rate={r2:.3f}")

# C3: high-rate rf2 variant (re-threshold preds[0] to rate 0.35)
preds=np.load(os.path.join(DIR,"sweep100_preds.npz"))
rf2p=preds["0"]
r3,th3=write('cand_hirate.csv',rf2p,0.35)
print(f"C3: high-rate rf2 (rate 0.35) rate={r3:.3f}")

print("C4: RF nm_ticket+ticket title l10", flush=True)
o1,t1=build_one_key('nm_ticket','RF',10,'title'); o2,t2=build_one_key('ticket','RF',10,'title')
r4,th4=write('cand_title.csv',(t1+t2)/2,0.30)
print(f"   rate={r4:.3f}")

print("C5: RF nm_ticket+ticket deck l10", flush=True)
o1,t1=build_one_key('nm_ticket','RF',10,'deck'); o2,t2=build_one_key('ticket','RF',10,'deck')
r5,th5=write('cand_deck.csv',(t1+t2)/2,0.30)
print(f"   rate={r5:.3f}")

print(f"\nRATES: C1={r1:.3f} C2={r2:.3f} C3={r3:.3f} C4={r4:.3f} C5={r5:.3f}  TOTAL {time.time()-t0:.1f}s")
