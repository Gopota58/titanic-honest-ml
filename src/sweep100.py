import os, re, sys, time, warnings
import numpy as np, pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (train.csv/test.csv live here)
t0 = time.time()
train = pd.read_csv(os.path.join(DIR, "train.csv")); test = pd.read_csv(os.path.join(DIR, "test.csv"))
y = train["Survived"].values.astype(int); gm = y.mean(); N = len(train)
def lastname(n): return str(n).split(",")[0].strip().lower()
def tp(t):
    m = re.match(r"\s*([A-Za-z]+)", str(t)); return m.group(1).upper() if m else "NONE"
def cl(c):
    c = str(c); return c[0].upper() if c not in ("nan","") else "U"
for df in (train, test):
    df["_ln"] = df["Name"].map(lastname)
    df["_tk"] = df["Ticket"].astype(str).str.replace(r"[\.\s]+","",regex=True).str.upper()
    df["_tp"] = df["Ticket"].map(tp); df["_cl"] = df["Cabin"].map(cl)
KEYS      = {'nm_ticket':train["_ln"]+"|"+train["_tk"],'ticket':train["_tk"],'lastname':train["_ln"],'ticket_prefix':train["_tp"],'cabin_letter':train["_cl"]}
TEST_KEYS = {'nm_ticket':test["_ln"]+"|"+test["_tk"],'ticket':test["_tk"],'lastname':test["_ln"],'ticket_prefix':test["_tp"],'cabin_letter':test["_cl"]}
def build_features(df):
    d_ = {}
    age = df["Age"].fillna(df["Age"].median()); fare = df["Fare"].fillna(df["Fare"].median())
    d_['Pclass']=df['Pclass'].astype(int); d_['Sex']=(df['Sex']=='male').astype(int); d_['Age']=age.astype(float)
    d_['Fare_log']=np.log1p(fare); d_['Embarked']=df['Embarked'].fillna('S').map({'S':0,'C':1,'Q':2}).astype(int)
    d_['SibSp']=df['SibSp'].astype(int); d_['Parch']=df['Parch'].astype(int); fam=df['SibSp']+df['Parch']+1
    d_['Fam']=fam.astype(int); d_['IsAlone']=(fam==1).astype(int)
    title=df['Name'].str.extract(r",\s*([A-Za-z]+)\.")[0].str.lower()
    tm={'mr':0,'mrs':1,'miss':2,'master':3,'dr':4,'rev':4,'major':4,'col':4,'mlle':2,'ms':2,'lady':1,'countess':1,'sir':0,'capt':0,'jonkheer':0,'don':0,'mme':1,'dona':1}
    d_['Title']=title.map(lambda x:tm.get(x,4)).astype(int)
    deck=df['Cabin'].map(lambda c:str(c)[0].upper() if str(c) not in ('nan','') else 'U')
    d_['Deck']=deck.map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'U':7}).fillna(7).astype(int)
    d_['FarePerPerson']=(fare/(fam+1e-6)).values
    d_['Mother']=((d_['Sex']==0)&(age>35)&(df['Parch']>0)).astype(int)
    d_['WomanOrChild']=(((d_['Sex']==0)&(age<35))|(age<14)).astype(int)
    d_['AgeGroup']=pd.cut(age,[-1,14,35,60,100],labels=[0,1,2,3]).astype(int)
    d_['IsChild']=(age<14).astype(int); d_['NameLen']=df['Name'].str.len().values
    return pd.DataFrame(d_)
FX = build_features(train); FXt = build_features(test)
FEATS = {'base':['Pclass','Sex','Age','Fare_log','Embarked','SibSp','Parch','Fam','IsAlone'],
         'famcomp':['Pclass','Sex','Age','Fare_log','Embarked','SibSp','Parch','Fam','IsAlone','Mother','WomanOrChild','Title'],
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

# ---------- конфигурации (~100) ----------
keys5=['nm_ticket','ticket','lastname','ticket_prefix','cabin_letter']
two=[list(c) for c in combinations(keys5,2)]
three=[list(c) for c in combinations(keys5,3)]
configs=[]
for combo in two:
    for model in ['RF','ET']:
        for feat in ['base','famcomp']:
            for leaf in [10,20]:
                configs.append((combo,model,feat,leaf,'equal'))
for combo in two[:5]:
    for model in ['RF','ET']:
        configs.append((combo,model,'base',10,'cvw'))
for combo in two[:5]:
    configs.append((combo,'HGB','base',10,'equal'))
for combo in three[:5]:
    configs.append((combo,'RF','base',10,'cvw'))
for k in ['nm_ticket','ticket','lastname']:
    for model in ['RF','ET']:
        configs.append(([k],model,'base',10,'equal'))
configs=configs[:100]
print(f"TOTAL CONFIGS: {len(configs)}", flush=True)

base=pd.read_csv(os.path.join(DIR,"robust_rf2.csv"))['Survived'].values
rows=[]; preds={}
for i,(combo,model,feat,leaf,wt) in enumerate(configs):
    t1=time.time()
    oofs=[]; tests=[]
    for k in combo:
        op,tp_=build_one_key(k,model,leaf,feat); oofs.append(op); tests.append(tp_)
    oofs=np.array(oofs); tests=np.array(tests)
    if wt=='cvw':
        w=np.zeros(len(combo))
        for j in range(len(combo)):
            best,ba=0.5,-1
            for t in np.arange(0.30,0.701,0.02):
                ac=accuracy_score(y,(oofs[j]>=t).astype(int))
                if ac>ba: ba=ac; best=t
            w[j]=ba
        w=w-w.max(); w=np.exp(w*10); w/=w.sum()
    else:
        w=np.ones(len(combo))/len(combo)
    bo=(oofs.T*w).T.sum(0); bt=(tests.T*w).T.sum(0)
    best,ba=0.5,-1
    for t in np.arange(0.30,0.701,0.02):
        ac=accuracy_score(y,(bo>=t).astype(int))
        if ac>ba: ba=ac; best=t
    rate=float(np.mean((bt>=best).astype(int)))
    preds[i]=bt
    mine=(bt>=best).astype(int)
    diff=int(np.sum(mine!=base))
    cols="|".join(combo)
    rows.append(dict(idx=i,model=model,keys=cols,feat=feat,leaf=leaf,wt=wt,cv=round(ba,4),rate=round(rate,3),diff=diff))
    print(f"[{i:03d}] {model} {cols[:28]:28s} {feat} l{leaf} {wt:4s} cv={ba:.4f} rate={rate:.3f} diff={diff} t={time.time()-t1:.1f}s", flush=True)

mdf=pd.DataFrame(rows)
mdf.to_csv(os.path.join(DIR,"sweep100_metrics.csv"),index=False)
np.savez_compressed(os.path.join(DIR,"sweep100_preds.npz"), **{str(k):v for k,v in preds.items()})
# саммари
healthy=mdf[(mdf.rate>=0.28)&(mdf.rate<=0.44)].copy()
print("\n=== TOP 15 HEALTHY by CV ===")
for _,r in healthy.sort_values('cv',ascending=False).head(15).iterrows():
    print(f"[{int(r.idx):03d}] cv={r.cv:.4f} rate={r.rate:.3f} diff={int(r.diff)} {r.model} {r.keys} {r.feat} l{int(r.leaf)} {r.wt}")
print(f"\nhealthy count={len(healthy)} / {len(mdf)}  TOTAL {time.time()-t0:.1f}s")
