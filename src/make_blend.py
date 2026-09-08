import os, re, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (train.csv/test.csv live here)
keyA, keyB, a, feat, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
t0=time.time()
train=pd.read_csv(os.path.join(DIR,"train.csv")); test=pd.read_csv(os.path.join(DIR,"test.csv"))
y=train["Survived"].values.astype(int); gm=y.mean(); N=len(train)
def lastname(n): return str(n).split(",")[0].strip().lower()
def tp(t):
    m=re.match(r"\s*([A-Za-z]+)",str(t)); return m.group(1).upper() if m else "NONE"
def cl(c):
    c=str(c); return c[0].upper() if c not in ("nan","") else "U"
for df in (train,test):
    df["_ln"]=df["Name"].map(lastname); df["_tk"]=df["Ticket"].astype(str).str.replace(r"[\.\s]+","",regex=True).str.upper(); df["_tp"]=df["Ticket"].map(tp); df["_cl"]=df["Cabin"].map(cl)
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
    d_['FarePerPerson']=(fare/(fam+1e-6)).values
    d_['Mother']=((d_['Sex']==0)&(age>35)&(df['Parch']>0)).astype(int)
    d_['WomanOrChild']=(((d_['Sex']==0)&(age<35))|(age<14)).astype(int)
    d_['AgeGroup']=pd.cut(age,[-1,14,35,60,100],labels=[0,1,2,3]).astype(int)
    d_['IsChild']=(age<14).astype(int); d_['NameLen']=df['Name'].str.len().values
    return pd.DataFrame(d_)
FX=build_features(train); FXt=build_features(test)
FEATSETS={'base':['Pclass','Sex','Age','Fare_log','Embarked','SibSp','Parch','Fam','IsAlone'],
 'famcomp':['Pclass','Sex','Age','Fare_log','Embarked','SibSp','Parch','Fam','IsAlone','Mother','WomanOrChild','Title'],
 'deck':['Pclass','Sex','Age','Fare_log','Embarked','SibSp','Parch','Fam','IsAlone','Deck','Title'],
 'rich':['Pclass','Sex','Age','Fare_log','Embarked','SibSp','Parch','Fam','IsAlone','Title','Deck','FarePerPerson','Mother','WomanOrChild','AgeGroup','IsChild','NameLen'],
 'title':['Pclass','Sex','Age','Fare_log','Embarked','SibSp','Parch','Fam','IsAlone','Title']}
def group_rate_idx(idx,karr,al):
    df=pd.DataFrame({'k':karr[idx],'y':y[idx]}); g=df.groupby('k')['y'].agg(['mean','size']); return (g['mean']*g['size']+al*gm)/(g['size']+al)
cols=FEATSETS[feat]
def fam_arrays(key):
    karr=KEYS[key].values; tarr=TEST_KEYS[key].values
    oof=np.zeros(N)
    for tr,va in StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(N),y):
        rate=group_rate_idx(tr,karr,a); oof[va]=pd.Series(karr[va]).map(rate).fillna(gm).values
    rf=group_rate_idx(np.arange(N),karr,a); ft=pd.Series(tarr).map(rf).fillna(gm).values
    return oof,ft
oofA,ftA=fam_arrays(keyA); oofB,ftB=fam_arrays(keyB)
XA=np.column_stack([FX[cols].values,oofA]); XAt=np.column_stack([FXt[cols].values,ftA])
XB=np.column_stack([FX[cols].values,oofB]); XBt=np.column_stack([FXt[cols].values,ftB])
def mkinst(): return ExtraTreesClassifier(n_estimators=800,max_depth=10,min_samples_leaf=10,n_jobs=-1,random_state=0)
# OOF blend for threshold
skf=StratifiedKFold(5,shuffle=True,random_state=0); oofpA=np.zeros(N); oofpB=np.zeros(N)
for tr,va in skf.split(XA,y):
    mA=mkinst(); mA.fit(XA[tr],y[tr]); oofpA[va]=mA.predict_proba(XA[va])[:,1]
    mB=mkinst(); mB.fit(XB[tr],y[tr]); oofpB[va]=mB.predict_proba(XB[va])[:,1]
blend_oof=(oofpA+oofpB)/2
best,ba=0.5,-1
for t in np.arange(0.30,0.701,0.02):
    ac=accuracy_score(y,(blend_oof>=t).astype(int))
    if ac>ba: ba=ac; best=t
# full-fit test preds
mA=mkinst(); mA.fit(XA,y); pA=mA.predict_proba(XAt)[:,1]
mB=mkinst(); mB.fit(XB,y); pB=mB.predict_proba(XBt)[:,1]
blend_test=(pA+pB)/2
rate=float(np.mean((blend_test>=best).astype(int)))
out_df=pd.DataFrame({'PassengerId':test['PassengerId'].values,'Survived':(blend_test>=best).astype(int)})
out_df.to_csv(os.path.join(DIR,out),index=False)
print(f"{out}: blend CV={ba:.4f} rate={rate:.3f} ET({keyA})+ET({keyB}) {feat} d10l10", flush=True)
