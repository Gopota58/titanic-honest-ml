import os, re, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (train.csv/test.csv live here)
t0 = time.time()

train = pd.read_csv(os.path.join(DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DIR, "test.csv"))
y = train["Survived"].values.astype(int)
gm = y.mean(); N = len(train)

def lastname(n): return str(n).split(",")[0].strip().lower()
def tp(t):
    m = re.match(r"\s*([A-Za-z]+)", str(t)); return m.group(1).upper() if m else "NONE"
def cl(c):
    c = str(c); return c[0].upper() if c not in ("nan","") else "U"
for df in (train, test):
    df["_ln"] = df["Name"].map(lastname)
    df["_tk"] = df["Ticket"].astype(str).str.replace(r"[\.\s]+","",regex=True).str.upper()
    df["_tp"] = df["Ticket"].map(tp)
    df["_cl"] = df["Cabin"].map(cl)
KEYS      = {'nm_ticket':train["_ln"]+"|"+train["_tk"],'ticket':train["_tk"],'lastname':train["_ln"],'ticket_prefix':train["_tp"],'cabin_letter':train["_cl"]}
TEST_KEYS = {'nm_ticket':test["_ln"]+"|"+test["_tk"],'ticket':test["_tk"],'lastname':test["_ln"],'ticket_prefix':test["_tp"],'cabin_letter':test["_cl"]}

def build_features(df):
    d_ = {}
    age = df["Age"].fillna(df["Age"].median()); fare = df["Fare"].fillna(df["Fare"].median())
    d_['Pclass'] = df['Pclass'].astype(int); d_['Sex'] = (df['Sex']=='male').astype(int); d_['Age'] = age.astype(float)
    d_['Fare_log'] = np.log1p(fare); d_['Embarked'] = df['Embarked'].fillna('S').map({'S':0,'C':1,'Q':2}).astype(int)
    d_['SibSp'] = df['SibSp'].astype(int); d_['Parch'] = df['Parch'].astype(int); fam = df['SibSp']+df['Parch']+1
    d_['Fam'] = fam.astype(int); d_['IsAlone'] = (fam==1).astype(int)
    title = df['Name'].str.extract(r",\s*([A-Za-z]+)\.")[0].str.lower()
    tm = {'mr':0,'mrs':1,'miss':2,'master':3,'dr':4,'rev':4,'major':4,'col':4,'mlle':2,'ms':2,'lady':1,'countess':1,'sir':0,'capt':0,'jonkheer':0,'don':0,'mme':1,'dona':1}
    d_['Title'] = title.map(lambda x: tm.get(x,4)).astype(int)
    deck = df['Cabin'].map(lambda c: str(c)[0].upper() if str(c) not in ('nan','') else 'U')
    d_['Deck'] = deck.map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'U':7}).fillna(7).astype(int)
    d_['FarePerPerson'] = (fare/(fam+1e-6)).values
    d_['Mother'] = ((d_['Sex']==0)&(age>35)&(df['Parch']>0)).astype(int)
    d_['WomanOrChild'] = (((d_['Sex']==0)&(age<35))|(age<14)).astype(int)
    d_['AgeGroup'] = pd.cut(age,[-1,14,35,60,100],labels=[0,1,2,3]).astype(int)
    d_['IsChild'] = (age<14).astype(int); d_['NameLen'] = df['Name'].str.len().values
    return pd.DataFrame(d_)

FX = build_features(train); FXt = build_features(test)
FEAT = ['Pclass','Sex','Age','Fare_log','Embarked','SibSp','Parch','Fam','IsAlone']  # 'base'

def group_rate_idx(idx, karr, al):
    df = pd.DataFrame({'k':karr[idx],'y':y[idx]})
    g = df.groupby('k')['y'].agg(['mean','size'])
    return (g['mean']*g['size'] + al*gm)/(g['size']+al)

def fam_arrays(key, a):
    karr = KEYS[key].values; tarr = TEST_KEYS[key].values
    oof = np.zeros(N)
    for tr,va in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(N), y):
        rate = group_rate_idx(tr, karr, a)
        oof[va] = pd.Series(karr[va]).map(rate).fillna(gm).values
    rf = group_rate_idx(np.arange(N), karr, a)
    ft = pd.Series(tarr).map(rf).fillna(gm).values
    return oof, ft

def mkinst(model):
    if model == 'ET':
        return ExtraTreesClassifier(n_estimators=800, max_depth=10, min_samples_leaf=10, n_jobs=-1, random_state=0)
    return RandomForestClassifier(n_estimators=800, max_depth=10, min_samples_leaf=10, n_jobs=-1, random_state=0)

def build_one_key(key, model, a):
    """Возвращает (oof_probs N, test_probs 418) для одного ключа."""
    oof, ft = fam_arrays(key, a)
    X  = np.column_stack([FX[FEAT].values,  oof])
    Xt = np.column_stack([FXt[FEAT].values, ft])
    # OOF предсказания (для подбора порога бленда)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    oofp = np.zeros(N)
    for tr,va in skf.split(X, y):
        m = mkinst(model); m.fit(X[tr], y[tr]); oofp[va] = m.predict_proba(X[va])[:,1]
    # полный фит -> тест
    m = mkinst(model); m.fit(X, y)
    testp = m.predict_proba(Xt)[:,1]
    return oofp, testp

# ---- конфигурации: (имя, [ключи], модель) ----
CONFIGS = [
    ("robust_3a", ["nm_ticket","ticket","ticket_prefix"], "ET"),
    ("robust_3b", ["nm_ticket","ticket","cabin_letter"],  "ET"),
    ("robust_3c", ["nm_ticket","ticket","lastname"],      "ET"),
    ("robust_4",  ["nm_ticket","ticket","ticket_prefix","cabin_letter"], "ET"),
    ("robust_5",  ["nm_ticket","ticket","lastname","ticket_prefix","cabin_letter"], "ET"),
    ("robust_rf2",["nm_ticket","ticket"], "RF"),
]

A = 10
results = []
for name, keys, model in CONFIGS:
    t1 = time.time()
    oof_list, test_list = [], []
    for k in keys:
        op, tp_ = build_one_key(k, model, A)
        oof_list.append(op); test_list.append(tp_)
    blend_oof = np.mean(oof_list, axis=0)
    blend_test = np.mean(test_list, axis=0)
    # подбор порога по OOF-CV
    best, ba = 0.5, -1
    for t in np.arange(0.30, 0.701, 0.02):
        ac = accuracy_score(y, (blend_oof >= t).astype(int))
        if ac > ba: ba = ac; best = t
    rate = float(np.mean((blend_test >= best).astype(int)))
    out = pd.DataFrame({'PassengerId': test['PassengerId'].values,
                        'Survived': (blend_test >= best).astype(int)})
    out.to_csv(os.path.join(DIR, name + ".csv"), index=False)
    # разнообразие vs iter5 (эталон)
    base = pd.read_csv(os.path.join(DIR, "iter5_et_blend.csv"))['Survived'].values
    mine = out['Survived'].values
    diff = int(np.sum(mine != base))
    results.append((name, len(keys), model, round(ba,4), round(rate,3), diff, round(time.time()-t1,1)))
    print(f"{name}: keys={len(keys)} {model} cv={ba:.4f} rate={rate:.3f} diff_vs_iter5={diff} t={time.time()-t1:.1f}s", flush=True)

print("\n=== SUMMARY (sorted by cv) ===")
for r in sorted(results, key=lambda x:-x[3]):
    print(f"{r[0]:12s} k={r[1]} {r[2]:3s} cv={r[3]:.4f} rate={r[4]:.3f} diff={r[5]} t={r[6]}s")
print(f"TOTAL {time.time()-t0:.1f}s")
