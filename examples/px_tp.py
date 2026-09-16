"""
Пример: воспроизведение рекордной посылки PX_TP (public LB 0.82057) в минимальном виде.

Отличие от n3_ageimp.py: добавлен КОМПЛЕМЕНТАРНЫЙ сигнал family-survival по ПРЕФИКСУ БИЛЕТА
(fs_tp) как дополнительная базовая фича ко всем 3 ключевым моделям. Это и есть принципиально
иная ось, давшая честный прирост +2 строки поверх N3 (0.81578 -> 0.82057). Всё остальное — движок N3:
   1. сглаженный OOF family-survival (alpha=10) по ключам nm_ticket/ticket/lastname;
   2. RandomForest (depth=10, leaf=10, 800 деревьев) на base-фичах + family-survival;
   3. cv-взвешенный бленд ключей (softmax по OOF-CV, T=10); test_rate 0.30;
   4. Age импутирована медианой Title x Pclass (как в N3).

Запуск из корня репозитория:
    python examples/px_tp.py
Генерирует submissions/px_tp.csv (идентичен att_px_tp.csv, diff=0).
"""
import os, re
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
train = pd.read_csv(os.path.join(ROOT, "train.csv"))
test  = pd.read_csv(os.path.join(ROOT, "test.csv"))
y = train["Survived"].values.astype(int)
N = len(train)
gm = y.mean()
ALPHA = 10

# ---- семейные ключи ----
def lastname(n): return str(n).split(",")[0].strip().lower()
def clean(t):   return str(t).replace(".", "").replace(" ", "").upper()
def tprefix(t):
    m = re.match(r"\s*([A-Za-z]+)", str(t))
    return m.group(1).upper() if m else "NONE"
for df in (train, test):
    df["_ln"] = df["Name"].map(lastname)
    df["_tk"] = df["Ticket"].astype(str).map(clean)
    df["_tp"] = df["Ticket"].astype(str).map(tprefix)
KEYS = {"nm_ticket": train["_ln"]+"|"+train["_tk"],
        "ticket":   train["_tk"],
        "lastname": train["_ln"],
        "ticket_prefix": train["_tp"]}
TEST_KEYS = {"nm_ticket": test["_ln"]+"|"+test["_tk"],
             "ticket":   test["_tk"],
             "lastname": test["_ln"],
             "ticket_prefix": test["_tp"]}

# ---- базовые фичи: Age импутируем медианой по Title x Pclass (честно, по train) ----
def title_of(df):
    return df["Name"].str.extract(r",\s*([A-Za-z]+)\.")[0].str.lower()
tr_title, te_title = title_of(train), title_of(test)
age_med = train.groupby([tr_title, train["Pclass"]])["Age"].median().to_dict()
age_global = train["Age"].median()
def impute_age(df, title):
    key = list(zip(title, df["Pclass"]))
    return df["Age"].fillna(pd.Series([age_med.get((a, b), age_global) for a, b in key],
                                      index=df.index))

def features(df, title):
    age = impute_age(df, title)
    fare = df["Fare"].fillna(df["Fare"].median())
    return pd.DataFrame({
        "Pclass": df["Pclass"].astype(int),
        "Sex": (df["Sex"] == "male").astype(int),
        "Age": age.astype(float),
        "Fare_log": np.log1p(fare),
        "Embarked": df["Embarked"].fillna("S").map({"S":0,"C":1,"Q":2}).astype(int),
        "SibSp": df["SibSp"].astype(int),
        "Parch": df["Parch"].astype(int),
        "Fam": (df["SibSp"]+df["Parch"]+1).astype(int),
        "IsAlone": ((df["SibSp"]+df["Parch"]+1) == 1).astype(int),
    })
FX, FXt = features(train, tr_title), features(test, te_title)
FEATS = list(FX.columns)

# ---- комплементарный family-survival по ПРЕФИКСУ БИЛЕТА (OOF, честно) ----
def fam_arrays(key):
    karr = KEYS[key].values; tarr = TEST_KEYS[key].values; oof = np.zeros(N)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(N), y):
        g = pd.DataFrame({"k": karr[tr], "y": y[tr]}).groupby("k")["y"].agg(["mean","size"])
        rate = (g["mean"]*g["size"] + ALPHA*gm) / (g["size"] + ALPHA)
        oof[va] = pd.Series(karr[va]).map(rate).fillna(gm).values
    g = pd.DataFrame({"k": karr, "y": y}).groupby("k")["y"].agg(["mean","size"])
    rate = (g["mean"]*g["size"] + ALPHA*gm) / (g["size"] + ALPHA)
    ft = pd.Series(tarr).map(rate).fillna(gm).values
    return oof, ft

oof_tp, ft_tp = fam_arrays("ticket_prefix")
FX["fs_tp"] = oof_tp.astype(float)
FXt["fs_tp"] = ft_tp.astype(float)
FEATS = list(FX.columns)   # теперь включает fs_tp

# ---- одна модель-ключ: OOF + тест-вероятности ----
def build_key(key):
    oof, ft = fam_arrays(key)
    X  = np.column_stack([FX[FEATS].values, oof])
    Xt = np.column_stack([FXt[FEATS].values, ft])
    skf = StratifiedKFold(5, shuffle=True, random_state=0); oofp = np.zeros(N)
    for tr, va in skf.split(X, y):
        m = RandomForestClassifier(n_estimators=800, max_depth=10, min_samples_leaf=10,
                                   n_jobs=-1, random_state=0)
        m.fit(X[tr], y[tr]); oofp[va] = m.predict_proba(X[va])[:, 1]
    m = RandomForestClassifier(n_estimators=800, max_depth=10, min_samples_leaf=10,
                               n_jobs=-1, random_state=0)
    m.fit(X, y); testp = m.predict_proba(Xt)[:, 1]
    return oofp, testp

# ---- cv-взвешивание (softmax по OOF-CV точности) ----
keys = ["nm_ticket", "ticket", "lastname"]
oof_stack, test_stack = [], []
for k in keys:
    o, t = build_key(k); oof_stack.append(o); test_stack.append(t)
oof_stack, test_stack = np.array(oof_stack), np.array(test_stack)

w = np.zeros(3)
for j in range(3):
    best, ba = 0.5, -1
    for th in np.arange(0.30, 0.701, 0.02):
        ac = accuracy_score(y, (oof_stack[j] >= th).astype(int))
        if ac > ba: ba, best = ac, th
    w[j] = ba
w = w - w.max(); w = np.exp(w * 10); w /= w.sum()
print("веса ключей:", np.round(w, 3))

final = (test_stack.T * w).T.sum(0)
target_rate = 0.30
k = int(round(target_rate * len(final)))
thr = np.sort(final)[::-1][k - 1]
pred = (final >= thr).astype(int)
print(f"test_rate = {pred.mean():.3f}")

out = pd.DataFrame({"PassengerId": test["PassengerId"].values, "Survived": pred})
os.makedirs(os.path.join(ROOT, "submissions"), exist_ok=True)
out.to_csv(os.path.join(ROOT, "submissions", "px_tp.csv"), index=False)
print("записано submissions/px_tp.csv")

# сверка с эталоном att_px_tp.csv (если есть в submissions/)
ref_path = os.path.join(ROOT, "submissions", "att_px_tp.csv")
if os.path.exists(ref_path):
    ref = pd.read_csv(ref_path)["Survived"].values.astype(int)
    print("diff vs att_px_tp.csv:", int((pred != ref).sum()), "(ожидается 0)")
