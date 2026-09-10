"""
Пример: воспроизведение рекордной посылки C2 (public LB 0.81100) в минимальном виде.

Суть метода (честный ML, без утечки меток):
  1. Для каждого «семейного» ключа считаем сглаженный OOF target-encoding выживаемости семьи
     (family-survival, alpha=10). Ключ группировки — номер билета / фамилия (семьи делят билет).
  2. Обучаем RandomForest на базовых фичах + family-survival КАЖДОГО ключа отдельно (OOF-предсказания).
  3. Блендуем ключи cv-взвешенно: вес ключа = softmax от его OOF-CV точности (temperature=10),
     НЕ equal-weight. Итоговый порог подбираем под test_rate ~0.30.

Запуск из корня репозитория:
    python examples/c2_minimal.py
Генерирует submissions/c2_minimal.csv (идентичен cand_3keycvw.csv).
"""
import os, re
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # корень репозитория
train = pd.read_csv(os.path.join(ROOT, "train.csv"))
test  = pd.read_csv(os.path.join(ROOT, "test.csv"))
y = train["Survived"].values.astype(int)
N = len(train)
gm = y.mean()
ALPHA = 10

# ---- семейные ключи ----
def lastname(n): return str(n).split(",")[0].strip().lower()
def clean(t):   return str(t).replace(".", "").replace(" ", "").upper()
for df in (train, test):
    df["_ln"] = df["Name"].map(lastname)
    df["_tk"] = df["Ticket"].astype(str).map(clean)
KEYS = {"nm_ticket": train["_ln"]+"|"+train["_tk"],
        "ticket":   train["_tk"],
        "lastname": train["_ln"]}
TEST_KEYS = {"nm_ticket": test["_ln"]+"|"+test["_tk"],
             "ticket":   test["_tk"],
             "lastname": test["_ln"]}

# ---- базовые фичи (без famcomp/rich!) ----
def features(df):
    age = df["Age"].fillna(df["Age"].median())
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
FX, FXt = features(train), features(test)
FEATS = list(FX.columns)

# ---- сглаженный OOF family-survival по одному ключу ----
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
w = w - w.max(); w = np.exp(w * 10); w /= w.sum()   # softmax, temperature=10
print("веса ключей:", np.round(w, 3))

final = (test_stack.T * w).T.sum(0)                 # бленд
target_rate = 0.30
k = int(round(target_rate * len(final)))
thr = np.sort(final)[::-1][k - 1]
pred = (final >= thr).astype(int)
print(f"test_rate = {pred.mean():.3f}")

out = pd.DataFrame({"PassengerId": test["PassengerId"].values, "Survived": pred})
os.makedirs(os.path.join(ROOT, "submissions"), exist_ok=True)
out.to_csv(os.path.join(ROOT, "submissions", "c2_minimal.csv"), index=False)
print("записано submissions/c2_minimal.csv")
