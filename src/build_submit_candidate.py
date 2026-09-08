import os, numpy as np, pandas as pd
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (train.csv/test.csv live here)
preds = np.load(os.path.join(DIR, "sweep100_preds.npz"))
test = pd.read_csv(os.path.join(DIR, "test.csv"))
base = pd.read_csv(os.path.join(DIR, "robust_rf2.csv"))["Survived"].values
anchor = preds["0"]

def make(name, prob, target_rate=0.30):
    k = int(round(target_rate*len(prob)))
    thr = np.sort(prob)[::-1][k-1] if k>0 else 1.0
    s = (prob>=thr).astype(int)
    rate = s.mean(); diff = int(np.sum(s!=base))
    print(f"{name:34s} rate={rate:.3f} diff_vs_rf2={diff:3d} thr={thr:.3f}")
    return s, rate, diff

print("=== варианты (порог -> rate 0.30) ===")
# A: solo[2]
sA,_,_ = make("A solo[2] RF nm_tk|tk famcomp", preds["2"])
# B: blend 3 RF-famcomp
B = np.mean([preds["2"], preds["10"], preds["34"]], axis=0)
sB,_,_ = make("B blend[2,10,34] RF-famcomp", B)
# C: 0.35 anchor + 0.65 famcomp_top8
fc = np.mean([preds[str(i)] for i in [2,6,10,14,18,34,38,42]], axis=0)
C = 0.35*anchor + 0.65*fc
sC,_,_ = make("C 0.35anchor+0.65famcomp8", C)

# выбор по суждению: B (robust + famcomp signal)
chosen, cname = sB, "B"
out = pd.DataFrame({"PassengerId": test["PassengerId"].values, "Survived": chosen})
out.to_csv(os.path.join(DIR, "sweep_winner.csv"), index=False)
print(f"\nCHOSEN: {cname} -> sweep_winner.csv  (rate={chosen.mean():.3f} diff_vs_rf2={int(np.sum(chosen!=base))})")
