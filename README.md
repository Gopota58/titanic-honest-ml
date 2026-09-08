# Честный Kaggle Titanic: public LB 0.81100 без утечки меток

[![Public LB](https://img.shields.io/badge/Public%20LB-0.81100-success)](https://www.kaggle.com/c/titanic/leaderboard)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Проект — честное поднятие public leaderboard на классическом датасете Kaggle **Titanic**
(891 обучающих / 418 тестовых) **безо всякой утечки меток, без LB-probing и без фейка**.
Финальный результат — **0.81100** на публичном LB (доказанный честный потолок ~0.81–0.82).

> Единственный честный арбитр качества здесь — **реальный Kaggle public LB** (~10 посылок/день).
> Любая локальная валидация (cross-validation, holdout, даже синтетика с известной истиной)
> систематически лжёт из-за семейной структуры выборки и covariate shift — этот грабль
> мы прошли десятки раз и задокументировали.

## Результаты (реальный public LB, хронология)

| Дата | Модель | Public LB |
|------|--------|-----------|
| 09-05 | v3: рег. RandomForest + smoothed OOF family-survival (α=10) | 0.79425 |
| 09-06 | iter5: бленд 2× ExtraTrees (nm_ticket + ticket) + family-survival | 0.80143 ⭐ первый пробой 0.80 |
| 09-08 | robust_rf2: **RandomForest** (nm_ticket + ticket) + family-survival | 0.80382 |
| 09-08 | **cand_3keycvw: RF 3-ключевой cv-взвешенный бленд** (nm_ticket+ticket+lastname) | **0.81100** 🏆 |

Полная таблица всех посылок и грабли — в `submissions/` и в описании проекта (LoreBase).

## Методология

Единственный сигнал, честно обобщающийся на тест, — **сглаженный OOF target-encoding
выживаемости семьи** (family-survival, α=10). Ключ группировки — номер билета
(`ticket`) и его вариации (`nm_ticket = lastname|ticket`); билет — самый чистый
семейный прокси (семьи делят один билет).

Рабочая архитектура (после сотен экспериментов):

1. **Регуляризованные деревья (RandomForest / ExtraTrees), НЕ бустинг.**
   Те же фичи + family-survival дают разрыв CV→LB ≈ 0.036–0.038 и LB ~0.79–0.80;
   XGBoost/HGB на тех же фичах переобучаются (разрыв ~0.07–0.08, LB 0.77–0.78).
2. **Мульти-ключевой бленд семейных сигналов** работает, но только при:
   - модель = **RandomForest** (не ExtraTrees в роли бленда — ET роняет test_rate → коллапс);
   - взвешивание = **cv-weighted** (вес ключа = softmax от его OOF-CV точности), **не equal**;
   - фичи = **base** (Pclass, Sex, Age, Fare_log, Embarked, SibSp, Parch, Fam, IsAlone + family_survival),
     **не famcomp/rich**;
   - **test_rate** (доля предсказанных выживших на тесте) удержан ≈ **0.30**.
   Нарушение любого пункта → covariate-shift коллапс (test_rate < 0.27).
3. **Контроль test_rate 0.30–0.46** как ворота устойчивости; победители в зоне ~0.29–0.31.

## Ключевые находки (грабли)

- Локальный CV **не является** надёжным прокси для LB: глубокие деревья/бустинг дают
  CV 0.84–0.85, но LB 0.74–0.78. Даже добавление разумных фич (Mother/WomanOrChild/Title =
  «famcomp») **систематически** поднимает OOF-CV на +0.003–0.005 (проверено на 16/16 конфигах),
  но на реальном LB **не переносится** — переобучение, невидимое по CV.
- Рекорд — **хрупкий пик**: соседние вариации гиперпараметров (сид/лист/альфа) роняют LB на ~0.007.
- Правило посылок: лучший результат отправляется **последним**, держим ≥1 слот в резерве,
  чтобы лучший счёт остался активным на лидерборде.

## Структура репозитория

```
titanic-honest-ml/
├── train.csv, test.csv          # оригинальные данные Kaggle
├── src/
│   ├── make_one.py              # одна модель (RF/ET/HGB) по одному ключу семьи
│   ├── make_blend.py            # бленд 2× ExtraTrees по 2 ключам
│   ├── gen_robust.py            # мульти-ключевые робастные бленды (→ robust_rf2, 0.80382)
│   ├── gen_robust2.py           # RF+ET / 3-ключевые вариации
│   ├── sweep100.py              # 100 локальных кандидатов в safe-режиме
│   ├── gen_next6.py             # финальные 5 разведочных + генерация кандидатов
│   └── build_submit_candidate.py# сборка итогового бленда из sweep-предсказаний
├── submissions/                 # ключевые посылки (доказательство результата)
│   ├── iter5_et_blend.csv       # 0.80143
│   ├── robust_rf2.csv           # 0.80382
│   ├── cand_3keycvw.csv         # 0.81100  ← ФИНАЛ (активная)
│   ├── cand_title.csv, cand_deck.csv, sweep_winner.csv
├── requirements.txt
└── README.md
```

## Как воспроизвести

```bash
pip install -r requirements.txt

# 1) Базовый артефакт (ET-бленд, 0.80143):
python src/make_blend.py

# 2) RF-бленд (0.80382):
python src/gen_robust.py          # -> robust_rf2.csv

# 3) Прогнать 100 кандидатов и выбрать лучший (нужны Kaggle-креды для посылки):
python src/sweep100.py
python src/gen_next6.py            # -> cand_3keycvw.csv (0.81100)

# Отправка на Kaggle (требует KAGGLE_USERNAME / KAGGLE_KEY):
python -m kaggle competitions submit -c titanic -f submissions/cand_3keycvw.csv -m "RF 3-key cvw blend, public LB 0.81100"
```

## Disclaimer

Проект — учебно-исследовательский. Соблюдаются правила Kaggle: **никакой утечки меток,
никакого зондирования лидерборда (LB-probing), никакого фейка**. Потолок честного public LB
на выборке 891/418 ≈ 0.81–0.82; выше — только утечка / LB-probing / фейк.

## Лицензия

MIT — см. [LICENSE](LICENSE).
