for d in omnifood8k nutrition5k acetada; do venv/bin/python calibration_check.py kizart_results/n50_s42_gemini_flash_lite_pre --dataset $d; done
# Az utólagos kalibráció elvetése – a dolgozatban hivatkozott számok forrása

Dolgozat: Tervezés fejezet, „Az utólagos kalibráció elvetése” szakasz.
Minden szám az alábbi két parancs kimenetéből származik (API-hívás nélkül).
Futtatás helye: a `datasets` mappa (ahol a `benchmark.py` és az `analyze.py` van).

## Felhasznált futás

| Futás | Modell | Prompt | Adathalmazok | n | seed | Dátum |
|---|---|---|---|---|---|---|
| `kizart_results/n50_s42_gemini_flash_lite_pre` | gemini-flash-lite (gemini-3.5-flash-lite) | P3 | omnifood8k, nutrition5k, acetada (és további) | 50 / adathalmaz | 42 | 2026-10-23 (első mérési kör) |

A futás a `kizart_results` mappában van. Az első kör ismert pipeline-hibája (a DFood-TJ
„both” nézete egyetlen képet küldött) csak a DFood-TJ-t érintette; az itt használt három
adathalmaz eredményeit nem.

## 1. Torzítás és kalibráció: `calibration_check.py`

```
for d in omnifood8k nutrition5k acetada; do venv/bin/python calibration_check.py kizart_results/n50_s42_gemini_flash_lite_pre --dataset $d; done
```

Kimenet (2026-11-13, a `calibration_check.py` első változatával, `pMAE_loglin` oszlop nélkül):

```
run                                                        n     bias%_kcal   bias%_mass_g   bias%_carb_g bias%_protein_g    bias%_fat_g  pMAE_raw pMAE_scale pMAE_linear
kizart_results/n50_s42_gemini_flash_lite_pre              50          -45.0           11.1              -              -              -      50.4       38.4        32.1
run                                                        n     bias%_kcal   bias%_mass_g   bias%_carb_g bias%_protein_g    bias%_fat_g  pMAE_raw pMAE_scale pMAE_linear
kizart_results/n50_s42_gemini_flash_lite_pre              50          -11.1          -11.5              -              -              -      31.7       33.7        34.1
run                                                        n     bias%_kcal   bias%_mass_g   bias%_carb_g bias%_protein_g    bias%_fat_g  pMAE_raw pMAE_scale pMAE_linear
kizart_results/n50_s42_gemini_flash_lite_pre              50           -1.6            0.3              -              -              -      22.6       23.1        23.6
```

(Sorrend: omnifood8k, nutrition5k, acetada.)

## 2. Meredekség: `analyze.py compare`

```
for d in omnifood8k nutrition5k acetada; do venv/bin/python analyze.py compare kizart_results/n50_s42_gemini_flash_lite_pre --dataset $d --target kcal; done
```

Kimenet (2026-11-14):

```
target kcal: 50 meals in all 1 specs (per spec: [50])
mean truth 389.1   (logSD = SD of ln(pred/true): spread without bias)

spec                                                          pMAE%     MAE  bias%     r slope  skill  logSD
n50_s42_gemini_flash_lite_pre:kcal                             50.4   196.1  -45.0  0.34  0.21  -0.65  0.527
target kcal: 50 meals in all 1 specs (per spec: [50])
mean truth 260.2   (logSD = SD of ln(pred/true): spread without bias)

spec                                                          pMAE%     MAE  bias%     r slope  skill  logSD
n50_s42_gemini_flash_lite_pre:kcal                             31.7    82.5  -11.1  0.83  0.80   0.49  0.457
target kcal: 50 meals in all 1 specs (per spec: [50])
mean truth 718.4   (logSD = SD of ln(pred/true): spread without bias)

spec                                                          pMAE%     MAE  bias%     r slope  skill  logSD
n50_s42_gemini_flash_lite_pre:kcal                             22.6   162.1   -1.6  0.72  0.64   0.34  0.346
```

## A dolgozat állításai és forrásuk

| Állítás a szövegben | Forrás | Érték |
|---|---|---|
| A nagyobb adagokat a modell rendre alulbecsüli | 2. parancs, `slope` | 0,21 / 0,80 / 0,64 (mind < 1) |
| Torzítás: ACETADA −1,6%, Nutrition5k −11,1%, OmniFood8k −45,0% | 1. parancs, `bias%_kcal` (a 2. parancs `bias%` oszlopa is) | −1,6 / −11,1 / −45,0 |
| OmniFood8k: a lineáris korrekció 50,4%-ról 32,1%-ra csökkenti a pMAE-t | 1. parancs, `pMAE_raw` → `pMAE_linear` | 50,4 → 32,1 |
| Nutrition5k: a korrekció ront (31,7% → 34,1%) | 1. parancs, `pMAE_raw` → `pMAE_linear` | 31,7 → 34,1 |
| ACETADA: a korrekció ront (22,6% → 23,6%) | 1. parancs, `pMAE_raw` → `pMAE_linear` | 22,6 → 23,6 |

A „kihagyásos (leave-one-out)” korrekció: minden étel korrekcióját a többi 49 étel alapján
illesztett `valós = a · becslés + b` egyenes adja (`calibration_check.py`, `loo_linear`).

## Eltérés a korábbi összefoglalótól

A `claude/first-round-findings.md` összefoglaló (2026-10-23) „OmniFood 50.4 → 31.9” értéket
közöl. Ez egy repón kívüli, ideiglenes elemzésből származott, és a log-lineáris korrekciót
használta (`ln valós = a · ln becslés + b`), nem a lineárist. A dolgozat a fenti, a repóból
reprodukálható lineáris értéket (32,1%) használja. A log-lineáris érték a `calibration_check.py`
frissített változatával (`pMAE_loglin`oszlop) ellenőrizhető.


