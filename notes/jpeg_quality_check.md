# JPEG-minőség (80 vs 90) – a dolgozatban hivatkozott számok forrása

Dolgozat: Tervezés fejezet, „A mérési környezet kialakítása” szakasz, a képek
előfeldolgozásáról szóló bekezdés (a 90-es JPEG-minőség indoklása).
Futtatás helye: a `datasets` mappa. Mérés: 2026-11-14.

## Futások

| Futás | Program | JPEG-minőség | Modell | Prompt | Adathalmaz | n | seed |
|---|---|---|---|---|---|---|---|
| `n5k_lite_macros_q90_n50_s42` | `benchmark.py` | 90 | gemini-flash-lite | P3_macros | nutrition5k | 50 | 42 |
| `n5k_lite_macros_q80_n50_s42` | `jpeg80_benchmark.py` | 80 | gemini-flash-lite | P3_macros | nutrition5k | 50 | 42 |

A manifest nem rögzíti a JPEG-minőséget (`jpeg_quality` mező nincs), ezért a minőséget a
futtatott program azonosítja: a `benchmark.py`-ban `JPEG_QUALITY = 90`, a
`jpeg80_benchmark.py`-ban `JPEG_QUALITY = 80`. A futtatott kódváltozatot a két manifest
`benchmark_sha256` mezője rögzíti. A két program a Nutrition5k-konfigurációban és a
Flash-Lite-bejegyzésben megegyezik; egyéb eltéréseik (SNAPMe-beállítás, Claude-bejegyzések,
NutriBench-prompt) ezt a mérést nem érintik. Mindkét futás ugyanazt az 50 ételt használta
(`analyze.py compare`: „50 meals in all 2 specs”).

## Parancsok és kimenetek

```
venv/bin/python benchmark.py --datasets nutrition5k --models gemini-flash-lite --prompt P3_macros --n 50 --run-name n5k_lite_macros_q90_n50_s42
venv/bin/python jpeg80_benchmark.py --datasets nutrition5k --models gemini-flash-lite --prompt P3_macros --n 50 --run-name n5k_lite_macros_q80_n50_s42
```

### Energia

```
venv/bin/python analyze.py compare n5k_lite_macros_q90_n50_s42 n5k_lite_macros_q80_n50_s42 --target kcal
```

```
target kcal: 50 meals in all 2 specs (per spec: [50, 50])
mean truth 260.2   (logSD = SD of ln(pred/true): spread without bias)

spec                                                          pMAE%     MAE  bias%     r slope  skill  logSD
n5k_lite_macros_q90_n50_s42:kcal                               30.2    78.7  -16.4  0.83  0.73   0.51  0.438
n5k_lite_macros_q80_n50_s42:kcal                               33.2    86.3  -13.5  0.79  0.73   0.47  0.466

paired vs n5k_lite_macros_q90_n50_s42:kcal (negative = better):
  n5k_lite_macros_q80_n50_s42:kcal                           Δ pMAE  +2.9 pp  CI [-2.7, +9.1]  Wilcoxon p 0.55  meals better 38%
```

### Szénhidrát

```
venv/bin/python analyze.py compare n5k_lite_macros_q90_n50_s42 n5k_lite_macros_q80_n50_s42 --target carb_g
```

```
target carb_g: 50 meals in all 2 specs (per spec: [50, 50])
mean truth 19.7   (logSD = SD of ln(pred/true): spread without bias)

spec                                                          pMAE%     MAE  bias%     r slope  skill  logSD
n5k_lite_macros_q90_n50_s42:carb_g                             31.9     6.3   -7.9  0.83  0.86   0.51  0.654
n5k_lite_macros_q80_n50_s42:carb_g                             36.3     7.2   -7.8  0.78  0.83   0.44  0.703

paired vs n5k_lite_macros_q90_n50_s42:carb_g (negative = better):
  n5k_lite_macros_q80_n50_s42:carb_g                         Δ pMAE  +4.4 pp  CI [-3.3, +12.5]  Wilcoxon p 0.14  meals better 34%
```

### Tokenszám és költség

```
grep -h . results/n5k_lite_macros_q90_n50_s42/summary.csv results/n5k_lite_macros_q80_n50_s42/summary.csv | cut -d, -f1-4,11-13
```

```
dataset,model,prompt,output,images_per_call,tokens_in_p50,cost_usd
nutrition5k,gemini-flash-lite,P3_macros,schema,1.0,1326.0,0.0483
dataset,model,prompt,output,images_per_call,tokens_in_p50,cost_usd
nutrition5k,gemini-flash-lite,P3_macros,schema,1.0,1326.0,0.0487
```

(Első sor: q90, második sor: q80.)

## A dolgozat állításai és forrásuk

| Állítás a szövegben | Forrás | Érték |
|---|---|---|
| 80-as minőséggel az energia pMAE-je 2,9 pp-vel nagyobb | `compare --target kcal`, Δ pMAE | +2,9 pp |
| 80-as minőséggel a szénhidrát pMAE-je 4,4 pp-vel nagyobb | `compare --target carb_g`, Δ pMAE | +4,4 pp |
| A különbség nem szignifikáns; 95%-os CI | ugyanott, CI | −2,7 … +9,1 pp; −3,3 … +12,5 pp |
| A bemeneti tokenszám a két beállítás mellett azonos | `summary.csv`, `tokens_in_p50` | 1326 mindkettőben |

Értelmezés: a nem szignifikáns különbség nem bizonyítja, hogy a tömörítésnek nincs
hatása (a konfidenciaintervallumok szélesek), de a pontbecslés mindkét célmennyiségre a
90-es minőség mellett szól, és a 90-es minőség költségtöbbletet nem okoz. Az 1024 képpontos
méretkorlátot ez a mérés nem vizsgálja.
