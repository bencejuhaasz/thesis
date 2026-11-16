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

### Pontosság és konfidenciaintervallum (a dolgozatban ez szerepel)

A dolgozat minden páros konfidenciaintervalluma delta-módszeres standard hibával és
Student-eloszlással ($n-1$ szabadsági fok) készül, a döntési méréssel azonos módon.

```
venv/bin/python pvalues.py --runs n5k_lite_macros_q90_n50_s42 n5k_lite_macros_q80_n50_s42 \
  --metrics kcal_pMAE% carb_g_pMAE% --no-unpaired --out pvalues_jpeg
venv/bin/python -c "
import pandas as pd
d=pd.read_csv('pvalues_jpeg/pvalues.csv')
print(d[['metric','A','B','delta_pp','CI95_low','CI95_high','p']].to_string(index=False))"
```

```
      metric     A     B  delta_pp  CI95_low  CI95_high     p
carb_g_pMAE% 31.85 36.30      4.45     -3.67      12.57 0.276
  kcal_pMAE% 30.25 33.19      2.94     -3.24       9.12 0.344
```

A `pvalues.py` a számokat két tizedesre kerekítve menti, a konzolkimenete ezt még egyszer
egy tizedesre kerekíti, ezért ott a szénhidrát Δ értéke +4.5. A pontos különbség a nyers
adatból:

```
venv/bin/python -c "
import json
R={}
for r in ['n5k_lite_macros_q90_n50_s42','n5k_lite_macros_q80_n50_s42']:
    L=[json.loads(l) for l in open(f'results/{r}/results.jsonl')]
    R[r]={x['sample_id']:x for x in L if x.get('pred')}
a,b=R.values(); ids=[i for i in a if i in b]
for k in ('kcal','carb_g'):
    t=sum(a[i]['gt'][k] for i in ids)
    d=sum(abs(b[i]['pred'][k]-b[i]['gt'][k])-abs(a[i]['pred'][k]-a[i]['gt'][k]) for i in ids)
    print(k, len(ids), 100*d/t)"
```

```
kcal 50 2.937145378298291
carb_g 50 4.447572647564867
```

A q90 futás `results.jsonl`-je 51 sort tartalmaz: az első hívás helyi „No API key”
hibával meghiúsult, majd sikeresen újrafutott. Az elemzés a sikeres hívást használja,
ezért mindkét futásban 50 étel szerepel.

### Korábbi, bootstrap-alapú összevetés (a dolgozatból 2026-11-16-án kivéve)

```
venv/bin/python analyze.py compare n5k_lite_macros_q90_n50_s42 n5k_lite_macros_q80_n50_s42 --target kcal
venv/bin/python analyze.py compare n5k_lite_macros_q90_n50_s42 n5k_lite_macros_q80_n50_s42 --target carb_g
```

Energia: Δ pMAE +2.9 pp, CI [-2.7, +9.1]; szénhidrát: Δ pMAE +4.4 pp, CI [-3.3, +12.5]
(5000 visszatevéses újramintavételezés, seed 0). Ugyanarra a következtetésre jut, mint a
delta-módszer.

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
| 80-as minőséggel az energia pMAE-je 2,9 pp-vel nagyobb | nyers adat, `kcal` | +2,937 pp |
| 80-as minőséggel a szénhidrát pMAE-je 4,4 pp-vel nagyobb | nyers adat, `carb_g` | +4,448 pp |
| A különbség nem szignifikáns; 95%-os CI | `pvalues_jpeg/pvalues.csv`, `CI95_low`, `CI95_high`, `p` | −3,2 … +9,1 pp (p = 0,344); −3,7 … +12,6 pp (p = 0,276) |
| A bemeneti tokenszám a két beállítás mellett azonos | `summary.csv`, `tokens_in_p50` | 1326 mindkettőben |

Értelmezés: a nem szignifikáns különbség nem bizonyítja, hogy a tömörítésnek nincs
hatása (a konfidenciaintervallumok szélesek), de a pontbecslés mindkét célmennyiségre a
90-es minőség mellett szól, és a 90-es minőség költségtöbbletet nem okoz. Az 1024 képpontos
méretkorlátot ez a mérés nem vizsgálja.
