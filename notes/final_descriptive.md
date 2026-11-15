# A végső kör leíró eredményei, hihetőség és konzisztencia – források

Dolgozat: Eredmények fejezet (leíró eredmények, strukturált kimenet / stabilitás,
konzisztencia). Az adatok a `data-freeze-2026-11-15` állapotban vannak; az alábbi
elemzések új API-hívást nem végeztek, csak a fagyasztott válaszokat dolgozzák fel.
Futtatás helye: a `datasets` mappa. Elemzés: 2026-11-15.

A pre-regisztrált hipotézistesztek: `prereg_tests.py` → `prereg_tests/prereg_tests.csv`
(lásd `notes/prereg_final_2026-11-12.md`). Ez a jegyzet a teszteken kívüli, leíró részt
dokumentálja.

## 1. Leíró pontosság és működés: `final_descriptive.py`

```
venv/bin/python final_descriptive.py
```

Kimenet: `final_descriptive/accuracy.csv`, `pairs.csv`, `ops.csv`. Minden szám a tesztelt
pozíciókon (N5k, ACETADA, SNAPMe 51–250., OmniFood8k 101–300., DFood-TJ 351–600.),
ugyanazzal a betöltővel, mint a tesztek.

- `accuracy.csv`: ágonként és célmennyiségenként pMAE és bias% 95%-os CI-vel (delta-módszer,
  Student t), meredekség (becslés a valós értékre), skill (kihagyásos mediántipphez képest),
  bias% a valós érték szerinti alsó / középső / felső harmadban.
- `pairs.csv`: páros különbségek 95%-os CI-vel, döntés és Holm-korrekció nélkül
  (az A blokk modellpárjai energiára, tömegre, fehérjére, zsírra; Sonnet vs. Sol minden
  célmennyiségre).
- `ops.csv`: hívásszám, api_fail, parse_fail (summary.csv, minden pozíció), válaszidő átlaga
  és mediánja (tesztelt pozíciók), költség 1000 hívásra.

Fő megfigyelések (forrás: `accuracy.csv`, `pairs.csv`, `ops.csv`):

| Megfigyelés | Forrás | Értékek |
|---|---|---|
| A drága modellek a szénhidrátot rendszeresen túlbecsülik | accuracy, carb_g, `bias%` | N5k: Lite +12, Sonnet +49, Sol +50; OmniFood: +10 / +63 / +45; ACETADA: +11 / +18 / +27; DFood: −19 / −10 / +10 |
| A túlbecslés a szénhidrátszegény ételeken a legnagyobb | accuracy, carb_g, `bias%_low3` | N5k: Sonnet +157, Sol +133; OmniFood: Sonnet +314, Sol +284 |
| A szénhidrát-definíció (P3_recipe) nem változtat a túlbecslésen | accuracy, carb_g, `bias%` | Sol: +45,2 (macros) / +45,5 (recipe); Lite OmniFood: +10,4 / +9,3 |
| Energiában és tömegben más a sorrend | pairs | OmniFood: energia Sonnet −10,1, Sol −12,1 pp; tömeg +22,5 / +21,3 pp (vs. Lite) |
| A skill több helyen ≤ 0 | accuracy, `skill` | DFood Lite szénhidrát −0,25; OmniFood Sonnet / Sol szénhidrát −0,40 / −0,20 |
| Nincs API- vagy feldolgozási hiba | ops | api_fail = parse_fail = 0 minden ágban |
| Válaszidő (medián) | ops, `latency_p50_s` | Lite 2,1–2,6 s; Sonnet 3,5–4,3 s; Sol 8,1–12,5 s |
| Költség 1000 hívásra | ops, `cost_per_1000_usd` | Lite 0,9–1,5; Sonnet 6,2–9,5; Sol 5,6–8,7 USD |

## 2. Kiugró érték: DFood-TJ, Sonnet, zsír (2995-ös étel)

A Sonnet egy tételre („Stir-fried napa cabbage with shredded pork and carrot”, 90 g, 70 kcal)
4122 g zsírt adott; az étel összesített zsírbecslése 4132,5 g (valós: 18,4 g). A válasz a
sémának megfelelt (feldolgozható volt). Utólagos érzékenységi elemzés:

```
venv/bin/python -c "
import numpy as np, prereg_tests as P, significance as sig
from pathlib import Path
d,ds,n=P.load_run('final_dfood_sonnet_macros', Path('results'))
for drop in (False, True):
    x=d.drop(index='2995') if drop else d
    a=sig.metric_arrays(x,'fat_g','pMAE'); v,se,_=sig.ratio_stats(a[1],a[2])
    print('2995 nélkül' if drop else 'mind', len(a[2]), round(v,1), '+-', round(1.97*se,1))
"
```

```
mind 250 121.8 +- 154.2
2995 nélkül 249 43.6 +- 4.1
```

A zsír nem tesztelt célmennyiség a DFood-TJ-n; a tesztekre nincs hatása.

## 3. Hihetőségi ellenőrzés: `plausibility_check.py`

```
venv/bin/python plausibility_check.py
```

Kimenet: `plausibility/plausibility.csv`, `plausibility/flagged_items.csv` (a 17 végső futás).
Szabályok tételenként: makrotápanyagok tömege > tétel tömege; |4·szénhidrát + 4·fehérje +
9·zsír − kcal| > 50%·kcal (kcal ≥ 20). Kb. 22 000 tételből 39 jelölt.

A 39 tétel besorolása **kézi átnézéssel** készült (nem szkript):

| Csoport | Db | Megjegyzés |
|---|---|---|
| A szabály korlátja: alkohol (bor, sör, vodka) | 12 | az alkohol energiáját a 4/4/9 képlet nem számolja; a válasz helyes |
| A szabály korlátja: citrom | 4 | rost és szerves savak; a válasz helyes |
| Kerekítés, víztartalom nélküli ételek (csokoládé, vaj, cukor, méz, olaj, chips, aszalt füge) | 17 | a makrók tömege 1–2 g-mal több a tétel tömegénél, az energia egyezik |
| Kisebb energia-következetlenség (tej 30 g / 48 kcal, spenót 40 g / 25 kcal) | 2 | |
| Durva, nagyságrendi hiba | 4 | lásd alább |

Durva hibák a végső körben:

| Futás | Étel | Tétel | Hibás érték |
|---|---|---|---|
| dfood_sonnet_macros | 2995 | napa cabbage, 90 g, 70 kcal | zsír 4122 g |
| dfood_sonnet_macros | 1657 | white gourd, 120 g, 70 kcal | zsír 41 g |
| omni_sonnet_macros | 2452 | bamboo shoot strips, 55 g, 70 kcal | zsír 41 g |
| n5k_lite_recipe | dish_1561575506 | cauliflower, 70 g, 30 kcal | szénhidrát 35 g |

Egy korábbi futásban (`plausibility_check.py omnifood8k_1image_sonnet_n50_s42`) egy további:
`omnifood8k_1image_sonnet_n50_s42`, 3734-es étel, „Stir-fried sliced meat with green
onion/vegetables”, 50 g, 90 kcal, zsír 6006 g. Összesen tehát négy nagyságrendi zsírhiba,
mind a Sonnettől, és egy szénhidráthiba a Flash-Lite-tól (P3_recipe).

A SNAPMe magasabb jelölési aránya (3–6%) az adathalmaz jellegéből adódik (italok, alkohol,
csomagolt ételek); a jelölések többsége ott a szabályok korlátja.

## 4. Konzisztencia: `consistency_pairs.py`

```
venv/bin/python consistency_pairs.py --pair final_n5k_lite_macros=nutrition5k_macros_n50_s42
```

Kimenet: `consistency_pairs/consistency_pairs.csv`, `consistency_pairs/pairs_used.csv`.
A végső futások a kizárt pozíciókon (N5k, ACETADA, SNAPMe, DFood 1–50.; OmniFood Lite
1–100.) összevetve egy korábbi, azonos beállítású futással (azonos modellazonosító és
beállítás, promptszöveg, use_text, képméret, nézetbeállítás). Ételenként két független
hívás; mérőszámok a `consistency.py` szerint, k = 2.

A `--pair` a Nutrition5k Flash-Lite ágnál kell: automatikusan három azonos beállítású futás
is egyezett (50–50 közös étellel), és döntetlennél az `n5k_lite_macros_q90_n50_s42`
(a végső kör után, 2026-11-14-én készült JPEG-mérés) lett volna kiválasztva. A
pre-regisztráció korábbi futást ír elő, ezért az eredeti előmérés
(`nutrition5k_macros_n50_s42`, 2026-10-27) a pár.

A párok (`pairs_used.csv`):

| Végső futás | Korábbi futás | Ételek |
|---|---|---|
| final_acetada_lite_macros | acetada_lite_macros_n50_s42 | 50 |
| final_acetada_sol_macros | acetada_gpt-sol_macros_n50_s42 | 50 |
| final_acetada_sonnet_macros | acetada_sonnet_macros_n50_s42 | 50 |
| final_dfood_lite_macros | dfood_lite_macros_oblique_n50_s42 | 50 |
| final_dfood_sol_macros | dfood_gpt-sol_macros_oblique_n50_s42 | 50 |
| final_dfood_sonnet_macros | dfood_sonnet_macros_oblique_n50_s42 | 50 |
| final_n5k_lite_macros | nutrition5k_macros_n50_s42 | 50 |
| final_n5k_lite_recipe | nutrition5k_lite_recipe_n50_s42 | 50 |
| final_n5k_sol_macros | nutrition5k_sol_macros_n50_s42 | 50 |
| final_n5k_sonnet_macros | nutrition5k_sonnet_macros_n50_s42 | 50 |
| final_omni_lite_macros | omnifood8k_lite_macros_confirm_s42 | 100 |
| final_omni_lite_recipe | omnifood8k_lite_recipe_confirm_s42 | 100 |
| final_omni_sol_macros | omnifood8k_1image_sol_n50_s42 | 50 |
| final_omni_sol_recipe | omnifood8k_1image_sol_recipe_n50_s42 | 50 |
| final_omni_sonnet_macros | omnifood8k_1image_sonnet_n50_s42 | 50 |
| final_snapme_lite_hint | snapme_lite_hint_n50_s42 | 49 |
| final_snapme_lite_macros | snapme_lite_macros_n50_s42 | 49 |

Fő megfigyelések (energia, `consistency_pairs.csv`):

| Modell | within_% | RC_% | ICC |
|---|---|---|---|
| Flash-Lite (P3_macros) | 5,6–11,8 | 15,5–32,7 | 0,885–0,994 |
| Sonnet | 3,9–7,6 | 10,7–21,1 | 0,893–0,993 |
| Sol | 4,7–11,6 | 13,1–32,1 | 0,858–0,992 |

Szénhidrátban az RC_% 12,9–75,3. A Flash-Lite (temperature 0,1) nem konzisztensebb a
Sonnetnél (alapértelmezett temperature). Korlát: a két hívás között napok teltek el, így
a szolgáltató oldali esetleges változások is benne vannak.

Kiugró érték: OmniFood8k, Sonnet, zsír (within_% 2127,5), a korábbi futás 3734-es ételének
hibája miatt (3. pont). Utólagos érzékenységi elemzés:

```
venv/bin/python -c "
import numpy as np, consistency as C, consistency_pairs as CP
from pathlib import Path
R=Path('results'); excl,ds,view=CP.excluded_units('final_omni_sonnet_macros',R)
fp=CP.preds(R/'final_omni_sonnet_macros',ds,view); ep=CP.preds(R/'omnifood8k_1image_sonnet_n50_s42',ds,view)
for drop in (False,True):
    u=[x for x in sorted(excl) if x in fp and x in ep and not (drop and x=='3734')]
    P=np.array([[fp[x][0]['fat_g'],ep[x][0]['fat_g']] for x in u]); t=np.array([fp[x][1]['fat_g'] for x in u])
    s=C.target_stats(P,t,0,np.random.default_rng(0))
    print('3734 nélkül' if drop else 'mind', s['meals'], 'within%',s['within_%'],'RC%',s['RC_%'],'ICC',s['ICC'],'pMAE final/korábbi',s['pMAE_r0'],s['pMAE_r1'])
"
```

```
mind 50 within% 2127.5 RC% 5893.1 ICC 0.003 pMAE final/korábbi 60.2 482.7
3734 nélkül 49 within% 4.7 RC% 13.0 ICC 0.977 pMAE final/korábbi 60.6 60.3
```

