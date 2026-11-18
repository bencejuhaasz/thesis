# Sikertelen hívások kezelése (F2, témavezetői visszajelzés 2026-11-16)

## Szabály (rögzítve 2026-11-16)

- Sikertelen hívás: API-hiba (újrapróbálás után is) vagy fel nem dolgozható válasz.
  Becslése 0, vagyis a hibája a teljes referenciaérték (az alkalmazás ilyenkor nem
  rögzít semmit).
- Két pontosságot közlünk: pMAE a sikeres hívásokon (`pMAE`) és pMAE a teljes
  hívásszámon (`pMAE_all`), valamint a sikeres hívások arányát (`success_rate`).
- Mindhárom a vizsgált (döntési mérésben: tesztelt) pozíciókon számolódik, nem a
  `summary.csv`-ből (az az összes pozícióra vonatkozik).
- Újrapróbált API-hiba: a hívásszámba a végső hívás számít (a `results.jsonl`-ban
  pozíciónként az utolsó rekord).
- Bemeneti hiba (olvashatatlan kép, hívás nem történt) nem sikertelen hívás: az
  adathalmazt jellemzi, nem az API-t. SNAPMe: 2 ilyen kép (32. és 139. pozíció,
  l. prereg_final_2026-11-12.md); a mintavételt az összehasonlíthatóság miatt nem
  módosítottam, ezért a SNAPMe döntési mérése 199 ételen fut.

## Kód (main `b5aa417`)

- `significance.py` `load_group`: minden hívást beolvas; ételenként `n_calls`, `n_ok`,
  és `z_k` (sikertelen hívás = 0). A `p_k` és a `metric_arrays` változatlan, így minden
  korábbi pontossági szám változatlan (ellenőrizve: prereg_tests, final_descriptive,
  ci.py, pvalues.py teljes kimenete).
- `metric_arrays_all`: |z − t| a teljes hívásszámon.
- `final_descriptive.py`: accuracy.csv `n_calls`, `success_rate`, `pMAE_all` (+ CI);
  ops.csv `calls_tested`, `failed_tested`.
- `ci.py`: summary_ci.csv célonként `_n_calls`, `_success_rate`, `_pMAE_all` (+ CI).
- Egyetlen szándékos változás a korábbi kimenetekhez képest: a válaszidő-átlag és a
  címke n-je a fel nem dolgozható hívásokat is tartalmazza (ezek is lefutottak).

## Eredmény

Döntési mérés (`final_descriptive/ops.csv`, accuracy.csv, main `fb6029a`): minden futás
minden tesztelt pozícióján sikeres a hívás (`failed_tested` = 0, `success_rate` = 1),
így `pMAE_all` = `pMAE`.

Előmérés (`ci.py`, 125 futás × modell): egyetlen futásban volt sikertelen hívás:
`nutrition5k_sol_med_t1000_q80_n50_s42` (gpt-6.1-sol, reasoning_effort medium,
max_completion_tokens 1000): 50-ből 5 válasz üres — a gondolkodás a teljes 1000 tokenes
keretet elhasználta (a sikeres válaszok kimenetének mediánja 480 token).

| cél | n / hívás | pMAE (sikeres) | pMAE (teljes hívásszám) |
|---|---|---|---|
| kcal | 45 / 50 | 44,7 (31,8; 57,6) | 55,4 (41,4; 69,5) |
| tömeg | 45 / 50 | 30,2 (21,2; 39,2) | 44,9 (30,9; 59,0) |
| szénhidrát | 45 / 50 | 64,6 (47,4; 81,8) | 71,7 (56,7; 86,6) |
| fehérje | 45 / 50 | 49,9 (34,8; 65,1) | 59,6 (44,0; 75,2) |
| zsír | 45 / 50 | 58,5 (40,1; 76,9) | 66,2 (49,2; 83,2) |

## Reprodukálás

    venv/bin/python final_descriptive.py
    venv/bin/python ci.py --runs nutrition5k_sol_med_t1000_q80_n50_s42 --out ci_f2
