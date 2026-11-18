# Pre-regisztráció – végső mérési kör (2026-11-12)

Rögzítve az első API-hívás előtt. Terv: „Végső mérési terv, nagy elemszám” táblázat
(témavezető jóváhagyta 2026-11-12). Eltérést csak dokumentálva, indoklással lehet tenni.

## Modellek (pontos azonosító a futtatás napján)
206:    dict(label="gemini-flash-lite", provider="gemini", model="gemini-3.5-flash-lite",
207-         enabled=False, temperature=0.1, extra={}, price={"in": 0.30, "out": 2.50},
208-         min_interval_s=6.0),        # round 1 model; label kept so old results still match
--
253:    dict(label="claude-sonnet-low", provider="anthropic", model="claude-sonnet-5-5",
254-         enabled=False, temperature=None, extra={"effort": "low", "max_tokens": 4000},
255-         price={"in": 2.00, "out": 10.00}, min_interval_s=0.0),
--
269:    dict(label="gpt-sol", provider="openai", model="gpt-6.1-sol", enabled=False,
270-     temperature=None, extra={"reasoning_effort": "low"},
271-     price={"in": 2.00, "out": 10.00}, min_interval_s=0.0),




Beállítások: a meglévő konfiguráció (low reasoning), 1 kép hívásonként, 1 ismétlés,
seed 42 szerinti véletlen sorrend, DFood-TJ: `DFOOD_VIEWS = "oblique"`.

## Pozíciók

Csak olyan ételek kerülnek a tesztekbe, amelyek eredményét korábbi elemzés nem látta.

| Adathalmaz | --n | Tesztelt pozíciók | Kizárt (újrafutó) pozíciók |
|---|---|---|---|
| Nutrition5k | 250 | 51–250 | 1–50 |
| OmniFood8k | 300 | 101–300 | 1–100 |
| ACETADA | 250 | 51–250 | 1–50 |
| DFood-TJ (oblique) | 600 | 351–600 | 1–350 |
| SNAPMe | 250 | 51–250 | 1–50 |

A kizárt pozíciók csak a konzisztencia-elemzéshez használhatók.

## Hipotézisek és tesztek

Minden teszt: kétoldali páros t-próba a pMAE-különbségre (`pvalues.py`, delta-módszeres SE,
df = n − 1). Döntés: szignifikáns, ha a Holm-korrigált p < 0,05 a családon belül.
A hatás nagyságát 95%-os konfidenciaintervallummal közöljük (korrekció nélkül).

| Család | Kérdés | Összevetés | Adathalmaz | Cél | m |
|---|---|---|---|---|---|
| A | Pontosabb-e szénhidrátban a Flash-Lite, mint a Sonnet és a gpt-sol? | Flash-Lite vs. Sonnet; Flash-Lite vs. gpt-sol (P3_macros) | N5k, OmniFood, ACETADA, DFood-TJ | szénhidrát | 8 |
| B | Javít-e a szöveges ételleírás a csak képes becslésen? | P3_macros_hint (kép + szöveg) vs. P3_macros, Flash-Lite | SNAPMe | szénhidrát, zsír | 2 |
| C1 | Megismétlődik-e a P3_recipe zsírjavulása nagyobb mintán? | P3_recipe vs. P3_macros, Flash-Lite | OmniFood | zsír | 1 |
| C2 | Változtat-e a P3_recipe a szénhidrát-pMAE-n? | P3_recipe vs. P3_macros, Flash-Lite | OmniFood, N5k | szénhidrát | 2 |
| C3 | A gpt-solon is javít-e a P3_recipe a zsír-pMAE-n? | P3_recipe vs. P3_macros, gpt-sol | OmniFood | zsír | 1 |

Az A család adathalmazait előre rögzített erőszabály választotta ki (felderítő |Δ| ≥ MDD
n = 200-nál, adathalmazonkénti σ_d-vel); a SNAPMe ezen kiesett.
Reprodukálás: `venv/bin/python mdd_selection.py` (2026-11-18-án pótolva; a számítás változatlan).

## Adatkezelési szabályok

- Nincs kiugró érték szerinti kizárás.
- Parse-hibás vagy hiányzó válasz: a pár kiesik, a számát armonként közlöm; ha egy armban
  10%-nál több hiányzik, külön jelzem. Parse-hibás hívást nem futtatok újra; API-hibás
  hívást ugyanazzal a paranccsal újrafuttatok.
- A tesztekhez csak akkor nyúlok, ha az adott blokk minden armja teljes. Futás közben csak
  működési adatot nézek (hívásszám, api_fail, parse_fail, költség), pontosságot nem.

## Leíró eredmények (teszt nélkül, 95%-os CI-vel)

Energia, fehérje, tömeg (az A blokkban a zsír is); Sonnet vs. gpt-sol; késleltetés, költség,
parse_fail- és api_fail-arány modellenként; konzisztencia a kizárt pozíciókon (csak ahol
azonos konfigurációjú korábbi futás van: DFood 1–50; OmniFood Sonnet és gpt-sol 1–50;
OmniFood Flash-Lite 1–100; a többi adathalmazon 1–50).


## Eltérések

1. **Újrafuttatott hívások (Nutrition5k, Flash-Lite, P3_macros).** 14 hívás az első
   kísérletre helyi beállítási hibával zárult („No API key was provided”: az API-kulcs
   hiányzott a környezetből, a kérés nem jutott el a szolgáltatóhoz). A pre-regisztráció
   szerint ugyanazzal a paranccsal újrafuttattam őket; mind a 14 étel egy sikeres
   választ kapott. Az elemzés ételenként az utolsó, sikeres sort használja. Mivel a hiba
   nem a szolgáltatónál keletkezett, a stabilitási mutatókba nem számít bele.
2. **SNAPMe elemszám.** A `--n 250` kiválasztásból 248 étel került lekérésre mindkét
   ágban (a két ág ételei azonosak). A tesztelt 51–250. pozícióból ezért 199 pár került a
   B család tesztjeibe 200 helyett.
   A két hiányzó étel a 32. (`7e2a432f45ae8c667ffc4619bb2e.jpeg`) és a 139.
   (`a1fd87844e09b6f54f0480949ae7.jpeg`) pozíció: a képfájlokat a Pillow nem ismeri fel
   (`UnidentifiedImageError`), ezért a `benchmark.py` API-hívás nélkül kihagyta őket.
3. **Az elemzési szkript.** A teszteket a `prereg_tests.py` végezte, amelyet a futtatás
   előtt commitoltam (2026-11-14 14:28, `prereq_tests.py` néven; 2026-11-15-én
   átnevezve). A futtatás: 2026-11-14 19:10.
4. **A végső körtől független mérés.** 2026-11-14-én egy külön páros előmérés futott a
   JPEG-minőség hatásáról (Flash-Lite, Nutrition5k, P3_macros, a sorrend 1–50. étele,
   minőség 80 és 90). A végső kör ágait és tesztjeit nem érinti; leírása és eredménye:
   `notes/jpeg_quality_check.md`.
5. **DFood-TJ: 25 tesztelt étel korábbi futásban (észlelve 2026-11-17-én, utólag).** A
   351–600. pozíció 250 ételéből 25 szerepelt a `both_dfood_tj_n50_s42_gemini_flash_lite_pre`
   és az `overhead_dfood_tj_n50_s42_gemini_flash_lite_pre` futásban (2026-10-23, első mérési
   kör, P3 prompt: csak energia és tömeg, sorok szerinti mintavétel a későbbi étkezés szerinti
   helyett). A szénhidrátot ezek a futások nem becsülték, így az A család hipotézisét nem
   befolyásolhatták. Érzékenységvizsgálat a 25 étel nélkül (225 pár): Sonnet Δ −4,87
   (−6,67; −3,07), gpt-sol Δ −7,00 (−9,69; −4,32); mindkettő Holm-korrekció után is
   szignifikáns, a következtetés nem változik. Reprodukálás:

       venv/bin/python - <<'EOF'
       import json, sys; from pathlib import Path
       sys.path.insert(0, "."); import significance as sig, prereg_tests as P
       R = Path("results"); ex = set()
       for run in ("both_dfood_tj_n50_s42_gemini_flash_lite_pre", "overhead_dfood_tj_n50_s42_gemini_flash_lite_pre"):
           v = sig.manifest_view(R / run / "results.jsonl")
           ex |= {sig.unit_of("dfood_tj", json.loads(l)["sample_id"], v) for l in (R / run / "results.jsonl").open()}
       for A, B, k in [f for f in P.FAMILIES["A"] if "dfood" in f[0]]:
           dA, dB = P.load_run(A, R)[0], P.load_run(B, R)[0]
           print(B, len(set(dA["unit"]) & ex), P.test(dA[~dA["unit"].isin(ex)], dB[~dB["unit"].isin(ex)], k))
       EOF
Más eltérés nem történt: a modellek, promptok, pozíciók, családok és döntési szabályok
a fentiek szerint változatlanok.

