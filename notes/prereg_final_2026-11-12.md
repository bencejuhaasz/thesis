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

(futás után kitöltendő)
