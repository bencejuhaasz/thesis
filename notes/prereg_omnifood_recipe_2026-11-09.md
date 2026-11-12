# Előzetes rögzítés: P3_recipe vs. P3_macros, OmniFood8k zsír (2026-11-09, a megerősítő futás előtt)

Hipotézis: a P3_recipe prompt kisebb zsír-pMAE-t ad, mint a P3_macros, az OmniFood8k-n.
Előzmény (felderítés, 1-50. étel, 4 kipróbált prompt): zsír Delta = -8,2 pp [-11,0; -5,3].

Beállítás: modell gemini-flash-lite (benchmark.py config szerint), JSON-séma kimenet, OmniFood8k 1 kép,
ugyanaz a benchmark.py, mint a felderítésben (sha256 lent). Mindkét prompt ugyanazon a napon fut.
Minta: OmniFood8k, a seed-42 sorrend 51-100. étele (a felderítésben nem használt ételek), n = 50.
Kiértékelés csak azokon az ételeken, amelyek mindkét futásban értékelhetők.

Elsődleges teszt (egyetlen, m = 1):
  Delta = pMAE_zsír(P3_recipe) - pMAE_zsír(P3_macros), páros, ugyanazokon az ételeken;
  standard hiba delta-módszerrel (mint a pvalues.py), kétoldali t-próba (df = n - 1), alfa = 0,05.
  Siker: p < 0,05 és Delta < 0. Közlés: Delta és 95% CI.
Robusztusság (nem döntési alap): bootstrap 95% CI, Wilcoxon-p.
Másodlagos, csak leíró (CI, szignifikancia-állítás nélkül): kcal, szénhidrát, fehérje, gramm.
Az eredmény láttán nem változik sem a teszt, sem a minta; újrafuttatás csak API-hiba pótlására.
d696c489433c14a80a03f5ca6982911ca5e74c93cb4d1bc290b5aeb2ee63034f  benchmark.py

friss ételek: P3_macros 50, P3_recipe 50; mindkettőben értékelhető: 50
            cél          n   macros   recipe    delta         95% CI (t)     p (t) 95% CI (bootstrap) p (Wilc.)
ELSŐDLEGES  fat_g       50    60.2%    55.4%     -4.9       [-8.2; -1.6]    0.0040       [-8.1; -1.9]    0.0060
másodlagos  kcal        50    41.7%    39.7%     -2.1       [-4.9; +0.8]    0.1517       [-4.8; +0.8]    0.1857
másodlagos  carb_g      50    62.6%    60.2%     -2.4       [-8.2; +3.5]    0.4205       [-8.3; +3.2]    0.3803
másodlagos  protein_g   50    63.4%    63.1%     -0.3       [-8.3; +7.8]    0.9474       [-8.0; +7.7]    0.6188
másodlagos  mass_g      50    17.8%    16.9%     -0.9       [-4.4; +2.6]    0.6072       [-4.2; +2.5]    0.4663

ELSŐDLEGES TESZT (m = 1, alfa = 0,05, páros t, delta-módszeres SE): igazolva: a P3_recipe szignifikánsan kisebb zsírhibát ad
delta = pMAE(P3_recipe) - pMAE(P3_macros), pp; negatív = a P3_recipe jobb. Döntés csak az elsődleges sor p (t) értékéből.

Az előre rögzített megerősítő tesztben (OmniFood8k, 50 friss étel, páros t-próba) a receptalapú prompt (P3_recipe) a P3_macros-hoz képest 4,9 százalékponttal csökkentette a zsírbecslés pMAE-jét (95% CI: 1,6–8,2 pp; p = 0,004).

Ami ebből nem következik, és a korlátok közé kell írni:

Csak az OmniFood8k-ra érvényes. A DFood-TJ-n a felderítés nem mutatott hatást (−0,5 pp), így általános „rejtett zsír” javításról nem beszélhetsz.
A zsírhiba így is nagy (55%). A javulás statisztikailag valós, de gyakorlatilag mérsékelt.
A szénhidrát nem javult szignifikánsan. A cukorbeteg-alkalmazás szempontjából ez a fontosabb cél, és erre továbbra sincs igazolt prompt.
Az OmniFood zsírcímkéinek megbízhatósága továbbra is nyitott kérdés.
