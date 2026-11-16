# Etap 3 — pomiary w danych (ściany, bomby, łuki, łańcuchy)

Data: 2026-11-16. Materiał wejściowy do specyfikacji Etapu 3, nie specyfikacja. Pomiar zrobił
agent (sam odczyt); skrypty i surowe wyniki: `%TEMP%\bsm-stage3-survey\` (`collect.py`,
`analyze.py`, `analysis.txt`). Reguły klas to interpretacja agenta, nie pomiar.

Metoda: próba 6000 map (ziarno 0, najwyższa przyjęta trudność z mapy; v4 pominięte — 599
trudności w całym zbiorze); liczby „cały zbiór" to pełny przebieg po 39 903 przyjętych
trudnościach. Sekundy liczone stałym BPM z Info.

## 1. Jak często występują

| | bomby | ściany | łuki | łańcuchy |
|---|---|---|---|---|
| udział trudności z co najmniej jednym | 51,9% | 90,6% | 25,4% | 10,0% |
| na minutę, mediana / p90 (gdy są) | 9,6 / 56 | 22,8 / 112 | 18,4 / 39 | 3,7 / 18 |
| tylko v3 | 51,7% | 93,8% | 87,8% | 34,7% |

Łuki i łańcuchy istnieją tylko w formacie v3 (nowsze mapy).

## 2. Ściany

- 52% ścian trwa krócej niż 0,05 s (mediana 0,04 s = 1/8 bitu) — to błyski, nie przeszkody.
- 86% ścian v2 i 61% ścian v3 to pojedyncza skrajna kolumna (0 albo 3).
- v3: 19% ścian leży poza siatką (x = −1 lub 4), mimo że mapy z rozszerzeniami są odrzucane.
- v2 (70% wszystkich ścian) nie ma rzędu ani wysokości: typ 0 = pełna, typ 1 = do kucania.
- Jednocześnie na ekranie: mediana 2, p90 3–4.

Klasy (reguły po kolei): **błysk** — czas ≤ 0 lub < 0,05 s; **dekoracja boczna** — nie obejmuje
kolumn 1 ani 2; **kucanie** — obejmuje kolumny 1 i 2 od rzędu 2 w górę; **zawężenie toru** —
w trakcie ≥ 3 nuty w wolnych polach i ≥ 2 nuty/s; **unik** — reszta.

| klasa | udział ścian | mapy ze ścianami, które ją mają | ściany w całym zbiorze | map |
|---|---|---|---|---|
| błysk | 52,5% | 70,0% | 2 892 923 | 11 564 |
| dekoracja boczna | 41,4% | 91,5% | 2 343 489 | 14 723 |
| unik | 4,8% | 57,1% | 247 478 | 9 822 |
| zawężenie toru | 0,8% | 26,2% | 30 929 | 4 713 |
| kucanie | 0,4% | 13,1% | 20 727 | 2 337 |

Ściany rozgrywkowe to ok. 6% wszystkich. Nie stoją w przerwach między nutami: 54% ścian
środkowych nie ma w trakcie żadnych nut, ale lokalna gęstość nut wokół nich jest typowa
(mediana 0,95 średniej mapy). Tylko 0,1% nachodzi na nuty.

## 3. Bomby

- Pozycja: 67% w dolnym rzędzie, 21% w górnym, 12% w środkowym.
- Tylko 5,7% bomb dzieli czas z nutą. Mediana odległości od najbliższej nuty: 0,63 bitu.
- Grupy jednoczesne: pojedyncza 41%, dwie 23%, trzy 9,5%, cztery 19,5%, pięć i więcej 7%.

Klasy: **wzór/dekoracja** — grupa ≥ 3 naraz, linia w rzędzie lub kolumnie, albo ≥ 3 bomby
w tym samym polu w ±1 bit; **reset** — bomba w polu poprzedniej nuty ręki lub tuż „za nią"
(do 1 bitu po nucie); **blokada toru** — bomba na przedłużeniu zamachu; **inne**.

| klasa | udział bomb | bomby w całym zbiorze | map |
|---|---|---|---|
| wzór/dekoracja | 79,5% | 1 082 303 | 8 287 |
| inne | 17,6% | 255 836 | 6 555 |
| reset | 2,4% | 28 964 | 3 299 |
| blokada toru | 0,4% | 4 398 | 970 |

Progi (okno 1 bitu, pola) mocno wpływają na podział reset/blokada.

## 4. Łuki i łańcuchy (v3)

- Łuki: 561 986 w 4 383 mapach. Długość: mediana 1 bit (p90 2). Nuta na początku w 98,4%,
  na końcu tylko w 18,8%.
- Łańcuchy: 74 715 w 1 779 mapach. Nuta na początku w 99,7%. Liczba segmentów: 5 (33%),
  4 (32%), 6 (11%). Czas trwania to stała narzędzia (ok. 0,04 bitu).
- Danych na łuki jest dużo; łańcuchów ok. 7× mniej, ale wystarczająco.

## 5. Wnioski dla projektu

- Dekoracja i rozgrywka muszą być osobnymi torami: dekoracja to 94% ścian i 80% bomb.
- Elementów rozgrywkowych jest mało (ok. 300 tys. ścian, ok. 33 tys. bomb reset/blokada) —
  dla nich reguły mogą być lepsze niż model.
- Łuki zawsze zaczynają się na nucie, więc są dodatkiem do modelu wzorców, nie osobnym bytem.
- Ściany poza siatką (x = −1, 4) w v3 trzeba albo odtwarzać, albo świadomie pominąć.
- Korelacja liczby ścian ze światłami jest słaba (0,19).
