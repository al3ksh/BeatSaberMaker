# Etap 4 — pomiary w danych (światła i tła)

Data: 2026-11-16. Materiał wejściowy do specyfikacji świateł, nie specyfikacja. Pomiar zrobił
agent (sam odczyt); skrypty: `%TEMP%\bsm-stage4-survey\` (`extract.py`, `analyze.py`,
`extra.py`). Próba: 5000 map z przyjętą trudnością (ziarno 0, najwyższa przyjęta trudność).
Odczyt v4 (58 map) niezweryfikowany ze specyfikacją formatu. Wnioski z punktu 6 to ocena agenta.

## 1. Ile świateł mają mapy

| format | map | bez zdarzeń | < 50 zdarzeń | zdarzeń/min p10 / p50 / p90 | światła grupowe v3 |
|---|---|---|---|---|---|
| v2 | 3541 | 1,4% | 1,8% | 253 / 936 / 2800 | — |
| v3 | 1401 | 2,0% | 6,0% | 160 / 902 / 2602 | 15,6% |
| v4 | 58 | 12% | 47% | 0 / 19 / 957 | 79% |
| razem | 5000 | 1,7% | 3,5% | 235 / 916 / 2748 | 5,3% |

Mediana to ok. 15 zdarzeń na sekundę. 36,8% zdarzeń ma dane kolorów Chroma (kilka map daje
większość); 14,1% map ma takie zdarzenia. Tylko 62% map ma identyczne światła we wszystkich
trudnościach.

## 2. Czego używają

- Typy: pierścienie (1) 26%, lasery lewe/prawe, światła tylne i środkowe po ok. 13%, obrót
  pierścieni (8) 8%, prędkości laserów (12, 13) po 4%, zoom (9) 1%.
- Wartości świateł: włącz 39,5%, zanikanie 28,9%, wyłącz 23,4%, błysk 5,4%.
- Kolor: czerwony 47,7%, niebieski 27,4%, biały 1%.
- Obrót pierścieni: 89% map, mediana 48/min. Zoom: 68% map, mediana 18/min.

## 3. Związek z muzyką i nutami

| miara | wynik | kontrola |
|---|---|---|
| zdarzenia świateł w 30 ms od nuty | 51,0% | 28,5% (po przesunięciu o 233 ms) |
| w 30 ms od 1/4 bitu | 75,8% | ok. 48% losowo |
| korelacja gęstości świateł i nut w oknach 8 s (mediana po mapach) | 0,75 | — |
| siła onsetu w chwili zdarzenia (× średnia) | 2,54 | losowo 1,47; nuty 3,46 |

## 4. Struktura

- Średnio 2,65 zdarzenia na znacznik czasu; 57% znaczników ma ≥ 2 zdarzenia.
- Udział zdarzeń w taktach identycznych z wcześniejszym taktem: mediana 0,17 (p90 0,57).
- Jawne „wyłącz" następuje po 27% zdarzeń włączających; reszta to łańcuchy włącz/zanikaj.

## 5. Jakość i tła

- Ocena i kuracja nie przewidują bogatszych świateł.
- Mapy z gęstym, pełnym pokazem na starych grupach świateł: ok. 15 000 w całym zbiorze.
  Mapy ze światłami grupowymi v3: ok. 930.

| tło | udział |
|---|---|
| Default | 41,1% |
| BigMirror | 14,7% |
| Nice | 6,4% |
| Panic | 4,7% |
| BTS | 3,8% |
| Origins | 3,6% |
| Triangle | 3,4% |
| FitBeat | 3,1% |
| Dragons | 2,7% |
| Timbaland | 2,3% |

48 różnych teł. Tagi gatunku prawie nie przewidują tła (Default wygrywa w każdym tagu).
Najnowsze tła (TheSecond, RockMixtape, Lizzo, Panic2, TheWeeknd) działają na światłach
grupowych v3 — stare zdarzenia dają w nich pusty lub ubogi pokaz. Pozostałe popularne tła
działają na pięciu starych grupach. 38,7% map ma własne kolory.

## 6. Wnioski (ocena)

- Danych na stare grupy świateł jest bardzo dużo; model zdarzeń warunkowany dźwiękiem i nutami
  jest realny dla pięciu grup plus obrót i zoom pierścieni.
- Światła grupowe v3 mają za mało danych na model — reguły/szablony albo pominięcie.
- Sensowny start: tła działające na starych grupach (Default, BigMirror, Nice, Panic, Origins,
  Triangle, Dragons, Timbaland, BTS, Kaleidoscope — razem ok. 85% map).
- Sama reguła „światło na nucie/bicie" nie odda ludzkiej różnorodności (gęstość p10–p90 od
  235 do 2750 zdarzeń/min); potrzebne sterowanie gęstością/stylem pokazu.
