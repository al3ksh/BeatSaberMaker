# BeatSaberMaker — Etap 2: drugi trening (spec)

Data: 2026-11-13. Poprzedni spec: `2026-11-10-nuty-v1-design.md` (Etap 1, zamknięty).

## 1. Cel

Wytrenować drugą generację obu modeli (rytm i wzorce) tak, żeby poprawić trzy rzeczy
zmierzone po Etapie 1, i zrobić to na większym, lepiej ważonym zbiorze:

1. **Ręka siedząca nisko.** Model wzorców dwa razy częściej niż ludzie tnie w górę na dolnym
   rzędzie (20,5% nut wobec 10,8%) i o połowę rzadziej używa skosów w środkowym rzędzie.
2. **Słaby rytm na rzadkich mapach.** F1 onsetów 0,48–0,56 poniżej 1,5 nps (średnio 0,76).
3. **Niewykorzystane dane.** 7 882 trudności odrzucone wyłącznie za „zmiany BPM”.

Stan wyjściowy (modele generacji 1, zbiór walidacyjny): rytm F1@25 ms 0,759, wzorce CE 1,802,
odległość rozkładu kierunków od ludzkiego (TV) 0,197 po poprawce losowania.

## 2. Kryteria sukcesu

1. Nowy model rytmu: F1@25 ms na walidacji nie niższe niż 0,759 i F1 w koszyku 0,5–1,5 nps
   wyższe niż 0,559.
2. Nowy model wzorców, próbkowany na ludzkich czasach nut (40 utworów walidacyjnych, te same
   ustawienia losowania co w produkcji): udział cięć w górę na dolnym rzędzie bliżej ludzkiego
   niż 20,5%, a TV łącznego rozkładu (rząd × kierunek) niższe niż u generacji 1; zero twardych
   naruszeń grywalności.
3. Zbiór testowy jest użyty raz, na końcu, dla wybranej pary modeli.
4. Użytkownik ogląda lub gra co najmniej 3 mapy z nowych modeli i mówi, czy jest lepiej.
   Jeśli nie jest, zostają modele generacji 1 (nie są nadpisywane).

## 3. Zakres

### 3.1 Mapy ze zmianami BPM

Podział 7 882 trudności według sposobu zapisu (zmierzone):

| Zapis | Trudności | Decyzja |
|---|---|---|
| v2 `_customData._BPMChanges` | 4 039 | Przyjąć ze stałym BPM. Założenie: to siatka edytora, czasy nut są w bitach BPM z `Info.dat`. Sprawdzane kontrolą z 3.2. |
| v3 `bpmEvents` (w tym z `BPMInfo.dat`) | 3 171 | Przeliczyć czasy nut odcinkami według `bpmEvents`. |
| v2 zdarzenia typu 100 | 378 | Przeliczyć jak v3 (`_time`, `_floatValue`). |
| tylko `BPMInfo.dat` / `AudioData.dat` | 284 | Zostają odrzucone (`tempo_external`). |

Trudność z jakąkolwiek wartością BPM poniżej 20 lub powyżej 1000 zostaje odrzucona
(`bpm_trick`, ok. 1 300 trudności) — to sztuczki mapperów, nie tempo.

### 3.2 Kontrola zgodności z dźwiękiem

Źle przeliczone czasy zatrułyby model rytmu, więc żadna z tych map nie wchodzi do zbioru bez
pomiaru. Dla każdej kandydującej trudności liczona jest **zgodność z dźwiękiem**: średnia
wartość obwiedni onsetów (strumień widmowy z zapisanego mel-spektrogramu) w klatkach nut,
podzielona przez średnią obwiedni całego utworu. Próg to 5. percentyl tej miary na próbie
1 000 już przyjętych trudności treningowych. Trudność poniżej progu zostaje odrzucona
(`tempo_unverified`).

Dla trudności z mapą tempa liczone są oba warianty (stałe BPM i mapa tempa) i raportowane
obok siebie; używany jest wariant z mapą tempa, a jeśli stałe BPM wypada wyraźnie lepiej na
większości takich map, to jest błąd w konwersji i zadanie wraca do poprawki, zamiast
przyjmować dane.

### 3.3 Podział zbiorów

Bez zmian i bez `data resplit`. Odrzucone dotąd trudności mają już przypisany zbiór według
piosenki; zachowują go. Dzięki temu walidacja i test pozostają porównywalne z generacją 1,
z jednym zastrzeżeniem: zbiory walidacyjny i testowy też urosną o odzyskane mapy, więc
porównania z generacją 1 liczone są na **starej liście** trudności walidacyjnych (zapisanej
przed przebudową cache) i dodatkowo na nowej.

### 3.4 Ważenie próbek

Dziś każda trudność jest losowana równie często. Nowa waga trudności to iloczyn:

- **jakości:** `0,25 + 0,75 · clip((score − 0,80) / 0,15, 0, 1)`, gdzie `score` to ocena
  z BeatSavera (mediana 0,935); brak oceny → 0,6;
- **wyrównania gęstości:** `(mediana liczności koszyków nps / liczność koszyka)^0,5`,
  obcięte do [0,5; 3] — rzadkie koszyki (0,5–1,5 nps to 4,4% zbioru) są losowane częściej.

Wagi trafiają do losowania z powtórzeniami w pętli treningowej. Walidacja nie jest ważona.

### 3.5 Model wzorców odporny na własne błędy

W treningu model widzi wyłącznie prawdziwe poprzednie nuty, a przy generowaniu własne.
W nowym treningu każda pozycja wejścia „poprzednia nuta” jest z prawdopodobieństwem `p`
podmieniana (obie ręce naraz) na parę wziętą z innej, losowej pozycji tej samej sekwencji.
Cele uczenia się nie zmieniają. Domyślnie `p = 0,1`; jeden przebieg kontrolny z `p = 0`.

### 3.6 Rozmiary modeli

| Model | Rozmiar | d_model / warstwy / głowice | Parametry (ok.) |
|---|---|---|---|
| wzorce | S (obecny) | 384 / 8 / 6 | 15 mln |
| wzorce | M | 512 / 12 / 8 | 38 mln |
| wzorce | L | 768 / 12 / 12 | 85 mln |
| rytm | S (obecny) | 256 / 6 / 4 | 5 mln |
| rytm | M | 384 / 8 / 6 | 14 mln |

Kolejność przebiegów: wzorce S (p = 0,1), wzorce S (p = 0), wzorce M, rytm S, rytm M.
Wzorce L tylko wtedy, gdy M wygrywa z S na walidacji. Wybór modelu wzorców: najniższe TV
(rząd × kierunek) spośród przebiegów, których CE nie jest gorsze od najlepszego o więcej
niż 0,03. Wybór modelu rytmu: najwyższe F1@25 ms.

### 3.7 Ewaluacja

`bsmaker eval` dostaje: ścieżki checkpointów jako opcje (porównywanie przebiegów), listę
trudności z pliku (stara lista walidacyjna), F1 rytmu także przy ±10 ms, oraz sekcję
„wzorce na ludzkich czasach”: łączny rozkład rząd × kierunek z TV, udział cięć w górę na
dolnym rzędzie i w dół na górnym.

### 3.8 Trening przez Colab CLI

Trening odpalany z Raspberry Pi użytkownika narzędziem `colab` (zalogowane 2026-11-12).
Skrypty działające na maszynie Colab: przygotowanie (rozpakowanie starego archiwum danych
i paczki uzupełniającej z Dysku, kod z zipa) i pojedynczy przebieg treningu z checkpointami
na Dysku. Podpięcie Dysku (`colab drivemount`) wykonuje użytkownik — narzędzie nie pozwala
zrobić tego bez człowieka. Przebieg treningu jest trzymany jako aktywne polecenie w `tmux`
na Pi, żeby Colab nie uznał środowiska za bezczynne; po zerwaniu wznawia się z `last.pt`.

Paczka uzupełniająca zawiera tylko cechy nowych map, nowy cache zdarzeń i statystyki
(ok. 10 GB); stare archiwum 40 GB zostaje na Dysku bez zmian.

Budżet: do 120 jednostek obliczeniowych (stan 381). Każde środowisko jest zatrzymywane po
przebiegu, a stan jednostek raportowany przed i po.

## 4. Poza zakresem

- Bomby, ściany, łuki, łańcuchy; światła; GUI (Etapy 3–5 bez zmian).
- Trening z własnych próbek modelu (scheduled sampling) — tylko jeśli podmiana z 3.5 nie pomoże.
- Mapy tylko z `BPMInfo.dat` / `AudioData.dat`.
- Zmiany w generatorze poza wczytaniem nowych checkpointów.

## 5. Ryzyka

| Ryzyko | Zabezpieczenie |
|---|---|
| Błędna konwersja tempa zatruwa dane | Kontrola z 3.2, raport obu wariantów, próg z przyjętych map |
| Brak miejsca na F: (17 GB wolne, nowe cechy ok. 9 GB) | Użytkownik usuwa `upload\` (40 GB) przed liczeniem cech |
| Większy model się przeucza | Wybór po walidacji; L tylko warunkowo; generacja 1 zostaje |
| Colab zabiera środowisko | Checkpoint co 1000 kroków na Dysku, automatyczne wznawianie |
| Statystyki zbioru zmieniają się po przebudowie | Są zapisane w checkpointach; generacja 1 używa swoich |
