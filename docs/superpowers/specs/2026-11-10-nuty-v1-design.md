# BeatSaberMaker — Etap 1: generator nut v1 (spec)

Data: 2026-11-10 · Status: do akceptacji

## 1. Cel i kontekst

Narzędzie, które z dowolnej piosenki generuje grywalną mapę Beat Sabera. Budowane głównie
do własnego użytku i nauki, ale ma być produktem dobrej jakości. **Priorytet nr 1 to
grywalność map**; interfejs i dodatki są wtórne.

Ten spec obejmuje **Etap 1**: same nuty, jedna trudność na uruchomienie, CLI, model
wytrenowany na lokalnych mapach, scraper kuratorowanych danych działający w tle.

### Ustalenia z użytkownikiem

- Rdzeń neuronowy, architektura z dwóch modeli: timing (kiedy pada nuta) + patterny (jaka nuta).
- Twardy checker grywalności pilnuje wyniku podczas generowania.
- Sterowanie: suwak trudności (docelowa gęstość nut) i wybór stylu.
- Interfejs: CLI, mapa zapisywana prosto do `CustomLevels`.
- Dane: najpierw lokalne 1418 map, równolegle scraping kuratorowanych map z BeatSavera.
- Dane treningowe na dysku F:, kod w `C:\Users\alex\Desktop\BeatSaberMaker`.

### Fakty o danych lokalnych (pełny skan z 2026-11-10)

| | |
|---|---|
| Mapy / pliki trudności | 1418 / 3777, wszystkie się parsują |
| Pliki trudności Standard | 3439 (Easy 309, Normal 440, Hard 727, Expert 1004, Expert+ 959) |
| Audio | 75,9 h; 1415 × `.egg`, 2 × `.ogg`, 1 × `.wav` |
| Nuty | 2,66 mln |
| Formaty plików trudności | v2: 2966, v3: 785, v4: 26; 58 plików ma błędną lub brakującą etykietę wersji |
| Mapy ze zmianami BPM | 199 |
| Wymagane mody | Noodle 90, Chroma 49, Vivify 37, Mapping Extensions 24 |
| Duplikaty | 11 par tego samego klucza BeatSaver; 30 folderów bez klucza |
| Mediana gęstości (nuty/s) | Easy 1,36 · Normal 1,96 · Hard 2,92 · Expert 3,80 · Expert+ 4,74 |
| Środowisko | Beat Saber 1.37.1, Python 3.13.2, ffmpeg 8.1, RTX 4060 Ti 16 GB |

### Kryteria sukcesu Etapu 1

1. `bsmaker generate piosenka.mp3 --nps 4.5 --style balanced` tworzy folder w `CustomLevels`,
   który ładuje się w Beat Saberze 1.37.1 bez żadnych wymaganych modów.
2. Wygenerowana mapa ma zero twardych naruszeń grywalności według checkera (z konstrukcji).
3. Osiągnięta gęstość nut mieści się w ±10% wartości zadanej suwakiem.
4. Timing modelu na zbiorze walidacyjnym: F1 onsetów ≥ 0,70 przy tolerancji ±25 ms.
   To próg „model działa”, nie cel końcowy.
5. Użytkownik gra w VR co najmniej 3 wygenerowane mapy z różnych gatunków i spisuje uwagi,
   które zasilają Etap 2.

### Poza zakresem Etapu 1

- Bomby, ściany, arki, chainy (Etap 3).
- Światła (Etap 4), GUI z podglądem (Etap 5).
- Chroma / Noodle Extensions (Etap 6), Vivify z gotowych szablonów (Etap 7).
- Kilka trudności w jednym uruchomieniu, tryby inne niż Standard.
- Warunkowanie stylem konkretnego mappera.
- Poprawna siatka beatów dla utworów o zmiennym tempie (takie utwory dostają nuty
  niedociągnięte do siatki — nadal grywalne, ale trudniejsze do ręcznej edycji).
- Publikowanie map na BeatSaver. Gdyby kiedyś do tego doszło, mapy trzeba oznaczać jako
  wygenerowane przez AI zgodnie z zasadami serwisu.

## 2. Architektura

```
piosenka.mp3
   │  ffmpeg → mono 22,05 kHz → log-mel
   ▼
[audio]  ──────────────► wykrycie BPM i siatki beatów
   │
   ▼
[model timingu]  (mel, nps, styl) → lista czasów zdarzeń
   │
   ▼
[model patternów] (czasy, mel wokół zdarzeń, nps, styl) → nuty lewej i prawej ręki
   │        ▲
   │        └── [checker grywalności] maskuje niedozwolone nuty przy każdym kroku
   ▼
[eksport]  dociągnięcie do siatki → Info.dat + trudność v3 + song.egg + okładka
   ▼
CustomLevels\BSM - <tytuł> [nps X, styl]\
```

### Moduły

Pakiet `bsmaker` w `src/bsmaker/`. Każdy moduł ma jedną rolę i jest testowalny osobno.

| Moduł | Rola | Zależy od |
|---|---|---|
| `formats` | Czyta mapy v2/v3/v4 do wspólnej reprezentacji; zapisuje gotową mapę | — |
| `audio` | Dekodowanie przez ffmpeg, log-mel, wykrywanie BPM/beatów | — |
| `playability` | Symulator ruchu rąk: reguły twarde i miękkie, raport, maska dozwolonych nut | `formats` |
| `data` | Manifest map, filtry, podział na zbiory, cache cech, scraper BeatSavera | `formats`, `audio`, `playability` |
| `models.timing` | Model timingu, trening, inferencja | `data` |
| `models.pattern` | Model patternów, trening, próbkowanie z maską | `data`, `playability` |
| `generate` | Spina pipeline od pliku audio do folderu mapy | wszystkie powyższe |
| `cli` | Komendy `bsmaker ...` | wszystkie powyższe |

### Stack

- Python 3.13, środowisko wirtualne `.venv` (pip; `uv` nie jest zainstalowany).
- PyTorch z CUDA; trening w mixed precision.
- `numpy`, `soundfile`, `librosa` (mel), `httpx` (scraper), `typer` (CLI), `pytest`.
- Manifest danych w SQLite (biblioteka standardowa).
- Wykrywanie beatów: `beat_this`; jeśli nie zainstaluje się w tym środowisku, zastępuje go
  `librosa.beat.beat_track`. Zgodność PyTorch/CUDA i `beat_this` z Pythonem 3.13 jest
  sprawdzana w pierwszym zadaniu planu; w razie problemu środowisko schodzi na Python 3.12.

## 3. Wspólna reprezentacja mapy (`formats`)

```python
@dataclass(frozen=True)
class Note:
    time_s: float      # czas w sekundach od początku audio
    beat: float        # pozycja w beatach (wg BPM mapy)
    x: int             # kolumna 0–3 (od lewej)
    y: int             # rząd 0–2 (od dołu)
    color: int         # 0 = lewa (czerwona), 1 = prawa (niebieska)
    direction: int     # 0 góra, 1 dół, 2 lewo, 3 prawo, 4–7 skosy, 8 kropka

@dataclass
class Difficulty:
    characteristic: str          # "Standard", ...
    rank: str                    # "Easy" … "ExpertPlus"
    njs: float
    njs_offset: float
    notes: list[Note]
    bombs, obstacles, arcs, chains   # parsowane i liczone, w Etapie 1 nieużywane
    flags: DifficultyFlags       # np. has_bpm_changes, has_offgrid_notes, has_custom_note_data

@dataclass
class BeatMap:
    key: str | None              # klucz BeatSaver z nazwy folderu
    title, artist, mapper: str
    bpm: float
    audio_path: Path
    requirements, suggestions: list[str]
    difficulties: list[Difficulty]
```

**Czytanie.** Format wykrywany po obecnych kluczach, nie po polu wersji: `_notes` → v2,
`colorNotes` bez `colorNotesData` → v3, `colorNotesData` → v4. Parser toleruje BOM,
wielkość liter w nazwach plików (`Info.dat` / `info.dat`) i brakujące pola opcjonalne.
Błąd parsowania jednej trudności nie przerywa czytania pozostałych.

**Zmiany BPM.** Czytnik wykrywa je ze wszystkich źródeł (`_BPMChanges`, eventy typu 100,
`bpmEvents`, wieloregionowy `BPMInfo.dat`, `AudioData` w v4) i ustawia flagę
`has_bpm_changes`. W Etapie 1 takie mapy są wykluczane z treningu, więc konwersja
beat → sekundy zakłada stały BPM. Niezerowy `_songTimeOffset` też wyklucza mapę.

**Zapisywanie.** Folder mapy zawiera:
- `Info.dat` w schemacie v2.1.0, jedna trudność Standard, `_levelAuthorName: "BeatSaberMaker"`;
- plik trudności w schemacie v3.3.0 (tylko `colorNotes`; pozostałe tablice puste);
- `song.egg` — wejściowe audio przekodowane przez ffmpeg do Ogg Vorbis;
- `cover.png` — prosta, generowana okładka.

Ranga trudności w `Info.dat` to ta, której mediana gęstości z danych jest najbliższa
zadanemu `--nps`. NJS i offset pochodzą z tabeli median policzonej na danych
treningowych w przedziałach gęstości.

## 4. Dane (`data`)

### Układ na dysku

```
F:\BeatSaberMakerData\
  manifest.sqlite          # mapy, trudności, metadane BeatSaver, werdykty filtrów
  raw\<key>\               # pobrane mapy: pliki .dat + audio.ogg (mono 22,05 kHz, ~64 kb/s)
  features\<key>.npy       # log-mel, float16
  checkpoints\             # wagi modeli
  reports\                 # raporty ewaluacji i kalibracji
```

Lokalny folder `CustomLevels` jest źródłem **tylko do odczytu**. Jedyny zapis do niego to
nowe foldery z prefiksem `BSM - ` tworzone przez `generate`.

### Cechy audio

Mono 22 050 Hz, log-mel: 80 pasm, okno 1024, krok 256 próbek (11,6 ms, ~86 klatek/s).
Około 2,7 MB na utwór, ~4 GB dla danych lokalnych. Parametry są zapisane w nagłówku cache;
zmiana parametrów unieważnia cache.

Dla pobranych map oryginalne audio jest zastępowane kompaktową kopią mono (~1,6 MB/utwór),
żeby zmieścić 10–20 tys. map na F: i móc później przeliczyć cechy bez ponownego pobierania.

### Filtry plików trudności

Plik trafia do treningu, jeśli spełnia wszystkie warunki. Każde odrzucenie zapisuje powód
w manifeście.

1. Charakterystyka `Standard`.
2. Mapa bez wymaganych modów (sugestie typu Chroma są dozwolone — dotyczą świateł).
3. Wszystkie nuty na siatce 4×3 i z kierunkiem 0–8; brak danych Heck/Noodle na nutach.
4. Stały BPM, zerowy offset audio.
5. Co najmniej 80 nut; długość utworu 60–600 s; gęstość 0,5–12 nut/s.
6. Nie jest oznaczona na BeatSaver jako automapper/AI.
7. Udział twardych naruszeń checkera ≤ 2% nut (próg w konfiguracji, patrz sekcja 6).
8. Deduplikacja po kluczu i hashu mapy.

### Uproszczenia reprezentacji treningowej

- Gęstość (`nps`) = liczba nut / czas między pierwszą a ostatnią nutą.
- Czasy nut zaokrąglane do klatek mela (11,6 ms).
- Kilka nut tego samego koloru w odstępie < 40 ms (stacki, wieże, slidery z kropek) liczy
  się jako jedno cięcie: zostaje pierwsza nuta. Udział takich przypadków jest raportowany.

### Podział na zbiory

Po utworze (`song_id`): mapy tego samego utworu (ten sam znormalizowany tytuł i artysta albo
ten sam tytuł i czas różniący się o ≤ 2 s) trafiają do jednego zbioru, więc wszystkie trudności
wszystkich takich map są razem. Deterministycznie z hasha `song_id`: 90% trening, 5% walidacja,
5% test. Filtr grywalności przy budowie danych używa profilu `data`, a generowanie (maska) profilu
`mask`.

### Style

Słownik stylów pochodzi z tagów BeatSavera: `tech`, `dance`, `speed`, `accuracy`,
`balanced`, `challenge`, `fitness`, plus `unknown`. Mapa może mieć kilka tagów.
Metadane dla lokalnych map pobierane są po kluczu z nazwy folderu (1388 z 1418).
Mapy bez tagów dostają `unknown`; dodatkowo w treningu styl jest losowo zastępowany przez
`unknown` w 15% przykładów, żeby model działał też bez podanego stylu.

### Scraper BeatSavera

- Publiczne API `api.beatsaver.com`. Dokładne endpointy, pola i limity zapytań są
  weryfikowane z dokumentacją API w pierwszym kroku zadania scrapera.
- Zbiór „kuratorowany” = suma: mapy rankedowe (ScoreSaber lub BeatLeader), mapy oznaczone
  jako curated, mapy z oceną ≥ 85% przy ≥ 100 głosach. Bez map oznaczonych jako automapper/AI.
  Cel: 10–20 tys. map.
- Grzeczny klient: ograniczenie tempa, backoff przy błędach 429/5xx, najwyżej 2 pobierania
  naraz, nagłówek `User-Agent: BeatSaberMaker/0.1` bez danych osobowych.
- Wznawialny: stan w manifeście, przerwane pobieranie kontynuuje od miejsca przerwania.
- Bezpiecznik miejsca: zatrzymuje się, gdy na F: zostaje mniej niż 20 GB.
- Pobrane archiwa to dane niezaufane: rozpakowywane tylko pliki `.dat` i audio, z kontrolą
  ścieżek (bez wychodzenia poza folder docelowy) i limitem rozmiaru; nic z nich nie jest
  wykonywane.

## 5. Modele

Interfejsy są ustalone; rozmiary i hiperparametry to wartości startowe do strojenia.

### 5.1 Model timingu

```python
predict(mel, nps: float, styles: set[str]) -> list[float]   # czasy zdarzeń w sekundach
```

- **Zdarzenie** = moment, w którym pada co najmniej jedna nuta (pojedyncza lub dubel).
- **Wejście:** log-mel w oknach ~12 s (1024 klatki), z zakładką przy inferencji.
- **Architektura:** stem konwolucyjny redukujący oś częstotliwości → 6 warstw enkodera
  transformera (d=256, dwukierunkowy) → logit zdarzenia na klatkę. Około 5 mln parametrów.
- **Warunkowanie:** embedding `log(nps)` i embedding stylów dodawane do każdej klatki.
- **Trening:** BCE z rozmytymi etykietami (sąsiednie klatki z wagą 0,25) i wagą klasy
  pozytywnej.
- **Dekodowanie:** wybór lokalnych maksimów powyżej progu. Próg jest dobierany
  wyszukiwaniem binarnym tak, żeby liczba zdarzeń odpowiadała zadanemu `nps`
  (z poprawką na oczekiwany udział dubli z tabeli policzonej na danych).

### 5.2 Model patternów

```python
sample(event_times, mel, nps, styles, seed, checker) -> list[Note]
```

- **Sekwencja:** jedno zdarzenie = para tokenów (lewa ręka, prawa ręka).
  Słownik jednej ręki: 109 tokenów = `brak` + 4 kolumny × 3 rzędy × 9 kierunków.
  W każdym zdarzeniu co najmniej jedna ręka ma nutę.
- **Wejście na zdarzenie:** odstęp od poprzedniego zdarzenia, odstępy do 4 następnych
  (czasy są znane z góry), wektor audio z małej sieci konwolucyjnej nad wycinkiem mela
  ±0,2 s wokół zdarzenia, tokeny poprzedniego zdarzenia.
- **Architektura:** przyczynowy transformer (8 warstw, d=384, kontekst 256 zdarzeń),
  około 15 mln parametrów. Wyjście sfaktoryzowane: najpierw lewa ręka, potem prawa
  pod warunkiem lewej.
- **Warunkowanie:** prefiks z embeddingiem `log(nps)` i stylów.
- **Trening:** cross-entropy z teacher forcing na przefiltrowanych mapach ludzkich.
- **Próbkowanie:** temperatura + top-p. Przed losowaniem każdego tokenu checker zwraca
  maskę dozwolonych nut; niedozwolone dostają prawdopodobieństwo zero.
- **Ślepy zaułek:** jeśli maska blokuje wszystkie nuty danej ręki, ręka dostaje `brak`;
  jeśli obie ręce mają `brak`, zdarzenie jest pomijane i liczone w raporcie.
- **Warianty:** `--seed` daje inną mapę dla tej samej piosenki.

### 5.3 Pętla gęstości

Po wygenerowaniu liczona jest faktyczna gęstość. Jeśli odbiega od zadanej o więcej niż 10%,
próg timingu jest korygowany i generowanie powtarzane, najwyżej 3 razy.

### 5.4 Siatka beatów przy eksporcie

Modele pracują w sekundach i nie zależą od BPM. Przy eksporcie:
1. Wykrywane są beaty; dopasowywany jest jeden BPM i przesunięcie.
2. Jeśli dopasowanie jest dobre (stałe tempo), czasy nut są dociągane do najbliższej
   pozycji siatki 1/16 lub 1/12 beatu, o ile różnica nie przekracza 25 ms.
3. Jeśli tempo jest zmienne lub wykrycie niepewne, nuty zostają w oryginalnych czasach,
   a do `Info.dat` trafia mediana BPM. Komenda wypisuje ostrzeżenie.

## 6. Checker grywalności (`playability`)

Deterministyczny symulator bez ML. Stan każdej ręki: ostatnia pozycja, ostatni kierunek
cięcia, parity (forehand/backhand), czas ostatniego cięcia.

```python
check(difficulty) -> Report                      # naruszenia twarde i miękkie z czasami
allowed(state, time_s, hand) -> mask[109]        # dozwolone tokeny dla następnej nuty
advance(state, time_s, left_token, right_token) -> state
```

### Reguły twarde (maskowane przy generowaniu)

1. **Parity.** Kolejne cięcia tej samej ręki muszą iść naprzemiennie. Nowy kierunek musi
   leżeć w zakresie ±45° od kierunku przeciwnego do poprzedniego cięcia (±90° przy wolnym
   tempie). Kropka dziedziczy oczekiwany kierunek.
2. **Reset tylko z czasem.** Powtórzenie kierunku tą samą ręką (double directional) jest
   dozwolone dopiero po przerwie dłuższej niż próg resetu.
3. **Limit szybkości ręki.** Minimalny odstęp między cięciami tej samej ręki, zależny od
   odległości między nutami.
4. **Duble.** Dwie nuty w jednym zdarzeniu nie mogą zajmować tej samej komórki, nie mogą
   tworzyć handclapu (cięcia do siebie w sąsiednich komórkach) ani skrzyżowania rąk.
5. **Zasłanianie.** Nuta w środkowych komórkach środkowego rzędu nie może bezpośrednio
   poprzedzać innej nuty w odstępie krótszym niż próg widoczności.
6. **Rzadkie kombinacje.** Kombinacje (ręka, kolumna, rząd, kierunek), które w ludzkich
   mapach występują rzadziej niż 0,01%, są blokowane (np. cięcia wymagające nienaturalnego
   skrętu nadgarstka).

### Reguły miękkie (tylko metryki)

Resety po przerwie, duże skoki pozycji przy krótkim odstępie, nierównowaga rąk, długie
serie w jednym rzędzie, nuty odwrócone (inwerty).

### Kalibracja

Progi (kąty, czasy, częstości) są w konfiguracji. Wartości startowe ustawiane są z
percentyli zmierzonych na ludzkich mapach. **Warunek poprawności checkera:** co najmniej
95% lokalnych plików Standard bez wymaganych modów ma ≤ 1% twardych naruszeń. Jeśli
checker odrzuca więcej, to reguły są zbyt ostre i wymagają korekty, zanim zostaną użyte
jako filtr danych i maska. Raport kalibracji trafia do `reports\`.

## 7. CLI

| Komenda | Działanie |
|---|---|
| `bsmaker data index-local` | Skan `CustomLevels` do manifestu, filtry, statystyki |
| `bsmaker data fetch-metadata` | Tagi, oceny i statusy z BeatSavera dla znanych kluczy |
| `bsmaker data scrape` | Pobieranie zbioru kuratorowanego (wznawialne) |
| `bsmaker data features` | Liczenie i cache log-meli |
| `bsmaker check <folder mapy>` | Raport grywalności dowolnej mapy |
| `bsmaker train timing` / `train pattern` | Trening z checkpointami i wznawianiem |
| `bsmaker eval` | Metryki na zbiorze walidacyjnym/testowym, raport |
| `bsmaker generate <audio> --nps N [--style S ...] [--seed K] [--out DIR] [--overwrite]` | Generowanie mapy |
| `bsmaker preview <folder mapy>` | Plik audio z kliknięciami w momentach nut — szybka kontrola synchronizacji bez VR |

Ścieżki (`CustomLevels`, katalog danych, checkpointy) są w pliku `bsmaker.toml`
w katalogu projektu.

## 8. Obsługa błędów

- **Dane:** uszkodzona mapa lub audio → wpis w manifeście z powodem, przetwarzanie idzie dalej.
- **Scraper:** ponowienia z backoffem; po wyczerpaniu prób mapa oznaczona jako nieudana
  i pomijana; brak miejsca → czyste zatrzymanie z komunikatem.
- **Trening:** checkpoint co ustaloną liczbę kroków, wznawianie po przerwaniu.
- **Generowanie:** czytelny komunikat i niezerowy kod wyjścia przy braku ffmpeg, braku
  checkpointu, nieobsługiwanym pliku audio lub utworze krótszym niż 30 s. Istniejący folder
  wyjściowy nie jest nadpisywany bez `--overwrite`, a `--overwrite` działa tylko na
  folderach z prefiksem `BSM - `.
- Ostrzeżenia (niepewny BPM, pominięte zdarzenia, gęstość poza tolerancją) są wypisywane
  w podsumowaniu generowania, a nie ukrywane.

## 9. Testy i ewaluacja

### Testy automatyczne (pytest)

- `formats`: małe syntetyczne mapy v2/v3/v4 w repo (bez cudzych utworów); zapis → odczyt
  daje te same nuty; wykrywanie formatu po kluczach; flagi zmian BPM i nut poza siatką.
- `playability`: ręcznie napisane wzorce dobre (naprzemienny stream, poprawne duble) i złe
  (double directional, handclap, zasłonięcie) z oczekiwanym wynikiem dla każdej reguły;
  własność: sekwencja zbudowana wyłącznie z `allowed()` ma zero twardych naruszeń w `check()`.
- `data`: każdy filtr na spreparowanym przykładzie; deterministyczny podział; unieważnianie cache.
- `scraper`: na nagranych odpowiedziach API, bez sieci; test bezpiecznego rozpakowywania.
- `models`: test przeuczenia na 2 mapach (strata spada do zera) na CPU; kształty tensorów;
  maskowane próbkowanie nigdy nie zwraca zablokowanego tokenu.
- `generate`: test end-to-end z malutkim modelem o losowych wagach na syntetycznym audio —
  powstaje folder, który nasz czytnik parsuje bez błędów.

### Metryki modeli

- Timing: precision/recall/F1 onsetów (±25 ms) na walidacji przy gęstości mapy ludzkiej.
- Patterny: cross-entropy na walidacji; porównanie rozkładów map wygenerowanych i ludzkich
  (pozycje, kierunki, udział dubli, równowaga rąk); częstość reguł miękkich; częstość
  ślepych zaułków.
- Całość: generowanie dla 10 utworów testowych przy gęstości mapy ludzkiej, raport w `reports\`.

### Test ręczny

`bsmaker preview` do odsłuchu synchronizacji, potem gra w VR. Uwagi użytkownika są
zapisywane w `docs/feedback/` i stanowią wejście do Etapu 2.

## 10. Kolejność budowy

| # | Kamień milowy | Wynik do sprawdzenia |
|---|---|---|
| M0 | Środowisko: venv, PyTorch+CUDA, szkielet pakietu, pytest | `pytest` przechodzi, GPU widoczne w PyTorch |
| M1 | `formats` | Wszystkie 3777 lokalnych plików się czyta; round-trip na fixture'ach |
| M2 | `audio` + `data` (manifest, filtry, cechy dla danych lokalnych) | Manifest ze statystykami i powodami odrzuceń |
| M3 | `playability` + raport kalibracji | Warunek 95% spełniony na danych lokalnych |
| M4 | Scraper + metadane lokalnych map | Tagi stylu w manifeście; pobieranie kuratorowanych map rusza w tle |
| M5 | Model timingu | F1 ≥ 0,70 na walidacji |
| M6 | Model patternów | Próbki bez twardych naruszeń, rozkłady zbliżone do ludzkich |
| M7 | `generate`, `preview`, CLI | Pierwsza mapa w `CustomLevels` |
| M8 | Raport ewaluacji + test w VR | Lista uwag do Etapu 2 |

M1 jest pierwsze, bo definiuje wspólną reprezentację. Potem M2, M3 i M4 mogą iść
równolegle (osobni agenci, rozłączne pliki), podobnie M5 i M6 po zakończeniu M2 i M3.

## 11. Dalsze etapy (poza tym specem)

2. Trening na danych kuratorowanych, większe modele, poprawki po testach w VR.
3. Bomby, ściany, arki, chainy.
4. Światła (osobny model; 1285 lokalnych map ma sensowny lightshow).
5. GUI z podglądem.
6. Chroma / Noodle Extensions.
7. Vivify z biblioteki gotowych szablonów sterowanych strukturą utworu.
