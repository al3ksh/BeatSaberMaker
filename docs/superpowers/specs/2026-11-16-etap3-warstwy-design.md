# BeatSaberMaker — Etap 3: warstwy (łuki, łańcuchy, ściany, bomby, światła) — spec

Data: 2026-11-16. Wcześniejsze: `2026-11-10-nuty-v1-design.md` (Etap 1),
`2026-11-13-etap2-trening-design.md` (Etap 2). Pomiary wejściowe:
`2026-11-16-etap3-pomiary.md`, `2026-11-16-etap4-pomiary.md`.

## 1. Cel

Wygenerowana mapa ma, oprócz nut, trzy opcjonalne warstwy uczone na ludzkich mapach:

1. **Ozdoby nut** — łuki (arcs) i łańcuchy (chains).
2. **Dekoracja** — ozdobne ściany i bomby; do tego rzadkie, łagodne przeszkody rozgrywkowe z reguł.
3. **Światła** — pokaz na klasycznych grupach świateł, z wyborem tła.

Dochodzi też zapis kilku trudności w jednej mapie. Modele nut (generacja 2) zostają bez zmian.

## 2. Ustalenia z użytkownikiem (wiążące)

- Ściany i bomby to głównie ozdoba; rozgrywkowo tylko czasem (kucnij, przesuń się), nic intensywnego.
- Map typu fitbeat nie robimy.
- Łuki i łańcuchy z umiarem: mapy „skillowe" (gęste) mało, mapy „vibe" więcej.
- Światła są osobną warstwą, którą można wyłączyć.
- Tło wybiera użytkownik opcją; domyślne: Default.
- Wszystko w jednym planie i jednej sesji treningowej na Colab.
- „Shut Up and Dance" (3 nps) w obecnej postaci jest wzorcem dobrego charakteru nut — warstwy nie
  mogą zmieniać samych nut.

## 3. Kryteria sukcesu

1. `bsmaker generate piosenka --levels hard,expert` tworzy jedną mapę z dwiema trudnościami,
   łukami, łańcuchami, ozdobami, światłami i wybranym tłem; ładuje się w Beat Saber 1.37.1 bez modów.
2. Zero naruszeń twardych reguł, starych i nowych (ściana nie nachodzi na nuty, bomba nie stoi na
   torze zamachu, łańcuch nie koliduje z nutami).
3. Każda warstwa da się wyłączyć; mapa bez warstw jest identyczna z dzisiejszą dla tego samego ziarna.
4. Ilość elementów na poziomie `--ozdoby 2` mieści się między 25. a 75. percentylem ludzkich map
   tej samej gęstości; poziomy 1 i 3 dają wyraźnie mniej i więcej.
5. Użytkownik ogląda lub gra co najmniej 3 mapy i ocenia każdą warstwę osobno. Warstwa oceniona
   źle zostaje domyślnie wyłączona, bez blokowania pozostałych.

## 4. Architektura

Warstwy powtarzają sprawdzony układ „kiedy" + „co" z nut, na tym samym kodzie:

| Warstwa | „Kiedy" (tagger klatek, jak model rytmu) | „Co" (model sekwencji, jak model wzorców) |
|---|---|---|
| Ozdoby nut | — (decyzja zapada przy każdej nucie) | tagger nut: dla każdej nuty łańcuch? łuk? |
| Dekoracja | 4 kanały: ściana lewa, ściana prawa, bomby, (rezerwa) | geometria i długość ściany; maska pól bomb |
| Światła | 9 kanałów: 5 grup świateł, obrót pierścieni, zoom, prędkość laserów L i P | wartość zdarzenia (akcja × kolor, prędkość) |

Wspólne wejścia: mel-spektrogram (jak dotąd), ślad nut (dwa dodatkowe kanały klatek: uderzenia
lewej i prawej ręki), gęstość nut, poziom warstwy. Przy generowaniu ślad nut pochodzi z już
wygenerowanych nut, więc warstwy nie wpływają na nuty.

Kolejność generowania: nuty → ozdoby nut → dekoracja → przeszkody z reguł → światła → kontrola → zapis.

### 4.1 Ozdoby nut

Tagger dwukierunkowy nad sekwencją nut jednej trudności (wszystkie nuty są już znane). Dla każdej
nuty przewiduje:

- **łańcuch:** brak albo liczba segmentów 3–7; koniec łańcucha wyliczany regułą (jedno pole
  w kierunku cięcia, w granicach siatki), ściśnięcie 1,0;
- **łuk:** brak albo długość (0,5 / 1 / 1,5 / 2 / 3 / 4 bity), pole końca (12) i kierunek końca (9).

Dane: tylko mapy v3 (łuki 4 383 map, łańcuchy 1 779). Losowanie z kalibracją progu do zadanej
liczby na minutę. Twarde reguły: łańcuch tylko wtedy, gdy następna nuta tej samej ręki jest co
najmniej 0,25 s później, a pola łańcucha są wolne od innych nut w tym czasie; łuk nie może kończyć
się po następnej nucie tej samej ręki.

### 4.2 Dekoracja

- **Ściany ozdobne:** kanały „lewa" i „prawa". Wartość = wariant położenia (kolumna skrajna albo
  tuż poza siatką), wariant wysokości (pełna albo wysoka), długość (1/8, 1/4, 1/2, 1, 2, 4 bity).
  Uczone na ścianach klas „błysk" i „dekoracja boczna" z pomiaru; ściany środkowe nie wchodzą do treningu.
- **Bomby:** jeden kanał; wartość = maska pól (słownik najczęstszych masek pokrywających 95% grup bomb).
- **Twarde reguły (maska przy losowaniu, jak przy nutach):** ściana nie obejmuje pola, w którym
  od 0,1 s przed jej początkiem do 0,1 s po końcu stoi nuta; bomba nie stoi w polu nuty ani w polu
  sąsiednim wzdłuż kierunku cięcia dla nut w odległości ±0,4 s.
- **Przeszkody rozgrywkowe (reguły, bez modelu):** tylko przy `--ozdoby` ≥ 2, najwyżej jedna
  sekwencja na 45 s. Krok w bok: ściana na dwóch kolumnach po jednej stronie przez 1–2 bity, gdy
  wszystkie nuty w tym czasie leżą po drugiej stronie. Kucnięcie: wysoka ściana przez środek przez
  1 bit, tylko w przerwie bez nut trwającej co najmniej 2 bity.

### 4.3 Światła

- Obsługiwane tła (klasyczne grupy świateł): Default, BigMirror, Nice, Triangle, Panic, Origins,
  Dragons, Timbaland, BTS, Kaleidoscope. Trening na mapach z tymi tłami, bez świateł grupowych v3,
  z co najmniej 100 zdarzeniami na minutę; jedna trudność na mapę.
- „Kiedy": 9 kanałów na klatkach mel. „Co": dla grup świateł — wyłącz albo akcja (włącz, błysk,
  zanikanie) × kolor (dwa); dla prędkości laserów — wartość 0–9; obrót i zoom bez wartości.
- Dane Chroma w zdarzeniach są ignorowane (brane są pola podstawowe). Kolory mapy zostają domyślne.
- **Plan B:** jeśli model świateł wypadnie źle w ocenie użytkownika, generator reguł: błysk na
  mocnym uderzeniu, zmiana koloru co 8 bitów, obrót pierścieni na akcentach. Reguły są też punktem
  odniesienia w ewaluacji.

## 5. Sterowanie

| Opcja | Znaczenie | Domyślnie |
|---|---|---|
| `--levels` | lista trudności (easy, normal, hard, expert, expertplus) albo jak dotąd `--nps` | jedna trudność z `--nps` |
| `--ozdoby 0–3` | ilość łuków, łańcuchów, ścian i bomb; 0 = brak | auto: 2 poniżej 4,5 nps, 1 dla 4,5–7 nps, 0 powyżej |
| `--swiatla 0–3` | intensywność pokazu; 0 = brak | 2 |
| `--tlo` | nazwa tła z listy obsługiwanych | Default |

Poziomy 1 / 2 / 3 odpowiadają 25. / 50. / 75. percentylowi ludzkich map o tej gęstości (wśród map,
które dany element mają). Przy kilku trudnościach światła i dekoracja są wspólne (liczone dla
najgęstszej), a ozdoby nut osobne dla każdej.

## 6. Dane

- Czytnik map dostaje pełne parsowanie ścian, bomb, łuków, łańcuchów i zdarzeń świateł (v2 i v3;
  v4 pomijane) oraz nazwy tła. Czasy przez mapę tempa, jak dla nut.
- Nowy cache warstw obok cache zdarzeń: dla każdej przyjętej trudności zdarzenia każdej warstwy
  w sekundach. Klasy ścian i bomb liczone regułami z pomiaru.
- Podział train/val/test bez zmian.
- Na Colab jedzie tylko nowy cache i zip z kodem (cechy dźwięku już tam są).

## 7. Trening i ewaluacja

- Pięć małych modeli w jednej kolejce na A100: tagger ozdób nut, „kiedy" i „co" dekoracji,
  „kiedy" i „co" świateł. Szacunek 5–7 godzin, do 70 jednostek; od użytkownika jedno wgranie
  pliku i jedno `drivemount`.
- `bsmaker eval` dostaje sekcje warstw: liczba elementów na minutę względem ludzi na każdym
  poziomie, odległość rozkładów wartości od ludzkich, udział zdarzeń świateł w 30 ms od ćwierćbitu
  i ich siła onsetu (ludzie: 76% i 2,5×), naruszenia reguł (0).
- Wybór i domyślne włączenie każdej warstwy dopiero po ocenie użytkownika.

## 8. Poza zakresem

Światła grupowe v3 i najnowsze tła; Chroma i Noodle; własne kolory mapy; mapy fitbeat; tryby 360/90;
format v4; ponowny trening modeli nut; interfejs graficzny.

## 9. Ryzyka

| Ryzyko | Zabezpieczenie |
|---|---|
| Światła chaotyczne (ludzkie pokazy są gęste i bardzo różne, brak sygnału jakości) | Filtr danych, sterowanie intensywnością, plan B z reguł, warstwa wyłączalna |
| Dekoracja przeszkadza w grze | Twarde reguły jako maska przy losowaniu + kontrola po fakcie |
| Mało danych na łańcuchy (1 779 map) | Mały tagger, kalibracja liczby, poziom 0 dla gęstych map |
| Trzy tory kodu naraz kolidują | Osobne kopie robocze repo na tor; wspólne zmiany (czytnik, cache, zapis) najpierw, w jednym torze |
| Reguły klas ścian/bomb z pomiaru są umowne | Progi w jednym miejscu, raport liczności klas przy budowie cache |
