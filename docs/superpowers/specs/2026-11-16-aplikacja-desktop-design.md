# BeatSaberMaker — Etap 4: aplikacja desktopowa — spec

Data: 2026-11-16. Wcześniejsze: `2026-11-16-etap3-warstwy-design.md` (warstwy).
Silnik generowania istnieje (`bsmaker generate`); ten etap daje mu okno, instalator i aktualizacje.

## Po co

Alex chce generować mapy bez terminala i dać narzędzie znajomym. Wklejasz link z YouTube
albo wyszukujesz utwór, wybierasz jaka ma być mapa, klikasz — mapa ląduje w grze. Ma
wyglądać jak wydany produkt (porządny, „korporacyjny” interfejs: animacje, ikony, żadnych
tracebacków), bo idzie do ludzi i do portfolio. Nie musi być pełną profeską (bez podpisu
kodu, telemetrii, kont).

Sukces: znajomy pobiera instalator z GitHuba, instaluje, w kilka kliknięć ma mapę w
Beat Saberze, a nowe wersje przychodzą same.

## Cele

1. Aplikacja Windows (10/11, x64) z instalatorem i automatycznymi aktualizacjami z
   GitHub Releases (repo `al3ksh/BeatSaberMaker` zostanie publiczne — przełącza Alex).
2. Kreator w trzech krokach (decyzja Alexa: kierunek „B — kroki”):
   1. **Utwór** — pole wyszukiwania YouTube z listą wyników (miniatura, tytuł, kanał,
      czas) albo wklejony link (rozpoznawany automatycznie). *Moja decyzja:* dodatkowo
      „z pliku” (mp3/ogg/wav z dysku) — silnik i tak to umie, koszt mały.
   2. **Mapa** — dwa tryby:
      - **Pakiet** (domyślny): Hard + Expert + Expert+, światła 2, ozdoby auto, otoczenie
        Default, losowy seed. *Moja decyzja* (Alex nie pamiętał, jak „pakiet” miał wyglądać).
      - **Własne**: poziomy (checkboxy easy…expertplus) albo jedno nps (suwak 0,5–12),
        style (chipy z `STYLES`), ozdoby 0–3, światła 0–3, otoczenie (lista z
        `parse_environment`), seed; w „zaawansowanych” przełącznik świateł regułowych.
   3. **Gotowe** — postęp z etapami (pobieranie → analiza audio → nuty → warstwy → zapis),
      potem „Mapa jest w grze”, przyciski: otwórz folder, eksport zip, zrób kolejną.
3. **Biblioteka** (decyzja Alexa): lista map zrobionych w aplikacji — kafelki z okładką,
   tytułem, poziomami, datą. Akcje: **wygeneruj ponownie** (otwiera krok 2 z poprzednimi
   ustawieniami, można zmienić seed/ustawienia), **eksport zip** (do wysłania znajomemu
   albo ręcznego wgrania na Questa), **usuń**.
4. **Ustawienia**: folder `CustomLevels` (wykrywany automatycznie z bibliotek Steama,
   ręczna zmiana), urządzenie (auto / GPU / CPU), folder eksportu, wersja + „sprawdź
   aktualizacje”.
5. Płynność: okno nigdy nie zamarza; generowanie w tle z postępem i możliwością anulowania.

## Nie-cele

- Quest standalone bezpośrednio (wysyłanie na gogle) — tylko eksport zip.
- macOS / Linux.
- Podpisywanie kodu (SmartScreen pokaże ostrzeżenie — akceptowane).
- Konta, telemetria, upload na BeatSaver, edytor map, podgląd 3D mapy w aplikacji.
- Porządkowanie repo pod publikację (README, zrzuty, licencja, przegląd historii) —
  **osobny projekt po tej aplikacji**, przed pierwszym publicznym release'em. README ma
  być pisane po ludzku, nie „typowo AI”.

## Anty-cele (porażka, nawet jeśli działa)

- Wygląda tanio / jak formularz z lat 2000.
- Użytkownik widzi traceback Pythona albo angielski komunikat z wnętrza silnika.
- Okno zawiesza się na czas generowania.
- Aktualizacja gubi bibliotekę albo ustawienia.
- Aplikacja zapisuje w `CustomLevels` cokolwiek poza swoimi folderami `BSM - …` albo
  usuwa cudze mapy.

## Ograniczenia

- Silnik to istniejący pakiet `bsmaker` (Python). Nie przepisujemy go; dokładamy cienką
  warstwę dla aplikacji.
- `generate` w runtime potrzebuje: `bsmaker.toml` (`load_config`: `custom_levels`,
  `checkpoints_dir`, `base_dir`), `playability.toml`, checkpointów timing/pattern
  (+ opcjonalnie warstw), `layer_stats.json`. Aplikacja musi je dostarczyć z paczki
  (modele) i z ustawień użytkownika (`CustomLevels`).
- Checkpointy w paczce bez stanu optymalizatora (~70 MB zamiast ~360 MB).
- Warstwy (Plan 4) mogą jeszcze nie mieć modeli; `generate` już degraduje się łagodnie
  (ostrzeżenie + reguły/wyłączenie) — aplikacja pokazuje to jako informację, nie błąd.
- GitHub Releases: pojedynczy plik ≤ 2 GB. PyTorch z CUDA (~2,5 GB) trzeba dzielić.
- GPU jest współdzielone z grami użytkownika — aplikacja nie trzyma modeli na GPU, gdy
  nic nie generuje (*moja decyzja*: zwalnia po zadaniu).
- Pobieranie audio z YouTube łamie regulamin YT; narzędzie do użytku prywatnego,
  README to mówi.
- Wykonawcy: kod pisze Gemini Hydrogen (`agy`), Argon tylko jako doradca/recenzent w
  kluczowych miejscach (mały limit). Zasady zasobów jak w poprzednich planach
  (`OMP_NUM_THREADS=4`, jeden proces Pythona na agenta, ≤ ~4 agentów z kodem naraz).

## Podejścia (wybrane: A)

- **A. Electron + silnik Pythona jako proces w tle** — *wybrane.* Interfejs webowy
  (najłatwiej o „wydany” wygląd, Gemini najmocniejszy w tym), electron-builder (instalator
  NSIS) + electron-updater (GitHub Releases, aktualizacje różnicowe). Koszt: dwa języki,
  dwa buildy.
- **B. Całość w Pythonie (PySide6/Qt)** — jeden język, ale wygląd „systemowy” i
  aktualizator do napisania od zera. Odrzucone: przegrywa z głównym celem (wygląd).
- **C. Tauri** — lżejsze, wbudowany updater, ale wymaga Rusta + MSVC (tarcie dla
  agentów na Windowsie), a zysk na rozmiarze znika przy PyTorchu. Odrzucone.

## Jak (decyzje)

**Interfejs** (*moje decyzje*, Alex dopracuje wygląd po wszystkim): Electron + TypeScript
+ React + Vite, Tailwind, komponenty w stylu shadcn/ui, animacje Framer Motion, ikony
Lucide. Ciemny motyw, akcenty czerwony/niebieski (miecze). Interfejs po polsku, napisy w
jednym pliku (łatwo dodać EN).

**Silnik** (*moja decyzja*): długożyjący proces `bsmaker-engine` (nowa komenda, np.
`bsmaker engine`) rozmawiający z Electronem przez JSON-lines na stdin/stdout. Modele
ładowane raz → ponowne generowanie szybkie. Komendy: `search`, `resolve` (link → metadane),
`download`, `generate`, `export_zip`, `cancel`, `info` (GPU, wersje, wykryty CustomLevels).
Zdarzenia: `progress {stage, pct}`, `result`, `error {code, message_pl}`. Jedno zadanie
naraz, kolejne czekają w kolejce w UI. Silnik pakowany PyInstallerem (onedir).

**YouTube**: yt-dlp (dołączony) — wyszukiwanie `ytsearch10:`, pobranie najlepszego audio;
ffmpeg dołączony (wersja LGPL). Tytuł/artysta z tytułu filmu („Artysta - Tytuł”, edytowalne
przed generowaniem), okładka z miniatury.

**Rozmiar / GPU** — decyzja Alexa: robimy głównie pod kartę graficzną (NVIDIA); znajomi
mają pecety, przypadkowi użytkownicy nie są celem.
- Instalator zawiera tylko aplikację (mała). **Pakiet silnika** (Python + PyTorch CUDA +
  modele, ~2,5 GB) jest pobierany przy pierwszym uruchomieniu z tego samego GitHub
  Release, w częściach ≤ 2 GB, ze sprawdzeniem SHA256, z paskiem postępu i wznawianiem.
- Pakiet silnika ma własną wersję; aktualizacja aplikacji pobiera go ponownie tylko, gdy
  ta wersja się zmieni.
- Bez karty NVIDIA silnik działa na CPU (ten sam PyTorch) — wolniej, z informacją w UI;
  nie jest to scenariusz optymalizowany.
Eksport do ONNX/DirectML — poza zakresem.

**Biblioteka**: `%APPDATA%\BeatSaberMaker\library.json` + kopia audio i okładki każdej
mapy (potrzebne do ponownego generowania). Ponowne generowanie nadpisuje ten sam folder
`BSM - …` po potwierdzeniu. Usuwanie: wpis + folder mapy do Kosza (nie trwałe). Biblioteka
pokazuje tylko mapy zrobione w aplikacji.

**Logi**: plik w `%APPDATA%\BeatSaberMaker\logs`, przycisk „otwórz logi” w ustawieniach.

**Wydania**: GitHub Actions (runner Windows) na tagu `v*` buduje silnik + aplikację,
publikuje release; electron-updater sprawdza przy starcie i proponuje restart.

**Układ repo**: `app/` (Electron), `src/bsmaker/engine/` (protokół), `packaging/`
(PyInstaller, skrypt odchudzania checkpointów), `.github/workflows/release.yml`.

## Rozstrzygnięcia po przeglądzie (moje decyzje)

1. **Postęp i anulowanie**: `generate_map` dostaje opcjonalny `progress(stage, pct)`
   i sprawdzanie flagi anulowania między etapami/porcjami (zmiana w rdzeniu, wsteczna
   zgodność: domyślnie `None`). Anulowanie = przerwanie na najbliższym punkcie kontrolnym,
   sprzątnięcie częściowego folderu. Ostatnia deska ratunku: zabicie procesu silnika i
   restart (UI to ukrywa).
2. Logika ładowania modeli z `generate_cmd` wyciągnięta do wspólnej funkcji
   (np. `bsmaker.pipeline`), używanej przez CLI i silnik. Zachowanie CLI bez zmian
   (testy CLI muszą przejść).
3. Writer dostaje `artist`, `title` i opcjonalną okładkę (`coverImageFilename` w
   `info.dat`, plik jpg w folderze). Mapy z CLI bez tych opcji — bez zmian.
4. Konfiguracja: nowa ścieżka `Config` z parametrów (bez `bsmaker.toml`); silnik dostaje
   ścieżki zasobów paczki (modele, `playability.toml`, `layer_stats.json`) i
   `CustomLevels` z ustawień aplikacji. `require_ffmpeg` przyjmuje jawną ścieżkę do
   dołączonego ffmpeg.
5. **Biblioteka trzyma ścieżkę folderu każdej mapy.** Ponowne generowanie: nowa mapa
   powstaje obok, po sukcesie stary folder idzie do Kosza, wpis biblioteki wskazuje nowy.
   Seed: Pakiet losuje seed, zapisuje go we wpisie (widoczny w „Własne”).
6. Usuwanie dotyka tylko folderów zapisanych w bibliotece i tylko gdy nazwa zaczyna się
   od `BSM - ` i folder leży w `CustomLevels`; brakujący folder → wpis znika bez błędu.
7. Modele w pakiecie silnika; nowe modele = nowa wersja pakietu silnika. Jeśli w chwili builda
   modeli warstw brak — release z warstwami regułowymi/wyłączonymi jest akceptowalny.
8. ffmpeg: przypięta wersja LGPL (URL + SHA256 w `packaging/`), yt-dlp tak samo;
   pobierane w CI, nie commitowane.
9. `CustomLevels`: wykrywanie Steam (rejestr + `libraryfolders.vdf`, pierwsza biblioteka
   z Beat Saberem); brak → pierwsze uruchomienie prosi o wskazanie folderu gry i nie
   puszcza dalej, dopóki nie jest ustawiony.
10. Błędy: silnik zwraca kody (`ffmpeg_missing`, `download_failed`, `no_models`,
    `cancelled`, `internal`…); polskie teksty to tabela w UI. Nieznany wyjątek →
    `internal` + „zobacz logi”.
11. Jedna instancja aplikacji; `library.json` zapisywany atomowo (plik tymczasowy +
    rename).
12. Testy: Python — jednostkowe dla protokołu silnika i nowych opcji writera; UI —
    testy komponentów + e2e (Playwright dla Electrona) z fałszywym silnikiem. Build z
    torch w CI tylko na tagu.

## Zostawione wykonawcy

Konkretne komponenty, animacje i układ ekranów w ramach kroków powyżej; nazwy komunikatów
błędów; dokładny format wiadomości protokołu; wykrywanie Steama (rejestr +
`libraryfolders.vdf`); strategia testów UI.
