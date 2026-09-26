---
title: Ustawienia
seo_title: Zmienne środowiskowe i katalogi LibreYOLO
description: >-
  Wszystkie zmienne środowiskowe odczytywane przez LibreYOLO, zapisywane
  katalogi, wymagane tokeny i przełączniki zmieniające używaną ścieżkę kodu.
lead: >-
  LibreYOLO nie ma pliku konfiguracyjnego. Zachowaniem, które nie jest
  argumentem funkcji, sterują zmienne środowiskowe i niewielka liczba umownych
  katalogów, wszystkie wymienione tutaj.
keywords:
  - LIBREYOLO_DATASETS_DIR
  - LIBREYOLO_KERNELS
  - LIBREYOLO_FASTER_COCO_EVAL
  - HF_TOKEN
  - katalog wag LibreYOLO
  - pamięć podręczna LibreYOLO
last_verified: 1.5.0
verification: >-
  Zmienne znaleziono przez wyszukanie os.environ i os.getenv w libreyolo/**/*.py
  w wersji v1.5.0; semantykę odczytano w każdym miejscu użycia. Konwencje
  katalogów odczytano z libreyolo/data/utils.py, libreyolo/utils/download.py,
  libreyolo/export/exporter.py, libreyolo/models/base/model.py i
  libreyolo/models/sam3dbody/mhr_body.py.
snippets:
  usage:
    - label: Ustawienie innego katalogu głównego zbiorów danych
      language: bash
      code: |
        export LIBREYOLO_DATASETS_DIR=/data/datasets
        python -c "from libreyolo.data import DATASETS_DIR; print(DATASETS_DIR)"
    - label: Odczyt rozwiązanej wartości w Pythonie
      language: python
      code: >
        from libreyolo.data import DATASETS_DIR


        # Domyślnie ~/datasets; LIBREYOLO_DATASETS_DIR nadpisuje wartość podczas
        importu.

        print(DATASETS_DIR)
source_hash: 462f1288582225ce
---

## Zmienne środowiskowe

| Zmienna | Wartość domyślna | Działanie |
|---|---|---|
| `LIBREYOLO_DATASETS_DIR` | `~/datasets` | Katalog główny zbiorów danych. Odczytywany raz podczas importu do `libreyolo.data.DATASETS_DIR` |
| `LIBREYOLO_FASTER_COCO_EVAL` | nieustawiona | Nadpisuje flagę walidacji `faster_coco_eval`. `1`, `true`, `yes` lub `on` wymusza szybszy backend, każda inna wartość go wyłącza, a brak wartości pozostawia decyzję fladze konfiguracji |
| `LIBREYOLO_KERNELS` | nieustawiona | Wybór kerneli. `off` albo `reference` wymusza implementacje referencyjne; każda inna wartość wybiera wyłącznie implementacje zarejestrowane pod tą nazwą |
| `LIBREYOLO_QUANT_KERNELS` | nieustawiona | Starszy alias `LIBREYOLO_KERNELS`, odczytywany tylko wtedy, gdy tamta zmienna nie jest ustawiona |
| `LIBREYOLO_HUB_KERNELS` | nieustawiona | `0`, `false`, `off` albo `no` wyłącza ładowanie kerneli z Hugging Face Hub. Każda inna wartość, w tym brak wartości, pozostawia je włączone |
| `LIBREYOLO_MHR_PATH` | `~/.cache/libreyolo/mhr/mhr_model.pt` | Położenie modelu ciała MHR używanego przez zadanie `mesh` |
| `LIBRELABEL_ENABLE_LOCATE` | nieustawiona | Musi mieć dokładnie wartość `1`, `true`, `yes` albo `on`, aby udostępnić asystenta LocateAnything w narzędziu do etykietowania. Każda inna wartość pozostawia go wyłączonego |
| `SAM_3D_BODY_PATH` | nieustawiona | Ścieżka do pakietu SAM 3D Body dla rodziny siatki, gdy nie przekazano jej do konstruktora |
| `HF_TOKEN` | nieustawiona | Token dostępu Hugging Face używany dla repozytoriów ograniczonych |

<code-tabs name="usage" />

`LIBREYOLO_DATASETS_DIR` jest odczytywane podczas importu, dlatego ustawienie go
po zaimportowaniu `libreyolo.data` nie wpływa na `DATASETS_DIR`.

Kernele Hub wymagają dwuetapowej zgody. Pobieranie podczas działania następuje
tylko po zainstalowaniu opcjonalnego pakietu `kernels`, dlatego instalacja
`libreyolo[hub-kernels]` włącza funkcję, a `LIBREYOLO_HUB_KERNELS=0` ją wyłącza.
Instalacja bez dodatku pozostaje niezmieniona niezależnie od tej wartości.

Wybór kerneli przerywa również importy: gdy `LIBREYOLO_KERNELS` wymusza `off`
albo `reference`, wbudowani przyspieszeni dostawcy w ogóle nie są importowani.
Rejestr sterowany przez te trzy zmienne opisano w sekcji
[kernele](/docs/reference/kernels).

## Zmienne ustawiane przez bibliotekę

Te zmienne są zapisywane, a nie odczytywane, dlatego ich ręczne ustawianie nie
jest obsługiwaną ścieżką.

| Zmienna | Ustawiana przez |
|---|---|
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | Pomocniczy mechanizm uruchamiania DDP, po jednej wartości na proces roboczy |
| `CUDA_VISIBLE_DEVICES` | Tymczasowo zawężana podczas konfiguracji rozproszonej, a następnie przywracana |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Ustawiana na `1` przez trenerów EC za pomocą `setdefault`, więc istniejąca wartość ma pierwszeństwo |
| `MOMENTUM_ENABLED` | Ustawiana przez moduł ładujący rodziny siatki za pomocą `setdefault` |

`LOCAL_RANK` pełni również funkcję sygnału trybu rozproszonego: obecność tej
zmiennej w środowisku informuje kod trenowania, że działa pod DDP.

## Zmienne loggerów

Opcjonalne loggery trenowania używają wartości środowiskowych jako nazw projektu.

| Zmienna | Wartość domyślna | Używana przez |
|---|---|---|
| `WANDB_PROJECT` | `libreyolo` | Logger Weights and Biases, gdy nie przekazano projektu |
| `COMET_PROJECT_NAME` | `libreyolo` | Logger Comet, gdy nie przekazano projektu |

Uwierzytelnianie w tych usługach odbywa się przez ich własne narzędzia, a nie LibreYOLO.

## Tokeny

`HF_TOKEN` jest tokenem dostępu Hugging Face. Gdy nie jest ustawiony, token jest
odczytywany z `~/.cache/huggingface/token`, gdzie zapisuje go logowanie przez CLI
Hugging Face. Obie ścieżki działają.

Token jest potrzebny tylko dla repozytoriów ograniczonych. Dostarczanym przykładem
jest SAM 3: jego wagi są pobierane z ograniczonego repozytorium na licencji
niestandardowej, dlatego warunki trzeba zaakceptować na stronie repozytorium,
a sesja musi być uwierzytelniona.

## Katalogi

| Ścieżka | Zawartość |
|---|---|
| `weights/` | Pobrane punkty kontrolne, pobrane migawki Hugging Face i wyeksportowane artefakty |
| `~/datasets` | Katalog główny zbiorów danych, chyba że `LIBREYOLO_DATASETS_DIR` wskazuje inaczej |
| `~/.cache/huggingface/token` | Token Hugging Face, gdy nie znajduje się w `HF_TOKEN` |
| `~/.cache/libreyolo/mhr/mhr_model.pt` | Model ciała MHR, chyba że `LIBREYOLO_MHR_PATH` wskazuje inaczej |
| `runs/track/` | Domyślny wynik `model.track(save=True)` |

`weights/` jest względne wobec katalogu roboczego. Sama nazwa pliku jest
rozwiązywana przez ten katalog, dlatego `LibreYOLO("LibreYOLO9t.pt")` szuka
`weights/LibreYOLO9t.pt` i pobiera go tam, gdy pliku brakuje. `model.export()`
zapisuje do tego samego katalogu, gdy nie podano `output_path`. Warstwy siostrzane
pobierają migawki wieloplikowe do `weights/<Prefix><size>/`.

## Zachowanie pobierania

Pobieranie wag jest ponawiane trzy razy z narastającym opóźnieniem, wznawiane z
pliku częściowego i chronione plikiem blokady, aby dwa procesy nie pobierały
jednocześnie tego samego punktu kontrolnego. Rodzina pobierająca z hosta innej
firmy może przypiąć sumę kontrolną i bezpiecznie przerwać pracę przy niezgodności.

Niektóre pobrania wyświetlają informację o licencji przed rozpoczęciem. Te
informacje są częścią ścieżki pobierania i nie można ich wyłączyć w konfiguracji.

## Backend walidacji

`model.val()` domyślnie przyjmuje `faster_coco_eval=True` i wraca do pycocotools,
gdy pakiet nie jest zainstalowany, jednokrotnie zgłaszając ostrzeżenie. Ustawienie
`LIBREYOLO_FASTER_COCO_EVAL` nadpisuje flagę pojedynczego wywołania, co jest
przeznaczone dla zestawu benchmarkowego, który nie może zmieniać konfiguracji
każdego przebiegu. Rzeczywiście użyty backend jest raportowany w
`model.last_eval_backend`.

## Skrypty pobierania zbiorów danych

Plik YAML zbioru danych może zawierać pole `download` z kodem Python. Nie jest ono
wykonywane bez przekazania `allow_download_scripts=True` do wywołania, które je
odczytuje. Jest to argument funkcji `val()` i `export()`, a nie zmienna środowiskowa.

