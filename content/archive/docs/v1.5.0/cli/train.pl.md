---
title: libreyolo train
seo_title: dokumentacja polecenia libreyolo train
description: >-
  Trenowanie modelu z wiersza poleceń: wszystkie 59 argumentów wraz z
  wartościami domyślnymi, sposób, w jaki zastępują je ustawienia rodziny modeli,
  oraz argumenty, które rodzina ignoruje.
lead: >-
  Trenuje jeden model na jednym zbiorze danych i zapisuje checkpointy, metryki
  oraz logi w katalogu uruchomienia. Każdy argument poniżej ma wartość domyślną
  z definicji polecenia, którą może zastąpić własna konfiguracja trenowania
  rodziny modeli.
keywords:
  - libreyolo train cli
  - trenowanie yolo z wiersza poleceń
  - polecenie libreyolo train
  - argumenty libreyolo train
  - trenowanie yolo na własnych danych
  - zamrażanie warstw yolo
last_verified: 1.5.0
meta:
  - label: Polecenie
    value: libreyolo train
    mono: true
  - label: Wymagany
    value: data
    mono: true
  - label: Wynik
    value: 'Checkpointy, metryki i logi w runs/train/exp'
snippets:
  examples:
    - label: Podstawowy
      language: bash
      code: >
        # coco8.yaml jest częścią pakietu i przy pierwszym użyciu pobiera swoje
        8 obrazów.

        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10 imgsz=640
        batch=8
    - label: Najpierw sprawdzenie ustalonej konfiguracji
      language: bash
      code: >
        # Wypisuje ustawienia, których użyłoby uruchomienie, łącznie z
        domyślnymi

        # ustawieniami rodziny, i kończy działanie bez trenowania i wczytywania
        danych.

        libreyolo train model=LibreDFINEn.pt data=coco8.yaml epochs=10
        dry_run=true
    - label: Nazwane uruchomienie z jawnym przepisem
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml \
          epochs=50 batch=8 optimizer=adamw lr0=0.001 weight_decay=0.0001 \
          patience=20 save_period=5 project=runs/train name=yolo9s-coco8 exist_ok=true
source_hash: 3aad4298310d3081
---

## Składnia

```bash
libreyolo train data=<dataset.yaml> [model=<name|path>] [key=value ...]
```

Argumenty to pary `key=value`, działa też forma POSIX, więc `epochs=50` i
`--epochs 50` to ten sam argument. Wartości logiczne przyjmują `true` i `false`:
`amp=false` odpowiada `--no-amp` tam, gdzie flaga ma formę przeczącą.

## Argumenty

### Model i dane

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `data` | | Ścieżka do pliku YAML zbioru danych (format YOLO, np. `coco8.yaml`). Wymagany |
| `model` | `yolox-s` | Nazwa modelu lub ścieżka do wag |
| `task` | | Jawne nadpisanie zadania: `detect`, `segment`, `semantic`, `pose`, `classify`, `gaze`, `obb`, `point`, `depth` |
| `pretrained` | `true` | Użycie wstępnie wytrenowanych wag. `false` buduje architekturę i trenuje od zera |
| `allow_download_scripts` | `false` | Zezwolenie na osadzony kod Pythona w blokach pobierania w pliku YAML zbioru danych |

### Pętla trenowania

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `epochs` | `300` | Epoki trenowania |
| `batch` | `16` | Rozmiar batcha na urządzenie |
| `imgsz` | `640` | Rozmiar obrazu treningowego: `640` (kwadrat) lub `480x640` (HxW) |
| `device` | `auto` | Urządzenie: `0`, `cpu`, `mps`, `auto` |
| `workers` | `4` | Procesy robocze dataloadera |
| `cache` | `false` | Zapisywanie obrazów w pamięci podręcznej, aby przyspieszyć wczytywanie danych: `ram`, `disk`, `true`, `false` |
| `seed` | `0` | Ziarno losowości |
| `resume` | | Wznowienie trenowania: `true` lub ścieżka do checkpointu |
| `amp` | `true` | Automatyczna mieszana precyzja |
| `amp_dtype` | `float16` | Typ danych AMP na CUDA: `float16` lub `bfloat16` |
| `cuda_graph` | `false` | Przechwycenie przejścia w przód i wstecz do grafów CUDA. Tylko pojedyncze GPU i tylko obsługiwane rodziny; pozostałe działają w trybie eager |
| `lora` | `false` | Dostrajanie LoRA, dla rodzin transformerowych wymienionych w sekcji Uwagi |
| `freeze` | | Zamrożenie warstw: liczba całkowita, lista indeksów lub nazwy modułów |

### Destylacja

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `distill_model` | | Nauczyciel: checkpoint detektora albo identyfikator nauczyciela fundamentalnego, taki jak `dinov2`, do destylacji cech z backbone |
| `dis` | | Waga funkcji straty destylacji. Gdy nieustawione, publikowana wartość domyślna dla danego typu straty |
| `distill_loss_type` | `mgd` | Strata na cechach dla nauczycieli będących detektorami: `mgd`, `cwd`. Nauczyciele fundamentalni zawsze używają `feat_mse` |

### Optymalizator

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `optimizer` | `sgd` | Optymalizator: `sgd`, `adam`, `adamw` |
| `lr0` | `0.01` | Początkowy współczynnik uczenia |
| `momentum` | `0.937` | Momentum SGD, a dla optymalizatorów Adam współczynnik pierwszego momentu |
| `weight_decay` | `0.0005` | Regularyzacja L2 |
| `nesterov` | `true` | Momentum Nesterova |

### Harmonogram

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `scheduler` | `yoloxwarmcos` | Typ harmonogramu LR |
| `warmup_epochs` | `5` | Czas trwania rozgrzewki |
| `warmup_lr_start` | `0.0` | Początkowy LR rozgrzewki |
| `min_lr_ratio` | `0.05` | Minimalny współczynnik LR |
| `lr_drop` | `100` | Epoka skokowego obniżenia LR w RF-DETR |

### Augmentacja

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `mosaic` | `1.0` | Prawdopodobieństwo mosaic |
| `mixup` | `1.0` | Prawdopodobieństwo mixup |
| `hsv_prob` | `1.0` | Prawdopodobieństwo jitteru HSV |
| `flip_prob` | `0.5` | Prawdopodobieństwo odbicia poziomego |
| `degrees` | `10.0` | Zakres obrotu, plus minus, w stopniach |
| `translate` | `0.1` | Współczynnik przesunięcia |
| `shear` | `2.0` | Kąt ścinania |
| `mosaic_scale` | `(0.1,2.0)` | Zakres skali mosaic |
| `mixup_scale` | `(0.5,1.5)` | Zakres skali mixup |
| `no_aug_epochs` | `15` | Wyłączenie augmentacji przez ostatnie N epok |

### EMA

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `ema` | `true` | Wykładnicza średnia krocząca |
| `ema_decay` | `0.9998` | Współczynnik zaniku EMA |

### Walidacja w trakcie trenowania

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `val` | `true` | Walidacja w trakcie trenowania |
| `eval_interval` | `10` | Walidacja co N epok |
| `max_det` | `300` | Maksymalna liczba predykcji na obraz po NMS walidacji |
| `eval_max_det` | | Limit ewaluatora COCO. Gdy nieustawione, konwencja AP@100 z pycocotools |
| `faster_coco_eval` | `true` | Użycie backendu C++ faster-coco-eval do metryk COCO, gdy jest zainstalowany; w przeciwnym razie powrót do pycocotools |
| `save_plots` | `false` | Zapisanie końcowych wykresów walidacji w trakcie trenowania |
| `patience` | `50` | Cierpliwość early stopping (wczesne zatrzymanie). `0` wyłącza tę funkcję |

### Wyjście

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `project` | `runs/train` | Katalog główny wyników |
| `name` | `exp` | Nazwa eksperymentu |
| `exist_ok` | `false` | Ponowne użycie istniejącego katalogu wyjściowego |
| `save_period` | `10` | Zapis checkpointu co N epok |
| `log_interval` | `10` | Logowanie straty co N batchy |

### Flagi dla agentów

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `json` | `false` | Wyjście JSON na stdout |
| `quiet` | `false` | Wyciszenie stderr |
| `dry_run` | `false` | Ustalenie i wypisanie konfiguracji bez jej wykonania |
| `help_json` | `false` | Zrzut schematu polecenia jako JSON i zakończenie |

## Przykłady

<code-tabs name="examples" />

## Uwagi

### Powyższe wartości domyślne nie zawsze są tymi używanymi

Każda rodzina modeli ma własną konfigurację trenowania i tam, gdzie różni się
ona od bazowej, jej wartość zastępuje domyślną wartość polecenia dla każdego
argumentu, który nie został ustawiony jawnie. Samodzielne ustawienie argumentu
zawsze ma pierwszeństwo. `libreyolo cfg` wypisuje bazowe wartości domyślne oraz
nadpisania dla poszczególnych rodzin, i to jest sposób na sprawdzenie, czego
dana rodzina faktycznie użyje.

`imgsz` to argument, dla którego ma to największe znaczenie. Domyślną wartością
polecenia jest `640`, co nie jest natywnym wejściem każdego checkpointu:
publikowane rozmiary detekcji RF-DETR to 384, 512, 576 i 704, a checkpointy
YOLOX `n` i `t` mają 416. W przypadku RF-DETR i DEIMv2 `imgsz` jest przekazywany
tylko wtedy, gdy został ustawiony jawnie, więc poza tym obowiązuje ich własny
rozmiar. Pozostałe rodziny dostają podaną wartość i trenują z nią. FOMO jest tu
najbardziej rygorystyczny: każdy rozmiar przyjmuje wyłącznie swoje natywne
wejście (96, 192 i 224), więc uruchomienie FOMO wymaga dopasowania `imgsz`,
inaczej kończy się błędem. RF-DETR wymaga dodatkowo, aby wartość dzieliła się
przez rozmiar patcha pomnożony przez liczbę okien, i podaje dwa najbliższe
dopuszczalne rozmiary, gdy tak nie jest.

### Argumenty ignorowane przez rodzinę

Nie każda rodzina czyta każdy argument, a najbardziej widać to przy argumentach
augmentacji. RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETRv4 i DINOv2 trenują przez
pipeline'y typu pass-through, bez mosaic, bez mixup i bez przekształcenia
afinicznego, więc `mosaic`, `mixup`, `hsv_prob`, `degrees`, `translate`,
`shear`, `mosaic_scale` i `mixup_scale` nic tam nie zmieniają. EC korzysta
z tego samego pipeline'u, ale czyta `hsv_prob`, `degrees` i `translate`, gdy
jego zadaniem jest estymacja pozy. Rodziny klasyfikacyjne, SegFormer i NAFNet
ignorują cały ten zestaw, a wraz z nim `flip_prob`, ponieważ ich odbicie działa
ze stałym prawdopodobieństwem, a nie konfigurowalnym. YOLO-NAS ignoruje sam
`mosaic`, ponieważ zamiast tego stosuje stale włączone przekształcenie
afiniczne na każdej próbce. RF-DETR ignoruje dodatkowo trzy kolejne argumenty
spoza tej listy: `optimizer`, `momentum` i `nesterov`.

Ustawienie jednego z nich nie jest błędem. Uruchomienie zapisuje na stderr
wiersz z nazwą rodziny i argumentami, które zostaną zignorowane, a następnie
trenuje, i ten wiersz jest miarodajną listą dla zainstalowanej wersji. Jest to
również jedyny sygnał, więc skryptowe uruchomienie z `quiet=true` wycisza to
ostrzeżenie razem ze wszystkim innym na stderr.

`val=false` to powiązany przypadek. Dla większości rodzin ustawia
`eval_interval` na `0`; RF-DETR nie potrafi w ten sposób wyłączyć walidacji
i zapisuje w logu, że zignorował to żądanie.

### Inne zachowania, które warto znać

`lora=true` jest przyjmowane przez RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1,
v2 i v4, EC oraz ConvNeXt. Każda inna rodzina kończy działanie z
`config_unsupported`, zamiast trenować bez tego.

`pretrained=false` w połączeniu z `resume` jest odrzucane w rodzinach, które
obsługują trenowanie od zera, ponieważ oba te ustawienia żądają przeciwnych
rzeczy.

`mosaic` i `mixup` to zapis w wierszu poleceń pól konfiguracyjnych
`mosaic_prob` i `mixup_prob`. W rodzinach, w których mixup działa tylko na
próbkach mosaic, `mixup` powyżej zera przy `mosaic` równym zero nigdy się nie
uruchomi, o czym uruchomienie informuje.

`dry_run=true` rozwiązuje odwołanie do modelu, stosuje domyślne ustawienia
rodziny i wypisuje konfigurację, z którą trenowałoby uruchomienie. Nie wczytuje
zbioru danych, więc jest to tani sposób na potwierdzenie, że argument przyjął
oczekiwaną wartość.

stdout przenosi końcowy obiekt wyniku; postęp i ostrzeżenia trafiają na stderr.
Kod wyjścia to `0` przy powodzeniu, `2` przy błędzie użycia lub konfiguracji,
`3` gdy nie można znaleźć lub odczytać zbioru danych, `4` gdy nie można wczytać
modelu, oraz `1` przy innych awariach w trakcie działania.

Powiązane: [`libreyolo doctor`](/docs/cli/doctor) do sprawdzenia zbioru danych
przed zdecydowaniem się na uruchomienie, [`libreyolo monitor`](/docs/cli/monitor)
do obserwowania uruchomienia w przeglądarce, [`libreyolo val`](/docs/cli/val) do
zmierzenia wyniku.
