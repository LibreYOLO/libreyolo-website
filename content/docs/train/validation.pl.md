---
title: Walidacja i metryki
seo_title: Walidacja i metryki w LibreYOLO
description: >-
  Uruchom val() dla dowolnego modelu, odczytaj klucze metryk zwracane przez
  poszczególne zadania, wybierz backend ewaluacji i włącz stratę walidacyjną
  obok metryki dokładności.
lead: >-
  Walidacja wykonuje model na podziale zbioru danych za pomocą val() i zwraca
  płaski słownik kluczy metryk oraz wartości zmiennoprzecinkowych. Klucze są
  dosłownymi ciągami znaków, a ich zestaw zależy od zadania, nie od rodziny.
keywords:
  - map50-95
  - ewaluacja coco
  - metryki walidacyjne modelu
  - faster-coco-eval
  - pycocotools
  - strata walidacyjna
  - miou segmentacja
  - jakość segmentacji panoptycznej
  - top1 accuracy
last_verified: 1.5.0
snippets:
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["speed/total_ms"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Na innym podziale
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml", split="train", batch=4)

        print(metrics)
  valloss:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, val_loss=True)
  json:
    - label: Zapis predykcji w formacie COCO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.val(data="coco8.yaml", save_json=True, save_dir="runs/val/exp")
source_hash: d907183492fa3f57
---

## Uruchamianie walidacji

`val()` przyjmuje zbiór danych i zwraca metryki.

<code-tabs name="val" />

Wartość zwracana to zwykły `dict[str, float]`. Każdy klucz jest dosłowny,
dlatego należy odczytywać go według nazwy, a nie pozycji.

Główne argumenty to `data`, `split`, `batch`, `imgsz`, `conf`, `iou`, `workers`,
`device`, `augment`, `save_json` i `verbose`. Domyślna wartość `conf` to `0.001`,
a `iou` to `0.6`. Obie są znacznie mniej restrykcyjne niż wartości domyślne dla
predykcji, ponieważ przegląd mAP wymaga ogona predykcji o niskiej pewności.
Domyślna wartość `imgsz` odpowiada własnemu rozmiarowi wejścia modelu, a nie
stałej liczbie. `split` przyjmuje wyłącznie `val`, `test` albo `train`.

Każde inne pole konfiguracji walidacji jest przekazywane jako argument nazwany,
w tym `save_dir`, `max_det`, `eval_max_det`, `half`, `amp_dtype`, `cache` i
`save_plots`.

## Klucze metryk według zadania

Detekcja zwraca zestaw wartości z rodziny COCO:

```text
metrics/mAP50-95   metrics/mAP50    metrics/mAP75
metrics/mAP_small  metrics/mAP_medium  metrics/mAP_large
metrics/AR1  metrics/AR10  metrics/AR100  metrics/AR_max_det
metrics/AR_small  metrics/AR_medium  metrics/AR_large
metrics/precision  metrics/recall
metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)  metrics/mAP50-95(B)
```

Dwa z nich mogą być mylące. `metrics/precision` i `metrics/recall` to aliasy
zachowane dla zgodności wstecznej. Zawierają wartości mAP 50-95 i AR@100, a nie
parę precision i recall. Należy używać nazwanych kluczy.

Segmentacja instancji zwraca powyższe wartości mAP i AR dla masek pod kluczami
bez sufiksu, wersje dla ramek pod sufiksem `(B)`, a wersje dla masek są
powtórzone pod sufiksem `(M)`. Dla tego zadania precision i recall występują
wyłącznie w postaci z sufiksami jako `metrics/precision(B)`/`metrics/recall(B)`
oraz `metrics/precision(M)`/`metrics/recall(M)`. Obie pary zawierają te same
wartości aliasów co w detekcji. Para `(B)` to mAP50-95 ramek i AR@100 ramek, a
para `(M)` to mAP50-95 masek i AR@100 masek.

| Zadanie | Klucze |
|---|---|
| detect | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75` oraz opisane wyżej podziały według rozmiaru i recall |
| segment | wersje kluczy detekcji dla masek (klucze bez sufiksu dotyczą masek); `precision`/`recall` występują tylko jako `(B)`/`(M)`, a oba zestawy są aliasami według tej samej zasady |
| pose | `metrics/keypoints_mAP50-95`, `metrics/keypoints_mAP50`, `metrics/keypoints_mAP75`, `metrics/keypoints_mAP_M`, `metrics/keypoints_mAP_L` oraz odpowiadające im klucze `keypoints_AR` |
| obb | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, `metrics/precision`, `metrics/recall` oraz kopie z sufiksem `(OBB)` |
| classify | `metrics/accuracy_top1`, `metrics/accuracy_top5` |
| semantic | `metrics/mIoU`, `metrics/pixel_accuracy` |
| panoptic | `metrics/PQ`, `metrics/SQ`, `metrics/RQ`, `metrics/PQ_things`, `metrics/PQ_stuff`, `metrics/categories` |
| depth | `metrics/abs_rel`, `metrics/rmse`, `metrics/delta1`, `metrics/delta2`, `metrics/delta3` |
| normal | `metrics/mean_angular_error`, `metrics/median_angular_error`, `metrics/within_11_25`, `metrics/within_22_5`, `metrics/within_30` |
| edge | `metrics/ODS`, `metrics/OIS`, `metrics/best_threshold` |
| restore | `metrics/PSNR`, `metrics/SSIM` |
| matte | `metrics/MAE`, `metrics/Smeasure` |
| ocr | `metrics/det_precision`, `metrics/det_recall`, `metrics/det_hmean`, `metrics/e2e_precision`, `metrics/e2e_recall`, `metrics/e2e_f1`, `metrics/rec_1-NED` |
| point | `metrics/precision`, `metrics/recall`, `metrics/f1`, `metrics/MLE`, `metrics/MAE`, `metrics/RMSE` oraz klucz przeglądu mAP |

W zadaniu OBB klucze `metrics/precision` i `metrics/recall` nie są aliasami.
Zawierają rzeczywiste precision i recall przy IoU 0.50, obliczone dla najmniej
restrykcyjnego punktu pracy, czyli każdej predykcji, która przeszła przez `conf`
o domyślnej wartości `0.001`. Kopie z sufiksem `(OBB)` powtarzają te same cztery
wartości pod nazwą właściwą dla zadania, zgodnie z tą samą konwencją co `(B)` i
`(M)` powyżej.

`accuracy_top5` to w rzeczywistości top-`min(5, num_classes)`, więc na zbiorze z
trzema klasami jest to top-3. Każda próbka spełnia ten warunek, dlatego wartość
wynosi 1.0.

Klucz przeglądu dla zadania point jest tworzony z progów odległości. Przy
wartościach domyślnych ma postać `metrics/mAP@[0.01:0.10]`, a klucz pojedynczego
progu to `metrics/mAP@0.01`. Przekazanie `dist_thresholds` zmienia oba ciągi.

Większość zadań zwraca również klucz `fitness`, pojedynczą wartość domyślnie
używaną do wyboru najlepszego checkpointu. Detekcja, segmentacja i OBB go nie
zawierają. Ich rodziny są wybierane według `metrics/mAP50-95`, który zwracają ich
słowniki. Estymacja pozy nie zwraca ani `fitness`, ani `metrics/mAP50-95`. Jej
trenery ustawiają zamiast tego `best_metric_key` na
`metrics/keypoints_mAP50-95`.

## Klucze szybkości

Każdy walidator dodaje pomiary czasu:

```text
speed/preprocess_ms   speed/inference_ms   speed/postprocess_ms
speed/total_ms        speed/total_s        speed/images_seen
```

Są to czasy w milisekundach na obraz, uśrednione dla całego przebiegu. Opisują
maszynę i użyte ustawienia, dlatego uzyskana z nich wartość ma sens tylko wtedy,
gdy podano wraz z nią sprzęt, rozmiar batcha i precyzję.

## Backend ewaluacji

Metryki detekcji i segmentacji są obliczane przez ewaluator COCO, a domyślne
`faster_coco_eval=True` wybiera backend C++, jeśli zainstalowano pakiet
`faster-coco-eval`. W przeciwnym razie przebieg wraca do pycocotools i wyświetla
jedno ostrzeżenie na proces:

```text
faster_coco_eval requested but not installed; falling back to pycocotools.
Install with: pip install faster-coco-eval
```

Rzeczywiście użyty backend jest zapisany w modelu jako `last_eval_backend`, a
CLI podaje go w wynikach zadań typu detekcyjnego. Ustaw
`LIBREYOLO_FASTER_COCO_EVAL`, aby nadpisać wartość konfiguracji za pomocą
zmiennej środowiskowej.

`iou_thresholds` jest uwzględniane wyłącznie na ścieżce OBB. Ścieżka COCO
wykonuje własny stały przegląd od 0.50 do 0.95 i ignoruje tę wartość.

## Strata walidacyjna

Domyślnie walidacja raportuje tylko dokładność. `val_loss=True` oblicza również
funkcję celu trenowania danej rodziny na batchach walidacyjnych.

<code-tabs name="valloss" />

Zwracany jest klucz `metrics/loss` oraz po jednym
`metrics/loss/<component>` na składnik. Składniki są ważone dokładnie tak samo
jak podczas trenowania, więc ich suma daje wartość całkowitą. W loggerze
występują jako `val/loss` i `val/loss/<component>`, a `libreyolo monitor`
nakłada `metrics/loss` na `train/loss`.

Składniki są właściwe dla danej rodziny:

| Zadanie | Rodziny | Składniki |
|---|---|---|
| detect | `yolo9`, `yolo9_p2`, `yolo9_e2e` | `box`, `cls`, `dfl` |
| detect | `yolonas` | `cls`, `iou`, `dfl` |
| detect | `rfdetr` | `ce`, `bbox`, `giou` |
| detect | `rtdetr`, `rtdetrv2` | `vfl`, `bbox`, `giou` |
| detect | `dfine` | `vfl`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `domedetr` | `vfl`, `bbox`, `giou`, `fgl`, `ddf`, `defe_density`, `defe_reg` |
| detect | `deim`, `deimv2`, `rtdetrv4`, `ec` | `mal`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `rtmdet` | `cls`, `bbox` |
| detect | `picodet` | `cls`, `bbox`, `dfl` |
| detect | `yolox` | `iou`, `obj`, `cls`, `l1` |
| detect | `yolo7` | `iou`, `obj`, `cls` |
| point | `fomo` | `ce` |
| classify | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` | `ce` |
| semantic | `segformer`, `lingbotvision`, `dinov2` | `sem` |
| restore | `nafnet` | `restore` |

Funkcja jest domyślnie wyłączona, ponieważ przypisywanie danych referencyjnych
zwiększa czas i zużycie pamięci podczas walidacji. Walidator ponownie wykorzystuje
wynik modelu już obliczony dla metryki dokładności, zamiast wykonywać drugie
przejście w przód. Działa w `no_grad` na modelu ewaluacyjnym lub EMA, a podczas
trenowania na wielu GPU jest obliczany lokalnie przez rangę 0 bez operacji
kolektywnych. Wybór najlepszego checkpointu nadal opiera się na metryce
dokładności.

Celowo nie robi trzech rzeczy. Nigdy nie uwzględnia składników kontrastowego
odszumiania, ponieważ wymagają one danych referencyjnych podczas przejścia w
przód, a przejście walidacyjne ich nie otrzymuje. Raportuje model w trybie
ewaluacji, więc tam, gdzie przejścia rodziny w trybie trenowania i ewaluacji
rzeczywiście się różnią, na przykład statystykami BatchNorm albo głębokością
stochastyczną, wartość odzwierciedla tryb ewaluacji. Jest to zamierzone
porównanie. Jeśli funkcja nie została zaimplementowana dla danego zadania w
rodzinie, podczas konfiguracji zgłaszany jest błąd zamiast cichego pominięcia:

```text
val_loss=True currently supports RF-DETR detection only; segment, pose, OBB,
classify, and semantic tasks are not supported
```

FOMO jest wyjątkiem, dla którego nic się nie zmienia. Jego walidator zawsze
obliczał tę funkcję straty, a `val_loss=True` wpływa wyłącznie na klucze, pod
którymi jest publikowana.

Walidacji z augmentacją nie można łączyć ze stratą walidacyjną. Żądanie obu
powoduje błąd.

## Pliki zapisywane przez walidację

`val()` zawsze zapisuje `config.yaml` w swoim katalogu wynikowym. Jeśli nie
podano `save_dir`, domyślnie jest to
`runs/val/<model>_<size>_<timestamp>`.

<code-tabs name="json" />

`save_json=True` zapisuje `predictions.json` dla detekcji oraz
`predictions_bbox.json` i `predictions_masks.json` dla segmentacji. OBB nie
obsługuje tej opcji i wyświetla odpowiedni komunikat.

`save_plots=True` zapisuje pliki w podkatalogu `plots/`. Dla detekcji powstają
`box_metrics.png`, wykresy AP i recall dla poszczególnych klas, krzywe
precision-recall i pewności, macierz pomyłek oraz obrazy próbek z adnotacjami,
jeśli zainstalowano OpenCV. Segmentacja dodaje odpowiedniki każdego wykresu dla
masek, a estymacja pozy otrzymuje własny zestaw metryk i krzywych. Pozostałe
walidatory nie implementują wykresów. Klasyfikacja, segmentacja semantyczna,
segmentacja panoptyczna, głębia, normalne, krawędzie, rekonstrukcja, matting,
OCR, OBB i point nie zapisują tam niczego. Błąd tworzenia wykresu powoduje
ostrzeżenie i nigdy nie przerywa przebiegu.

## Walidacja podczas trenowania

Trenowanie wykonuje walidację co `eval_interval` epok na podziale `val` zbioru
danych. Uzyskane metryki sterują wyborem `best.pt`, mechanizmem early stopping
`patience` i kluczami `val/` w każdym loggerze. Walidacja korzysta z wag EMA, gdy
EMA jest włączone.

Zobacz [Hiperparametry](/docs/train/hyperparameters), aby poznać `eval_interval`,
`patience` i `save_plots`, oraz [Loggery eksperymentów](/docs/train/loggers), aby
sprawdzić, dokąd trafiają wartości.

## Powiązane

- [Zbiory danych](/docs/train/datasets) opisujące klucze podziałów i formaty
  odczytywane przez walidatory.
