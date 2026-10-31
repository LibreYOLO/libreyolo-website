---
title: EoMT
families:
  - eomt
seo_title: 'EoMT: predykcja segmentacji semantycznej, instancji i panoptycznej'
description: >-
  Używaj EoMT w LibreYOLO do segmentacji semantycznej, instancji i panoptycznej
  na prostym transformerze wizyjnym DINOv2, bez dekodera. Licencja MIT.
lead: >-
  Sieć segmentacji oparta na prostym transformerze wizyjnym bez dedykowanego
  dekodera pikseli: dodatkowe wyuczone zapytania dodane do samego enkodera
  przewidują maski. LibreYOLO obsługuje ją do segmentacji semantycznej,
  instancji i panoptycznej.
keywords:
  - EoMT
  - transformer masek tylko z enkoderem
  - DINOv2
  - segmentacja panoptyczna
  - segmentacja instancji
  - segmentacja semantyczna
last_verified: 1.5.0
snippets:
  predict:
    - label: Segmentacja semantyczna
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreEoMTl-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # (H, W), identyfikatory klas

        print(mask.classes)      # posortowane identyfikatory klas obecnych na
        obrazie
    - label: Segmentacja instancji
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sufiks -seg w nazwie pliku wybiera zadanie segmentacji instancji,
        # dlatego nie trzeba podawać argumentu zadania.
        model = LibreYOLO("LibreEoMTl-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)
    - label: Segmentacja panoptyczna
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W), identyfikatory segmentów
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEoMTl-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Segmentacja semantyczna
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Segmentacja instancji
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # maski
        print(metrics["metrics/mAP50-95(B)"])   # ramki
    - label: Segmentacja panoptyczna
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEoMTl-sem.pt format=onnx
        libreyolo export model=LibreEoMTl-sem.pt format=tensorrt half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreEoMTl-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 64b2da642999f150
---

## Instalacja

EoMT nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej. Sufiks zadania w nazwie pliku (`-sem`, `-seg`, `-panoptic`)
wybiera zadanie, a `LibreYOLO()` wnioskuje je z tej nazwy, więc argument `task=`
nie jest potrzebny.

<code-tabs name="predict" />

Segmentacja semantyczna wypełnia `result.semantic_mask`, czyli tablicę
identyfikatorów klas `(H, W)` w polu `.data`. Segmentacja instancji wypełnia
`result.boxes` i `result.masks` o takim samym kształcie, jaki zwracają inne
rodziny segmentacji. Segmentacja panoptyczna wypełnia `result.panoptic`: mapę
identyfikatorów segmentów `(H, W)` w polu `.data` oraz `.segments_info`, listę
słowników `{"id", "category_id"}`, po jednym na segment. Argument `conf`
filtruje wybór zapytań. `iou` nie ma wpływu na zadanie semantyczne, ponieważ
wybiera ono argmax dla każdego piksela bez etapu NMS. Więcej informacji o
źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępne są trzy rozmiary enkodera, s/b/l, wszystkie z backbone DINOv2.
Checkpoint semantyczny wytrenowano na ADE20K przy 512 px, a checkpointy
instancji i panoptyczny na COCO przy 640 px. Drugi checkpoint instancji
wytrenowano przy 1280 px. Projekt źródłowy publikuje wagi segmentacji instancji
z DINOv2 tylko w rozmiarze l. Rozmiary s i b są dostępne wyłącznie dla
segmentacji semantycznej i panoptycznej. Warianty EoMT z backbone DINOv3 istnieją
w projekcie źródłowym, ale nie są tu udostępniane, ponieważ zależą od
kontrolowanych, niekomercyjnych wag DINOv3.

LibreYOLO nie trenuje EoMT: `train()` zgłasza `NotImplementedError` dla tej
rodziny, którą powyższy [poziom obsługi](/docs/models) oznacza jako przeznaczoną
wyłącznie do inferencji.

## Walidacja

`val()` wybiera ścieżkę według zadania. Segmentacja semantyczna zwraca
`metrics/mIoU` i `metrics/pixel_accuracy`. Segmentacja instancji zwraca te same
klucze mAP masek i ramek co inne rodziny segmentacji. Segmentacja panoptyczna
zwraca Panoptic Quality jako `metrics/PQ`, podzielone na `metrics/SQ` (jakość
segmentacji) i `metrics/RQ` (jakość rozpoznawania), a także `metrics/PQ_things`
i `metrics/PQ_stuff`.

<code-tabs name="val" />

## Eksport

<export-matrix />

Obecnie eksportowane jest tylko zadanie semantyczne. Wywołanie `export()` dla
segmentacji instancji lub panoptycznej zgłasza `NotImplementedError`, ponieważ
ich wyjście zapytań i masek nie ma jeszcze kontraktu eksportu środowiska
uruchomieniowego. Wyeksportowany artefakt semantyczny jest ponownie ładowany
przez `LibreYOLO()` na podstawie rozszerzenia pliku, więc plik `.onnx` lub
`.engine` zachowuje się jak checkpoint i zwraca ten sam obiekt `Results`.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
