---
title: Faster R-CNN
families:
  - faster_rcnn
seo_title: 'Faster R-CNN w LibreYOLO: predykcja, walidacja i eksport'
description: >-
  Uruchamiaj Faster R-CNN w LibreYOLO do detekcji obiektów z czterema backbone.
  Instaluj, przewiduj, waliduj i eksportuj port torchvision na licencji
  BSD-3-Clause.
lead: >-
  Faster R-CNN wykrywa obiekty za pomocą sieci propozycji regionów zasilającej
  dwustopniowy klasyfikator. To architektura, która włączyła propozycje regionów
  do tej samej trenowanej sieci zamiast osobnego etapu. LibreYOLO przenosi
  implementację torchvision do detekcji.
keywords:
  - Faster R-CNN
  - detekcja obiektów
  - sieć propozycji regionów
  - detektor dwustopniowy
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFasterRCNNl.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFasterRCNNl.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFasterRCNNl.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFasterRCNNl.pt format=onnx imgsz=800
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreFasterRCNNl.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 3fd82eb835399560
---

## Instalacja

Faster R-CNN nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane
elementy znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zmiana
detektora wymaga zmiany jednego wiersza. Argumenty `conf` i `iou` ustawiają progi
pewności i NMS. Faster R-CNN zachowuje źródłowy etap NMS, w przeciwieństwie do
detektora opartego na zapytaniach. Więcej informacji o źródłach, streamingu i
obsłudze wyników zawiera strona [predykcji](/docs/predict).

## Warianty

Dostępne są cztery rozmiary, z których każdy stanowi inną konfigurację
torchvision, a nie przeskalowaną wersję tej samej konfiguracji. `n` to
MobileNetV3-Large z wejściem 320 px, `s` używa tego samego backbone przy 800 px,
`m` to ResNet-50 z piramidą cech, a `l` jest wersją v2 z głębszą głowicą
propozycji regionów i czterokonwolucyjną głowicą ramek zamiast głowicy wariantu
`m`. Warianty `n` i `s` poświęcają dokładność na rzecz lżejszego backbone.

## Walidacja

`val()` zwraca słownik kluczy `metrics/`, który obejmuje precision, recall,
mAP 50 i mAP 50-95, zmierzone względem dowolnego zbioru danych w formacie użytym
do trenowania.

<code-tabs name="val" />

## Eksport

<export-matrix />

Faster R-CNN eksportuje wyłącznie do ONNX z rozmiarem batcha 1. Wyeksportowany
graf zachowuje wewnątrz źródłowy etap zmiany rozmiaru, dlatego LibreYOLO wymusza
`dynamic=True` niezależnie od przekazanej wartości, aby graf działał dla źródeł
niebędących kwadratami. Wyeksportowany plik `.onnx` jest ponownie ładowany przez
`LibreYOLO()` na podstawie rozszerzenia i zwraca ten sam obiekt `Results`.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
