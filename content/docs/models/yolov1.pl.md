---
title: YOLOv1
families:
  - yolo1
seo_title: 'YOLOv1 w LibreYOLO: predykcja, walidacja i eksport'
description: >-
  Uruchamiaj oryginalny detektor YOLOv1 w LibreYOLO: zamrożoną, muzealną rodzinę
  wyłącznie do inferencji. Przewiduj, waliduj i eksportuj na licencji domeny
  publicznej.
lead: >-
  YOLOv1 to oryginalny detektor z 2016 roku, od którego nazwę wzięła rodzina
  YOLO: jedna sieć konwolucyjna z głowicą w pełni połączoną przewiduje wszystkie
  ramki i wyniki klas w jednym przebiegu, bez ramek kotwic. LibreYOLO udostępnia
  go jako zamrożony eksponat przeznaczony wyłącznie do inferencji.
keywords:
  - YOLOv1
  - YOLO v1
  - Darknet
  - detekcja obiektów
  - Pascal VOC
  - modele muzealne
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO1b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO1b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO1b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO1b.pt format=onnx
        libreyolo export model=LibreYOLO1b.pt format=tensorrt half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreYOLO1b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: a786372dba86f2f8
---

## Instalacja

YOLOv1 nie wymaga żadnego dodatku poza pakietem podstawowym.

```bash
pip install libreyolo
```

## Predykcja

Ta rodzina służy wyłącznie do inferencji: `train()` zgłasza
`NotImplementedError`, dlatego strona nie zawiera sekcji Trenowanie. Obsługiwane
są predykcja, walidacja i eksport. Przy pierwszym użyciu wagi są pobierane z
Hugging Face i zapisywane w lokalnej pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zmiana
detektora wymaga zmiany jednego wiersza. Dwie kwestie są charakterystyczne dla
tej rodziny. Opublikowany checkpoint wytrenowano na Pascal VOC (2007+2012), a
nie COCO, dlatego `box.cls` indeksuje 20 kategorii VOC (aeroplane, bicycle,
bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse,
motorbike, person, pottedplant, sheep, sofa, train, tvmonitor) zamiast 80
kategorii COCO. Ponadto w pełni połączona głowica detekcji przyjmuje jeden obraz
na raz, dlatego lista źródeł jest przetwarzana w pętli zamiast w prawdziwym
batchu. Więcej informacji o źródłach, streamingu i obsłudze wyników zawiera
strona [predykcji](/docs/predict).

## Walidacja

`val()` zwraca słownik kluczy `metrics/`, który obejmuje precision, recall,
mAP 50 i mAP 50-95, zmierzone względem zbioru danych w tej samej przestrzeni
etykiet w stylu VOC, na której wytrenowano checkpoint.

<code-tabs name="val" />

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ponownie ładowany przez `LibreYOLO()` na podstawie
rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint
i zwraca ten sam obiekt `Results`. Obsługiwane jest także uruchomienie grafu w
samym środowisku uruchomieniowym bez zainstalowanego LibreYOLO, ale wtedy należy
samodzielnie napisać przetwarzanie wstępne i końcowe.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>
