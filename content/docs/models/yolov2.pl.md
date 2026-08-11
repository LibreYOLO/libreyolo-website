---
title: YOLOv2
families:
  - yolo2
seo_title: 'YOLOv2 w LibreYOLO: predykcja, walidacja i eksport'
description: >-
  Uruchamiaj YOLOv2 (YOLO9000) w LibreYOLO: zamrożoną, muzealną rodzinę
  wyłącznie do inferencji. Przewiduj, waliduj i eksportuj na licencji domeny
  publicznej.
lead: >-
  YOLOv2, opublikowany także jako YOLO9000, to detektor Darknet-19, który
  wprowadził do rodziny YOLO ramki kotwic i warstwę passthrough. LibreYOLO
  udostępnia go jako zamrożony eksponat przeznaczony wyłącznie do inferencji.
keywords:
  - YOLOv2
  - YOLO9000
  - Darknet
  - Darknet-19
  - detekcja obiektów
  - ramki kotwic
  - modele muzealne
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO2b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO2b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO2b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO2b.pt format=onnx
        libreyolo export model=LibreYOLO2b.pt format=tensorrt half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreYOLO2b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: ba2884a2f6e1b0da
---

## Instalacja

YOLOv2 nie wymaga żadnego dodatku poza pakietem podstawowym.

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
detektora wymaga zmiany jednego wiersza. Argument `conf` filtruje próg pewności,
a `iou` próg NMS, stosowane do predykcji głowicy `region` opartych na kotwicach.
Więcej informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Walidacja

`val()` zwraca słownik kluczy `metrics/`, który obejmuje precision, recall,
mAP 50 i mAP 50-95, zmierzone względem dowolnego zbioru danych w formacie użytym
do walidacji.

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
