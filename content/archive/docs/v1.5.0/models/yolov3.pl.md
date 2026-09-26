---
title: YOLOv3
families:
  - yolo3
seo_title: 'YOLOv3 w LibreYOLO: predykcja, walidacja i eksport'
description: >-
  Uruchamiaj YOLOv3 w LibreYOLO: zamrożoną, muzealną rodzinę wyłącznie do
  inferencji w rozmiarach tiny, base i SPP. Przewiduj, waliduj i eksportuj na
  licencji domeny publicznej.
lead: >-
  YOLOv3 to detektor Darknet-53, który dodał do rodziny YOLO predykcję
  wieloskalową i niezależne klasyfikatory logistyczne. LibreYOLO udostępnia go
  jako zamrożony eksponat przeznaczony wyłącznie do inferencji, w rozmiarach
  tiny, base i SPP.
keywords:
  - YOLOv3
  - Darknet
  - Darknet-53
  - detekcja obiektów
  - detekcja wieloskalowa
  - modele muzealne
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO3b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO3b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Rozmiar SPP
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Wariant SPP dodaje blok spatial pyramid pooling przed głowicami
        # detekcji i działa przy własnym natywnym rozmiarze wejścia.
        model = LibreYOLO("LibreYOLO3spp.pt")
        result = model(SAMPLE_IMAGE)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO3b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO3b.pt format=onnx
        libreyolo export model=LibreYOLO3b.pt format=tensorrt half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreYOLO3b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: a4c652bb2707fc8f
---

## Instalacja

YOLOv3 nie wymaga żadnego dodatku poza pakietem podstawowym.

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
a `iou` próg NMS, stosowane osobno dla każdej skali przed połączeniem ramek ze
wszystkich trzech głowic. Więcej informacji o źródłach, streamingu i obsłudze
wyników zawiera strona [predykcji](/docs/predict).

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
