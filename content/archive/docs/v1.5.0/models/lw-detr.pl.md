---
title: LW-DETR
families:
  - lwdetr
seo_title: 'LW-DETR: predykcja i eksport na licencji Apache-2.0'
description: >-
  Uruchamiaj LW-DETR w LibreYOLO do detekcji obiektów w czasie rzeczywistym.
  Instaluj, przewiduj, waliduj i eksportuj pięć rozmiarów opartych na ViT,
  wszystkie na licencji Apache-2.0.
lead: >-
  Transformer do detekcji oparty na zwykłym ViT, przedstawiony przez Baidu jako
  alternatywa czasu rzeczywistego dla detektorów YOLO. LibreYOLO udostępnia pięć
  rozmiarów do detekcji, wyłącznie do inferencji.
keywords:
  - LW-DETR
  - transformer do detekcji
  - detekcja obiektów w czasie rzeczywistym
  - zwykły ViT
  - DETR
  - Baidu
  - Atten4Vis
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLWDETRt.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLWDETRt.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")

        # val() zwraca zwykły słownik, a nie obiekt
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLWDETRt.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreLWDETRt.pt format=onnx imgsz=640

        libreyolo export model=LibreLWDETRt.pt format=tensorrt imgsz=640
        half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreLWDETRt.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: badd1d8255df5bbd
---

## Instalacja

LW-DETR nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zmiana
detektora wymaga zmiany jednego wiersza. Argumenty `conf` i `max_det` filtrują
wybór zapytań. Argument `iou` jest przyjmowany dla zgodności API, ale nie ma
wpływu na wynik, ponieważ dekoder jest predyktorem zbioru bez etapu NMS. Więcej
informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

W LibreYOLO model LW-DETR służy wyłącznie do inferencji. Implementacja źródłowa
trenuje go z nadzorem jeden do wielu Group-DETR w wielu grupach zapytań i
funkcją straty klasyfikacji uwzględniającą IoU. Ta procedura nie jest tutaj
podłączona, dlatego `train()` zgłasza `NotImplementedError`.

## Warianty

Dostępnych jest pięć rozmiarów, wszystkie ze wspólnym enkoderem zwykłego ViT,
projektorem wieloskalowym i dekoderem deformable DETR oraz przy tej samej
rozdzielczości wejściowej. Dwa najmniejsze mają tę samą szerokość enkodera, a
różnią się głębokością bloków. Kolejne dwa mają wspólny szerszy enkoder i różnią
się liczbą poziomów projektora zasilających dekoder. Największy używa
najszerszego enkodera.

## Walidacja

`val()` zwraca słownik kluczy `metrics/`, który obejmuje precision, recall,
mAP 50 i mAP 50-95, zmierzone względem dowolnego zbioru danych w formacie użytym
do trenowania.

<code-tabs name="val" />

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ponownie ładowany przez `LibreYOLO()` na podstawie
rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint
i zwraca ten sam obiekt `Results`. Strona [eksportu](/docs/export) wymienia
argumenty obsługiwane przez każdy format.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
