---
title: YOLOX
families:
  - yolox
seo_title: 'YOLOX: predykcja, trenowanie i eksport na licencji Apache-2.0'
description: >-
  Używaj YOLOX w LibreYOLO do detekcji obiektów: instaluj, przewiduj, trenuj,
  waliduj i eksportuj na licencji Apache-2.0.
lead: >-
  YOLOX to jednostopniowy detektor bez kotwic (anchor-free) z rozdzieloną
  głowicą klasyfikacji i regresji, trenowany z przypisywaniem etykiet SimOTA.
  LibreYOLO obsługuje go do detekcji.
keywords:
  - YOLOX
  - detekcja obiektów
  - detekcja bez kotwic
  - rozdzielona głowica
  - SimOTA
  - detekcja obiektów w czasie rzeczywistym
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLOXs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLOXs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLOXs.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLOXs.pt data=my-dataset.yaml
    - label: Porównanie z COCO
      language: bash
      code: >
        # Dołączony plik yaml COCO zawiera osadzony skrypt pobierania, dlatego
        wymaga

        # jawnego zezwolenia, chyba że zbiór danych jest już dostępny lokalnie.

        libreyolo val model=LibreYOLOXn.pt data=coco.yaml imgsz=416 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLOXs.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLOXs.pt format=tensorrt imgsz=640
        half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreYOLOXs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: f5ab735a29f85a95
---

## Instalacja

YOLOX nie wymaga żadnego dodatku poza pakietem podstawowym.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zmiana
detektora wymaga zmiany jednego wiersza. Argument `conf` ustawia próg pewności,
a `iou` próg NMS stosowany do trzech rozdzielonych skal predykcji. Więcej
informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

Sześć rozmiarów współdzieli ten sam backbone CSP i neck PAFPN. Dwa najmniejsze,
`n` i `t`, działają przy mniejszej stałej rozdzielczości wejściowej niż pozostałe
cztery. Poniższa tabela benchmarku podaje dokładną wartość dla każdego rozmiaru.

<benchmark-table task="detect" />

<va-embed />

## Trenowanie

<code-tabs name="train" />

Bez zmian konfiguracji trener wykonuje 300 epok z `lr0=0.01`, momentum SGD 0.9,
5 epokami rozgrzewki oraz augmentacjami mosaic i mixup wyłączonymi na ostatnie
15 epok. `train()` przyjmuje także argument `pretrained`, ale jego wartość nie
jest odczytywana wewnątrz metody. Trenowanie zawsze jest kontynuowane od wag, z
którymi utworzono model, dlatego `pretrained=False` nie inicjuje sieci ponownie.

`imgsz` domyślnie przyjmuje stałą wartość z podstawowej konfiguracji trenowania,
a nie natywną rozdzielczość wczytanego checkpointu. Ma to szczególne znaczenie
dla checkpointów `n` i `t`. Kontynuowanie trenowania któregokolwiek bez jawnego
ustawienia `imgsz` przełącza go na większą wartość domyślną zamiast mniejszego
rozmiaru, w którym został opublikowany.

Informacje o zbiorach danych, augmentacji, wielu GPU i loggerach zawiera strona
[trenowania](/docs/train).

## Walidacja

`val()` zwraca słownik kluczy `metrics/`, który obejmuje precision, recall,
mAP 50 i mAP 50-95, zmierzone względem dowolnego zbioru danych w formacie użytym
do trenowania.

<code-tabs name="val" />

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ponownie ładowany przez `LibreYOLO()` na podstawie
rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint
i zwraca ten sam obiekt `Results`. Obsługiwane jest także uruchomienie grafu w
samym środowisku uruchomieniowym bez zainstalowanego LibreYOLO, ale wtedy należy
samodzielnie napisać przetwarzanie wstępne i końcowe. Eksport CoreML może
wbudować NMS w graf za pomocą `nms=True`. YOLOX i YOLOv9 są jedynymi dwiema
rodzinami, które obecnie przyjmują tę flagę.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
