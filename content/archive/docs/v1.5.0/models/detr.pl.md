---
title: DETR
families:
  - detr
seo_title: 'DETR: przewiduj i eksportuj pod Apache-2.0'
description: >-
  Uruchom DETR, oryginalny transformer detekcji, w LibreYOLO. Zainstaluj,
  przewiduj, waliduj i eksportuj cztery rozmiary ResNet-based, wszystkie na
  licencji Apache-2.0.
lead: >-
  DETR to oryginalny transformator detekcji, przewidujący stały zestaw obiektów
  przy użyciu dekodera transformatora dopasowanego węgiersko, zamiast kotwic lub
  gęstej siatki. LibreYOLO oferuje cztery rozmiary do detekcji, tylko do
  inferencji.
keywords:
  - DETR
  - transformer wykrywania
  - detekcja obiektów
  - węgierskie dopasowanie
  - dekoder transformera
  - Meta AI
  - Facebook AI Badania
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")

        # val() zwraca zwykły słownik, a nie obiekt
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Użyj wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka kieruje na podstawie rozszerzenia pliku, więc eksportowany
        artefakt się ładuje

        # jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreDETRr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: c5549a596742d2a5
---
## Instalacja

DETR nie potrzebuje żadnych dodatkowych opcji. Wszystko, co importuje, znajduje się w instalacji bazowej.

```bash
pip install libreyolo
```

## Predykcja

Wagi są pobierane z Hugging Face przy pierwszym użyciu i są przechowywane lokalnie w pamięci podręcznej.

<code-tabs name="predict" />

Zwrócony obiekt `Results` jest tym, który zwraca każda rodzina, więc zamiana na inny detektor to zmiana jednej linii. `conf` i `max_det` filtrują wybór zapytania; `iou` jest akceptowany dla zgodności z API, ale nie ma efektu, ponieważ dekoder jest przewidującym zestaw, który nie ma kroku NMS. Zobacz [predykcja](/docs/predict) dla źródeł, streaming i obsługi wyników.

DETR jest tylko do inferencji w LibreYOLO. Wstępne trenowanie odbywa się przez 500 epok z dopasowaniem węgierskim; ta receptura nie jest tutaj zaimplementowana, więc `train()` generuje `NotImplementedError`.

## Warianty

Cztery checkpointy łączą dwie głębokości backbone, ResNet-50 lub ResNet-101, z opcjonalnym etapem rozszerzonym C5: warianty DC5 zachowują ostatni etap backbone w pełnej rozdzielczości zamiast dalszego próbkowania w dół, dzięki czemu dekoder odczytuje dokładniejszą mapę cech z tego samego rozmiaru wejściowego. Wszystkie cztery dzielą 100 wyuczonych zapytań dotyczących obiektów oraz sześciowarstwowy enkoder-dekoder transformera, a wszystkie działają przy tej samej rozdzielczości wejściowej.

## Walidacja

`val()` zwraca słownik kluczy `metrics/` obejmujących dokładność, recall, mAP 50 i mAP 50-95, mierzone względem dowolnego zbioru danych w formacie, na którym byłeś trenowany.

<code-tabs name="val" />

## Eksport

<export-matrix />

Eksportowany artefakt ładuje się z powrotem przez `LibreYOLO()` na podstawie jego rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint i zwraca ten sam `Results`. [Eksport](/docs/export) wymienia argumenty akceptowane przez każdy format.

<code-tabs name="export" />

## Checkpointy

Każdy opublikowany plik wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box> 
