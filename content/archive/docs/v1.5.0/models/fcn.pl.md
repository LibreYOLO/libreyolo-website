---
title: FCN
families:
  - fcn
seo_title: 'FCN: predykcja i eksport FCN z ResNet na licencji BSD-3-Clause'
description: >-
  Używaj FCN w LibreYOLO do segmentacji semantycznej. Instaluj, przewiduj,
  waliduj i eksportuj checkpointy FCN z dylatowanym ResNet z torchvision.
lead: >-
  Gęsty klasyfikator dla każdego piksela, który zastępuje warstwy w pełni
  połączone detektora splotami, dzięki czemu zwraca mapę klas w pełnej
  rozdzielczości zamiast ramek. LibreYOLO udostępnia go wyłącznie do segmentacji
  semantycznej.
keywords:
  - FCN
  - w pełni konwolucyjna sieć
  - segmentacja semantyczna
  - gęsta predykcja
  - ResNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreFCNr50.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # (H, W), identyfikatory klas

        print(mask.classes)      # posortowane identyfikatory klas obecnych na
        obrazie
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCNr50.pt format=onnx
        libreyolo export model=LibreFCNr50.pt format=tensorrt half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreFCNr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 7776b0fc85a208fb
---

## Instalacja

FCN nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Segmentacja semantyczna zwraca jeden identyfikator klasy na piksel, a nie ramki.
Dlatego `result.semantic_mask` zawiera tablicę `(H, W)` w polu `.data` oraz listę
identyfikatorów klas obecnych na obrazie w `.classes`. Argumenty `conf`, `iou` i
`max_det` są przyjmowane dla zgodności API, ale nie mają wpływu na wynik. Model
przypisuje klasę każdemu pikselowi przez argmax, bez progu pewności ani etapu
NMS. Więcej informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępne są dwie głębokości ResNet, obie ze stałym wejściem 520 px. Graf
inferencji biblioteki jest FCN z dylatowanym ResNet z torchvision, a nie opartą
na VGG siecią FCN-8s z połączeniami pomijającymi z oryginalnej publikacji.

LibreYOLO nie trenuje FCN: `train()` zgłasza `NotImplementedError` dla tej
rodziny, którą powyższy [poziom obsługi](/docs/models) oznacza jako przeznaczoną
wyłącznie do inferencji. Dwa opublikowane checkpointy to własne wagi torchvision
wytrenowane na COCO i przekonwertowane dla loadera LibreYOLO.

## Walidacja

`val()` zwraca `metrics/mIoU` i `metrics/pixel_accuracy`, zmierzone względem
dowolnego zbioru danych w formacie użytym do trenowania.

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
