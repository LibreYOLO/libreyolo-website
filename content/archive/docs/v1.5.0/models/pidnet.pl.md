---
title: PIDNet
families:
  - pidnet
seo_title: 'PIDNet: predykcja i eksport segmentacji czasu rzeczywistego na licencji MIT'
description: >-
  Używaj PIDNet w LibreYOLO do segmentacji semantycznej czasu rzeczywistego.
  Instaluj, przewiduj, waliduj i eksportuj checkpointy s/m/l Cityscapes na
  licencji MIT.
lead: >-
  Trójgałęziowa sieć segmentacji semantycznej, która dodaje dedykowaną gałąź
  granic do projektu inspirowanego regulatorem
  proporcjonalno-całkująco-różniczkującym, z myślą o inferencji czasu
  rzeczywistego. LibreYOLO udostępnia ją wyłącznie do segmentacji semantycznej.
keywords:
  - PIDNet
  - segmentacja semantyczna w czasie rzeczywistym
  - segmentacja z granicami
  - Cityscapes
  - gęsta predykcja
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePIDNets-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # (H, W), identyfikatory klas

        print(mask.classes)      # posortowane identyfikatory klas obecnych na
        obrazie
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePIDNets-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePIDNets-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePIDNets-sem.pt format=onnx
        libreyolo export model=LibrePIDNets-sem.pt format=tensorrt half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibrePIDNets-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 489db64a39e3a61a
---

## Instalacja

PIDNet nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej. Sufiks nazwy pliku `-sem` jest wymagany dla tej rodziny.

<code-tabs name="predict" />

Segmentacja semantyczna zwraca jeden identyfikator klasy na piksel, a nie ramki.
Dlatego `result.semantic_mask` zawiera tablicę `(H, W)` w polu `.data` oraz listę
identyfikatorów klas obecnych na obrazie w `.classes`. Argumenty `conf`, `iou` i
`max_det` są przyjmowane dla zgodności API, ale nie mają wpływu na wynik. Model
przypisuje klasę każdemu pikselowi przez argmax, bez progu pewności ani etapu
NMS. Więcej informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępne są trzy rozmiary, wszystkie ze stałym wejściem 1024 px. Opublikowane
checkpointy są konwersjami oficjalnych wag PIDNet wytrenowanych na Cityscapes z
19 klasami.

LibreYOLO nie trenuje PIDNet: `train()` zgłasza `NotImplementedError` dla tej
rodziny, którą powyższy [poziom obsługi](/docs/models) oznacza jako przeznaczoną
wyłącznie do inferencji.

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

## Cytowanie

<citation-block />
