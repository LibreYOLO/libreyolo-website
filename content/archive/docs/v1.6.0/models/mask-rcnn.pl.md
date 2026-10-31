---
title: Mask R-CNN
families:
  - mask_rcnn
seo_title: 'Mask R-CNN w LibreYOLO: predykcja, walidacja i eksport'
description: >-
  Uruchamiaj Mask R-CNN w LibreYOLO do detekcji obiektów i segmentacji
  instancji. Instaluj, przewiduj, waliduj i eksportuj port torchvision na
  licencji BSD-3-Clause.
lead: >-
  Mask R-CNN dodaje do Faster R-CNN gałąź maski dla każdego regionu, przewidując
  maskę segmentacji obok każdej wykrytej ramki. LibreYOLO przenosi implementację
  torchvision do detekcji i segmentacji instancji.
keywords:
  - Mask R-CNN
  - segmentacja instancji
  - detekcja obiektów
  - Faster R-CNN
  - torchvision
  - detektor dwustopniowy
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMaskRCNNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Tylko ramki
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # task="detect" pomija głowicę masek i zwraca ramki z tego samego
        # checkpointu, bez masek w wyniku.
        model = LibreYOLO("LibreMaskRCNNr50.pt", task="detect")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])      # maski
        print(metrics["metrics/mAP50-95(B)"])   # ramki
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMaskRCNNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMaskRCNNr50.pt format=onnx imgsz=800
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreMaskRCNNr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.masks.data.shape)
source_hash: 9608459b801aa6d5
---

## Instalacja

Mask R-CNN nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane
elementy znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zmiana
detektora wymaga zmiany jednego wiersza. Wczytanie checkpointu bez argumentu
`task` zwraca maski instancji, ponieważ segmentacja jest domyślnym zadaniem tej
rodziny. `result.masks` zawiera je wtedy obok ramek. Przekazanie
`task="detect"` wczytuje te same wagi bez głowicy masek i zwraca wyłącznie
ramki. Argumenty `conf` i `iou` ustawiają progi pewności i NMS. Mask R-CNN
zachowuje źródłowy etap NMS, w przeciwieństwie do detektora opartego na
zapytaniach. Więcej informacji o źródłach, streamingu i obsłudze wyników zawiera
strona [predykcji](/docs/predict).

## Warianty

Dostępny jest jeden backbone, ResNet-50 z piramidą cech, korzystający z buildera
Mask R-CNN v2 w torchvision. Opublikowany checkpoint podlega licencji
BSD-3-Clause i obsługuje oba zadania tej rodziny, dlatego nie ma rozmiaru do
wyboru.

## Walidacja

`val()` zwraca słownik kluczy `metrics/`. Dla domyślnego zadania segmentacji
tego checkpointu zwykły klucz `metrics/mAP50-95` zawiera wynik maski, a ten sam
przebieg zwraca wynik ramek pod sufiksem `(B)`, dzięki czemu oba są dostępne
jednocześnie.

<code-tabs name="val" />

## Eksport

<export-matrix />

Mask R-CNN eksportuje wyłącznie do ONNX z rozmiarem batcha 1. Wyeksportowany
graf zachowuje wewnątrz źródłowe etapy zmiany rozmiaru i wklejania maski,
dlatego LibreYOLO wymusza `dynamic=True` niezależnie od przekazanej wartości,
aby graf działał dla źródeł niebędących kwadratami. Wyeksportowany plik `.onnx`
jest ponownie ładowany przez `LibreYOLO()` na podstawie rozszerzenia i zwraca ten
sam obiekt `Results`.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny. Poniższy pojedynczy checkpoint
jest wymieniony pod detekcją, ale ten sam plik wczytuje się także do segmentacji.
Bez argumentu `task` domyślnie zwraca maski.

<checkpoint-table />

## Licencja

<provenance-box>

Mask R-CNN jest zbudowany jako podklasa wrappera Faster R-CNN w LibreYOLO.
Współdzieli to samo źródło torchvision i licencję BSD-3-Clause oraz dodaje
predyktor masek i głowicę RoI masek z tego samego przeniesionego commita.

</provenance-box>
