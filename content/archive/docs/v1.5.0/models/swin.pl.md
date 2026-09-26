---
title: Swin Transformer
families:
  - swin
seo_title: 'Swin Transformer: klasyfikacja obrazów modelem LibreSwin z LibreYOLO'
description: >-
  Przewiduj, waliduj i eksportuj klasyfikatory Swin Transformer za pomocą
  LibreYOLO. Wagi na licencji MIT. Dostrajanie nie jest jeszcze obsługiwane.
lead: >-
  Swin Transformer V1: hierarchiczny transformer wizyjny, który oblicza uwagę
  wewnątrz przesuniętych okien lokalnych zamiast na całym obrazie. LibreYOLO
  udostępnia cztery rozmiary do klasyfikacji obrazów.
keywords:
  - Swin Transformer
  - hierarchiczny transformer wizyjny
  - uwaga w przesuniętych oknach
  - klasyfikacja obrazów
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwint-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwint-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")

        # data to katalog główny z podziałem train/ i val/ na foldery klas
        # (układ ImageFolder), a nie plik YAML zbioru danych.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwint-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwint-cls.pt format=onnx
        libreyolo export model=LibreSwint-cls.pt format=tensorrt half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreSwint-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: faa6bbacae62d88e
---

## Instalacja

Swin nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Klasyfikator zwraca `result.probs` zamiast `result.boxes`. Pola `top1` i `top5`
zawierają indeksy klas, a `top1conf` i `top5conf` ich pewności. Każdy rozmiar ma
stałe wejście 224 px, ponieważ ostatni etap uwagi jest zbudowany dla tej
rozdzielczości. Predykcja, walidacja i eksport zgłaszają błąd po przekazaniu
innego `imgsz`. Więcej informacji o źródłach, streamingu i obsłudze wyników
zawiera strona [predykcji](/docs/predict).

## Warianty

Dostępne są cztery rozmiary, od tiny do large. Powstały z tej samej wieży
przesuniętych okien, a różnią się szerokością embeddingu i głębokością etapów.
Large jest wstępnie wytrenowany na ImageNet-22k i dostrojony na ImageNet-1k,
natomiast pozostałe trzy wytrenowano bezpośrednio na ImageNet-1k. LibreYOLO
udostępnia tę rodzinę wyłącznie do inferencji. Obsługiwane są predykcja,
walidacja top-1/top-5 w stylu ImageNet oraz eksport, a źródłowa procedura
trenowania ImageNet nie jest zaimplementowana.

## Walidacja

`val()` działa na podziale w stylu ImageFolder (katalog z podfolderami `train/`
i `val/`, po jednym folderze na klasę) i zwraca metryki accuracy top-1 i top-5.

<code-tabs name="val" />

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ponownie ładowany przez `LibreYOLO()` na podstawie
rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint
i zwraca ten sam obiekt `Results`. Strona [eksportu](/docs/export) wymienia
argumenty obsługiwane przez każdy format oraz dodatki wymagane przez niektóre z
nich.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
