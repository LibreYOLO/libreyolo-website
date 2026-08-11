---
title: VGG
families:
  - vgg
seo_title: 'VGG: uruchamianie klasyfikatorów obrazów VGG-16/19 w LibreYOLO'
description: >-
  Przewiduj, waliduj i eksportuj klasyfikatory VGG za pomocą LibreYOLO. Wagi
  torchvision na licencji BSD-3-Clause. Dostrajanie nie jest jeszcze
  obsługiwane.
lead: >-
  VGG to konwolucyjny klasyfikator obrazów zbudowany z jednolitych stosów małych
  splotów 3x3 zamiast większych filtrów. LibreYOLO udostępnia do klasyfikacji
  obrazów rozmiary 16- i 19-warstwowe, w wersji zwykłej i z normalizacją batcha.
keywords:
  - VGG
  - VGG-16
  - VGG-19
  - konwolucyjna sieć neuronowa
  - klasyfikacja obrazów
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVGG16-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreVGG16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")

        # data to katalog główny z podziałem train/ i val/ na foldery klas
        # (układ ImageFolder), a nie plik YAML zbioru danych.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreVGG16-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreVGG16-cls.pt format=onnx
        libreyolo export model=LibreVGG16-cls.pt format=tensorrt half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreVGG16-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 26eb6ff5811533fd
---

## Instalacja

VGG nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Klasyfikator zwraca `result.probs` zamiast `result.boxes`. Pola `top1` i `top5`
zawierają indeksy klas, a `top1conf` i `top5conf` ich pewności. Predykcja działa
ze stałym wejściem 224 px i zgłasza błąd po przekazaniu innego `imgsz`. Więcej
informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępne są cztery rozmiary: 16 i 19 warstw konwolucyjnych, każdy w wersji
zwykłej i z normalizacją batcha. Udostępnione wagi pochodzą z późniejszego
trenowania torchvision od zera na ImageNet, a nie z konwersji oryginalnego
wydania Caffe grupy Oxford z 2014 roku. LibreYOLO udostępnia tę rodzinę
wyłącznie do inferencji. Obsługiwane są predykcja, walidacja top-1/top-5 w stylu
ImageNet oraz eksport, a dostrajanie nie jest zaimplementowane.

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
