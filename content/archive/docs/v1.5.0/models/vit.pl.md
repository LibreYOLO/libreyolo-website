---
title: ViT
families:
  - vit
seo_title: 'ViT: uruchamianie klasycznych klasyfikatorów Vision Transformer w LibreYOLO'
description: >-
  Przewiduj, waliduj i eksportuj klasyfikatory ViT za pomocą LibreYOLO. Wagi
  AugReg na licencji Apache-2.0. Dostrajanie nie jest jeszcze obsługiwane.
lead: >-
  Klasyczny Vision Transformer: czysty transformer stosowany do patchy obrazu o
  stałym rozmiarze, z wyuczonym tokenem klasy i bez splotów. LibreYOLO
  udostępnia do klasyfikacji obrazów cztery rozmiary wstępnie wytrenowane metodą
  AugReg.
keywords:
  - ViT
  - Vision Transformer
  - AugReg
  - klasyfikacja obrazów
  - klasyfikator transformerowy
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTti-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreViTti-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")

        # data to katalog główny z podziałem train/ i val/ na foldery klas
        # (układ ImageFolder), a nie plik YAML zbioru danych.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreViTti-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreViTti-cls.pt format=onnx
        libreyolo export model=LibreViTti-cls.pt format=tensorrt half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreViTti-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: f63e98454913765a
---

## Instalacja

ViT nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Klasyfikator zwraca `result.probs` zamiast `result.boxes`. Pola `top1` i `top5`
zawierają indeksy klas, a `top1conf` i `top5conf` ich pewności. Przetwarzanie
wstępne skaluje i przycina centralnie do stałego wejścia 224 px, używając
procedury ewaluacyjnej AugReg z timm: interpolacji dwusześciennej przy
współczynniku przycięcia 0.9. Więcej informacji o źródłach, streamingu i
obsłudze wyników zawiera strona [predykcji](/docs/predict).

## Warianty

Dostępne są cztery rozmiary, od tiny do large. Współdzielą jeden stały graf
224 px z patchami 16, a różnią się szerokością embeddingu i głębokością
transformera. LibreYOLO udostępnia tę rodzinę wyłącznie do inferencji.
Obsługiwane są predykcja, walidacja top-1/top-5 w stylu ImageNet oraz eksport,
a procedura dostrajania AugReg nie jest zaimplementowana.

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
