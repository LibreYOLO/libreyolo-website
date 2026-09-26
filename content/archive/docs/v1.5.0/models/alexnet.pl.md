---
title: AlexNet
families:
  - alexnet
seo_title: 'AlexNet: uruchamianie klasycznego klasyfikatora ImageNet w LibreYOLO'
description: >-
  Predykcja, walidacja i eksport AlexNet w LibreYOLO. Wagi torchvision na
  licencji BSD-3-Clause; dostrajanie nie jest jeszcze obsługiwane.
lead: >-
  AlexNet to sieć konwolucyjna, która wygrała ILSVRC 2012 i pomogła
  zapoczątkować erę uczenia głębokiego w wizji komputerowej. LibreYOLO dostarcza
  późniejszą, jednowieżową rewizję tej architektury do klasyfikacji obrazów.
keywords:
  - AlexNet
  - ImageNet
  - sieć konwolucyjna
  - klasyfikacja obrazów python
  - wstępnie wytrenowany klasyfikator obrazów
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreAlexNetb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")

        # data to katalog główny z podziałami train/ i val/ w folderach klas
        # (układ ImageFolder), a nie YAML zbioru danych.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreAlexNetb-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreAlexNetb-cls.pt format=onnx
        libreyolo export model=LibreAlexNetb-cls.pt format=tensorrt half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Fabryka wybiera ścieżkę na podstawie sufiksu pliku, więc
        # wyeksportowany artefakt ładuje się jak każdy checkpoint i zwraca
        # ten sam obiekt Results.
        model = LibreYOLO("LibreAlexNetb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 68c09f080c74bb87
---

## Instalacja

AlexNet nie wymaga opcjonalnego extra. Wszystko, co importuje, znajduje się
w instalacji bazowej.

```bash
pip install libreyolo
```

## Predykcja

Wagi są pobierane z Hugging Face przy pierwszym użyciu i zapisywane lokalnie
w pamięci podręcznej.

<code-tabs name="predict" />

Klasyfikator zwraca `result.probs` zamiast `result.boxes`: `top1`
i `top5` podają indeksy klas, a `top1conf` i `top5conf` ich wskaźniki
pewności. Informacje o źródłach, streamingu i obsłudze wyników zawiera
[predykcja](/docs/predict).

## Warianty

Jeden rozmiar. Dostarczany graf to późniejsza, jednowieżowa rewizja wydana
przez torchvision, z 64 filtrami w pierwszej warstwie i bez local response
normalization, a nie oryginalna architektura z 2012 roku działająca na dwóch
GPU. LibreYOLO dostarcza tę rodzinę wyłącznie do inferencji: predykcja,
walidacja top-1/top-5 w stylu ImageNet oraz eksport są obsługiwane,
a dostrajanie nie zostało zaimplementowane.

## Walidacja

`val()` działa na podziale w formacie ImageFolder (katalog z podfolderami
`train/` i `val/`, po jednym folderze na klasę) i zwraca dokładność top-1 i top-5.

<code-tabs name="val" />

## Eksport

<export-matrix />

Wyeksportowany artefakt ładuje się z powrotem przez `LibreYOLO()` na podstawie
sufiksu pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint
i zwraca ten sam `Results`. [Eksport](/docs/export) wymienia argumenty
przyjmowane przez każdy format oraz te dodatkowe, które wnoszą niektóre z nich.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box>
