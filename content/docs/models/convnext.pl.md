---
title: ConvNeXt
families:
  - convnext
seo_title: 'ConvNeXt: trenowanie, walidacja i eksport na licencji Apache-2.0'
description: >-
  Używaj ConvNeXt w LibreYOLO do klasyfikacji obrazów. Instaluj, uruchamiaj
  predykcję, dostrajaj za pomocą LoRA, waliduj i eksportuj modele LibreConvNeXt
  tiny/small/base.
lead: >-
  ConvNeXt to klasyfikator obrazów zbudowany wyłącznie ze standardowych splotów,
  modernizowany blok po bloku od ResNet w kierunku rozwiązań stosowanych w
  transformerach wizyjnych. LibreYOLO obsługuje go w jednym zadaniu:
  klasyfikacji.
keywords:
  - ConvNeXt
  - ConvNeXt tiny
  - klasyfikacja obrazów
  - konwolucyjna sieć neuronowa
  - klasyfikator ImageNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreConvNeXtt-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 epochs=5
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreConvNeXtt-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreConvNeXtt-cls.pt format=onnx
        libreyolo export model=LibreConvNeXtt-cls.pt format=tensorrt half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie sufiksu pliku, więc
        wyeksportowany

        # artefakt wczytuje się jak każdy checkpoint i zwraca ten sam obiekt
        Results.

        model = LibreYOLO("LibreConvNeXtt-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 1682cc69cf2925e6
---

## Instalacja

ConvNeXt nie wymaga opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji bazowej.

```bash
pip install libreyolo
```

Wyjątkiem jest dostrajanie adapterów z `lora=True`, które wymaga dodatku `lora`.

```bash
pip install "libreyolo[lora]"
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane lokalnie
w pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, dlatego zamiana
modelu wymaga zmiany jednego wiersza. Klasyfikator nie zawiera ramek ani masek.
Pole `result.probs` przechowuje predykcję dla całego obrazu i udostępnia
`top1`, `top5`, `top1conf` oraz `top5conf`. Argumenty `conf`, `iou` i `max_det`
są akceptowane dla zgodności API, ale nie mają wpływu na wynik, ponieważ
w pojedynczym wektorze prawdopodobieństwa nie ma wartości do progowania ani
tłumienia. Zobacz stronę [predykcji](/docs/predict), aby poznać źródła,
streaming i obsługę wyników.

## Warianty

Dostępne są trzy rozmiary: tiny, small i base. Wszystkie są trenowane i oceniane
w ten sam sposób, dlatego wybór sprowadza się do kompromisu między liczbą
parametrów a accuracy. Zadanie jest stałe: każdy rozmiar obsługuje tylko
klasyfikację. Nazwa pliku wag dla każdego rozmiaru kończy się na `-cls.pt`.
Fabryka odczytuje ten sufiks, aby wybrać rodzinę, więc argument `task=` nie jest
potrzebny.

## Trenowanie

Dostrajanie rozpoczyna się od opublikowanego backbone ImageNet, a końcowa
warstwa klasyfikatora jest automatycznie przebudowywana zgodnie z liczbą klas
w docelowym zbiorze danych.

<code-tabs name="train" />

Przy ustawieniach domyślnych moduł trenujący wykonuje 100 epok z `lr0=1e-3`,
optymalizatorem AdamW, batchem 64 i early stopping po 50 epokach bez poprawy.
Argument `data` przyjmuje katalog główny zbioru danych (`train/` i `val/`,
po jednym folderze na klasę), znaną krótką nazwę, na przykład `imagenette160`,
albo adres URL pliku `.zip`. Bloki ConvNeXt zawierają warstwy MLP `nn.Linear`
wymagane przez LoRA, dlatego `lora=True` jest tu obsługiwane i wstrzykuje
adaptery do warstw MLP bloków zamiast dostrajać cały backbone.

Zobacz stronę [trenowania](/docs/train), aby poznać zbiory danych, augmentację,
obsługę wielu GPU i loggery.

## Walidacja

Metoda `val()` zwraca słownik kluczy `metrics/`. Dla klasyfikacji są to
accuracy top-1 i top-5 w podziale walidacyjnym.

<code-tabs name="val" />

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ponownie wczytywany przez `LibreYOLO()` na
podstawie sufiksu pliku, dlatego plik `.onnx` lub `.engine` zachowuje się jak
checkpoint i zwraca ten sam obiekt `Results`. Strona [Eksport](/docs/export)
zawiera argumenty obsługiwane przez każdy format oraz dodatki wymagane przez
niektóre z nich.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box>

Ta rodzina udostępnia tylko ConvNeXt V1. Małe, wstępnie wytrenowane checkpointy
ConvNeXt-V2 są objęte licencją CC-BY-NC 4.0 i celowo je wykluczono, ponieważ wag
na licencji niekomercyjnej nie można rozpowszechniać w bibliotece przeznaczonej
do zastosowań komercyjnych i objętej licencją MIT.

</provenance-box>

## Cytowanie

<citation-block />
