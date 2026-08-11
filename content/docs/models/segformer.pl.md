---
title: SegFormer
families:
  - segformer
seo_title: 'SegFormer: segmentacja semantyczna w LibreYOLO'
description: >-
  Używaj modelu SegFormer w LibreYOLO do segmentacji semantycznej ADE20K w
  rozmiarach b0-b5. Instalacja, predykcja, trenowanie i eksport; wagi wstępnie
  wytrenowane są przeznaczone do użytku niekomercyjnego.
lead: >-
  SegFormer to transformer do segmentacji semantycznej, który łączy
  hierarchiczny enkoder Mix Transformer (MiT) z lekkim dekoderem opartym
  wyłącznie na MLP, unikając ciężkich dekoderów i stałych kodowań pozycyjnych
  potrzebnych we wcześniejszych transformerach do segmentacji. LibreYOLO
  obsługuje go w jednym zadaniu, segmentacji semantycznej, w sześciu rozmiarach.
keywords:
  - SegFormer
  - segmentacja semantyczna
  - Mix Transformer
  - MiT
  - transformer do segmentacji
  - ADE20K
  - predykcja gęsta
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSegformerb0-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (dostrajanie)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: Od podstaw
      language: python
      code: >
        from libreyolo.models.segformer.model import LibreSegformer


        # Brak model_path oznacza losową inicjalizację bez pobierania. To jedyna

        # droga do wag wolnych od niekomercyjnych warunków wstępnie
        wytrenowanych punktów kontrolnych.

        model = LibreSegformer(size="b0", nb_classes=150)

        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: Wiele GPU
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512

        libreyolo export model=LibreSegformerb0-sem.pt format=tensorrt imgsz=512
        half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, dlatego

        # artefakt ładuje się jak punkt kontrolny i zwraca ten sam obiekt
        Results.

        model = LibreYOLO("LibreSegformerb0-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: c236895b991beabf
---

## Instalacja

SegFormer nie wymaga opcjonalnych dodatków. Wszystkie importowane przez niego elementy znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej pamięci podręcznej.

<code-tabs name="predict" />

`result.semantic_mask` zawiera gęstą mapę klas: `.data` to tensor `(H, W)`
z identyfikatorami klas w oryginalnym rozmiarze obrazu, a `.classes` zawiera
identyfikatory klas rzeczywiście obecnych na obrazie. `result.boxes` ma wartość
`None`, ponieważ nie ma detekcji poszczególnych instancji. Parametry `conf` i
`iou` są przyjmowane dla zgodności API, ale nie zmieniają wyniku: model zwraca
jedną klasę na piksel, a nie detekcje instancji wymagające filtrowania lub
usuwania duplikatów. Informacje o źródłach, strumieniowaniu i obsłudze wyników
znajdziesz w sekcji [predykcja](/docs/predict).

## Warianty

Dostępnych jest sześć rozmiarów, od b0 do b5. Każdy kolejny poszerza i pogłębia
enkoder Mix Transformer, zachowując tę samą konstrukcję dekodera opartą wyłącznie na MLP.

<checkpoint-table />

## Trenowanie

Metoda `train()` domyślnie dostraja opublikowany punkt kontrolny. Zamiast tego
wywołaj `LibreSegformer(...)` bez `model_path`, aby zbudować model z losowo
zainicjalizowanym enkoderem i głowicą oraz trenować go od podstaw. Jest to jedyna
droga do wag nieobjętych niekomercyjnym ograniczeniem wstępnie wytrenowanych
punktów kontrolnych (zobacz [Licencjonowanie](#licensing)).

<code-tabs name="train" />

Przy ustawieniach domyślnych trener stosuje konfigurację ADE20K z publikacji
SegFormer: AdamW z podstawową szybkością uczenia dla backbone i głowicą dekodera
trenowaną z szybkością 10 razy większą, zanik wag wszędzie poza LayerNorm i
konwolucją pozycyjną Mix-FFN oraz liniowy harmonogram zaniku z rozgrzewką.
Zbieżność większych rozmiarów, od b3 do b5, nie została zweryfikowana kompleksowo.

Informacje o zbiorach danych, augmentacji, wielu GPU i loggerach znajdziesz w sekcji [trenowanie](/docs/train).

## Walidacja

Metoda `val()` zwraca słownik kluczy `metrics/`: mIoU i dokładność pikselową,
mierzone na dowolnym zbiorze danych w formacie użytym do trenowania.

<code-tabs name="val" />

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ponownie ładowany przez `LibreYOLO()` na podstawie
rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak punkt
kontrolny i zwraca ten sam obiekt `Results`. Sekcja [eksport](/docs/export)
zawiera argumenty akceptowane przez każdy format.

<code-tabs name="export" />

## Punkty kontrolne

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box>

Enkoder i głowica dekodera LibreSegformer są portem implementacji SegFormer
z Hugging Face Transformers na licencji Apache-2.0, a nie portem
NVlabs/SegFormer. Oryginalnego repozytorium NVIDIA nigdy nie odczytywano ani
nie kopiowano, a wymieniono je tutaj wyłącznie w celu przypisania autorstwa
autorom publikacji. Niekomercyjne ograniczenie NVIDIA dotyczy tylko powyższych
wstępnie wytrenowanych punktów kontrolnych. Architektura i własny kod LibreYOLO
pozostają w całości na licencji MIT.

</provenance-box>

## Cytowanie

<citation-block />

