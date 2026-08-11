---
title: LingBot-Vision
families:
  - lingbotvision
seo_title: 'LingBot-Vision: segmentacja semantyczna w LibreYOLO'
description: >-
  Używaj modelu LingBot-Vision w LibreYOLO do segmentacji semantycznej z
  backbone ViT na licencji Apache-2.0. Instalacja, predykcja, trenowanie,
  walidacja i eksport, rozmiary s/b/l.
lead: >-
  LingBot-Vision to rodzina backbone'ów wizyjnych transformer trenowanych
  samonadzorowanie za pomocą modelowania maskowanego skupionego na granicach do
  gęstej percepcji przestrzennej, opublikowana przez Robbyant. LibreYOLO łączy
  backbone z głowicą gęstą i obsługuje go w jednym zadaniu, segmentacji
  semantycznej.
keywords:
  - LingBot-Vision
  - segmentacja semantyczna
  - transformer wizyjny
  - uczenie samonadzorowane
  - modelowanie granic
  - Robbyant
  - predykcja gęsta
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLingBotVisions-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (sonda liniowa)
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Backbone jest domyślnie zamrożony, zgodnie z protokołem oceny
        # projektu źródłowego: trenowana jest tylko głowica gęsta 1x1.
        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(data="my-dataset.yaml", epochs=20, imgsz=512, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 imgsz=512 batch=16
    - label: Pełne dostrajanie
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(
            data="my-dataset.yaml", epochs=20, imgsz=512, batch=16,
            freeze_backbone=False,
        )
    - label: Wiele GPU
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLingBotVisions-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="coreai", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLingBotVisions-sem.pt format=onnx imgsz=512
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, dlatego

        # artefakt ładuje się jak punkt kontrolny i zwraca ten sam obiekt
        Results.

        model = LibreYOLO("LibreLingBotVisions-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: c47b33fdc6fa1139
---

## Instalacja

LingBot-Vision nie wymaga opcjonalnych dodatków. Wszystkie importowane przez
niego elementy znajdują się w instalacji podstawowej.

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
`iou` są przyjmowane dla zgodności API, ale nie zmieniają wyniku, ponieważ model
zwraca jedną klasę na piksel, a nie detekcje wymagające filtrowania. Zobacz
[predykcję](/docs/predict), aby poznać źródła, strumieniowanie i obsługę wyników.

## Warianty

Opublikowano trzy rozmiary, s, b i l, wydestylowane z modelu nauczyciela ViT-g/16
o 1,1 mld parametrów. Sam nauczyciel w rozmiarze `g` może być ładowany i
dostrajany w LibreYOLO, ale LibreYOLO nie udostępnia własnego punktu kontrolnego `g`.

<checkpoint-table />

## Trenowanie

Metoda `train()` dostraja opublikowany punkt kontrolny. Domyślna konfiguracja
odpowiada sondzie liniowej z raportu źródłowego: backbone ViT jest zamrożony,
a trenowana jest wyłącznie głowica gęsta 1x1, tak jak podczas tworzenia wag
udostępnianych przez LibreYOLO powyżej. Przekaż `freeze_backbone=False`, aby
zamiast tego dostroić całą sieć, i odpowiednio zmniejsz `lr0`.

<code-tabs name="train" />

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

W dokumentacji projektu źródłowego opisano ViT jako zbudowany na architekturze
DINOv2/DINOv3 opublikowanej przez Meta AI. Robbyant rozpowszechnia swoją
implementację na licencji Apache-2.0, a ten port LibreYOLO utworzono wyłącznie
na podstawie repozytorium Robbyant, nigdy na podstawie kodu DINOv2 lub DINOv3
firmy Meta.

</provenance-box>

## Cytowanie

<citation-block />

