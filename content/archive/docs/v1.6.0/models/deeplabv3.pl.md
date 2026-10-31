---
title: DeepLabv3
families:
  - deeplabv3
seo_title: 'DeepLabv3: predykcja i eksport segmentacji semantycznej ASPP'
description: >-
  Używaj DeepLabv3 w LibreYOLO do segmentacji semantycznej. Instaluj, uruchamiaj
  predykcję, waliduj i eksportuj checkpointy ResNet i MobileNetV3 z torchvision.
lead: >-
  Sieć segmentacji semantycznej, która łączy cechy z kilkoma współczynnikami
  dylatacji równolegle (łączenie piramid przestrzennych z dylatacją) przed
  sklasyfikowaniem każdego piksela. LibreYOLO dostarcza go wyłącznie do
  segmentacji semantycznej.
keywords:
  - DeepLabv3
  - łączenie piramid przestrzennych z dylatacją
  - ASPP
  - segmentacja semantyczna
  - gęsta prognoza
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # (H, W) identyfikatory klas

        print(mask.classes)      # posortowane identyfikatory klas obecne na
        obrazku
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeepLabv3r50-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeepLabv3r50-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=onnx

        libreyolo export model=LibreDeepLabv3r50-sem.pt format=tensorrt
        half=True
    - label: Użyj wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Trasy fabryczne na sufiksie pliku, więc wyeksportowany artefakt
        zostaje załadowany

        # jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreDeepLabv3r50-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 7abf11ebb6cece18
---
## Instalacja

DeepLabv3 nie wymaga opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji bazowej.

```bash
pip install libreyolo
```

## Predykcja

Wagi pobierane są z Hugging Face przy pierwszym użyciu i zapisywane lokalnie w pamięci podręcznej. Dla tej rodziny wymagany jest przyrostek nazwy pliku `-sem`.

<code-tabs name="predict" />

Segmentacja semantyczna zwraca jeden identyfikator klasy na piksel, a nie
ramki. Pole `result.semantic_mask.data` zawiera tablicę `(H, W)`, natomiast
`result.semantic_mask.classes` listę identyfikatorów klas obecnych na obrazie.
Argumenty `conf`, `iou` i `max_det` są akceptowane dla zgodności API, ale nie
mają wpływu na wynik. Model przypisuje klasę do każdego piksela przez argmax,
bez progu pewności ani etapu NMS. Zobacz stronę
[predykcji](/docs/predict), aby poznać źródła, streaming i obsługę wyników.

## Warianty

Trzy warianty backbone: ResNet-50 z dylatacją, ResNet-101 z dylatacją oraz
MobileNetV3-Large z dylatacją. Jest to DeepLabv3, a nie DeepLabv3+, dlatego nie
ma etapu dekodera ani udoskonalania CRF. Odpowiada to implementacji torchvision,
a nie kodowi referencyjnemu publikacji.

LibreYOLO nie trenuje DeepLabv3. Metoda `train()` zgłasza
`NotImplementedError` dla tej rodziny, którą [poziom wsparcia](/docs/models)
oznacza jako przeznaczoną wyłącznie do inferencji. Trzy opublikowane checkpointy
zawierają wagi torchvision COCO-with-VOC-label przekonwertowane dla modułu
wczytującego LibreYOLO.

## Walidacja

`val()` zwraca `metrics/mIoU` i `metrics/pixel_accuracy`, mierzone względem dowolnego zbioru danych w formacie, na którym trenowałeś.

<code-tabs name="val" />

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ponownie wczytywany przez `LibreYOLO()` na
podstawie sufiksu pliku, dlatego plik `.onnx` lub `.engine` zachowuje się jak
checkpoint i zwraca ten sam obiekt `Results`. Strona [Eksport](/docs/export)
zawiera argumenty akceptowane przez każdy format.

<code-tabs name="export" />

## Checkpointy

Każdy opublikowany plik wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box>
