---
title: DeepLabv3
families:
  - deeplabv3
seo_title: 'DeepLabv3: predykcja i eksportowanie segmentacji semantycznej ASPP'
description: >-
  Użyj DeepLabv3 w LibreYOLO do segmentacji semantycznej. Instaluj, przewidywaj,
  sprawdzaj i eksportuj checkpointy ResNet i MobileNetV3 firmy Torchvision.
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

DeepLabv3 nie wymaga wyposażenia dodatkowego. Wszystko, co importuje, znajduje się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Wagi pobierane są z Hugging Face przy pierwszym użyciu i zapisywane lokalnie w pamięci podręcznej. Dla tej rodziny wymagany jest przyrostek nazwy pliku `-sem`.

<code-tabs name="predict" />

Semantyczna segmentacja zwraca jeden identyfikator klasy na piksel, a nie pola, więc `result.semantic_mask` zawiera tablicę `(H, W)` na `.data` i listę identyfikatorów klas obecnych w obrazie na `.classes`. `conf`, `iou` i `max_det` są akceptowane dla zgodności API, ale nie mają żadnego efektu: model przypisuje klasę do każdego piksela przez argmax, bez progu ufności lub kroku NMS. Zobacz [predykcja](/docs/predict) dla źródeł, streaming i obsługi wyników.

## Warianty

Trzy backbones: rozszerzone ResNet-50, rozszerzone ResNet-101 i rozszerzone MobileNetV3-Large. To jest DeepLabv3, a nie DeepLabv3+, więc nie ma tu żadnego stopnia dekodera ani udoskonalenia CRF, pasującego do implementacji torchvision, a nie do własnego kodu referencyjnego artykułu.

LibreYOLO nie trenuje DeepLabv3: `train()` podnosi `NotImplementedError` dla tej rodziny, co [poziom wsparcia](/docs/models) powyżej oznacza jedynie jako inferencja. Trzy opublikowane checkpointy to własne wagi COCO-with-VOC-label firmy Torchvision, przeliczone dla ładowarki LibreYOLO.

## Walidacja

`val()` zwraca `metrics/mIoU` i `metrics/pixel_accuracy`, mierzone względem dowolnego zbioru danych w formacie, na którym trenowałeś.

<code-tabs name="val" />

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ładowany ponownie poprzez `LibreYOLO()` zgodnie z sufiksem pliku, zatem plik `.onnx` lub `.engine` zachowuje się jak checkpoint i zwraca ten sam komunikat `Results`. [Eksport](/docs/export) wyświetla listę argumentów akceptowanych przez każdy format.

<code-tabs name="export" />

## Checkpointy

Każdy opublikowany plik wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box>
