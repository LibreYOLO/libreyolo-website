---
title: DeepLabv3
families:
  - deeplabv3
seo_title: 'DeepLabv3: przewidywanie i eksportowanie segmentacji semantycznej ASPP'
description: >-
  Użyj DeepLabv3 w LibreYOLO do segmentacji semantycznej. Instaluj, przewidywaj,
  sprawdzaj i eksportuj checkpointy ResNet i MobileNetV3 firmy Torchvision.
lead: >-
  Sieć segmentacji semantycznej, która łączy cechy z kilkoma współczynnikami
  dylatacji równolegle (okropne łączenie piramid przestrzennych) przed
  sklasyfikowaniem każdego piksela. LibreYOLO dostarcza go wyłącznie do
  segmentacji semantycznej.
keywords:
  - DeepLabv3
  - okropne przestrzenne piramidowanie
  - ASPP
  - segmentacja semantyczna
  - gęsta prognoza
last_verified: 1.5.0
snippets:
  predict:
    - label: Pyton
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
    - label: Pyton
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
    - label: Pyton
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

        # jak każdy punkt kontrolny i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreDeepLabv3r50-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 7abf11ebb6cece18
---
## Zainstaluj

DeepLabv3 nie wymaga wyposażenia dodatkowego. Wszystko, co importuje, znajduje się w bazie
zainstaluj.

ZXQBLOCK0QXZ

## Przewiduj

Ciężary pobierane są z Hugging Face przy pierwszym użyciu i zapisywane lokalnie w pamięci podręcznej. The
Dla tej rodziny wymagany jest przyrostek nazwy pliku `-sem`.

<code-tabs name="predict" />

Segmentacja semantyczna zwraca jeden identyfikator klasy na piksel, a nie pola, tzw
`result.semantic_mask` zawiera tablicę `(H, W)` na `.data` i listę
identyfikatory klas obecne na obrazie na `.classes`. `conf`, `iou` i `max_det` są
akceptowane dla parzystości API, ale nie mają żadnego efektu: model przypisuje każdemu klasę
piksel według argmax, bez progu ufności lub kroku NMS. Zobacz
[przewidywanie](/docs/predict) dla źródeł, przesyłania strumieniowego i obsługi wyników.

## Warianty

Trzy kręgosłupy: rozszerzony ResNet-50, rozszerzony ResNet-101 i rozszerzony
MobileNetV3-Large. To jest DeepLabv3, a nie DeepLabv3+, więc nie ma dekodera
etap lub udoskonalenie CRF, pasujące raczej do implementacji Torchvision niż do
własny kod referencyjny papieru.

LibreYOLO nie trenuje DeepLabv3: `train()` podnosi `NotImplementedError` dla
ta rodzina, którą [poziom wsparcia] (/docs/models) powyżej oznacza jako wniosek
tylko. Trzy opublikowane checkpointy to COCO-with-VOC-label firmy Torchvision
obciążniki przeliczone dla ładowarki LibreYOLO.

## Sprawdź

`val()` zwraca `metrics/mIoU` i `metrics/pixel_accuracy`, mierzone względem
dowolny zbiór danych w formacie, na którym trenowałeś.

<code-tabs name="val" />

## Eksportuj

<export-matrix />

Wyeksportowany artefakt wczytuje się ponownie poprzez `LibreYOLO()` zgodnie z sufiksem pliku, więc
Plik `.onnx` lub `.engine` zachowuje się jak punkt kontrolny i zwraca to samo
`Results`. [Eksport](/docs/export) wyświetla listę argumentów akceptowanych przez każdy format.

<code-tabs name="export" />

## Punkty kontrolne

Każdy opublikowany plik wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box>
