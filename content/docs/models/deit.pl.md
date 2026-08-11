---
title: DeiT
families:
  - deit
seo_title: 'Klasyfikator obrazów DeiT: przewiduj, waliduj, eksportuj'
description: >-
  Uruchom klasyfikatory obrazów DeiT w LibreYOLO: zamrożone, tylko do
  wnioskowania, rodziny muzealne w rozmiarach tiny, small i base, pod
  Apache-2.0.
lead: >-
  DeiT (Data-efficient image Transformer) jest zwykłym klasyfikatorem Vision
  Transformer trenowanym wyłącznie na ImageNet-1k, bez dodatkowych danych do
  wstępnego treningu. LibreYOLO zawiera wersje tiny, small i base o rozmiarze
  patch-16 jako zamrożone, wyłącznie do wnioskowania.
keywords:
  - DeiT
  - Transformator Wizji
  - ViT
  - klasyfikacja obrazów
  - ImageNet
  - wydajne pod względem danych trenowanie
  - rodzina muzeum
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeiTb-cls.pt")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeiTb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeiTb-cls.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeiTb-cls.pt format=onnx
        libreyolo export model=LibreDeiTb-cls.pt format=tensorrt half=True
    - label: Użyj wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka kieruje na podstawie rozszerzenia pliku, więc eksportowany
        artefakt się ładuje

        # jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreDeiTb-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 9c67c8554b2af5c6
---
## Instalacja

DeiT nie potrzebuje niczego dodatkowego poza podstawowym pakietem.

```bash
pip install libreyolo
```

## Predykcja

Ta rodzina jest tylko do wnioskowania: `train()` podnosi `NotImplementedError`, więc ta strona nie ma sekcji Train. Obsługiwane są Predict, validate i export. Wagi są pobierane z Hugging Face przy pierwszym użyciu i są przechowywane lokalnie w pamięci podręcznej. Sufiks `-cls` w nazwie pliku jest wymagany i wybiera zadanie klasyfikacji.

<code-tabs name="predict" />

Zwrócony obiekt `Results` zawiera tensor `probs` zamiast `boxes`; `top1` i `top5` indeksują 1 000 klas ImageNet-1k, a `top1conf` jest wynikiem softmax dla najlepszej prognozy. Każdy rozmiar ma stałą rozdzielczość wejściową z osadzenia pozycyjnego: wstępne przetwarzanie zmienia rozmiar i przycina do środka do tej rozdzielczości, a podanie innego `imgsz` powoduje błąd zamiast cichego przeskalowania. Zobacz [prediction](/docs/predict) dla źródeł, streaming i obsługi wyników.

## Walidacja

`val()` zwraca słownik z dokładnością top-1 i top-5, mierzony względem zbioru danych ułożonego w konwencjonalnej strukturze folderów `train/<class>/` i `val/<class>/`.

<code-tabs name="val" />

## Eksport

<export-matrix />

Eksportowany artefakt ładuje się z powrotem przez `LibreYOLO()` na podstawie jego rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint i zwraca ten sam `Results`. Uruchamianie grafu w środowisku uruchomieniowym bez żadnego zainstalowanego LibreYOLO jest również obsługiwane, ale wtedy wstępne i końcowe przetwarzanie należy napisać samodzielnie.

<code-tabs name="export" />

## Checkpointy

Każdy opublikowany plik wagowy dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box>

## Cytowanie

<citation-block /> 
