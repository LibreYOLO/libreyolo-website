---
title: DeiT
families:
  - deit
seo_title: 'Klasyfikator obrazów DeiT: przewiduj, waliduj, eksportuj'
description: >-
  Uruchamiaj klasyfikatory obrazów DeiT w LibreYOLO. Ta historyczna rodzina,
  przeznaczona wyłącznie do inferencji, obejmuje rozmiary tiny, small i base
  na licencji Apache-2.0.
lead: >-
  DeiT (Data-efficient image Transformer) jest zwykłym klasyfikatorem Vision
  Transformer trenowanym wyłącznie na ImageNet-1k, bez dodatkowych danych do
  wstępnego treningu. LibreYOLO zawiera wersje tiny, small i base o rozmiarze
  patch-16 jako zamrożone modele przeznaczone wyłącznie do inferencji.
keywords:
  - DeiT
  - transformer wizyjny
  - ViT
  - klasyfikacja obrazów
  - ImageNet
  - trenowanie efektywne pod względem danych
  - historyczne modele klasyfikacyjne
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

Ta rodzina służy wyłącznie do inferencji. Metoda `train()` zgłasza
`NotImplementedError`, dlatego na tej stronie nie ma sekcji Trenowanie.
Obsługiwane są predykcja, walidacja i eksport. Przy pierwszym użyciu wagi są
pobierane z Hugging Face i zapisywane lokalnie w pamięci podręcznej. Wymagany
sufiks nazwy pliku `-cls` wybiera zadanie klasyfikacji.

<code-tabs name="predict" />

Zwracany obiekt `Results` zawiera tensor `probs` zamiast `boxes`. Pola `top1`
i `top5` indeksują 1000 klas ImageNet-1k, a `top1conf` jest wynikiem softmax
dla najlepszej predykcji. Każdy rozmiar ma stałą rozdzielczość wejściową
wynikającą z embeddingu pozycyjnego. Przetwarzanie wstępne zmienia rozmiar
i wykonuje kadrowanie centralne do tej rozdzielczości, natomiast podanie innego
`imgsz` zgłasza błąd zamiast niejawnie przeskalować dane. Zobacz stronę
[predykcji](/docs/predict), aby poznać źródła, streaming i obsługę wyników.

## Walidacja

`val()` zwraca słownik z dokładnością top-1 i top-5, mierzony względem zbioru danych ułożonego w konwencjonalnej strukturze folderów `train/<class>/` i `val/<class>/`.

<code-tabs name="val" />

## Eksport

<export-matrix />

Eksportowany artefakt ładuje się z powrotem przez `LibreYOLO()` na podstawie jego rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint i zwraca ten sam `Results`. Uruchamianie grafu w środowisku uruchomieniowym bez żadnego zainstalowanego LibreYOLO jest również obsługiwane, ale wtedy wstępne i końcowe przetwarzanie należy napisać samodzielnie.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencjonowanie

<provenance-box></provenance-box>

## Cytowanie

<citation-block /> 
