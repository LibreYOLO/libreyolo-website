---
title: ResNet
families:
  - resnet
seo_title: 'ResNet: trenowanie, walidacja i eksport na licencji Apache-2.0'
description: >-
  Używaj ResNet w LibreYOLO do klasyfikacji obrazów. Instaluj, przewiduj,
  dostrajaj, waliduj i eksportuj LibreResNet18/34/50/101.
lead: >-
  ResNet to klasyfikator obrazów zbudowany z bloków resztkowych, czyli połączeń
  pomijających, które pozwalają dodawać do sieci znacznie więcej warstw bez
  spadku dokładności dotykającego głębokie zwykłe stosy konwolucyjne. LibreYOLO
  obsługuje go w jednym zadaniu: klasyfikacji.
keywords:
  - ResNet
  - ResNet50
  - klasyfikacja obrazów
  - uczenie resztkowe
  - głębokie sieci resztkowe
  - klasyfikator ImageNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
        libreyolo export model=LibreResNet50-cls.pt format=tensorrt half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreResNet50-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: e2f46c73716af1b7
---

## Instalacja

ResNet nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zmiana modelu
wymaga zmiany jednego wiersza. Klasyfikator nie zawiera ramek ani masek:
`result.probs` przechowuje predykcję całego obrazu z polami `top1`, `top5`,
`top1conf` i `top5conf`. Argumenty `conf`, `iou` i `max_det` są przyjmowane dla
zgodności API, ale nie mają wpływu na wynik, ponieważ w pojedynczym wektorze
prawdopodobieństwa nie ma czego progować ani tłumić. Więcej informacji o
źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępne są cztery głębokości, wszystkie trenowane i ewaluowane w ten sam
sposób, dlatego wybór jest bezpośrednim kompromisem między liczbą parametrów a
dokładnością. Zadanie jest stałe: każdy rozmiar obsługuje wyłącznie
klasyfikację. Nazwa pliku wag każdego rozmiaru kończy się na `-cls.pt`, a
fabryka odczytuje ten sufiks, aby wybrać rodzinę. Argument `task=` nie jest
potrzebny.

## Trenowanie

Dostrajanie rozpoczyna się od opublikowanego backbone ImageNet, a ostatnia
warstwa klasyfikatora jest automatycznie przebudowywana do liczby klas docelowego
zbioru danych.

<code-tabs name="train" />

Bez zmian konfiguracji trener wykonuje 100 epok z `lr0=1e-3`, optymalizatorem
AdamW, batchem 64 i early stopping (wczesnym zatrzymaniem) po 50 epokach bez
poprawy. Argument `data` przyjmuje katalog główny zbioru danych (`train/` i
`val/`, po jednym folderze na klasę), znaną krótką nazwę, taką jak
`imagenette160`, albo adres URL pliku `.zip`. Ustawienie `lora=True` nie jest tu
obsługiwane i powoduje błąd, ponieważ LoRA w LibreYOLO działa na komponentach
transformerowych z warstwami `nn.Linear`, których ResNet nie ma.

Informacje o zbiorach danych, augmentacji, wielu GPU i loggerach zawiera strona
[trenowania](/docs/train).

## Walidacja

`val()` zwraca słownik kluczy `metrics/`. Dla klasyfikacji są to metryki
accuracy top-1 i top-5 dla podzbioru walidacyjnego.

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
