---
title: MobileNetV4
families:
  - mobilenetv4
seo_title: 'MobileNetV4: trenowanie, walidacja i eksport na licencji Apache-2.0'
description: >-
  Używaj MobileNetV4 w LibreYOLO do klasyfikacji obrazów. Instaluj, przewiduj,
  dostrajaj, waliduj i eksportuj LibreMobileNetV4 small/medium/large.
lead: >-
  MobileNetV4 to klasyfikator obrazów zbudowany dla sprzętu mobilnego i
  brzegowego, wykorzystujący blok Universal Inverted Bottleneck do połączenia
  kilku wcześniejszych projektów bloków mobilnych w jedną przeszukiwalną
  strukturę. LibreYOLO obsługuje go w jednym zadaniu: klasyfikacji.
keywords:
  - MobileNetV4
  - MobileNetV4 conv
  - klasyfikacja obrazów
  - inferencja mobilna
  - klasyfikator na urządzenie brzegowe
  - klasyfikator ImageNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMobileNetV4s-cls.pt source=cat.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreMobileNetV4s-cls.pt data=imagenette160
        epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreMobileNetV4s-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMobileNetV4s-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMobileNetV4s-cls.pt format=onnx

        libreyolo export model=LibreMobileNetV4s-cls.pt format=tensorrt
        half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreMobileNetV4s-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 4a9a1b392ffb136d
---

## Instalacja

MobileNetV4 nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane
elementy znajdują się w instalacji podstawowej.

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

Dostępne są trzy rozmiary, small/medium/large, wszystkie oparte wyłącznie na
splotach. Ta rodzina nie zawiera wariantów hybrydowych dodających uwagę Mobile
MQA. Wybór rozmiaru jest bezpośrednim kompromisem między liczbą parametrów a
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
transformerowych z warstwami `nn.Linear`, których nie ma w blokach UIB tej
rodziny.

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
