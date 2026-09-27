---
title: RetinaNet
families:
  - retinanet
seo_title: 'RetinaNet w LibreYOLO: predykcja, walidacja i eksport'
description: >-
  Uruchamiaj RetinaNet w LibreYOLO do jednostopniowej detekcji obiektów z
  ogniskową funkcją straty. Instaluj, przewiduj, waliduj i eksportuj port
  torchvision na licencji BSD-3-Clause.
lead: >-
  RetinaNet to jednostopniowy detektor trenowany z ogniskową funkcją straty,
  która zmniejsza wagę łatwych przykładów negatywnych, dzięki czemu gęsta siatka
  kotwic zachowuje dokładność bez osobnego etapu propozycji. LibreYOLO przenosi
  implementację torchvision do detekcji.
keywords:
  - RetinaNet
  - ogniskowa funkcja straty
  - detekcja obiektów
  - detektor jednostopniowy
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRetinaNetr50v2.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRetinaNetr50v2.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRetinaNetr50v2.pt format=onnx imgsz=800
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreRetinaNetr50v2.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 1cc7ceb6de290bdb
---

## Instalacja

RetinaNet nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane
elementy znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zmiana
detektora wymaga zmiany jednego wiersza. Argumenty `conf` i `iou` ustawiają progi
pewności i NMS. RetinaNet zachowuje źródłowy etap NMS dla gęstej siatki kotwic.
Więcej informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępne są dwa rozmiary, oba oparte na ResNet-50 z piramidą cech. `r50` ma
oryginalną głowicę, a `r50v2` zastępuje ją głowicą GroupNorm i szerszym blokiem
P6 zasilanym z ostatniego etapu backbone zamiast z wyjścia FPN.

## Walidacja

`val()` zwraca słownik kluczy `metrics/`, który obejmuje precision, recall,
mAP 50 i mAP 50-95, zmierzone względem dowolnego zbioru danych w formacie użytym
do trenowania.

<code-tabs name="val" />

## Eksport

<export-matrix />

RetinaNet eksportuje wyłącznie do ONNX z rozmiarem batcha 1. RetinaNet zmienia
rozmiar wejścia przy zachowaniu proporcji, a jego wynikowy rozmiar jest zmienny,
dlatego LibreYOLO wymusza `dynamic=True` niezależnie od przekazanej wartości,
aby graf działał dla źródeł o różnych kształtach. Wyeksportowany plik `.onnx`
jest ponownie ładowany przez `LibreYOLO()` na podstawie rozszerzenia i zwraca ten
sam obiekt `Results`.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>
