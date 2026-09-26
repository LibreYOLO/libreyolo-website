---
title: FCOS
families:
  - fcos
seo_title: 'FCOS w LibreYOLO: predykcja, walidacja i eksport'
description: >-
  Uruchamiaj FCOS w LibreYOLO do detekcji obiektów bez kotwic (anchor-free).
  Instaluj, przewiduj, waliduj i eksportuj port torchvision ResNet-50/FPN na
  licencji BSD-3-Clause.
lead: >-
  FCOS wykrywa obiekty dla każdego piksela zamiast polegać na zbiorze wstępnie
  zdefiniowanych ramek kotwic, przewidując ramkę i wskaźnik wyśrodkowania w
  każdym położeniu mapy cech. LibreYOLO przenosi implementację torchvision do
  detekcji.
keywords:
  - FCOS
  - detekcja bez kotwic
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

        model = LibreYOLO("LibreFCOSr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCOSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCOSr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="torchscript", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCOSr50.pt format=onnx imgsz=800
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreFCOSr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 60bd7b8dfd903a8c
---

## Instalacja

FCOS nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane elementy
znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zmiana
detektora wymaga zmiany jednego wiersza. Wywołanie modelu bez argumentów progów
stosuje własne opublikowane wartości domyślne FCOS: `conf=0.2`, `iou=0.6` i
`max_det=100`. Przekazanie dowolnego z tych trzech argumentów zastępuje jego
wartość. FCOS zachowuje końcowy etap NMS dla predykcji poszczególnych pikseli.
Więcej informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępny jest jeden rozmiar: ResNet-50 z piramidą cech, jedyny wariant
rozpoznawany przez tę rodzinę.

## Walidacja

`val()` zwraca słownik kluczy `metrics/`, który obejmuje precision, recall,
mAP 50 i mAP 50-95, zmierzone względem dowolnego zbioru danych w formacie użytym
do trenowania.

<code-tabs name="val" />

## Eksport

<export-matrix />

FCOS eksportuje do ONNX, TorchScript i OpenVINO. FCOS zachowuje proporcje źródła
przed uruchomieniem grafu, dlatego LibreYOLO wymusza `dynamic=True` dla ścieżek
ONNX i OpenVINO niezależnie od przekazanej wartości, aby graf działał dla
dopełnionych kształtów wejściowych. Wyeksportowany plik `.onnx` jest ponownie
ładowany przez `LibreYOLO()` na podstawie rozszerzenia i zwraca ten sam obiekt
`Results`.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
