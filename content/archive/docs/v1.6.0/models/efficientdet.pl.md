---
title: EfficientDet
families:
  - efficientdet
seo_title: 'EfficientDet: detekcja obiektów w LibreYOLO'
description: >-
  Uruchamiaj EfficientDet D0-D4 w LibreYOLO: detektory BiFPN do predykcji,
  walidacji i eksportu do ONNX, TensorRT i OpenVINO na licencji Apache-2.0.
lead: >-
  EfficientDet łączy backbone EfficientNet z powtarzaną dwukierunkową piramidą
  cech (BiFPN) oraz wspólnie skaluje głębokość, szerokość i rozdzielczość w
  pięciu rozmiarach. LibreYOLO udostępnia go jako detektor wyłącznie do
  inferencji.
keywords:
  - EfficientDet
  - BiFPN
  - EfficientNet
  - detekcja obiektów
  - skalowanie złożone
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientDetd0.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientDetd0.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientDetd0.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEfficientDetd0.pt format=onnx
        libreyolo export model=LibreEfficientDetd0.pt format=tensorrt half=True
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Fabryka wybiera ścieżkę na podstawie rozszerzenia pliku, więc
        wyeksportowany artefakt

        # ładuje się jak każdy checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("LibreEfficientDetd0.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 12c61fb0035437ce
---

## Instalacja

EfficientDet nie wymaga żadnego opcjonalnego dodatku. Wszystkie importowane
elementy znajdują się w instalacji podstawowej.

```bash
pip install libreyolo
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

Zwracany obiekt `Results` jest taki sam dla każdej rodziny, więc zmiana
detektora wymaga zmiany jednego wiersza. EfficientDet dekoduje kandydatów
opartych na kotwicach, a następnie uruchamia tłumienie niemaksymalne osobno dla
każdej klasy, dlatego `conf`, `iou` i `max_det` mają tu rzeczywisty wpływ.
Więcej informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępnych jest pięć rozmiarów, od D0 do D4. Każdy kolejny łączy większy
backbone EfficientNet z głębszą i szerszą BiFPN oraz głębszą głowicą predykcji,
dlatego liczba parametrów i koszt obliczeniowy rosną wspólnie zgodnie z regułą
skalowania złożonego opisaną w publikacji.

## Walidacja

`val()` zwraca słownik kluczy `metrics/`, który obejmuje precision, recall,
mAP 50 i mAP 50-95, zmierzone względem dowolnego zbioru danych w formacie użytym
do trenowania.

<code-tabs name="val" />

## Eksport

<export-matrix />

Wyeksportowany artefakt jest ponownie ładowany przez `LibreYOLO()` na podstawie
rozszerzenia pliku, więc plik `.onnx` lub `.engine` zachowuje się jak checkpoint
i zwraca ten sam obiekt `Results`.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box>

Checkpointy D0-D4 LibreYOLO są konwertowane za pośrednictwem projektu
rwightman/efficientdet-pytorch na licencji Apache-2.0, który z kolei tworzy
kopię lustrzaną oficjalnych wag wytrenowanych w TensorFlow z google/automl bez
zmiany wyuczonych tensorów. Nie korzystano z żadnego kodu projektu
zylo117/Yet-Another-EfficientDet-Pytorch na licencji LGPL ani się z nim nie
zapoznawano.

</provenance-box>
