---
title: HRNet
families:
  - hrnet
seo_title: 'HRNet: odgórna estymacja pozy w LibreYOLO'
description: >-
  Używaj HRNet w LibreYOLO do odgórnej estymacji pozy COCO-17. Instaluj,
  przewiduj, waliduj i eksportuj checkpointy W32 i W48 na licencji MIT.
lead: >-
  HRNet to sieć konwolucyjna, która utrzymuje strumień cech o wysokiej
  rozdzielczości przez wielokrotne scalanie wieloskalowe zamiast odzyskiwać
  rozdzielczość po jej zmniejszeniu. LibreYOLO opakowuje oficjalny odgórny
  wariant estymacji pozy do inferencji i walidacji.
keywords:
  - HRNet
  - estymacja pozy człowieka
  - odgórna estymacja pozy
  - punkty kluczowe COCO-17
  - sieć wysokiej rozdzielczości
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Bez podanego źródła osób HRNet automatycznie łączy się z lekkim
        # detektorem LibreYOLO9t i jednorazowo zapisuje ten wybór w logu.
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreHRNetw32-pose.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Źródło osób
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreHRNetw32-pose.pt")


        # Całkowite pominięcie detekcji: potraktowanie całego obrazu jako jednej
        osoby.

        result = model(SAMPLE_IMAGE, cropped=True)


        # Albo przekazanie HRNet ramek z wcześniej uruchomionego detektora.

        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])


        # Albo połączenie z konkretnym detektorem LibreYOLO zamiast domyślnego

        # LibreYOLO9t.

        result = model(SAMPLE_IMAGE, person_detector="rfdetr")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreHRNetw32-pose.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreHRNetw32-pose.pt format=onnx
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Wyeksportowany graf zawiera wyłącznie głowicę map cieplnych o stałym
        obszarze.

        # Przyjmuje batch już przyciętych i znormalizowanych obrazów osób oraz
        zwraca

        # surowe mapy cieplne. Detekcja osób, geometria przycięcia, dekodowanie
        map cieplnych

        # i tłumienie OKS nie są częścią grafu. Uruchomienie poza LibreYOLO
        wymaga

        # samodzielnej implementacji tego etapu dekodowania.

        session = ort.InferenceSession("LibreHRNetw32-pose.onnx")

        name = session.get_inputs()[0].name

        heatmaps = session.run(
            None, {name: np.zeros((1, 3, 256, 192), dtype=np.float32)}
        )[0]
source_hash: 5a5540fd54ee6f23
---

## Instalacja

HRNet nie wymaga żadnego dodatku poza pakietem podstawowym.

```bash
pip install libreyolo
```

Jego domyślny detektor osób, lekki checkpoint LibreYOLO9t, jest pobierany
automatycznie przy pierwszym połączeniu z HRNet.

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej.

<code-tabs name="predict" />

HRNet jest odgórnym estymatorem pozy. Wymaga ramki osoby przed uruchomieniem
głowicy pozy, dlatego każde wywołanie ustala takie ramki. Bez dodatkowej
konfiguracji przy pierwszym użyciu łączy się z detektorem LibreYOLO9t i zapisuje
ten wybór w logu. `cropped=True` pomija detekcję i traktuje cały obraz jako
jedną osobę. `person_boxes` przyjmuje ramki z wcześniej uruchomionego detektora.
`person_detector` przyjmuje `"auto"`, `"rfdetr"`, dowolny model detekcji
LibreYOLO albo zwykły obiekt wywoływalny. `flip_test=True` uruchamia model także
na poziomo odbitym przycięciu i uśrednia obie mapy cieplne, co stanowi własną
augmentację HRNet podczas testowania. Ogólne `augment=True` nie jest tutaj
zdefiniowane. Źródła z wieloma obrazami działają sekwencyjnie. Detektor HRNet i
zmienna liczba osób na obraz nie obsługują predykcji w złożonym batchu. Więcej
informacji o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępne są dwa rozmiary, `w32` i `w48`. Oba przewidują standardowy zbiór
punktów kluczowych COCO-17 ze stałej rozdzielczości przycięcia osoby. `w48` jest
szerszym backbone.

Źródłowy katalog modeli podaje dokładność pozy każdego rozmiaru z jego własnym
detektorem osób, konfiguracją testu odbicia i oficjalnym protokołem ewaluacji
COCO. Domyślne połączenie LibreYOLO używa innego detektora, dlatego walidacja
tutaj mierzy tę kombinację, a nie kombinację źródłową. Odtworzenie wyników
źródłowych wymaga tych samych ramek osób, wyników detektora i ustawienia odbicia
co w oryginalnej ewaluacji.

## Walidacja

`val()` uruchamia OKS-AP punktów kluczowych w stylu COCO i przyjmuje plik
`data.yaml` pozy YOLO albo plik JSON punktów kluczowych COCO z katalogiem
obrazów. Backend metryk domyślnie używa faster-coco-eval, a gdy nie jest on
zainstalowany, automatycznie przechodzi na `pycocotools`.
`faster_coco_eval=False` wymusza ścieżkę `pycocotools`.

<code-tabs name="val" />

Walidacja wewnętrznie korzysta z własnego `predict()` HRNet, dlatego używa
detektora osób, z którym zbudowano lub wywołano model. Aby źródło pozostało
stałe między przebiegami, należy utworzyć model z jawnym `person_detector=`,
zamiast pozwalać każdemu wywołaniu ponownie ustalać wartość domyślną.

## Eksport

<export-matrix />

Kontrakt eksportu HRNet obejmuje tylko ONNX, TorchScript, OpenVINO i TensorRT.
Każdy inny format zgłasza błąd przed rozpoczęciem śledzenia. Każdy eksport jest
wyłącznie głowicą map cieplnych o stałym obszarze, batchu 1 i precyzji FP32,
przyjmującą przycięcie osoby i zwracającą surowe mapy cieplne. Geometria
przycięcia przed nią oraz dekodowanie map cieplnych, odtwarzanie odbicia i
tłumienie OKS za nią pozostają w Pythonie, dlatego pełny pipeline od obrazu do
punktów kluczowych nadal wymaga LibreYOLO po drugiej stronie.

<code-tabs name="export" />

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />
