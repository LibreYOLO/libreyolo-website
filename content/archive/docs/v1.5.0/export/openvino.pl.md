---
title: OpenVINO
seo_title: Eksport do OpenVINO IR z LibreYOLO
description: >-
  Konwersja modelu LibreYOLO do OpenVINO IR: para model.xml i model.bin,
  kompresja wag do FP16, INT8 przez NNCF oraz inferencja na CPU, GPU lub NPU.
lead: >-
  OpenVINO IR to format środowiska uruchomieniowego Intela, czyli graf model.xml
  obok blobu wag model.bin. LibreYOLO eksportuje pośredni model ONNX, konwertuje
  go przez ov.convert_model i zapisuje plik metadata.yaml w tym samym katalogu.
keywords:
  - eksport yolo do openvino
  - openvino ir
  - model.xml model.bin
  - ov.convert_model
  - kwantyzacja int8 nncf
  - openvino npu
  - compress_to_fp16
last_verified: 1.5.0
meta:
  - label: Flaga
    value: export(format="openvino")
    mono: true
  - label: Zapisuje
    value: 'Katalog z plikami model.xml, model.bin i metadata.yaml'
  - label: Dodatkowo
    value: 'pip install "libreyolo[onnx,openvino]"'
    mono: true
  - label: Wczytywanie z powrotem
    value: LibreYOLO("weights/LibreYOLO9t_openvino")
    mono: true
  - label: Kształty
    value: 'Zgodne z pośrednim modelem ONNX: dynamiczny batch przy dynamic=True'
  - label: Precyzja
    value: >-
      FP32, kompresja wag do FP16 (half=True), INT8 przez NNCF (int8=True wraz z
      data=)
verification: >-
  Odczytane z libreyolo/export/openvino.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/openvino.py i pyproject.toml
  na gałęzi dev.
snippets:
  install:
    - label: Instalacja
      language: bash
      code: >
        # IR jest konwertowany z pośredniego modelu ONNX, więc potrzebne są oba
        dodatki.

        pip install "libreyolo[onnx,openvino]"
    - label: INT8 wymaga dodatkowo NNCF
      language: bash
      code: |
        pip install nncf
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Zapisuje katalog weights/LibreYOLO9t_openvino
        path = model.export(format="openvino")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format openvino
    - label: Argumenty
      language: python
      code: |
        model.export(
            format="openvino",
            imgsz=640,
            batch=1,
            dynamic=False,    # True zachowuje dynamiczną oś batcha w całym IR
            half=False,       # True zapisuje wagi w FP16
            int8=False,       # True uruchamia kwantyzację potreningową NNCF
            data=None,        # wymagane, gdy int8=True
            output_path=None, # None zapisuje weights/<stem>_openvino
        )
  int8:
    - label: INT8 z danymi kalibracyjnymi
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="openvino",
            int8=True,
            data="coco128.yaml",   # wymagane: ten format nie ma wartości domyślnej
            fraction=1.0,
        )
  run:
    - label: Przez LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_openvino")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Wybór urządzenia
      language: python
      code: >
        from libreyolo import LibreYOLO


        # "auto" i "cpu" mapują się na CPU, "gpu" i "cuda" na GPU,

        # każda inna wartość jest przekazywana wielkimi literami, na przykład
        "npu" -> NPU.

        model = LibreYOLO("weights/LibreYOLO9t_openvino", device="gpu")
    - label: Samo OpenVINO
      language: python
      code: >
        import numpy as np

        import openvino as ov

        import yaml


        core = ov.Core()

        print(core.available_devices)


        compiled = core.compile_model("weights/LibreYOLO9t_openvino/model.xml",
        "CPU")

        outputs = compiled(np.zeros((1, 3, 640, 640), dtype=np.float32))

        print([tensor.shape for tensor in outputs.values()])


        # Nazwy klas, zadanie i rozmiar wejścia znajdują się w metadata.yaml
        obok IR.

        meta =
        yaml.safe_load(open("weights/LibreYOLO9t_openvino/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Wstępne i końcowe przetwarzanie są na tej ścieżce po stronie
        użytkownika.
  support:
    - label: Sprawdzenie jednej rodziny i zadania przed eksportem
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 519816615e3aca3c
---

## Instalacja

<code-tabs name="install" />

Konwersja przechodzi przez pośredni model ONNX, więc dodatek `onnx` jest częścią
wymagań, a nie opcjonalnym uzupełnieniem. NNCF instaluje się osobno i jest
potrzebny tylko przy `int8=True`.

## Eksport

<code-tabs name="export" />

Artefaktem jest katalog, a nie plik. `weights/LibreYOLO9t_openvino` zawiera
`model.xml`, `model.bin` i `metadata.yaml`, a przy `half=True` przed sufiksem
wstawiane jest `_fp16`. Przenoś lub kopiuj cały katalog; te trzy pliki to jeden
artefakt.

`half=True` ustawia `compress_to_fp16` przy zapisie. To kompresja wag w IR, a nie
zmiana precyzji inferencji, którą urządzenie wybiera w czasie działania.

### INT8

<code-tabs name="int8" />

`int8=True` uruchamia kwantyzację potreningową NNCF na kalibracyjnym loaderze
danych LibreYOLO z presetem mixed, a `data` jest obowiązkowe: ten format nie ma
awaryjnego wariantu z ośmioma obrazami. Brak NNCF powoduje zgłoszenie
`ImportError` z nazwą polecenia instalacji.

## Uruchamianie artefaktu

<code-tabs name="run" />

`LibreYOLO()` rozpoznaje każdy katalog zawierający `model.xml` i zwraca ten sam
obiekt `Results` co checkpoint, odczytując nazwy klas, zadanie, rozmiar wejścia i
schemat pozy z `metadata.yaml`.

Ciąg znaków urządzenia jest mapowany, a nie przekazywany wprost. `auto` i `cpu`
kompilują się na CPU, `gpu` i `cuda` kompilują się na GPU, a każda inna wartość
jest zamieniana na wielkie litery i przekazywana do OpenVINO, i w ten sposób
wskazuje się NPU jako cel.

Trzeci fragment kodu jest przeznaczony dla osób bez zainstalowanego LibreYOLO.
Wstępne przetwarzanie, dekodowanie, NMS i przeskalowanie współrzędnych są tam po
stronie użytkownika, a nazwy klas istnieją wyłącznie w `metadata.yaml`.

## Ograniczenia

IR bez pliku `metadata.yaml` nadal się wczytuje, ale backend przyjmuje wtedy
awaryjnie 80 klas i zadanie detekcji, co jest błędne dla wszystkiego innego.
Zachowaj katalog w nienaruszonym stanie.

Zablokowane przed trasowaniem: segmentacja YOLO9, segmentacja RTMDet-Ins,
detekcja SSD, Faster R-CNN i RetinaNet oraz matting BiRefNet lub FeyNobg, gdzie
OpenVINO 2026.2 nie potrafi odwzorować na operacje niższego poziomu standardowej
operacji ONNX `DeformConv-19` we współdzielonym dekoderze matte.

Tam, gdzie kombinacja nie jest ani zwalidowana, ani zablokowana, ścieżka
konwertera jest dostępna, a projekt nie odnotował dla niej zgodności działania w
środowisku uruchomieniowym OpenVINO. Kilka kombinacji zwalidowano z dołączonym
jawnym kontekstem, na przykład segmentację semantyczną DeepLabV3 przy stałym
wejściu 520 na 520 w OpenVINO 2026.2 z domyślną precyzją inferencji na CPU oraz
L2CS gaze przy stałym wycinku twarzy 448 na 448. `libreyolo formats` wypisuje ten
kontekst dla każdej kombinacji.

Pełną siatkę rodzin i zadań zawiera
[macierz eksportu](/docs/reference/export-matrix). Dla jednej kombinacji:

<code-tabs name="support" />
