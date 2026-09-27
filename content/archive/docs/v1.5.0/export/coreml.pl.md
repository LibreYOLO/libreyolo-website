---
title: Core ML
seo_title: Eksport do Core ML z LibreYOLO
description: >-
  Eksport detektora LibreYOLO do pakietu Core ML .mlpackage: kontrakt wejścia
  ImageType, FP16, compute units, wbudowany NMS i cztery obsługiwane rodziny.
lead: >-
  Core ML to format modeli Apple działających na urządzeniu. LibreYOLO trasuje
  detektor za opakowaniem wstępnego przetwarzania właściwym dla każdej rodziny,
  dzięki czemu przekonwertowany graf zawsze przyjmuje kanoniczne wejście obrazu
  RGB, a następnie zapisuje pakiet .mlpackage w formacie ML Program z
  dołączonymi metadanymi modelu.
keywords:
  - eksport yolo do coreml
  - mlpackage
  - coremltools
  - ct.ImageType
  - apple neural engine
  - compute_units
  - coreml nms pipeline
last_verified: 1.5.0
meta:
  - label: Flaga
    value: export(format="coreml")
    mono: true
  - label: Zapisuje
    value: Jeden pakiet .mlpackage (katalog) w formacie ML Program
  - label: Dodatkowo
    value: 'pip install "libreyolo[coreml]"'
    mono: true
  - label: Wczytywanie z powrotem
    value: LibreYOLO("weights/LibreYOLO9t.mlpackage") na macOS
    mono: true
  - label: Kształty
    value: Stałe. Wejściem jest sztywno określony ct.ImageType.
  - label: Precyzja
    value: 'FP32, FP16 (half=True). Brak INT8.'
  - label: Rodziny
    value: 'Tylko detekcja, dla yolox, yolo9, rtdetr i rfdetr'
verification: >-
  Odczytane z libreyolo/export/coreml.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/coreml.py i pyproject.toml na
  gałęzi dev.
snippets:
  install:
    - label: Instalacja
      language: bash
      code: |
        pip install "libreyolo[coreml]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Zapisuje pakiet weights/LibreYOLO9t.mlpackage
        path = model.export(format="coreml")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml
    - label: Argumenty
      language: python
      code: |
        model.export(
            format="coreml",
            imgsz=640,
            batch=1,
            half=False,           # True konwertuje z precyzją obliczeń FLOAT16
            compute_units="all",  # all | cpu_and_gpu | cpu_and_ne | cpu_only
            output_path=None,     # None zapisuje weights/<stem>.mlpackage
        )

        # dynamic jest przyjmowane, ale wejściem jest ct.ImageType o stałym
        # kształcie, a osadzone metadane i tak zapisują dynamic=False.
  nms:
    - label: Wbudowanie warstwy NMS od Apple
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Tylko detekcja w YOLOX i YOLO9, batch 1.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="coreml",
            nms=True,
            conf=0.25,
            iou=0.45,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml --nms \
          --conf 0.25 --iou 0.45
  run:
    - label: 'Przez LibreYOLO, na macOS'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO(
            "weights/LibreYOLO9t.mlpackage",
            compute_units="all",   # lub cpu_and_ne, aby przypiąć Neural Engine
        )
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Samo coremltools
      language: python
      code: >
        import coremltools as ct

        from PIL import Image


        mlmodel = ct.models.MLModel("weights/LibreYOLO9t.mlpackage")

        print(mlmodel.user_defined_metadata["model_family"])

        print(mlmodel.user_defined_metadata["names"])


        # Wejściem jest obraz o nazwie "image" w stałym rozmiarze eksportu.

        image = Image.open(SAMPLE_IMAGE).convert("RGB").resize((640, 640))

        out = mlmodel.predict({"image": image})

        print({name: value.shape for name, value in out.items()})


        # Letterboxing i postprocessing są na tej ścieżce po stronie
        użytkownika.
  support:
    - label: Sprawdzenie jednej rodziny i zadania przed eksportem
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 09c5394e3837eca2
---

## Instalacja

<code-tabs name="install" />

Predykcja wymaga macOS. `LibreYOLO()` odrzuca `.mlpackage` na każdej innej
platformie, podając w komunikacie nazwę bieżącej, a macierz wsparcia zapisuje te
kombinacje jako dostępne, ponieważ zgodność środowiska uruchomieniowego wymaga
runnera z macOS.

## Eksport

<code-tabs name="export" />

Pakiet jest zapisywany w `weights/` pod rdzeniem nazwy checkpointu, z dopisanym
`_fp16`, gdy `half=True`. `.mlpackage` jest katalogiem, więc należy skopiować
całe drzewo.

Każda rodzina jest trasowana za opakowaniem wstępnego przetwarzania, więc
przekonwertowany graf przyjmuje jedno kanoniczne wejście: RGB, `scale=1/255`,
bez biasu, zadeklarowane jako `ct.ImageType`. Opakowanie pochłania własną
konwencję rodziny, czyli BGR w zakresie od 0 do 255 dla YOLOX, średnią i
odchylenie standardowe ImageNet dla RF-DETR oraz przekształcenie identycznościowe
dla YOLO9 i RT-DETR. Dlatego konsument Core ML podaje zwykły obraz, a nie tensor
właściwy dla rodziny.

Konwersja celuje w ML Program z minimalnym celem wdrożenia iOS 15.
`compute_units` jest zapisywane w przekonwertowanym modelu i można je nadpisać
ponownie przy wczytywaniu artefaktu.

Metadane modelu trafiają do `user_defined_metadata` jako łańcuchy znaków i to
stamtąd backend odczytuje rodzinę, zadanie, nazwy klas, rozmiar wejścia i schemat
pozy.

### Wbudowany NMS

<code-tabs name="nms" />

`nms=True` opakowuje model w pipeline Core ML zakończony warstwą
`NonMaximumSuppression` od Apple. Wynik ma dwa wyjścia: `confidence` o kształcie
`N` na liczbę klas oraz `coordinates` o kształcie `N` na 4, jako znormalizowane
`xywh`.

Dotyczy to wyłącznie detekcji w YOLOX i YOLO9 i wymaga batcha 1. Rodziny w stylu
DETR są odrzucane po nazwie, ponieważ predykcja zbioru bierze top-k po zapytaniach
i klasach, bez kroku IoU, i nie może użyć tej warstwy. `max_det` również nie jest
tutaj udostępniane; gdy limit liczby detekcji ma znaczenie, należy użyć
[wbudowanego NMS w ONNX](/docs/export/onnx).

## Uruchamianie artefaktu

<code-tabs name="run" />

`LibreYOLO()` rozpoznaje katalog z sufiksem `.mlpackage` i zwraca ten sam obiekt
`Results` co checkpoint. `compute_units` to jedyny argument, który fabryka
przepuszcza dla tego formatu, i przyjmuje wartości `all`, `cpu_and_gpu`,
`cpu_and_ne` oraz `cpu_only`. Argument `device` jest ignorowany, ponieważ Core ML
kieruje pracę przez compute units.

Drugi fragment kodu to ścieżka samego środowiska uruchomieniowego. Letterboxing,
dekodowanie, NMS i przeskalowanie współrzędnych są tam po stronie użytkownika, a
nazwy klas znajdują się w `user_defined_metadata`.

## Ograniczenia

Cztery rodziny, tylko detekcja: `yolox`, `yolo9`, `rtdetr` i `rfdetr`. Wszystko
inne jest odrzucane w preflight, ponieważ to świadome rodziny opakowanie wstępnego
przetwarzania sprawia, że kontrakt stałego wejścia obrazowego jest poprawny, a
rodzina spoza tej listy przekonwertowałaby się z błędną normalizacją. Komunikat
błędu wskazuje ONNX i TorchScript jako alternatywy.

Kształt wejścia jest sztywno ustalony przez `ct.ImageType`, więc `dynamic=True`
niczego nie zmienia, a metadane zapisują `dynamic=False`. Wyeksportuj drugi
pakiet dla drugiej rozdzielczości.

`half=True` konwertuje z precyzją obliczeń FP16. Ten eksporter nie ma ścieżki
INT8.

Pełną siatkę rodzin i zadań zawiera
[macierz eksportu](/docs/reference/export-matrix). Nowszy format Apple do pracy na
urządzeniu opisuje [Core AI](/docs/export/coreai). Dla jednej kombinacji:

<code-tabs name="support" />
