---
title: TensorRT
seo_title: Eksport do TensorRT z LibreYOLO
description: >-
  Budowanie silnika TensorRT z modelu LibreYOLO: pośredni plik ONNX, buildy FP16
  i INT8, profile dynamicznego batcha oraz granice przenośności silnika.
lead: >-
  TensorRT kompiluje graf do silnika dostrojonego pod jedno GPU. LibreYOLO
  najpierw eksportuje pośredni plik ONNX, parsuje go parserem ONNX z TensorRT,
  buduje silnik i zapisuje obok metadane modelu w pliku sidecar JSON.
keywords:
  - eksport yolo do tensorrt
  - silnik tensorrt
  - trt fp16
  - kalibracja int8 tensorrt
  - profil optymalizacji tensorrt
  - dynamiczny batch tensorrt
  - zgodność sprzętowa tensorrt
last_verified: 1.5.0
meta:
  - label: Flaga
    value: export(format="tensorrt")
    mono: true
  - label: Zapisuje
    value: Jeden plik .engine oraz plik sidecar .engine.json z metadanymi
  - label: Dodatkowo
    value: 'pip install "libreyolo[onnx,tensorrt]"'
    mono: true
  - label: Wczytywanie z powrotem
    value: LibreYOLO("weights/LibreYOLO9t.engine")
    mono: true
  - label: Kształty
    value: Domyślnie stałe; dynamic=True dodaje profil optymalizacji dla osi batcha
  - label: Precyzja
    value: 'FP32, FP16 (half=True), INT8 (int8=True z data=)'
  - label: Wymagania
    value: >-
      GPU NVIDIA podczas budowania i podczas uruchamiania. Silniki nie przenoszą
      się między architekturami GPU.
verification: >-
  Odczytane z libreyolo/export/tensorrt.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tensorrt.py i pyproject.toml
  na gałęzi dev.
snippets:
  install:
    - label: Instalacja
      language: bash
      code: >
        # Silnik jest budowany z pośredniego pliku ONNX, więc potrzebne są oba
        dodatki.

        pip install "libreyolo[onnx,tensorrt]"
    - label: Sprawdzenie łańcucha narzędzi przed budowaniem
      language: bash
      code: >
        python -c "import tensorrt, torch; print(tensorrt.__version__,
        torch.cuda.is_available())"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Zapisuje weights/LibreYOLO9t_fp16.engine i
        weights/LibreYOLO9t_fp16.engine.json

        path = model.export(format="tensorrt", half=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tensorrt --half
    - label: Argumenty
      language: python
      code: |
        model.export(
            format="tensorrt",
            imgsz=640,
            batch=1,
            half=False,
            int8=False,
            data=None,                      # wymagane, gdy int8=True
            dynamic=False,
            workspace=4.0,                  # pamięć robocza budowania w GiB
            min_batch=1,                    # granice profilu dynamicznego
            opt_batch=1,
            max_batch=8,
            hardware_compatibility="none",  # albo "ampere_plus"
            gpu_device=0,                   # urządzenie budowania na hoście z wieloma GPU
            verbose=False,
        )
  dynamic:
    - label: Silnik z dynamicznym batchem
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Pośredni plik ONNX musi mieć dynamiczną oś batcha, aby profil
        # miał się do czego przypiąć.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            dynamic=True,
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            half=True,
        )
  int8:
    - label: INT8 z danymi kalibracyjnymi
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            int8=True,
            data="coco128.yaml",   # wymagane: ten format nie ma wartości domyślnej
            fraction=1.0,
        )
  run:
    - label: Przez LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_fp16.engine")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Samo TensorRT
      language: python
      code: >
        import json


        import tensorrt as trt


        path = "weights/LibreYOLO9t_fp16.engine"

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        with open(path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(engine.get_tensor_mode(name), name, engine.get_tensor_shape(name))

        # Nazwy klas, zadanie i rozmiar wejścia są w pliku sidecar, nie w
        silniku.

        # Alokacja buforów, wstępne i końcowe przetwarzanie są tu po stronie
        użytkownika.

        print(json.load(open(path + ".json"))["names"])
  support:
    - label: Sprawdzenie jednej rodziny i zadania przed budowaniem
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cb90fc98ab735233
---

## Instalacja

Zarówno budowanie, jak i uruchamianie wymagają GPU NVIDIA z działającym stosem
CUDA. Ten format nie ma awaryjnej ścieżki na CPU.

<code-tabs name="install" />

Dodatek `tensorrt` przypina `tensorrt-cu12` i `pycuda`, a marker pomija oba na
macOS. Na urządzeniu Jetson nie należy używać tego dodatku: przypina on build dla
CUDA 12 na platformie z CUDA 13. Zamiast tego należy użyć TensorRT instalowanego
przez JetPack, jak opisano na stronie [NVIDIA Jetson](/docs/export/jetson).

## Eksport

<code-tabs name="export" />

Eksport przebiega w dwóch krokach. Pierwszy zapisuje pośredni plik ONNX w ścieżce
tymczasowej, drugi parsuje go i buduje silnik, a plik pośredni jest potem
usuwany. `workspace` to pamięć robocza używana podczas budowania, podana w GiB;
większa wartość pozwala builderowi wypróbować więcej kerneli i nie wpływa na
pamięć podczas inferencji.

Plik sidecar z metadanymi jest zapisywany obok silnika jako `<engine>.json` i
odnotowuje precyzję, którą build faktycznie zrealizował. Gdy GPU nie ma szybkiego
FP16 ani szybkiego INT8, builder ostrzega i schodzi niżej, a plik sidecar podaje
precyzję, która wyszła, a nie tę, o którą poproszono.

Przy FP16 backbone ViT w grafie jest wykrywany, a jego warstwy
zmiennoprzecinkowe zostają przypięte do FP32. Bloki backbone w stylu DINOv2
przepełniają się w FP16 i dają NaN, więc build ustawia
`OBEY_PRECISION_CONSTRAINTS` i raportuje `FP16 (FP32 ViT backbone)`. Dla bloków
backbone typu CNN ten krok nic nie zmienia.

### Dynamiczny batch

<code-tabs name="dynamic" />

`dynamic=True` dodaje jeden profil optymalizacji obejmujący zakres od
`min_batch` do `max_batch`, zoptymalizowany dla `opt_batch`, i zapisuje te trzy
wartości w pliku sidecar. Profil jest dodawany tylko wtedy, gdy pośredni plik
ONNX rzeczywiście ma dynamiczny wymiar batcha; w przeciwnym razie build zapisuje
w logu, że używa optymalizacji statycznej, i kontynuuje.

### INT8

<code-tabs name="int8" />

INT8 korzysta z kalibratora entropijnego z TensorRT nałożonego na loader
kalibracyjny LibreYOLO, a `data` jest obowiązkowe: ten format nie ma awaryjnej
ścieżki z ośmioma obrazami. Kalibracja wymaga `cuda-python` lub `pycuda` do
bufora na urządzeniu. Kluczem pamięci podręcznej kalibracji jest hash bajtów
ONNX, więc skale z jednego modelu nigdy nie są ponownie używane dla innego,
który akurat zapisuje do tej samej ścieżki wyjściowej.

`half=True` i `int8=True` naraz powodują ostrzeżenie i build w INT8, który
zachowuje awaryjną ścieżkę FP16 dla warstw, których TensorRT nie potrafi
skwantyzować.

## Uruchamianie artefaktu

<code-tabs name="run" />

`LibreYOLO()` rozpoznaje sufiks `.engine`, odczytuje z pliku sidecar nazwy klas,
zadanie i schemat pozy oraz zwraca ten sam obiekt `Results` co checkpoint. Gdy
nie ma urządzenia z CUDA, zgłasza błąd natychmiast.

Drugi fragment kodu to ścieżka samego środowiska uruchomieniowego. Alokacja
buforów po stronie hosta i urządzenia, wstępne przetwarzanie, dekodowanie, NMS i
przeskalowanie współrzędnych są tam po stronie użytkownika, a sam silnik nie
zawiera nazw klas, więc plik sidecar musi mu towarzyszyć.

## Ograniczenia

Zserializowany silnik jest związany z architekturą GPU, stosem sterowników i
wersją TensorRT, która go zbudowała. Silnik zbudowany na stacji roboczej nie
wczyta się na innej architekturze i dlatego krok budowania uruchamia się na
maszynie docelowej. `hardware_compatibility="ampere_plus"` oddaje część
wydajności w zamian za przenośność w obrębie Ampere i nowszych. Wartość
`"same_compute_capability"` mapuje się na `NONE` i powoduje ostrzeżenie: silnik
jest zoptymalizowany wyłącznie pod bieżące GPU, a eksport mówi to wprost, zamiast
deklarować przenośność, której nie zastosował.

Profilowana jest wyłącznie oś batcha. Build z dynamicznymi wymiarami
przestrzennymi nie należy do tego kontraktu i dlatego FCOS jest zablokowany:
potrzebuje dynamicznej, dopełnionej wysokości i szerokości, aby zachować swoje
przekształcenie proporcji 800 na 1333.

Zablokowane przed trasowaniem: segmentacja w YOLO9, segmentacja w RTMDet-Ins,
detekcja w SSD, Faster R-CNN i RetinaNet oraz matting w BiRefNet lub FeyNobg,
gdzie TensorRT 10.16 dochodzi do wspólnego węzła ONNX `DeformConv` i nie potrafi
go sparsować, ponieważ `ModulatedDeformConv2d` nie ma w rejestrze pluginów.

Gdy kombinacja nie jest ani zwalidowana, ani zablokowana, ścieżka konwertera jest
dostępna, a projekt nie odnotował dla niej zgodności środowiska uruchomieniowego
TensorRT. To stwierdzenie o dowodach, a nie o tym, czy build się powiedzie.

Pełną siatkę rodzin i zadań zawiera
[macierz eksportu](/docs/reference/export-matrix). Dla jednej kombinacji:

<code-tabs name="support" />
