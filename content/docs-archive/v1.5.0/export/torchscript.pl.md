---
title: TorchScript
seo_title: Eksport do TorchScript z LibreYOLO
description: >-
  Eksport modelu LibreYOLO do TorchScript: śledzone archiwum .torchscript z
  metadanymi LibreYOLO w środku, wczytywalne z Pythona lub z libtorch.
lead: >-
  TorchScript to własny format serializowanego grafu w PyTorch. LibreYOLO śledzi
  model za pomocą torch.jit.trace i zapisuje wynik razem z dodatkowym plikiem
  libreyolo_metadata.json, dzięki czemu archiwum niesie rodzinę, zadanie, nazwy
  klas i rozmiar wejścia.
keywords:
  - eksport yolo do torchscript
  - torch.jit.trace
  - torch.jit.load
  - wdrożenie libtorch
  - metadane torchscript
  - extra_files
last_verified: 1.5.0
meta:
  - label: Flaga
    value: export(format="torchscript")
    mono: true
  - label: Zapisuje
    value: Jedno archiwum .torchscript z dodatkowym plikiem libreyolo_metadata.json
  - label: Extra
    value: Brak. TorchScript jest dostarczany razem z PyTorch.
  - label: Ponowne wczytanie
    value: LibreYOLO("weights/LibreYOLO9t.torchscript")
    mono: true
  - label: Kształty
    value: Stałe. Graf jest śledzony przy jednym kształcie wejścia.
  - label: Precyzja
    value: 'FP32, FP16 (half=True). Bez INT8.'
verification: >-
  Odczytane z libreyolo/export/torchscript.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py i libreyolo/backends/torchscript.py na gałęzi dev.
snippets:
  install:
    - label: Instalacja
      language: bash
      code: |
        pip install libreyolo
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Zapisuje weights/LibreYOLO9t.torchscript
        path = model.export(format="torchscript")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format torchscript
    - label: Argumenty
      language: python
      code: |
        model.export(
            format="torchscript",
            imgsz=640,        # int lub (wysokość, szerokość)
            batch=1,
            half=False,       # wagi i aktywacje w FP16
            device=None,      # None śledzi na CPU dla tego formatu
            output_path=None, # None zapisuje weights/<stem>.torchscript
        )

        # dynamic jest akceptowane, ale archiwum zawsze zawiera ślad o stałym
        # kształcie, a osadzone metadane i tak zapisują dynamic=False.
  run:
    - label: Przez LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.torchscript")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Czysty PyTorch
      language: python
      code: >
        import json


        import torch


        extra_files = {"libreyolo_metadata.json": ""}

        module = torch.jit.load(
            "weights/LibreYOLO9t.torchscript",
            map_location="cpu",
            _extra_files=extra_files,
        )

        module.eval()


        metadata = json.loads(extra_files["libreyolo_metadata.json"])

        print(metadata["model_family"], metadata["task"], metadata["imgsz"])


        # Na tej ścieżce preprocessing i postprocessing są po stronie
        użytkownika.

        with torch.no_grad():
            out = module(torch.zeros(1, 3, 640, 640))
        print(out.shape if torch.is_tensor(out) else [t.shape for t in out])
  support:
    - label: Sprawdzenie jednej rodziny i zadania przed eksportem
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 286a082969ccd604
---

## Instalacja

<code-tabs name="install" />

TorchScript nie wymaga niczego poza podstawową instalacją, ponieważ `torch.jit`
jest dostarczany razem z PyTorch. To jedyny cel eksportu bez opcjonalnej
zależności i bez zewnętrznego konwertera, co czyni go przydatnym pierwszym
sprawdzeniem, gdy dłuższy toolchain zawodzi.

## Eksport

<code-tabs name="export" />

Śledzenie odbywa się na CPU, o ile nie wskazano urządzenia, a archiwum jest
zapisywane w `weights/` pod nazwą rdzenia checkpointu, gdy pominięto
`output_path`.

Kontrola ponownego śledzenia, którą `torch.jit.trace` normalnie wykonuje, jest
wyłączona. Kilka wrapperów eksportu zapisuje w pamięci podręcznej zależne od
kształtu kotwice podczas pierwszego przebiegu w przód, więc drugie śledzenie
obserwuje inną ścieżkę w Pythonie, mimo że zarejestrowany graf o stałym
kształcie jest poprawny. Testy parzystości weryfikują zamiast tego bezpośrednio
zapisany moduł.

Metadane nie znajdują się w pliku sidecar. `torch.jit.save` zapisuje
`libreyolo_metadata.json` wewnątrz archiwum, a `torch.jit.load` zwraca je przez
`_extra_files`.

## Uruchomienie artefaktu

<code-tabs name="run" />

`LibreYOLO()` rozpoznaje ścieżkę po sufiksie `.torchscript` i zwraca ten sam
obiekt `Results` co checkpoint, z którego powstało archiwum. Przy
`device="auto"` moduł jest mapowany na CUDA, jeśli jest dostępna, następnie na
MPS, a na końcu na CPU.

Drugi snippet to ścieżka dla czytelnika, który nie ma zainstalowanego
LibreYOLO, oraz dla wdrożenia w C++ przez libtorch, gdzie to samo archiwum
wczytuje się za pomocą `torch::jit::load`. Preprocessing, dekodowanie, NMS i
przeskalowanie współrzędnych są tam po stronie użytkownika. Dodatkowy plik
metadanych nadal daje się odczytać i jest jedynym miejscem, w którym istnieją
nazwy klas.

## Ograniczenia

Graf jest śladem przy jednym kształcie wejścia. `dynamic=True` jest akceptowane
dla symetrii interfejsu, ale nic nie zmienia, a osadzone metadane raportują
`dynamic=False`, żeby backend nigdy nie zakładał osi, której nie może użyć. Dla
drugiej rozdzielczości należy wyeksportować drugie archiwum.

`half=True` rzutuje model i wejście śledzenia na FP16. Nie ma ścieżki INT8:
`int8=True` zgłasza `NotImplementedError` podczas walidacji.

Prostokątny `imgsz` działa dla rodzin YOLO9, HRNet, NAFNet i Real-ESRGAN, a jest
odrzucany dla rodzin ze stałym kwadratowym kontraktem.

Pięć kombinacji jest odrzucanych przed śledzeniem. Segmentacja YOLO9, ponieważ
YOLO9 w LibreYOLO służy tylko do detekcji. Segmentacja RTMDet-Ins, której
dekodowanie masek z dynamicznym jądrem nie ma kontraktu dla eksportowanego
środowiska uruchomieniowego. Detekcja SSD, Faster R-CNN i RetinaNet, których
grafy o zmiennej długości lub z dynamicznymi kotwicami mają dowody parzystości
wyłącznie w ramach kontraktu ONNX Runtime.

Pełna siatka rodzin i zadań znajduje się w
[macierzy eksportu](/docs/reference/export-matrix). Dla jednej kombinacji:

<code-tabs name="support" />
