---
title: MNN
seo_title: Eksport do MNN z LibreYOLO
description: >-
  Eksport detektora LibreYOLO do MNN przez ONNX i mnnconvert: stały kształt
  NCHW, FP32 na CPU oraz towarzyszący plik metadanych wymagany przez kontrakt
  środowiska uruchomieniowego.
lead: >-
  MNN to lekki silnik inferencji firmy Alibaba. LibreYOLO eksportuje statyczny
  graf ONNX, konwertuje go narzędziem mnnconvert dostarczanym z pakietem MNN i
  zapisuje towarzyszący plik JSON z nazwami wejść i wyjść, stałym kształtem
  wejścia oraz nazwami klas.
keywords:
  - eksport yolo do mnn
  - mnnconvert
  - mnn inferencja
  - inferencja na urządzeniach mobilnych
  - stały kształt nchw
last_verified: 1.5.0
meta:
  - label: Flaga
    value: export(format="mnn")
    mono: true
  - label: Zapisuje
    value: Jeden plik .mnn oraz towarzyszący plik metadanych .mnn.json
  - label: Dodatkowo
    value: 'pip install "libreyolo[mnn]"'
    mono: true
  - label: Wczytywanie z powrotem
    value: LibreYOLO("weights/LibreYOLO9t.mnn")
    mono: true
  - label: Kształty
    value: Stały NCHW. dynamic=True jest odrzucane.
  - label: Precyzja
    value: 'Tylko FP32, tylko CPU.'
  - label: Zadania
    value: W tej wersji tylko detekcja
verification: >-
  Odczytane z libreyolo/export/mnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/mnn.py i pyproject.toml na
  gałęzi dev.
snippets:
  install:
    - label: Instalacja
      language: bash
      code: >
        # Dodatek zawiera libreyolo[onnx]: MNN konwertuje z pośredniego pliku
        ONNX.

        pip install "libreyolo[mnn]"
    - label: 'Sprawdzenie, czy konwerter jest dostępny w PATH'
      language: bash
      code: |
        mnnconvert --version
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Zapisuje weights/LibreYOLO9t.mnn i weights/LibreYOLO9t.mnn.json
        path = model.export(format="mnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format mnn --imgsz 640
    - label: Argumenty
      language: python
      code: |
        model.export(
            format="mnn",
            imgsz=640,        # int lub (wysokość, szerokość)
            batch=1,          # wbudowany na stałe w artefakt
            simplify=True,    # onnxsim na pośrednim pliku ONNX
            output_path=None, # None zapisuje weights/<stem>.mnn
            verbose=False,    # True wypisuje na bieżąco log mnnconvert
        )

        # dynamic=True zgłasza ValueError. half=True i int8=True są odrzucane.
  run:
    - label: Przez LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.mnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Samo MNN
      language: python
      code: >
        import json


        import MNN

        import numpy as np


        meta = json.load(open("weights/LibreYOLO9t.mnn.json"))

        print(meta["mnn_input_names"], meta["mnn_output_names"],
        meta["mnn_input_shape"])


        runtime = MNN.nn.create_runtime_manager(
            ({"backend": 0, "precision": 1, "numThread": 4},)
        )

        module = MNN.nn.load_module_from_file(
            "weights/LibreYOLO9t.mnn",
            meta["mnn_input_names"],
            meta["mnn_output_names"],
            runtime_manager=runtime,
            dynamic=False,
            shape_mutable=False,
        )


        blob = np.zeros(meta["mnn_input_shape"], dtype=np.float32)

        input_var = MNN.expr.const(
            blob, list(blob.shape), MNN.expr.NCHW, MNN.expr.float
        )

        outputs = module.forward([input_var])

        for out in outputs:
            print(np.array(MNN.expr.convert(out, MNN.expr.NCHW).read()).shape)

        # Wstępne i końcowe przetwarzanie są na tej ścieżce po stronie
        użytkownika.
  support:
    - label: Sprawdzenie jednej rodziny i zadania przed eksportem
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 68fad34d07aea149
---

## Instalacja

<code-tabs name="install" />

Dodatek zawiera `libreyolo[onnx]`, ponieważ konwersja przebiega przez pośredni
plik ONNX. Instaluje też program `mnnconvert`, którego eksporter szuka najpierw
obok aktywnego interpretera Pythona, a dopiero potem w `PATH`. Brak konwertera
powoduje zgłoszenie `ImportError` z nazwą polecenia instalacji, zamiast awarii w
połowie konwersji.

## Eksport

<code-tabs name="export" />

Zanim graf zostanie przekazany dalej, eksporter odczytuje kontrakt wejścia ONNX i
odrzuca wszystko, czego nie potrafi wyrazić: więcej niż jedno wejście obrazowe lub
kształt wejścia z wymiarem symbolicznym. MNN w tej wersji wymaga w pełni
ustalonego kształtu NCHW, a `batch` jest wbudowany na stałe w artefakt, zamiast
być ustalany przy wczytywaniu.

Towarzyszący plik nie jest opcjonalną formalnością. `weights/LibreYOLO9t.mnn.json`
zapisuje nazwy wejść i wyjść, stały kształt wejścia, batch, nazwy klas, użytą
wersję MNN oraz backend, pod który zbudowano artefakt, a środowisko uruchomieniowe
sprawdza każde z tych pól przy wczytywaniu.

W systemie Windows MNN 3.6.1 czasami kończy konwersję, a następnie przerywa
działanie podczas zamykania procesu, zgłaszając naruszenie ochrony pamięci lub
status fail-fast. Eksporter rozpoznaje te konkretne kody wyjścia i traktuje
konwersję jako udaną, gdy plik wynikowy istnieje.

## Uruchamianie artefaktu

<code-tabs name="run" />

`LibreYOLO()` rozpoznaje rozszerzenie `.mnn` i zwraca ten sam obiekt `Results` co
checkpoint. Wczytywanie jest z założenia rygorystyczne: towarzyszący plik musi
deklarować `format=mnn`, `mnn_backend=cpu`, `dynamic=false`, `precision=fp32`,
rozmiar, zadanie detekcji, stały dodatni kształt NCHW zgodny z zapisanym rozmiarem
obrazu oraz nazwy klas obejmujące każdy indeks od 0 do `nc - 1`. Każda
niezgodność powoduje zgłoszenie błędu, zamiast zgadywania.

Predykcja przy innym `imgsz` niż ten, pod który zbudowano artefakt, również
zgłasza błąd, a `device` jest ignorowane z ostrzeżeniem, ponieważ eksporty MNN
działają tutaj na CPU.

Drugi fragment pokazuje ścieżkę przez samo środowisko uruchomieniowe. Wstępne
przetwarzanie, dekodowanie, NMS i przeskalowanie współrzędnych są tam po stronie
użytkownika, a nazwy wejść i wyjść pochodzą z towarzyszącego pliku, ponieważ moduł
ładujący w MNN wymaga ich podania wprost.

## Ograniczenia

Tylko detekcja. Backend odrzuca przy wczytywaniu każde inne zadanie, a strona
eksportu zachowuje się tak samo: poza zapisanymi kombinacjami kontrola wstępna
zgłasza błąd „MNN v1 has no implemented runtime contract for this family and
task”.

FP32, CPU, stały kształt. `dynamic=True` zgłasza `ValueError`, a `half=True` i
`int8=True` są odrzucane podczas walidacji.

Zwalidowane rodziny detekcyjne to YOLO9, YOLO9-E2E, YOLO9-P2, RF-DETR, EC,
RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM i YOLO-NAS, a każdą z nich obejmuje
konwersja, ponowne wczytanie świeżego artefaktu, wykonanie na CPU w MNN,
sprawdzenie metadanych oraz zgodność detekcji po NMS z modelem PyTorch. DEIMv2
konwertuje się, wczytuje ponownie, wykonuje i zachowuje detekcje po NMS, ale jego
pośrednia ścieżka ONNX ma niepełną zgodność wyników na poziomie zapytań, dlatego
jest zapisany jako dostępny, a nie zwalidowany.

Pełną siatkę rodzin i zadań zawiera
[macierz eksportu](/docs/reference/export-matrix). Dla jednej kombinacji:

<code-tabs name="support" />
