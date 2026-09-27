---
title: TFLite
seo_title: Eksport do TFLite (LiteRT) z LibreYOLO
description: >-
  Eksportuj model LibreYOLO do bufora FlatBuffer .tflite za pomocą onnx2tf:
  statyczne kształty, tylko FP32, wejścia NHWC i rodziny, które konwertują się
  bez problemów.
lead: >-
  TFLite to format FlatBuffer wykonywany przez LiteRT na urządzeniach mobilnych
  i wbudowanych. LibreYOLO eksportuje statyczny graf ONNX, konwertuje go za
  pomocą onnx2tf w trybie flatbuffer-direct i zapisuje metadane modelu obok
  artefaktu w pliku pomocniczym JSON.
keywords:
  - eksport yolo do tflite
  - litert
  - onnx2tf
  - ai-edge-litert
  - tflite flatbuffer
  - wejście nhwc tflite
  - inferencja na urządzeniach brzegowych
last_verified: 1.5.0
meta:
  - label: Flaga
    value: export(format="tflite")
    mono: true
  - label: Zapisywane pliki
    value: Jeden plik .tflite oraz plik pomocniczy metadanych .tflite.json
  - label: Dodatek
    value: 'pip install "libreyolo[tflite]"'
    mono: true
  - label: Ponowne wczytanie
    value: LibreYOLO("weights/LibreYOLO9t.tflite")
    mono: true
  - label: Kształty
    value: Tylko statyczne. Ustawienie dynamic=True jest odrzucane.
  - label: Precyzja
    value: Tylko FP32. Ustawienia half=True i int8=True są odrzucane.
  - label: Wymagania
    value: >-
      Python 3.12 lub nowszy, ponieważ onnx2tf 2.4.x nie udostępnia pakietów
      wheel dla starszych wersji
verification: >-
  Opracowano na podstawie plików libreyolo/export/tflite.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py,
  libreyolo/backends/tflite.py i pyproject.toml w gałęzi dev.
snippets:
  install:
    - label: Instalacja
      language: bash
      code: >
        # LiteRT to obecna nazwa TensorFlow Lite używana przez Google. Oba
        dodatki

        # instalują ten sam zestaw narzędzi i tworzą ten sam plik wyjściowy
        .tflite.

        pip install "libreyolo[tflite]"
    - label: Najpierw sprawdź wersję Pythona
      language: bash
      code: |
        python -c "import sys; print(sys.version_info >= (3, 12))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Zapisuje weights/LibreYOLO9t.tflite i weights/LibreYOLO9t.tflite.json
        path = model.export(format="tflite", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tflite --imgsz 640

        # "litert" jest akceptowany jako alias i wskazuje ten sam eksporter.
        libreyolo export --model LibreYOLO9t.pt --format litert --imgsz 640
    - label: Argumenty
      language: python
      code: >
        model.export(
            format="tflite",
            imgsz=640,        # int lub (wysokość, szerokość)
            batch=1,
            simplify=True,    # onnxsim na pośrednim pliku ONNX
            output_path=None, # None zapisuje weights/<stem>.tflite
            verbose=False,    # True przesyła dziennik onnx2tf na bieżąco
        )


        # dynamic=True zgłasza ValueError: konwerter wymaga statycznych
        kształtów.

        # half=True i int8=True są odrzucane przed śledzeniem grafu.
  run:
    - label: Przez LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.tflite")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Czyste LiteRT
      language: python
      code: >
        import json


        import numpy as np

        from ai_edge_litert.interpreter import Interpreter


        interpreter = Interpreter(model_path="weights/LibreYOLO9t.tflite")

        interpreter.allocate_tensors()

        detail = interpreter.get_input_details()[0]

        print(detail["shape"], detail["dtype"])   # NHWC, nie NCHW


        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"],
        np.float32))

        interpreter.invoke()

        for output in interpreter.get_output_details():
            print(output["name"], interpreter.get_tensor(output["index"]).shape)

        # Nazwy klas, zadanie i rozmiar wejścia znajdują się w pliku
        pomocniczym.

        meta = json.load(open("weights/LibreYOLO9t.tflite.json"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Za wstępne przetwarzanie, transpozycję NCHW na NHWC i przetwarzanie
        końcowe odpowiada użytkownik.
  support:
    - label: Sprawdzenie rodziny i zadania przed eksportem
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: fa2deaa0ef6d9978
---

## Instalacja

<code-tabs name="install" />

Dodatek instaluje `onnx2tf` do konwersji oraz `ai-edge-litert` do uruchamiania
wyniku. Oba pakiety są objęte warunkiem wersji Python 3.12. W starszym
interpreterze eksport zgłasza `ImportError` z wymaganiem dotyczącym wersji,
zamiast zakończyć się błędem wewnątrz konwertera.

`libreyolo[litert]` instaluje dokładnie to samo. Ciąg formatu `litert` jest
aliasem `tflite`, a plikiem wyjściowym w obu przypadkach jest `.tflite`.

## Eksport

<code-tabs name="export" />

Rodzina i zadanie są sprawdzane przed wykonaniem innych operacji, dlatego
nieobsługiwana kombinacja natychmiast zgłasza konkretny błąd konwertera lub
środowiska uruchomieniowego, który spowodował jej wykluczenie, zamiast
ogólnego komunikatu. Sama konwersja odbywa się przez wywołanie `onnx2tf`
w podprocesie, w trybie `flatbuffer_direct`, na statycznym pliku pośrednim ONNX.

Metadane znajdują się w pliku pomocniczym. Plik
`weights/LibreYOLO9t.tflite.json` zawiera rodzinę, zadanie, nazwy klas, rozmiar
wejścia i schemat pozy. Sam FlatBuffer nie ma pola metadanych LibreYOLO, dlatego
oba pliki należy przenosić razem.

## Uruchamianie artefaktu

<code-tabs name="run" />

`LibreYOLO()` wybiera backend na podstawie sufiksu `.tflite` i zwraca ten sam
obiekt `Results` co checkpoint. Backend odczytuje plik pomocniczy, transponuje
blob NCHW do NHWC, gdy interpreter wymaga wejścia z kanałami na końcu, stosuje
skalę kwantyzacji i punkt zerowy interpretera, jeśli są dostępne, a następnie
transponuje dane wyjściowe z powrotem do układu oczekiwanego przez przetwarzanie
końcowe LibreYOLO.

Drugi fragment przedstawia ścieżkę z samym środowiskiem uruchomieniowym. W tym
przypadku za wstępne przetwarzanie, transpozycję układu, dekodowanie, NMS
i przeskalowanie współrzędnych odpowiada użytkownik. Najłatwiej przeoczyć układ:
onnx2tf generuje wejścia z kanałami na końcu, więc nie można powiązać blobu
o kształcie `(1, 3, 640, 640)`.

## Ograniczenia

Obsługiwane są tylko statyczne kształty. Ustawienie `dynamic=True` zgłasza
`ValueError` przed śledzeniem grafu, a obszar eksportu jest ustalony na wartość,
do której zostało przeliczone `imgsz`.

Obsługiwane jest tylko FP32. Ustawienia `half=True` i `int8=True` są odrzucane
podczas walidacji, więc ten eksporter nie pozwala obecnie na wdrożenie
skwantyzowanego modelu.

Zakres obsługi jest tu węższy niż w przypadku formatów grafu i wynika z pomiarów,
a nie z samej rodziny. Zweryfikowane kombinacje obejmują detekcję w modelach
YOLO9, YOLOX i YOLO-NAS, segmentację semantyczną w PIDNet, cztery rodziny
klasyfikacyjnych sieci CNN, embeddingi w DINOv2 i SigLIP2, klasyfikację
w SigLIP2, detekcję krawędzi w TEED i DexiNed oraz rekonstrukcję obrazu
w Real-ESRGAN i SwinIR. Z modelem SwinIR wiąże się dodatkowe zastrzeżenie:
zgodność zostaje zachowana, gdy wymiary źródłowe dokładnie odpowiadają obszarowi
eksportu. Mniejsze źródła są dopełniane do tego obszaru przed uruchomieniem
transformera, co może dawać inne wyniki niż natywna inferencja ze zmiennym
rozmiarem.

Zablokowane pozycje zawierają dokładną przyczynę błędu, z którą warto zapoznać
się przed próbą obejścia problemu. Kilka przykładów: detekcja RF-DETR konwertuje
się z natywnym obszarem 384, ale LiteRT nie może przydzielić jej pamięci, ponieważ
`STRIDED_SLICE` otrzymuje wejście o randze większej niż obsługiwane 5 wymiarów.
PicoDet jest odrzucany, ponieważ operacja `RESHAPE` mapuje 19 200 elementów
wejściowych na 9 600 elementów wyjściowych. D-FINE powoduje awarię konwertera
podczas obsługi kształtu `GatherElements`. RTMDet eksportuje się i ponownie
wczytuje z zachowaną zgodnością surowych danych, lecz wartość IoU publicznych
ramek spada do 0.911 przy przesunięciu współrzędnych wynoszącym 29.9 px.

Pełną siatkę rodzin i zadań zawiera
[macierz eksportu](/docs/reference/export-matrix). Aby sprawdzić jedną kombinację,
w tym przyczynę jej zablokowania:

<code-tabs name="support" />
