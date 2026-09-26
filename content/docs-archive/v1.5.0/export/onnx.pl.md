---
title: ONNX
seo_title: Eksport do ONNX z LibreYOLO
description: >-
  Eksport modelu LibreYOLO do formatu ONNX: opset dobierany dla każdej rodziny,
  dynamiczne osie, wbudowany NMS, INT8 i ponowne wczytanie grafu.
lead: >-
  ONNX to przenośny format grafu. LibreYOLO trasuje model przez
  torch.onnx.export, opcjonalnie upraszcza graf i zapisuje rodzinę, zadanie,
  nazwy klas oraz rozmiar wejścia w metadanych samego pliku, dzięki czemu każdy
  backend LibreYOLO potrafi odtworzyć postprocessing.
keywords:
  - eksport yolo do onnx
  - onnxruntime
  - torch.onnx.export
  - onnx opset
  - dynamiczne osie onnx
  - wbudowany nms onnx
  - onnx int8 qdq
  - onnx metadata_props
last_verified: 1.5.0
meta:
  - label: Flaga
    value: export(format="onnx")
    mono: true
  - label: Zapisuje
    value: 'Jeden plik .onnx, metadane osadzone w grafie'
  - label: Dodatkowo
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Wczytywanie z powrotem
    value: LibreYOLO("weights/LibreYOLO9t.onnx")
    mono: true
  - label: Kształty
    value: >-
      Domyślnie dynamiczny batch w Pythonie; wyjątki dla poszczególnych zadań
      poniżej
  - label: Precyzja
    value: 'FP32, FP16 (half=True), INT8 (int8=True, detekcja w YOLO9)'
verification: >-
  Odczytane z libreyolo/export/onnx.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/onnx.py i
  libreyolo/cli/commands/export.py na gałęzi dev.
snippets:
  install:
    - label: Instalacja
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Zapisuje weights/LibreYOLO9t.onnx
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx
    - label: Argumenty
      language: python
      code: |
        model.export(
            format="onnx",
            imgsz=640,        # int albo (wysokość, szerokość)
            batch=1,
            dynamic=True,     # domyślne w Pythonie; w CLI domyślnie False
            simplify=True,    # uruchamia onnxsim na grafie
            opset=None,       # None wybiera 13 albo 17 dla rodzin w stylu DETR
            half=False,       # wagi i aktywacje w FP16
            int8=False,       # QDQ INT8, tylko detekcja w YOLO9
            data=None,        # data.yaml do kalibracji, tylko INT8
            device=None,      # urządzenie trasowania; None bierze urządzenie modelu
            output_path=None, # None zapisuje weights/<stem>.onnx
        )
  nms:
    - label: Wbudowanie NMS w graf
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Tylko detekcja w YOLO9, batch 1. dynamic jest wymuszane na False.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            nms=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx --nms \
          --conf 0.25 --iou 0.45 --max-det 300
  int8:
    - label: INT8 z danymi kalibracyjnymi
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            int8=True,
            data="coco128.yaml",   # kilkaset reprezentatywnych obrazów
            fraction=1.0,
        )
  run:
    - label: Przez LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.onnx")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Samo ONNX Runtime
      language: python
      code: >
        import numpy as np

        import onnx

        import onnxruntime as ort


        session = ort.InferenceSession(
            "weights/LibreYOLO9t.onnx",
            providers=["CPUExecutionProvider"],
        )


        # Wstępne przetwarzanie i postprocessing są tu po stronie użytkownika.

        batch = np.zeros((1, 3, 640, 640), dtype=np.float32)

        outputs = session.run(None, {session.get_inputs()[0].name: batch})

        print([out.shape for out in outputs])


        # Graf niesie rodzinę, zadanie, nazwy klas i rozmiar wejścia.

        meta = {p.key: p.value for p in
        onnx.load("weights/LibreYOLO9t.onnx").metadata_props}

        print(meta["model_family"], meta["task"], meta["imgsz"])
  support:
    - label: Sprawdzenie jednej rodziny i zadania przed eksportem
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cee78250fc7189a3
---

## Instalacja

<code-tabs name="install" />

Dodatek instaluje `onnx`, `onnxsim` i `onnxruntime`. Sam `onnx` wystarczy, aby
zapisać plik; `onnxsim` wykonuje przebieg upraszczania, a `onnxruntime` uruchamia
artefakt i przeprowadza kalibrację INT8.

## Eksport

<code-tabs name="export" />

Bez `output_path` plik trafia do `weights/` pod rdzeniem nazwy checkpointu, z
dopisanym `_fp16` lub `_int8`, gdy zażądano tej precyzji.

`dynamic` ma domyślnie wartość `True` w Pythonie i `False` w CLI. Gdy jest
włączone, oś batcha staje się symboliczna, a kilka zadań otwiera się jeszcze
szerzej: segmentacja semantyczna otwiera dodatkowo wysokość i szerokość maski,
restauracja w Real-ESRGAN otwiera osie przestrzenne, a detektory dwuetapowe
zachowują dynamiczną wysokość i szerokość źródła, ponieważ ich skalowanie odbywa
się wewnątrz grafu.

`opset` jest przy pominięciu wybierany osobno dla każdej rodziny. Rodziny w stylu
DETR (`detr`, `deformable_detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`,
`lwdetr`, `rfdetr`, `rtdetr`, `rtdetrv2`, `rtdetrv4`) oraz `deit`, `midas` i
`moge2` dostają opset 17, bo dopiero tam `aten::scaled_dot_product` ma swoje
odwzorowanie. Reszta dostaje 13. Matting jest niezależnie od tego podnoszony do
19, ponieważ dekoder w BiRefNet potrzebuje operatora `DeformConv`, który ONNX
definiuje od opsetu 19.

`simplify=True` uruchamia `onnxsim` i zachowuje pierwotny graf, jeśli przebieg
się nie powiedzie, więc błąd upraszczania jest ostrzeżeniem, a nie niepowodzeniem
eksportu. Na macOS arm64 z `onnx` w wersji 1.22 lub nowszej i `onnxsim` w wersji
0.6.5 lub starszej przebieg jest pomijany w całości, ponieważ to zestawienie może
przerwać proces Pythona.

### Wbudowany NMS

<code-tabs name="nms" />

`nms=True` dotyczy wyłącznie detekcji w YOLO9 i wymaga batcha 1; zażądanie go
razem z `dynamic=True` zapisuje ostrzeżenie i wyłącza tryb dynamiczny. Graf ma
wtedy dwa wyjścia: `output` o kształcie `(batch, max_det, 6)` oraz `raw`, czyli
niezdekodowany tensor detektora, którego używa własny backend LibreYOLO, aby
postprocessing pozostał identyczny jak na ścieżce PyTorch.

### DeepStream

`deepstream=True` to opcja dostępna wyłącznie dla ONNX. Eksportuje graf w
układzie, jakiego oczekuje parser NVIDIA DeepStream, i zapisuje obok dwa pliki
towarzyszące, `config_infer_primary_<stem>.txt` oraz `<stem>_labels.txt`, dzięki
czemu artefakt wchodzi do pipeline'u bez ręcznie pisanej konfiguracji.

Wyklucza się wzajemnie z `nms=True`, a żądanie obu naraz zgłasza `ValueError`:
DeepStream wykonuje tłumienie we własnym etapie klasteryzacji. Przekazanie tej
opcji do formatu innego niż ONNX również zgłasza błąd. Siatkę obsługiwanych
rodzin i zadań oraz budowanie parsera opisuje
[DeepStream](/docs/export/deepstream).

### INT8

<code-tabs name="int8" />

`int8=True` uruchamia statyczną kwantyzację ONNX Runtime i zapisuje graf QDQ z
wejściami i wyjściami w float32. Kwantyzowane są tylko węzły `Conv` i `Gemm`.
Pozostawienie dekodowania w głowicy detekcji w float32 jest celowe: ta
konkatenacja miesza współrzędne ramek w skali pikseli z wynikami klas z zakresu
od 0 do 1, a pojedyncza skala aktywacji na tensor, zdominowana przez wielkość
ramek, sprowadziłaby każdy wynik do zera.

Ta flaga dotyczy obecnie wyłącznie detekcji w YOLO9, a wszystko inne zgłasza
`NotImplementedError` w preflight. Pominięcie `data` cofa się do `coco8.yaml` z
ostrzeżeniem; osiem obrazów to nie jest reprezentatywny zbiór kalibracyjny. Model
skwantyzowany już w PyTorch idzie inną drogą, opisaną na stronie
[Kwantyzacja](/docs/export/quantization).

## Uruchamianie artefaktu

<code-tabs name="run" />

`LibreYOLO()` rozpoznaje sufiks `.onnx` i zwraca ten sam obiekt `Results` co
checkpoint `.pt`, ponieważ nazwy klas, zadanie, rozmiar wejścia i schemat pozy
zostały podczas eksportu zapisane w `metadata_props` grafu. Przy `device="auto"`
sesja bierze `CUDAExecutionProvider`, gdy ONNX Runtime go zgłasza, a w przeciwnym
razie wraca do CPU.

Drugi fragment kodu jest dla czytelników bez zainstalowanego LibreYOLO. Wstępne
przetwarzanie, dekodowanie, NMS i przeskalowanie współrzędnych są na tej ścieżce
po stronie użytkownika; blok metadanych nadal można odczytać.

## Ograniczenia

Nazwy tensorów wyjściowych są ustalone dla każdego zadania i to do nich musi się
dopasować konsument, który nie czyta metadanych:

| Zadanie | Nazwy wyjść |
|---|---|
| Detekcja, głowice siatkowe i kotwicowe | `output` |
| Detekcja, w stylu DETR | `pred_logits`, `pred_boxes` |
| Detekcja, RF-DETR | `dets`, `labels` |
| Klasyfikacja | `output` |
| Segmentacja semantyczna | `semantic_logits` |
| Głębia | `depth` |
| Normalna powierzchni | `normal` |
| Krawędzie | `edges` |
| Restauracja | `restored` |
| Matting | `matte` |
| Spojrzenie | `yaw_logits`, `pitch_logits` |

RF-DETR jest też jedyną rodziną, której tensor wejściowy nazywa się `input`, a
nie `images`.

Kilka zadań ma w tej wersji kontrakt środowiska uruchomieniowego o stałej
rozdzielczości. Głębia, normalna powierzchni i krawędzie odrzucają `batch != 1` i
wymuszają `dynamic=False`. Matting wymusza natywny kwadrat 1024, ponieważ tablice
pozycji względnych w Swin z BiRefNet są związane ze swoją rozdzielczością.
Restauracja wymusza stałe płótno dla każdej rodziny poza Real-ESRGAN, którego
generator jest w pełni konwolucyjny.

Prostokątny `imgsz` działa dla rodzin YOLO9, HRNet, NAFNet i Real-ESRGAN.
Rodziny ze stałym kontraktem kwadratowym (`clip`, `deformable_detr`, `detr`,
`dinodetr`, `dfine`, `deim`, `deimv2`, `ec`, `lwdetr`, `moge2`, `rtdetr`,
`rtdetrv2`, `rtdetrv4`, `rfdetr`, `siglip2`, `ssd`) odrzucają go wprost.

Dwie kombinacje są odrzucane przed trasowaniem: segmentacja w YOLO9, ponieważ
YOLO9 obsługuje w LibreYOLO tylko detekcję, oraz segmentacja w RTMDet-Ins, której
dekodowanie masek z dynamicznymi jądrami nie ma kontraktu dla wyeksportowanego
środowiska uruchomieniowego.

Pełną siatkę rodzin i zadań zawiera
[macierz eksportu](/docs/reference/export-matrix). Dla jednej kombinacji można
zapytać bibliotekę wprost:

<code-tabs name="support" />
