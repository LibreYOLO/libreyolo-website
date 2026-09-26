---
title: ExecuTorch
seo_title: Eksport do ExecuTorch z LibreYOLO
description: >-
  Eksport modelu LibreYOLO do programu .pte ExecuTorch z delegacją XNNPACK:
  stały kształt, batch 1, FP32 i wymagany przez niego plik metadanych sidecar.
lead: >-
  ExecuTorch uruchamia programy PyTorch na urządzeniach brzegowych. LibreYOLO
  przechwytuje model za pomocą torch.export w trybie strict, przeprowadza
  lowering do XNNPACK i zapisuje program .pte razem z plikiem metadanych JSON
  jako jedną całość.
keywords:
  - eksport yolo do executorch
  - program .pte
  - xnnpack partitioner
  - torch.export strict
  - executorch runtime
  - inferencja pytorch edge
last_verified: 1.5.0
meta:
  - label: Flaga
    value: export(format="executorch")
    mono: true
  - label: Zapisuje
    value: Jeden program .pte plus plik metadanych .pte.json
  - label: Extra
    value: 'pip install "libreyolo[executorch]"'
    mono: true
  - label: Ponowne wczytanie
    value: LibreYOLO("weights/LibreYOLO9t.pte")
    mono: true
  - label: Kształty
    value: Stałe. dynamic=True i batch != 1 są odrzucane.
  - label: Precyzja
    value: Tylko FP32. half=True i int8=True są odrzucane.
  - label: Delegat
    value: 'XNNPACK, CPU. delegate=''xnnpack'' to jedyna akceptowana wartość.'
verification: >-
  Odczytane z libreyolo/export/executorch.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/executorch.py i pyproject.toml
  na gałęzi dev.
snippets:
  install:
    - label: Instalacja
      language: bash
      code: |
        # Celowo poza libreyolo[all]: ExecuTorch ogranicza, z którą
        # wersją Torch może być łączony.
        pip install "libreyolo[executorch]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Zapisuje weights/LibreYOLO9t.pte i weights/LibreYOLO9t.pte.json
        path = model.export(format="executorch", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format executorch --imgsz 640
    - label: Argumenty
      language: python
      code: |
        model.export(
            format="executorch",
            imgsz=640,             # int lub (wysokość, szerokość)
            batch=1,               # każda inna wartość zgłasza ValueError
            dynamic=False,         # True zgłasza ValueError
            delegate="xnnpack",    # jedyna akceptowana wartość
            device="cpu",          # każde inne urządzenie zgłasza ValueError
            output_path=None,      # None zapisuje weights/<stem>.pte
        )
  run:
    - label: Przez LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.pte")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Samo środowisko uruchomieniowe ExecuTorch
      language: python
      code: >
        import json

        from pathlib import Path


        import torch

        from executorch.runtime import Runtime


        runtime = Runtime.get()

        print(runtime.backend_registry.is_available("XnnpackBackend"))


        program =
        runtime.load_program(Path("weights/LibreYOLO9t.pte").read_bytes())

        method = program.load_method("forward")


        # Na tej ścieżce preprocessing i postprocessing są po stronie
        użytkownika.

        outputs = method.execute((torch.zeros(1, 3, 640, 640),))

        print([tensor.shape for tensor in outputs])


        meta = json.load(open("weights/LibreYOLO9t.pte.json"))

        print(meta["model_family"], meta["task"], meta["executorch_delegate"])
  support:
    - label: Sprawdzenie jednej rodziny i zadania przed eksportem
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c2c354a76ee33157
---

## Instalacja

<code-tabs name="install" />

Ten extra celowo znajduje się poza `libreyolo[all]`, ponieważ ExecuTorch przypina
wersję Torch, z którą działa, a jego instalacja przeciągnęłaby całe środowisko na
tę parę. Zainstaluj go w środowisku, w którym takie ograniczenie jest akceptowalne.

W systemie Windows krok lowering wywołuje plik wykonywalny `flatc` dostarczany z
ExecuTorch. Jeśli nie ma go w `PATH`, eksport zgłasza `RuntimeError` z takim
komunikatem, a rozwiązaniem jest uruchomienie z Developer PowerShell dla Visual
Studio 2022.

## Eksport

<code-tabs name="export" />

Przechwytywanie odbywa się przez `torch.export.export(..., strict=True)`, czyli
rzeczywiste przechwycenie grafu z guardami, a nie zapisany trace. Odczyty skalarów
po stronie hosta i przepływ sterowania zależny od danych są odrzucane, zamiast być
po cichu wpisywane na stałe, więc kilka rodzin zawodzi w tym miejscu, choć gdzie
indziej trace kończy się powodzeniem; przyczyny są zapisane dla każdej kombinacji w
macierzy wsparcia.

Krok lowering uruchamia `to_edge_transform_and_lower` z partitionerem XNNPACK.
Jeśli wynik nie zawiera żadnych partycji delegowanych, eksport zgłasza błąd, zamiast
oznaczać jako XNNPACK program korzystający wyłącznie z przenośnych kerneli.

Program i sidecar są zapisywane razem. Oba są przygotowywane, oba są podmieniane, a
błąd cofa wszystko do poprzedniego stanu, więc niekompletna para nigdy nie trafia na
dysk.

## Uruchomienie artefaktu

<code-tabs name="run" />

`LibreYOLO()` rozpoznaje sufiks `.pte` i zwraca ten sam obiekt `Results` co
checkpoint. Sidecar jest obowiązkowy przy wczytywaniu: bez pliku
`<program>.pte.json` backend zgłasza `FileNotFoundError`, ponieważ sam program nie
niesie nazw klas, zadania ani rozmiaru wejścia. Backend sprawdza też, czy
zainstalowane środowisko uruchomieniowe udostępnia `XnnpackBackend`, zanim zacznie
wczytywanie, i czyta program z bajtów zamiast mapować plik, co pozwala uniknąć
trzymania blokady pliku w systemie Windows przez cały czas życia backendu.

Drugi snippet to ścieżka przez samo środowisko uruchomieniowe. Preprocessing,
dekodowanie, NMS i przeskalowanie współrzędnych są tam po stronie użytkownika.

## Ograniczenia

Batch 1, stały kształt, FP32, CPU. Zarówno `batch != 1`, jak i `dynamic=True`
zgłaszają `ValueError`, zanim eksport cokolwiek zmieni, `half=True` i `int8=True` są
odrzucane podczas walidacji, a urządzenie inne niż CPU nie jest przyjmowane.

`delegate` przyjmuje w tej wersji `"xnnpack"` i nic więcej.

Eksporty klasyfikacji niosą dwa dodatkowe klucze metadanych, `crop_pct` i
`interpolation`, dzięki czemu środowisko uruchomieniowe może odtworzyć stosowaną
przez daną rodzinę politykę zmiany rozmiaru i przycięcia środka.

Zablokowane pozycje wskazują konkretny błąd, a nie kategorię. Detekcja i segmentacja
w D-FINE trafiają na nieobsługiwany odczyt `ContextVar` w deformowalnej uwadze przy
przechwytywaniu w trybie strict, a wymuszenie ręcznej ścieżki grid-sample serializuje
się, ale potem zawodzi w czasie działania na nieprawidłowej kolejności wymiarów
delegowanego tensora. DEIM i DEIMv2 przechodzą przechwytywanie, lowering i
serializację, po czym zawodzą podczas wykonania. Segmentacja semantyczna w EoMT
zawodzi na wyrażeniu symbolicznym zależnym od danych w ścieżce masek. Matting w
BiRefNet przechwytuje się przy 1024 na 1024, ale nie ma wariantu out dla
`torchvision::deform_conv2d`. Restauracja w SwinIR wczytuje się ponownie, a potem
zawodzi w `aten::alias_copy.out` z powodu niezgodnych kolejności wymiarów.

Pełną siatkę rodzin i zadań zawiera
[macierz eksportu](/docs/reference/export-matrix). Dla jednej kombinacji:

<code-tabs name="support" />
