---
title: Paddle
seo_title: Eksport do PaddlePaddle z LibreYOLO
description: >-
  Konwersja detektora LibreYOLO na model inferencyjny PaddlePaddle przez
  X2Paddle: przypięty toolchain, statyczne grafy FP32 z batchem 1 i inferencja
  na CPU.
lead: >-
  Modele inferencyjne PaddlePaddle to graf model.pdmodel obok pliku wag
  model.pdiparams. LibreYOLO eksportuje statyczny graf ONNX w opset 15,
  konwertuje go za pomocą X2Paddle i pakuje wynik razem z plikiem metadata.yaml,
  dzięki czemu wczytuje się on przez tę samą fabrykę co każde inne środowisko
  uruchomieniowe.
keywords:
  - eksport yolo do paddle
  - paddlepaddle inferencja
  - x2paddle
  - model.pdmodel
  - model.pdiparams
  - onnx opset 15
last_verified: 1.5.0
meta:
  - label: Flaga
    value: export(format="paddle")
    mono: true
  - label: Zapisuje
    value: 'Katalog z plikami model.pdmodel, model.pdiparams i metadata.yaml'
  - label: Dodatkowo
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: Wczytywanie z powrotem
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: Backend
    value: libreyolo.backends.paddle.PaddleBackend
    mono: true
  - label: Kształty
    value: 'Statyczne, batch 1, opset 15. Wszystkie trzy są wymuszane.'
  - label: Precyzja
    value: 'Tylko FP32, tylko CPU.'
  - label: Toolchain
    value: >-
      PaddlePaddle 2.6.2, X2Paddle 1.6.0, ONNX 1.17 lub starszy, sprawdzane
      dokładnie
verification: >-
  Odczytane z libreyolo/export/paddle.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/paddle.py, docs/paddle.md i
  pyproject.toml na gałęzi dev.
snippets:
  install:
    - label: Instalacja
      language: bash
      code: >
        # Python od 3.10 do 3.12. Zwalidowaną ścieżką na Windows jest WSL2 z
        Ubuntu 22.04.

        pip install "libreyolo[paddle]"
    - label: Sprawdzenie przypiętych wersji
      language: bash
      code: >
        python -c "from importlib.metadata import version;
        print(version('paddlepaddle'), version('x2paddle'), version('onnx'))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Zapisuje katalog weights/LibreYOLO9t_paddle
        path = model.export(format="paddle")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format paddle
    - label: Argumenty
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # int; kwadratowe płótno tej rodziny
            batch=1,          # każda inna wartość zgłasza ValueError
            dynamic=False,    # True zgłasza ValueError
            simplify=True,    # False zgłasza ValueError
            opset=15,         # każda inna wartość zgłasza ValueError
            output_path=None, # None zapisuje weights/<stem>_paddle
        )
  run:
    - label: Przez LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: CLI
      language: bash
      code: |
        libreyolo predict --model weights/LibreYOLO9t_paddle \
          --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --device cpu --save
    - label: Bezpośrednie użycie backendu
      language: python
      code: |
        from libreyolo.backends.paddle import PaddleBackend

        # To, co LibreYOLO() konstruuje dla katalogu Paddle. Ten sam obiekt
        # Results, bez routingu fabryki po drodze.
        backend = PaddleBackend("weights/LibreYOLO9t_paddle", device="cpu")
        result = backend.predict("parkour.jpg")
        print(result.boxes.xyxy[:3])
    - label: Samo Paddle
      language: python
      code: >
        import numpy as np

        import paddle.inference as paddle_infer

        import yaml


        directory = "weights/LibreYOLO9t_paddle"

        config = paddle_infer.Config(
            f"{directory}/model.pdmodel", f"{directory}/model.pdiparams"
        )

        config.disable_gpu()

        config.disable_mkldnn()

        config.switch_ir_optim(False)


        predictor = paddle_infer.create_predictor(config)

        handle = predictor.get_input_handle(predictor.get_input_names()[0])

        handle.reshape([1, 3, 640, 640])

        handle.copy_from_cpu(np.zeros((1, 3, 640, 640), dtype=np.float32))

        predictor.run()

        for name in predictor.get_output_names():
            print(name, predictor.get_output_handle(name).copy_to_cpu().shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Na tej ścieżce preprocessing i postprocessing leżą po stronie
        użytkownika.
  support:
    - label: Sprawdzenie jednej rodziny i zadania przed eksportem
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cdd8bf12286e2f53
---

## Instalacja

<code-tabs name="install" />

Ten extra przypina dokładnie ten stos, na którym mierzono parytet: PaddlePaddle
2.6.2, X2Paddle 1.6.0 oraz ONNX 1.17 lub starszy. Te przypięcia są sprawdzane w
momencie eksportu, a nie tylko podczas instalacji, a inna wersja zgłasza
`ImportError` z nazwą wersji oczekiwanej. Nowsze wydania Paddle odrzucają część
statycznego kodu generowanego przez X2Paddle 1.6.0, więc wcześniejszy błąd jest
lepszy niż wytworzenie artefaktu, którego nikt nie zwalidował.

## Eksport

<code-tabs name="export" />

Cztery argumenty są ustalone na sztywno, a nie tylko mają wartości domyślne.
`dynamic` musi być `False`, `batch` musi wynosić 1, `simplify` musi być `True`,
aby graf konwersji był w pełni statyczny, a `opset` musi wynosić 15, czyli tyle,
ile najwyżej przyjmuje X2Paddle 1.6.0. Przekazanie czegokolwiek innego zgłasza
błąd jeszcze przed trasowaniem.

Na grafie pośrednim wykonuje się jedna normalizacja. ONNX definiuje pominiętą
dylatację MaxPool jako jedynkę, PyTorch zapisuje jawny atrybut złożony z samych
jedynek, a X2Paddle 1.6.0 go odrzuca, więc eksporter usuwa tę zbędną wartość
domyślną i pozostawia opisaną operację bez zmian.

Artefaktem jest katalog: `model.pdmodel`, `model.pdiparams` i `metadata.yaml`.
Kod w Pythonie, który X2Paddle generuje podczas konwersji, nie wchodzi w jego
skład.

## Uruchamianie artefaktu

<code-tabs name="run" />

`LibreYOLO()` rozpoznaje każdy katalog zawierający jednocześnie `model.pdmodel`
i `model.pdiparams`, odczytuje `metadata.yaml` i zwraca ten sam obiekt `Results`
co checkpoint. Urządzenie inne niż `auto` lub `cpu` zgłasza błąd: ten backend
działa wyłącznie na CPU.

Fabryka konstruuje `PaddleBackend`, eksportowany z `libreyolo` i importowalny
jako `libreyolo.backends.paddle.PaddleBackend`. Skonstruuj go samodzielnie, gdy
potrzebny jest backend bez routingu po sufiksach wykonywanego przez fabrykę, na
przykład aby jawnie przekazać `task=` dla katalogu, którego `metadata.yaml` nie
został napisany samodzielnie. Jego `predict()` przyjmuje te same źródła i zwraca
te same wyniki.

Fragment kodu z samym środowiskiem uruchomieniowym odzwierciedla to, co
konfiguruje backend, a trzy wyłączone opcje są celowe. Pipeline fuzji na CPU w
Paddle 2.6 potrafi się wywrócić podczas optymalizowania dużych grafów gather i
scatter emitowanych dla deformable attention, więc parytet mierzono względem
przenośnego, niescalonego grafu statycznego. Na tej ścieżce preprocessing,
dekodowanie, NMS i przeskalowanie współrzędnych leżą po stronie użytkownika.

## Ograniczenia

Brak dynamicznych kształtów, brak FP16, brak INT8, brak wbudowanego NMS, brak
środowiska uruchomieniowego na GPU.

Zwalidowane kombinacje to detekcja YOLO9, detekcja YOLO9-E2E i YOLO9-P2,
detekcja, estymacja pozy i segmentacja w rodzinie EC, detekcja RT-DETRv4,
D-FINE, DEIM i DEIMv2 oraz detekcja i estymacja pozy YOLO-NAS. Każdą z nich
pokrywa konwersja, ponowne wczytanie w środowisku uruchomieniowym na CPU,
parytet surowych wyjść i zgodność z opublikowanymi wynikami.

Zablokowane, z przyczyną zapisaną dla każdej kombinacji:

| Kombinacja | Dlaczego |
|---|---|
| RF-DETR, wszystkie zadania | Wymaga ONNX w opset 17 i GridSample; X2Paddle 1.6.0 przyjmuje opset 15 lub niższy i nie ma mapowania dla GridSample |
| Detekcja RT-DETR i RT-DETRv2 | Wytrenowane grafy potrzebują GridSample w opset 16 lub nowszym |
| Segmentacja D-FINE | Konwertuje się i wczytuje ponownie, ale względny błąd RMS logitów maski wynosi 3.52%, a minimalne IoU dopasowanych masek to 0.582 |
| Segmentacja YOLO9 | W LibreYOLO YOLO9 obsługuje wyłącznie detekcję |
| Segmentacja RTMDet-Ins | Dekodowanie masek z dynamicznym jądrem nie ma kontraktu dla wyeksportowanego środowiska uruchomieniowego |

Wszystko, co nie zostało wymienione jako zwalidowane ani zablokowane, jest
odrzucane z adnotacją, że nie przeszło walidacji na ścieżce konwersji z ONNX do
Paddle.

Pełną siatkę rodzin i zadań zawiera
[macierz eksportu](/docs/reference/export-matrix). Dla jednej kombinacji:

<code-tabs name="support" />
