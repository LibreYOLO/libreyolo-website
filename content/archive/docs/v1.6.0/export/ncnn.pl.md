---
title: ncnn
seo_title: Eksport do ncnn z LibreYOLO
description: >-
  Eksport modelu LibreYOLO do ncnn przez PNNX: para plików param i bin, stałe
  płótno eksportu, przepisanie warstwy Focus w YOLOX i rodziny, które się
  konwertują.
lead: >-
  ncnn to biblioteka Tencent do inferencji na CPU, przeznaczona na urządzenia
  mobilne. LibreYOLO konwertuje przez PNNX, zapisując graf model.ncnn.param obok
  pliku wag model.ncnn.bin oraz plik metadata.yaml, który niesie rodzinę,
  zadanie i nazwy klas.
keywords:
  - eksport yolo do ncnn
  - pnnx
  - model.ncnn.param
  - yolo na androidzie
  - ncnn extractor
  - focus pixel_unshuffle
last_verified: 1.5.0
meta:
  - label: Flaga
    value: export(format="ncnn")
    mono: true
  - label: Zapisuje
    value: 'Katalog z plikami model.ncnn.param, model.ncnn.bin i metadata.yaml'
  - label: Dodatkowo
    value: 'pip install "libreyolo[ncnn]"'
    mono: true
  - label: Wczytywanie z powrotem
    value: LibreYOLO("weights/LibreYOLO9t_ncnn")
    mono: true
  - label: Kształty
    value: Stałe. Metadane zapisują dynamic=False niezależnie od flagi.
  - label: Precyzja
    value: Tylko FP32. half=True i int8=True są odrzucane.
verification: >-
  Odczytane z libreyolo/export/ncnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/ncnn.py i pyproject.toml na
  gałęzi dev.
snippets:
  install:
    - label: Instalacja
      language: bash
      code: |
        # pnnx konwertuje, ncnn uruchamia wynik.
        pip install "libreyolo[ncnn]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Zapisuje katalog weights/LibreYOLO9t_ncnn
        path = model.export(format="ncnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format ncnn --imgsz 640
    - label: Argumenty
      language: python
      code: |
        model.export(
            format="ncnn",
            imgsz=640,        # int lub (wysokość, szerokość)
            batch=1,
            simplify=True,    # dotyczy wyłącznie zapasowej ścieżki przez ONNX
            opset=None,       # auto; dotyczy wyłącznie zapasowej ścieżki przez ONNX
            output_path=None, # None zapisuje weights/<stem>_ncnn
        )

        # half=True i int8=True są odrzucane podczas walidacji.
  run:
    - label: Przez LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_ncnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Samo ncnn
      language: python
      code: >
        import ncnn

        import numpy as np

        import yaml


        directory = "weights/LibreYOLO9t_ncnn"

        net = ncnn.Net()

        net.load_param(f"{directory}/model.ncnn.param")

        net.load_model(f"{directory}/model.ncnn.bin")


        # ncnn przyjmuje pojedynczy obraz CHW, a nie batch.

        mat_in = ncnn.Mat(np.zeros((3, 640, 640), dtype=np.float32))

        extractor = net.create_extractor()

        extractor.input("in0", mat_in)

        ret, mat_out = extractor.extract("out0")

        print(ret, np.array(mat_out).shape)


        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Wstępne i końcowe przetwarzanie są na tej ścieżce po stronie
        użytkownika.
  support:
    - label: Sprawdzenie jednej rodziny i zadania przed eksportem
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 9a849a16a3b32334
---

## Instalacja

<code-tabs name="install" />

Ten extra instaluje obie połowy toolchainu: `pnnx` wykonuje konwersję, a `ncnn`
uruchamia wynik. Na ścieżce podstawowej żadne z nich nie przechodzi przez ONNX.

## Eksport

<code-tabs name="export" />

Artefaktem jest katalog. `weights/LibreYOLO9t_ncnn` zawiera
`model.ncnn.param`, `model.ncnn.bin` i `metadata.yaml`; wszystkie trzy pliki to
jeden artefakt i przenosi się je razem.

Konwersja najpierw próbuje użyć PNNX bezpośrednio z modelu PyTorch. Jeśli to się
nie uda, eksportuje statyczny graf ONNX do katalogu tymczasowego i wywołuje na nim
narzędzie wiersza poleceń `pnnx`, a błąd jest zgłaszany dopiero wtedy, gdy zawiodą
obie ścieżki, z opisem obu błędów. Dlatego `opset` i `simplify` wpływają wyłącznie
na ścieżkę zapasową.

YOLOX wymaga jednego przepisania, aby konwersja w ogóle się powiodła. Jego warstwa
Focus korzysta z krokowego wycinania, którego PNNX nie potrafi odwzorować, więc
eksport zamienia ją na `pixel_unshuffle` i permutuje kanały wejściowe następnej
konwolucji, aby skompensować inną kolejność kanałów. Wynik jest numerycznie
identyczny, a oryginalne wagi są przywracane po eksporcie.

## Uruchamianie artefaktu

<code-tabs name="run" />

`LibreYOLO()` rozpoznaje każdy katalog zawierający `model.ncnn.param` i
`model.ncnn.bin`, odczytuje `metadata.yaml` i zwraca ten sam obiekt `Results` co
checkpoint.

Drugi fragment kodu to ścieżka samego środowiska uruchomieniowego i dwa szczegóły
odróżniają ją od każdego innego formatu tutaj. ncnn pracuje na pojedynczym obrazie
CHW, a nie na batchu, więc nie ma wiodącej osi batcha. Nazwy blobów pochodzą z
pliku `.param`; PNNX zapisuje umownie `in0` i `out0`, a backend parsuje ten plik,
zamiast je zakładać. Wstępne przetwarzanie, dekodowanie, NMS i przeskalowanie
współrzędnych są na tej ścieżce po stronie użytkownika.

## Ograniczenia

FP32 na stałym płótnie. `half=True` i `int8=True` są odrzucane podczas walidacji,
a wyeksportowane metadane zapisują `dynamic=False` niezależnie od tego, co mówiła
flaga, więc żaden backend nie zakłada osi, której graf nie ma.

Każda rodzina w stylu DETR jest odrzucana w preflight: `detr`, `deformable_detr`,
`dinodetr`, `dfine`, `lwdetr`, `deim`, `deimv2`, `rtdetr`, `rtdetrv2`, `rtdetrv4`,
`rfdetr` i `ec`. Komunikat jest dla wszystkich taki sam, że model wymaga operacji
dekodera lub próbkowania niedostępnych w ncnn, i wskazuje w zamian ONNX, OpenVINO,
TorchScript lub TensorRT.

To, co się konwertuje, jest szerokie po stronie sieci konwolucyjnych: YOLO9 i
YOLO9-E2E, YOLOX, PicoDet, detekcja i estymacja pozy w YOLO-NAS, starsze detektory
YOLO1, YOLO3, YOLO4 i YOLO7, cztery rodziny klasyfikacyjne CNN, segmentacja
semantyczna PIDNet, detekcja punktowa FOMO w stałym rozmiarze 96 na 96, ZipDepth,
NAFNet i Real-ESRGAN.

Zablokowane pozycje wskazują konkretną przyczynę niepowodzenia. Grafy
transformerowe zwykle zostawiają nieobsługiwane węzły `pnnx.Expression`, co daje
sieć bez uruchamialnego bloba wejściowego, i właśnie to zatrzymuje DINOv2, CLIP,
SigLIP2 oraz SegFormer. BiRefNet wymaga deformowalnej konwolucji z torchvision,
której PNNX nie potrafi odwzorować. Przekonwertowany graf YOLO2 kończy działanie
środowiska uruchomieniowego ncnn w systemie Windows natywnym dzieleniem
całkowitym przez zero podczas pobierania wyjścia.

Pełną siatkę rodzin i zadań zawiera
[macierz eksportu](/docs/reference/export-matrix). Dla jednej kombinacji:

<code-tabs name="support" />
