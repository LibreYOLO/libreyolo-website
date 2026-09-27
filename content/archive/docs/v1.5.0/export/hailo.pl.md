---
title: Hailo
seo_title: Uruchamianie modeli LibreYOLO na akceleratorach Hailo
description: >-
  Wdrożenie modelu LibreYOLO na Hailo-8 lub Hailo-8L: statyczny eksport do ONNX,
  etap Dataflow Compiler uruchamiany samodzielnie oraz to, które architektury
  się kompilują.
lead: >-
  Akceleratory Hailo są programowane przez Hailo Dataflow Compiler, zamknięty
  SDK dystrybuowany przez Developer Zone firmy Hailo. Częścią tego przepływu po
  stronie biblioteki LibreYOLO jest zwykły statyczny eksport do ONNX;
  parsowanie, kwantyzacja i kompilacja do HEF odbywają się potem w DFC.
keywords:
  - libreyolo hailo
  - hailo-8
  - hailo-8l
  - raspberry pi ai kit
  - ai hat+
  - hailo dataflow compiler
  - kompilacja hef
  - hailortcli
last_verified: 1.5.0
meta:
  - label: Krok LibreYOLO
    value: 'export(format="onnx", imgsz=640, dynamic=False)'
    mono: true
  - label: To nie jest format
    value: Nie istnieje format="hef". DFC nie może być zależnością pip.
  - label: Extra
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Host kompilacji
    value: >-
      Linux x86_64, w tym WSL2 z Ubuntu 22.04. Kompilacja nie może działać na
      ARM.
  - label: Kompiluje się
    value: >-
      Czyste grafy CNN o stałych kształtach. Mechanizmy uwagi, dynamiczne
      kształty i projekty zdominowane przez LayerNorm już nie.
  - label: Status
    value: >-
      Żadna rodzina LibreYOLO nie została jeszcze przeprowadzona od początku do
      końca przez DFC aż do działającego HEF.
verification: >-
  Odczytane z skills/libreyolo-export-hailo/SKILL.md, libreyolo/export/onnx.py i
  libreyolo/cli/commands/export.py na gałęzi dev. Ograniczenia DFC to te
  zapisane w tym pliku skill; żaden HEF z LibreYOLO nie został skompilowany ani
  zmierzony.
snippets:
  install:
    - label: Strona LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'Strona Hailo, instalowana samodzielnie'
      language: text
      code: >
        Prerequisites, none of them installable from PyPI:


        - A Linux x86_64 machine. WSL2 Ubuntu 22.04 works. The Raspberry Pi is a
          runtime target, never the compile host.
        - The Dataflow Compiler wheel (hailo_sdk_client) from the Hailo
        Developer
          Zone, which is free to register for.
        - For Hailo-8 and Hailo-8L, the Hailo Model Zoo v2.x line, for its
          recipes and NMS configurations.
        - A GPU on the compile host is strongly recommended: the quantization
          step takes hours without one.
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Hailo wymaga batcha 1, stałej rozdzielczości i braku dynamicznych osi.

        # API Pythona domyślnie ustawia dynamic=True, więc trzeba je jawnie
        wyłączyć.

        model = LibreYOLO("LibreYOLOXs.pt")

        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    - label: CLI
      language: bash
      code: |
        # CLI domyślnie używa już statycznych kształtów.
        libreyolo export --model LibreYOLOXs.pt --format onnx --imgsz 640
    - label: 'Sprawdzenie przed kompilacją, czy graf jest statyczny'
      language: python
      code: |
        import onnx

        graph = onnx.load("weights/LibreYOLOXs.onnx").graph
        shape = graph.input[0].type.tensor_type.shape
        print([d.dim_value or d.dim_param for d in shape.dim])
  compile:
    - label: 'Parsowanie, kwantyzacja i kompilacja'
      language: python
      code: >
        from pathlib import Path


        import numpy as np

        from hailo_sdk_client import ClientRunner

        from PIL import Image


        ONNX = "weights/LibreYOLOXs.onnx"

        HW_ARCH = "hailo8"     # hailo8 | hailo8l | hailo10h

        IMGSZ = 640


        runner = ClientRunner(hw_arch=HW_ARCH)


        # Dla YOLOX pierwszą translację wykonaj bez end_node_names: log DFC

        # wypisze proponowane węzły końcowe. Uruchom ponownie z nimi.

        runner.translate_onnx_model(ONNX)


        # Normalizacja musi odpowiadać wstępnemu przetwarzaniu w LibreYOLO.
        YOLOX

        # i YOLO9 nie potrzebują średniej ani odchylenia standardowego, tylko

        # skali z 0-255 na 0-1.

        script = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0,
        255.0])\n"


        # Opcjonalnie: NMS może przejąć Hailo. Konfiguracja zależy zarówno od

        # liczby klas, jak i od rozmiaru wejścia, więc konfiguracja dla COCO-80

        # jest błędna dla dostrojonego modelu trzyklasowego. Bez tej linii HEF

        # zwraca surowe tensory z głowicy, a dekoduje je aplikacja.

        # script += 'nms_postprocess("yolox_nms_config.json", meta_arch=yolox,
        engine=cpu)\n'


        runner.load_model_script(script)


        # Obrazy kalibracyjne muszą być reprezentatywne dla danych
        wdrożeniowych.

        # Losowe obrazy skompilują się i po cichu zniszczą dokładność.

        calib_paths = sorted(Path("calib_images").glob("*.jpg"))[:128]

        calib = np.stack([
            np.asarray(
                Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ)),
                dtype=np.float32,
            )
            for p in calib_paths
        ])


        runner.optimize(calib)

        Path("libreyoloxs.hef").write_bytes(runner.compile())
    - label: Węzły końcowe dla YOLO9
      language: python
      code: |
        # Grafy LibreYOLO używają prefiksu "/head/...", a nie prefiksu "model.N"
        # znanego z konfiguracji pisanych pod inne eksporty. Skopiowana
        # konfiguracja nie będzie pasować. Jeśli parsowanie zawiedzie, sprawdź
        # nazwy we własnym grafie.
        END_NODES = [
            "/head/cv2.0/cv2.0.2/Conv", "/head/cv3.0/cv3.0.2/Conv",
            "/head/cv2.1/cv2.1.2/Conv", "/head/cv3.1/cv3.1.2/Conv",
            "/head/cv2.2/cv2.2.2/Conv", "/head/cv3.2/cv3.2.2/Conv",
        ]
        runner.translate_onnx_model(ONNX, end_node_names=END_NODES)
  device:
    - label: Raspberry Pi 5 z AI Kit lub AI HAT+
      language: bash
      code: >
        sudo apt install dkms hailo-all

        hailortcli fw-control identify       # kontrola urządzenia, podaje też
        architekturę

        hailortcli run libreyoloxs.hef       # test dymny i przepustowość
source_hash: 33b077f1c23d5535
---

## Instalacja

W bibliotece LibreYOLO nie ma `format="hef"` i nie będzie. Hailo Dataflow
Compiler to zamknięty SDK dystrybuowany jako prywatny plik wheel dostępny po
rejestracji w Developer Zone, więc nie może być zależnością ani opcjonalnym
dodatkiem (extra). Wdrożenie ma dwa etapy: LibreYOLO zapisuje statyczny plik
ONNX, a uruchomienie na nim DFC pozostaje po stronie użytkownika.

```text
Libre<Model>.pt  ->  ONNX  ->  HAR (parse)  ->  HAR (quantize INT8)  ->  HEF
                 [libreyolo]           [Hailo DFC, installed by you]
```

<code-tabs name="install" />

## Eksport

<code-tabs name="export" />

Nie należy przekazywać `half=True`. DFC przyjmuje plik ONNX w FP32 i wykonuje
własną kwantyzację INT8. Nie należy też przekazywać `nms=True`: NMS przejmuje
albo Hailo przez `nms_postprocess`, albo aplikacja, a podgraf NMS jest martwym
balastem za węzłami końcowymi. Domyślny opset działa; jeśli parser DFC
zaprotestuje, wyeksportuj ponownie z `opset=11`.

DFC tnie graf w podanych węzłach końcowych, którymi są konwolucje głowicy
detekcji, i odrzuca wszystko poniżej. Zwykły zdekodowany plik ONNX z LibreYOLO
jest więc akceptowalnym wejściem: końcówka dekodująca jest po prostu ignorowana
przez parser.

## Kompilacja

<code-tabs name="compile" />

Wartość `hw_arch` dobiera się do celu: `hailo8` dla Hailo-8, dla AI HAT+ 26 TOPS
oraz modułów M.2 i PCIe; `hailo8l` dla Hailo-8L, Raspberry Pi AI Kit i AI HAT+ 13
TOPS; `hailo10h` dla Hailo-10H, który wymaga odpowiednio nowszego DFC i Model
Zoo. W razie wątpliwości odpowiedzi udziela `hailortcli fw-control identify`
uruchomione na urządzeniu.

Dwie rodziny odwzorowują się na metaarchitekturę NMS z HailoRT, więc Hailo może
przejąć supresję wewnątrz skompilowanego pipeline'u: YOLOX przez
`meta_arch=yolox`, a YOLO9 przez metaarchitekturę Hailo z rozdzieloną głowicą,
której układ głowicy jest identyczny. Odpowiednią konfigurację `nms_postprocess`
należy wziąć z Hailo Model Zoo i dostosować do własnej liczby klas oraz rozmiaru
wejścia. Każdy inny detektor konwolucyjny kompiluje się jako graf bez pasującej
metaarchitektury: HEF zwraca surowe tensory z głowicy, a dekodowanie i NMS
wykonuje aplikacja na CPU.

Gdy coś zawiedzie, warto zachować log kompilacji. Każda poprawka opiera się na
dokładnej nazwie warstwy lub operatora, który zawiódł.

## Uruchomienie artefaktu

<code-tabs name="device" />

Inferencja w aplikacji korzysta z API `hailo_platform` w Pythonie. Gdy
`nms_postprocess` jest wkompilowane, wyjściem jest
`(batch, num_classes, max_dets, 5)` niosące `[y1, x1, y2, x2, score]` we
współrzędnych modelu, które trzeba samodzielnie przeskalować z powrotem do
obrazu źródłowego. Pipeline `Results` z LibreYOLO nie bierze udziału w czasie
działania; HEF jest samodzielnym artefaktem, a wstępne i końcowe przetwarzanie
należy do aplikacji.

## Ograniczenia

To, czy model może celować w Hailo-8 lub Hailo-8L, jest właściwością jego
architektury, a nie nazwy, więc poniższa reguła dotyczy także rodzin dodanych po
powstaniu tej strony.

Model nie skompiluje się, jeśli zawiera którykolwiek z tych elementów:

- Mechanizmy uwagi dowolnego rodzaju: self, cross, deformable czy windowed. To
  wyklucza każdy detektor w stylu DETR, każdy detektor z otwartym słownikiem lub
  warunkowany tekstem, każdy backbone ViT oraz każdą wieżę językową lub
  wizyjno-językową. Katalog modeli samego Hailo zawiera kilka ręcznie
  dostrojonych plików HEF z transformerami; to praca wykonana przez dostawcę pod
  konkretny przypadek, a nie dowód, że skompiluje się dowolny graf z uwagą.
- Dynamiczne kształty lub przepływ sterowania zależny od danych. DFC kompiluje
  jeden stały kształt wejścia i statyczny graf, więc zmienna liczba zapytań,
  prompty tekstowe, dynamiczne top-k, `NonZero`, `Gather` lub `TopK` z
  dynamicznymi indeksami oraz `grid_sample` odpadają.
- Projekt zdominowany przez LayerNorm lub GELU. BatchNorm składa się czysto z
  konwolucjami; wsparcie dla LayerNorm jest słabe, a GELU nie jest natywną
  funkcją aktywacji, więc stos w stylu ConvNeXt pasuje źle, choć nominalnie jest
  konwolucyjny.
- Praca obraz na obraz w natywnej rozdzielczości. Modele rekonstrukcji działają
  w pełnej rozdzielczości wejścia i przekraczają praktyczne budżety pamięci SRAM
  w Hailo.

Rodzina jest kandydatem, gdy składa się wyłącznie z konwolucji, używa BatchNorm
z ReLU lub SiLU i ma stały rozmiar wejścia. W tej bibliotece oznacza to
jednoetapowe detektory CNN, z YOLOX i YOLO9 jako głównymi celami; inne detektory
konwolucyjne, takie jak PicoDet, YOLO-NAS i RTMDet, z dekodowaniem po stronie
aplikacji; klasyfikatory CNN ResNet, MobileNetV4-conv i EfficientNetV2, spośród
których ResNet jest wspierany najlepiej, bo Hailo Model Zoo dostarcza dla niego
gotowe recepty; oraz małe konwolucyjne głowice zadaniowe, takie jak detekcja
punktów FOMO i estymacja spojrzenia L2CS na backbone ResNet, które w zasadzie da
się skompilować, ale nie mają recepty Hailo.

Jedno zastrzeżenie co do statusu, będące powodem, dla którego nic na tej stronie
nie jest przedstawiane jako wspierane: żadna rodzina LibreYOLO nie została
przeprowadzona od początku do końca przez DFC aż do działającego HEF. Powyższe
reguły przewidują kompilowalność na podstawie architektury. Zachowanie parsera,
kwantyzacja i dokładność pozostają niepotwierdzone, dopóki HEF nie zostanie
skompilowany i zmierzony, więc każdego kandydata należy traktować jako
wymagającego własnych, zapisanych dowodów: skompilowanego HEF z dokładnie tego
checkpointu, z odnotowanymi wersjami DFC, Model Zoo i HailoRT, udokumentowanej
kalibracji oraz porównania dokładności na urządzeniu z bazową wersją FP32,
zamiast liczby opisującej przepustowość.

Jeśli model jest zdyskwalifikowany, alternatywą są środowiska uruchomieniowe z
odnotowaną zgodnością: [ONNX](/docs/export/onnx),
[TensorRT](/docs/export/tensorrt) i [OpenVINO](/docs/export/openvino).
