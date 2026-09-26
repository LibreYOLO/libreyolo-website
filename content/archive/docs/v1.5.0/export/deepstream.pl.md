---
title: NVIDIA DeepStream
seo_title: Uruchamianie modeli YOLO na NVIDIA DeepStream
description: >-
  Eksport modelu LibreYOLO na NVIDIA DeepStream: graf ONNX oraz wygenerowana
  konfiguracja nvinfer. Dokładne polecenia do zbudowania parsera i do
  uruchomienia pipeline'u.
lead: >-
  NVIDIA DeepStream uruchamia inferencję przez element nvinfer, który potrzebuje
  grafu ONNX, pasującego pliku konfiguracyjnego i parsera ramek ograniczających.
  Ustawienie deepstream=True przy eksporcie do ONNX zapisuje dwa pierwsze i
  podłącza je do trzeciego.
keywords:
  - NVIDIA DeepStream
  - DeepStream YOLO
  - nvinfer
  - parser bounding box deepstream
  - config_infer_primary
  - NvDsInferParseYolo
  - deepstream-app
  - TensorRT engine
  - yolo na jetson
meta:
  - label: Flaga
    value: 'export(format="onnx", deepstream=True)'
    mono: true
  - label: Zapisuje
    value: 'Graf ONNX, config_infer_primary_<stem>.txt i <stem>_labels.txt'
  - label: Zakres
    value: 43 kombinacje rodziny i zadania w dziewięciu zadaniach
  - label: Parser
    value: >-
      NvDsInferParseYolo, z projektu DeepStream-Yolo na licencji MIT autorstwa
      Marcosa Luciano. Budowany raz na urządzenie.
    links:
      - label: github.com/marcoslucianops/DeepStream-Yolo
        href: 'https://github.com/marcoslucianops/DeepStream-Yolo'
  - label: Dostępność
    value: Dostępne w v1.5.0. Scalone do dev 2026-08-08 w pull request 728.
    links:
      - label: pull request 728
        href: 'https://github.com/LibreYOLO/libreyolo/pull/728'
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
  - label: Walidacja w środowisku uruchomieniowym
    value: 'DeepStream 8.0.0 na RTX 5070 Ti, tylko detekcja, 2026-08-08'
verification: >-
  Napisane na podstawie walidacji w środowisku uruchomieniowym z 2026-08-08.
  Listy rodzin, klucze konfiguracji i wartości domyślne odczytane z
  libreyolo/export/deepstream.py oraz libreyolo/export/exporter.py przy commicie
  5f81e11e, który tego samego dnia został scalony do dev w pull request 728.
snippets:
  install:
    - label: Instalacja
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO9, LibreDFINE


        # Zapisuje libreyolo9s.onnx, config_infer_primary_libreyolo9s.txt

        # oraz libreyolo9s_labels.txt w katalogu roboczym.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="onnx",
        deepstream=True)


        # Każdy model detekcji trzymaj w osobnym katalogu: każda konfiguracja

        # detekcji wskazuje ten sam plik pamięci podręcznej silnika. Zobacz
        „Znane pułapki”.

        LibreDFINE("LibreDFINEs.pt", size="s").export(format="onnx",
        deepstream=True)
    - label: Argumenty
      language: python
      code: >
        model.export(
            format="onnx",     # deepstream=True jest odrzucane dla każdego innego formatu
            deepstream=True,
            conf=0.25,         # ustawia pre-cluster-threshold (oraz classifier-threshold,
                               # segmentation-threshold w tych zadaniach)
            iou=0.45,          # ustawia nms-iou-threshold, pomijane przy cluster-mode=4
            batch=1,           # ustawia batch-size i nazwę pliku pamięci podręcznej silnika
            half=False,        # True zapisuje w konfiguracji network-mode=2 (kompilacja fp16)
            int8=False,        # True zapisuje w konfiguracji network-mode=1
            dynamic=True,      # dynamiczna oś batcha w grafie ONNX
            imgsz=640,         # ustawia infer-dims=3;H;W
        )


        # deepstream=True i nms=True wzajemnie się wykluczają: DeepStream
        wykonuje

        # tłumienie na etapie klastrowania, więc nic nie jest osadzane w grafie.
    - label: Najpierw pobranie wag D-FINE
      language: bash
      code: |
        curl -L -o LibreDFINEs.pt \
          https://huggingface.co/LibreYOLO/LibreDFINEs/resolve/main/LibreDFINEs.pt
  gpu:
    - label: Sprawdzenie przekazania GPU przed wszystkim innym
      language: bash
      code: |
        docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.04-py3 \
          nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
      expect: |
        name, driver_version, compute_cap
        NVIDIA GeForce RTX 5070 Ti, 591.86, 12.0
  parser:
    - label: 'build_parser.sh, uruchamiany w kontenerze DeepStream'
      language: bash
      code: >
        set -e

        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo.git


        # /usr/local/cuda-12 w tym obrazie to zaślepka i kompilacja się na niej
        wywraca z

        # komunikatem "fatal error: crt/host_defines.h: No such file or
        directory". Trzeba

        # wskazać toolkit, który faktycznie zawiera ten nagłówek; w obrazie 8.0
        jest to cuda-12.5.

        CUDA_DIR=$(readlink -f /usr/local/cuda)

        [ -f "$CUDA_DIR/include/crt/host_defines.h" ] || \
          CUDA_DIR=$(ls -d /usr/local/cuda-*.* | sort -Vr | \
                     while read d; do [ -f "$d/include/crt/host_defines.h" ] && echo "$d" && break; done)

        # Obraz zawiera libcublas.so.12 i libcublas.so.12.8.4.1, ale nie zawiera

        # nieoznaczonego wersją libcublas.so, którego wymaga -lcublas, więc
        linkowanie

        # kończy się błędem "/usr/bin/ld: cannot find -lcublas". Podaj linkerowi
        nazwy,

        # których oczekuje.

        mkdir -p /tmp/cudalibs

        for lib in cublas cublasLt cudart; do
          real=$(find /usr/local -name "lib${lib}.so.1*" | grep -v stubs | sort -V | tail -1)
          ln -sf "$real" "/tmp/cudalibs/lib${lib}.so"
        done

        export LIBRARY_PATH="/tmp/cudalibs:$LIBRARY_PATH"


        make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
        CUDA_VER="${CUDA_DIR##*/cuda-}"
    - label: Segmentacja instancji używa innego parsera
      language: bash
      code: >
        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git

        make -C DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg \
          CUDA_VER="${CUDA_DIR##*/cuda-}"
  run:
    - label: deepstream_app_config.txt
      language: text
      code: >
        [application]

        enable-perf-measurement=1

        perf-measurement-interval-sec=5

        gie-kitti-output-dir=kitti


        [tiled-display]

        enable=0


        [source0]

        enable=1

        type=3

        uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

        num-sources=1

        gpu-id=0


        [streammux]

        gpu-id=0

        batch-size=1

        batched-push-timeout=40000

        width=1920

        height=1080

        live-source=0


        [primary-gie]

        enable=1

        gpu-id=0

        gie-unique-id=1

        config-file=config_infer_primary_libreyolo9s.txt


        [osd]

        enable=1

        border-width=2

        text-size=15


        [sink0]

        enable=1

        type=1

        sync=0


        [tests]

        file-loop=0
    - label: Uruchomienie
      language: bash
      code: |
        deepstream-app -c deepstream_app_config.txt
      expect: |
        App run successful
    - label: Oba kroki w jednym kontenerze
      language: bash
      code: |
        docker run --rm --gpus all -v "$PWD:/work" -w /work \
          nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
          bash -c "bash build_parser.sh && deepstream-app -c deepstream_app_config.txt"
source_hash: 1ee91c265753dd9a
---

## Dostępność

Eksport do DeepStream jest dostępny od wersji v1.5.0. Został scalony do gałęzi
`dev` 2026-08-08 w pull request 728, więc aktualna instalacja go zawiera i nie
trzeba przypinać żadnej gałęzi.

<code-tabs name="install" />

Jeśli gałąź `deepstream-export` została sklonowana przed 2026-08-08, należy ją
wymienić. Ta gałąź została zrebasowana i wypchnięta z wymuszeniem, a starsza
historia nie zawiera poprawki, bez której te eksporty w ogóle nie uruchomią się
na maszynie z CUDA.

## Co zapisuje eksport

`model.export(format="onnx", deepstream=True)` zapisuje obok siebie trzy pliki.
Dla `libreyolo9s.pt`:

- `libreyolo9s.onnx`, graf detekcji, jeden tensor wyjściowy o kształcie
  `(batch, num_detections, 6)`, każdy wiersz to `[x1, y1, x2, y2, score, class_id]`
  we współrzędnych pikselowych wejścia sieci.
- `config_infer_primary_libreyolo9s.txt`, konfiguracja `nvinfer` zawierająca
  stałe preprocessingu danej rodziny, liczbę klas, progi i podłączenie parsera.
- `libreyolo9s_labels.txt`, jedna nazwa klasy w wierszu.

Plik etykiet powstaje zawsze wtedy, gdy checkpoint zawiera nazwy klas. Modele
głębi ich nie mają, więc nie dostają ani tego pliku, ani klucza `labelfile-path`.

Biblioteka LibreYOLO nie generuje pliku `.so`. Plik `.so`, który ładuje
DeepStream, to parser ramek ograniczających z `marcoslucianops/DeepStream-Yolo`,
kompilowany raz na urządzenie, i jest to ten sam plik binarny niezależnie od
tego, na który detektor LibreYOLO zostanie skierowany. Modelem jest plik ONNX.
Klasyfikacja i segmentacja semantyczna nie potrzebują parsera w ogóle, ponieważ
`nvinfer` sam wykonuje dla nich postprocessing.

## Eksport modelu

<code-tabs name="export" />

`LibreDFINE._load_weights` zgłasza `FileNotFoundError`, gdy pliku nie ma jeszcze
na dysku, i nie próbuje go pobrać, więc `LibreDFINEs.pt` trzeba pobrać wcześniej
samodzielnie. Ta luka jest śledzona jako
[issue #727](https://github.com/LibreYOLO/libreyolo/issues/727). Wagi YOLO9
pobierają się przy pierwszym użyciu.

Flaga działa tylko w Pythonie. `libreyolo export` w tej gałęzi nie ma opcji
`deepstream`, a CLI buduje argumenty eksportu ze stałej listy, zamiast
przepuszczać nieznane klucze.

## Budowa parsera ramek ograniczających

Detekcja wymaga biblioteki parsera, segmentacja instancji wymaga innej, a
pozostałe zadania nie wymagają żadnej. Dwie rzeczy w obrazie DeepStream 8.0
psują udokumentowane polecenie budowania i obie są problemami środowiska, a nie
biblioteki LibreYOLO.

Obraz zawiera `cuda`, `cuda-12`, `cuda-12.5`, `cuda-12.8` i `cuda-12.9` w
`/usr/local`. Kompletny toolkit ma tylko `cuda-12.5`. Zawiera też
`libcublas.so.12` i `libcublas.so.12.8.4.1`, ale nie nieoznaczony wersją
`libcublas.so`, do którego odwołuje się `-lcublas`. Poniższy skrypt obchodzi oba
problemy.

<code-tabs name="parser" />

Następnie należy skierować `custom-lib-path` w wygenerowanej konfiguracji na
zbudowany `libnvdsinfer_custom_impl_Yolo.so`. Wygenerowaną wartością jest
ścieżka względna `nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`,
która działa, gdy `deepstream-app` jest uruchamiany z katalogu repozytorium
`DeepStream-Yolo`, a poza tym przypadkiem wymaga zmiany.

## Uruchomienie pipeline'u

Zanim poświęci się czas na cokolwiek innego, warto sprawdzić, czy kontener widzi
GPU. To pierwsza kontrola, jaką wykonał przebieg walidacyjny, na karcie
Blackwell pod WSL2.

<code-tabs name="gpu" />

Przebieg walidacyjny sterował `deepstream-app` z jednym źródłem plikowym, bez
sinka wyświetlania, z włączonym on-screen display i ustawionym
`gie-kitti-output-dir`, tak aby detekcje z każdej klatki trafiały na dysk jako
tekst KITTI. Konfiguracja z tymi ustawieniami:

<code-tabs name="run" />

`nvinfer` buduje silnik TensorRT z pliku ONNX przy pierwszym uruchomieniu i
zapisuje go w pamięci podręcznej obok modelu, więc pierwsze uruchomienie płaci
za budowę silnika, a kolejne wczytują pamięć podręczną.

## Wygenerowana konfiguracja

Obie poniższe konfiguracje zostały zapisane przez eksporter na potrzeby
przebiegu walidacyjnego i nie były później edytowane.

| Klucz | YOLO9-s | D-FINE-s |
|---|---|---|
| `net-scale-factor` | 0.003921568627 | 0.003921568627 |
| `model-color-format` | 0 | 0 |
| `infer-dims` | 3;640;640 | 3;640;640 |
| `maintain-aspect-ratio` | 1 | 0 |
| `symmetric-padding` | 0 | 0 |
| `network-type` | 0 | 0 |
| `num-detected-classes` | 80 | 80 |
| `cluster-mode` | 2 | 4 |
| `parse-bbox-func-name` | NvDsInferParseYolo | NvDsInferParseYolo |
| `pre-cluster-threshold` | 0.25 | 0.25 |
| `nms-iou-threshold` | 0.45 | |
| `topk` | 300 | 300 |

Obie konfiguracje różnią się w trzech miejscach: `maintain-aspect-ratio`,
`cluster-mode` oraz to, czy `nms-iou-threshold` w ogóle występuje. Konfiguracja
D-FINE pomija ten klucz całkowicie, czego wymaga `cluster-mode=4`.

Głowice, które zwracają najwyżej jedną predykcję na obiekt, dostają
`cluster-mode=4`, więc DeepStream nie klastruje ich wyników; klastrowanie
scalałoby faktycznie odrębne detekcje. Dotyczy to `rfdetr`, `dfine`, `deim`,
`deimv2`, `ec`, `rtdetr`, `rtdetrv2`, `rtdetrv4` i `yolo9_e2e`. Głowice siatkowe
i kotwicowe dostają `cluster-mode=2` oraz `nms-iou-threshold`.

Konfiguracje detekcji zawierają też
`engine-create-func-name=NvDsInferYoloCudaEngineGet`, co przekazuje budowanie
silnika bibliotece parsera. To właśnie ustala na stałe nazwę pliku pamięci
podręcznej silnika i to jest źródłem kolizji opisanej w znanych pułapkach.

## Obsługiwane zadania i rodziny

Eksportuje się czterdzieści trzy kombinacje rodziny i zadania.
`deepstream_supported_tasks()` i `deepstream_supported_families(task)` w
`libreyolo/export/deepstream.py` zwracają te same listy w czasie działania.

| Zadanie | `network-type` | Biblioteka parsera | Rodziny |
|---|---|---|---|
| Detekcja | 0 | DeepStream-Yolo | yolo9, yolo9_p2, yolo9_e2e, yolo1, yolo2, yolo3, yolo4, yolo7, yolox, yolonas, rtmdet, picodet, rfdetr, dfine, deim, deimv2, ec, rtdetr, rtdetrv2, rtdetrv4 |
| Klasyfikacja | 1 | Niepotrzebna | mobilenetv4, convnext, efficientnetv2, resnet, dinov2 |
| Segmentacja semantyczna | 2 | Niepotrzebna | pidnet, eomt, dinov2, lingbotvision |
| Segmentacja instancji | 3 | DeepStream-Yolo-Seg | rfdetr, dfine, ec |
| Poza | 100 | Niepotrzebna | yolo9, yolonas, rfdetr, ec |
| Głębia | 100 | Niepotrzebna | depth_anything, zipdepth |
| Restauracja | 100 | Niepotrzebna | nafnet, realesrgan, swinir |
| Matting | 100 | Niepotrzebna | birefnet |
| Spojrzenie | 100 | Niepotrzebna | l2cs |

`network-type=100` oznacza, że DeepStream nie ma postprocesora dla danego
zadania. Takie konfiguracje ustawiają `output-tensor-meta=1`, natywne wyjścia
grafu przechodzą nietknięte, a aplikacja dekoduje je z metadanych tensora. Grafy
z wieloma wyjściami są tu w porządku: każda warstwa wyjściowa trafia do
metadanych z tymi samymi nazwami wyjść i dynamicznymi osiami co przy zwykłym
eksporcie do ONNX.

Wiersze segmentacji instancji to wiersz detekcji, po którym następuje maska danej
instancji, spłaszczona przy `(netH / 4, netW / 4)`, czyli w rozdzielczości
zapisanej na stałe w parserze segmentacji, jako prawdopodobieństwa dla
`segmentation-threshold`.

Klasyfikacja i spojrzenie działają jako inferencja wtórna. Aby umieścić
klasyfikator za detektorem, należy ustawić `process-mode=2` i
`operate-on-gie-id` w wygenerowanej konfiguracji. Spojrzenie to kontrakt
obejmujący samą głowicę, jeden wycinek twarzy na wejście, więc wymaga przed sobą
detektora twarzy.

Trzy rodziny są nieobecne celowo. `segformer` nie jest podłączony do wspólnego
kontraktu eksportu semantycznego i nie eksportuje się do ONNX w żadnej postaci.
RTMDet-Ins i YOLO9 mają eksport segmentacji instancji zablokowany w samej
bibliotece LibreYOLO. `depth_anything3` nie ma implementacji eksportu.

Za dwoma wierszami tabeli kryją się braki checkpointów. Opublikowany jest tylko
semantyczny checkpoint EoMT w rozmiarze `l`, a klasyfikacja DINOv2 nie ma
opublikowanego checkpointu w ogóle, więc ta kombinacja wymaga własnych
dostrojonych wag.

## Różnice w preprocessingu

`nvinfer` oblicza `net-scale-factor * (x - offsets)` dla każdego kanału ze
skalarną skalą, co nie pozwala wyrazić odchylenia standardowego osobno dla
kanału. Rodziny, które go potrzebują (`rfdetr`, `ec`, rozmiary `deimv2` z
backbone DINO, `rtmdet`, `picodet` oraz wszystkie rodziny klasyfikacyjne), mają
normalizację wbudowaną w eksportowany graf, a wygenerowana konfiguracja podaje
grafowi pasującą surową przestrzeń wejściową.

Geometria to miejsce, w którym własne pipeline'y LibreYOLO w Pythonie i
`nvinfer` nadal się rozjeżdżają:

- Rodziny stosujące letterbox (`yolo9`, `yolox`, `yolonas`, `rtmdet`, `yolo2`,
  `yolo3`, `yolo4`, `yolo7`) natywnie dopełniają szarością. `nvinfer` dopełnia
  czernią.
- Detekcja `yolonas` natywnie skaluje najdłuższy bok do 636 wewnątrz płótna 640.
  `maintain-aspect-ratio` w `nvinfer` używa pełnych 640.
- Klasyfikacja natywnie skaluje najkrótszy bok, a potem wycina środek. `nvinfer`
  rozciąga klatkę lub ROI obiektu do wejścia sieci, więc ciasno wykadrowane
  obiekty wypadają inaczej.
- EoMT natywnie przetwarza kafelki metodą przesuwnego okna na potrzeby
  segmentacji semantycznej. Eksportowany graf to jedno rozciągnięte płótno, co
  jest szybsze i mniej dokładne.
- `pidnet` zwraca mapę klas w 1/8 rozdzielczości wejścia, a `lingbotvision` w
  1/16. DeepStream skaluje mapę klas w górę na potrzeby wyświetlania.

Bramka zgodności ONNX podaje już wstępnie przetworzone tensory, więc sprawdza
wyjścia grafu i nie wykryje złej kolejności kanałów ani złej polityki
dopełniania w konfiguracji. Przed wdrożeniem zadania wymagającego dokładnej
zgodności należy zwalidować wyniki na własnych danych.

## Znane pułapki

### Dwa modele detekcji w jednym katalogu wczytują nawzajem swoje silniki

Każda konfiguracja detekcji zawiera ten sam wiersz:

```ini
model-engine-file=model_b1_gpu0_fp32.engine
```

Konstruktor silnika w parserze wymaga tej nazwy pliku i nie zmienia się ona
zależnie od modelu. Po wyeksportowaniu drugiego modelu detekcji do tego samego
katalogu drugie uruchomienie wczytuje silnik z pamięci podręcznej pierwszego
modelu. Nic się nie wywraca; po prostu ramki są błędne. Każdy model detekcji
powinien mieć własny katalog. Przebieg walidacyjny musiał najpierw odizolować
D-FINE do osobnego katalogu, zanim w ogóle dało się go przetestować.

### Ramka może nieść tylko jedną klasę

Format wiersza w `nvinfer` to `[x1, y1, x2, y2, score, class_id]`, jedna klasa
na ramkę, więc eksport sprowadza wyniki klas do ich argmax. Ramka, którą
`predict` zgłasza pod dwiema klasami, przetrwa pod jedną. Zmierzony przypadek:
LibreYOLO zgłasza `vase 0.773` i `bottle 0.383` na tej samej ramce, a graf
DeepStream zachowuje `vase`. Wynika to z formatu wiersza w parserze i nie da się
tego zmienić bez porzucenia tego kontraktu, więc jest to zachowanie oczekiwane,
a nie regresja.

## Zwalidowane

`deepstream-app` dobiegł do EOS z komunikatem `App run successful` na obu typach
głowic detektora, na dołączonym przez NVIDIA pliku `sample_1080p_h264.mp4`
(1443 klatki), z włączonymi zrzutami KITTI dla każdej klatki.

| | YOLO9-s | D-FINE-s |
|---|---|---|
| Typ głowicy | siatkowa | jeden do jednego |
| `cluster-mode` | 2 | 4 |
| `maintain-aspect-ratio` | 1 | 0 |
| Klatki z detekcjami | 1443 | 1443 |
| Łączna liczba detekcji | 18031 | 71105 |

Histogramy klas ze wszystkich 1443 klatek stawiają dla obu modeli samochody na
pierwszym miejscu, a ludzi na drugim, co jest prawidłowe dla sceny ulicznej.
Czterokrotna różnica w liczbie detekcji to działanie różnicy w `cluster-mode`:
D-FINE przy `cluster-mode=4` nie klastruje wyników, więc przetrwa każde
zapytanie powyżej progu, wraz z niemal duplikatami.

Dwa niezależnie wytrenowane modele umieszczają dominujący obiekt w tym samym
miejscu:

```text
YOLO9  bus  [706.72,  0.82, 1916.34, 1062.97]  conf 0.965
D-FINE bus  [702.73,  2.93, 1916.24, 1069.32]  conf 0.965
```

Ten przebieg dowodzi pięciu rzeczy: TensorRT buduje silnik z wyeksportowanego
pliku ONNX na sm_120, `nvinfer` akceptuje każdy klucz w wygenerowanej
konfiguracji, `NvDsInferParseYolo` poprawnie odczytuje układ tensora, ramki
trafiają we współrzędne źródłowej rozdzielczości 1920x1080, a etykiety
rozwiązują się względem wygenerowanego pliku etykiet.

Środowisko, w którym to działało:

| Komponent | Wartość |
|---|---|
| System hosta | Windows 11 Pro 26200 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| Sterownik | 591.86 |
| Compute capability | 12.0 (Blackwell, sm_120) |
| Środowisko uruchomieniowe kontenerów | Docker Desktop 29.4.3, backend WSL2 |
| Obraz DeepStream | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` |
| Wersja DeepStream | 8.0.0 |
| CUDA w kontenerze | 12.8.1 |
| Parser | `marcoslucianops/DeepStream-Yolo` na HEAD |

Poza przebiegiem pipeline'u `tests/unit/test_deepstream_export.py` pokrywa
adaptery grafu i klucze wygenerowanej konfiguracji, a jego 35 testów przechodzi
na tym commicie.

## Niezwalidowane

Wymienione po to, aby powyższy zakres nie był czytany szerzej, niż jest.

- Jetson i aarch64. Kontrakt eksportu nie zależy od architektury, ale pipeline
  uruchomiono wyłącznie na dedykowanym GPU x86.
- Czterdzieści jeden z 43 kombinacji. Przez DeepStream przeszła tylko detekcja z
  `yolo9` i detekcja z `dfine`. Klasyfikacja, segmentacja semantyczna,
  segmentacja instancji i zadania z surowymi tensorami są pokryte testami
  jednostkowymi i sprawdzeniami zgodności ONNX, a nie przebiegiem pipeline'u.
- FP16 i INT8. Sprawdzono tylko `network-mode=0`.
- Wiele strumieni i batchowanie. Jedno źródło, `batch-size=1`.
- Dokładność względem zbioru danych referencyjnych (ground truth). Detekcje
  sprawdzono pod kątem sensowności semantycznej i zgodności między modelami, nie
  oceniano ich jako mAP przez DeepStream.
