---
title: NVIDIA Jetson
seo_title: Instalacja LibreYOLO i PyTorch na NVIDIA Jetson
description: >-
  Instalacja LibreYOLO na NVIDIA Jetson: cztery biblioteki CUDA pomijane przez
  JetPack, krok --no-deps wymagany przez PyTorch i zmierzone wyniki z Orin Nano.
lead: >-
  Płytki NVIDIA Jetson uruchamiają LibreYOLO na standardowych pakietach wheel
  PyTorch dla aarch64. Nie jest do tego potrzebna żadna kompilacja torch
  specyficzna dla Jetson, ale JetPack pomija cztery biblioteki, z którymi torch
  się linkuje, i instalacja musi je dostarczyć.
keywords:
  - NVIDIA Jetson
  - Jetson Orin Nano
  - JetPack 7.2
  - instalacja pytorch na jetson
  - nvidia-cudnn-cu13
  - nvidia-nccl-cu13
  - nvidia-cusparselt-cu13
  - nvidia-nvshmem-cu13
  - torch.cuda.is_available
  - no kernel image is available for execution on the device
  - tensorrt na jetson
  - pytorch aarch64 wheel
last_verified: 1.4.0
meta:
  - label: Płytka
    value: 'Jetson Orin Nano Super Developer Kit, 8 GB, GPU compute capability 8.7'
  - label: Platforma
    value: 'JetPack 7.2 (L4T R39.2), Ubuntu 24.04, CUDA 13, Python 3.12.3, aarch64'
  - label: Testowany stos
    value: >-
      libreyolo 1.4.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, opencv
      5.0.0, numpy 2.5.1, dnia 2026-07-27
  - label: Brakuje w JetPack
    value: >-
      nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-cusparselt-cu13,
      nvidia-nvshmem-cu13
    mono: true
  - label: Benchmarki
    value: >-
      223 zweryfikowane uruchomienia na tej płytce, 58 modeli z 12 rodzin, w
      PyTorch, ONNX Runtime i TensorRT
    links:
      - label: visionanalysis.org/hardware/jetson_orin
        href: 'https://www.visionanalysis.org/hardware/jetson_orin'
  - label: Śledzone w
    value: Część zgłoszenia 648 poświęcona Jetson
    links:
      - label: zgłoszenie 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
verification: >-
  Przepis instalacji i oczekiwane wyjście pochodzą z uruchomienia instalacji z
  2026-07-27 na Jetson Orin Nano Super. Wiersze opóźnienia i dokładności
  pochodzą ze zrzutu zweryfikowanych wyników stojącego za visionanalysis.org,
  przefiltrowanego do sprzętu jetson_orin, pomiar z czerwca 2026 na libreyolo
  1.2.0.dev0. Zachowanie eksportu i loadera odczytane z
  libreyolo/export/exporter.py, libreyolo/export/tensorrt.py i
  libreyolo/models/__init__.py.
snippets:
  prep:
    - label: Pakiety systemowe i wirtualne środowisko
      language: bash
      code: |
        # JetPack nie ma preinstalowanego pip ani modułu venv.
        sudo apt update
        sudo apt install -y python3.12-venv python3-pip

        python3 -m venv ~/libreyolo
        source ~/libreyolo/bin/activate
        pip install -U pip wheel setuptools
  torch:
    - label: PyTorch z indeksu pakietów wheel dla CUDA 13
      language: bash
      code: |
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu130 \
          --extra-index-url https://pypi.org/simple
    - label: 'Cztery biblioteki, których nie dostarcza JetPack'
      language: bash
      code: |
        pip install nvidia-cudnn-cu13 nvidia-nccl-cu13 \
                    nvidia-cusparselt-cu13 nvidia-nvshmem-cu13
    - label: 'Jeśli pip żąda cuda-toolkit 13.0.3, instalacja z --no-deps'
      language: bash
      code: >
        # --no-deps oznacza, że zależności Pythona dla torch też trzeba wymienić
        ręcznie.

        pip install --no-deps \
          torch torchvision \
          nvidia-cudnn-cu13 nvidia-nccl-cu13 \
          nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 \
          filelock typing_extensions sympy networkx jinja2 markupsafe mpmath \
          fsspec numpy pillow
  ldd:
    - label: Wskazanie kolejnej brakującej biblioteki zamiast zgadywania
      language: bash
      code: >
        ldd
        "$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" \
          | grep "not found"

        # Wszystko, czego wciąż brakuje we wszystkich bibliotekach torch, w
        jednym przebiegu:

        ldd "$VIRTUAL_ENV"/lib/python3.12/site-packages/torch/lib/*.so
        2>/dev/null \
          | grep "not found" | sort -u
  install:
    - label: 'Instalacja LibreYOLO po torch, a nie przed'
      language: bash
      code: >
        # torch jest już spełniony, więc pip zostawia kompilację CUDA na
        miejscu.

        pip install libreyolo


        # Dodatek ONNX jest potrzebny tylko do eksportu. Eksport do TensorRT
        idzie

        # przez ONNX, więc trzeba go dodać przed sekcją eksportu poniżej.

        pip install "libreyolo[onnx]"
  verify:
    - label: Wersje i urządzenie
      language: python
      code: |
        import cv2
        import numpy
        import torch

        import libreyolo

        print("torch", torch.__version__, "cuda", torch.cuda.is_available())
        print("gpu", torch.cuda.get_device_name(0))
        print("libreyolo", libreyolo.__version__)
        print("cv2", cv2.__version__, "numpy", numpy.__version__)
      expect: |
        torch 2.13.0+cu130 cuda True
        gpu Orin
        libreyolo 1.4.0
        cv2 5.0.0 numpy 2.5.1
    - label: Następnie uruchomienie prawdziwego kernela
      language: python
      code: |
        import torch

        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        # Przy pierwszym użyciu pobiera checkpoint.
        model = LibreYOLO9("libreyolo9s.pt", size="s")

        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict --source
        https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        --model libreyolo9s.pt --save
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreYOLO9, SAMPLE_IMAGE


        # Zapisuje libreyolo9s.onnx, a potem buduje z niego libreyolo9s.engine.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="tensorrt",
        half=True)


        # Silnik wczytuje się z powrotem przez ten sam punkt wejścia.

        result = LibreYOLO("libreyolo9s.engine").predict(SAMPLE_IMAGE)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model libreyolo9s.pt --format tensorrt --half
  power:
    - label: Tryb zasilania i zegary
      language: bash
      code: >
        sudo nvpmodel -q      # jakie tryby udostępnia ta płytka i który jest
        aktywny

        sudo nvpmodel -m 0    # najwyższy tryb na testowanej tutaj płytce

        sudo jetson_clocks


        tegrastats            # obciążenie na żywo; nvidia-smi jest ograniczone
        na Tegra
source_hash: c07ff908503e89b5
---

## Co dokumentuje ta strona

Ta strona dokumentuje jedną konfigurację zweryfikowaną od początku do końca, a
nie macierz wsparcia. Płytką był Jetson Orin Nano Super Developer Kit z 8 GB
pamięci, z systemem JetPack 7.2 (L4T R39.2, Ubuntu 24.04, CUDA 13, Python
3.12.3), a stos, który na nim wystartował, to `libreyolo 1.4.0` z
`torch 2.13.0+cu130`, OpenCV 5.0.0 i NumPy 2.5.1. `torch.cuda.is_available()`
zwróciło `True`, a GPU zgłosiło się jako `Orin`.

Inne wydania JetPack, inne płytki Jetson i inne wersje CUDA nie były testowane.
Poniższy przepis to ten, który zadziałał na tej kombinacji.

Tamto uruchomienie odbyło się 2026-07-27 na LibreYOLO 1.4.0 i nie zostało
powtórzone na sprzęcie z 1.5.0: to jedyna strona w drzewie 1.5.0, która wciąż
niesie weryfikację z 1.4.0, dlatego jej front matter podaje
`last_verified: "1.4.0"`. Nic w zmianach z 1.5.0 nie dotyka ścieżki instalacji,
czterech brakujących bibliotek ani opisanych tutaj flag eksportu, więc polecenia
powinny pozostać aktualne, ale numery wersji w poniższych wynikach to te, które
wypisała wersja 1.4.0, a nie pomiar z 1.5.0.

Dwie rzeczy w tej konfiguracji przeczą temu, co mówi większość poradników o
Jetson. Pakiety wheel to zwykłe kompilacje aarch64 publikowane dla CUDA 13, więc
żadna kompilacja torch specyficzna dla Jetson nie jest potrzebna. A JetPack nie
dostarcza czterech bibliotek, z którymi te pakiety się linkują, więc
`import torch` zawodzi po jednej bibliotece naraz, dopóki wszystkie cztery nie
zostaną zainstalowane.

## Instalacja

Obrazy JetPack przychodzą bez pip i bez modułu `venv`, więc oba idą jako
pierwsze.

<code-tabs name="prep" />

Płytka z 8 GB jest ciasna dla większych checkpointów. Dodanie swapu na NVMe
przed ich wczytaniem pozwala uniknąć zabicia procesu z braku pamięci w trakcie
uruchomienia.

Następnie PyTorch. Indeks CUDA 13 zawiera pakiety wheel dla aarch64; dodatkowy
indeks dostarcza zależności czysto pythonowe z PyPI.

<code-tabs name="torch" />

Cztery pakiety wheel `nvidia-*-cu13` to ta część, którą łatwo przeoczyć. JetPack
dostarcza sterownik GPU, a nie cuDNN, NCCL, cuSPARSELt czy NVSHMEM, i bez nich
torch odmawia importu. Instalacja wszystkich czterech naraz jest szybsza niż
odkrywanie ich po jednym wyjątku naraz.

Trzeci fragment obejmuje konkretną awarię: metadane zależności torch dla
kompilacji CUDA 13 wymagają `cuda-toolkit==13.0.3`, dla którego nie ma pakietu
wheel dla aarch64 na PyPI, więc rozwiązywanie zależności zawodzi, zanim
cokolwiek się pobierze. `--no-deps` pomija resolver, co oznacza, że każdą
zależność trzeba wymienić w wierszu poleceń.

LibreYOLO instaluje się na końcu. Instalacja go jako pierwszego pozwala, by pip
wybrał własny torch, który na tej platformie nie jest kompilacją CUDA.

<code-tabs name="install" />

Każda pozostała zależność rozwiązuje się do gotowego pakietu wheel dla aarch64,
w tym OpenCV, NumPy, SciPy, pycocotools i safetensors. Nic nie kompiluje się ze
źródeł.

## Sprawdzenie, czy CUDA działa

<code-tabs name="verify" />

Drugi fragment jest równie ważny jak pierwszy. Pakiet wheel zbudowany dla
niewłaściwej architektury GPU nadal zgłasza
`torch.cuda.is_available() == True`, a potem zawodzi przy pierwszej prawdziwej
operacji z komunikatem `CUDA error: no kernel image is available for execution
on the device`. Mnożenie macierzy na urządzeniu to test, który to wychwytuje.

## Uruchomienie predykcji

<code-tabs name="predict" />

`predict` zwraca ten sam obiekt `Results` co na każdej innej platformie, więc
strony modeli obowiązują bez zmian.

## Eksport do TensorRT

Na tej płytce TensorRT był szybszy niż PyTorch i ONNX Runtime dla wszystkich 55
modeli, które zmierzono w każdym środowisku uruchomieniowym.

<code-tabs name="export" />

`format="tensorrt"` najpierw zapisuje graf ONNX i buduje z niego silnik, więc
dodatek `onnx` musi być zainstalowany. `LibreYOLO()` rozpoznaje sufiks pliku,
więc plik `.engine` wczytuje się tym samym wywołaniem co checkpoint `.pt`.

Nie należy używać dodatku pip `tensorrt` na Jetson. Przypina on
`tensorrt-cu12`, kompilację dla CUDA 12, do platformy z CUDA 13. Zamiast tego
należy używać TensorRT instalowanego przez JetPack. Jeśli `import tensorrt`
zawodzi wewnątrz wirtualnego środowiska, a poza nim działa, należy odtworzyć
środowisko z `--system-site-packages`, aby moduł systemowy był widoczny.

Zserializowane silniki TensorRT są związane z urządzeniem, architekturą GPU i
wersją TensorRT, która je zbudowała. Silnik zbudowany na stacji roboczej nie
wczyta się na Jetson, więc krok budowania wykonuje się na płytce.

## Zmierzone na tej płytce

Opóźnienie na obraz, rozmiar batcha 1, od początku do końca, wraz z
preprocessingiem i postprocessingiem, na COCO val2017 (podzbiór 500 obrazów)
przy `conf=0.001` i `max_det=300`. Pięć modeli z 58 zmierzonych:

| Model | Wejście (px) | PyTorch FP32 (ms) | ONNX FP32 (ms) | TensorRT FP32 (ms) | TensorRT FP16 (ms) | mAP 50-95 |
|---|---:|---:|---:|---:|---:|---:|
| DEIMv2-Atto | 320 | 64.9 | 22.8 | 12.3 | 11.2 | 27.49 |
| YOLOX-Tiny | 416 | 49.2 | 31.8 | 23.0 | 19.4 | 35.45 |
| YOLO9-t | 640 | 101.2 | 53.8 | 36.0 | 29.1 | 41.78 |
| RT-DETR-r18 | 640 | 98.3 | 103.7 | 45.3 | 25.7 | 49.72 |
| D-FINE-s | 640 | 96.8 | 96.1 | 44.7 | 33.1 | 53.45 |

Kolumna mAP to własny wynik uruchomienia TensorRT FP16. Wśród 55 modeli
zmierzonych we wszystkich czterech środowiskach uruchomieniowych największa
różnica między wynikiem PyTorch FP32 a wynikiem TensorRT FP16 wyniosła 0.59
punktu, dla DEIMv2-X. Środowiska uruchomieniowe różnią się szybkością, a nie
dokładnością.

TensorRT FP32 był szybszy niż PyTorch i ONNX Runtime dla wszystkich 55 tych
modeli. TensorRT FP16 również był szybszy niż PyTorch FP32 dla wszystkich 55, od
1.68x do 6.22x, z medianą 3.39x. To ONNX Runtime jest tym, który się waha: był
wolniejszy niż PyTorch dla 23 z 55, w tym w wierszu RT-DETR-r18.

Warunki stojące za każdą liczbą: `libreyolo 1.2.0.dev0`, `torch 2.12.0+cu130`,
Python 3.12.3, CUDA 13, sterownik 595.78, ONNX Runtime 1.24.0, pomiar z czerwca
2026. Opóźnienie na Jetson zależy też od aktywnego trybu zasilania, którego
rekordy benchmarku nie zawierają.

<code-tabs name="power" />

Wszystkie 223 uruchomienia, w tym pozostałe 53 modele i pełne kolumny
dokładności, są opublikowane na
[stronie Jetson Orin w Vision Analysis](https://www.visionanalysis.org/hardware/jetson_orin).

## Rozwiązywanie problemów

### import torch zawodzi z nazwą biblioteki współdzielonej

Brakuje jednej z czterech powyższych bibliotek. Zamiast zgadywać której, można
odczytać ją z pliku binarnego:

<code-tabs name="ldd" />

Każdy brakujący wpis odpowiada jednemu pakietowi wheel:

| Brakująca biblioteka | Pakiet wheel |
|---|---|
| cuDNN | `nvidia-cudnn-cu13` |
| NCCL | `nvidia-nccl-cu13` |
| cuSPARSELt | `nvidia-cusparselt-cu13` |
| NVSHMEM | `nvidia-nvshmem-cu13` |

### torch ostrzega, że żadna kompilacja nie obsługuje tego GPU

Pierwsze wywołanie CUDA na działającej konfiguracji wypisuje to:

```text
UserWarning: Found GPU0 Orin which is of compute capability (CC) 8.7.
The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:
- 8.0 which supports hardware CC >=8.0,<9.0 except {8.7}
- 9.0 which supports hardware CC >=9.0,<10.0
- 10.0 which supports hardware CC >=10.0,<11.0 except {10.1}
- 11.0 which supports hardware CC >=11.0,<12.0
- 12.0 which supports hardware CC >=12.0,<13.0
No published PyTorch CUDA builds for release 2.13.0+cu130 support this GPU.
```

Na tej płytce ostrzeżenie jest kosmetyczne. Pakiet wheel niesie kernele `sm_80`,
a Orin je wykonuje. To samo ostrzeżenie pojawiało się przy wcześniejszym
pakiecie wheel z tego indeksu, tym, który dał każdy wiersz benchmarku powyżej.
Potwierdź to mnożeniem macierzy ze sprawdzenia CUDA, zamiast ufać temu
komunikatowi lub mu nie ufać.

### CUDA error: no kernel image is available for execution on the device

Zainstalowany pakiet wheel został zbudowany dla innej architektury GPU. Tak
dzieje się z pakietami wheel z indeksu `sbsa` firmy NVIDIA, które celują w
serwerowe GPU ARM, a nie w układy Jetson. Zainstaluj ponownie z indeksu CUDA 13
z sekcji instalacji.

### pip nie znajduje cuda-toolkit 13.0.3

Nie ma dla niego pakietu wheel dla aarch64. Użyj formy `--no-deps` z sekcji
instalacji i wymień zależności torch jawnie.

### libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file

Pakiet wheel torch dla aarch64 linkuje NVIDIA Performance Libraries do obliczeń
na CPU. Zainstaluj je i dodaj do ścieżki bibliotek:

```bash
pip install nvpl-lapack nvpl-blas --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
```

Ten indeks jest w porządku dla tych dwóch bibliotek CPU. To jego kompilacje
torch dają opisaną wyżej awarię „no kernel image”.

### Źródła pakietów wheel, które nie pasują do JetPack 7.2

| Źródło | Wynik na Orin Nano Super |
|---|---|
| torch z `pypi.jetson-ai-lab.io/sbsa/cu130` | Zbudowany dla serwerowych GPU ARM. Importuje się, zgłasza dostępność CUDA, a potem zawodzi z komunikatem „no kernel image is available for execution on the device”. |
| torch z `pypi.jetson-ai-lab.io/jp6/*` | Kompilacje dla CUDA 12 i Pythona 3.10. Nie instalują się na Pythonie 3.12 z tego obrazu. |
| Kontenery PyTorch dla JetPack 6 | Inicjalizacja CUDA zawodzi z błędem 801 na hoście z JetPack 7. |
| Budowanie torch ze źródeł | Działa, ale zajmuje godziny na płytce z 8 GB i jest zbędne, gdy zainstalowane są pakiety wheel dla CUDA 13. |

## DeepStream

Dla pełnego pipeline'u wideo zamiast pętli w Pythonie należy wyeksportować z
`deepstream=True` i uruchomić graf przez `nvinfer`. Ta ścieżka ma własną stronę,
wraz z wygenerowaną konfiguracją `nvinfer`, budową parsera ramek ograniczających
i znanymi pułapkami: [DeepStream](/docs/export/deepstream).

Sam pipeline DeepStream został zwalidowany na dedykowanym GPU x86, a nie na
Jetson. Kontrakt eksportu nie zależy od architektury, ale uruchomienie
pipeline'u na aarch64 wciąż pozostaje do zrobienia.

## Nie zweryfikowano

- Wydania JetPack inne niż 7.2 i wydania L4T inne niż R39.2.
- Płytki Jetson inne niż Orin Nano Super 8 GB.
- Trenowanie na płytce. Inferencja i eksport zostały przećwiczone; uruchomienie
  trenowania nie.
- Silniki INT8. Dla tej płytki istnieją tylko wiersze FP32 i FP16.
- Rozmiary batcha powyżej 1. Każdy powyższy pomiar to batch 1.
