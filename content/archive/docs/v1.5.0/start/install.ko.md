---
title: 설치
seo_title: LibreYOLO 설치
description: >-
  PyPI에서 LibreYOLO를 설치하고, 모델 군 또는 내보내기 대상이 필요한 선택적 추가 기능을 선택한 후, PyTorch가 GPU를
  인식하는지 확인하십시오.
lead: >-
  LibreYOLO는 PyPI에 libreyolo로 공개되어 있습니다. 기본 패키지는 PyTorch만으로 충분한 모델 계열에 대한 예측,
  학습, 검증을 포함하며; 선택적 추가 기능은 나머지를 제공합니다.
keywords:
  - libreyolo 설치
  - pip 설치 libreyolo
  - libreyolo 추가 기능
  - 리브리욜로 쿠다
  - 리브리욜로 GPU
  - libreyolo 요구 사항
last_verified: 1.5.0
meta:
  - label: 패키지
    value: libreyolo
    mono: true
  - label: Python
    value: 3.10 이상
  - label: 코드 라이선스
    value: MIT
  - label: 핵심 의존성
    value: PyTorch 2.4 이상
snippets:
  install:
    - label: 파이프
      language: bash
      code: |
        pip install libreyolo
    - label: 추가 항목 포함
      language: bash
      code: |
        # 여러 개를 하나의 설치에 합치려면 쉼표로 구분하십시오.
        pip install "libreyolo[rfdetr,onnx]"
    - label: 모든 것
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: 출처에서
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # 파이썬, 토치, CUDA, cuDNN, 모든 보이는 GPU, 그리고 어느 것
        # 선택적 패키지가 설치되었습니다.
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: 모델 재고
      language: bash
      code: |
        # 모든 등록된 계열과 그들의 작업, 규모 및 입력
        # 결의안. 여분이 없는 계열은 다음과 같이 나열됩니다
        # 그들을 가능하게 하는 pip 명령어.
        libreyolo models
source_hash: 34fc6d3e24d03fb4
---

## 설치

<code-tabs name="install" />

Python 3.10 이상이 필요합니다. 기본 설치는 PyTorch, torchvision, NumPy, Pillow, OpenCV, PyYAML, requests, mss, tqdm, pycocotools, typer, click, safetensors 및 SciPy를 포함하므로, 추가로 필요한 것이 없는 YOLOv9 및 다른 계열는 `pip install libreyolo` 후 바로 작동합니다.

클론은 이 문서와 일치하는 코드를 가진 안정 브랜치 `release`를 확인합니다. 미출시 작업을 포함하는 통합 브랜치는 `dev`입니다.

## 선택 사항 추가

추가는 한 모델 계열이나 한 내보내기 대상이 필요로 하는 종속 항목을 추가하는 괄호로 묶인 이름입니다. 다른 것은 변경되지 않습니다: 추가가 있든 없든 API는 동일합니다.

### 모델 계열

| 추가 | 추가함 |
|---|---|
| `rfdetr` | RF-DETR 백본을 공급하는 `transformers` |
| `eomt` | `transformers` |
| `midas` | `timm` 1.0.x, MiDaS의 ViT-L/16 및 EfficientNet-Lite3 인코더를 제공하는 |
| `vlm` | `transformers`, `num2words`, `decord`, `lmdb`, `peft` |
| `sam` | `transformers`, `timm` |
| `openvocab` | `transformers`, `timm`, `regex`, `ftfy` |
| `sensenova` | macOS에서 `transformers`, `accelerate`, 및 `bitsandbytes` |
| `modus` | `transformers`, `accelerate` |
| `clip` | `regex` 및 `ftfy`, 공급된 CLIP 텍스트 토크나이저에 필요함 |
| `siglip2` | 다국어 SigLIP 2 토크나이저에서 필요한 `sentencepiece` |
| `gaze` | `gdown`, L2CS 체크포인트의 자동 다운로드를 켭니다 |
| `rtdetr` | 없음. RT-DETR은 추가적인 의존성이 필요하지 않으며, 이름은 안정적으로 유지됩니다. |

### 내보내기 및 런타임

| 추가 | 추가합니다 |
|---|---|
| `onnx` | `onnx`, `onnxsim`, `onnxruntime` |
| `tensorrt` | `tensorrt-cu12` 10.16.1.11 및 `pycuda`, macOS 종료 |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`, macOS 전용 |
| `tflite`, 별칭 `litert` | `libreyolo[onnx]`와 `onnx2tf`, `ai-edge-litert`, `onnx-graphsurgeon` 및 `onnx-simplifier` |
| `mnn` | `libreyolo[onnx]` 더하기 `MNN` |
| `ncnn` | `pnnx` 및 `ncnn` |
| `paddle` | `libreyolo[onnx]` 플러스 `paddlepaddle` 2.6.2 및 `x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | HTTP 및 HTTPS V2 추론을 위한 `tritonclient[http]` |

### 학습, 평가 및 기록

| 추가 | 추가함 |
|---|---|
| `lora` | `libreyolo[rfdetr]` 플러스 `peft`, `lora=True` 파인튜닝을 위해 |
| `plots` | `matplotlib` |
| `fast-eval` | `faster-coco-eval`, C++ COCO 평가 백엔드 |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`, 별칭 `dvc` | `dvclive` |

`fast-eval`는 하드 의존성이 아니라 선택적(opt-in) 의존성이므로, 사전 빌드된 휠(wheel)이 없는 플랫폼에서도 일반 설치가 중단되지 않습니다. 패키지가 없는 경우, COCO 평가 평가가 pycocotools로 대체되며 실행은 계속됩니다.

### 공구

| 추가 | 추가함 |
|---|---|
| `stream` | `yt-dlp`, YouTube 페이지 URL을 해결하는 데만 필요 |
| `tracking` | 없습니다. 모든 추적 의존성은 이미 핵심 의존성입니다 |
| `label` | `libreyolo[sam]`, `libreyolo label`에서 클릭으로 마스크 지원을 가능하게 합니다 |
| `hub-kernels` | `kernels`, 컴파일된 허브 커널용 선택적 로더입니다. 설치하면 RF-DETR 예측이 부동 소수점 허용 오차에서 이동할 수 있다고 언급한 [kernels](/docs/reference/kernels)를 참조하십시오. |
| `clip-convert` | 중량 변환 및 패리티 검사를 위해 `libreyolo[clip]`와 `open_clip_torch` |
| `siglip2-convert` | 같은 이유로 `libreyolo[siglip2]`와 `transformers` |

웹캠, RTSP, RTMP, TCP, UDP, HLS 및 로컬 다중 스트림 목록은 추가가 필요 없습니다. 오직 유튜브 페이지 URL만 필요합니다.

### 총 추가

`libreyolo[all]`는 모델, 내보내기, 추적 및 로깅 추가 기능을 한 명령으로 설치합니다. 일부는 의도적으로 제외되어 있습니다. `neptune`는 안정적인 `neptune-scale`가 protobuf를 7 미만으로 요구하는 반면 TFLite 경로는 protobuf 7을 요구하기 때문에 제외됩니다. `executorch`는 ExecuTorch가 함께 사용할 PyTorch 버전을 제한하기 때문에 제외되며, `coreai`는 `coreai-torch`가 PyTorch를 2.11.x로 고정하며 전체 환경을 해당 버전으로 끌어올리기 때문에 제외됩니다. `fast-eval`, `hub-kernels`, `clip-convert` 및 `siglip2-convert`도 제외됩니다. 이름으로 설치할 수 있습니다.

## 플랫폼 제약

세 개의 추가 기능은 종속성 표시기에 의해 플랫폼 범위로 지정되므로 설치가 모든 곳에서 성공하며 휠이 존재하지 않는 곳에서는 단순히 적게 설치됩니다.

| 추가 | 제약 |
|---|---|
| `coreai` | macOS 전용입니다. Core AI 툴체인은 다른 곳에서는 변환하거나 실행하지 않습니다. |
| `tensorrt` | CUDA가 없는 macOS에서 건너뜀 |
| `tflite`, `litert` | `onnx2tf`와 `ai-edge-litert`는 Python 3.12 이상이 필요합니다 |

`sensenova`는 macOS에서 `bitsandbytes`를 건너뛰는데, 해당 플랫폼에는 휠이 게시되지 않습니다. 나머지 추가 항목들은 정상적으로 설치됩니다.

디스크가 병목이라면 대부분은 PyTorch이고, PyTorch의 대부분은 기본 휠에 포함된 CUDA 페이로드입니다. CPU 전용 휠은 아무것도 포기하지 않고 그것을 제거합니다. 토치를 전혀 설치하지 않아야 하는 머신에서 ONNX 탐지를 위해서는 [경량 설치](/docs/lightweight-install)를 참조하십시오.

## GPU와 CUDA

장치 선택은 모델이 구성될 때 이루어집니다. 기본값인 `device="auto"`는 `torch.cuda.is_available()`가 참일 때 CUDA를 사용하고, `torch.backends.mps.is_available()`가 참일 때 Metal Performance Shaders를 사용하며, 그렇지 않으면 CPU를 사용합니다. 라이브러리의 다른 부분은 하드웨어를 검사하지 않으므로, PyTorch가 GPU를 인식하지 못하면 LibreYOLO도 인식하지 못합니다.

대신 장치를 고정하려면 모델에 `device`를 전달하거나 `predict`, `train`, `val` 및 `export`를 전달하십시오. `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, `0`와 같은 정수 또는 `"0"`와 같은 숫자 문자열을 허용합니다; 마지막 두 가지는 `cuda:<n>`로 확장됩니다.

`libreyolo checks`로 시작합니다. 이는 Torch 버전, Torch가 빌드된 CUDA 및 cuDNN 버전, 그리고 메모리를 포함한 모든 가시적인 GPU를 출력합니다. NVIDIA 카드가 있는 머신에서 CUDA가 없다고 보고되면, pip로 설치된 PyTorch 휠이 CPU 빌드입니다. 먼저 PyTorch 인덱스에서 CUDA 빌드를 설치한 후, LibreYOLO를 설치하십시오:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

그것은 저장소가 Linux와 Windows에서 자체 uv 관리 환경에 대해 고정하는 동일한 인덱스입니다. 이는 CUDA 12.8 런타임 요구 사항인 NVIDIA 드라이버 555 이상이 필요합니다. macOS는 PyPI 휠을 유지하는데, 이는 PyTorch 다운로드 호스트가 Darwin 빌드를 게시하지 않기 때문입니다.

## 설치를 확인

<code-tabs name="verify" />

`libreyolo models`는 추가 기능이 적용되었는지 확인하는 가장 빠른 방법입니다: 의존성이 누락된 패키지는 이를 활성화하는 정확한 pip 명령어와 함께 출력됩니다. 두 명령 모두 `--json`를 허용하며, 이는 동일한 데이터를 stdout에 기계가 읽을 수 있는 객체로 출력합니다.
