---
title: 설치
seo_title: LibreYOLO 설치
description: >-
  PyPI에서 LibreYOLO를 설치하고, 선택적 추가 항목 중 모델 패밀리나 내보내기 대상이 필요로 하는 항목을 선택한 후 PyTorch가
  GPU를 인식하는지 확인하십시오.
lead: >-
  LibreYOLO는 PyPI에 libreyolo로 게시되어 있습니다. 기본 패키지는 예측, 학습, 검증 및 PyTorch만 필요로 하는 모델
  패밀리를 포함합니다; 선택적 추가 항목은 나머지를 추가합니다.
keywords:
  - libreyolo 설치
  - pip install libreyolo
  - libreyolo 추가 항목
  - libreyolo cuda
  - libreyolo gpu
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
    - label: pip
      language: bash
      code: |
        pip install libreyolo
    - label: 추가 항목 포함
      language: bash
      code: |
        # 여러 항목을 하나의 설치에 결합하려면 쉼표로 구분하세요.
        pip install "libreyolo[rfdetr,onnx]"
    - label: 모든 항목
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: 소스에서
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # Python, Torch, CUDA, cuDNN, 모든 보이는 GPU, 그리고 어떤
        # 선택적 패키지가 설치되어 있는지.
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: 모델 인벤토리
      language: bash
      code: |
        # 등록된 모든 패밀리와 그 작업, 크기 및 입력
        # 해상도. 추가 항목이 누락된 패밀리는
        # 이를 활성화하는 pip 명령과 함께 나열됩니다.
        libreyolo models
source_hash: 34fc6d3e24d03fb4
---

## 설치

<code-tabs name="install" />

Python 3.10 이상이 필요합니다. 기본 설치는 PyTorch, torchvision, NumPy, Pillow, OpenCV, PyYAML, requests, mss, tqdm, pycocotools, typer, click, safetensors 및 SciPy를 설치하므로, YOLOv9 및 추가 설치가 필요 없는 다른 패밀리는 `pip install libreyolo` 이후 바로 작동합니다.

클론은 `release`를 확인하는데, 이 안정 브랜치의 코드는 이 문서와 일치합니다. 출시되지 않은 작업을 포함하는 통합 브랜치는 `dev`입니다.

## 선택적 추가 항목

추가 항목은 한 모델 패밀리나 하나의 내보내기 대상이 필요로 하는 종속성을 추가하는 괄호로 묶인 이름입니다. 다른 것은 변하지 않습니다: 추가 항목의 유무와 상관없이 API는 동일합니다.

### 모델 패밀리

| 추가 항목 | 추가 |
|---|---|
| `rfdetr` | RF-DETR 백본을 제공하는 `transformers` |
| `eomt` | `transformers` |
| `midas` | MiDaS의 ViT-L/16와 EfficientNet-Lite3 인코더를 제공하는 `timm` 1.0.x |
| `vlm` | `transformers`, `num2words`, `decord`, `lmdb`, `peft` |
| `sam` | `transformers`, `timm` |
| `openvocab` | `transformers`, `timm`, `regex`, `ftfy` |
| `sensenova` | macOS에서 `transformers`, `accelerate`, `bitsandbytes` |
| `modus` | `transformers`, `accelerate` |
| `clip` | `regex` 및 `ftfy`, 공급된 CLIP 텍스트 토크나이저에 필요함 |
| `siglip2` | `sentencepiece`, 다국어 SigLIP 2 토크나이저에 필요함 |
| `gaze` | `gdown`, L2CS 체크포인트 자동 다운로드를 켬 |
| `rtdetr` | 없음. RT-DETR에는 추가 의존성이 필요하지 않음; 이름은 유지됨 |

### 내보내기 및 런타임

| 추가 | 추가함 |
|---|---|
| `onnx` | `onnx`, `onnxsim`, `onnxruntime` |
| `tensorrt` | `tensorrt-cu12` 10.16.1.11 및 `pycuda`, macOS 제외 |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`, macOS 전용 |
| `tflite`, 별칭 `litert` | `libreyolo[onnx]` 및 `onnx2tf`, `ai-edge-litert`, `onnx-graphsurgeon`와 `onnx-simplifier` |
| `mnn` | `libreyolo[onnx]` 및 `MNN` |
| `ncnn` | `pnnx`와 `ncnn` |
| `paddle` | `libreyolo[onnx]` 및 `paddlepaddle` 2.6.2와 `x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | HTTP 및 HTTPS V2 추론용 `tritonclient[http]` |

### 학습, 평가 및 로깅

| 추가 | 추가 기능 |
|---|---|
| `lora` | `libreyolo[rfdetr]` 및 `peft`, `lora=True` 파인튜닝용 |
| `plots` | `matplotlib` |
| `fast-eval` | `faster-coco-eval`, C++ COCO 평가 백엔드 |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`, 별칭 `dvc` | `dvclive` |

`fast-eval`는 사전 구축된 휠이 없는 플랫폼이 일반 설치를 깨뜨리지 않도록 강제 의존성이 아닌 선택적 의존성입니다. 패키지가 없는 경우 COCO 평가가 pycocotools로 대체되며 실행은 계속됩니다.

### 도구

| 추가 | 추가합니다 |
|---|---|
| `stream` | `yt-dlp`, YouTube 페이지 URL을 해결하는 데만 필요합니다 |
| `tracking` | 없음. 모든 추적 의존성은 이미 핵심 의존성입니다 |
| `label` | `libreyolo[sam]`, `libreyolo label`에서 클릭 투 마스크 지원을 가능하게 함 |
| `hub-kernels` | `kernels`, 컴파일된 허브 커널용 선택적 로더. 설치하면 RF-DETR 예측이 부동소수점 허용 오차 범위에서 변할 수 있다는 내용은 [kernels](/docs/reference/kernels) 참조 |
| `clip-convert` | `libreyolo[clip]` 및 `open_clip_torch`, 가중치 변환 및 패리티 검사용 |
| `siglip2-convert` | `libreyolo[siglip2]` 및 `transformers`, 같은 이유로 |

웹캠, RTSP, RTMP, TCP, UDP, HLS 및 로컬 멀티 스트림 목록에는 추가 사항이 필요 없음. 유튜브 페이지 URL만 필요함.

### 집계 추가

`libreyolo[all]`는 모델, 내보내기, 추적 및 로깅 추가 기능을 한 명령어로 설치합니다. 일부는 의도적으로 제외되어 있습니다. `neptune`는 안정적인 `neptune-scale`가 protobuf 7 미만을 요구하는 반면 TFLite 경로는 protobuf 7을 요구하기 때문에 제외됩니다. `executorch`는 ExecuTorch가 결합할 PyTorch 버전을 제한하기 때문에 제외되며, `coreai`는 `coreai-torch`가 PyTorch를 2.11.x로 고정하여 전체 환경을 해당 버전으로 끌어올리기 때문에 제외됩니다. `fast-eval`, `hub-kernels`, `clip-convert` 및 `siglip2-convert`도 제외됩니다. 이름으로 설치할 수 있습니다.

## 플랫폼 제약

세 가지 추가 기능은 종속성 마커로 인해 플랫폼 범위가 지정되므로 설치가 모든 곳에서 성공하며 휠이 없는 곳에서는 단순히 적게 설치됩니다.

| 추가 기능 | 제약 |
|---|---|
| `coreai` | macOS 전용. 코어 AI 툴체인은 다른 곳에서는 변환하거나 실행하지 않습니다. |
| `tensorrt` | CUDA가 없는 macOS에서 건너뜀 |
| `tflite`, `litert` | `onnx2tf` 및 `ai-edge-litert`는 Python 3.12 이상 필요 |

`sensenova`는 macOS에서 `bitsandbytes`를 건너뜀(휠이 게시되지 않음); 나머지 추가 설치는 정상적으로 진행됨.

디스크가 제약이라면 대부분이 PyTorch이고, 대부분의 PyTorch는 기본 휠에 포함된 CUDA 페이로드입니다. CPU 전용 휠은 아무 것도 포기하지 않고 이를 제거합니다. Torch가 전혀 없어야 하는 머신에서 ONNX 감지를 위해 [경량 설치](/docs/lightweight-install)를 참고하세요.

## GPU 및 CUDA

모델이 구성될 때 장치 선택이 이루어집니다. 기본값인 `device="auto"`는 `torch.cuda.is_available()`가 true일 때 CUDA를 사용하고, `torch.backends.mps.is_available()`가 true일 때 Metal Performance Shaders를 사용하며, 그 외에는 CPU를 사용합니다. 라이브러리의 다른 부분에서는 하드웨어를 검사하지 않으므로, PyTorch가 GPU를 인식하지 못하면 LibreYOLO도 인식하지 못합니다.

장치를 고정하려면 모델이나 `predict`, `train`, `val`, `export`에 `device`를 전달하면 됩니다. 이 값은 `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, `0`와 같은 정수 또는 `"0"`와 같은 숫자 문자열을 허용하며; 마지막 두 가지는 `cuda:<n>`로 확장됩니다.

`libreyolo checks`로 시작하세요. 이는 Torch 버전, Torch가 빌드된 CUDA 및 cuDNN 버전, 그리고 메모리를 포함한 모든 보이는 GPU를 출력합니다. NVIDIA 카드가 있는 머신에서 CUDA가 없다고 보고되면, pip로 해결된 PyTorch 휠이 CPU 빌드입니다. 먼저 PyTorch 인덱스에서 CUDA 빌드를 설치한 다음 LibreYOLO를 설치하세요:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

이는 저장소가 Linux 및 Windows에서 자체 uv-managed 환경을 위해 고정하는 것과 동일한 인덱스입니다. NVIDIA 드라이버 555 이상이 필요하며, 이는 CUDA 12.8 런타임 요구 사항입니다. macOS는 PyPI 휠을 유지합니다. PyTorch 다운로드 호스트가 Darwin 빌드를 제공하지 않기 때문입니다.

## 설치를 확인

<code-tabs name="verify" />

`libreyolo models`은 추가 기능이 효과가 있었는지 여부를 확인하는 가장 빠른 방법입니다: 종속성이 누락된 계열은 해당 종속성을 활성화하는 정확한 pip 명령어와 함께 출력됩니다. 두 명령어 모두 `--json`를 받아들이며, 이는 동일한 데이터를 기계가 읽을 수 있는 객체 형태로 stdout에 출력합니다.
