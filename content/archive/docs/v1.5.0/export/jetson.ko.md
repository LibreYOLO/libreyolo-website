---
title: NVIDIA Jetson
seo_title: NVIDIA Jetson에 LibreYOLO와 PyTorch 설치
description: >-
  NVIDIA Jetson에 LibreYOLO 설치하기: JetPack이 제외하는 네 개의 CUDA 라이브러리, PyTorch에 필요한
  --no-deps 단계, 그리고 측정된 Orin Nano 수치.
lead: >-
  NVIDIA Jetson 보드는 표준 aarch64 PyTorch 휠에서 LibreYOLO를 실행합니다. Jetson 전용 torch 빌드는
  포함되지 않지만, JetPack은 torch가 연결하는 네 개의 라이브러리를 생략하며, 설치 시 이를 제공해야 합니다.
keywords:
  - 엔비디아 제트슨
  - 제트슨 오린 나노
  - 제트팩 7.2
  - Jetson에 PyTorch 설치
  - 엔비디아-cudnn-cu13
  - 엔비디아-NCCL-CU13
  - nvidia-cusparselt-cu13
  - nvidia-nvshmem-cu13
  - torch.cuda.is_available
  - 디바이스에서 실행할 수 있는 커널 이미지가 없습니다
  - 제트슨에서의 TensorRT
  - aarch64 휠
last_verified: 1.4.0
meta:
  - label: 판
    value: '제트슨 오린 나노 슈퍼 개발자 키트, 8GB, GPU 연산 능력 8.7'
  - label: 플랫폼
    value: 'JetPack 7.2 (L4T R39.2), Ubuntu 24.04, CUDA 13, Python 3.12.3, aarch64'
  - label: 스택 테스트 완료
    value: >-
      libreyolo 1.4.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, opencv
      5.0.0, numpy 2.5.1, on 2026-07-27
  - label: JetPack에서 누락됨
    value: >-
      nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-cusparselt-cu13,
      nvidia-nvshmem-cu13
    mono: true
  - label: 벤치마킹된
    value: '이 보드에서 검증된 223회 실행, 12개 계열의 58개 모델, PyTorch, ONNX Runtime 및 TensorRT에서'
    links:
      - label: visionanalysis.org/hardware/jetson_orin
        href: 'https://www.visionanalysis.org/hardware/jetson_orin'
  - label: 추적됨
    value: 문제 648의 제트슨 편
    links:
      - label: 문제 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
verification: >-
  설치 레시피와 예상 출력은 2026-07-27에 Jetson Orin Nano Super에서 실행한 설치에서 가져왔습니다. 지연 시간 및
  정확도 행은 visionanalysis.org 뒤의 검증 결과 스냅샷에서 가져왔으며, 하드웨어 jetson_orin으로 필터링하고 2026년
  6월에 libreyolo 1.2.0.dev0에서 측정했습니다. 내보내기 및 로더 동작은 libreyolo/export/exporter.py,
  libreyolo/export/tensorrt.py, libreyolo/models/__init__.py.에서 읽었습니다.
snippets:
  prep:
    - label: 시스템 패키지와 가상 환경
      language: bash
      code: |
        # JetPack은 pip나 venv 모듈을 사전 설치하지 않습니다.
        sudo apt update
        sudo apt install -y python3.12-venv python3-pip

        python3 -m venv ~/libreyolo
        source ~/libreyolo/bin/activate
        pip install -U pip wheel setuptools
  torch:
    - label: CUDA 13 휠 인덱스에서 PyTorch
      language: bash
      code: |
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu130 \
          --extra-index-url https://pypi.org/simple
    - label: JetPack이 제공하지 않는 네 가지 라이브러리
      language: bash
      code: |
        pip install nvidia-cudnn-cu13 nvidia-nccl-cu13 \
                    nvidia-cusparselt-cu13 nvidia-nvshmem-cu13
    - label: 만약 pip가 cuda-toolkit 13.0.3을 요구하면 --no-deps 옵션을 사용하여 설치하십시오
      language: bash
      code: |
        # --no-deps는 torch의 파이썬 의존성도 수동으로 지정된다는 것을 의미합니다.
        pip install --no-deps \
          torch torchvision \
          nvidia-cudnn-cu13 nvidia-nccl-cu13 \
          nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 \
          filelock typing_extensions sympy networkx jinja2 markupsafe mpmath \
          fsspec numpy pillow
  ldd:
    - label: 추측하지 말고 다음 누락된 라이브러리의 이름을 지정하십시오
      language: bash
      code: >
        ldd
        "$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" \
          | grep "not found"

        # 토치의 모든 라이브러리에서 여전히 누락된 모든 것, 한 번에:

        ldd "$VIRTUAL_ENV"/lib/python3.12/site-packages/torch/lib/*.so
        2>/dev/null \
          | grep "not found" | sort -u
  install:
    - label: 'LibreYOLO는 torch 설치 후에 설치하십시오, 그 전에 설치하지 마십시오'
      language: bash
      code: |
        # torch는 이미 만족되었으므로, pip는 CUDA 빌드를 그대로 둡니다.
        pip install libreyolo

        # ONNX 추가 기능은 내보내기할 때만 필요합니다. TensorRT 내보내기는 실행됩니다.
        # ONNX를 통해, 따라서 아래의 내보내기 섹션 전에 그것을 추가하십시오.
        pip install "libreyolo[onnx]"
  verify:
    - label: 버전 및 장치
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
    - label: 그런 다음 실제 커널을 실행하십시오
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

        # 첫 사용 시 체크포인트를 다운로드합니다.
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


        # libreyolo9s.onnx를 작성한 다음, 그것으로 libreyolo9s.engine를 만듭니다.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="tensorrt",
        half=True)


        # 엔진은 동일한 진입점을 통해 다시 로드됩니다.

        result = LibreYOLO("libreyolo9s.engine").predict(SAMPLE_IMAGE)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model libreyolo9s.pt --format tensorrt --half
  power:
    - label: 전원 모드 및 클럭
      language: bash
      code: |
        sudo nvpmodel -q      # 이 보드가 제공하는 모드와 활성화된 모드
        sudo nvpmodel -m 0    # 여기에서 테스트된 보드의 최고 모드
        sudo jetson_clocks

        tegrastats            # 실시간 부하; nvidia-smi는 Tegra에서 제한됩니다
source_hash: c07ff908503e89b5
---

## 이 페이지가 기록하는 것

이 페이지는 지원 매트릭스가 아니라 엔드 투 엔드로 검증된 하나의 구성을 기록합니다. 보드는 JetPack 7.2(L4T R39.2, Ubuntu 24.04, CUDA 13, Python 3.12.3)를 실행하는 8GB 메모리의 Jetson Orin Nano Super Developer Kit이었고, 그 위에 올라간 스택은 `libreyolo 1.4.0`와 `torch 2.13.0+cu130`, OpenCV 5.0.0, NumPy 2.5.1이었습니다. `torch.cuda.is_available()`는 `True`를 반환했고 GPU는 스스로를 `Orin`로 보고했습니다.

다른 JetPack 릴리스, 다른 Jetson 보드 및 다른 CUDA 버전은 테스트되지 않았습니다. 아래 레시피는 해당 조합에서 작동한 것입니다.

그 실행은 2026-07-27에 LibreYOLO 1.4.0을 대상으로 진행되었으며, 1.5.0 하드웨어에서는 반복되지 않았습니다. 이것은 1.5.0 트리에서 여전히 1.4.0 검증을 유지하고 있는 유일한 페이지이며, 그래서 그 앞부분에 `last_verified: "1.4.0"`라고 표시되어 있습니다. 1.5.0의 변경 사항 중에는 설치 경로나 누락된 네 개의 라이브러리, 여기에서 설명된 내보내기 플래그를 건드리는 것이 없으므로, 명령은 그대로 적용될 것으로 예상되지만, 아래 출력에 나타난 버전 번호는 1.5.0의 측정값이 아닌 1.4.0이 출력한 값입니다.

이에 대해 두 가지는 대부분의 제트슨 가이드가 말하는 것과 다릅니다. 휠은 CUDA 13용으로 게시된 일반 aarch64 빌드이므로 제트슨 전용 토치 빌드가 필요하지 않습니다. 그리고 JetPack에는 그 휠이 연결하는 네 개의 라이브러리가 포함되어 있지 않아서, `import torch`는 네 개 모두 설치될 때까지 한 번에 하나씩 라이브러리 설치에 실패합니다.

## 설치

JetPack 이미지는 pip와 `venv` 모듈 없이 제공되므로 둘 다 먼저 설치해야 합니다.

<code-tabs name="prep" />

8GB 보드는 더 큰 체크포인트에는 부족합니다. 체크포인트를 로드하기 전에 NVMe에 스왑을 추가하면 실행 중 메모리 부족으로 프로세스가 종료되는 것을 방지할 수 있습니다.

그 다음은 PyTorch입니다. CUDA 13 인덱스에는 aarch64 휠이 포함되어 있고, 추가 인덱스는 PyPI에서 순수 파이썬 의존성을 제공합니다.

<code-tabs name="torch" />

네 개의 `nvidia-*-cu13` 휠은 놓치기 쉬운 부분입니다. JetPack은 GPU 드라이버를 제공하지만 cuDNN, NCCL, cuSPARSELt 또는 NVSHMEM은 제공하지 않으며, torch는 이들 없이는 임포트하는 것을 거부합니다. 한 번에 네 개를 모두 설치하는 것이 예외를 하나씩 발견하는 것보다 더 빠릅니다.

세 번째 코드 조각은 특정 실패를 다룹니다: CUDA 13 빌드용 torch의 의존성 메타데이터가 `cuda-toolkit==13.0.3`를 요구하는데, PyPI에 aarch64 휠이 없어 다운로드가 시작되기 전에 해결이 실패합니다. `--no-deps`는 해결기를 건너뛰므로, 모든 의존성은 명령줄에서 지정해야 합니다.

LibreYOLO는 마지막에 설치합니다. 먼저 설치하면 pip가 자체적인 torch를 선택하게 되는데, 이 플랫폼에서는 CUDA 빌드가 아닙니다.

<code-tabs name="install" />

남아 있는 모든 의존성은 OpenCV, NumPy, SciPy, pycocotools 및 safetensors를 포함하여 미리 빌드된 aarch64 휠로 해결됩니다. 소스에서 컴파일되는 것은 없습니다.

## CUDA가 작동하는지 확인

<code-tabs name="verify" />

두 번째 스니펫도 첫 번째만큼 중요합니다. 잘못된 GPU 아키텍처용으로 빌드된 휠은 여전히 `torch.cuda.is_available() == True`를 보고하고 첫 번째 실제 연산에서 `CUDA error: no kernel image is available for execution on the device`로 실패합니다. 장치에서의 행렬 곱셈이 이를 잡아내는 확인 방법입니다.

## 예측을 실행

<code-tabs name="predict" />

`predict`는 다른 플랫폼에서와 동일하게 `Results` 객체를 반환하므로 모델 페이지는 변경 없이 적용됩니다.

## TensorRT로 내보내기

이 보드에서 TensorRT는 모든 런타임에서 측정된 55개의 모델 모두에 대해 PyTorch와 ONNX Runtime보다 더 빨랐습니다.

<code-tabs name="export" />

`format="tensorrt"`는 먼저 ONNX 그래프를 작성하고 그로부터 엔진을 빌드하므로 `onnx` 추가 기능을 설치해야 합니다. `LibreYOLO()`는 파일 접미사에 따라 분기하므로, `.engine` 파일은 `.pt` 체크포인트와 동일한 호출을 통해 로드됩니다.

Jetson에서 `tensorrt` pip 추가 기능을 사용하지 마십시오. 이는 CUDA 12 빌드인 `tensorrt-cu12`를 CUDA 13 플랫폼에 고정시킵니다. 대신 JetPack이 설치하는 TensorRT를 사용하십시오. `import tensorrt`가 가상 환경 안에서는 실패하지만 밖에서는 작동한다면, 시스템 모듈이 보이도록 `--system-site-packages`로 환경을 재생성하십시오.

직렬화된 TensorRT 엔진은 장치, GPU 아키텍처 및 엔진을 구축한 TensorRT 버전에 연결되어 있습니다. 워크스테이션에서 구축된 엔진은 Jetson에서 로드되지 않으므로, 빌드 단계는 보드에서 실행됩니다.

## 이 보드에서 측정됨

이미지당 지연 시간, 배치 크기 1, 전처리 및 후처리를 포함한 엔드 투 엔드, COCO val2017 (500 이미지 하위 집합)에서 `conf=0.001` 및 `max_det=300`에서 측정. 측정된 58개 모델 중 5개 모델:

| 모델 | 입력 (px) | PyTorch FP32 (밀리초) | ONNX FP32 (밀리초) | TensorRT FP32 (밀리초) | TensorRT FP16 (밀리초) | mAP 50-95 |
|---|---:|---:|---:|---:|---:|---:|
| DEIMv2-Atto | 320 | 64.9 | 22.8 | 12.3 | 11.2 | 27.49 |
| YOLOX-타이니 | 416 | 49.2 | 31.8 | 23.0 | 19.4 | 35.45 |
| YOLO9-t | 640 | 101.2 | 53.8 | 36.0 | 29.1 | 41.78 |
| RT-DETR-r18 | 640 | 98.3 | 103.7 | 45.3 | 25.7 | 49.72 |
| 디-파인-에스 | 640 | 96.8 | 96.1 | 44.7 | 33.1 | 53.45 |

mAP 열은 TensorRT FP16 실행 자체의 점수입니다. 네 가지 런타임 모두에서 측정된 55개 모델 중, PyTorch FP32 점수와 TensorRT FP16 점수 간의 가장 큰 격차는 DEIMv2-X에서 0.59 포인트였습니다. 런타임은 속도는 다르지만 정확도는 다르지 않습니다.

TensorRT FP32는 55개의 모델 모두에서 PyTorch와 ONNX Runtime보다 빠르게 실행되었습니다. TensorRT FP16 역시 55개 모델 모두에서 PyTorch FP32보다 1.68배에서 6.22배 빠르며, 중앙값은 3.39배였습니다. ONNX Runtime은 변동이 컸습니다: 55개 중 23개 모델에서 PyTorch보다 느렸으며, 그 중에는 RT-DETR-r18 모델도 포함되어 있습니다.

모든 숫자 뒤에 있는 조건: `libreyolo 1.2.0.dev0`, `torch 2.12.0+cu130`, Python 3.12.3, CUDA 13, 드라이버 595.78, ONNX Runtime 1.24.0, 2026년 6월 측정. Jetson에서의 레이턴시는 또한 활성 전원 모드에 따라 달라지며, 벤치마크 기록에는 이 정보가 포함되어 있지 않습니다.

<code-tabs name="power" />

다른 53개 모델과 전체 정확도 열을 포함한 223개의 모든 실행 결과는 [Vision Analysis의 Jetson Orin 페이지](https://www.visionanalysis.org/hardware/jetson_orin)에서 공개됩니다.

## 문제 해결

### 공유 라이브러리 이름 지정 실패로 인해 torch를 가져오지 못함

위의 네 개 라이브러리 중 하나가 없습니다. 어느 것인지 추측하기보다는, 이진수를 읽어보십시오:

<code-tabs name="ldd" />

각 누락된 항목은 하나의 바퀴에 대응됩니다:

| 라이브러리 없음 | 바퀴 |
|---|---|
| cuDNN | `nvidia-cudnn-cu13` |
| NCCL | `nvidia-nccl-cu13` |
| cuSPARSELt | `nvidia-cusparselt-cu13` |
| NVSHMEM | `nvidia-nvshmem-cu13` |

### 토치는 이 GPU를 지원하는 빌드가 없다고 경고

작동 중인 구성에서 첫 번째 CUDA 호출은 다음을 출력합니다:

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

이 보드에서는 경고가 외형적인 것입니다. 휠은 `sm_80` 커널을 포함하고 있으며 Orin이 이를 실행합니다. 동일한 경고가 그 인덱스의 이전 휠에서도 나타났는데, 그 휠이 위의 모든 벤치마크 행을 생성했습니다. 메시지를 신뢰하거나 신뢰하지 않기보다는 CUDA 체크의 행렬 곱으로 확인하십시오.

### CUDA 오류: 장치 실행용 커널 이미지 부재

설치된 휠은 다른 GPU 아키텍처용으로 빌드되었습니다. 이는 NVIDIA의 `sbsa` 인덱스에서 가져온 휠에서 발생하는 일로, 서버용 ARM GPU를 대상으로 하고 Jetson 실리콘용은 아닙니다. 설치 섹션의 CUDA 13 인덱스에서 다시 설치하십시오.

### pip의 cuda-toolkit 13.0.3 탐색 실패

이를 위한 aarch64 휠이 없습니다. 설치 섹션에서 `--no-deps` 형식을 사용하고 torch의 종속 항목을 명시적으로 지정하십시오.

### libnvpl_lapack_lp64_gomp.so.0: 공유 객체 파일 열기 실패

aarch64 토치 휠은 CPU 수학을 위해 NVIDIA 성능 라이브러리를 연결합니다. 이를 설치하고 라이브러리 경로에 넣으십시오:

```bash
pip install nvpl-lapack nvpl-blas --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
```

그 인덱스는 이 두 개의 CPU 라이브러리에는 괜찮습니다. 위에서 '커널 이미지 없음' 오류를 발생시키는 것은 torch 빌드입니다.

### JetPack 7.2에 맞지 않는 휠 소스

| 소스 | 오린 나노 슈퍼의 결과 |
|---|---|
| `pypi.jetson-ai-lab.io/sbsa/cu130` 횃불 | 서버용 ARM GPU용으로 제작됨. 가져오기, CUDA 사용 가능 보고 후 '디바이스에서 실행할 커널 이미지가 없습니다'라는 오류와 함께 실패함. |
| `pypi.jetson-ai-lab.io/jp6/*` 횃불 | CUDA 12와 Python 3.10 빌드입니다. 이 이미지의 Python 3.12에는 설치되지 않습니다. |
| JetPack 6 PyTorch 컨테이너 | JetPack 7 호스트에서 CUDA 초기화가 오류 801과 함께 실패합니다. |
| 소스에서 토치 빌드하기 | 작동하지만 8GB 보드에서는 몇 시간이 걸리며, CUDA 13 휠이 설치되면 불필요합니다. |

## 딥스트림

Python 루프가 아닌 전체 비디오 파이프라인의 경우, `deepstream=True`로 내보내고 `nvinfer`를 통해 그래프를 실행하십시오. 해당 경로에는 생성된 `nvinfer` 구성, 바운딩 박스 파서 빌드 및 알려진 함정이 포함된 자체 페이지가 있습니다: [DeepStream](/docs/export/deepstream).

DeepStream 파이프라인 자체는 Jetson이 아닌 x86 독립 GPU에서 검증되었습니다. 내보내기 계약은 아키텍처에 의존하지 않지만 aarch64에서의 파이프라인 실행은 여전히 미완료 상태입니다.

## 확인되지 않음

- 7.2가 아닌 JetPack 릴리스 및 R39.2가 아닌 L4T 릴리스.
- Orin Nano 슈퍼 8GB를 제외한 Jetson 보드.
- 보드에서 학습 중. 추론과 내보내기는 수행되었으나, 학습 실행은 이루어지지 않았습니다.
- INT8 엔진. 이 보드에는 FP32와 FP16 행만 존재합니다.
- 배치 크기는 1 이상입니다. 위의 모든 측정은 배치 1입니다.
