---
title: 경량 설치
seo_title: PyTorch 없이 LibreYOLO ONNX 추론 실행
description: >-
  LibreYOLO를 --no-deps 옵션으로 설치하고, 디스크에 torch 없이 numpy만으로 ONNX 탐지를 실행하십시오. 이 기술,
  한계, 그리고 정확한 패키지 목록.
lead: >-
  LibreYOLO의 ONNX 추론 경로는 디코드와 NMS를 포함하여 끝에서 끝까지 numpy로 되어 있습니다. 실행 시점에 PyTorch가
  필요하지 않으므로, 의존성 확인을 건너뛴 설치도 머신에 torch가 없어도 검출을 실행할 수 있습니다.
keywords:
  - 토치 없는 추론
  - 횃불 자유
  - 파이토치 없이 libreyolo
  - torch 없이 onnx 추론
  - libreyolo 경량 설치
  - pip 설치 no-deps
  - libreyolo 디스크 공간
  - onnxruntime 추론
last_verified: 1.5.0
meta:
  - label: 적용 대상
    value: 'ONNX 탐지, 일곱 가지 모델 계열'
  - label: 진입점
    value: libreyolo.backends.onnx.OnnxBackend
    mono: true
  - label: 지원 수준
    value: '최선의 노력, 별도의 배포가 아님'
snippets:
  install:
    - label: 경량
      language: bash
      code: |
        # 의존성 목록 없이 패키지를 설치한 다음, 그런 다음 제공하십시오
        # ONNX 탐지 경로가 실제로 가져오는 네 개의 패키지.
        pip install --no-deps libreyolo
        pip install numpy pillow opencv-python-headless onnxruntime
    - label: CPU 전용 토치
      language: bash
      code: |
        # 먼저 이것을 시도해 보십시오. 모든 기능을 유지하고 CUDA 휠을 피합니다.
        # 그것이 디스크의 대부분이 차지되는 곳입니다.
        pip install libreyolo --index-url https://download.pytorch.org/whl/cpu
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo.backends.onnx import OnnxBackend


        model = OnnxBackend("libreyolo9t.onnx")

        result =
        model.predict("https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg")


        # xyxy는 여기서 torch 텐서가 아니라 numpy ndarray입니다.

        print(result.boxes.xyxy)

        print(result.boxes.conf)

        print(result.boxes.cls)
source_hash: e60e83d32d13026e
---

## 왜 이 방법이 효과가 있는가

`pip install --no-deps libreyolo`는 패키지를 설치하고 그 의존성 목록을 완전히 건너뜁니다. 어떤 의존성도 대신 해결되지 않으므로 실제로 사용할 패키지를 직접 설치해야 합니다.

그것은 원하는 코드 경로가 실제로 건너뛴 종속성을 필요로 하지 않을 때만 유용하며, ONNX 탐지의 경우에는 필요하지 않습니다. 디코드, 비최대 억제를 포함하여, numpy입니다. 전처리 레시피도 numpy입니다. PyTorch는 학습 및 즉시 추론 종속성이며, 이 경로에서는 절대 호출되지 않습니다.

이 릴리스 이전에는 어쨌든 가져오기에 실패했습니다: `libreyolo.models` 아래의 어떤 것이든 가져오면 체크포인트 자동 탐지 레지스트리를 채우기 위해 모든 모델 클래스가 생성되었고, 그 클래스들은 `torch.nn.Module` 하위 클래스입니다. 전처리 레시피는 이제 자체 패키지 `libreyolo.preprocess`에 있으며, torch 가져오기는 torch 속성에 접근할 때까지 지연됩니다. 따라서 ONNX 경로는 머신에 torch가 없어도 가져올 수 있습니다. 그 패키지는 각 계열에 대해 numpy 네이티브 전처리기를 보유하고 있습니다: `yolo9`, `yolonas`, `yolox`, `ec`, `rtdetr`, `rfdetr`, `dfine`, `deim` 및 `deimv2`으로, 아래에서 끝까지 검증된 7개 계열보다 두 개 더 많습니다. 각 `libreyolo/models/<family>/utils.py`는 그것에서 다시 내보내므로 기존의 가져오기 경로가 계속 작동합니다.

## CPU 전용 휠 우선 시도

이를 요청하는 대부분의 사람들은 다중 기가바이트 설치를 피하고 싶어하며, 용량은 한 곳에 집중되어 있습니다: 기본 `torch` 휠이 CUDA를 포함합니다. CPU 전용 빌드는 그 일부에 불과하며 특별한 설치 경로가 필요하지 않습니다.

<code-tabs name="install" />

CPU 전용 옵션은 모든 LibreYOLO 기능을 유지합니다: 학습, 검증, 모든 작업, 모든 계열, CLI. 단순히 최소한으로 사용하는 것이 아니라, 기계에 토치를 전혀 설치하지 않으려 할 때 가벼운 경로를 선택하십시오.

## 경량 설치가 포함하는 내용

| | |
|---|---|
| 작업 | 탐지 |
| 형식 | ONNX |
| 진입점 | `OnnxBackend` |
| 인터페이스 | 파이썬 라이브러리 |

이 경로에서는 다음 일곱 계열이 확인되었습니다: [YOLOv9](/docs/models/yolov9), [YOLO-NAS](/docs/models/yolo-nas), [EdgeCrafter](/docs/models/edgecrafter), [RT-DETR](/docs/models/rt-detr), [RF-DETR](/docs/models/rf-detr), [D-FINE](/docs/models/d-fine) 및 [DEIM](/docs/models/deim), 각 계열의 변종을 함께 계산했습니다.

그것은 검증된 범위이지, 라이브러리가 강제하는 경계가 아닙니다. 다른 작업과 다른 계열는 단순히 확인된 범위 밖에 있습니다: 일부는 호출 시 Torch를 불러오고, 몇몇은 우연히 작동할 수도 있습니다. 이 목록을 벗어나는 것은 지원되거나 고장난 것으로 간주하지 말고 테스트되지 않은 것으로 취급하십시오.

내부에서는 결과가 단순히 비슷한 것이 아니라 일반 설치와 동일합니다. 각 계열는 ONNX로 내보내지고 두 번 실행되었습니다. 한 번은 정상적으로, 한 번은 torch가 차단된 상태에서 실행되었으며; 박스, 점수 및 클래스가 정확히 일치했습니다. 스위트 내의 동등성 테스트는 그 계약이 벗어나지 않도록 유지합니다.

## 사람들의 시선을 사로잡는 다섯 가지

**모델 클래스가 아닌 `OnnxBackend`를 사용하십시오.** `LibreYOLO9("model.onnx")`는 여전히 torch가 필요합니다. 왜냐하면 `LibreYOLO9` 자체가 `nn.Module` 하위 클래스이기 때문입니다. 이 문서의 다른 모든 페이지가 모델을 클래스나 `LibreYOLO()`를 통해 로드하기 때문에, 이것이 가장 가능성이 높은 실수입니다.

**다른 곳으로 내보내기.** `.onnx` 파일을 생성하려면 torch가 필요하므로, 경량 머신에서는 파일을 만들 수 없습니다. 개발용 또는 CI 머신에서 내보낸 후 아티팩트를 슬림 대상 머신으로 전송하십시오.

**결과는 numpy 배열을 포함합니다.** `result.boxes.xyxy`는 여기에서 `ndarray`입니다. 컨테이너는 두 가지 유형 모두를 허용하므로 속성 이름은 변경되지 않지만, 결과에서 `.cpu()` 또는 `.numpy()`를 호출하는 코드는 실패할 것입니다.

**단일 이미지가 단일 `Results`를 반환합니다.** `predict()`는 한 이미지에 대해 하나의 `Results`를 반환하고 여러 이미지에 대해서는 리스트를 반환합니다. 단일 결과를 `[0]`로 인덱싱하면 첫 번째 이미지를 선택하는 것이 아니라 첫 번째 검출을 선택하게 되며, 이는 예외를 발생시키지 않고 하나의 박스 결과만 조용히 제공합니다.

**CLI는 작동하지 않습니다.** `typer`와 `click`는 네 개의 패키지에 없으므로 `libreyolo` 명령을 사용할 수 없습니다. 이것은 라이브러리 설치입니다.

## 예측

<code-tabs name="predict" />

CUDA에서 실행하려면 `onnxruntime`를 `onnxruntime-gpu`로 교체하십시오. 네 개의 패키지는 실제로 전체 torch-free `predict()`가 가져오는 것들이며, 추론된 것이 아니라 호출 중에 기록된 것입니다. `opencv-python-headless`는 선언된 `opencv-python`를 대신합니다: 동일한 모듈, GUI 라이브러리 없음, 디스크 상에서 더 작음.

나머지 선언된 종속성 중에서, `requests`는 URL에서 이미지를 로드하는 데만 필요하고, `pycocotools`와 `scipy`는 검증과 평가를 위한 것이며, `typer`와 `click`는 CLI입니다.

## 의도적으로 변동되는 목록

위의 패키지 목록은 이 페이지 상단에 명시된 릴리스에 대해 정확합니다. `--no-deps`는 의존성 해결을 선택 해제하므로 아무 것도 이를 확인하지 않으며, 나중 릴리스에서는 여기 나열되지 않은 것을 가져올 수 있습니다.

만약 `ModuleNotFoundError`를 접했다면, 이미 그 기술을 이해하고 있는 것입니다: 누락된 패키지를 설치하십시오. 이것은 버그 리포트가 아니라 의도된 유지 관리 모델입니다. 이 경로는 최선의 노력을 다하는 방법이며 별도로 지원되는 배포판이 아니므로, PyPI에 두 번째 경량 패키지가 없고 그에 대한 계획도 없는 이유이기도 합니다.

설치된 복사본으로 조용히 되돌아가지 않고 환경이 실제로 토치 없는 상태인지 확인하려면, 다음을 확인하십시오:

```python
import importlib.util

assert importlib.util.find_spec("torch") is None, "torch is installed"
```

그 체크는 슬림 이미지용 CI에서 유지할 가치가 있습니다. 이것이 없으면 우연히 torch가 있는 환경은 모든 테스트를 통과하게 되어 아무 것도 알려주지 않습니다.
