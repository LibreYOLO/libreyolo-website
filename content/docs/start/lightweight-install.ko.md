---
title: 경량 설치
seo_title: PyTorch 없이 LibreYOLO ONNX 추론 실행
description: >-
  --no-deps 옵션으로 LibreYOLO 설치 및 numpy만으로 ONNX 탐지 실행, 디스크에 torch 없음. 기술, 한계, 정확한
  패키지 목록.
lead: >-
  LibreYOLO의 ONNX 추론 경로는 디코드와 NMS를 포함하여 numpy 엔드 투 엔드입니다. 실행 시간에 PyTorch가 필요하지
  않으므로, 의존성 해결을 건너뛴 설치도 머신에 torch 없이 탐지를 수행할 수 있습니다.
keywords:
  - torch 없는 추론
  - torch 없음
  - pytorch 없는 libreyolo
  - torch 없는 onnx 추론
  - libreyolo 경량 설치
  - pip 설치 no-deps
  - libreyolo 디스크 공간
  - onnxruntime 추론
last_verified: 1.5.0
meta:
  - label: 적용 대상
    value: 'ONNX 탐지, 7개 모델 패밀리'
  - label: 진입점
    value: libreyolo.backends.onnx.OnnxBackend
    mono: true
  - label: 지원 수준
    value: '최선의 노력, 별도의 배포 아님'
snippets:
  install:
    - label: 경량
      language: bash
      code: |
        # 패키지의 의존성 목록 없이 설치한 다음, 실제로 ONNX 감지 경로에서 가져오는 네 가지 패키지를 제공하세요.
        # 정의되지 않음
        pip install --no-deps libreyolo
        pip install numpy pillow opencv-python-headless onnxruntime
    - label: CPU 전용 torch
      language: bash
      code: |
        # 먼저 이것을 시도하세요. 모든 기능을 유지하면서 CUDA 휠을 피할 수 있으며,
        # 대부분의 디스크 공간이 거기에 사용됩니다.
        pip install libreyolo --index-url https://download.pytorch.org/whl/cpu
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo.backends.onnx import OnnxBackend


        model = OnnxBackend("libreyolo9t.onnx")

        result =
        model.predict("https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg")


        # xyxy는 여기에서 torch 텐서가 아닌 numpy ndarray입니다.

        print(result.boxes.xyxy)

        print(result.boxes.conf)

        print(result.boxes.cls)
source_hash: e60e83d32d13026e
---

## 이것이 작동하는 이유

`pip install --no-deps libreyolo`는 패키지를 설치하고 의존성 목록을 완전히 건너뜁니다. 사용자를 대신해 아무것도 해결되지 않으며, 실제로 사용하는 것을 설치하는 책임은 사용자에게 있습니다.

이것은 원하는 코드 경로가 정말로 건너뛴 종속성을 필요로 하지 않을 경우에만 유용하며, ONNX 감지의 경우에는 그렇지 않습니다. 디코딩은 비최대 억제(non-maximum suppression)를 포함하여 numpy입니다. 전처리 레시피도 numpy입니다. PyTorch는 학습 및 즉시 추론(eager-inference) 종속성이며, 이 경로에서는 절대 호출되지 않습니다.

이 릴리스 이전에는 가져오기가 어쨌든 실패했습니다: `libreyolo.models` 아래의 모든 것을 가져오면 체크포인트 자동 감지 레지스트리를 채우기 위해 모든 모델 클래스가 빌드되었고, 그 클래스들은 `torch.nn.Module` 하위 클래스입니다. 전처리 레시피는 이제 자체 패키지 `libreyolo.preprocess`에 있으며, torch 가져오기는 torch 속성을 다루는 무언가가 나타날 때까지 지연되므로, ONNX 경로는 기계에 torch가 없어도 가져올 수 있습니다. 그 패키지에는 각 패밀리별로 numpy-기반 전처기가 포함되어 있습니다: `yolo9`, `yolonas`, `yolox`, `ec`, `rtdetr`, `rfdetr`, `dfine`, `deim` 및 `deimv2`, 아래에서 끝까지 검증된 7개 패밀리보다 2개 더 많습니다. 각 `libreyolo/models/<family>/utils.py`는 그것에서 다시 내보내므로 기존 가져오기 경로는 계속 작동합니다.

## 먼저 CPU 전용 휠을 시도해 보세요

이 파일을 요청하는 대부분의 사람들은 다중 기가바이트 설치를 피하고 싶어하며, 크기는 한 곳에 집중되어 있습니다: 기본 `torch` 휠에는 CUDA가 포함되어 있습니다. CPU 전용 빌드는 그 일부에 불과하며 특별한 설치 경로가 필요하지 않습니다.

<code-tabs name="install" />

CPU 전용 옵션은 모든 LibreYOLO 기능을 유지합니다: 학습, 검증, 모든 작업, 모든 패밀리, CLI. 머신에 torch를 전혀 설치하고 싶을 때는 경량 경로를 선택하세요, 단순히 torch를 줄이는 것이 아닙니다.

## 경량 설치가 포함하는 것

| | |
|---|---|
| 작업 | 검출 |
| 형식 | ONNX |
| 진입점 | `OnnxBackend` |
| 인터페이스 | Python 라이브러리 |

이 경로에서 7개의 패밀리가 확인되었습니다: [YOLOv9](/docs/models/yolov9), [YOLO-NAS](/docs/models/yolo-nas), [EdgeCrafter](/docs/models/edgecrafter), [RT-DETR](/docs/models/rt-detr), [RF-DETR](/docs/models/rf-detr), [D-FINE](/docs/models/d-fine) 및 [DEIM](/docs/models/deim), 각 패밀리의 변형을 포함하여 계산됩니다.

이는 검증된 범위를 의미하며, 라이브러리가 강제하는 경계가 아닙니다. 다른 작업과 다른 패밀리는 단순히 확인되지 않은 곳에 있으며: 일부는 호출 시 torch를 가져올 수 있고, 일부는 우연히 작동할 수도 있습니다. 이 목록을 벗어나는 항목은 지원되거나 고장난 것으로 생각하지 말고 테스트되지 않은 것으로 간주하십시오.

내부에서는 결과가 단순히 유사한 것이 아니라 일반 설치와 동일합니다. 각 패밀리는 ONNX로 내보낸 뒤 두 번 실행되었으며, 한 번은 일반적으로 실행하고 한 번은 torch가 차단된 상태에서 실행했습니다; 박스, 점수 및 클래스가 정확히 일치했습니다. 테스트 모음에 있는 일관성 테스트는 그 계약이 변하지 않도록 유지합니다.

## 사람들을 사로잡는 다섯 가지

**모델 클래스를 사용하지 말고 `OnnxBackend`를 사용하십시오.** `LibreYOLO9("model.onnx")`는 여전히 torch가 필요합니다, 왜냐하면 `LibreYOLO9` 자체가 `nn.Module`의 하위 클래스이기 때문입니다. 이는 가장 흔한 실수입니다, 왜냐하면 이 문서의 다른 모든 페이지는 모델을 클래스나 `LibreYOLO()`를 통해 로드하기 때문입니다.

**다른 곳에 내보내세요.** `.onnx` 파일을 생성하려면 torch가 필요하므로, 경량 머신에서는 생성할 수 없습니다. 개발 또는 CI 머신에서 내보내고 아티팩트를 슬림 타겟으로 전달하세요.

**결과는 numpy 배열을 포함합니다.** `result.boxes.xyxy`는 여기서 `ndarray`입니다. 컨테이너는 어느 타입이든 허용하므로 속성 이름은 변경되지 않지만, 결과에서 `.cpu()`나 `.numpy()`를 호출하는 코드는 실패합니다.

**단일 이미지는 단일 `Results`를 반환합니다.** `predict()`는 한 이미지에 대해 하나의 `Results`를 반환하고, 여러 이미지에는 리스트를 반환합니다. 단일 결과를 `[0]`로 인덱싱하면 첫 번째 이미지를 선택하는 것이 아니라 첫 번째 감지를 선택하며, 이는 오류를 발생시키는 대신 조용히 한 박스 결과를 제공합니다.

**CLI는 작동하지 않습니다.** `typer`와 `click`는 네 개의 패키지에 포함되어 있지 않으므로 `libreyolo` 명령은 사용할 수 없습니다. 이는 라이브러리 설치입니다.

## 예측

<code-tabs name="predict" />

CUDA에서 실행하려면 `onnxruntime`을 `onnxruntime-gpu`로 교체하십시오. 네 개의 패키지는 완전한 torch-free `predict()`가 실제로 import하는 것들이며, 추론이 아닌 호출 중에 기록된 것입니다. `opencv-python-headless`는 선언된 `opencv-python`를 대신하며: 동일한 모듈, GUI 라이브러리 없음, 디스크 상에서 더 작음.

나머지 선언된 종속성 중에서, `requests`는 URL에서 이미지를 로드하는 데만 필요하며, `pycocotools`와 `scipy`는 검증 및 평가를 위한 것이고, `typer`와 `click`는 CLI입니다.

## 이 목록은 의도적으로 변동될 수 있습니다.

위의 패키지 목록은 이 페이지 상단에 명시된 릴리스에 대해 정확합니다. `--no-deps`는 종속성 해결을 선택 해제하므로 아무 것도 자동으로 확인하지 않으며, 이후 릴리스에서는 여기 나열되지 않은 항목을 가져올 수도 있습니다.

`ModuleNotFoundError`가 발생하면 이미 기술을 이해한 것입니다: 누락된 패키지를 설치하면 됩니다. 이는 버그 보고가 아니라 의도된 유지 관리 모델입니다. 이 경로는 최선의 노력 기준이며 별도로 지원되는 배포판이 아니므로 PyPI에 두 번째 경량 패키지가 없고 계획도 없는 이유이기도 합니다.

환경이 설치된 torch로 조용히 되돌아가는 것이 아니라 정말로 torch가 없는지 확인하려면 다음과 같이 확인하십시오:

```python
import importlib.util

assert importlib.util.find_spec("torch") is None, "torch is installed"
```

이 검사는 슬림 이미지용 CI에 유지할 가치가 있습니다. 이 검사가 없으면 우연히 torch가 있는 환경이 모든 테스트를 통과하고 아무것도 알려주지 않습니다.
