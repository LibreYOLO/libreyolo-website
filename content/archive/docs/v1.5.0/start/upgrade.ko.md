---
title: 1.5.0으로 업그레이드
seo_title: LibreYOLO 1.4.0을 1.5.0으로 업그레이드
description: >-
  버전 1.5.0에서 필요한 네 가지 코드 변경 사항, 메트릭을 이동하는 세 가지 변경 사항, 그리고 실행을 비교하기 전에 알아두면 좋은 작은
  행동 변화들.
lead: >-
  공개 모델 API에서 아무것도 제거되지 않았습니다: 1.4.0에서 작동하던 모든 클래스와 함수는 여전히 임포트됩니다. 네 개의 인자는 형태가
  변경되었고, 세 개의 기본값은 비교하고 있을 수 있는 숫자를 이동했습니다.
keywords:
  - libreyolo 업그레이드
  - libreyolo 1.5.0 마이그레이션
  - allow_experimental 제거됨
  - libreyolo 주요 변경 사항
  - yolox bn eps
  - faster-coco-eval 기본값
last_verified: 1.5.0
meta:
  - label: 적용 대상
    value: 1.4.0에서 1.5.0으로
  - label: 코드 변경 필요
    value: '넷, 모두 좁다'
  - label: 결과를 만드는
    value: 'COCO 백엔드, YOLOX BN eps, D-FINE 다중 스케일'
  - label: 공개 API 제거
    value: None
source_hash: ab38d8ef7b53f596
---

이 페이지는 LibreYOLO 자체를 업그레이드하는 것에 관한 것입니다. 업스트림 프로젝트에서 체크포인트를 로드하는 방법을 찾고 있다면, 이는 [기존 가중치 가져오기](/docs/migrate)로, 다른 주제입니다.

릴리스의 전체 변경 사항은 [변경 로그](/docs/changelog)에 있습니다. 아래에는 조치가 필요한 항목만 나옵니다.

## 해야 하는 코드 변경

### `allow_experimental=True` 제거

승인 게이트와 그 뒤에 있는 `ddp_aware(experimental_key=...)` 메커니즘이 사라졌습니다. 이전에는 EC, RTMDet, PicoDet 및 FOMO 학습 및 내보내기에 해당 인수가 필요했기 때문에, 이러한 계열 중 하나를 학습하는 모든 스크립트에 영향을 미칩니다.

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0: 인수를 삭제
model.train(data="data.yaml", epochs=100)
```

더 이상 사용 중단 대체 코드는 없습니다. 여전히 그것을 전달하는 호출은 `TypeError`를 발생시킵니다. `BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES`는 그것과 함께 제거되었습니다. `get_download_notice()` 훅은 남아 있으며 여전히 MiDaS, SegFormer 및 YOLO9-P2에 의해 재정의됩니다.

지원 수준은 여전히 게시되지만 더 이상 인수가 아닙니다: [안정성 등급](/docs/reference/stability-tiers)을 참조하십시오.

### 내보내기 계층 `"experimental"` 제거

```python
from libreyolo.export.support import Tier

# 1.4.0: 리터럴["validated", "experimental", "blocked"]
# 1.5.0: 문자 그대로["검증됨", "사용 가능", "차단됨"]
```

티어 문자열에서 코드 분기는 `"experimental"`로 읽히던 곳을 `"available"`로 읽어야 합니다. `BaseExporter`는 더 이상 해당 형식에 대해 `RuntimeWarning`를 방출하지 않습니다. 형식별 상태는 [export matrix](/docs/reference/export-matrix)에 나와 있습니다.

### `pretrained=False`와 `resume` 조합 거부

이전에 결합은 일관성 없이 진행되었습니다. 이제 다음을 제기합니다:

```
ValueError: pretrained=False cannot be combined with resume.
```

하나를 선택하십시오. `pretrained=False`는 새로 초기화된 시드에서 시작하며, 1.5.0에서는 세 가지가 아니라 모든 학습 가능한 계열에 대해 작동합니다. `resume`는 체크포인트에서 중단된 실행을 계속합니다. 두 가지 모두 [학습](/docs/train) 아래에 문서화되어 있습니다.

### CLI `--imgsz`는 문자열이지 정수가 아닙니다

말하는 것보다 좁습니다. 이 두 가지는 영향을 받지 않습니다:

```bash
libreyolo predict --model yolo9-t --source img.jpg --imgsz 640   # 아직 괜찮다
```

```python
model.predict("img.jpg", imgsz=640)   # 아직 괜찮다
```

Python에서 [CLI](/docs/cli) 명령 함수들을 직접 호출하는 코드만 변경할 필요가 있습니다. 왜냐하면 `predict`, `train`, `val`가 `--imgsz`를 `int`에서 `str`로 확장하여 직사각형 크기를 허용할 수 있게 했기 때문입니다:

```python
from libreyolo.cli.commands.predict import predict_cmd

predict_cmd(..., imgsz=640)      # 1.4.0
predict_cmd(..., imgsz="640")    # 1.5.0 및 "480x640"도 이제 작동합니다
```

`train`의 기본값은 이제 문자열 `"640"`입니다. `export --imgsz`는 이미 문자열이었고, `profile`는 변경되지 않았습니다.

## 변하는 숫자들

세 가지 변경 사항이 기본 설정에서 지표를 움직입니다. 버전별 결과를 추적하는 경우, 1.5.0 실행과 1.4.0 실행을 비교하기 전에 이것들을 읽으십시오.

### 기본 COCO 지표 백엔드 faster-coco-eval

`val()`와 에포크별 학습 검증은 이제 pycocotools 대신 faster-coco-eval C++ 백엔드를 사용하여 COCO 지표를 계산합니다.

스위치는 100개의 RF100-VL 테스트 분할 전반에 걸친 측정된 동등성을 기준으로 결정되었습니다: 1400개의 메트릭 값 중 1381개가 비트 단위로 동일하며, 최대 편차는 2.22e-16, 주요 차이는 정확히 0이며, 전체적으로 15.6배 더 빠르고 탐지 밀도가 높은 데이터셋에서는 56배 더 빠릅니다. 숫자는 변경되지 않아야 합니다. 어쨌든 다른 구현에 의해 생성된 것이므로 이것이 리스트에 있는 이유입니다.

faster-coco-eval이 설치되어 있지 않을 때 pycocotools가 자동 대체로 유지됩니다. 강제로 사용하려면:

```bash
libreyolo val --model yolo9-t --data coco.yaml --no-faster-coco-eval
```

```python
model.val(data="coco.yaml", faster_coco_eval=False)
```

`LIBREYOLO_FASTER_COCO_EVAL=0`는 전 세계적으로 동일한 작업을 수행합니다. 실제로 사용되는 백엔드는 INFO에 기록되며, `val()` 후에는 `model.last_eval_backend`로 노출되고, [CLI](/docs/cli/val) JSON 페이로드에 `eval_backend`로 포함됩니다. `pip install libreyolo[fast-eval]`로 빠른 경로를 설치하십시오.

### YOLOX 1.5.0 이전에 학습된 체크포인트는 eps 재정의가 필요

이것은 릴리스의 함정입니다. [YOLOX](/docs/models/yolox)를 파인튜닝한 경우 읽어보십시오.

YOLOX는 BatchNorm `eps=1e-3` 및 `momentum=0.03`를 지정합니다. 1.5.0 이전까지 이러한 값들은 후속 보정으로 적용되었으며, 데이터셋의 `nc`가 체크포인트의 것과 다를 때 `train()`가 수행하는 클래스 수 재구성에서는 유지되지 않았습니다. 이러한 세부 조정은 torch의 기본 `eps=1e-5`에서 학습 중 검증을 보고한 후 `1e-3`에서 추론을 위해 다시 로드되었습니다: 서로 다른 정규화에서 동일한 텐서들입니다.

일반 컨브 크기는 거의 움직이지 않습니다. Depthwise `n`는 많이 움직이는데, 이는 채널별 `running_var`가 eps가 지배할 만큼 충분히 작기 때문입니다. RF100-VL `ball`에서는 동일한 나노 체크포인트가 학습된 eps에서 평가했을 때 **0.566** mAP50-95을 기록하고, 초기 상태로 다시 로드한 후에는 **0.151**을 기록합니다.

1.5.0 이전에 학습된 체크포인트는 eps=1e-5 의미를 가집니다. 이를 위해 정확한 수치를 보고하려면, BN eps를 1e-5로 오버라이드하여 평가하십시오:

```python
import torch
from libreyolo import LibreYOLOX

model = LibreYOLOX("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

또는 `sqrt((var + 1e-3) / (var + 1e-5))`를 BN 가중치에 한 번 접어 결과를 저장하십시오. 1.5.0 이후에 학습된 체크포인트는 필요하지 않습니다.

### D-FINE 다중 스케일 학습은 업스트림 크기별 레시피를 사용

`base_size_repeat`는 모든 크기에 대해 3으로 하드코딩되어 있었습니다. 이제 업스트림에서 지정한 대로 크기별로 해결됩니다: **n**은 멀티 스케일이 꺼진 상태에서 고정 크기로 학습, **s** 20, **m** 6, **l** 4, **x** 3. 이전에는 x만 일치했기 때문에, n, s, m, l은 다른 스케일 분포를 보게 되며 다른 지표로 수렴합니다.

이전 동작을 복원하려면 명시적으로 설정하십시오:

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIM은 여전히 하드코딩된 3을 사용합니다. 계열 세부 사항은 [D-FINE](/docs/models/d-fine)에 있습니다.

## 알아둘 만함, 조치 필요 없음

- **직사각형 `imgsz` 결과가 이전에 잘못되어 변경되었습니다.** 박스 좌표, RTMDet 마스크 크기 조정, YOLO-NAS 재스케일링 및 검증기 실제 값 스케일링은 이제 하나의 스칼라 대신 축별 높이와 너비를 사용합니다. 정사각형 `imgsz`는 거의 변경되지 않았습니다. 1.4.0에서의 직사각형 추론 또는 검증 실행은 잘못 스케일링되었습니다. YOLO-NAS는 이제 잘못된 출력을 조용히 생성하는 대신 직사각형 `imgsz`를 아예 거부합니다.
- **측정 지표 사전이 키를 얻었습니다.** COCO 평가기에서는 `max_det`, `ar_max_det` 및 `AR_max_det`를, FOMO에서는 `metrics/loss`와 `metrics/loss/ce`를 추가했습니다. 기본값의 값은 변경되지 않았지만, 사용자 정의 [로거](/docs/train/loggers)와 CSV 헤더를 포함하여 측정 지표 키를 반복하는 모든 항목은 새로운 열을 확인합니다.
- **헤드 재구성을 트리거하는 시드 YOLO9 실행**은 재구성 전에 시드가 적용되기 때문에 서로 다른 초기화에서 시작됩니다. 다른 클래스 수에 대해 시드 1.4.0로 파인튜닝한 결과는 1.5.0에서 비트 단위로 재현할 수 없습니다.
- **`libreyolo[hub-kernels]`가 이제 CUDA에서 실제로 네이티브 MS-deform-attn 커널을 사용합니다.** 1.4.0에서는 RF-DETR가 절대 사용하지 않는 조건 뒤에 이를 제한했기 때문에 커널이 실행되지 않았습니다. RF-DETR 및 다른 변형 주의(attention) 계열에서 예측 값이 부동소수점 허용 오차 내에서 변동할 수 있습니다. 기본 설치에는 영향을 주지 않으며, `LIBREYOLO_HUB_KERNELS=0`는 이를 비활성화합니다.
- **`libreyolo predict`은 오류를 발생시키는 대신 지원되지 않는 옵션을 제거합니다.** CLI는 모델의 `__call__` 시그니처에 대해 kwargs를 필터링하므로, 계열가 허용하지 않는 옵션은 `TypeError`을 발생시키는 대신 무시됩니다. 플래그 이름의 오타도 이제 조용히 무시됩니다.
- **실시간 소스는 JSON 출력 형태를 변경합니다.** 웹캠, RTSP 스트림 및 화면 캡처는 암묵적으로 스트리밍을 활성화하며, 호출당 하나의 레코드가 아니라 프레임당 하나의 레코드를 생성합니다. 이 [소스](/docs/predict/sources)는 1.5.0에서 새로 추가되었으므로 1.4.0 스크립트에는 영향을 주지 않습니다.
- **`rfdetr-pose` 또는 `yolonas-pose`를 ONNX로 다시 내보내면 출력 이름이 다르게 나옵니다.** 1.4.0은 출력 수 기반의 휴리스틱을 통해 이들의 다중 텐서 포즈 헤드를 세분화(segmentation)로 잘못 읽었습니다. 디스크에 있는 기존 `.onnx` 파일은 그대로 유지됩니다.
- **토치가 없는 설치**에서는 결과가 `torch.Tensor`가 아닌 numpy 배열을 유지하므로, `.boxes.data`는 다른 유형을 반환하며 NMS 동점 처리 방식이 torchvision과 다를 수 있습니다. 토치가 설치되어 있으면 동작은 바이트 단위까지 동일합니다. [경량 설치](/docs/lightweight-install)를 참조하십시오.
- **구성 객체는 생성 시 더 많이 검증됩니다.** `TrainConfig`는 이전에는 없던 `__post_init__`를 얻었으므로, 이미 유효하지 않았던 구성은 실행 중 깊이에서 실패하는 대신 즉시 오류를 발생시킵니다. `ValidationConfig` 직렬화는 `edge_thresholds` 키를 얻었으며, 이는 1.4.0 덤프에서 엄격한 `ValidationConfig(**dump)` 왕복을 깨뜨립니다.
- **작업 접미사가 있는 계열의 가중치 파일 이름이 다르게 해석됩니다.** `segformer-b0`는 이제 `LibreSegformerb0-sem.pt`로 해석됩니다. 이는 자동 다운로드 404 오류를 수정하고, 이전에 접미사가 없는 파일 이름을 하드코딩한 모든 스크립트를 깨뜨립니다.
- **pytest 마커 `experimental_backend`는 이제 `extended_backend`입니다.** `-m`로 테스트 스위트를 실행할 때만 관련이 있습니다.

## 체크포인트와 데이터셋

1.4.0으로 작성된 체크포인트는 변경 없이 로드됩니다. [스키마](/docs/reference/checkpoint-schema)는 직사각형 모델을 위해 `imgsz_h`와 `imgsz_w`를 추가했으며, 이전 리더를 위해 여전히 스칼라 `imgsz = max(h, w)`를 작성합니다. [ExecuTorch](/docs/export/executorch)와 [MNN](/docs/export/mnn) 내보내기는 이제 각각 사이드카 `<program>.pte.json`와 `<model>.mnn.json`가 필요하며, HRNet 내보내기는 `pose_input: "person_crop"`를 포함합니다. 데이터셋 형식은 변경되지 않았습니다.
