---
title: 1.5.0으로 업그레이드
seo_title: LibreYOLO 1.4.0을 1.5.0으로 업그레이드
description: >-
  1.5.0이 요구하는 네 가지 코드 변경 사항, 메트릭을 이동시키는 세 가지 변경 사항, 실행 비교 전에 알아둘 가치가 있는 작은 동작
  변화들.
lead: >-
  공개 모델 API에서 삭제된 것은 없습니다: 1.4.0에서 작동하던 모든 클래스와 함수는 여전히 가져올 수 있습니다. 네 개의 인수 모양이
  변경되었고, 세 개의 기본 값이 비교할 수 있는 숫자로 이동했습니다.
keywords:
  - libreyolo 업그레이드
  - libreyolo 1.5.0 마이그레이션
  - allow_experimental 제거
  - libreyolo 주요 변경 사항
  - yolox bn eps
  - faster-coco-eval 기본값
last_verified: 1.5.0
meta:
  - label: 적용 대상
    value: 1.4.0에서 1.5.0으로
  - label: 필요한 코드 변경 사항
    value: 네 가지 모두 좁은 변경
  - label: 이동하는 결과
    value: 'COCO 백엔드, YOLOX BN eps, D-FINE 다중 스케일'
  - label: 공개 API 제거
    value: None
source_hash: ab38d8ef7b53f596
---

이 페이지는 LibreYOLO 자체 업그레이드에 관한 것입니다. 만약 상위 프로젝트에서 체크포인트를 로드하는 방법을 찾고 있다면, 그것은 [기존 가중치 가져오기](/docs/migrate)라는 다른 주제입니다.

릴리스에 대한 전체 항목은 [변경 로그](/docs/changelog)를 참조하십시오. 다음은 귀하에게 무언가를 요구하는 부분만입니다.

## 반드시 수행해야 하는 코드 변경

### `allow_experimental=True`는 더 이상 존재하지 않습니다

인식 확인 게이트와 그 뒤에 있는 `ddp_aware(experimental_key=...)` 메커니즘이 제거되었습니다. EC, RTMDet, PicoDet 및 FOMO 학습 및 내보내기는 이전에 해당 인수가 필요했으므로 이러한 계열 중 하나를 학습하는 모든 스크립트에 영향을 미칩니다.

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0: 인수 삭제
model.train(data="data.yaml", epochs=100)
```

더 이상 사용되지 않는 셰임이 없습니다. 여전히 이를 전달하는 호출은 `TypeError`를 발생시킵니다. `BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES`는 함께 제거되었습니다. `get_download_notice()` 훅은 여전히 존재하며 MiDaS, SegFormer 및 YOLO9-P2에 의해 여전히 오버라이드됩니다.

지원 수준은 여전히 게시되지만 더 이상 인수로 사용되지 않습니다: [stability tiers](/docs/reference/stability-tiers)를 참조하십시오.

### 내보내기 계층 `"experimental"`는 더 이상 존재하지 않습니다.

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

계층 문자열에 기반한 코드 분기는 `"experimental"` 대신 `"available"`를 읽어야 합니다. `BaseExporter`는 이제 해당 형식에 대해 `RuntimeWarning`를 발생시키지 않습니다. 형식별 상태는 [export matrix](/docs/reference/export-matrix)에 나와 있습니다.

### `resume`와 함께하는 `pretrained=False`는 이제 거부됩니다.

이전에는 조합이 일관성 없이 진행되었습니다. 이제는 다음을 발생시킵니다:

```
ValueError: pretrained=False cannot be combined with resume.
```

하나를 선택하십시오. `pretrained=False`는 새로운 시드 초기화로 시작하며, 1.5.0에서는 세 가지가 아닌 모든 학습 가능한 패밀리에서 작동하며, `resume`는 체크포인트에서 중단된 실행을 계속합니다. 둘 다 [training](/docs/train) 아래에 문서화되어 있습니다.

### CLI `--imgsz`는 정수가 아닌 문자열입니다.

생각보다 좁습니다. 둘 다 영향을 받지 않습니다:

```bash
libreyolo predict --model yolo9-t --source img.jpg --imgsz 640   # 여전히 괜찮습니다.
```

```python
model.predict("img.jpg", imgsz=640)   # 여전히 괜찮습니다.
```

Python에서 [CLI](/docs/cli) 명령 함수를 직접 호출하는 코드만 변경이 필요합니다. `predict`, `train`와 `val`가 `--imgsz`를 `int`에서 `str`로 확장하여 직사각형 크기도 허용하게 되었기 때문입니다:

```python
from libreyolo.cli.commands.predict import predict_cmd

predict_cmd(..., imgsz=640)      # 1.4.0
predict_cmd(..., imgsz="640")    # 1.5.0에서는 "480x640"도 이제 작동합니다.
```

`train`의 기본값은 이제 문자열 `"640"`입니다. `export --imgsz`는 이미 문자열이었고, `profile`는 변경되지 않았습니다.

## 변경되는 숫자

세 가지 변경 사항이 기본 설정에서 지표를 이동시킵니다. 버전별로 결과를 추적하는 경우 1.5.0 실행을 1.4.0 실행과 비교하기 전에 이를 읽으십시오.

### faster-coco-eval이 기본 COCO 지표 백엔드입니다.

`val()` 및 에포크별 학습 검증이 이제 pycocotools 대신 faster-coco-eval C++ 백엔드로 COCO 지표를 계산합니다.

이 전환은 모든 100개의 RF100-VL 테스트 분할에서 측정된 동등성을 기준으로 결정되었습니다: 1400개의 지표 값 중 1381개가 비트 단위로 동일하고, 최대 편차는 2.22e-16이며, 주된 변동 값은 정확히 0이고, 전체적으로 15.6배 빠르며 탐지 밀집 데이터셋에서는 56배 빠릅니다. 숫자는 변경되지 않아야 합니다. 어쨌든 다른 구현에 의해 생성되므로 이 목록에 포함된 이유입니다.

pycocotools는 faster-coco-eval이 설치되지 않았을 때 자동으로 대체됩니다. 강제로 설정하려면:

```bash
libreyolo val --model yolo9-t --data coco.yaml --no-faster-coco-eval
```

```python
model.val(data="coco.yaml", faster_coco_eval=False)
```

`LIBREYOLO_FASTER_COCO_EVAL=0`는 전역적으로 동일한 기능을 수행합니다. 실제로 사용되는 백엔드는 INFO에 기록되며, `val()` 이후 `model.last_eval_backend`로 노출되고, [CLI](/docs/cli/val) JSON 페이로드에 `eval_backend`로 포함됩니다. `pip install libreyolo[fast-eval]`로 빠른 경로를 설치하세요.

### 1.5.0 이전에 학습된 YOLOX 체크포인트는 eps 오버라이드가 필요

이것이 릴리스에서의 함정입니다. 파인튜닝된 [YOLOX](/docs/models/yolox)를 사용한 경우 읽어보세요.

YOLOX는 BatchNorm `eps=1e-3` 및 `momentum=0.03`를 지정합니다. 1.5.0 이전까지 이러한 값들은 클래스 수 재구성 `train()`가 데이터셋의 `nc`가 체크포인트와 다를 때 생존하지 못하는 사후 보정으로 적용되었습니다. 이러한 파인튜닝은 torch의 기본 `eps=1e-5`에서 학습 중 검증 결과를 보고한 후 `1e-3`에서 추론을 위해 다시 로드되었습니다: 다른 정규화에서 동일한 텐서.

일반 합성곱(conv) 크기는 거의 움직이지 않습니다. 깊이별 합성곱(depthwise) `n`은 많이 움직이는데, 채널별 `running_var`가 eps보다 충분히 작기 때문입니다. RF100-VL `ball`에서 동일한 나노 체크포인트는 학습된 eps에서 평가 시 **0.566** mAP50-95를 기록하고, 기본 재로드 후 **0.151**를 기록합니다.

1.5.0 이전에 학습된 체크포인트는 eps=1e-5 의미를 가집니다. 이를 위해 정확한 수치를 보고하려면 BN eps를 1e-5로 덮어쓰고 평가해야 합니다:

```python
import torch
from libreyolo import LibreYOLOX

model = LibreYOLOX("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

또는 `sqrt((var + 1e-3) / (var + 1e-5))`를 BN 가중치에 한 번 접어서 결과를 저장합니다. 1.5.0 이상에서 학습된 체크포인트는 필요하지 않습니다.

### D-FINE 다중 스케일 학습은 상위 레시피를 사용

`base_size_repeat`는 모든 크기에 대해 하드코딩되어 3이었습니다. 이제 상위에서 지정한 대로 각 크기별로 해결됩니다: **n**은 다중 스케일이 꺼진 상태에서 고정 크기로 학습, **s** 20, **m** 6, **l** 4, **x** 3. 이전에는 x만 일치하여 n, s, m, l은 다른 스케일 분포를 보이며 다른 지표로 수렴합니다.

이전 동작을 복원하려면 명시적으로 설정하십시오:

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIM은 여전히 하드코딩된 3을 사용합니다. 세부 정보는 [D-FINE](/docs/models/d-fine)에서 확인할 수 있습니다.

## 참고만 하면 되며, 별도의 조치는 필요하지 않습니다

- **직사각형 `imgsz` 결과가 이전에는 잘못되어 변경되었습니다.** 박스 좌표, RTMDet 마스크 리사이징, YOLO-NAS 리스케일링 및 검증자 실제 값 스케일링이 이제 단일 스칼라가 아닌 각각 축별 높이와 너비를 사용합니다. 정사각형 `imgsz`는 거의 변경되지 않았습니다. 1.4.0에서 직사각형 추론 또는 검증 실행이 잘못 스케일링되었습니다. YOLO-NAS는 이제 잘못된 출력을 조용히 내보내는 대신 직사각형 `imgsz`를 완전히 거부합니다.
- **메트릭 사전이 키를 추가했습니다.** COCO 평가기의 `max_det`, `ar_max_det` 및 `AR_max_det`와 FOMO의 `metrics/loss` 및 `metrics/loss/ce`. 기본값의 값은 변경되지 않았지만, 사용자 정의 [로거](/docs/train/loggers) 및 CSV 헤더를 포함하여 메트릭 키를 반복하는 모든 것에서 새 열이 표시됩니다.
- **헤드 재구성을 트리거하는 시드 YOLO9 실행**은 이제 재구성 전에 시드를 적용하기 때문에 초기화가 달라집니다. 다른 클래스 수에 대해 시드가 적용된 1.4.0 파인튜닝은 1.5.0에서 비트 단위로 재현할 수 없습니다.
- **CUDA에서 `libreyolo[hub-kernels]`가 이제 실제로 네이티브 MS-deform-attn 커널을 작동시킵니다.** 1.4.0에서는 RF-DETR이 절대 사용하지 않는 조건 뒤에 차단되어 커널이 실행되지 않았습니다. RF-DETR 및 다른 변형 주의 계열에서 예측은 부동 소수점 허용 범위 내에서 변동할 수 있습니다. 일반 설치에는 영향이 없으며 `LIBREYOLO_HUB_KERNELS=0`가 이를 비활성화합니다.
- **`libreyolo predict`는 오류를 발생시키는 대신 지원되지 않는 옵션을 제거합니다.** CLI는 모델의 `__call__` 시그니처에 맞춰 kwargs를 필터링하므로 계열에서 허용하지 않는 옵션은 `TypeError`를 발생시키는 대신 무시됩니다. 플래그 이름의 오타도 이제는 조용히 무시됩니다.
- **실시간 소스는 JSON 출력 형식을 변경합니다.** 웹캠, RTSP 스트림 및 화면 캡처는 암묵적으로 스트리밍을 활성화하며, 이는 호출당 하나의 레코드가 아니라 프레임당 하나의 레코드를 출력합니다. 이러한 [소스](/docs/predict/sources)는 1.5.0에서 새로 추가되었으므로 1.4.0 스크립트에는 영향을 주지 않습니다.
- **`rfdetr-pose` 또는 `yolonas-pose`를 ONNX로 다시 내보내면 출력 이름이 달라집니다.** 1.4.0에서는 다중 텐서 포즈 헤드를 출력 수 추정을 통해 세분화로 잘못 인식했습니다. 디스크에 있는 기존 `.onnx` 파일은 그대로 유지됩니다.
- **Torch가 없는 설치에서는** 결과가 `torch.Tensor` 대신 numpy 배열을 유지하므로 `.boxes.data`는 다른 타입을 반환하며 NMS 동점 처리 방식이 torchvision과 다를 수 있습니다. Torch가 설치된 경우 동작은 바이트 단위까지 동일합니다. [경량 설치](/docs/lightweight-install)를 참조하십시오.
- **구성 객체가 생성 시 더 많은 검증을 수행합니다.** `TrainConfig`는 없었던 `__post_init__`를 갖게 되었으므로, 이미 잘못된 구성은 실행 도중 깊이에서 실패하는 대신 즉시 오류를 발생시킵니다. `ValidationConfig` 직렬화는 `edge_thresholds` 키를 추가했으며, 이는 1.4.0 덤프에서 엄격한 `ValidationConfig(**dump)` 왕복을 깨뜨립니다.
- **작업 접미사가 붙은 패밀리의 가중치 파일 이름이 다르게 해결됩니다.** `segformer-b0`는 이제 `LibreSegformerb0-sem.pt`로 해결됩니다. 이는 자동 다운로드 404 오류를 수정하고, 이전 접미사가 없는 파일 이름을 하드코딩한 스크립트는 작동하지 않게 됩니다.
- **pytest 마커 `experimental_backend`가 이제 `extended_backend`입니다.** 테스트 스위트를 `-m`로 실행하는 경우에만 관련이 있습니다.

## 체크포인트 및 데이터셋

1.4.0에서 작성된 체크포인트는 그대로 로드됩니다. [스키마](/docs/reference/checkpoint-schema)는 직사각형 모델을 위해 `imgsz_h`와 `imgsz_w`를 추가했으며, 이전 리더를 위해 여전히 스칼라 `imgsz = max(h, w)`를 작성합니다. [ExecuTorch](/docs/export/executorch)와 [MNN](/docs/export/mnn) 내보내기는 이제 각각 사이드카 `<program>.pte.json`와 `<model>.mnn.json`가 필요하며, HRNet 내보내기는 `pose_input: "person_crop"`를 포함합니다. 데이터셋 형식은 변경되지 않았습니다.
