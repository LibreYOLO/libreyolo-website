---
title: 1.7.0으로 업그레이드
seo_title: LibreYOLO 1.6.0에서 1.7.0으로 업그레이드
description: >-
  LibreYOLO 1.7.0의 YOLO9 체크포인트, 모듈 이름, 학습 레시피, CLI 실행 폴더, 혼동 행렬, 새 오류에 대한 마이그레이션
  절차입니다.
lead: >-
  1.7.0은 이전 YOLO9 체크포인트를 같은 float 박스로 계속 불러옵니다. YOLO9 계열의 학습, 코드가 네트워크 내부에서 보는
  이름, 그리고 몇 가지 검증과 예측 계약이 바뀝니다.
keywords:
  - libreyolo 업그레이드
  - libreyolo 1.6.0 마이그레이션
  - libreyolo 주요 변경 사항
  - yolo9 모듈 이름
  - clip_max_norm
  - 혼동 행렬 confusion matrix
last_verified: 1.7.0
source_hash: 895bcd7425f0052f
---

## 1.6.0에서 1.7.0으로

- 1.7.0이 저장한 YOLO9, YOLO9-E2E, YOLO9-P2 체크포인트는 새 텐서 이름을 사용하며 1.6.x에서 불러올 수 없습니다. 1.6.x는 YOLO9와 YOLO9-P2 체크포인트를 누락된 키 경고와 함께 불러오고 박스를 하나도 예측하지 않으며, YOLO9-E2E 체크포인트에는 `model_family` 불일치 오류를 발생시킵니다. 이를 읽는 모든 환경을 업그레이드합니다. 이전 릴리스가 기록한 float, 양자화, 학습 체크포인트는 1.7.0에서 불러올 수 있으며 ONNX 내보내기는 영향을 받지 않습니다.

- YOLO9 계열 모델의 `model.model`에 접근하는 코드는 별칭 없이 MultimediaTechLab/YOLO 모듈 이름을 보게 됩니다. `DDetect`는 `YOLO9Head`(`anchor_convs` / `class_convs`), `RepConvN`은 `RepConv`, `RepNBottleneck`은 `Bottleneck`, `DFL`은 `Anchor2Vec`(`anchor2vec`), `nc`는 `num_classes`이며, 블록 하위 레이어는 `conv1`/`conv2`/`bottleneck`입니다. YOLO9-E2E에서는 `YOLO9E2EDetect`가 `YOLO9E2EHead`, `one2one_cv2` / `one2one_cv3`가 `one_to_one_anchor_convs` / `one_to_one_class_convs`입니다. `LibreYOLO9Model.fuse()`, `RepConvN.fuse_convs()`, `Conv.forward_without_bn()`, `Concat`, `yolo9.utils.decode_boxes`는 제거되었습니다. 모든 이름 변경은 [변경 이력](/docs/changelog)에 나열되어 있습니다.

- ONNX 노드 이름은 헤드, 백본, 넥에서 새 모듈 이름을 따르며, 예를 들어 `/head/anchor_convs.0/...`입니다. Hailo 끝 노드 설정처럼 노드 이름을 기준으로 하는 도구를 갱신합니다. 모델 입력, 출력, 출력 이름은 바뀌지 않습니다.

- PGI 텐서가 없는 가중치에서 시작하는 YOLO9 파인튜닝은 공개된 `LibreYOLO9{t,s,m,c}.pt`를 포함해 1.5.0처럼 기본 헤드만 학습합니다. 1.6.0은 `aux_weight=0.25`에서 무작위로 초기화된 PGI 분기를 붙였습니다. PGI 텐서가 있는 가중치와 처음부터 학습하는 실행은 여전히 PGI로 학습합니다. 그래도 새 분기를 붙이려면 `aux_weight=0.25`를, 확실히 끄려면 `aux_weight=0`을 전달합니다.

- PGI가 붙으면 YOLO9-m과 YOLO9-c는 MultimediaTechLab의 보조 분기를 만듭니다. 클래스 80개 기준으로 학습 전용 파라미터가 기존 7.5M과 10.2M에서 12.8M(m)과 25.7M(c)으로 늘어나므로, 이 크기의 PGI 학습은 스텝당 메모리와 시간이 더 필요합니다. 1.6.0 m/c 학습 체크포인트는 학습에 사용한 분기 그대로 학습을 재개하고 파인튜닝합니다. 추론과 YOLO9-t/s는 바뀌지 않습니다.

- YOLO9, YOLO9-E2E, YOLO9-P2는 그래디언트 L2 노름을 10으로 클리핑하며, 학습 할당기는 앵커 점수를 fp32로 계산하므로 fp16 AMP에서 선택되던 양성 샘플이 달라질 수 있습니다. 1.6.0의 클리핑을 쓰려면 `clip_max_norm=0`을 전달합니다. 1.6.0에서 시작해 1.7.0에서 재개한 실행도 이를 전달하지 않으면 클리핑합니다. fp32 할당을 되돌리는 설정은 없습니다.

- 0이 아닌 `perspective`는 공유 아핀 워프를 쓰는 모든 계열(YOLO9와 그 E2E, P2 변형, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE, RTMDet, PicoDet, RT-DETR, RT-DETRv2)에서 모서리 변위로 왜곡하며, 이를 설정한 1.6.0 체크포인트에서 재개한 실행도 마찬가지입니다. `0.002` 이상의 값은 같은 최대 왜곡을 주고, 음수 값은 `ValueError`를 발생시킵니다. 1.6.0의 왜곡을 재현하는 설정은 없으며, 기본값 `perspective=0`은 바뀌지 않습니다.

- YOLO9 모자이크 파이프라인은 mixup 비율을 Beta(32, 32) 대신 Beta(1, 1)에서 추출하며, 이전 값을 위한 설정은 없습니다. YOLOv7은 mixup을 기본으로 켜므로 기본 YOLOv7 학습이 달라집니다. YOLO9, YOLO9-E2E, YOLO9-P2, RT-DETR, RT-DETRv2는 mixup을 켠 경우에만 달라집니다.

- `name=` 없이 `libreyolo train`을 실행하면 EC, D-FINE, SegFormer, LingBot-Vision 실행은 `runs/train/exp` 대신 `runs/train/ec_exp`, `dfine_exp`, `segformer_exp`, `lingbotvision_exp`에, V-JEPA 2 프로브는 `runs/train/train`에, YOLO9-P2 실행은 `runs/train/yolo9_exp`에 기록됩니다. 이전 경로를 쓰려면 `name=exp`를, YOLO9-P2에는 `name=yolo9_p2_exp`를 전달합니다.

- `libreyolo train`은 D-FINE, SegFormer, LingBot-Vision을 각자의 레시피로 시작합니다. D-FINE의 경우 132 에폭, `lr0=2e-4`, `weight_decay=1e-4`, 워밍업 2 에폭이며 모자이크, mixup, HSV, 아핀 증강은 없습니다. 1.6.0이 보내던 값을 보내려면 `epochs=300 optimizer=sgd lr0=0.01 weight_decay=5e-4 warmup_epochs=5 mosaic=1.0 mixup=1.0 hsv_prob=1.0 degrees=10 translate=0.1 shear=2.0`을 전달합니다.

- `libreyolo.validation.ConfusionMatrix`는 `matrix[predicted, true]`로 인덱싱하고(이전에는 `[actual, predicted]`), `conf_thres=0.25`에서 탐지를 세며(이전에는 0.15), `names=`와 `task=`는 키워드 인자로만 받고, `matrix`는 더 이상 다시 할당할 수 없습니다. 이전 행렬을 인덱싱하던 코드는 전치하고, 이전 집계를 쓰려면 `conf_thres=0.15`를 전달합니다. `plots=True`는 개수를 `confusion_matrix.png`에, 정규화된 그림을 `confusion_matrix_normalized.png`에 기록합니다.

- `predict(agnostic_nms=True)`는 경고하는 대신 클래스에 걸쳐 겹치는 박스를 억제합니다. 박스가 없는 작업은 `ValueError`를, `LibreEnsemble`과 HRNet은 `NotImplementedError`를 발생시키고, SAM 계층은 이 키워드를 `TypeError`로 거부하므로 해당 호출에서는 이 인자를 제거합니다.

- 알 수 없는 분류기 `interpolation`은 torchvision 모드인 `LANCZOS`, `BOX`, `HAMMING`, `NEAREST_EXACT`를 포함해 `ValueError`를 발생시킵니다. `bilinear`, `bicubic`, `nearest`, `bilinear_no_antialias`를 사용합니다.

- 1.6.0 수치와 비교하기 전에 여러 클래스로 학습한 모델의 `val(single_cls=True)`, 박스만 있는 행과 폴리곤 행이 섞인 분할 데이터셋, 그리고 기본적으로 이미지당 최대 `max_det=300`개의 마스크를 유지하는 RTMDet-Ins에서 검증을 다시 실행합니다. MMDetection의 설정을 쓰려면 `max_det=100`을 전달합니다.

- `libreyolo[rfdetr]`가 없으면 RF-DETR 또는 DINOv2 모델 이름으로 CLI를 호출할 때 `model_not_found` 대신 JSON 오류 코드 `model_load_failed`로 종료하며, 메시지는 `pip install libreyolo[rfdetr]`를 알려 줍니다.

전체 릴리스는 [변경 이력](/docs/changelog)을 참조하십시오.

## 1.5.0에서 1.6.0으로

- ONNX 추가 패키지를 설치하기 전에 ONNX Runtime 1.18 미만으로 고정된 환경을 `onnxruntime>=1.18.0`으로 업그레이드합니다.

- SAM 3D Body는 검토된 스냅샷 파일 집합을 받습니다. `libreyolo[hf]`를 설치하고 접근 권한을 얻어 자동 다운로드를 사용하거나, 신뢰할 수 있는 로컬 파일시스템에서 변경되지 않은 스냅샷 디렉터리와 고정된 MHR 파일을 제공합니다.

- RF-DETR의 자세 이외 작업 크기 조정과 분류 모델 정규화 수정 이후 검증을 다시 실행합니다. 1.5.0 결과와 비교하기 전에 배포된 신뢰도 임계값을 다시 확인하며, 기존 전처리 플래그는 없습니다.

- D-FINE, DEIM, RT-DETRv4, YOLO-NAS 탐지는 기본적으로 FP16 AMP를 활성화합니다. FP32를 유지하려면 `amp=False`를 전달합니다.

- 이전 YOLO9 파인튜닝 설정을 사용하려면 `aux_weight=0`, `max_labels=100`, `warmup_momentum=0.937`을 설정합니다. center가 새로 기록된 변환에서 기존 기하를 재현해야 하면 `letterbox_pad="topleft"`를 설정합니다. 기존 단일 헤드 체크포인트는 원래 그래프로 학습을 재개합니다. 새 YOLO9 실행은 GPU 메모리를 더 많이 사용하므로, 1.5.0의 배치 크기에서 메모리가 부족하면 `batch`를 낮춥니다.

- RF-DETR과 DINOv2는 계열 이름에 번호가 증가하는 실행 디렉터리를 생성합니다. 산출물 경로를 사용하는 코드를 갱신하거나, 기존 위치와 재사용 동작을 유지하려면 `output_dir="runs/train", exist_ok=True`를 설정합니다.

- 분류의 `mixup + cutmix <= 1`을 유지합니다. 잘못된 조합은 이제 설정 중 오류를 발생시킵니다.

- QAT는 EMA, SyncBatchNorm, 체크포인트 평균을 비활성화합니다. 이 상태에 의존하지 않고 QAT의 best/last 체크포인트를 사용합니다.

- 데이터셋 변경 훅이 있는 사용자 정의 로더는 `persistent_workers=False`를 사용하거나 변경 후 작업자를 다시 생성해야 합니다. 호환되지 않는 영구 다중 작업자 사본은 이제 오류를 발생시킵니다.

- NumPy 이미지 배열은 OpenCV 순서인 BGR로 읽힙니다. `np.asarray(pil_image)` 같은 RGB 배열을 전달하는 곳에서는 `color_format="rgb"`를 전달하며, `cv2.imread()` 출력과 비디오 프레임은 변경할 필요가 없습니다.

- 4D NumPy 배열이나 텐서는 배치입니다: `predict()`는 첫 번째 이미지만 사용하는 대신 이미지마다 `Results` 하나씩을 담은 리스트를 반환합니다.

- `train(resume=True)`와 `train(resume="path/to/last.pt")`는 실행에 저장된 학습 인수를 복원하고 그 실행 디렉터리에 계속 기록하며, 명시적으로 전달한 인수가 우선합니다. 공개된 가중치나 이미 `epochs`에 도달한 실행을 재개하면 `ValueError`가 발생합니다.

- 검증이 켜져 있으면 마지막 에폭은 항상 검증하므로, `eval_interval`보다 짧은 실행도 이제 지표를 보고하고 `best.pt`를 기록합니다. `val=False`는 마지막 에폭을 포함해 검증을 끕니다. 학습 중 검증은 `runs/val/` 대신 `<run>/val`에 기록합니다.

- CLI `train` 실패는 오류 유형을 보고하므로, 설정 오류는 `io_error`와 함께 코드 1로 종료하는 대신 코드 2로 종료합니다.

전체 릴리스는 [변경 기록](/docs/changelog)을, 체크포인트 변환은 [가중치 가져오기](/docs/migrate)를 참조하십시오.

## 1.4.0에서 1.5.0으로

이 페이지는 LibreYOLO 자체를 업그레이드하는 것에 관한 것입니다. 업스트림 프로젝트에서 체크포인트를 로드하는 방법을 찾고 있다면, 이는 [기존 가중치 가져오기](/docs/migrate)로, 다른 주제입니다.

릴리스의 전체 변경 사항은 [변경 로그](/docs/changelog)에 있습니다. 아래에는 조치가 필요한 항목만 나옵니다.

### 해야 하는 코드 변경

#### `allow_experimental=True` 제거

승인 게이트와 그 뒤에 있는 `ddp_aware(experimental_key=...)` 메커니즘이 사라졌습니다. 이전에는 EC, RTMDet, PicoDet 및 FOMO 학습 및 내보내기에 해당 인수가 필요했기 때문에, 이러한 계열 중 하나를 학습하는 모든 스크립트에 영향을 미칩니다.

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0: 인수를 삭제
model.train(data="data.yaml", epochs=100)
```

더 이상 사용 중단 대체 코드는 없습니다. 여전히 그것을 전달하는 호출은 `Unknown training config keys (ignored)` 경고를 받으며, 인수는 아무 효과가 없습니다. `BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES`는 그것과 함께 제거되었습니다. `get_download_notice()` 훅은 남아 있으며 여전히 MiDaS, SegFormer 및 YOLO9-P2에 의해 재정의됩니다.

지원 수준은 여전히 게시되지만 더 이상 인수가 아닙니다: [안정성 등급](/docs/reference/stability-tiers)을 참조하십시오.

#### 내보내기 계층 `"experimental"` 제거

```python
from libreyolo.export.support import Tier

# 1.4.0: 리터럴["validated", "experimental", "blocked"]
# 1.5.0: 문자 그대로["검증됨", "사용 가능", "차단됨"]
```

티어 문자열에서 코드 분기는 `"experimental"`로 읽히던 곳을 `"available"`로 읽어야 합니다. `BaseExporter`는 더 이상 해당 형식에 대해 `RuntimeWarning`를 방출하지 않습니다. 형식별 상태는 [export matrix](/docs/reference/export-matrix)에 나와 있습니다.

#### `pretrained=False`와 `resume` 조합 거부

이전에 결합은 일관성 없이 진행되었습니다. 이제 다음을 제기합니다:

```
ValueError: pretrained=False cannot be combined with resume.
```

하나를 선택하십시오. `pretrained=False`는 새로 초기화된 시드에서 시작하며, 1.5.0에서는 세 가지가 아니라 모든 학습 가능한 계열에 대해 작동합니다. `resume`는 체크포인트에서 중단된 실행을 계속합니다. 두 가지 모두 [학습](/docs/train) 아래에 문서화되어 있습니다.

#### CLI `--imgsz`는 문자열이지 정수가 아닙니다

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

### 변하는 숫자들

세 가지 변경 사항이 기본 설정에서 지표를 움직입니다. 버전별 결과를 추적하는 경우, 1.5.0 실행과 1.4.0 실행을 비교하기 전에 이것들을 읽으십시오.

#### 기본 COCO 지표 백엔드 faster-coco-eval

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

#### YOLOX 1.5.0 이전에 학습된 체크포인트는 eps 재정의가 필요

이것은 릴리스의 함정입니다. [YOLOX](/docs/models/yolox)를 파인튜닝한 경우 읽어보십시오.

YOLOX는 BatchNorm `eps=1e-3` 및 `momentum=0.03`를 지정합니다. 1.5.0 이전까지 이러한 값들은 후속 보정으로 적용되었으며, 데이터셋의 `nc`가 체크포인트의 것과 다를 때 `train()`가 수행하는 클래스 수 재구성에서는 유지되지 않았습니다. 이러한 세부 조정은 torch의 기본 `eps=1e-5`에서 학습 중 검증을 보고한 후 `1e-3`에서 추론을 위해 다시 로드되었습니다: 서로 다른 정규화에서 동일한 텐서들입니다.

일반 컨브 크기는 거의 움직이지 않습니다. Depthwise `n`는 많이 움직이는데, 이는 채널별 `running_var`가 eps가 지배할 만큼 충분히 작기 때문입니다. RF100-VL `ball`에서는 동일한 나노 체크포인트가 학습된 eps에서 평가했을 때 **0.566** mAP50-95을 기록하고, 초기 상태로 다시 로드한 후에는 **0.151**을 기록합니다.

1.5.0 이전에 학습된 체크포인트는 eps=1e-5 의미를 가집니다. 이를 위해 정확한 수치를 보고하려면, BN eps를 1e-5로 오버라이드하여 평가하십시오:

```python
import torch
from libreyolo import LibreYOLO

model = LibreYOLO("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

또는 `sqrt((var + 1e-3) / (var + 1e-5))`를 BN 가중치에 한 번 접어 결과를 저장하십시오. 1.5.0 이후에 학습된 체크포인트는 필요하지 않습니다.

#### D-FINE 다중 스케일 학습은 업스트림 크기별 레시피를 사용

`base_size_repeat`는 모든 크기에 대해 3으로 하드코딩되어 있었습니다. 이제 업스트림에서 지정한 대로 크기별로 해결됩니다: **n**은 멀티 스케일이 꺼진 상태에서 고정 크기로 학습, **s** 20, **m** 6, **l** 4, **x** 3. 이전에는 x만 일치했기 때문에, n, s, m, l은 다른 스케일 분포를 보게 되며 다른 지표로 수렴합니다.

이전 동작을 복원하려면 명시적으로 설정하십시오:

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIM은 여전히 하드코딩된 3을 사용합니다. 계열 세부 사항은 [D-FINE](/docs/models/d-fine)에 있습니다.

### 알아둘 만함, 조치 필요 없음

- **직사각형 `imgsz` 결과가 이전에 잘못되어 변경되었습니다.** 박스 좌표, RTMDet 마스크 크기 조정, YOLO-NAS 재스케일링 및 검증기 실제 값 스케일링은 이제 하나의 스칼라 대신 축별 높이와 너비를 사용합니다. 정사각형 `imgsz`는 거의 변경되지 않았습니다. 1.4.0에서의 직사각형 추론 또는 검증 실행은 잘못 스케일링되었습니다. YOLO-NAS는 이제 잘못된 출력을 조용히 생성하는 대신 직사각형 `imgsz`를 아예 거부합니다.
- **측정 지표 사전이 키를 얻었습니다.** COCO 평가기에서는 `max_det`, `ar_max_det` 및 `AR_max_det`를, FOMO에서는 `metrics/loss`와 `metrics/loss/ce`를 추가했습니다. 기본값의 값은 변경되지 않았지만, 사용자 정의 [로거](/docs/train/loggers)와 CSV 헤더를 포함하여 측정 지표 키를 반복하는 모든 항목은 새로운 열을 확인합니다.
- **헤드 재구성을 트리거하는 시드 YOLO9 실행**은 재구성 전에 시드가 적용되기 때문에 서로 다른 초기화에서 시작됩니다. 다른 클래스 수에 대해 시드 1.4.0로 파인튜닝한 결과는 1.5.0에서 비트 단위로 재현할 수 없습니다.
- **`libreyolo[hub-kernels]`가 이제 CUDA에서 실제로 네이티브 MS-deform-attn 커널을 사용합니다.** 1.4.0에서는 RF-DETR가 절대 사용하지 않는 조건 뒤에 이를 제한했기 때문에 커널이 실행되지 않았습니다. RF-DETR 및 다른 변형 주의(attention) 계열에서 예측 값이 부동소수점 허용 오차 내에서 변동할 수 있습니다. 기본 설치에는 영향을 주지 않으며, `LIBREYOLO_HUB_KERNELS=0`는 이를 비활성화합니다.
- **`libreyolo predict`은 오류를 발생시키는 대신 지원되지 않는 옵션을 제거합니다.** CLI는 모델의 `__call__` 시그니처에 대해 kwargs를 필터링하므로, 계열가 허용하지 않는 옵션은 `TypeError`을 발생시키는 대신 무시됩니다. 알 수 없는 플래그 이름은 여전히 `No such option`으로 거부됩니다.
- **실시간 소스는 JSON 출력 형태를 변경합니다.** 웹캠, RTSP 스트림 및 화면 캡처는 암묵적으로 스트리밍을 활성화하며, 호출당 하나의 레코드가 아니라 프레임당 하나의 레코드를 생성합니다. 이 [소스](/docs/predict/sources)는 1.5.0에서 새로 추가되었으므로 1.4.0 스크립트에는 영향을 주지 않습니다.
- **`rfdetr-pose` 또는 `yolonas-pose`를 ONNX로 다시 내보내면 출력 이름이 다르게 나옵니다.** 1.4.0은 출력 수 기반의 휴리스틱을 통해 이들의 다중 텐서 포즈 헤드를 세분화(segmentation)로 잘못 읽었습니다. 디스크에 있는 기존 `.onnx` 파일은 그대로 유지됩니다.
- **토치가 없는 설치**에서는 결과가 `torch.Tensor`가 아닌 numpy 배열을 유지하므로, `.boxes.data`는 다른 유형을 반환하며 NMS 동점 처리 방식이 torchvision과 다를 수 있습니다. 토치가 설치되어 있으면 동작은 바이트 단위까지 동일합니다. [경량 설치](/docs/lightweight-install)를 참조하십시오.
- **구성 객체는 생성 시 더 많이 검증됩니다.** `TrainConfig`는 이전에는 없던 `__post_init__`를 얻었으므로, 이미 유효하지 않았던 구성은 실행 중 깊이에서 실패하는 대신 즉시 오류를 발생시킵니다. `ValidationConfig` 직렬화는 `edge_thresholds` 키를 얻었으며, 이는 1.4.0 덤프에서 엄격한 `ValidationConfig(**dump)` 왕복을 깨뜨립니다.
- **작업 접미사가 있는 계열의 가중치 파일 이름이 다르게 해석됩니다.** `segformer-b0`는 이제 `LibreSegformerb0-sem.pt`로 해석됩니다. 이는 자동 다운로드 404 오류를 수정하고, 이전에 접미사가 없는 파일 이름을 하드코딩한 모든 스크립트를 깨뜨립니다.
- **pytest 마커 `experimental_backend`는 이제 `extended_backend`입니다.** `-m`로 테스트 스위트를 실행할 때만 관련이 있습니다.

### 체크포인트와 데이터셋

1.4.0으로 작성된 체크포인트는 변경 없이 로드됩니다. [스키마](/docs/reference/checkpoint-schema)는 직사각형 모델을 위해 `imgsz_h`와 `imgsz_w`를 추가했으며, 이전 리더를 위해 여전히 스칼라 `imgsz = max(h, w)`를 작성합니다. [ExecuTorch](/docs/export/executorch)와 [MNN](/docs/export/mnn) 내보내기는 이제 각각 사이드카 `<program>.pte.json`와 `<model>.mnn.json`가 필요하며, HRNet 내보내기는 `pose_input: "person_crop"`를 포함합니다. 데이터셋 형식은 변경되지 않았습니다.
