---
title: 검증 및 지표
seo_title: LibreYOLO에서의 검증 및 지표
description: >-
  어떤 모델에서든 val()을 실행하고, 각 작업이 반환하는 메트릭 키를 읽고, 평가 백엔드를 선택하며, 정확도 메트릭과 함께 검증 손실을
  켭니다.
lead: >-
  검증(validation)은 모델을 val()을 통해 나눈 데이터셋에 실행하고, 메트릭 키와 실수값(Float value)으로 이루어진 평면
  딕셔너리를 반환합니다. 키는 문자열이며, 어떤 키를 얻는지는 모델 계열이 아니라 작업(task)에 따라 달라집니다.
keywords:
  - map50-95
  - 코코 평가
  - 검증 지표
  - 더 빠른 코코 평가
  - 파이코코툴스
  - 검증 손실
  - 미우
  - 전면적 품질
  - Top1 정확도
last_verified: 1.5.0
snippets:
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["speed/total_ms"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: 다른 분할에서
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml", split="train", batch=4)

        print(metrics)
  valloss:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, val_loss=True)
  json:
    - label: COCO 형식 예측 작성
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.val(data="coco8.yaml", save_json=True, save_dir="runs/val/exp")
source_hash: d907183492fa3f57
---

## 검증을 실행

`val()`는 데이터셋을 가져와 메트릭을 반환합니다.

<code-tabs name="val" />

반환 값은 일반 `dict[str, float]`입니다. 모든 키는 문자 그대로이므로 위치가 아니라 이름으로 읽으십시오.

주요 인수는 `data`, `split`, `batch`, `imgsz`, `conf`, `iou`, `workers`, `device`, `augment`, `save_json` 및 `verbose`입니다. `conf`는 `0.001`로 기본 설정되며, `iou`는 `0.6`로 기본 설정되는데, 둘 다 예측 기본값보다 훨씬 느슨합니다. 이는 mAP 스윕이 낮은 신뢰도의 꼬리값을 필요로 하기 때문입니다. `imgsz`는 고정 숫자 대신 모델 자체 입력 크기를 기본값으로 사용합니다. `split`는 `val`, `test` 또는 `train`만 허용하며 그 외는 허용하지 않습니다.

검증 구성의 다른 모든 필드는 키워드 인수로 전달되며, 여기에는 `save_dir`, `max_det`, `eval_max_det`, `half`, `amp_dtype`, `cache` 및 `save_plots`가 포함됩니다.

## 작업별 메트릭 키

탐지는 COCO 계열 숫자를 반환합니다:

```text
metrics/mAP50-95   metrics/mAP50    metrics/mAP75
metrics/mAP_small  metrics/mAP_medium  metrics/mAP_large
metrics/AR1  metrics/AR10  metrics/AR100  metrics/AR_max_det
metrics/AR_small  metrics/AR_medium  metrics/AR_large
metrics/precision  metrics/recall
metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)  metrics/mAP50-95(B)
```

그 중 두 개는 함정입니다. `metrics/precision`와 `metrics/recall`는 이전 호환성을 위해 보유된 별칭입니다: 이들은 정밀도와 재현율 쌍이 아니라 mAP 50-95와 AR@100 값을 포함합니다. 이름이 지정된 키를 사용하십시오.

인스턴스 분할은 위의 mAP 및 AR 수치를 접미사가 없는 키 아래에서 마스크 수치로 반환하며, 박스 버전은 `(B)` 접미사 아래에, 마스크 버전은 `(M)` 아래에 반복됩니다. 정밀도와 재현율은 이 작업에서는 접미사 형태로만 존재하며 `metrics/precision(B)`/`metrics/recall(B)` 및 `metrics/precision(M)`/`metrics/recall(M)`로 나타나고, 두 쌍 모두 detect의 별칭 값과 동일합니다: `(B)` 쌍은 박스 mAP50-95와 박스 AR@100이며, `(M)` 쌍은 마스크 mAP50-95와 마스크 AR@100입니다.

| 작업 | 열쇠 |
|---|---|
| 탐지하다 | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, 위의 크기 및 리콜 분류와 함께 |
| 세그먼트 | 위의 탐지 키의 마스크 버전(접미사가 없는 키는 마스크임); `precision`/`recall`는 `(B)`/`(M)`로만 존재하며, 둘 다 동일하게 별칭이 지정됨 |
| 자세 | `metrics/keypoints_mAP50-95`, `metrics/keypoints_mAP50`, `metrics/keypoints_mAP75`, `metrics/keypoints_mAP_M`, `metrics/keypoints_mAP_L`, 그리고 일치하는 `keypoints_AR` 키 |
| obb | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, `metrics/precision`, `metrics/recall`, 그리고 `(OBB)`가 접미사로 붙은 복사본 |
| 분류하다 | `metrics/accuracy_top1`, `metrics/accuracy_top5` |
| 의미론의 | `metrics/mIoU`, `metrics/pixel_accuracy` |
| 파노프틱 | `metrics/PQ`, `metrics/SQ`, `metrics/RQ`, `metrics/PQ_things`, `metrics/PQ_stuff`, `metrics/categories` |
| 깊이 | `metrics/abs_rel`, `metrics/rmse`, `metrics/delta1`, `metrics/delta2`, `metrics/delta3` |
| 보통 | `metrics/mean_angular_error`, `metrics/median_angular_error`, `metrics/within_11_25`, `metrics/within_22_5`, `metrics/within_30` |
| 모서리 | `metrics/ODS`, `metrics/OIS`, `metrics/best_threshold` |
| 복원하다 | `metrics/PSNR`, `metrics/SSIM` |
| 매트 | `metrics/MAE`, `metrics/Smeasure` |
| 광학 문자 인식 | `metrics/det_precision`, `metrics/det_recall`, `metrics/det_hmean`, `metrics/e2e_precision`, `metrics/e2e_recall`, `metrics/e2e_f1`, `metrics/rec_1-NED` |
| 점 | `metrics/precision`, `metrics/recall`, `metrics/f1`, `metrics/MLE`, `metrics/MAE`, `metrics/RMSE`, 그리고 mAP 스윕 키 |

OBB의 `metrics/precision`와 `metrics/recall`는 별칭이 아닙니다: 이들은 실제 정밀도와 IoU 0.50에서의 재현율로, 가장 느슨한 작동 지점에서 취한 값입니다(모든 `conf`를 통과한 예측, 기본값 `0.001`). `(OBB)`로 끝나는 복사본들은 작업별 이름 아래 같은 네 가지 값을 반복하며, 위의 `(B)`와 `(M)`와 동일한 규칙을 따릅니다.

`accuracy_top5`는 실제로 top-`min(5, num_classes)`이므로, 세 가지 클래스 데이터셋에서는 top-3가 되며, 이는 모든 샘플이 만족하므로 1.0으로 읽습니다.

점 작업의 스윕 키는 거리 임계값으로 작성되므로, 기본값으로는 `metrics/mAP@[0.01:0.10]`를 읽고 단일 임계값 키는 `metrics/mAP@0.01`를 읽습니다. `dist_thresholds`를 전달하면 두 문자열이 모두 변경됩니다.

대부분의 작업은 또한 `fitness` 키를 반환하며, 단일 숫자 최적 체크포인트 선택에서 기본적으로 사용됩니다. 탐지, 분할 및 OBB는 이를 포함하지 않으며; 그들의 계열는 `metrics/mAP50-95`에서 선택되며, 해당 딕셔너리는 반환합니다. 포즈는 `fitness`나 `metrics/mAP50-95`를 반환하지 않으며; 트레이너는 대신 `best_metric_key`를 `metrics/keypoints_mAP50-95`로 설정합니다.

## 속도 키

모든 검증자가 타이밍을 추가합니다:

```text
speed/preprocess_ms   speed/inference_ms   speed/postprocess_ms
speed/total_ms        speed/total_s        speed/images_seen
```

이 값들은 실행 동안 이미지당 평균 밀리초를 나타냅니다. 이것들은 실행한 기계와 설정을 설명하므로, 여기서 얻은 수치는 하드웨어, 배치 크기, 정밀도와 함께 보고될 때만 의미가 있습니다.

## 평가 백엔드

검출 및 분할 지표는 COCO 평가기를 통해 계산되며, 기본값인 `faster_coco_eval=True`는 `faster-coco-eval` 패키지가 설치되면 C++ 백엔드를 선택합니다. 설치되지 않은 경우, 실행은 프로세스당 하나의 경고와 함께 pycocotools로 대체됩니다:

```text
faster_coco_eval requested but not installed; falling back to pycocotools.
Install with: pip install faster-coco-eval
```

실제로 실행된 백엔드는 모델에 `last_eval_backend`로 기록되며, CLI는 검출 스타일 작업의 출력에서 이를 보고합니다. `LIBREYOLO_FASTER_COCO_EVAL`를 설정하여 환경에서 가져온 구성 값을 덮어쓸 수 있습니다.

`iou_thresholds`는 OBB 경로에서만 인정됩니다. COCO 경로는 자체 고정된 0.50에서 0.95 범위로 평가하며 값을 무시합니다.

## 검증 손실

기본적으로 검증은 정확도만 보고합니다. `val_loss=True`는 검증 배치에서 계열의 학습 목표도 계산합니다.

<code-tabs name="valloss" />

그것은 `metrics/loss`를 방출하고 각 항마다 `metrics/loss/<component>`를 추가하며, 학습 시 가중치가 매기는 방식과 정확히 동일하게 가중치를 적용하여 구성 요소들이 총합이 되도록 합니다. 로거를 통해 이것들은 `val/loss`와 `val/loss/<component>`로 나타나며, `libreyolo monitor`는 `metrics/loss` 위에 `train/loss`를 오버레이합니다.

구성 요소는 계열 자신의 것입니다:

| 작업 | 계열들 | 구성 요소 |
|---|---|---|
| 탐지하다 | `yolo9`, `yolo9_p2`, `yolo9_e2e` | `box`, `cls`, `dfl` |
| 탐지하다 | `yolonas` | `cls`, `iou`, `dfl` |
| 탐지하다 | `rfdetr` | `ce`, `bbox`, `giou` |
| 탐지하다 | `rtdetr`, `rtdetrv2` | `vfl`, `bbox`, `giou` |
| 탐지하다 | `dfine` | `vfl`, `bbox`, `giou`, `fgl`, `ddf` |
| 탐지하다 | `domedetr` | `vfl`, `bbox`, `giou`, `fgl`, `ddf`, `defe_density`, `defe_reg` |
| 탐지하다 | `deim`, `deimv2`, `rtdetrv4`, `ec` | `mal`, `bbox`, `giou`, `fgl`, `ddf` |
| 탐지하다 | `rtmdet` | `cls`, `bbox` |
| 탐지하다 | `picodet` | `cls`, `bbox`, `dfl` |
| 탐지하다 | `yolox` | `iou`, `obj`, `cls`, `l1` |
| 탐지하다 | `yolo7` | `iou`, `obj`, `cls` |
| 점 | `fomo` | `ce` |
| 분류하다 | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` | `ce` |
| 의미론의 | `segformer`, `lingbotvision`, `dinov2` | `sem` |
| 복원하다 | `nafnet` | `restore` |

기본적으로 비활성화되어 있습니다. 목표 할당이 검증에 시간과 메모리를 추가하기 때문입니다. 검증기는 두 번째 순방향 계산을 수행하지 않고 정확도 지표를 위해 이미 생성된 모델 출력을 재사용하며, 평가 또는 EMA 모델에서는 `no_grad` 아래에서 실행되고, 다중 GPU 학습에서는 수집 작업 없이 rank 0에서 로컬로 계산됩니다. 최적 체크포인트 선택은 정확도 지표에 따라 유지됩니다.

의도적으로 하지 않는 세 가지가 있습니다. 첫째, 대비-디노이징(contrastive-denoising) 항목을 포함하지 않는데, 이는 전방 계산 시 실제 값이 필요하고 검증 전방 계산에서는 필요하지 않기 때문입니다. 둘째, 평가 모드 모델을 보고하기 때문에, 만약 어떤 계열의 학습 및 평가 전방 계산이 실제로 다르다면(BatchNorm 통계나 확률적 깊이(stochastic depth)에서), 그 수치는 평가 모드(eval mode)를 반영합니다; 이것이 의도된 비교입니다. 셋째, 계열가 구현하지 않은 작업(task)에 대해서는 조용히 건너뛰지 않고 설정(setup) 시 구성 오류(configuration error)를 발생시킵니다:

```text
val_loss=True currently supports RF-DETR detection only; segment, pose, OBB,
classify, and semantic tasks are not supported
```

FOMO는 아무것도 바꾸지 않는 예외입니다: 그 검증자는 항상 이 손실을 계산했고, `val_loss=True`는 단지 어느 키로 게시되는지에만 영향을 줍니다.

증강된 검증과 검증 손실은 결합할 수 없으며, 둘 다 요청하면 오류가 발생합니다.

## 파일이 검증을 작성

`val()`는 항상 `config.yaml`를 자신의 저장 디렉토리에 작성하며, `save_dir`가 제공되지 않은 경우 기본적으로 `runs/val/<model>_<size>_<timestamp>`를 사용합니다.

<code-tabs name="json" />

`save_json=True`는 탐지를 위해 `predictions.json`를 작성하고, 분할을 위해 `predictions_bbox.json`와 `predictions_masks.json`를 작성합니다. OBB는 이를 지원하지 않으며 그렇게 말합니다.

`save_plots=True`는 `plots/` 하위 디렉토리에 작성합니다. 탐지에서는 OpenCV가 설치된 경우 `box_metrics.png`, 클래스별 AP 및 리콜 차트, 정밀도-재현율 및 신뢰도 곡선, 혼동 행렬, 주석이 달린 샘플 이미지를 가져옵니다. 분할(segmentation)은 각 항목의 마스크 측 사본을 추가하며, 포즈(pose)는 자체 메트릭 및 곡선 세트를 가져옵니다. 다른 검증기는 플롯을 구현하지 않습니다. 분류, 의미론적, 파노라마, 깊이, 노멀, 엣지, 복원, 매트, OCR, OBB 및 포인트는 모두 해당 위치에 아무것도 작성하지 않습니다. 플로팅 실패는 경고만 하며 실행을 절대 중단하지 않습니다.

## 학습 중 검증

학습은 데이터셋의 `val` 분할에 대해 매 `eval_interval` 에폭마다 검증을 수행하며, 생성되는 지표는 `best.pt` 선택, `patience` 조기 종료, 그리고 모든 로거의 `val/` 키를 결정하는 데 사용됩니다. EMA가 켜져 있을 때 검증은 EMA 가중치로 실행됩니다.

`eval_interval`, `patience` 및 `save_plots`에 대해서는 [하이퍼파라미터](/docs/train/hyperparameters)를 참조하고, 숫자가 어디에 들어가는지에 대해서는 [실험 로거](/docs/train/loggers)를 참조하십시오.

## 관련된

- [데이터셋](/docs/train/datasets)은 분할 키와 포맷 검증기를 읽습니다.
