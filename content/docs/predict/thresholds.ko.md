---
title: 임계값과 필터링
seo_title: 'LibreYOLO에서 conf, iou 및 max_det'
description: >-
  예측 시 conf, iou, max_det, classes, agnostic_nms가 실제로 무엇을 하는지, 그리고 어떤 계열이 자체
  NMS를 실행하지 않아 iou를 무시하는지.
lead: >-
  네 가지 인자가 어떤 예측이 살아남는지를 결정합니다: conf, iou, max_det 및 classes. 그 중 두 가지만 모든 계열에
  적용되는데, 이는 세트 예측기가 고정된 쿼리 세트를 디코딩하고 NMS를 절대 실행하지 않기 때문입니다.
keywords:
  - yolo 신뢰도 임계값
  - iou 임계값 nms
  - 최대_탐지
  - 클래스 필터 탐지 파이썬
  - 불가지론적 NMS
  - nms 무료 detR
  - 탐지 신뢰도 임계값
  - 클래스 필터링 추론
last_verified: 1.6.0
verification: >-
  libreyolo/models/base/inference.py.의 InferenceRunner.__call__에서 따온 기본값,
  libreyolo/postprocess/의 모든 모듈에서 읽은 계열별 NMS 동작 및 libreyolo/backends/base.py.의
  _is_nms_free_family와 교차 확인, InferenceRunner._apply_classes_filter 및
  _wrap_results에서의 클래스 필터링, libreyolo/utils/predict_args.py.의
  NOOP_PREDICT_KWARGS에서의 agnostic_nms 상태, libreyolo/models/openvocab/base.py.의
  NMS_THRESHOLD에서의 오픈보캐뷸러리 처리, BaseModel.val에서의 검증 기본값.
snippets:
  basic:
    - label: 네 가지 논점
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(
            SAMPLE_IMAGE,
            conf=0.25,      # 예측을 이 점수 이상으로 유지하십시오
            iou=0.45,       # NMS 겹침 임계값, NMS가 실행되는 위치
            max_det=300,    # 이미지당 캡
            classes=None,   # 또는 클래스 ID 목록
        )
        print(len(result.boxes))
    - label: 스위핑 설정
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        for conf in (0.1, 0.25, 0.5, 0.75):
            result = model(SAMPLE_IMAGE, conf=conf)
            print(conf, len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt conf=0.4 iou=0.5 max_det=100 \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  classes:
    - label: 특정 클래스 필터링
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # 클래스 ID는 model.names의 인덱스입니다. COCO에서 0은 사람입니다.
        result = model(SAMPLE_IMAGE, classes=[0])

        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: 이름에 대한 아이디 찾기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        wanted = {"person", "backpack"}
        ids = [i for i, name in result.names.items() if name in wanted]
        print(ids)

        filtered = model(SAMPLE_IMAGE, classes=ids)
        print(len(filtered.boxes))
  nmsfree:
    - label: NMS를 운영하지 않는 계열에게 빚을 지다
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # RF-DETR는 고정된 쿼리 세트를 디코딩하므로 여기서 IOU는 아무 것도 바꾸지 않습니다.
        model = LibreYOLO("LibreRFDETRs.pt")

        loose = model(SAMPLE_IMAGE, iou=0.9)
        tight = model(SAMPLE_IMAGE, iou=0.1)

        # 어느 쪽이든 동일한 수치입니다. conf와 max_det가 작동하는 제어값입니다.
        print(len(loose.boxes), len(tight.boxes))
  agnostic:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        per_class = model(SAMPLE_IMAGE)
        agnostic = model(SAMPLE_IMAGE, agnostic_nms=True)

        # 클래스가 다른 겹치는 박스는 점수가 가장 높은 하나로 합쳐집니다.
        print(len(per_class.boxes), len(agnostic.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt agnostic_nms=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: fa0e6cee02b3e336
---

## 네 가지 논점

| 논쟁 | 기본값 | 적용 대상 |
|---|---|---|
| `conf` | `0.25` | 모든 계열 |
| `iou` | `0.45` | 비최대 억제를 실행하는 계열들 |
| `max_det` | `300` | 모든 계열 |
| `classes` | `None` | 모든 계열 |

<code-tabs name="basic" />

이 중 두 가지는 보편적이고 두 가지는 그렇지 않은데, 이는 무엇이든 조정하기 전에 아는 것이 가장 유용한 한 가지입니다.

검증은 의도적으로 다른 기본값을 사용합니다: `val()`는 `conf=0.001`와 `iou=0.6`에서 실행됩니다. 이는 평균 정밀도가 전체 정밀도-재현율 곡선을 기준으로 계산되며, 0.25 컷오프는 이를 잘라버리기 때문입니다.

## 컨프

`conf`은 예측이 폐기되는 점수입니다. 이는 NMS를 실행하지 않는 계열를 포함한 모든 계열에 적용되며, 탐지가 너무 많거나 너무 적을 때 가장 먼저 조정하는 기준입니다.

`0.25`의 기본값은 사진을 보는 데 적합합니다. 하류 시스템에 공급할 때는 보통 더 높게 설정하기를 원하며, 정확도를 측정할 때는 훨씬 낮게 설정하기를 원합니다.

탐지 검증은 IoU 0.50에서 F1을 기준으로 `metrics/best_conf`와 클래스별 임계값을 제공합니다. 이를 배포 임계값의 측정된 시작점으로 사용하며, [검증](/docs/train/validation)을 참조하십시오. 분할은 이 키를 제공하지 않습니다.

## IOU

`iou`는 같은 클래스의 두 박스 중 점수가 낮은 박스를 제거하는 비최대 억제의 기준이 되는 겹침 정도입니다. 이는 해당 계열가 억제를 실행할 때만 의미가 있습니다.

세트 예측기는 고정된 수의 쿼리를 디코딩하고 점수가 높은 것들을 선택합니다. 중복은 후처리 단계가 아니라 학습 중 아키텍처 내에서 억제되므로 조정할 임계값이 없습니다. 이러한 계열는 API 일관성을 위해 `iou`를 허용하며 이를 무시합니다:

CenterNet, DEIM, DETR, Deformable DETR, D-FINE, DINO-DETR, EdgeCrafter, Faster R-CNN, LW-DETR, Mask R-CNN, RF-DETR, RT-DETR, 그리고 end-to-end YOLOv9 헤드. 이러한 디코더를 기반으로 구축된 변형들은 그 동작을 그대로 계승합니다.

<code-tabs name="nmsfree" />

대부분은 후처리 도큐스트링에서 그렇게 말하지만, 실행 시 경고는 발생하지 않으므로 RF-DETR에서 `iou`를 전체적으로 수행하면 오류 대신 평평한 선이 나타납니다. Faster R-CNN과 Mask R-CNN은 약간 다른 경우입니다. 두 모델 모두 이미 모델 내부에서 NMS를 수행했으며, `iou`가 변경할 수 있는 방법을 지원하지 않는 고정된 업스트림 임계값을 사용합니다.

이 계열들은 이것을 사용합니다: YOLOv1부터 YOLOv4, YOLOv7, YOLOv9, YOLOX, YOLO-NAS, RTMDet, PicoDet, EfficientDet, FCOS, RetinaNet 그리고 SSD.

세 가지 예측 시 옵션은 셋 모두 모델이 끝난 후 박스에 작용하기 때문에 세트 예측기에서도 `iou`를 중요하게 만듭니다:

- `tiling=True`은 `iou`에서 클래스별 NMS로 겹치는 타일을 조정합니다
- `augment=True`는 각 클래스 NMS와 뒤집힌 뷰를 `iou`에서 병합합니다
- `agnostic_nms=True`는 `iou`에서 클래스에 걸쳐 겹치는 박스를 억제하며,
  [아래](#agnostic_nms)에서 설명합니다

앞의 둘은 [추론 성능](/docs/predict/performance)에 포함되어 있습니다.

오픈-어휘 탐지기는 자체 규칙을 가지고 있습니다. 프로세서가 NMS를 실행하는 계열는 자체 기본 임계값을 선언하고 `iou`를 준수하며, 이는 OMDet-Turbo의 경우에 해당합니다. 아무 것도 억제하지 않는 계열인 Grounding DINO, OWLv2 및 OV-DEIM은 `iou`가 전달되면 경고를 발생시킵니다. 그 경고는 라이브러리에서 유일한 경고입니다.

## 최대_탐지

`max_det`는 한 이미지에 대해 몇 개의 예측이 반환되는지를 제한합니다. 이는 모든 곳에 적용되지만, 서로 다른 메커니즘을 통해 적용됩니다: NMS 계열은 억제 후에 잘라내고, 세트 예측기는 이를 상위 k 선택의 크기로 사용합니다.

어떤 계열들은 업스트림 참조 구성 때문에 요청하는 것과 상관없이 아래로 제한합니다. SSD는 200으로 한정되고, FCOS는 이미지당 자체 탐지 한도에 따라 제한됩니다. `max_det`를 그 이상으로 올려도 효과가 없습니다.

`max_det`가 각 계열별이 아니라 중앙에서 적용되는 경우는 두 가지입니다. 하나는 타일 추론으로, 병합된 목록이 타일이 조정된 후 잘립니다. 다른 하나는 `classes`나 `agnostic_nms=True`를 쓰는 호출로, 필터나 억제 뒤에 잘립니다.

## 클래스 필터링

<code-tabs name="classes" />

`classes`는 클래스 ID 목록을 받아 해당 클래스가 목록에 있는 예측만 유지합니다. ID는 `result.names`를 인덱싱하며, 가장 확실한 방법은 데이터셋 순서를 가정하기보다는 결과에서 `names`를 읽는 것입니다.

필터링은 각 계열의 후처리 후 중앙에서 발생하며, 모든 예측 경로가 통과하는 단일 깔때기에서 이루어집니다. 이것은 알아둘 가치가 있는 두 가지 결과를 가져옵니다. 모든 계열에서 작동하며, NMS가 없는 계열도 포함됩니다. 또한 상자와 정렬된 페이로드를 필터링하므로, 마스크, 키포인트 및 방향이 지정된 상자도 함께 잘려서 불일치되지 않습니다.

명령줄에서 `classes`는 단일 정수, 리스트 또는 쉼표로 구분된 문자열을 받습니다:

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

필터링이 정확도를 저절로 높이지는 않습니다. 모델은 여전히 나중에 버리는 클래스를 예측하는 데 연산 예산을 사용합니다. `classes`를 설정하면 계열은 박스 300개, 또는 `max_det`가 더 크면 `max_det`만큼의 후보 예산을 유지하며, `max_det`로 자르는 것은 필터 뒤에 이루어지므로 작은 `max_det`가 관심 클래스를 밀어내지 않습니다. 내보낸 백엔드도 배치 TensorRT 엔진을 포함해 같은 예산을 사용합니다. 점수가 더 높은 원치 않는 클래스의 박스가 300개를 넘는 이미지는 관심 클래스에 도달하기 전에 여전히 예산을 채울 수 있습니다. 그런 경우 `max_det`를 300보다 높이십시오.

## 애그노스틱_nms

모든 계열의 자체 NMS는 클래스를 구분하므로, 기본적으로 클래스가 다른 두 개의 겹치는 박스는 어떤 `iou`에서든 모두 살아남습니다. `agnostic_nms=True`는 클래스를 구분하지 않는 단계를 하나 추가합니다. `iou`를 넘어 겹치는 박스 중 클래스와 관계없이 점수가 가장 높은 것 하나만 남습니다.

<code-tabs name="agnostic" />

이 단계는 계열이 끝낸 탐지 결과에 대해 `classes` 필터 뒤, `max_det`로 자르기 전에 실행됩니다. 따라서 자체 NMS를 실행하지 않는 세트 예측기, 내보낸 백엔드, `augment=True`와 `tiling=True`에서도 똑같이 동작합니다. 마스크와 키포인트는 해당 박스와 함께 제거되고, 회전 박스는 회전 IoU로 비교됩니다. 박스를 반환하는 작업인 탐지, 분할, 자세, OBB를 지원합니다.

계열은 이 단계가 처리할 후보로 박스 300개, 또는 `max_det`가 더 크면 `max_det`만큼의 예산을 유지합니다. `max_det`가 300보다 작으면 점수가 더 낮은 후보가 억제로 비게 된 자리를 다시 채웁니다. 기본값을 포함해 300 이상이면 그 자리는 빈 채로 남습니다.

적용되지 않는 경우는 다음과 같습니다.

- 표준 predict 경로에서 박스가 없는 작업은 `ValueError`를 발생시킵니다.
- `LibreEnsemble`과 [HRNet](/docs/models/hrnet)은 `NotImplementedError`를 발생시킵니다.
- `LibreSAM` 모델과 [3D 탐지기](/docs/tasks/3d-object-detection)인 WildDet3D, 3D-MOOD, FCOS3D, DetAny3D는 이런 인자를 받지 않으며 `TypeError`를 발생시킵니다.
- [L2CS-Net](/docs/models/l2cs), [PP-OCRv5](/docs/models/pp-ocrv5), [SAM 3D Body](/docs/models/sam-3d-body), [LibreFaceRec](/docs/models/librefacerec)은 자체 predict 경로를 실행하며 이를 무시합니다.

`val(agnostic_nms=True)`는 탐지와 분할 검증에 같은 단계를 적용합니다. [검증과 지표](/docs/train/validation)를 참조하십시오.

## 무엇을 예측이 거부하는가

두 개의 인수가 경고 대신 발생합니다: `visualize`와 `embed`는 둘 다 `NotImplementedError`를 발생시킵니다. 임베딩의 경우, `task="embed"`로 모델을 로드하고 `predict` 또는 `embed`를 정상적으로 호출하십시오. `agnostic_nms=True`는 위에 나열한 대로 적용되지 않는 곳에서 예외를 발생시킵니다.

인식되지 않는 항목은 지원되는 옵션을 나열하는 `TypeError`를 발생시키므로, 오타는 조용히 무시되는 대신 즉시 실패합니다.

다음은 수락되었고, 경고되었으며, 폐기되었습니다: `boxes`, `dnn`, `half`, `line_width`, `retina_masks`, `show_conf`, `show_labels` 및 `verbose`.
