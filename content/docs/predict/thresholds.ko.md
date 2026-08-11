---
title: 임계값 및 필터링
seo_title: 'LibreYOLO에서 conf, iou 및 max_det'
description: >-
  예측 시 conf, iou, max_det 및 classes가 실제로 수행하는 것, NMS를 실행하지 않기 때문에 iou를 무시하는 계열,
  그리고 agnostic_nms가 작동하지 않는 이유
lead: >-
  어떤 예측이 살아남을지 결정하는 네 가지 인수: conf, iou, max_det 및 classes. 세트 예측자는 고정된 쿼리 세트를
  디코딩하고 NMS를 실행하지 않기 때문에 이 중 두 가지만 모든 계열에 적용됨
keywords:
  - yolo conf 임계값
  - iou 임계값 nms
  - max_det
  - 클래스 필터링 detection python
  - agnostic nms
  - NMS 없이 detr
  - 검출 신뢰도 임계값
  - 클래스 필터링 추론
last_verified: 1.5.0
verification: >-
  libreyolo/models/base/inference.py.의 InferenceRunner.__call__에서 기본값 인용,
  libreyolo/postprocess/의 모든 모듈에서 읽은 퍼-패밀리 NMS 동작 및 libreyolo/backends/base.py.의
  _is_nms_free_family와 교차 확인. InferenceRunner._apply_classes_filter와
  _wrap_results에서 클래스 필터링. libreyolo/utils/predict_args.py.의
  NOOP_PREDICT_KWARGS에서 agnostic_nms 상태. libreyolo/models/openvocab/base.py.의
  NMS_THRESHOLD에서 오픈-보캐블러리 처리. BaseModel.val에서 검증 기본값.
snippets:
  basic:
    - label: 네 가지 인수
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(
            SAMPLE_IMAGE,
            conf=0.25,      # 이 점수 이상인 예측을 유지
            iou=0.45,       # NMS가 실행되는 NMS 겹침 임계값
            max_det=300,    # 이미지당 상한
            classes=None,   # 또는 클래스 ID 목록
        )
        print(len(result.boxes))
    - label: conf 스윕
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

        # 클래스 ID는 model.names의 인덱스. COCO에서는 0이 사람.
        result = model(SAMPLE_IMAGE, classes=[0])

        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: 이름에 대한 ID 찾기
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
    - label: NMS를 수행하지 않는 패밀리에서의 iou
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # RF-DETR은 고정된 쿼리 집합을 디코딩하므로, iou는 여기서 아무 것도 변경하지 않습니다.
        model = LibreYOLO("LibreRFDETRs.pt")

        loose = model(SAMPLE_IMAGE, iou=0.9)
        tight = model(SAMPLE_IMAGE, iou=0.1)

        # 어느 쪽이든 동일한 개수입니다. conf와 max_det가 작동하는 제어입니다.
        print(len(loose.boxes), len(tight.boxes))
source_hash: 0b978963c356027d
---

## 네 개의 인수

| 인수 | 기본값 | 적용 대상 |
|---|---|---|
| `conf` | `0.25` | 모든 패밀리 |
| `iou` | `0.45` | 비최대 억제를 실행하는 패밀리 |
| `max_det` | `300` | 모든 패밀리 |
| `classes` | `None` | 모든 패밀리 |

<code-tabs name="basic" />

이 중 두 개는 보편적이고 두 개는 그렇지 않으며, 이는 어떤 조정을 하든 가장 먼저 알고 있어야 할 단 하나의 유용한 정보입니다.

검증은 일부러 다른 기본값을 사용합니다: `val()`는 `conf=0.001`와 `iou=0.6`에서 실행되며, 평균 정밀도는 전체 정밀도-재현 곡선을 기준으로 계산되므로 0.25 컷오프는 이를 잘라버릴 수 있습니다.

## conf

`conf`는 예측을 폐기하는 기준점 이하의 점수입니다. 이는 NMS를 전혀 실행하지 않는 경우를 포함하여 모든 패밀리에 적용되며, 감지가 너무 많거나 너무 적을 때 가장 먼저 확인할 제어 값입니다.

`0.25`의 기본값은 사진을 볼 때 적합합니다. 하류 시스템에 전달할 때는 보통 더 높게 설정하고, 정확도를 측정할 때는 훨씬 낮게 설정합니다.

## iou

`iou`는 비최대 억제가 같은 클래스의 두 박스 중 낮은 점수를 가진 박스를 제거하는 기준이 되는 겹침 정도입니다. 패밀리가 억제를 실행할 경우에만 의미가 있습니다.

세트 예측기는 고정된 수의 쿼리를 디코딩하고 점수가 높은 것들을 선택합니다. 중복은 후처리 단계가 아닌 학습 중 아키텍처 내부에서 억제되므로 조정할 임계값이 없습니다. 이러한 패밀리는 API 일관성을 위해 `iou`를 허용하지만 무시합니다:

CenterNet, DEIM, DETR, Deformable DETR, D-FINE, DINO-DETR, EdgeCrafter, Faster R-CNN, LW-DETR, Mask R-CNN, RF-DETR, RT-DETR, 그리고 end-to-end YOLOv9 헤드. 이러한 디코더 위에 구축된 변형들은 동일한 동작을 상속합니다.

<code-tabs name="nmsfree" />

대부분은 후처리 docstring에서 그렇게 명시하지만, 런타임에 경고가 발생하지 않으므로, RF-DETR에서 `iou`를 검색하면 오류 대신 평평한 선이 나타납니다. Faster R-CNN과 Mask R-CNN은 약간 다른 경우입니다: 두 모델 모두 이미 고정된 업스트림 임계값에서 NMS를 모델 내부에서 실행했으며, `iou`에는 이를 변경할 수 있는 지원 방법이 없습니다.

이러한 계열은 이를 사용합니다: YOLOv1부터 YOLOv4, YOLOv7, YOLOv9, YOLOX, YOLO-NAS, RTMDet, PicoDet, EfficientDet, FCOS, RetinaNet 및 SSD.

두 가지 예측 시 옵션은 모델이 완료된 후 박스를 병합하기 때문에, 세트 예측자에도 `iou`가 중요하게 작용합니다:

- `tiling=True`는 `iou`에서 클래스별 NMS로 겹치는 타일을 조정합니다.
- `augment=True`는 `iou`에서 클래스별 NMS로 뒤집힌 뷰를 병합합니다.

두 가지 모두 [추론 성능](/docs/predict/performance)에 다루어져 있습니다.

오픈 보캐뷸러리 탐지기는 자체 규칙을 가지고 있습니다. 프로세서가 NMS를 실행하는 패밀리는 자체 기본 임계값을 선언하고 `iou`를 준수하며, 이는 OMDet-Turbo의 경우입니다. 아무 것도 억제하지 않는 패밀리인 Grounding DINO, OWLv2 및 OV-DEIM은 `iou`가 전달될 때 경고를 발생시킵니다. 이 경고는 라이브러리에서 유일한 종류입니다.

## max_det

`max_det`는 하나의 이미지에서 얼마나 많은 예측이 반환될 수 있는지를 제한합니다. 이는 모든 곳에 적용되지만, 메커니즘은 다릅니다: NMS 패밀리는 억제 후 잘라내고, 세트 예측기는 이를 상위-k 선택의 크기로 사용합니다.

일부 계열들은 상류 참조 구성 때문에 무엇을 요청하든 그 아래로 값을 제한합니다. SSD는 최대 200, RTMDet 인스턴스 분할은 100, FCOS는 자체 이미지당 감지 한도까지 제한됩니다. `max_det`를 그 이상으로 올려도 효과가 없습니다.

`max_det`가 계열별이 아닌 중앙에서 적용되는 유일한 경우는 타일 추론이며, 여기서 병합된 목록은 타일이 조정된 후 잘립니다.

## 클래스 필터링

<code-tabs name="classes" />

`classes`는 클래스 ID 목록을 받아 해당 목록에 있는 클래스만 예측으로 유지합니다. ID는 `result.names`를 인덱싱하며, 가장 확실한 방법은 데이터셋 순서를 가정하지 않고 결과에서 `names`를 읽는 것입니다.

필터링은 각 패밀리의 후처리 후, 모든 예측 경로가 통과하는 단일 퍼널에서 중앙 집중식으로 발생합니다. 이것에는 알아둘 만한 두 가지 결과가 있습니다. 모든 패밀리에서 작동하며, NMS가 없는 패밀리도 포함됩니다. 또한, 필터링은 박스에 정렬된 페이로드에도 적용되므로 마스크, 키포인트, 방향 박스도 박스와 함께 축소되어 불일치 상태로 남지 않습니다.

명령줄에서 `classes`는 단일 정수, 목록, 또는 쉼표로 구분된 문자열을 허용합니다.

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

필터링은 정확도가 공짜가 아닙니다. 모델은 여전히 나중에 버리게 될 클래스를 예측하는 데 예산을 소비하며, `max_det`는 필터 이전에 패밀리에 의해 적용되므로, 원하지 않는 클래스로 이미지가 가득 차면 원하는 클래스에 도달하기 전에 한도를 초과할 수 있습니다. 그런 경우 `conf`를 낮추거나 `max_det`를 높이세요.

## agnostic_nms

`agnostic_nms`는 허용되며 아무 작업도 수행하지 않습니다. 이를 전달하면 명령줄 호환성을 위한 무작동(no-op)이라는 경고가 발생하며, 해당 인수는 무시됩니다.

클래스에 관계없는 억제 모드는 없습니다. 라이브러리 내 모든 NMS 호출은 클래스 인식 방식이므로, 서로 다른 클래스의 겹치는 두 박스는 항상 모두 살아남습니다. 이것이 문제라면 먼저 `classes`로 필터링하거나 `result.boxes`에서 직접 클래스 간 억제를 수행하십시오.

## 예측에서 거부되는 것

두 인수는 경고 대신 예외를 발생시킵니다: `visualize`와 `embed`는 모두 `NotImplementedError`를 발생시킵니다. 임베딩의 경우, `task="embed"`로 모델을 로드하고 `predict` 또는 `embed`를 정상적으로 호출하십시오.

인식되지 않는 항목은 지원되는 옵션을 명시하며 `TypeError`를 발생시키므로, 오타가 있어도 즉시 실패하며 조용히 무시되지 않습니다.

이것들은 수락되었고, 경고되었으며, 폐기되었습니다: `agnostic_nms`, `boxes`, `dnn`, `half`, `line_width`, `retina_masks`, `show_conf`, `show_labels` 그리고 `verbose`.
