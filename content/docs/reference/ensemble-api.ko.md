---
title: 앙상블 API
seo_title: LibreEnsemble API 및 퓨전 연산
description: >-
  LibreEnsemble, ExternalDetector 및 libreyolo.ops의 세 가지 퓨전 연산: 가중 박스 퓨전, 시드 변형,
  클래스 인식 NMS 퓨전.
lead: >-
  LibreEnsemble은 동일한 이미지에서 여러 탐지기를 실행하고 그 탐지 결과를 하나의 Results로 통합합니다. 퓨전은 각 멤버의
  자체 후처리 후에 이루어지므로, 멤버들은 자신만의 입력 크기, 정규화 및 억제를 유지합니다.
keywords:
  - LibreEnsemble
  - 가중 박스 퓨전
  - wbf
  - ExternalDetector
  - libreyolo.ops.fusion
  - 최소 투표 합의
last_verified: 1.5.0
verification: >-
  v1.5.0에서 libreyolo/ensemble/model.py 및 libreyolo/ops/fusion.py로부터 서명 및 기본값 읽기.
  설계 의도는 docs/adr/0004-model-ensembling.md. 기준.
snippets:
  usage:
    - label: '두 멤버, 기본 퓨전'
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])

        # 단일 이미지 소스는 리스트가 아닌 하나의 Results를 반환합니다.
        result = ens(SAMPLE_IMAGE, conf=0.25)

        print(result.boxes.xyxy)
        print(result.speed)
    - label: 합의 및 구성원별 임계값
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(
            ["LibreYOLO9t.pt", "LibreYOLO9s.pt"],
            weights=[1.0, 2.0],
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,
        )
        result = ens(SAMPLE_IMAGE, conf=[0.25, 0.4])
        print(len(result))
  ops:
    - label: '융합 연산, 모델 미관여'
      language: python
      code: >
        import torch

        from libreyolo.ops import weighted_boxes_fusion


        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [12.0, 11.0, 51.0,
        49.0]])

        scores = torch.tensor([0.9, 0.8])

        labels = torch.tensor([0, 0])

        model_ids = torch.tensor([0, 1])


        fused = weighted_boxes_fusion(
            boxes, scores, labels, model_ids, num_models=2, iou_thr=0.55
        )

        print(fused)
source_hash: 3834f628efb1193d
---

## LibreEnsemble

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

| 인수 | 기본값 | 의미 |
|---|---|---|
| `members` | | 두 개 이상의 탐지기 |
| `weights` | `None` | 구성원별 신뢰 요인; 생략 시 모두 `1.0` |
| `fusion` | `"wbf"` | `"wbf"`, `"wbf_seeded"`, `"nms"` 또는 호출 가능한 것 |
| `fusion_iou` | `0.55` | 융합 클러스터링을 위한 IoU 임계값 |
| `min_votes` | `1` | 최소 이 수 이상의 구성원으로 확인된 박스만 유지 |

구성원은 `LibreYOLO()` 팩토리를 통해 해결된 가중치 경로, 이미 구축된 모델, 내보낸 백엔드 또는 `ExternalDetector`입니다. 모든 구성원은 탐지 작업 모델이어야 합니다.

<code-tabs name="usage" />

구성은 두 명 미만의 구성원, 잘못된 길이의 `weights` 목록, 0 이하의 가중치, 양의 정수가 아닌 `min_votes` 및 구성원 수보다 큰 `min_votes`를 거부합니다. `fusion="nms"`가 `min_votes > 1`인 경우에도 오류가 발생하는데, 이는 NMS가 클러스터 멤버십을 폐기하고 투표를 셀 수 없기 때문입니다.

`weights`는 각 구성원에게 부여된 신뢰를 조정합니다. 가중치가 높을수록 융합된 좌표와 점수가 해당 구성원 쪽으로 더 끌립니다. 일반적인 관례는 이를 검증 mAP에 비례하도록 만드는 것입니다.

## 클래스 공간

동일한 `names`를 가진 멤버는 그대로 통과합니다. 그렇지 않으면 클래스 공간은 이름으로 합쳐지고, 멤버 클래스 ID는 조회 테이블을 통해 다시 매핑되며, 결합된 `Results.names`는 합집합입니다. 결합은 동일한 통합 클래스 내의 박스만 병합하므로 한 멤버만 아는 클래스는 결합되지 않고 통과합니다. 불일치가 있으면 생성 시 경고가 기록됩니다.

`min_votes`는 그 클래스가 멤버의 레이블 공간에 포함된 수로 클래스별로 제한되므로 부분적으로 공유된 어휘에서도 합의가 의미있게 유지됩니다.

## 앙상블 호출

```python
ens(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    output_path=None,
    color_format="auto",
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    **kwargs,
)
```

`predict`는 `__call__`의 별칭입니다. 반환은 일반적인 `Results`이며, 그 `speed`는 비용을 멤버별로 분해하고 `fusion` 항목을 추가합니다. 단일 이미지 소스는 그 중 하나를 반환하고, 목록이나 디렉토리는 목록을 반환하며, `stream=True`는 생성기를 반환합니다.

`conf`, `iou` 및 `device`는 모든 구성원에게 브로드캐스트하며 각 구성원마다 하나의 값을 허용하므로 `conf=[0.25, 0.4]`는 구성원 0에 0.25의 임계값을, 구성원 1에 0.4의 임계값을 제공합니다. `imgsz`는 int 또는 tuple일 때 브로드캐스트되며, list일 때만 구성원별로 적용되므로 `imgsz=(480, 640)`는 모든 사람에게 하나의 직사각형 크기를 제공하고 `imgsz=[480, 640]`는 구성원 0에 480, 구성원 1에 640을 제공합니다. 각 항목은 해당 구성원의 패밀리에 유효해야 합니다.

`augment`는 테스트 시 증강을 지원하는 구성원에게 브로드캐스트되며, 내보낸 백엔드는 이를 무시합니다. `classes`는 클래스 ID의 합집합을 취하고 `max_det`는 결합된 결과에 적용되므로 구성원은 관대하게 실행하고 앙상블은 한 번만 정리합니다. `batch`는 API 일관성을 위해 허용됩니다; 이미지들은 순차적으로 처리됩니다.

`val()` 및 `export()`는 `NotImplementedError`를 발생시킵니다. 구성원을 개별적으로 검증하고 내보내세요.

## 외부 탐지기

```python
ExternalDetector(fn: Callable, names: dict[int, str])
```

임의의 감지 호출을 멤버로 적응시킵니다. `fn`는 PIL 이미지를 받아 `(boxes, scores, labels)`를 반환하며, 여기서 상자는 원본 이미지 픽셀 단위의 xyxy이고 레이블은 `names`에서 유효한 클래스 ID입니다. 텐서, 배열 및 중첩 리스트 모두 작동합니다. LibreYOLO는 외부 코드에서 아무 것도 가져오지 않습니다.

어댑터는 반환값을 검증합니다: 3-튜플이어야 하며, 상자의 형태는 `(N, 4)`여야 하고, 세 배열은 길이가 같아야 하며, 모든 클래스 ID는 `names`에 나타나야 합니다. `conf` 이하의 감지는 융합 전에 제거됩니다.

## 융합 연산

융합 원시 연산은 `libreyolo.ops`에 있는 독립적인 torch 연산입니다. 이들은 모델에 독립적이며 단독으로 가져올 수 있으므로 앙상블과 별도로 내보내집니다.

<code-tabs name="ops" />

세 가지 모두 동일한 위치 인수 `boxes, scores, labels, model_ids`를 취하고 `(boxes, scores, labels)`를 반환합니다.

| 연산자 | 레지스트리 키 | 동작 |
|---|---|---|
| `weighted_boxes_fusion` | `wbf` | 순차적, 종이-충실 가중 상자 융합 |
| `wbf_seeded` | `wbf_seeded` | 동일한 축소의 병렬 일회 통과 변형 |
| `nms_fusion` | `nms` | 모든 것을 연결하고 클래스 인식 NMS를 적용 |

`FUSIONS`는 세 개의 레지스트리 키를 호출 가능한 객체에 매핑하고, `LibreEnsemble`는 거기서 `fusion=`를 조회합니다.

```python
weighted_boxes_fusion(
    boxes, scores, labels, model_ids,
    *,
    weights=None,
    num_models=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type="avg",
    min_votes=1,
    models_per_label=None,
    label_weights=None,
)
```

`wbf_seeded`는 동일한 시그니처를 취합니다. `nms_fusion`는 `conf_type`를 제외한 동일한 인수를 취하고, `min_votes > 1`일 때 `ValueError`를 발생시킵니다.

`weighted_boxes_fusion`에서는 탐지가 가중치로 조정된 신뢰도 순서대로 방문됩니다. 각각은 기존 클러스터에 합류하거나 동일한 레이블을 가진 IoU가 `iou_thr` 이상인 해당 클러스터의 진행 중인 통합 상자와 가장 잘 겹치거나, 새로운 클러스터를 시작합니다. 클러스터의 통합 상자는 그 구성원의 좌표를 신뢰도로 가중 평균한 것이며, 점수는 신뢰도를 가중 평균하거나 최대값으로 산출하고, 적은 모델에서 확인된 상자가 점수가 낮도록 재조정됩니다.

`wbf_seeded`는 클래스 인식 NMS를 `iou_thr`에서 사용하여 클러스터 시드를 선택하고, 모든 탐지를 동일 레이블의 최적 IoU 시드에 할당한 후 각 클러스터를 같은 방식으로 축소합니다. 클러스터 형태는 진행 중간에 이동하지 않으므로 전체 연산은 고정 형태 텐서 연산입니다. 두 가지 변형은 클러스터가 명확할 경우 일치하며 겹치는 클러스터 체인에서는 약간 다를 수 있습니다.

`nms_fusion` 각 겹치는 그룹에서 가장 높은 신뢰도의 박스를 그대로 유지합니다. 모델별로 `weights`는 억제 순위를 위해서만 신뢰도를 조정하며, 살아남은 박스는 원래 점수를 유지합니다.

## 맞춤 융합

`fusion=` 또한 위의 연산과 동일한 서명을 가진 호출 가능 객체를 받을 수 있습니다. 이름은 `ens.fusion`에 기록되거나, 이름이 없으면 `"custom"`에 기록됩니다. 반환값은 검증됩니다: 일관된 형태를 가진 `(boxes, scores, labels)` 삼중이어야 합니다.
