---
title: 앙상블 API
seo_title: LibreEnsemble API 및 통합 작업
description: >-
  LibreEnsemble, ExternalDetector, 그리고 libreyolo.ops의 세 가지 융합 연산: 가중 박스 융합, 그 시드
  변형, 그리고 클래스 인식 NMS 융합.
lead: >-
  LibreEnsemble은 동일한 이미지에 여러 탐지기를 실행하고 그 탐지 결과를 하나의 결과로 통합합니다. 통합은 각 멤버의 자체 후처리
  후에 발생하므로, 멤버들은 자신의 입력 크기, 정규화 및 억제를 유지합니다.
keywords:
  - 리브르앙상블
  - 가중치 박스 융합
  - wbf
  - 외부 탐지기
  - libreyolo.ops.융합
  - 최소 투표 합의
last_verified: 1.5.0
verification: >-
  v1.5.0에서 libreyolo/ensemble/model.py 및 libreyolo/ops/fusion.py에서 읽은 서명 및 기본값.
  docs/adr/0004-model-ensembling.md.에서 설계 의도
snippets:
  usage:
    - label: '두 멤버, 기본 융합'
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])

        # 단일 이미지 소스는 리스트가 아닌 하나의 결과를 반환합니다.
        result = ens(SAMPLE_IMAGE, conf=0.25)

        print(result.boxes.xyxy)
        print(result.speed)
    - label: 합의 및 회원별 기준
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
    - label: '퓨전 연산, 모델 없음'
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

## 리브르앙상블

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

| 논쟁 | 기본값 | 의미 |
|---|---|---|
| `members` | | 두 개 이상의 탐지기 |
| `weights` | `None` | 멤버별 신뢰 요소; 생략 시 모두 `1.0` |
| `fusion` | `"wbf"` | `"wbf"`, `"wbf_seeded"`, `"nms"` 또는 호출 가능 |
| `fusion_iou` | `0.55` | 융합 클러스터링을 위한 IoU 임계값 |
| `min_votes` | `1` | 적어도 이 수만큼의 회원이 확인한 상자만 유지하십시오 |

멤버는 `LibreYOLO()` 팩토리를 통해 해석된 가중치 경로, 이미 구성된 모델, 내보낸 백엔드 또는 `ExternalDetector`입니다. 모든 멤버는 검출 작업 모델이어야 합니다.

<code-tabs name="usage" />

구성은 두 명 이하의 구성원, 잘못된 길이의 `weights` 목록, 양수가 아닌 가중치, 양의 정수가 아닌 `min_votes` 및 구성원 수보다 큰 `min_votes`를 거부합니다. `min_votes > 1`가 있는 `fusion="nms"`도 발생합니다. NMS가 클러스터 구성원을 삭제하고 투표 수를 계산할 수 없기 때문입니다.

`weights`는 각 구성원에게 부여된 신뢰를 척도로 합니다. 높은 가중치는 융합된 좌표와 점수를 해당 구성원 쪽으로 끌어당깁니다. 관례상 이를 검증 mAP에 비례하도록 합니다.

## 수업 공간

`names`가 동일한 멤버는 그대로 통과합니다. 그렇지 않으면 클래스 공간은 이름별로 합쳐지고, 멤버 클래스 ID는 조회 테이블을 통해 재매핑되며, 통합된 `Results.names`가 합집합이 됩니다. 퓨전은 동일한 통합 클래스 내에서만 박스를 병합하므로, 클래스에 멤버가 하나만 있는 경우에는 병합되지 않고 통과합니다. 불일치가 있으면 생성 시 경고가 기록됩니다.

`min_votes`는 각 클래스별로 그 클래스가 포함된 멤버의 레이블 공간 수에 따라 제한되므로, 부분적으로 공유된 어휘에서도 합의가 의미 있게 유지됩니다.

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

`predict`는 `__call__`의 별칭입니다. 반환값은 일반적인 `Results`이며, 여기서 `speed`는 회원별 비용을 나누고 `fusion` 항목을 추가합니다. 단일 이미지 소스는 그 중 하나를 반환하고, 목록이나 디렉터리는 목록을 반환하며, `stream=True`는 제너레이터를 반환합니다.

`conf`, `iou` 및 `device`는 모든 구성원에게 브로드캐스트하며 구성원당 하나의 값도 허용하므로 `conf=[0.25, 0.4]`는 구성원 0에게 0.25의 임계값을, 구성원 1에게 0.4의 임계값을 제공합니다. `imgsz`는 int나 튜플일 때 브로드캐스트되며 리스트일 때만 구성원별로 적용되므로 `imgsz=(480, 640)`는 모두에게 하나의 직사각형 크기이고 `imgsz=[480, 640]`는 구성원 0에게 480, 구성원 1에게 640입니다. 각 항목은 해당 구성원의 계열에 대해 유효해야 합니다.

`augment`는 테스트 시 증강을 지원하는 멤버들에게 방송하며, 내보낸 백엔드는 이를 무시합니다. `classes`는 클래스 ID를 합집합으로 취하고 `max_det`는 융합된 결과에 적용하여 멤버들이 충분히 실행되고 앙상블은 한 번 트리밍합니다. `batch`는 API 일치를 위해 허용되며, 이미지가 순차적으로 처리됩니다.

`val()`와 `export()`는 `NotImplementedError`를 발생시킵니다. 구성원을 개별적으로 확인하고 내보내십시오.

## 외부 탐지기

```python
ExternalDetector(fn: Callable, names: dict[int, str])
```

모든 탐지 호출 가능 객체를 멤버로 적응시킵니다. `fn`는 PIL 이미지를 가져와 `(boxes, scores, labels)`를 반환하며, 여기서 박스는 원본 이미지 픽셀의 xyxy이고 레이블은 `names`에서 유효한 클래스 ID입니다. 텐서, 배열 및 중첩 리스트 모두 작동합니다. LibreYOLO는 외부 코드에서 아무것도 가져오지 않습니다.

어댑터는 반환 값을 검증합니다: 3-튜플이어야 하고, boxes는 `(N, 4)` 형상을 가져야 하며, 세 배열은 같은 길이여야 하고, 모든 클래스 ID가 `names`에 나타나야 합니다. `conf` 이하의 탐지는 융합 전에 제거됩니다.

## 융합 작전

퓨전 프리미티브는 `libreyolo.ops`의 독립적인 토치 연산입니다. 이들은 모델에 의존하지 않으며 자체적으로 임포트할 수 있기 때문에 앙상블과 별도로 내보내집니다.

<code-tabs name="ops" />

세 가지 모두 동일한 위치 인수 `boxes, scores, labels, model_ids`를 사용하고 `(boxes, scores, labels)`를 반환합니다.

| 작업 | 레지스트리 키 | 행동 |
|---|---|---|
| `weighted_boxes_fusion` | `wbf` | 순차적, 종이-충실 가중 박스 융합 |
| `wbf_seeded` | `wbf_seeded` | 같은 축소의 병렬 단일 통과 변형 |
| `nms_fusion` | `nms` | 모든 것을 연결하고 클래스 인식 NMS를 적용하십시오 |

`FUSIONS`는 세 개의 레지스트리 키를 호출 가능한 항목에 매핑하고, `LibreEnsemble`는 거기서 `fusion=`를 조회합니다.

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

`wbf_seeded`는 동일한 서명을 사용합니다. `nms_fusion`는 `conf_type`를 제외한 동일한 인수를 사용하며, `min_votes > 1`일 때 `ValueError`를 발생시킵니다.

`weighted_boxes_fusion`에서는 탐지가 가중치로 조정된 신뢰도 순서대로 방문됩니다. 각 탐지는 기존 클러스터 중 자신의 겹치는 누적 융합 박스와 가장 잘 맞는 것에 참여하거나, IoU가 `iou_thr` 이상이고 동일한 레이블을 가진 경우, 또는 새 클러스터를 시작합니다. 클러스터의 융합 박스는 구성원 좌표의 신뢰도 가중 평균이며, 점수는 신뢰도의 가중 평균 또는 최대값으로, 적은 수의 모델에서 확인된 박스는 점수가 낮게 재조정됩니다.

`wbf_seeded`는 `iou_thr`에서 클래스 인식 NMS로 클러스터 시드를 선택하고, 각 검출을 같은 레이블의 최적 IoU 시드에 할당한 다음, 각 클러스터를 같은 방식으로 축소합니다. 클러스터 모양은 중간 단계에서 절대 변하지 않으므로 전체 연산은 고정된 형태의 텐서 수학입니다. 두 가지 변형은 클러스터가 명확할 때 일치하며, 겹치는 클러스터 체인에서는 약간 다를 수 있습니다.

`nms_fusion`은 각 겹치는 그룹에서 가장 신뢰도가 높은 박스를 변경하지 않고 유지합니다. 모델별 `weights`는 억제 순위를 위해서만 신뢰도를 조정하며, 살아남은 박스는 원래 점수를 유지합니다.

## 맞춤 융합

`fusion=`는 위의 연산과 동일한 시그니처를 가진 호출 가능한(callable) 객체도 받습니다. 그 이름은 `ens.fusion`에 기록되며, 이름이 없으면 `"custom"`에 기록됩니다. 반환값은 검증됩니다: 반환값은 일관된 형태를 가진 `(boxes, scores, labels)` 삼중이어야 합니다.
