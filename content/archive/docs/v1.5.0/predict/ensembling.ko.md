---
title: 검출기 앙상블
seo_title: LibreYOLO에서 탐지기 앙상블하기
description: >-
  하나의 이미지에 여러 탐지기를 실행하고, 서로 다른 클래스 목록을 가진 모델을 포함하여, 가중치 박스 융합 또는 NMS로 그들의 박스를
  융합합니다.
lead: >-
  LibreEnsemble은 같은 디코딩된 이미지에 대해 두 개 이상의 탐지기를 실행하고 그들의 박스를 하나의 Results 객체로
  통합합니다. 멤버들은 자신들의 가중치, 임계값, 장치 및 클래스 목록을 유지합니다.
keywords:
  - 모델 앙상블 객체 탐지
  - 가중치 박스 융합
  - wbf 파이썬
  - 두 개의 탐지기를 결합하다
  - 바운딩 박스 합치기
  - 리브르앙상블
  - 앙상블 탐지 파이썬
  - 최소_투표수
last_verified: 1.5.0
verification: >-
  생성자 및 호출 시그니처, 기본값, 검증 오류, 클래스 공간 통합, 투표 집계 및 libreyolo/ensemble/model.py.
  Fusion 알고리즘에서 읽은 반환 결과와 libreyolo/ops/fusion.py.에서 가져온 인수,
  docs/adr/0004-model-ensembling.md.에서의 설계 의도, tests/unit/test_ensemble.py 및
  tests/unit/test_ops_fusion.py.와 교차 확인된 사용 패턴
snippets:
  basic:
    - label: '두 개의 검출기, 융합됨'
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        # 멤버는 체크포인트 경로나 이미 로드된 모델일 수 있습니다.
        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        result = ensemble(SAMPLE_IMAGE)
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: 가중치와 투표 요건
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(
            ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
            weights=[1.0, 1.3],   # 관례적으로, 검증 mAP에 비례하여
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,          # 두 멤버 모두 찾은 상자만 유지
        )

        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes), "agreed detections")
    - label: 회원별 기준치
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # 스칼라는 모든 멤버에 적용되며, 리스트는 멤버별로 읽힙니다.
        result = ensemble(SAMPLE_IMAGE, conf=[0.3, 0.5], iou=0.5)
        print(len(result.boxes))
  external:
    - label: 탐지기를 가져오면 LibreYOLO가 로드되지 않았습니다
      language: python
      code: |
        from libreyolo import ExternalDetector, LibreEnsemble, SAMPLE_IMAGE

        def my_detector(pil_image):
            # 반환 (boxes, scores, labels): 원본 이미지 픽셀에서 xyxy.
            return ([[100.0, 100.0, 200.0, 300.0]], [0.9], [0])

        external = ExternalDetector(my_detector, names={0: "person"})

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", external])
        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes))
  sources:
    - label: 단일 모델이 사용하는 동일한 소스
      language: python
      code: |
        from libreyolo import LibreEnsemble

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # clip.mp4를 디스크에 있는 비디오 파일로 교체하십시오.
        for result in ensemble("clip.mp4", stream=True, vid_stride=2):
            print(result.frame_idx, len(result.boxes))
source_hash: 6dcd2f84ec6f3f65
---

## 앙상블이란 무엇인가

`LibreEnsemble`는 두 개 이상의 탐지기를 사용하여 각 탐지기를 동일한 이미지에 적용한 후, 그 박스들을 하나의 `Results`로 결합합니다. 이것은 예측 시점의 구성으로, 학습할 것은 없으며, 구성원들은 독립적인 모델로 남아 각자 검증 및 내보내기가 가능합니다.

탐지는 그것이 지원하는 유일한 작업입니다. 작업이 다른 무엇이든인 구성원은 구성 시 `ValueError`를 발생시키며, 구성원 인덱스와 해당 작업을 표시합니다.

두 이름 모두 지연 방식으로 가져오므로 사용될 때까지 비용이 들지 않습니다:

```python
from libreyolo import LibreEnsemble, ExternalDetector
```

## 건물 하나

<code-tabs name="basic" />

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

`members`는 두 개 이상의 시퀀스입니다. `str` 또는 `Path` 항목은 `LibreYOLO()`를 통해 로드되며, 그 외의 경우는 호출 가능해야 하며 `names` 딕셔너리를 제공해야 합니다. 두 개 미만이면 `ValueError`가 발생하며, 일반 문자열을 전달하면 문자들을 반복하는 대신 `TypeError`가 발생합니다.

`weights`는 균일 가중치인 `None`로 기본 설정됩니다. 제공된 가중치는 각 구성원당 하나씩 있어야 하며 반드시 양수여야 하므로, 0 가중치는 구성원을 조용히 제거하는 대신 오류를 발생시킵니다. 문서화된 관례는 이를 각 구성원의 검증 mAP에 비례하도록 설정하는 것입니다.

`fusion_iou`는 기본적으로 `0.55`로 설정되며, 이는 서로 다른 멤버의 박스가 함께 클러스터링되는 IoU입니다. 이것은 멤버별 자체 NMS 설정인 호출당 `iou`와는 다른 임계값입니다.

`min_votes`는 기본적으로 `1`로 설정되며, 이는 단일 멤버가 상자를 운반할 수 있음을 의미합니다. 값을 높이면 해당 수만큼 서로 다른 멤버에 의해 확인된 클러스터만 유지됩니다. 이 값은 멤버 수를 초과하지 않는 양의 정수여야 하며, 실제로 해당 클래스를 아는 멤버 수만큼 클래스별로 제한됩니다. 따라서 단 한 명의 멤버만 학습한 클래스는 조용히 삭제되지 않습니다.

## 융합 방법

세 개는 이름으로 받을 수 있고, 호출 가능 객체도 사용할 수 있습니다.

| `fusion` | 행동 |
|---|---|
| `"wbf"` | 가중치 박스 융합, 순차적이며 논문 [1]에 충실함. 기본값 |
| `"wbf_seeded"` | 원패스 가중 박스 융합; 클래스 인식 NMS가 클러스터 시드를 선택함 |
| `"nms"` | 모든 멤버의 박스를 연결한 후, 클래스 인식 NMS를 수행 |

[1] Roman Solovyev, Weimin Wang, Tatiana Gabruseva, ["Weighted boxes fusion:
Ensembling boxes from different object detection models"](https://arxiv.org/abs/1910.13302),
arXiv:1910.13302.

가중 박스 융합은 신뢰도에 의해 가중된 클러스터의 좌표를 평균하여 단일 멤버가 제안하지 않은 박스를 생성합니다. 두 가지 가중 변형은 클러스터가 명확할 때 항상 일치하며, 겹치는 클러스터 체인에서는 약간 다를 수 있습니다. `"nms"`는 평균 대신 생존자를 선택하므로 생존자는 원래 점수를 유지하고, 가중치는 어떤 박스가 승리할지에만 영향을 미칩니다. 선택을 클러스터링이 아닌 방식으로 수행하기 때문에 투표를 계산할 수 없습니다: `fusion="nms"`와 `min_votes`를 결합하여 `1`보다 크면 `ValueError`가 증가합니다.

가중 박스 융합은 구성원의 가중치가 지지한 비율에 따라 클러스터의 점수를 재조정합니다. 두 명의 동등한 가중치 멤버가 있는 경우, 그중 한 명만 발견한 박스는 점수의 절반만 유지합니다: `0.9`는 `0.45`가 됩니다. 따라서 융합된 신뢰도는 각 멤버가 실행된 `conf`보다 낮아질 수 있으므로, 멤버 임계값이 여전히 적용된다고 가정하지 말고 융합된 점수를 기준으로 필터링해야 합니다.

## 다른 반 목록을 가진 회원들

회원들은 클래스 목록을 공유할 필요가 없습니다. 그들의 레이블 공간은 이름으로 합집합되며, 각 회원은 자신의 클래스 ID를 합집합으로 다시 매핑하는 조회 테이블을 받습니다. `ensemble.names`가 그 합집합이며, 반환된 `Results`가 그것을 담고 있습니다.

박스는 항상 같은 클래스 이름 내에서만 결합됩니다. 단 한 명의 멤버만 아는 클래스는 결합되지 않은 채로 통과하며, 이에 대한 벌점은 없습니다. 점수 재조정은 클래스별 분모를 사용하므로, 단독으로 알려진 클래스도 점수를 유지합니다.

부분 겹침은 모든 구성원이 공유하지 않는 클래스의 이름을 경고로 기록합니다. 그 경고는 주의 깊게 읽어야 할 사항입니다. 왜냐하면 클래스 이름이 `class_0`와 같은 자리 표시자인 체크포인트는 다른 모든 구성원과 분리된 합집합을 만들고, 구성원 간의 융합이 전혀 일어나지 않기 때문입니다.

자신의 클래스 ID가 아닌 `names`를 반환하는 멤버는 `RuntimeError`를 발생시킵니다.

## 외국 탐지기

<code-tabs name="external" />

`ExternalDetector(fn, names)`는 PIL 이미지를 입력으로 받고 원본 이미지 픽셀에서 xyxy 형식의 박스를 포함한 `(boxes, scores, labels)`를 반환하는 호출 가능한 객체를 감쌉니다. 이 함수는 인자의 수, 박스 형태, 길이 일치 여부 및 모든 클래스 ID가 `names`에 나타나는지를 검증하며, `conf` 임계값도 직접 적용합니다.

이것이 로드되지 않은 LibreYOLO 탐지기가 융합에 참여하는 방식입니다.

## 부르기

<code-tabs name="sources" />

호출 시그니처는 단일 모델의 시그니처를 반영하며, 동일한 소스를 허용합니다: 이미지, 폴더, 목록, 비디오, 화면 캡처, 웹캠 및 네트워크 스트림. 라이브 소스는 다른 곳과 동일한 이유로 `stream=True`를 필요로 합니다.

| 논쟁 | 기본값 | 노트 |
|---|---|---|
| `conf` | `0.25` | 회원별; 스칼라 브로드캐스트 또는 회원당 하나 |
| `iou` | `0.45` | 각 멤버의 자신의 NMS 임계값, 융합 임계값이 아님 |
| `imgsz` | `None` | `list`는 멤버별로 읽히며; `int`나 튜플은 브로드캐스트됩니다 |
| `device` | `None` | 스칼라 또는 멤버당 하나, 그래서 멤버들이 서로 다른 장치에 앉을 수 있습니다 |
| `classes` | `None` | 결합된 클래스 ID에서 융합 결과를 필터링합니다 |
| `max_det` | `300` | 합쳐진 결과에 적용됨 |

`list`가 `imgsz`의 회원당을 의미하기 때문에 `imgsz=[480, 640]`는 첫 번째 회원에게는 480이고 두 번째 회원에게는 640인 반면, `imgsz=(480, 640)`는 모두에게 하나의 직사각형 크기입니다. 그 구분은 쉽게 혼동될 수 있습니다.

회원들은 요청한 것과 상관없이 최소 300의 `max_det`로 호출되므로 각자는 후하게 실행되고 앙상블은 마지막에 한 번 다듬습니다.

이미지는 한 번 디코딩되며 동일한 객체가 모든 구성원에게 전달됩니다. `batch`는 패리티에 대해 허용되며 무시됩니다; 이미지는 순차적으로 처리됩니다.

## 무엇이 돌아오는가

보통의 `Results`, 단일 모델이 반환하는 동일한 유형으로, `names`가 연합 클래스 공간으로 설정됩니다. [결과 작업](/docs/predict/results)에 있는 모든 내용이 변함없이 적용됩니다.

한 가지 차이점은 `result.speed`로, 이는 앙상블이 채우는 것입니다. 그 키들은 `member_0`, `member_1` 등이 있으며, 여기에 `fusion`도 포함되어 있으며 단위는 밀리초입니다. 라이브러리에서 `speed`가 채워지는 유일한 장소입니다.

비한정 상자나 점수를 가진 행은 융합 전에 제거됩니다. 구성원이 서로 다른 장치에 있을 경우, 융합은 무엇인가를 반환한 첫 번째 구성원의 장치에서 실행됩니다.

## 앙상블이 할 수 없는 것

`val()`와 `export()`는 둘 다 `NotImplementedError`를 올리고 구성원들을 가리킵니다: 각각을 개별적으로 검증하고 내보내십시오. `train` 메서드는 전혀 없으므로, 호출하면 `AttributeError`가 발생합니다.

하프 정밀도는 앙상블 수준에서 처리되지 않습니다. `half=True`는 다른 모든 곳에서와 마찬가지로 동일한 경고된 무작동 경로를 따릅니다; 각 구성원에게 정밀도를 설정하십시오.

앙상블링에는 명령줄 인터페이스가 없습니다. 그것은 파이썬 API입니다.
