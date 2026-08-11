---
title: 탐지기 앙상블
seo_title: LibreYOLO에서 탐지기 앙상블
description: '하나의 이미지에서 여러 탐지기를 실행하고 가중치 박스 융합 또는 NMS로 박스를 결합하며, 다른 클래스 목록을 가진 모델도 포함'
lead: >-
  LibreEnsemble은 같은 디코딩된 이미지에서 두 개 이상의 탐지기를 실행하고 그들의 박스를 하나의 Results 객체로 결합.
  구성원은 자신의 가중치, 임계값, 장치 및 클래스 목록을 유지
keywords:
  - 모델 앙상블 객체 탐지
  - 가중치 박스 융합
  - WBF 파이썬
  - 두 탐지기 결합
  - 바운딩 박스 결합
  - LibreEnsemble
  - 앙상블 감지 파이썬
  - 최소 투표 수
last_verified: 1.5.0
verification: >-
  생성자 및 호출 서명, 기본값, 검증 오류, 클래스 공간 통합, 투표 집계 및 libreyolo/ensemble/model.py.에서 읽은
  반환된 결과, libreyolo/ops/fusion.py.에서의 융합 알고리즘과 그 인수,
  docs/adr/0004-model-ensembling.md.에서의 설계 의도, tests/unit/test_ensemble.py 및
  tests/unit/test_ops_fusion.py.와 교차 검증된 사용 패턴
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
    - label: 가중치 및 투표 요구조건
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(
            ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
            weights=[1.0, 1.3],   # 관례상, 검증 mAP에 비례
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,          # 두 멤버가 모두 찾은 박스만 유지
        )

        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes), "agreed detections")
    - label: 멤버별 임계값
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # 스칼라는 모든 멤버에 적용; 리스트는 멤버별로 읽음
        result = ensemble(SAMPLE_IMAGE, conf=[0.3, 0.5], iou=0.5)
        print(len(result.boxes))
  external:
    - label: LibreYOLO가 로드하지 않은 검출기를 가져오기
      language: python
      code: |
        from libreyolo import ExternalDetector, LibreEnsemble, SAMPLE_IMAGE

        def my_detector(pil_image):
            # 반환 (박스, 점수, 레이블): 원본 이미지 픽셀의 xyxy.
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
source_hash: 4f4c54c52b295795
---

## 앙상블이란 무엇인가

`LibreEnsemble`는 두 개 이상의 검출기를 사용하여 각 검출기를 동일한 이미지에 실행하고, 그들의 박스를 하나의 `Results`로 융합합니다. 이는 예측 시점의 구조로, 학습할 것은 없으며, 구성원들은 독립적인 모델로 유지되어 자체적으로 검증 및 내보내기가 가능합니다.

검출만이 지원되는 유일한 작업입니다. 다른 작업을 수행하는 구성원이 있으면 `ValueError`를 생성 시점에 발생시키며, 구성원 인덱스와 그 작업을 함께 명시합니다.

두 이름 모두 지연 로딩되어 사용될 때까지 비용이 들지 않습니다:

```python
from libreyolo import LibreEnsemble, ExternalDetector
```

## 하나 만들기

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

`members`는 두 개 이상의 시퀀스입니다. `str` 또는 `Path` 항목은 `LibreYOLO()`를 통해 로드되며, 그 외의 항목은 호출 가능해야 하고 `names` 딕셔너리를 노출해야 합니다. 두 개 미만이면 `ValueError`를 발생시키고, 일반 문자열을 전달하면 문자를 반복하지 않고 `TypeError`를 발생시킵니다.

`weights`의 기본값은 `None`이며, 이는 균일 가중치입니다. 제공된 가중치는 각 멤버마다 하나씩 있어야 하며 엄격히 양수여야 하므로, 0 가중치는 멤버를 조용히 제거하지 않고 대신 오류를 발생시킵니다. 문서화된 관례는 각 멤버의 검증 mAP에 비례하도록 설정하는 것입니다.

`fusion_iou`의 기본값은 `0.55`이며, 이는 다른 멤버의 박스들이 함께 클러스터링되는 IoU입니다. 이는 각 호출의 `iou`, 즉 각 멤버의 NMS 설정과는 다른 임계값입니다.

`min_votes`는 기본적으로 `1`로 설정되며, 이는 한 명의 구성원이라도 상자를 들 수 있다는 의미입니다. 값을 올리면 해당 만큼 서로 다른 구성원으로 확인된 클러스터만 유지됩니다. 이는 구성원 수를 넘지 않는 양의 정수여야 하며, 각 클래스별로 실제로 그 클래스를 아는 구성원 수로 제한되므로, 한 명의 구성원만 학습된 클래스가 조용히 제거되지 않습니다.

## 융합 방법

이름으로 세 가지가 허용되며, 호출 가능한 객체도 허용됩니다.

| `fusion` | 동작 |
|---|---|
| `"wbf"` | 가중 상자 융합, 순차적이며 논문에 충실함. 기본값 |
| `"wbf_seeded"` | 단일 통과 가중 상자 융합; 클래스 인식 NMS가 클러스터 시드를 선택합니다. |
| `"nms"` | 모든 구성원의 상자를 연결한 다음 클래스 인식 NMS 수행 |

가중 박스 융합은 신뢰도로 가중된 클러스터의 좌표 평균을 계산하여 단일 구성원이 제안하지 않은 박스를 생성합니다. 클러스터가 명확할 때 두 가지 가중 변형은 일치하며, 겹치는 클러스터 체인에서는 약간 다를 수 있습니다. `"nms"`는 평균을 내는 대신 생존자를 선택하므로, 생존자는 원래 점수를 유지하고 가중치는 어떤 박스가 승리할지에만 영향을 줍니다. 선택 방식을 사용하기 때문에 투표를 계산할 수 없으며, `fusion="nms"`와 `min_votes`를 조합하고 `1`보다 크면 `ValueError`가 올라갑니다.

가중 상자 융합은 클러스터 점수를 이를 지지한 구성원 가중치 비율로 재조정합니다. 가중치가 동일한 두 구성원이 있는 경우, 그 중 한 명만 발견한 상자는 점수의 절반만 유지합니다: `0.9`는 `0.45`가 됩니다. 따라서 융합된 신뢰도는 구성원 각각이 실행된 `conf` 이하로 떨어질 수 있으므로, 구성원 임계값이 여전히 유효하다고 가정하지 말고 융합된 점수를 기준으로 필터링해야 합니다.

## 서로 다른 클래스 목록을 가진 구성원

구성원은 클래스 목록을 공유할 필요가 없습니다. 그들의 레이블 공간은 이름별로 합집합되어, 각 구성원은 자신의 클래스 ID를 합집합으로 재매핑하는 조회 테이블을 받습니다. `ensemble.names`는 그 합집합이고, 반환되는 `Results`가 이를 운반합니다.

박스는 동일한 클래스 이름 내에서만 융합됩니다. 한 명의 멤버만 아는 클래스는 융합되지 않은 채로 통과하며, 이에 대한 불이익은 없습니다: 점수 재조정은 클래스별 분모를 사용하므로, 혼자 알려진 클래스는 점수를 유지합니다.

부분 겹침은 모든 멤버가 공유하지 않는 클래스의 이름을 경고로 기록합니다. 이 경고를 주의 깊게 읽어야 합니다. 왜냐하면 클래스 이름이 `class_0`와 같은 자리 표시자인 체크포인트는 모든 다른 멤버와 겹치지 않는 합집합을 생성하며, 멤버 간 융합이 전혀 일어나지 않기 때문입니다.

자신의 `names` 외부에서 클래스 ID를 반환하는 멤버는 `RuntimeError`를 발생시킵니다.

## 외부 탐지기

<code-tabs name="external" />

`ExternalDetector(fn, names)`는 PIL 이미지를 입력으로 받고 `(boxes, scores, labels)`를 반환하는 모든 호출 가능 객체를 래핑하며, 박스는 원본 이미지 픽셀 기준으로 xyxy 형식입니다. 이것은 함수 인자의 수, 박스 형상, 길이 일치 여부 및 모든 클래스 ID가 `names`에 나타나는지 검증하며, `conf` 임계값을 자체적으로 적용합니다.

이렇게 LibreYOLO가 로드되지 않은 탐지기가 융합에 참여합니다.

## 호출 방식

<code-tabs name="sources" />

호출 서명은 단일 모델과 동일하며, 이미지, 폴더, 리스트, 비디오, 화면 캡처, 웹캠 및 네트워크 스트림 등 동일한 소스를 받습니다. 라이브 소스는 다른 곳에서와 동일한 이유로 `stream=True`가 필요합니다.

| 인수 | 기본값 | 참고 |
|---|---|---|
| `conf` | `0.25` | 멤버별; 스칼라 브로드캐스트 또는 멤버당 하나 |
| `iou` | `0.45` | 각 멤버의 자체 NMS 임계값, 융합 임계값 아님 |
| `imgsz` | `None` | 멤버당 `list` 하나가 읽히며; `int` 또는 튜플은 브로드캐스트합니다 |
| `device` | `None` | 스칼라 또는 멤버당 하나씩, 그래서 멤버들이 다른 장치에 분산될 수 있습니다 |
| `classes` | `None` | 결합된 결과를 필터링하며, 유니언 클래스 ID에 적용됩니다 |
| `max_det` | `300` | 결합된 결과에 적용됩니다 |

`list`가 `imgsz` 당 멤버를 의미하기 때문에, `imgsz=[480, 640]`는 첫 번째 멤버는 480, 두 번째 멤버는 640이고, `imgsz=(480, 640)`는 모든 사람에게 하나의 직사각형 크기입니다. 이 구분은 쉽게 혼동될 수 있습니다.

멤버들은 요청한 것과 관계없이 최소 300의 `max_det`로 호출되므로, 각 멤버는 충분히 실행되고 앙상블은 마지막에 한 번 트리밍됩니다.

이미지는 한 번 디코딩되며 동일한 객체가 모든 구성원에게 전달됩니다. `batch`는 패리티용으로 받아들여지며 무시됩니다; 이미지는 순차적으로 처리됩니다.

## 반환되는 것

일반적인 `Results`, 단일 모델이 반환하는 동일한 유형이며 `names`는 합집합 클래스 공간으로 설정됩니다. [결과 작업](/docs/predict/results)에서 적용되는 모든 내용은 변경되지 않습니다.

유일한 차이점은 `result.speed`로, 앙상블이 채우는 값입니다. 그 키는 `member_0`, `member_1` 등이며, `fusion`는 밀리초 단위입니다. 라이브러리에서 `speed`가 채워지는 유일한 위치입니다.

유한하지 않은 박스 또는 점수를 가진 행은 융합 전에 삭제됩니다. 구성원이 다른 장치에서 실행되는 경우, 융합은 처음으로 데이터를 반환한 구성원의 장치에서 실행됩니다.

## 앙상블이 수행할 수 없는 것

`val()`와 `export()`는 모두 `NotImplementedError`를 발생시키며 멤버를 가리킵니다: 각 멤버를 개별적으로 검증하고 내보내세요. `train` 메서드는 전혀 없으므로, 호출하면 `AttributeError`가 발생합니다.

하프 정밀도는 앙상블 레벨에서 처리되지 않습니다. `half=True`는 다른 모든 경우와 마찬가지로 경고된 no-op 경로를 타며; 각 멤버에서 정밀도를 설정하십시오.

앙상블을 위한 명령줄 인터페이스는 없습니다. 이것은 Python API입니다.
