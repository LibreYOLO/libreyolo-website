---
title: 자세 추정
seo_title: LibreYOLO에서의 자세 추정
description: >-
  LibreYOLO에서 인스턴스별 주요점을 예측하기: 작업을 수행하는 계열, 레이블 형식, predict, train, validate 및
  export 호출.
lead: >-
  자세 추정은 각 인스턴스의 위치를 찾아 주고, 그것에 대한 명명된 키포인트의 순서화된 집합을 반환하므로 출력에는 객체의 단지 범위뿐만 아니라
  내부 구조가 담깁니다. 이 작업의 핵심은 포즈입니다.
keywords:
  - 자세 추정 파이썬
  - 키포인트 탐지
  - 인간 자세 모델
  - COCO 키포인트
  - OKS mAP
  - 포즈 모델 학습
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 파일 이름의 -pose 접미사는 키포인트 헤드를 선택하므로, 아니요
        # 작업 인수가 필요합니다.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy.shape)   # (N, K, 2) 픽셀 좌표
        print(result.boxes.xyxy.shape)     # (N, 4), 동일한 N개의 인스턴스
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs-pose.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 보이는 키포인트만
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreECs-pose.pt")(SAMPLE_IMAGE)
        kpts = result.keypoints

        # .has_visible는 세 번째 키포인트 열에서 파생되며, 그리고
        # 체크포인트가 오직 (x, y)만 예측할 때 모두 참
        for person, visible in zip(kpts.xy, kpts.has_visible):
            print(person[visible])
    - label: 대신 위에서 아래로
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # HRNet는 상향식입니다: 먼저 각 사람을 잘라냅니다. 사람 소스가 없습니다.
        # 주어진 것은 LibreYOLO9t 탐지기와 짝을 이루고 선택을 기록합니다.
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # coco8-pose.yaml에는 내장 다운로드 스크립트가 포함되어 있으므로 필요합니다
        # 데이터가 이미 로컬에 있는 경우를 제외하고 명시적인 허가가 필요합니다.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="coco8-pose.yaml",
            epochs=50,
            imgsz=640,
            batch=4,
            allow_download_scripts=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs-pose.pt data=coco8-pose.yaml \
          epochs=50 imgsz=640 batch=4 allow_download_scripts=True
    - label: 자신의 데이터셋
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yaml는 kpt_shape를 선언해야 하며, 레이블 행은 가져야 합니다
        # 정확히 5 + K * D 개의 필드.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(data="my-pose-dataset.yaml", epochs=50, imgsz=640, batch=8)
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreECs-pose.pt")


        # val()는 객체가 아니라 일반 딕셔너리를 반환합니다.

        metrics = model.val(data="coco8-pose.yaml", allow_download_scripts=True)


        print(metrics["metrics/keypoints_mAP50-95"])

        print(metrics["metrics/keypoints_mAP50"],
        metrics["metrics/keypoints_mAP75"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs-pose.pt data=coco8-pose.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 경로를 지정하므로, 내보낸 아티팩트가 로드됩니다
        # 체크포인트처럼 작동하며 같은 Results 객체를 반환합니다.
        model = LibreYOLO("LibreECs-pose.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
source_hash: 9de01d1f615bdf33
---

## 정의

자세 추정은 단순한 범위가 아니라 구조를 반환합니다. 각 인스턴스는 여전히 박스, 클래스 및 점수를 받으며, 또한 고정된 순서로 `K` 키포인트를 받습니다. 따라서 인덱스 5는 모든 인스턴스와 모든 이미지에서 동일한 신체 부위를 의미합니다. 레이블 세트가 그 순서를 정의하며, 출력에는 키포인트를 이름으로 식별하는 요소가 없습니다.

`pose`는 표준 작업 키이며, 체크포인트 파일 이름의 `-pose` 접미사가 이를 선택하므로, 공개된 가중치를 로드할 때 `task=`는 필요하지 않습니다.

`predict()`는 `result.boxes`와 함께 `result.keypoints`를 채웁니다. `.data`는 박스와 행이 정렬된 `(N, K, 2)` 또는 `(N, K, 3)`이므로, 하나의 인스턴스 `i`는 다른 인스턴스 `i`와 같습니다. `.xy`는 픽셀 좌표를 슬라이스하고 `.xyn`는 원본 이미지 크기로 정규화합니다. `.conf`는 체크포인트가 하나를 예측할 때 세 번째 열이고, 예측하지 않을 때 `None`이며, `.has_visible`는 여기에서 파생된 불리언 마스크로, 세 번째 열이 없으면 모두 참입니다.

두 가지 아키텍처가 이 출력을 도달합니다. 일단계 모델은 한 번의 패스로 박스와 키포인트를 예측합니다. 톱다운 모델은 먼저 검출기를 실행하고 각 객체를 자른 후, 그 안에서 키포인트를 회귀시키기 때문에 정확도는 앞에 있는 검출기에 달려 있습니다.

## 모델들

세 가지 계열 모두 학습과 예측을 수행합니다: [RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter) 및 [YOLO-NAS](/docs/models/yolo-nas), 모두 단일 단계 모델입니다. RF-DETR은 자체적인 추가 항목 `pip install "libreyolo[rfdetr]"`가 필요합니다. RF-DETR과 EdgeCrafter는 공개된 포즈 체크포인트를 제공하며, 둘 다 단일 클래스인 사람만 포함된 데이터셋에서 파인튜닝됩니다. EdgeCrafter의 키포인트 헤드는 생성 시 고정되어 있으며, 다른 개수를 선언하는 데이터셋은 거부하지만, RF-DETR은 하나를 위해 헤드를 재초기화합니다. YOLO-NAS는 비상업적 라이선스 하에 Deci.AI 자체 CDN에서 가중치를 가져오며, LibreYOLO는 그 어느 것도 공개하지 않습니다; 또한 그 포즈 헤드는 새로운 키포인트 수에 맞게 다시 빌드되며, 세 가지 중 유일하게 클래스 수가 하나로 고정되지 않아 다중 클래스나 인간이 아닌 골격, 예를 들어 동물 포즈를 위한 계열입니다.

[HRNet](/docs/models/hrnet)은 상향식 옵션입니다. 이 모델은 예측하고, 검증하며, 내보내고, 그 `train()`는 `NotImplementedError`를 발생시킵니다. 사람이 입력되지 않으면, 자동으로 LibreYOLO9t 탐지기와 페어링되며; `cropped=True`는 전체 이미지를 하나의 인스턴스로 취급하고, `person_boxes=`는 이미 가지고 있는 박스를 사용하며, `person_detector=`는 다른 탐지기를 지정합니다.

[SenseNova-Vision](/docs/models/sensenova-vision) 또한 키포인트를 출력합니다. 이것은 자체 팩토리 `LibreVLM`와 자체 익스트라를 가진 프롬프트 기반 생성 모델이며, 어휘 세트가 없으면 `set_task("pose")`는 사람 카테고리로 대체됩니다. 이 모델의 가중치는 비상업적용이며, 모든 예측이 디퓨전 디코드이기 때문에 이미지당 지연 시간은 목적에 맞게 제작된 포즈 헤드보다 훨씬 깁니다.

## 예측

가중치는 처음 사용할 때 Hugging Face에서 다운로드되며 로컬에 캐시됩니다.

<code-tabs name="predict" />

키포인트 수와 순서는 라이브러리의 속성이 아니라 체크포인트의 속성이므로, 다른 스켈레톤에서 학습된 모델은 다른 `K`와 인덱스마다 다른 의미를 반환합니다. 세 번째 키포인트 열에 무엇이 들어 있는지도 체크포인트 속성입니다: EdgeCrafter는 점별 점수 대신 상수 값을 거기에 기록하며, 박스 헤드가 전혀 없기 때문에 각 포즈 박스는 해당 인스턴스 자신의 키포인트에 대한 경계 범위입니다. 소스, 스트리밍 및 결과 처리는 [prediction](/docs/predict)을 참조하십시오.

## 데이터셋 형식

레이아웃은 검출 레이아웃입니다: 이미지당 하나의 `.txt` 레이블 파일이 있으며, 이미지 경로에서 `images`를 `labels`로 교체하고 확장자를 변경하여 찾습니다.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

행은 키포인트가 추가된 검출 행입니다:

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

필드 수는 정확히 `5 + K * D`이며, 여기서 `D`는 `kpt_shape`의 두 번째 값입니다. 박스 및 키포인트 좌표는 원본 이미지의 너비와 높이에 대한 정규화된 부동소수점 값입니다. 가시성 `v`는 `D`가 3일 때만 나타나며 `0`, `1` 또는 `2`입니다.

YAML은 공유 계약에 두 개의 키를 추가합니다:

```yaml
path: dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
```

`kpt_shape`가 필요하며, `[K, 2]` 또는 `[K, 3]`입니다. `flip_idx`는 선택 사항이며 `0..K-1`의 순열로, 각 키포인트가 수평 뒤집기 후에 가지는 인덱스를 제공합니다. 이렇게 하면 왼쪽 손목이 왼쪽 손목으로 유지됩니다. 이를 생략하면 키포인트에 대해 잘못된 인덱스 순서로 적용되는 대신 수평 뒤집기 증가가 꺼집니다.

## 학습

<code-tabs name="train" />

학습은 이미 키포인트 헤드를 포함하고 있는 `-pose` 체크포인트에서 계속 진행됩니다; 작업은 학습 시 전달된 플래그가 아니라 로드한 체크포인트에서 읽히므로, 탐지 체크포인트가 단순히 요청한다고 해서 포즈 실행으로 바뀌지는 않습니다. YAML에서 `kpt_shape`는 EdgeCrafter의 경우 헤드가 생성 시 고정되어 있기 때문에 정확히 일치해야 하며, 반면 RF-DETR와 YOLO-NAS는 다른 수에 맞춰 헤드를 조정합니다. 데이터셋, 증강, 다중 GPU 및 로거에 대한 내용은 [training](/docs/train)을 참조하십시오.

## 검증

`val()`는 `metrics/` 키의 일반 딕셔너리를 반환합니다. 점수 매김은 Object Keypoint Similarity를 기반으로 한 COCO 키포인트 평가로, 각 키포인트의 거리 오류를 인스턴스 스케일과 키포인트별 허용치로 가중치하여 상자에서 IoU가 하는 역할을 합니다. 이는 기본 설치에 포함된 `pycocotools`가 필요합니다.

<code-tabs name="val" />

`metrics/keypoints_mAP50-95`는 헤드라인 수치로, OKS 임계값 0.50에서 0.95까지 평균된 평균 정밀도를 의미하며, 학습에서는 가장 좋은 에포크를 선택할 때 사용됩니다. `metrics/keypoints_mAP50`와 `metrics/keypoints_mAP75`는 단일 임계값 버전이며, `metrics/keypoints_mAP_M`와 `metrics/keypoints_mAP_L`는 영역별 평균을 나눈 것으로, 중간과 큰 인스턴스에 대해 나누었습니다; COCO 키포인트 평가에서는 작은 버킷은 정의하지 않습니다. 대응하는 평균 재현율 수치는 `metrics/keypoints_AR50-95`, `metrics/keypoints_AR50`, `metrics/keypoints_AR75`, `metrics/keypoints_AR_M`와 `metrics/keypoints_AR_L`입니다. 이 작업의 모든 키는 `keypoints_`로 접두사가 붙어 있으므로, 상자가 `mAP` 키를 반환하는 경우에는 나타나지 않습니다.

## 내보내기

<code-tabs name="export" />

내보낸 아티팩트는 파일 접미사를 통해 `LibreYOLO()`로 다시 로드되므로, `.onnx` 또는 `.engine` 파일은 체크포인트처럼 작동하며 동일한 `Results`를 반환합니다. 형식 지원 범위는 계열별로 다르며, 각 모델 페이지의 매트릭스는 수작업으로 작성된 것이 아니라 검증된 세트에서 생성됩니다. 형식, 추가 기능 및 제약 사항은 [export and deploy](/docs/export)를 참조하십시오.
