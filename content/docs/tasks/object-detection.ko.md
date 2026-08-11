---
title: 객체 탐지
seo_title: LibreYOLO에서 객체 탐지
description: >-
  LibreYOLO에서 객체를 축 정렬 바운딩 박스로 탐지하기: 작업을 수행하는 계열, 레이블 형식, 그리고 예측, 학습, 검증 및 내보내기
  호출.
lead: >-
  객체 검출은 이미지 내 모든 객체 인스턴스를 찾아 각 객체에 대해 축에 맞춘 사각형, 클래스 레이블 및 점수를 반환합니다. 작업 키는
  detect입니다.
keywords:
  - 객체 탐지 파이썬
  - 이미지에서 객체 탐지
  - 바운딩 박스 탐지
  - MIT 객체 탐지 라이브러리
  - YOLO 대안
  - 객체 탐지기 학습
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9t.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: '또 다른 계열, 같은 부름'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 검문소에서 경로를 정하고, 모든 탐지기는 반환됩니다
        # 같은 Results 객체이므로, 계열를 변경하는 것은 한 줄만 바꾸면 됩니다.
        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy.shape)
    - label: 비디오와 스트림
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # 라이브러리가 허용하는 모든 소스: 파일, 폴더, URL, 웹캠 인덱스,
        # RTSP 스트림 또는 .streams 목록.
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco128.yaml는 처음 사용할 때 128개의 이미지 샘플을 다운로드합니다. 포인트 데이터
        # 실제 실행을 위해 자신의 데이터셋 YAML에서.
        model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 imgsz=640 batch=8
    - label: 멀티 GPU
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val()는 객체가 아니라 일반 딕셔너리를 반환합니다.
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/AR100"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9t.pt data=coco128.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=onnx imgsz=640
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 경로를 지정하므로, 내보낸 아티팩트가 로드됩니다
        # 체크포인트처럼 작동하며 같은 Results 객체를 반환합니다.
        model = LibreYOLO("LibreYOLO9t.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: c735b6e3de78dd2b
---

## 정의

객체 탐지는 각 객체가 어디에 있는지와 무엇인지에 대한 답을 제공합니다. 한 장의 이미지가 입력되면 각 인스턴스마다 한 줄이 출력됩니다: 사각형을 위한 네 개의 숫자, 클래스 인덱스와 점수입니다. 픽셀 모양, 방향 또는 부품에 대한 정보는 포함되지 않으며, 이것이 [인스턴스 분할](/docs/tasks/instance-segmentation), [방향 박스](/docs/tasks/oriented-detection), [포즈](/docs/tasks/pose-estimation)와 구분되는 점입니다.

`detect`는 표준 작업 키이며 기본값입니다: 파일 이름에 작업 접미사가 없는 체크포인트는 탐지기로 로드됩니다.

`predict()`는 `result.boxes`를 채웁니다. `.xyxy`는 원본 이미지 캔버스에서 픽셀 모서리를 제공하고, `.conf`는 점수를 제공하며, `.cls`는 `result.names`에 대한 클래스 인덱스를 제공합니다. `.xywh`, `.xyxyn` 및 `.xywhn`는 동일한 행에서 파생된 뷰이며, `.id`는 추적기가 연결되면 트랙 ID를 전달합니다. `Boxes` 객체를 반복하면 한 행씩 슬라이스가 생성되므로 `box.cls`, `box.conf` 및 `box.xyxy` 모두 검출마다 작동합니다.

## 모델들

12개의 계열는 학습과 예측을 모두 수행합니다: [YOLOv9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter), [RT-DETR](/docs/models/rt-detr), [D-FINE](/docs/models/d-fine), [DEIM](/docs/models/deim), [Dome-DETR](/docs/models/dome-detr), [YOLO-NAS](/docs/models/yolo-nas), [YOLOX](/docs/models/yolox), [YOLOv7](/docs/models/yolov7), [RTMDet](/docs/models/rtmdet) 및 [PicoDet](/docs/models/picodet). YOLOv9와 RF-DETR은 두 가지 주력 계열이며, 기능들이 먼저 이들에 적용됩니다. RF-DETR은 자체 추가 요소 `pip install "libreyolo[rfdetr]"`가 필요하며, 나머지는 기본 패키지에서 실행됩니다.

추가로 열 가지가 예측, 검증 및 내보내기를 수행하지만, 그들의 `train()`는 `NotImplementedError`를 발생시킵니다: [LW-DETR](/docs/models/lw-detr), [DETR](/docs/models/detr), [Deformable DETR](/docs/models/deformable-detr), [DINO-DETR](/docs/models/dino-detr), [Faster R-CNN](/docs/models/faster-rcnn), [Mask R-CNN](/docs/models/mask-rcnn), [FCOS](/docs/models/fcos), [RetinaNet](/docs/models/retinanet), [SSD](/docs/models/ssd), [CenterNet](/docs/models/centernet) 및 [EfficientDet](/docs/models/efficientdet).

다크넷 계열, [YOLOv1](/docs/models/yolov1), [YOLOv2](/docs/models/yolov2), [YOLOv3](/docs/models/yolov3) 및 [YOLOv4](/docs/models/yolov4)는 냉동된 전시물처럼 유지됩니다: 예측, 검증 및 내보내기는 가능하지만, 학습은 되지 않습니다.

별도의 그룹은 체크포인트에서 가져오는 대신 실행 시점에 클래스 목록을 가져오므로 학습에서 한 번도 본 적 없는 이름도 탐지합니다: [Grounding DINO](/docs/models/grounding-dino), [OWLv2](/docs/models/owlv2), [OMDet-Turbo](/docs/models/omdet-turbo) 및 [OV-DEIM](/docs/models/ov-deim), 그리고 비전-언어 계열 [Florence-2](/docs/models/florence-2), [Kosmos-2](/docs/models/kosmos-2), [Qwen3-VL](/docs/models/qwen3-vl), [SmolVLM2](/docs/models/smolvlm2), [InternVL3](/docs/models/internvl3), [LFM2-VL](/docs/models/lfm2-vl), [LocateAnything](/docs/models/locate-anything), [SenseNova-Vision](/docs/models/sensenova-vision) 및 [LibreMODUS](/docs/models/libremodus). 이들은 자체 팩토리과 추가 항목을 통해 로드되며; 각 모델 페이지는 정확한 호출을 포함합니다.

## 예측

가중치는 처음 사용할 때 Hugging Face에서 다운로드되며 로컬에 캐시됩니다.

<code-tabs name="predict" />

`conf`는 신뢰도 임계값을 설정하고 `max_det`는 행의 수를 제한합니다. `iou`는 NMS 임계값으로, NMS를 실행하는 계열에만 영향을 미칩니다. RF-DETR와 end-to-end YOLOv9 헤드는 고정된 예측을 디코딩하고 이를 무시합니다. 소스, 스트리밍 및 결과 처리는 [prediction](/docs/predict)를 참조하십시오.

## 데이터셋 형식

이미지마다 하나의 `.txt` 레이블 파일이 있으며, 이미지 경로에서 `images`를 `labels`로 바꾸고 확장자를 변경하여 찾을 수 있습니다.

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

각 행은 정확히 다섯 개의 필드로 구성되며, 클래스 인덱스 다음에 정규화된 중심과 크기 상자가 옵니다:

```text
<class_id> <cx> <cy> <w> <h>
```

좌표는 원본 이미지의 너비와 높이를 기준으로 `[0, 1]`에서 부동 소수점입니다. `w`와 `h`는 양수여야 합니다. 레이블 파일이 없거나 비어 있으면 이미지에 객체가 없음을 의미합니다. 행에는 신뢰도나 트랙 ID가 없습니다.

YAML은 분할과 클래스를 명명합니다:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

`train` 및 `val`는 이미지 디렉토리, 이미지 리스트 `.txt` 파일 또는 둘 중 하나의 리스트일 수 있습니다. `nc`는 선택 사항이며 존재하는 경우 `names`와 일치해야 합니다. 네이티브 COCO JSON도 작동합니다: 분할 이름에서 JSON 파일로의 `annotations` 매핑을 추가하면, 분할 경로가 이미지 루트를 제공합니다. `names`가 존재하면 레이블 ID를 정의하므로 JSON 카테고리 이름이 이에 맞아야 합니다.

## 학습

<code-tabs name="train" />

`epochs`, `imgsz`, `batch` 및 `lr0`는 먼저 이동하는 인수입니다. `lr0`는 계열 간에 전달되지 않는 인수입니다. 컨볼루션 탐지기가 허용하는 속도는 트랜스포머에서는 다르게 나타나므로, 다른 계열의 예가 아닌 모델 페이지에서 값을 가져오십시오. 계열은 또한 어떤 인수를 완전히 무시할 수도 있으며, 해당 페이지에 어떤 인수를 무시하는지 목록이 나와 있습니다. 데이터셋, 증강, 멀티 GPU 및 로거에 대해서는 [training](/docs/train)을 참조하십시오.

## 검증

`val()`는 데이터셋 YAML에서 `val`로 명명된 분할에 대해 COCO 평가로 계산된 `metrics/` 키의 일반 딕셔너리를 반환합니다.

<code-tabs name="val" />

`metrics/mAP50-95`는 IoU 임계값 0.50에서 0.95까지 평균한 평균 정밀도를 의미하며, 주요 지표입니다. `metrics/mAP50`와 `metrics/mAP75`는 단일 임계값 버전입니다. `metrics/mAP_small`, `metrics/mAP_medium`, `metrics/mAP_large`는 동일한 평균을 객체 크기별로 나눈 것이며, `metrics/AR1`, `metrics/AR10`, `metrics/AR100`, `metrics/AR_small`, `metrics/AR_medium`, `metrics/AR_large`는 매칭 평균 재현율 수치입니다. `metrics/AR_max_det`와 `metrics/max_det`는 실행에서 사용된 검출 상한값을 기록합니다.

이 작업에서 `metrics/precision`와 `metrics/recall`을 주의 깊게 읽으십시오. 이들은 이전 호환성을 위해 유지되며 별칭일 뿐, 작동 포인트가 아닙니다: `metrics/precision`는 `metrics/mAP50-95`와 동일한 값을 가지며, `metrics/recall`는 `metrics/AR100`와 동일한 값을 가집니다. 이를 정밀도-재현율 쌍으로 표시하면 하나의 숫자가 두 번 보고됩니다. 또한 `(B)` 접미사 아래 네 개의 키가 상자용으로 반복되어, 마스크도 예측하는 모델에서 탐지 키가 동일하게 읽히도록 합니다: `metrics/mAP50-95(B)`, `metrics/mAP50(B)`, `metrics/precision(B)` 및 `metrics/recall(B)`.

## 내보내기

<code-tabs name="export" />

내보낸 아티팩트는 파일 접미사를 통해 `LibreYOLO()`으로 다시 로드되므로, `.onnx` 또는 `.engine` 파일은 체크포인트처럼 작동하며 동일한 `Results`를 반환합니다. 형식 지원 범위는 계열마다 다릅니다. 각 모델 페이지의 매트릭스는 수동으로 입력된 것이 아니라 검증된 세트에서 생성됩니다. 형식, 추가 기능 및 제약 사항은 [export and deploy](/docs/export)를 참조하십시오.
