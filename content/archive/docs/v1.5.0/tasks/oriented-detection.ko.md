---
title: 방향 탐지
seo_title: LibreYOLO에서의 방향 탐지
description: >-
  LibreYOLO에서 회전된 객체 탐지: 방향이 있는 박스를 제공하는 계열, 네 모서리 레이블 행, 그리고 예측, 학습, 검증 및 내보내기
  호출.
lead: >-
  방향 객체 검출은 축에 맞춘 사각형이 아니라 회전된 사각형으로 각 객체를 위치시키므로, 기울어진 객체가 배경으로 가득 찬 상자가 아니라
  엄밀하게 경계지어집니다. 이 작업의 핵심은 obb입니다.
keywords:
  - 방향성 바운딩 박스 탐지
  - 회전된 객체 탐지
  - OBB 파이썬
  - DOTA 데이터셋
  - 공중 객체 탐지
  - 회전된 IoU
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        # rfdetr 추가 필요: pip install "libreyolo[rfdetr]"
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 파일 이름의 -obb 접미사는 작업을 선택하므로 작업이 없습니다
        # 주장이 필요합니다.
        model = LibreYOLO("LibreRFDETRs-obb.pt")
        result = model(SAMPLE_IMAGE, save=True)

        obb = result.obb
        print(obb.xywhr)   # (N, 5): 중심 x, 중심 y, 너비, 높이, 라디안
        print(obb.conf, obb.cls)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRFDETRs-obb.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 각도 대신 모서리
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreRFDETRs-obb.pt")(SAMPLE_IMAGE)
        obb = result.obb

        print(obb.xyxyxyxy.shape)    # (N, 4, 2) 픽셀 단위의 코너 포인트
        print(obb.xyxyxyxyn.shape)   # 같은, 정규화된
        print(obb.xyxy.shape)        # (N, 4) 축에 정렬된 바운딩 박스
    - label: 더 작은 검문소
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRn-obb.pt")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr.shape)
    - label: RT-DETRv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # DOTA v1.0 가중치, 1024 px에서 15개의 항공 클래스. 방향 그래프
        # 체크포인트 자체 텐서에서 인식되므로, 작업 인자가 없습니다.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)
        print(result.names)   # 비행기, 배, 항구, 헬리콥터, 그리고 11개 더
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # 출판 지향 가중치에서 계속 진행됩니다. 데이터는 반드시 가리켜야 합니다

        # 레이블 행에 네 개의 모서리가 있는 데이터셋.

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        model.train(data="my-obb-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: 탐지 가중치에서
      language: bash
      code: |
        # 탐지 가중치는 각도 예측을 포함하지 않으므로, 이것은 명시적입니다
        # 전송. 작업=obb를 요청하는 것이 그것을 허가하는 것입니다.
        libreyolo train model=LibreRFDETRs.pt data=my-obb-dataset.yaml \
          task=obb epochs=50 imgsz=512
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        # val()은 객체가 아니라 일반 딕셔너리를 반환합니다.
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml
    - label: RT-DETRv2
      language: bash
      code: |
        libreyolo val model=LibreRTDETRv2n-obb.pt data=my-obb-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRFDETRs-obb.pt format=onnx imgsz=512
    - label: RT-DETRv2
      language: bash
      code: >
        # ONNX와 TorchScript는 여기에서 FP32로 검증된 대상입니다.

        # 배치 1, 고정된 1024 x 1024 캔버스에서.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 경로를 지정하므로, 내보낸 아티팩트가 로드됩니다
        # 체크포인트처럼 작동하며 같은 Results 객체를 반환합니다.
        model = LibreYOLO("LibreRFDETRs-obb.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr)
source_hash: 0d605d956f3ea025
---

## 정의

방향 탐지는 탐지에 하나의 숫자를 추가합니다: 각도입니다. 각 객체는 회전된 사각형, 클래스, 점수를 받습니다. 그 이점은 정확성입니다. 45도 각도의 배, 창고 지붕, 줄 지어 주차된 트럭: 그 중 어느 것에든 축에 맞춘 상자는 대부분 배경이 되며, 객체가 겹치지 않아도 인접한 두 상자가 겹칩니다. 이것이 이 작업이 항공 이미지와 문서 레이아웃에서 표준인 이유이며, 이와 관련된 참조 데이터셋이 DOTA인 이유입니다.

`obb`는 표준 작업 키이며, 체크포인트 파일 이름의 `-obb` 접미사가 이를 선택하므로, 공개된 가중치를 로드할 때 `task=`는 필요하지 않습니다.

`predict()`는 `result.obb`를 채웁니다. `.xywhr`는 표준 `(N, 5)` 형식입니다: 중심 x, 중심 y, 너비, 높이, 그리고 중심을 기준으로 너비 면의 회전을 라디안 단위로 나타내는 각도. `.conf`와 `.cls`는 점수와 클래스 인덱스를 `result.names`로 전달하며, `.id`는 추적할 때 추적 ID를 전달합니다. `.xyxyxyxy`는 각 행을 `(N, 4, 2)` 픽셀로 된 네 개의 모서리 점으로 변환하고, `.xyxyxyxyn`는 그 모서리를 정규화하며, `.xyxy`는 이를 둘러싸는 축 정렬 박스를 제공합니다. 이는 후속 코드가 직사각형만 이해할 때 사용됩니다. `result.boxes`도 채워지며 축 정렬 형식으로 제공됩니다.

## 모델들

이 작업에는 두 개의 계열가 있으며, 어느 것을 선택할지는 학습이 필요한지에 따라 달라집니다.

[RF-DETR](/docs/models/rf-detr)은 학습하는 모델입니다. 이 모델은 예측하고, 학습하고, 검증하며, 방향이 있는 박스를 내보냅니다. 또한 네 가지 크기(n, s, m, l)로 공개된 방향 체크포인트를 제공합니다. 자체 추가 파일 `pip install "libreyolo[rfdetr]"`가 필요하며, 모델 페이지에는 가중치 라이선스와 출처가 표시되어 있습니다.

그 체크포인트들이 실제로 무엇을 예측하는지에 대한 아래 섹션을 읽은 후에 그것을 바탕으로 계획을 세우십시오.

[RT-DETRv2](/docs/models/rt-detr)은 항공 가중치를 가진 모델입니다. 이는 `LibreRTDETRv2x-obb.pt`를 통해 `LibreRTDETRv2n-obb.pt`를 게시하며, DOTA v1.0 공식 단일 스케일 체크포인트를 LibreYOLO 형식으로 변환한 것으로, 1024 px에서 DOTA의 15개 클래스를 포함합니다. 기본 패키지 외에는 추가가 필요 없고, 방향 그래프는 체크포인트 자체의 텐서에서 인식되며, 예측, 검증, ONNX와 TorchScript 내보내기가 모두 지원됩니다. 학습은 지원되지 않으며: 방향 작업은 해당 계열에서 추론만 가능하고, `train()`가 증가하며, 다른 백본을 사용하는 탐지 가중치에서 전이되지 않습니다. 방향 박스에 대한 추적 및 테스트 시 증강도 사용할 수 없습니다.

그래서: DOTA 카테고리는 바로 사용 가능, RT-DETRv2. 직접 지정한 레이블, RF-DETR.

## 예측

가중치는 처음 사용할 때 Hugging Face에서 다운로드되며 로컬에 캐시됩니다.

<code-tabs name="predict" />

RF-DETR의 공개된 체크포인트가 무엇인지 실행하기 전에 알아두십시오. 이 작업의 참조 벤치마크가 DOTA임에도 불구하고, 해당 가중치은 DOTA에서 학습된 것이 아닙니다. 네 개 모두 RF-DETR 탐지 가중치에서 초기화되었고, UAV 영상의 단일 Roboflow Universe 데이터셋에서 여섯 가지 차량 클래스(자전거, 버스, 자동차, 기타 차량, 택시, 트럭)로 파인튜닝되었습니다. 모델 카드에는 이들을 개발용 가중치로 설명하며, 방향 기반 학습 지원을 검증하는 과정에서 생성되었고, 생산용이나 공식 벤치마크 가중치로 간주해서는 안 된다고 명시하고 있습니다.

실제로 이는 위에서 본 차량의 방향성 박스를 위한 작업 시작점이자 파이프라인이 처음부터 끝까지 실행되는지를 검증하는 의미를 갖습니다. 다른 도메인의 경우에는 자체 방향성 레이블을 사용하여 학습해야 하며, 항공 분야 카테고리에서는 DOTA가 유명하며, RT-DETRv2 체크포인트가 실제로 해당 데이터로 학습된 모델입니다. `conf`와 `max_det`는 검출을 위해 출력 형식을 조정합니다. 소스, 스트리밍 및 결과 처리는 [예측](/docs/predict)을 참조하십시오.

## 데이터셋 형식

레이아웃은 검출 레이아웃입니다: 이미지당 하나의 `.txt` 레이블 파일이 있으며, 이미지 경로에서 `images`를 `labels`로 교체하고 확장자를 변경하여 찾습니다.

```text
dataset/
  data.yaml
  images/
    train/P0001.png
    val/P0101.png
  labels/
    train/P0001.txt
    val/P0101.txt
```

한 행은 정확히 아홉 개의 필드로 구성되며, 클래스 인덱스 하나와 순서대로 나열된 네 개의 꼭짓점 좌표를 포함합니다:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

네 점은 `[0, 1]`에서 정규화된 부동 소수점 값이며, 비퇴화된 방향성 사각형을 형성해야 합니다. 레이블 파일에는 각도가 저장되지 않으며, 로더가 모서리에서 정규화된 `xywhr`를 추론합니다. 파서는 기본적으로 엄격하여 범위를 벗어난 좌표를 거부하지만, 데이터셋 및 검증 데이터 수집 시에는 그렇지 않은 경우 유효한 잘린 경계 레이블을 위해 먼저 `[0, 1]`로 클리핑한 후에도 여전히 퇴화된 박스는 거부할 수 있습니다.

행 파싱은 작업 인식 방식입니다. 아홉 개 필드는 `obb` 모드에서는 방향이 있는 상자를 의미하지만, `segment` 모드에서는 동일한 행이 네 점으로 이루어진 다각형으로 읽힙니다.

YAML은 탐지 YAML입니다:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: plane
  1: ship
```

네이티브 COCO JSON도 로드되며, 분할 이름과 JSON 파일을 매핑하는 `annotations` 맵이 있습니다. 주석은 우선순위 순으로 읽습니다: 8개의 픽셀 공간 모서리를 가진 `obb` 필드, 라디안 각도를 가진 `[cx, cy, w, h, angle]`의 `obb` 필드, 최소 면적 사각형으로 조정된 `segmentation` 다각형 또는 RLE, 또는 동일축 정렬 사각형으로 처리되어 `xywhr`로 표준화되는 일반 COCO `bbox`.

표준 행 파서는 `libreyolo.data.parse_yolo_obb_label_line`입니다.

## 학습

<code-tabs name="train" />

이 작업에 대한 학습은 RF-DETR을 의미합니다. 기본적으로 출판된 `-obb` 체크포인트에서 학습이 계속됩니다. 검출 가중치에서 시작하는 것은 의도된 전이입니다: 해당 가중치는 각도를 예측하지 않으며, `task=obb`를 통과하는 것이 교환을 승인하는 것입니다. `lr0`는 계열의 다른 작업과 마찬가지로 `1e-4` 이하로 유지하십시오. RT-DETRv2의 방향 체크포인트는 파인튜닝할 수 없습니다; 그대로 사용하거나 RF-DETR 모델을 자체 레이블로 학습하십시오. 데이터셋, 증강, 다중 GPU 및 로거에 대한 내용은 [training](/docs/train)을 참조하십시오.

## 검증

`val()`는 `metrics/` 키의 일반 사전을 반환합니다. 매칭은 회전된 IoU를 사용하여 수행되며, 이는 축에 맞춰진 직사각형이 아니라 방향이 있는 직사각형 간에 계산됩니다. 따라서 위치는 맞지만 각도가 틀린 예측은 실패로 점수가 매겨집니다.

<code-tabs name="val" />

`metrics/mAP50-95`는 IoU 임계값 0.50에서 0.95까지 0.05 단위로 평균된 평균 정밀도(mean average precision)이며, 주요 수치입니다. 탐지에서 사용되는 COCO 경로와 달리, 이 작업은 검증 구성(validation config)에서 `iou_thresholds`를 존중하므로 스윕(sweep)을 변경할 수 있습니다. `metrics/mAP50`와 `metrics/mAP75`는 단일 임계값 버전입니다. `metrics/precision`와 `metrics/recall`는 IoU 0.50에서의 실제 정밀도와 재현율이며, 가장 느슨한 동작 지점(loose operating point)에서 측정됩니다. 즉, 신뢰도 임계값(confidence threshold)을 통과한 모든 예측이 포함되며, 검증 중에 해당 임계값의 기본값은 0.001입니다. 따라서 `conf`를 높이면 이 값들이 이동하지만, 전체 정밀도-재현율 곡선을 사용하는 mAP 수치는 그대로 유지됩니다. 이 중 네 개는 `(OBB)` 접미사 아래에서 반복되며, `metrics/mAP50-95(OBB)`, `metrics/mAP50(OBB)`, `metrics/precision(OBB)`, `metrics/recall(OBB)`로 이어지는데, 이는 호출자가 동일한 테이블에 있을 때 축 정렬 결과와 방향이 지정된 결과를 구분하는 방법입니다. `metrics/mAP75`에는 접미사가 붙은 쌍이 없습니다.

이 작업에서는 두 가지 옵션이 아무런 동작을 하지 않습니다. `save_json`와 `save_plots`는 허용되며 경고를 기록합니다: 방향 예측 덤프와 검증 플롯은 구현되지 않았습니다.

## 내보내기

<code-tabs name="export" />

내보낸 아티팩트는 파일 확장자를 통해 `LibreYOLO()`로 다시 로드되므로, `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다. 동일한 계열의 작업마다 형식 지원 범위가 다르며, 모델 페이지의 매트릭스는 검증된 세트에서 생성되며 목표가 사용할 수 없는 이유를 명시합니다. 형식, 추가 기능 및 제약조건에 대해서는 [내보내기 및 배포](/docs/export)를 참조하십시오.
