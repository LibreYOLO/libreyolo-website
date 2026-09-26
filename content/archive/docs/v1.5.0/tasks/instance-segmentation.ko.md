---
title: 인스턴스 분할
seo_title: LibreYOLO에서 인스턴스 분할
description: 'LibreYOLO에서 개별 객체를 분할하기: 작업을 수행하는 클래스, 폴리곤 레이블 형식, 그리고 예측, 학습, 검증 및 내보내기 호출.'
lead: >-
  인스턴스 분할은 모든 객체 인스턴스의 위치를 찾아 각 객체에 대한 픽셀 단위 마스크를 반환하며, 동시에 검출기가 반환하는 박스, 클래스 및
  점수도 제공합니다. 작업 키는 segment입니다.
keywords:
  - 인스턴스 분할 파이썬
  - 객체 마스크 예측
  - 분할 모델 학습
  - 다각형 레이블
  - MIT 분할 라이브러리
  - 마스크 mAP
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 파일 이름의 -seg 접미사는 마스크 헤드를 선택하므로 작업이 없습니다
        # 주장이 필요합니다.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)   # (N, H, W), 탐지당 하나의 마스크
        print(result.boxes.xyxy.shape)   # (N, 4), 동일한 N개의 행
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn-seg.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 마스크 윤곽
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE)

        # .xy는 픽셀 단위의 (P, 2) 윤곽선 목록이며, .xyn은 동일하게 정규화된 것입니다.
        for name, contour in zip(result.boxes.cls, result.masks.xy):
            print(result.names[int(name)], contour.shape)
    - label: '또 다른 계열, 같은 부름'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # 발표된 세분화 가중치에서 계속되며, 마스크 헤드가 포함되어 있습니다.

        # 데이터는 레이블이 폴리곤을 포함하는 데이터세트를 가리켜야 합니다.

        model = LibreYOLO("LibreDFINEn-seg.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: 탐지 가중치에서
      language: bash
      code: |
        # 탐지 가중치는 마스크 헤드를 가지고 있지 않으므로, 이것은 명시적입니다
        # 전이: 머리는 학습되지 않은 상태에서 시작합니다. 작업=세그먼트를 요청하는 중입니다
        # 무엇이 그것을 허가하는가.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])       # 마스크
        print(metrics["metrics/mAP50-95(M)"])    # 마스크, 명시적
        print(metrics["metrics/mAP50-95(B)"])    # 상자들
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn-seg.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn-seg.pt format=onnx imgsz=640
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 경로를 지정하므로, 내보낸 아티팩트가 로드됩니다
        # 체크포인트처럼 작동하며 같은 Results 객체를 반환합니다.
        model = LibreYOLO("LibreDFINEn-seg.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 33e331eac0f9b0af
---

## 정의

인스턴스 분할은 탐지에 형태를 더한 것입니다. 각 객체 인스턴스는 여전히 박스, 클래스, 점수를 가지며, 여기에 속하는 픽셀을 덮는 이진 마스크도 함께 부여됩니다. 마스크는 겹칠 수 있으며, 어떤 객체에도 속하지 않는 픽셀은 할당되지 않은 채 남겨지는데, 이것이 이 작업를 [의미 분할](/docs/tasks/semantic-segmentation) 및 [파노라마 분할](/docs/tasks/panoptic-segmentation)과 구분하는 점입니다.

`segment`는 표준 작업 키이며, 체크포인트 파일 이름의 `-seg` 접미사가 이를 선택하므로, 공개된 가중치를 로드할 때 `task=`는 필요하지 않습니다.

`predict()`는 `result.boxes`와 함께 `result.masks`를 채웁니다. `.data`는 원본 이미지 캔버스에 쌓인 `(N, H, W)` 스택으로, 상자들과 행이 정렬되어 있어 마스크 `i`는 상자 `i`에 속합니다. `.xy`는 각각의 마스크를 가장 큰 외곽선으로 `(P, 2)` 픽셀 배열로 변환하고, `.xyn`는 동일한 외곽선을 정규화된 형태로 제공합니다.

## 모델들

네 개의 계열가 마스크를 학습하고 예측합니다: [RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter), [D-FINE](/docs/models/d-fine) 및 [RTMDet](/docs/models/rtmdet). RF-DETR은 자체 추가 패키지 `pip install "libreyolo[rfdetr]"`가 필요하며, 나머지 세 개는 기본 패키지에서 실행됩니다.

[Mask R-CNN](/docs/models/mask-rcnn)은 마스크를 예측, 검증 및 내보내지만, `train()`은 `NotImplementedError`를 발생시킵니다.

[EoMT](/docs/models/eomt)는 마스크를 예측하고 검증하지만 학습할 수 없으며, 내보낸 모델은 더욱 제한적입니다: `export()`는 오직 의미적 작업만 허용하며, `segment`와 `panoptic`에 대해 `NotImplementedError`를 발생시키는데, 이는 두 개의 요청-마스크 런타임 계약이 정의되지 않았기 때문입니다. 파이썬에서 인스턴스 마스크에는 내보낸 그래프가 아닌 EoMT를 사용하십시오.

별도의 그룹이 클래스 목록이 아니라 프롬프트에서 세그먼트를 나눕니다: 클릭, 상자 또는 구문이 객체를 선택하면 모델이 해당 마스크를 반환합니다. [SAM](/docs/models/sam), [SAM 2](/docs/models/sam-2), [SAM 3](/docs/models/sam-3), [MobileSAM](/docs/models/mobilesam), [EdgeTAM](/docs/models/edgetam) 및 [PicoSAM3](/docs/models/picosam3)는 이 방식을 사용하며, [SenseNova-Vision](/docs/models/sensenova-vision)도 마찬가지로, 세그멘테이션이 참조 기반입니다: 하나의 객체를 이름으로 지정하는 구문을 받습니다. 이들은 자체 팩토리와 추가 기능을 통해 로드되며, 각 모델 페이지에는 정확한 호출 방법이 나와 있습니다.

## 예측

가중치는 처음 사용할 때 Hugging Face에서 다운로드되며 로컬에 캐시됩니다.

<code-tabs name="predict" />

`conf`와 `max_det`는 탐지할 때와 같은 방식으로 출력을 형성하며, 마스크는 그들이 속한 박스와 함께 필터링됩니다. 소스, 스트리밍 및 결과 처리는 [prediction](/docs/predict)을 참조하십시오.

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

변경되는 것은 행입니다. 세그먼트는 클래스 인덱스 다음에 평평한 다각형이 옵니다:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

최소 세 개의 점이어야 하며, 따라서 클래스 인덱스 이후의 좌표 수는 짝수이고 최소 여섯 개여야 하며, 폴리곤은 퇴화되지 않아야 합니다. 좌표는 원본 이미지의 너비와 높이를 기준으로 `[0, 1]` 단위의 실수입니다. 분할(segmentation) 데이터셋에서도 다섯 개 필드의 검출 행이 허용되며, 이는 사각형 세그먼트로 읽히므로 박스 전용 데이터셋도 변환 없이 로드할 수 있습니다.

YAML은 탐지 YAML입니다:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

네이티브 COCO JSON도 작동합니다: 분할 이름을 JSON 파일에 매핑하는 `annotations`를 추가하고, 분할 경로가 이미지 루트를 제공합니다.

## 학습

<code-tabs name="train" />

기본적으로 학습은 공개된 `-seg` 체크포인트에서 계속 진행됩니다. 탐지 가중치에서 시작하는 것도 가능하지만, 이는 의도적인 전송입니다: 해당 가중치에는 마스크 헤드가 없기 때문에 학습되지 않은 상태로 시작하며, `task=segment`를 전달하는 것이 교체를 승인하는 방법입니다. 데이터셋, 증강, 다중 GPU 및 로거에 대해서는 [학습](/docs/train)을 참조하십시오.

## 검증

`val()`는 `metrics/` 키의 일반 사전을 반환합니다. 박스와 마스크는 각각 COCO 평가로 점수가 매겨지며, 마스크 숫자가 주요 수치입니다.

<code-tabs name="val" />

접미사가 없는 키는 마스크 결과를 보관합니다: `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, 그 다음에 `metrics/mAP_small`, `metrics/mAP_medium`, `metrics/mAP_large`는 객체 영역별로, `metrics/AR1`, `metrics/AR10`, `metrics/AR100`, `metrics/AR_small`, `metrics/AR_medium`, `metrics/AR_large`는 평균 재현율을 위해 사용됩니다. `metrics/AR_max_det`와 `metrics/max_det`는 실행에서 사용된 탐지 한계를 기록합니다.

마스크에는 `(M)`, 박스에는 `(B)`라는 명시적인 접미사 아래에서도 네 가지 수치가 발행되어, 비교가 계열이 어떤 번호를 기본(primary)으로 부르기로 결정했는지에 의존하지 않도록 되어 있습니다: `metrics/mAP50-95(M)`와 `metrics/mAP50-95(B)`, `metrics/mAP50(M)`와 `metrics/mAP50(B)`, `metrics/precision(M)`와 `metrics/precision(B)`, `metrics/recall(M)`와 `metrics/recall(B)`. 이 작업에는 접미사가 없는 `metrics/precision`나 `metrics/recall`는 없습니다.

정밀도와 재현율 키를 주의 깊게 읽으십시오. 이들은 이전 버전과의 호환성을 위해 유지되며, 운영 포인트가 아니라 별칭입니다: `metrics/precision(M)`는 `metrics/mAP50-95(M)`와 동일한 값을 가지고 있으며, `metrics/recall(M)`는 100개의 탐지에서 mask AR과 동일한 값을 가지며, `(B)`는 박스에 대해 동일하게 작동합니다. 이 둘을 쌍으로 플로팅하면 한 숫자가 두 번 보고됩니다.

## 내보내기

<code-tabs name="export" />

내보낸 아티팩트는 파일 확장자를 통해 `LibreYOLO()`로 다시 로드되므로, `.onnx` 또는 `.engine` 파일은 체크포인트처럼 작동하며 동일한 `Results`를 반환합니다. 동일한 계열에서 세분화 커버리지는 탐지 커버리지보다 좁습니다. 각 모델 페이지의 매트릭스는 검증된 세트에서 생성되며, 대상이 사용 불가한 이유를 명시합니다. 형식, 추가 기능 및 제약 사항은 [내보내기 및 배포](/docs/export)를 참조하십시오.
