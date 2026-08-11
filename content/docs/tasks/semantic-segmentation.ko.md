---
title: 시맨틱 분할
seo_title: LibreYOLO에서의 의미론적 세분화
description: >-
  LibreYOLO에서 모든 픽셀을 클래스에 레이블링하십시오: 작업을 수행하는 계열들, 밀집 마스크 형식, 그리고 predict, train,
  validate 및 export 호출들.
lead: 시맨틱 분할은 이미지의 모든 픽셀에 클래스를 할당하고 동일 클래스의 인스턴스 간에는 구분을 두지 않습니다. 이 작업의 핵심은 시맨틱입니다.
keywords:
  - 시맨틱 분할 파이썬
  - 픽셀 분류
  - 밀집 예측
  - 세분화 모델 학습
  - 평균 교차 엔트로피 지수
  - MIT 분할 라이브러리
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 파일 이름의 -sem 접미사는 작업을 선택하므로 작업이 없습니다
        # 인수가 필요합니다.
        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # 원본 캔버스 위의 (H, W) 클래스 아이디
        print(mask.classes)      # 정렬된 클래스 ID가 존재하며, 255는 무시됨
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSegformerb0-sem.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 한 번에 한 수업
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreSegformerb0-sem.pt")(SAMPLE_IMAGE)
        mask = result.semantic_mask

        for class_id in mask.classes:
            pixels = mask.class_mask(class_id)   # 부울 (H, W)
            print(result.names[class_id], int(pixels.sum()))
    - label: '또 다른 계열, 같은 부름'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: ADE20K에 대하여
      language: bash
      code: |
        # ade20k.yaml는 약 1GB 크기의 내장 다운로드 스크립트를 포함하고 있습니다
        # 아카이브이므로 데이터가 로컬이 아닌 한 명시적인 허가가 필요합니다.
        libreyolo train model=LibreSegformerb0-sem.pt data=ade20k.yaml \
          epochs=160 imgsz=512 batch=8 allow_download_scripts=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        # val()은 객체가 아니라 일반 딕셔너리를 반환합니다.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 경로를 지정하므로, 내보낸 아티팩트가 로드됩니다
        # 체크포인트처럼 작동하며 같은 Results 객체를 반환합니다.
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 44b92d8ba6062f04
---

## 정의

시맨틱 분할은 객체가 아닌 픽셀에 레이블을 지정합니다. 모든 픽셀은 하나의 클래스 ID를 받으며, 이미지에서 서로 닿아 있는 두 대의 자동차는 경계 없이 자동차 클래스의 하나의 영역이 됩니다. 인스턴스를 세는 것은 [인스턴스 분할](/docs/tasks/instance-segmentation)입니다; 모든 픽셀에 레이블을 지정하고 동시에 인스턴스를 구분하는 것은 [파노라마 분할](/docs/tasks/panoptic-segmentation)입니다.

`semantic`는 표준 작업 키이며, 체크포인트 파일 이름의 `-sem` 접미사가 이를 선택하므로, 공개된 가중치를 로드할 때 `task=`는 필요하지 않습니다.

`predict()`는 `result.semantic_mask`를 채웁니다. `.data`는 원본 이미지 캔버스에서 `(H, W)` 정수 클래스 맵이며, `.classes`는 존재하는 id를 정렬된 순서로 나열하고, `.class_mask(id)`는 하나의 클래스에 대한 불리언 `(H, W)` 선택을 반환합니다. 값 `255`는 무시 레이블입니다: 이는 결코 클래스가 아니며, 손실 및 지표에서 제외되며, `.classes`는 이를 제외합니다.

## 모델들

세 가지 계열은 모두 학습과 예측을 수행합니다: [SegFormer](/docs/models/segformer), [LingBot-Vision](/docs/models/lingbot-vision), [DINOv2](/docs/models/dinov2). SegFormer과 LingBot-Vision은 기본 패키지에서 실행되며 공개된 가중치를 제공합니다. DINOv2는 `pip install "libreyolo[rfdetr]"`가 필요하며 LibreYOLO에 호스팅된 체크포인트가 없습니다: 업스트림(backbone)를 불러오고 그 밀집 헤드는 무작위 초기화에서 시작하므로, 이는 바로 예측할 수 있는 모델이라기보다는 학습을 시작하기 위한 출발점입니다.

네 가지를 더 예측, 검증 및 내보내지만, 그들의 `train()`는 `NotImplementedError`를 높입니다: [FCN](/docs/models/fcn), [DeepLabv3](/docs/models/deeplabv3), [PIDNet](/docs/models/pidnet) 및 [EoMT](/docs/models/eomt).

클래스 세트는 계열별로가 아니라 체크포인트별로 다릅니다. 공개된 가중치는 레이블 공간이 거의 공통점이 없는 데이터셋에서 나온 것이며, 예를 들어 ADE20K의 150개 클래스와 Cityscapes의 19개 클래스가 있습니다. 따라서 체크포인트의 `names`가 무엇을 레이블링할 수 있는지를 알려주며, 두 체크포인트는 동일한 데이터셋에서 학습된 경우에만 비교할 수 있습니다.

## 예측

가중치는 처음 사용할 때 Hugging Face에서 다운로드되며 로컬에 캐시됩니다.

<code-tabs name="predict" />

지도는 픽셀별 argmax이므로 NMS 단계가 없으며 `iou`는 전혀 영향을 미치지 않습니다. `conf`와 `max_det`는 API 일관성을 위해 허용되며 SegFormer, PIDNet 및 기타 밀집 예측기에서는 아무 역할도 하지 않습니다; 예외는 EoMT로, 여기서 `conf`는 쿼리 선택을 필터링합니다. 소스, 스트리밍 및 결과 처리는 [prediction](/docs/predict)를 참조하십시오.

## 데이터셋 형식

각 이미지에는 `.txt` 레이블 파일 대신 단일 채널의 밀집 마스크가 짝지어져 있으며, 이미지 경로에서 마스크 디렉터리로 `images`를 교체하면 찾을 수 있습니다.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  masks/
    train/000001.png
    val/000101.png
```

마스크는 손실 없는 단일 채널 이미지로, 보통 PNG이며, 팔레트 모드 PNG는 팔레트 인덱스로 읽습니다. 각 픽셀 값은 `0..nc-1`의 클래스 ID이고, 값 `255`는 무시를 의미하며, 마스크 해상도는 짝지어진 이미지 해상도와 같아야 합니다.

YAML은 공유 계약 위에 두 개의 키를 사용합니다:

```yaml
path: dataset
train: images/train
val: images/val
masks_dir: masks
nc: 19
names:
  0: road
  1: sidewalk
```

`masks_dir`는 `images`를 대체한 디렉토리 이름으로, 기본값은 `masks`입니다. `label_mapping`는 로드 시 픽셀 값을 마스킹하기 위해 적용되는 선택적 `{source_id: train_id}` 재매핑으로, 이를 통해 1에서 150까지 번호가 매겨진 데이터셋이 0에서 149가 됩니다; 매핑되지 않은 소스 값은 무시 처리되며, 모든 학습 ID는 `0..nc-1` 범위에 있어야 합니다.

`masks_dir`를 제외하면 로더가 대체 모드로 전환됩니다: 마스크는 일반적인 `images`에서 `labels` 규칙을 통해 해결된 폴리곤 레이블에서 로드 시점에 래스터화되며, 객체 클래스 뒤에 `background` 클래스가 추가되어 `nc`가 하나 증가합니다.

정식 로더는 `libreyolo.data.SemanticDataset`입니다.

## 학습

<code-tabs name="train" />

`imgsz`는 탐지기에서는 그렇지 않은 방식으로 여기에서 제한됩니다. 각 계열는 그 입력이 배수여야 하는 나눗수를 선언하며, 이는 패치 그리드나 출력 보폭에 의해 설정되고, 학습과 검증 모두 `imgsz`가 정확히 나누어떨어지지 않을 때 실행 시작 전에 `ValueError`를 올립니다. 나눗수는 SegFormer의 경우 32, LingBot-Vision과 EoMT의 경우 16, DINOv2의 경우 14, FCN과 PIDNet의 경우 8입니다. 데이터셋, 증강, 다중 GPU 및 로거에 대해서는 [training](/docs/train)을 참조하십시오.

## 검증

`val()`는 데이터셋 YAML에서 `val`로 이름 붙여진 분할에 대해 계산된 `metrics/` 키들의 일반 딕셔너리를 반환합니다.

<code-tabs name="val" />

`metrics/mIoU`은 클래스별로 예측된 픽셀과 실제 픽셀의 겹치는 부분을 합집합으로 나눈 값인 교차합 대비(intersection over union)를 의미하며, 클래스별로 평균을 냅니다. 이는 주요 지표 숫자이며 학습 중 최적의 에포크를 선택하는 데 사용됩니다. `metrics/pixel_accuracy`은 올바른 클래스로 지정된 픽셀의 비율을 의미하며, 큰 배경 클래스는 이를 과대 평가할 수 있으므로 비교할 때는 mIoU를 사용하는 것이 적절합니다. `255`로 표시된 픽셀은 어느 쪽에도 포함되지 않습니다. 사전에는 `fitness`도 포함되어 있는데, 이는 mIoU 값을 복사한 것입니다.

## 내보내기

<code-tabs name="export" />

내보낸 아티팩트는 파일 접미사를 통해 `LibreYOLO()`으로 다시 로드되므로, `.onnx` 또는 `.engine` 파일은 체크포인트처럼 작동하며 동일한 `Results`를 반환합니다. 형식 지원 범위는 계열별로 다르며, 각 모델 페이지의 매트릭스는 수작업으로 작성된 것이 아니라 검증된 세트에서 생성됩니다. 형식, 추가 기능 및 제약 사항은 [export and deploy](/docs/export)를 참조하십시오.
