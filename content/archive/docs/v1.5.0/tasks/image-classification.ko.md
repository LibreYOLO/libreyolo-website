---
title: 이미지 분류
seo_title: LibreYOLO에서 이미지 분류
description: >-
  LibreYOLO에서 전체 이미지를 레이블링하기: 작업을 수행하는 계열, ImageFolder 데이터셋 레이아웃, 그리고 predict,
  train, validate 및 export 호출.
lead: 이미지 분류는 전체 이미지에 하나의 레이블 분포를 할당하며 내부의 아무 것도 위치시키지 않습니다. 이 작업의 핵심은 분류입니다.
keywords:
  - 이미지 분류 파이썬
  - 이미지 분류기 학습
  - 이미지폴더 데이터셋
  - top-1 정확도
  - 제로샷 분류
  - MIT 분류 라이브러리
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 파일 이름의 -cls 접미사는 작업을 선택하므로 작업이 없습니다
        # 주장이 필요합니다.
        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.names[result.probs.top1], float(result.probs.top1conf))
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 전체 분포
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreResNet50-cls.pt")(SAMPLE_IMAGE)
        probs = result.probs

        # .data는 전체 (C,) 벡터입니다; top5/top5conf는 순서가 지정된 뷰입니다.
        print(probs.data.shape)
        for index, score in zip(probs.top5, probs.top5conf):
            print(result.names[index], float(score))
    - label: '제로샷, 학습 없음'
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # CLIP은 이미지를 텍스트 프롬프트와 비교하여 점수를 매기므로, 레이블 세트는

        # 체크포인트에 내장되지 않고 호출 시 설정됩니다.

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        model.set_classes(["a person jumping", "an empty street", "a parked
        car"])

        result = model(SAMPLE_IMAGE)


        print(model.names[result.probs.top1], float(result.probs.top1conf))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # imagenette160는 잘 알려진 데이터셋 이름이며 처음 사용할 때 다운로드됩니다.
        # 자신의 데이터에 대한 train/ 분할이 있는 디렉터리를 전달하십시오.
        model = LibreYOLO("LibreResNet50-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: 멀티 GPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")

        # val()은 객체가 아니라 일반 딕셔너리를 반환합니다.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 경로를 지정하므로, 내보낸 아티팩트가 로드됩니다
        # 체크포인트처럼 작동하며 같은 Results 객체를 반환합니다.
        model = LibreYOLO("LibreResNet50-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
source_hash: 836bea76cd2cdf92
---

## 정의

이미지 분류는 전체 이미지에 대해 클래스별로 하나의 점수만 생성하며 좌표는 전혀 제공하지 않습니다. 이는 그림 속에 무엇이 있는지를 알려줄 뿐, 위치는 알려주지 않으며, 이것이 [객체 검출](/docs/tasks/object-detection)과 구분되는 점입니다.

`classify`는 표준 작업 키이며, 체크포인트 파일 이름의 `-cls` 접미사가 그것을 선택합니다. 분류 계열에서는 그 접미사가 선택 사항이 아니라 필수이므로 `LibreResNet50.pt`는 분류기로 읽히지 않으며 오직 `LibreResNet50-cls.pt`만 읽힙니다.

`predict()`는 `result.probs`를 채우고 `boxes`를 비웁니다. `.data`는 최고 점수 벡터이고, `.top1`는 최고 점수의 인덱스이며 `.top1conf`는 그 값입니다. `.top5`는 내림차순으로 다섯 개의 최고 인덱스이며 `.top5conf`는 그 점수입니다. 인덱스는 `result.names`를 가리킵니다. `Results` 객체를 슬라이싱해도 `probs`는 절대 잘리지 않습니다. 벡터가 행 하나에 속하는 것이 아니라 이미지에 속하기 때문입니다.

## 모델들

다섯 가지 모델 계열 모두 학습과 예측을 수행합니다: [ResNet](/docs/models/resnet), [ConvNeXt](/docs/models/convnext), [MobileNetV4](/docs/models/mobilenetv4), [EfficientNetV2](/docs/models/efficientnetv2) 및 [DINOv2](/docs/models/dinov2). 처음 네 가지는 기본 패키지에서 실행되며 공개된 가중치를 제공합니다. DINOv2는 `pip install "libreyolo[rfdetr]"`가 필요하며 LibreYOLO에 호스팅된 체크포인트가 없습니다. 이 모델은 상위(backbone)는 로드하지만 선형 헤드는 무작위 초기화되며, 따라서 바로 사용할 수 있는 예측기가 아니라 파인튜닝 시작점입니다.

다섯 가지가 더 예측, 검증 및 내보내기를 수행하지만, 그들의 `train()`는 `NotImplementedError`를 발생시킵니다: [ViT](/docs/models/vit), [Swin](/docs/models/swin), [VGG](/docs/models/vgg), [AlexNet](/docs/models/alexnet) 및 [DeiT](/docs/models/deit).

[CLIP](/docs/models/clip)과 [SigLIP2](/docs/models/siglip2)는 고정된 레이블 집합 없이 분류합니다. 이들은 이미지를 텍스트 프롬프트와 비교하여 점수를 매기기 때문에 `set_classes()`는 호출 시 클래스들을 정의하며, 새로운 레이블 집합에 대한 학습 단계가 전혀 없습니다. 두 모델 모두 `embed` 작업에도 사용됩니다.

## 예측

가중치는 처음 사용할 때 Hugging Face에서 다운로드되며 로컬에 캐시됩니다.

<code-tabs name="predict" />

`conf`, `iou` 및 `max_det`는 여기서 아무 효과가 없습니다: 임계값을 적용하거나 억제할 후보가 없으며, 단 하나의 분포만 있습니다. 소스, 스트리밍 및 결과 처리에 대해서는 [prediction](/docs/predict)를 참조하십시오.

## 데이터셋 형식

분류는 레이블 파일이나 YAML이 아니라 디렉토리 트리를 사용합니다. `data`는 데이터셋 루트입니다.

```text
dataset/
  train/
    tench/000001.jpg
    parachute/000002.jpg
  val/
    tench/000101.jpg
    parachute/000102.jpg
```

`train/`는 학습을 위해 필요하며, 폴더 이름을 정렬하여 클래스-인덱스 매핑을 정의하므로 알파벳 순으로 첫 번째 폴더가 클래스 0이 됩니다. `val/`는 검증을 위해 필요합니다. `test/` 분할이 존재할 수 있으며, 기본 학습 및 검증 명령은 이를 사용하지 않습니다. `train` 이외의 다른 분할은 예상 클래스 집합과 동일한 클래스 폴더 이름을 포함해야 하며, 이는 불일치가 잘못된 예측으로 점수화되는 대신 명확하게 실패하게 만드는 이유입니다. 허용되는 이미지 확장자는 `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif` 및 `.tiff`입니다.

`data`는 세 가지를 허용합니다: `train/` 분할이 포함된 디렉토리 경로, `.zip` URL, 또는 알려진 데이터셋 이름 중 하나인 `imagenette160`와 `smoke10`로, 처음 사용할 때 다운로드되고 캐시됩니다.

정식 로더는 `libreyolo.data.classify_dataset`입니다.

## 학습

<code-tabs name="train" />

선언할 `nc`가 없습니다: 클래스 수는 `train/` 아래 폴더 이름에서 가져오며, 최종 선형 레이어는 이를 맞추기 위해 재구성되지만 백본은 변경 없이 전이됩니다. 데이터셋, 증강, 멀티 GPU 및 로거에 대한 내용은 [training](/docs/train)을 참조하십시오.

## 검증

`val()`는 데이터셋 루트의 `val/` 분할에 대해 계산된 `metrics/` 키의 일반 사전을 반환합니다.

<code-tabs name="val" />

`metrics/accuracy_top1`는 가장 높은 점수를 받은 클래스가 실제 클래스인 이미지의 비율이며, 이는 주요 수치로, 학습에서 최적의 에포크를 선택할 때 사용됩니다. `metrics/accuracy_top5`는 상위 5개의 점수 높은 클래스 중 어느 곳에든 실제 클래스가 나타나는 이미지의 비율로, 데이터셋의 클래스 수가 적을수록 의미가 줄어듭니다. 사전에는 또한 `fitness`가 포함되어 있으며, 이는 top-1 값을 복사한 것입니다.

## 내보내기

<code-tabs name="export" />

내보낸 아티팩트는 파일 접미사를 통해 `LibreYOLO()`로 다시 로드되므로, `.onnx` 또는 `.engine` 파일은 체크포인트처럼 작동하며 동일한 `Results`를 반환합니다. 형식 지원 범위는 계열별로 다르며, 각 모델 페이지의 매트릭스는 수작업으로 작성된 것이 아니라 검증된 세트에서 생성됩니다. 형식, 추가 기능 및 제약 사항은 [export and deploy](/docs/export)를 참조하십시오.
