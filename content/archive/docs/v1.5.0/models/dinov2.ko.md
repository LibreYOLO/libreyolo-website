---
title: DINOv2
families:
  - dinov2
seo_title: 'LibreYOLO의 DINOv2: 의미 분할, 분류 및 임베딩'
description: >-
  LibreYOLO에서 DINOv2-with-Registers 백본으로 의미 분할, 분류, 전체 이미지 임베딩을 수행합니다. 전체에
  Apache-2.0이 적용됩니다.
lead: >-
  DINOv2는 레이블 없이 범용 이미지 특징을 생성하도록 Meta AI가 학습한 자기 지도 vision transformer입니다.
  LibreYOLO는 DINOv2-with-Registers 백본을 의미 분할, 분류, 전체 이미지 임베딩의 세 작업용으로 래핑합니다.
keywords:
  - DINOv2 사용법
  - DINOv2 with registers
  - 자기 지도 학습
  - vision transformer
  - 의미 분할
  - 이미지 임베딩
  - 특징 추출
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: 의미 분할
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # 이 계열에는 LibreYOLO 호스팅 체크포인트가 없습니다. Meta Hugging Face
        # 조직에서 Apache-2.0 DINOv2-with-Registers-small 백본을 다운로드합니다.
        # 아래 학습을 수행하기 전까지 조밀 헤드는 무작위 초기화 상태입니다.
        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        result = model(SAMPLE_IMAGE)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: 분류
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # nb_classes=는 데이터셋 클래스 수입니다. 선형 헤드는 학습하기 전까지
        # 무작위 초기화 상태입니다.
        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
    - label: 임베딩
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # 모든 작업 헤드를 우회합니다. 백본만으로 충분하므로 파인튜닝 없이도
        # 유용하게 사용할 수 있습니다.
        model = LibreDINOv2(size="s", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (1, D), L2 정규화
    - label: 배치 임베딩
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # 편의 래퍼로 predict()를 실행하고 모든 행을 (N, D) 텐서 하나로 쌓습니다.
        features = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(features.shape)
  train:
    - label: 의미 분할
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: 분류
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: 다중 GPU
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(
            data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4,
            device="0,1",
        )
  val:
    - label: 의미 분할
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: 분류
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
  export:
    - label: 의미 분할
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.export(format="onnx")
    - label: 분류
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.export(format="onnx")
    - label: 임베딩
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        model.export(format="tflite")
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        # 내보내기는 작업을 기준으로 파일명을 정하며 여기서는 LibreDINOv2s-sem.onnx입니다.
        model = LibreYOLO("LibreDINOv2s-sem.onnx")
        result = model(SAMPLE_IMAGE)
source_hash: 4256e0a0398e5aaf
---

## 설치

LibreDINOv2는 RF-DETR의 DINOv2 백본에 필요한 것과 같은 선택적 종속성인 `transformers`가 설치된 경우에만 등록되므로 동일한 extra가 필요합니다.

```bash
pip install "libreyolo[rfdetr]"
```

## 예측

LibreYOLO는 LibreDINOv2 체크포인트를 게시하지 않습니다. 파일을 불러오는 대신 래퍼를 직접 생성합니다. 기본값인 `model_path=None`은 처음 사용할 때 Hugging Face에서 Meta의 Apache-2.0 `facebook/dinov2-with-registers-small` 백본을 다운로드합니다. `task=`가 그 위에서 실행할 작업을 선택합니다.

<code-tabs name="predict" />

`task="semantic"`과 `task="classify"`는 백본 위에 조밀 또는 선형 헤드를 추가합니다. 해당 헤드는 무작위로 초기화되어 학습한 뒤에만 유용합니다([학습](#train) 참조). `task="embed"`는 모든 헤드를 건너뛰고 백본의 최종 정규화된 CLS 토큰을 `result.embeddings`의 전체 이미지 행 하나로 반환하므로 학습이 전혀 필요하지 않습니다. 세 작업 모두 인스턴스별 탐지를 생성하지 않으므로 `result.boxes`는 항상 `None`입니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

`size`는 백본 자체가 아니라 그 위에 놓는 RF-DETR 방식 프로젝터 너비를 선택합니다. 모든 크기는 같은 DINOv2-S(small) 인코더를 공유합니다. 의미 분할은 DINOv2의 기본 정사각형 패치 그리드에서 실행되고 분류와 임베딩은 선형 프로브 학습에 사용된 더 작은 분류 해상도에서 실행됩니다.

## 학습

`task="semantic"`과 `task="classify"`는 모두 학습할 수 있습니다. `task="embed"`에는 맞출 클래스 종속 헤드가 없으므로 여기서 `train()`을 호출하면 `NotImplementedError`가 발생합니다.

<code-tabs name="train" />

여기서 주요 키워드 인수는 대부분 계열이 사용하는 `batch`와 `lr0`가 아니라 `batch_size`와 `lr`입니다. `batch`와 `lr0`도 허용되어 각각 매핑되지만 두 형식을 함께 전달하면 충돌 오류가 발생합니다. 실행 위치를 지정하는 기본 방법은 `project=`/`name=` 대신 기본값이 `"runs/train"`인 `output_dir=`이지만 `project=`/`name=`도 직접 전달할 수 있습니다. 데이터셋, 증강, 다중 GPU, 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 `task="semantic"`에서 mIoU와 픽셀 정확도, `task="classify"`에서 top-1과 top-5 정확도의 `metrics/` 키 사전을 반환합니다. `task="embed"`에는 점수를 매길 정답이 없으므로 여기서 `val()`을 호출하면 `NotImplementedError`가 발생합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

각 작업은 위에 표시된 서로 다른 형식 하위 집합을 지원합니다. 내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다. 각 형식이 받는 인수는 [내보내기](/docs/export)에 나와 있습니다.

<code-tabs name="export" />

## 라이선스

<provenance-box>

위 "Weights" 행은 적용되는 라이선스인 Apache-2.0을 표시하지만 이 계열의 파일은 LibreYOLO Hugging Face 조직에 실제로 다시 게시되지 않습니다. LibreYOLO는 자체 LibreDINOv2 체크포인트를 호스팅하지 않습니다. `LibreDINOv2(model_path=None)`이 다운로드하는 것은 수정되지 않은 Meta의 `facebook/dinov2-with-registers-small` 저장소입니다.

</provenance-box>

## 인용

<citation-block />
