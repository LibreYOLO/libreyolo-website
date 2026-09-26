---
title: CLIP
families:
  - clip
seo_title: 'LibreYOLO의 CLIP: 제로샷 분류와 임베딩'
description: >-
  LibreYOLO에서 CLIP으로 제로샷 이미지 분류와 이미지/텍스트 임베딩을 수행합니다. 학습은 없으며 set_classes()가 런타임
  레이블 집합을 정의합니다.
lead: >-
  CLIP은 고정된 레이블 집합 대신 이미지와 텍스트 프롬프트의 점수를 비교하는 이중 타워 모델입니다. LibreYOLO는 학습 단계 없이
  제로샷 분류와 이미지/텍스트 임베딩을 지원합니다.
keywords:
  - CLIP 사용법
  - OpenCLIP
  - 제로샷 이미지 분류
  - 이미지 임베딩
  - 텍스트 임베딩
  - 오픈 보캐뷸러리
  - LAION-2B
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # set_classes()를 호출하지 않으면 CLI predict는 모델이 기본으로 불러오는

        # ImageNet 클래스 이름 1,000개를 사용합니다.

        libreyolo predict model=LibreCLIPb32-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 이미지와 텍스트 임베딩
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        image_embed = model(SAMPLE_IMAGE).embeddings.data
        text_embed = model.embed_text("a photo of a forklift")

        # 둘 다 L2 정규화되므로 일반 내적이 코사인 유사도입니다.
        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        # data는 train/ 분할이 있는 ImageFolder 루트입니다. 폴더 이름이
        # 이번 실행의 제로샷 클래스 프롬프트가 됩니다.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCLIPb32-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # 현재 set_classes() 레이블과 입력 해상도가 그래프에 고정됩니다.
        # 둘 중 하나라도 바꾼 뒤에는 다시 내보냅니다.
    - label: CLI
      language: bash
      code: |
        # 여기서는 set_classes()를 호출하지 않으므로 모델이 불러온 기본
        # ImageNet 클래스 1,000개가 고정됩니다.
        libreyolo export model=LibreCLIPb32-cls.pt format=onnx
    - label: 임베딩 내보내기
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed"는 이미지 타워만 추적하므로 클래스가 필요하지 않습니다.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: ac7cfd75ad6c0fa7
---

## 설치

CLIP에는 전용 extra가 필요합니다. 이 extra는 벤더링된 BPE 토크나이저가 정확한 토큰 ID를 재현하는 데 사용하는 패키지를 설치합니다.

```bash
pip install "libreyolo[clip]"
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

`set_classes()`는 이 모델을 오픈 보캐뷸러리 분류기로 만드는 핵심 기능입니다. 각 레이블을 모든 프롬프트 템플릿으로 렌더링하고 결과를 인코딩해 평균한 뒤 생성된 `[K, D]` 행렬을 분류기 헤드로 캐시하므로 이미지마다 다시 계산하지 않습니다. 언제든 다시 호출해 클래스를 바꿀 수 있습니다. 호출하지 않으면 LibreCLIP은 ImageNet-1k 클래스 이름 1,000개가 이미 설정된 상태로 불러옵니다.

`task="embed"`를 사용하면 예측이 클래스 확률 대신 입력마다 L2 정규화된 이미지 벡터 하나를 반환합니다. `embed_text()`는 같은 벡터 공간에서 정규화된 텍스트 행을 반환하므로 둘 사이의 일반 내적이 코사인 유사도입니다. 두 작업 모두 NMS 단계가 없으므로 `iou`는 영향을 주지 않습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 검증

`val()`은 ImageFolder `train/` 분할 아래의 클래스 폴더 이름을 읽고 이를 사용해 `set_classes()`를 호출한 다음 제로샷 top-1 및 top-5 정확도를 측정합니다. 가중치 업데이트가 전혀 없으므로 정확도는 클래스 이름이 프롬프트로 어떻게 해석되는지에 따라 달라집니다. 검증은 `task="classify"`만 지원하며 `task="embed"`에는 데이터셋 검증기가 없습니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

내보내기는 모델의 현재 상태를 고정 그래프에 넣습니다. `task="classify"`의 경우 `set_classes()`가 마지막으로 설정한 레이블과 내보낼 때의 해상도가 최종 선형 계층에 고정됩니다. 따라서 내보낸 ONNX 또는 TensorRT 그래프는 텍스트 타워와 토크나이저가 없는 일반 `[B, K]` 이미지 분류기이며 클래스나 크기를 바꾸면 다시 내보내야 합니다. `task="embed"` 내보내기는 이미지 타워만 추적합니다. 둘 다 ONNX opset 14 이상이 필요하며 내보내기가 기본으로 이를 설정합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다. 두 체크포인트는 모두 COCO 학습 결과가 아니라 OpenCLIP의 LAION-2B 학습 체크포인트(`ViT-B-32`, `ViT-B-16`)를 변환한 것입니다.

<checkpoint-table />

LAION-2B 학습 데이터에는 CSAM 콘텐츠가 포함되었다는 기록이 있습니다(Stanford Internet Observatory, 2023년 12월). LAION은 이후 정제하여 다시 공개한 Re-LAION을 출시했습니다. 이러한 가중치를 다시 호스팅한다면 가능한 경우 Re-LAION 파생 체크포인트를 우선 사용합니다.

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
