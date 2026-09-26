---
title: SwinIR
families:
  - swinir
seo_title: 'SwinIR: LibreYOLO에서 4배 이미지 초해상도 실행'
description: >-
  LibreYOLO에서 SwinIR로 이미지를 4배 초해상도 처리합니다. lightweight, medium, large 체크포인트를 설치하고
  예측, 검증, 내보내기합니다.
lead: >-
  이미지 복원을 위한 Swin Transformer 네트워크입니다. LibreYOLO는 4배 초해상도 체크포인트인 공식 lightweight
  생성기, 실제 환경 medium 생성기, 실제 환경 large 생성기의 추론과 검증을 제공합니다.
keywords:
  - SwinIR 사용법
  - Swin Transformer
  - 이미지 초해상도
  - 이미지 복원
  - residual Swin Transformer block
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwinIRm-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 큰 이미지용 타일 처리
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRl-restore.pt")

        # tile은 순전파를 겹치는 타일로 나누고 이음새를 다시 블렌딩합니다.
        # tile_pad는 잘라내기 전에 각 타일 주변에 추가하는 halo입니다.
        # 둘 다 CLI 플래그가 아니라 Python 전용 키워드 인수입니다.
        result = model("large-photo.jpg", tile=512, tile_pad=16, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwinIRm-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")

        # imgsz를 생략하면 작업 해상도가 아니라 작은 내부 패치 크기가 기본값이므로
        # 배포 환경에서 실제로 모델에 공급하는 크기를 전달합니다.
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwinIRm-restore.pt format=onnx imgsz=512
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreSwinIRm-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
source_hash: 87fc3d5524480eec
---

## 설치

SwinIR에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

복원 결과에는 박스가 없습니다. `result.restored`는 입력의 각 차원보다 4배 큰 캔버스에 담긴 조밀한 `(H, W, 3)` uint8 RGB 이미지입니다. `save=True`는 주석이 달린 플롯 대신 해당 이미지를 직접 기록합니다. 입력은 크기를 조정하지 않고 8의 배수로 패딩하므로 예측이 사진 자체 해상도에서 실행됩니다. 메모리보다 큰 소스는 `tile`과 `tile_pad`로 나눌 수 있으며 출력에서 타일 이음새를 다시 블렌딩합니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

세 가지 크기는 모두 4배 업스케일로 고정됩니다. `s`는 residual Swin Transformer block(RSTB) 단계 네 개와 pixel-shuffle-direct 업샘플링을 사용하는 공식 lightweight 생성기입니다. `m`과 `l`은 각각 RSTB 단계 여섯 개와 아홉 개를 사용하고, bicubic 다운스케일링만이 아니라 실제 환경의 열화를 위해 구축된 nearest-neighbor-plus-convolution 업샘플러를 사용하는 실제 환경 medium 및 large 생성기입니다.

## 검증

`val()`은 복원된 출력과 깨끗한 대상 이미지 사이의 PSNR과 SSIM을 측정합니다. 둘 다 테두리 자르기나 크기 조정 없이 원본 캔버스의 RGB에서 계산합니다. SSIM은 sigma 1.5인 11x11 가우시안 창을 사용하고 세 색상 채널에서 평균합니다.

<code-tabs name="val" />

데이터셋 인수는 열화된 입력 이미지 디렉터리와 해상도가 일치하는 깨끗한 대상 이미지 디렉터리를 페어링하는 YAML입니다. 정확한 키는 [데이터셋 형식](/docs/reference/dataset-formats)을 참조합니다.

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러옵니다. 따라서 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다. ExecuTorch와 매트릭스에서 차단으로 표시된 모든 형식은 이 계열에서 사용할 수 없습니다. ONNX, TorchScript, TensorRT, OpenVINO, TFLite는 사용할 수 있습니다. 각 형식이 받는 인수와 일부 형식이 추가하는 extra는 [내보내기](/docs/export)에 나와 있습니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
