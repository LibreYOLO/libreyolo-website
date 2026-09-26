---
title: Real-ESRGAN
families:
  - realesrgan
seo_title: 'Real-ESRGAN: LibreYOLO의 이미지 초해상도'
description: >-
  LibreYOLO에서 실용적인 4x, 2x 및 빠른 4x 이미지 초해상도에 Real-ESRGAN을 사용합니다. 설치, 예측, 검증,
  내보내기를 지원합니다.
lead: >-
  bicubic 다운스케일링만이 아니라 합성 열화로 학습한 실용적인 블라인드 초해상도 업스케일러입니다. LibreYOLO는 4x, 2x, 빠른
  4x 체크포인트의 추론과 검증을 제공합니다.
keywords:
  - Real-ESRGAN 사용법
  - RRDBNet
  - SRVGGNetCompact
  - 이미지 초해상도
  - 이미지 복원
  - 블라인드 초해상도
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRealESRGANx4-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 대형 이미지용 타일
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # tile은 순전파를 겹치는 타일로 나누고 이음새를 다시 혼합합니다.
        # tile_pad는 다시 잘라내기 전에 각 타일 주변에 추가하는 여백입니다.
        # 둘 다 CLI 플래그가 아닌 Python 전용 키워드 인수입니다.
        result = model("large-photo.jpg", tile=512, tile_pad=10, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: >
        libreyolo val model=LibreRealESRGANx4-restore.pt
        data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # imgsz를 생략하면 작업 해상도가 아니라 작은 내부 패치 크기가
        # 기본값이므로 배포가 실제로 모델에 전달할 크기를 지정합니다.
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRealESRGANx4-restore.pt format=onnx
        imgsz=512
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreRealESRGANx4-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
source_hash: f0efb4f65d38e22d
---

## 설치

Real-ESRGAN에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에
포함됩니다.

```bash
pip install libreyolo
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

복원 결과에는 바운딩 박스가 없습니다. `result.restored`는 각 차원에서 입력보다
`Results.restore_scale`배 큰 캔버스의 밀집 `(H, W, 3)` uint8 RGB 이미지입니다.
`save=True`는 어노테이션 플롯 대신 해당 이미지를 직접 저장합니다. 입력은 RGB로
변환되고 알파 채널은 버립니다. 메모리보다 큰 소스는 `tile`과 `tile_pad`로 나눌 수
있으며 출력에서 타일 이음새를 다시 혼합합니다. 소스, 스트리밍, 결과 처리는
[예측](/docs/predict)을 참조합니다.

## 변형

업스케일 배율에 따라 이름을 붙인 체크포인트 3개가 있습니다. `x4`는 23개의
residual-in-residual 밀집 블록이 있는 RRDBNet(`RealESRGAN_x4plus`)이며 4x 품질
기본값입니다. `x2`는 2x의 같은 RRDBNet 아키텍처입니다. `x4t`는 비디오와 저지연
4x 사용을 위해 만든 더 작고 빠른 생성기 SRVGGNetCompact
(`realesr-general-x4v3`)입니다. 업스트림 범용 모델은 추론 시 혼합하는 별도의 노이즈
제거 강도 네트워크도 제공하지만, 이 강도 조절 기능은 기본 `x4t` 생성기를 실행하는
이 포트에 포함되지 않습니다.

## 검증

`val()`은 복원 출력과 깨끗한 대상 이미지 사이의 PSNR 및 SSIM을 측정하며, 둘 다
테두리 크롭과 크기 조정 없이 원본 캔버스의 RGB에서 계산합니다. SSIM은 sigma 1.5의
11x11 가우시안 윈도우를 사용하고 3개 색상 채널의 평균을 냅니다.

<code-tabs name="val" />

데이터셋 인수는 열화된 입력 이미지 디렉터리와 해상도가 일치하는 깨끗한 대상 이미지
디렉터리를 짝짓는 YAML입니다. 정확한 키는
[데이터셋 형식](/docs/reference/dataset-formats)을 참조합니다.

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는
`.engine` 파일이 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다.
[내보내기](/docs/export)에는 모든 형식이 허용하는 인수와 일부 형식이 추가하는 extra가
나열되어 있습니다.

<code-tabs name="export" />

## 체크포인트

이 계열에서 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
