---
title: BiRefNet
families:
  - birefnet
seo_title: 'BiRefNet: LibreYOLO의 배경 제거와 매팅'
description: >-
  LibreYOLO에서 BiRefNet으로 배경 제거와 이분 이미지 분할을 수행합니다. general 체크포인트를 설치하고 예측, 검증,
  내보내기합니다.
lead: >-
  피사체와 배경을 분리하는 부드러운 알파 매트를 예측하는 양방향 참조 네트워크입니다. LibreYOLO는 BiRefNet의 매트 작업에 추론과
  검증을 제공합니다.
keywords:
  - BiRefNet 사용법
  - 이미지 배경 제거
  - 이분 이미지 분할
  - 알파 매트
  - 이미지 매팅
  - 누끼 따기
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreBiRefNetl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 컷아웃
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: 소스 RGB에 매트를 알파 채널로 추가합니다.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # images/와 자동 탐지되는 매트 디렉터리(mattes/, matte/, gt/,
        # masks/, mask/ 또는 alpha/)를 포함한 디렉터리도 데이터셋 YAML
        # 대신 사용할 수 있습니다.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreBiRefNetl-matte.pt format=onnx
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreBiRefNetl-matte.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: 1af1bd7f4f905081
---

## 설치

BiRefNet에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

매트 결과에는 박스가 없습니다. `result.matte`는 `[0, 1]` 범위의 조밀한 `(H, W)` float32 배열이며 1은 완전한 전경, 0은 완전한 배경입니다. 이진 마스크와 달리 부드러운 매트는 머리카락이나 털처럼 앤티앨리어싱된 가장자리 세부 정보를 유지합니다. `result.cutout()`은 소스 이미지와 해당 알파 채널을 합성해 RGBA 배열을 만듭니다. `result.save(path)` 또는 예측 호출의 `save=True`는 이를 투명 배경 PNG로 바로 기록합니다. 모델은 고정된 기본 1024x1024 캔버스에서 실행됩니다. Swin 백본의 상대 위치 표가 이 크기에 연결되어 있어 다른 해상도에서는 예외 대신 부정확한 보간이 발생하므로 지원하지 않습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

공개된 체크포인트는 `l` 하나이며 Swin-L 티어 BiRefNet-general 모델이자 업스트림의 품질 기본값입니다. 계열 코드에는 Swin-T lite 티어인 `t`도 지원되지만 LibreYOLO 변환본은 아직 공개되지 않았습니다.

## 검증

`val()`은 페어링된 이미지/매트 폴더에서 두 지표를 보고합니다. 모두 `[0, 1]` 범위이고 해상도와 무관합니다. MAE는 정답 알파에 대한 평균 절대 오차로 낮을수록 좋습니다. S-measure(Fan 외, ICCV 2017)는 픽셀 MAE만으로는 놓치는 피사체의 형태와 구멍 보존을 반영하는 구조적 유사도로 높을수록 좋습니다. 검증은 모델 자체의 `predict`를 사용하므로 계열의 정확한 전처리를 적용합니다.

<code-tabs name="val" />

검증은 추론 전용입니다. 파인튜닝은 제공되는 기능이 아니라 문서화된 후속 작업입니다. 향후 학습기가 상속할 정확한 해상도 제약은 예측을 참조합니다.

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다. 검증된 경로는 TorchScript입니다. ONNX 변환은 실행되지만 같은 수준의 동등성 검증을 통과하지 않았습니다. 각 형식이 받는 인수와 일부 형식이 추가하는 extra는 [내보내기](/docs/export)에 나와 있습니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
