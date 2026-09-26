---
title: FeyNobg
families:
  - feynobg
seo_title: 'FeyNobg: LibreYOLO의 배경 제거'
description: >-
  LibreYOLO에서 Feyn Inc.가 BiRefNet을 심화한 변형인 FeyNobg를 사용해 배경 제거와 알파 매팅을 수행합니다. 설치,
  예측, 검증 방법을 설명합니다.
lead: >-
  Feyn Inc.가 BiRefNet 아키텍처를 심화하고 다시 학습한 배경 제거 모델입니다. LibreYOLO는 FeyNobg의 매트 작업을
  위한 추론과 검증을 제공합니다.
keywords:
  - FeyNobg 사용법
  - 배경 제거
  - 이분 이미지 분할
  - 알파 매트
  - 이미지 매팅
  - 누끼 따기
  - nobg
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFeyNobgl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 오려내기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: 소스 RGB와 알파 채널로 사용한 매트입니다.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFeyNobgl-matte.pt")

        # images/와 자동 탐지되는 매트 디렉터리(mattes/, matte/, gt/,
        # masks/, mask/ 또는 alpha/)가 있는 디렉터리도 데이터셋 YAML
        # 대신 사용할 수 있습니다.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
source_hash: 45de3b578d7ebbf2
---

## 설치

FeyNobg에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에
포함됩니다.

```bash
pip install libreyolo
```

## 예측

체크포인트는 다른 모든 계열과 마찬가지로 처음 사용할 때 Hugging Face의 LibreYOLO
조직에서 내려받아 로컬에 캐시되지만, 아직 이 페이지의 체크포인트 표에는 나열되지
않았습니다.

<code-tabs name="predict" />

매트 결과에는 바운딩 박스가 없습니다. `result.matte`는 `[0, 1]` 범위의 밀집
`(H, W)` float32 배열이며 1은 완전한 전경, 0은 완전한 배경입니다. 이진 마스크와
달리 소프트 매트는 머리카락과 털 같은 안티앨리어싱된 엣지 세부 정보를 유지합니다.
`result.cutout()`은 소스 이미지와 해당 알파 채널을 RGBA 배열로 합성하며,
`result.save(path)` 또는 예측 호출의 `save=True`는 투명 배경 PNG로 바로
저장합니다. 모델은 고정 네이티브 1024x1024 캔버스에서 실행합니다. Swin 백본의
상대 위치 테이블이 이 크기에 묶여 있고 크기가 일치하지 않으면 오류 대신 잘못된
보간이 일어나므로 다른 해상도는 지원하지 않습니다. 소스, 스트리밍, 결과 처리는
[예측](/docs/predict)을 참조합니다.

## 변형

공개된 크기는 Swin-L 계층 백본을 사용하는 `l` 하나입니다. FeyNobg는 BiRefNet의
아키텍처에서 세 번째 Swin 단계를 18개 블록에서 24개 블록으로 심화한 뒤 다시
학습합니다. 따라서 LibreYOLO 포트는 BiRefNet의 순전파 경로, 전처리, 단일 로짓
출력 계약을 재사용하며 예측, 검증, 체크포인트 처리는 `birefnet` 계열과 동일하게
동작합니다.

## 검증

`val()`은 짝을 이룬 이미지 및 매트 폴더에서 해상도와 무관한 `[0, 1]` 범위의
메트릭 두 가지를 보고합니다. MAE는 정답 알파 대비 평균 절대 오차로 낮을수록 좋고,
S-measure(Fan et al., ICCV 2017)는 픽셀 MAE만으로는 놓치는 피사체 형태와 구멍
보존을 반영하는 구조적 유사도로 높을수록 좋습니다. 검증은 모델 자체 `predict`를
구동하므로 계열의 정확한 전처리를 사용합니다.

<code-tabs name="val" />

검증은 추론 전용입니다. 업스트림 `nobg` 라이브러리는 Apache-2.0 학습 코드를
제공합니다. 현재 파인튜닝하려면 이 계열에서 `train()`을 호출하는 대신 업스트림에서
학습하고 LibreYOLO 자체 변환 스크립트로 결과를 변환해야 합니다. `train()`은 일부
트레이너를 실행하지 않고 오류를 발생시킵니다.

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
