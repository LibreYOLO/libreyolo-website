---
title: MiDaS
families:
  - midas
seo_title: 'MiDaS: LibreYOLO의 단안 깊이 추정'
description: LibreYOLO에서 MiDaS 상대 깊이 추론을 실행합니다. s와 l 체크포인트는 배포자의 MIT 허가에 따라 LibreYOLO 미러를 사용합니다.
lead: >-
  MiDaS는 여러 데이터셋에서 스케일 및 시프트 불변 손실로 학습한 단안 상대 깊이 추정 모델입니다. 이후 계열이 재사용하는 제로샷 깊이
  전이 프로토콜을 확립한 연구 계열입니다. LibreYOLO는 깊이 작업의 예측과 제로샷 검증을 지원하지만 학습 경로는 제공하지 않습니다.
keywords:
  - MiDaS 사용법
  - 단안 깊이 추정
  - DPT 모델
  - 상대 깊이
  - 깊이 맵
  - 제로샷 깊이 추정
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 처음 사용할 때 미러 체크포인트를 다운로드합니다.
        model = LibreYOLO("LibreMiDaSl-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreMiDaSl-depth.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: 작은 변형
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # EfficientNet-Lite3 인코더로 DPT-Large의 l 크기보다 작고 빠릅니다.
        model = LibreYOLO("LibreMiDaSs-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMiDaSl-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMiDaSl-depth.pt format=onnx
        libreyolo export model=LibreMiDaSl-depth.pt format=tensorrt half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreMiDaSl-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: 64537aa3ad3a6f8f
---

## 설치

MiDaS의 timm 인코더에는 `midas` extra가 필요합니다.

```bash
pip install "libreyolo[midas]"
```

## 예측

s와 l 체크포인트는 배포자의 MIT 허가에 따라 LibreYOLO 미러에서 다운로드되어 로컬에 캐시됩니다.

<code-tabs name="predict" />

`result.depth_map`에는 조밀한 상대 역깊이 맵이 들어 있습니다. 값이 클수록 카메라에 가깝고, 값에는 미터 단위나 이미지 간 공통 스케일이 없습니다. `save=True`는 이 맵에 컬러맵을 적용한 시각화를 디스크에 저장하며, `Results.plot()`은 깊이 맵을 렌더링합니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조하십시오.

## 변형

두 변형은 같은 모델의 단순한 크기 차이가 아니라 서로 다른 인코더를 사용합니다. `s`는 EfficientNet-Lite3 인코더를 사용하는 MiDaS v2.1 Small입니다. `l`은 ViT-L/16 인코더에 MiDaS가 조밀 예측용으로 도입한 DPT 디코더를 결합한 DPT-Large입니다. 전처리도 다릅니다. `s`는 ImageNet 평균/표준편차 정규화와 상한 종횡비 크기 조정을 사용하고 `l`은 최소 종횡비 크기 조정과 평균 및 표준편차 0.5를 사용합니다. 가벼운 CNN이 필요하면 `s`, transformer 디코더의 정확도가 필요하면 `l`을 선택합니다.

이 계열은 학습을 제공하지 않습니다. `LibreMiDaS.train()`은 조건 없이 `NotImplementedError`를 발생시킵니다.

## 검증

`val()`은 공유 깊이 검증기를 실행합니다. 이미지별 최소제곱 스케일과 시프트로 각 예측을 정답에 정렬한 다음 표준 제로샷 상대 깊이 지표인 AbsRel, RMSE, 세 가지 delta 임곗값을 보고합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러옵니다. 따라서 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 박스 대신 `depth_map`을 포함한 동일한 `Results`를 반환합니다.

<code-tabs name="export" />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
