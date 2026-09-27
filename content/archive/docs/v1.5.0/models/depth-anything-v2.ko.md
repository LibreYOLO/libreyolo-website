---
title: Depth Anything V2
families:
  - depth_anything
seo_title: 'Depth Anything V2: 단안 깊이 예측 및 검증'
description: >-
  LibreYOLO에서 Depth Anything V2로 단안 깊이를 추정합니다. 설치, 예측, 검증을 지원하며 Small은
  Apache-2.0, Base와 Large는 CC-BY-NC-4.0입니다.
lead: >-
  Depth Anything V2는 DINOv2 인코더와 DPT 디코더를 결합하여 이미지 하나에서 조밀한 상대 역깊이 맵을 예측합니다.
  LibreYOLO는 깊이 작업의 예측과 제로샷 검증을 지원하지만 학습 경로는 제공하지 않습니다.
keywords:
  - Depth Anything V2 사용법
  - 단안 깊이 추정
  - DPT
  - DINOv2
  - 상대 깊이
  - 깊이 맵
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnythingV2s-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 깊이 맵 읽기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map    # DepthMap: 조밀한 (H, W), 높을수록 가까움
        raw = depth.data                # 텐서, 미터법 단위나 이미지 간 공통 스케일 없음
        normalized = depth.normalized() # 시각화를 위해 [0, 1]로 재조정
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnythingV2s-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=onnx

        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=tensorrt
        half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e1043aba1b70b65c
---

## 설치

Depth Anything V2에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

`result.depth_map`은 조밀한 상대 역깊이 맵을 담습니다. 값이 높을수록 카메라에 가깝고 값에는 미터법 단위나 이미지 간 공통 스케일이 없습니다. `save=True`는 해당 맵의 컬러맵 시각화를 디스크에 기록합니다. `Results.plot()`은 표면 노멀과 엣지만을 위해 정의되어 있어 이 계열을 지원하지 않습니다. 입력 해상도는 DPT 헤드의 기반인 DINOv2 패치 그리드 크기 14로 나누어떨어져야 합니다. LibreYOLO는 실행 전에 이를 검사하고 조건을 충족하지 않으면 예외를 발생시킵니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

ViT-S/B/L/G에 해당하는 s/b/l/g의 네 인코더 크기가 있습니다. 아래 체크포인트 표에는 s, b, l만 있으며 Giant 체크포인트는 공개되지 않았습니다. 네 가지 모두 같은 입력 해상도를 사용하므로 크기 선택은 이미지 크기가 아닌 인코더 용량을 절충합니다. 라이선스도 선택 요소입니다. Small 체크포인트는 Apache-2.0이고 Base와 Large는 CC-BY-NC-4.0입니다. 아래 라이선스를 참조합니다.

이 계열은 학습과 파인튜닝을 제공하지 않습니다. `LibreDepthAnythingV2.train()`은 조건 없이 `NotImplementedError`를 발생시킵니다. 대신 `weights/convert_depth_anything_v2_weights.py`로 호환되는 업스트림 체크포인트를 변환합니다.

## 검증

`val()`은 공유 깊이 검증기를 실행합니다. 이미지별 최소제곱 스케일과 시프트로 각 예측을 정답에 정렬한 다음 표준 제로샷 상대 깊이 지표인 AbsRel, RMSE, 세 가지 delta 임곗값을 보고합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러옵니다. 따라서 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 박스 대신 `depth_map`을 포함한 동일한 `Results`를 반환합니다. 각 형식이 받는 인수는 [내보내기](/docs/export)에 나와 있습니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
