---
title: Depth Anything 3
families:
  - depth_anything3
seo_title: 'Depth Anything 3: LibreYOLO의 단안 깊이 예측'
description: >-
  LibreYOLO에서 Depth Anything 3으로 단안 깊이를 추정합니다. Apache-2.0의 DA3MONO-LARGE 체크포인트를
  설치하고 예측, 검증, 내보내기합니다.
lead: >-
  Depth Anything 3는 구조적 특화 없이 하나 이상의 뷰에서 깊이와 카메라 기하를 예측하도록 학습된 plain DINOv2
  transformer입니다. LibreYOLO는 깊이 작업용 DA3MONO-LARGE 체크포인트를 이식하여 예측과 제로샷 검증을 지원하지만
  학습 경로는 제공하지 않습니다.
keywords:
  - Depth Anything 3 사용법
  - DA3
  - 단안 깊이 추정
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

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnything3l-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 깊이 맵 읽기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map    # DepthMap: 조밀한 (H, W), 높을수록 가까움
        raw = depth.data                # 텐서, 미터법 단위나 이미지 간 공통 스케일 없음
        normalized = depth.normalized() # 시각화를 위해 [0, 1]로 재조정
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnything3l-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnything3l-depth.pt format=onnx

        libreyolo export model=LibreDepthAnything3l-depth.pt format=tensorrt
        half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreDepthAnything3l-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: 0ac96180165c4891
---

## 설치

Depth Anything 3에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

`result.depth_map`은 조밀한 상대 역깊이 맵을 담습니다. 값이 높을수록 카메라에 가깝고 값에는 미터법 단위나 이미지 간 공통 스케일이 없습니다. 업스트림 체크포인트는 양의 상대 깊이를 출력합니다. LibreYOLO의 네트워크 래퍼는 이를 역수로 바꾸고 공식 하늘 처리를 재현하여 출력이 LibreYOLO의 공유 깊이 계약을 따르게 합니다. `save=True`는 해당 맵의 컬러맵 시각화를 디스크에 기록합니다. `Results.plot()`은 표면 노멀과 엣지만을 위해 정의되어 있어 이 계열을 지원하지 않습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

고정 입력 해상도를 사용하는 크기 `l` 하나입니다. 업스트림 DA3는 Small과 Base any-view 체크포인트, metric-depth 체크포인트, Nested와 Giant 체크포인트도 공개하지만 LibreYOLO는 이들을 제공하지 않습니다. 미터 깊이에는 LibreYOLO의 상대 역깊이 작업과 다른 공개 계약이 필요하고 any-view 및 Nested 체크포인트에는 LibreYOLO가 제공하지 않는 다중 이미지 카메라 API가 필요합니다. Large와 Giant any-view 체크포인트는 CC-BY-NC-4.0이기도 하며 LibreYOLO 다운로드 경로에서 참조하지 않습니다.

이 계열은 학습을 제공하지 않습니다. `LibreDepthAnything3.train()`은 조건 없이 `NotImplementedError`를 발생시킵니다. 업스트림에서 학습한 뒤 `weights/convert_depth_anything3_weights.py`로 호환되는 DA3MONO-LARGE 체크포인트를 변환합니다.

## 검증

`val()`은 공유 깊이 검증기를 실행합니다. 이미지별 최소제곱 스케일과 시프트로 각 예측을 정답에 정렬한 다음 표준 제로샷 상대 깊이 지표인 AbsRel, RMSE, 세 가지 delta 임곗값을 보고합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

이 계열의 내보내기는 ONNX, TorchScript, ExecuTorch, TensorRT, OpenVINO의 다섯 형식으로 제한됩니다. 다른 형식을 요청하면 검증되지 않은 변환을 시도하지 않고 `NotImplementedError`를 발생시킵니다. 내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 박스 대신 `depth_map`을 포함한 동일한 `Results`를 반환합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
