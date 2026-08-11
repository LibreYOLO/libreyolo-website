---
title: ZipDepth
families:
  - zipdepth
seo_title: 'ZipDepth: LibreYOLO의 경량 단안 깊이 추정'
description: >-
  LibreYOLO에서 ZipDepth로 경량 단안 깊이를 추정합니다. MIT 라이선스의 체크포인트 두 개를 설치하고 예측, 검증,
  내보내기합니다.
lead: >-
  ZipDepth는 Depth Anything V2 Large에서 증류한 작고 재매개변수화 가능한 CNN으로 조밀한 상대 역깊이 맵을
  예측합니다. LibreYOLO는 깊이 작업의 예측과 제로샷 검증을 지원하지만 학습 경로는 제공하지 않습니다.
keywords:
  - ZipDepth 사용법
  - 경량 단안 깊이 추정
  - 엣지 깊이 모델
  - 상대 깊이
  - 깊이 맵
  - 재매개변수화 CNN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreZipDepthb-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: NPU/엣지 체크포인트
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 같은 인코더와 gather/unfold를 지원하지 않는 컴파일러용 unfold 없는
        # 업샘플링 헤드를 사용합니다. 출력은 b 체크포인트와 시각적으로 동등합니다.
        model = LibreYOLO("LibreZipDepthbnpu-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreZipDepthb-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        model.export(format="onnx")
        model.export(format="ncnn")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreZipDepthb-depth.pt format=onnx
        libreyolo export model=LibreZipDepthbnpu-depth.pt format=ncnn
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreZipDepthb-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: 891eaa1a42795a4c
---

## 설치

ZipDepth에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

`result.depth_map`은 조밀한 상대 역깊이 맵을 담습니다. 값이 높을수록 카메라에 가깝고 값에는 미터법 단위나 이미지 간 공통 스케일이 없습니다. `save=True`는 해당 맵의 컬러맵 시각화를 디스크에 기록합니다. `Results.plot()`은 표면 노멀과 엣지만을 위해 정의되어 있어 이 계열을 지원하지 않습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

두 체크포인트는 인코더 용량이 같고 학습된 업샘플링 헤드만 다릅니다. `b`는 볼록 업샘플링을 사용하며 GPU나 CPU에서 실행됩니다. `bnpu`는 gather/unfold를 지원하지 않는 NPU 및 엣지 컴파일러를 위해 unfold 없는 디코더를 사용하며 출력은 `b`와 시각적으로 동등하다고 문서화되어 있습니다. 내보내기 대상이 제약된 런타임이면 `bnpu`, 그 외에는 `b`를 선택합니다.

두 체크포인트 모두 Depth Anything V2 Large 의사 레이블에서 증류되었습니다. 따라서 이 계열은 더 큰 Depth Anything V2 인코더와 함께 LibreYOLO 깊이 작업의 작고 엣지 지향적인 티어를 이룹니다.

이 계열은 학습을 제공하지 않습니다. `LibreZipDepth.train()`은 조건 없이 `NotImplementedError`를 발생시킵니다. 업스트림 레시피는 대규모 이미지 집합에서 의사 레이블을 증류하므로 LibreYOLO 학습 실행으로 재현할 수 없습니다. [fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth)에서 업스트림 방식으로 학습한 뒤 `weights/convert_zipdepth_weights.py`로 결과를 변환합니다.

## 검증

`val()`은 공유 깊이 검증기를 실행합니다. 이미지별 최소제곱 스케일과 시프트로 각 예측을 정답에 정렬한 다음 표준 제로샷 상대 깊이 지표인 AbsRel, RMSE, 세 가지 delta 임곗값을 보고합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

내보내기는 고정 해상도 조밀 계약을 따릅니다. 소스 이미지를 내보낸 캔버스에 맞춰 늘여 크기를 조정하고 반환된 깊이 맵은 이후 원본 캔버스 크기로 되돌립니다. 내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는 `.ncnn` 파일은 체크포인트처럼 동작하며 박스 대신 `depth_map`을 포함한 동일한 `Results`를 반환합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
