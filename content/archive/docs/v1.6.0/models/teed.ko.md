---
title: TEED
families:
  - teed
seo_title: 'TEED: 체크포인트를 직접 준비하는 엣지 검출'
description: 'LibreYOLO에서 밀집 엣지 확률 예측에 TEED를 사용합니다. 라이선스를 보유한 체크포인트를 변환한 다음 예측, 검증, 내보내기합니다.'
lead: >-
  TEED(Tiny and Efficient Edge Detector)는 하나의 RGB 이미지에서 밀집 엣지 확률 맵을 예측하는 소형 컨볼루션
  네트워크입니다. LibreYOLO는 엣지 검출 전용 아키텍처를 래핑하며 라이브러리에 체크포인트는 포함되지 않습니다.
keywords:
  - TEED 사용법
  - Tiny and Efficient Edge Detector
  - 엣지 검출
  - BIPED
  - 밀집 예측
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)        # [0, 1] 범위의 (H, W) float32
        print(edges.binary(0.5).sum())  # 임계값을 적용한 엣지 픽셀 수
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=weights/LibreTEEDt-edge.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])   # 최적 데이터셋 스케일 F-measure
        print(metrics["metrics/OIS"])   # 최적 이미지 스케일 F-measure
    - label: CLI
      language: bash
      code: >
        libreyolo val model=weights/LibreTEEDt-edge.pt data=my-dataset.yaml
        imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=weights/LibreTEEDt-edge.pt format=onnx imgsz=352

        libreyolo export model=weights/LibreTEEDt-edge.pt format=tensorrt
        imgsz=352 half=True
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: c7203b254e460258
---

## 설치

TEED에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에
포함됩니다.

```bash
pip install libreyolo
```

## 예측

LibreYOLO는 TEED 체크포인트를 제공하지 않습니다. 공식 공개 가중치는 BIPED로
학습했으며 공개된 데이터셋 약관이 비상업적 용도로 제한하므로 LibreYOLO는 이를
미러링하지 않습니다. 라이선스를 보유한 체크포인트를
`weights/convert_teed_weights.py`로 변환합니다. 이 스크립트는 LibreYOLO가 직접
불러올 수 있는 파일을 쓰기 전에 런타임 아키텍처를 기준으로 텐서 키를 검사합니다.

```bash
python weights/convert_teed_weights.py upstream.pth weights/LibreTEEDt-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges`에는 `[0, 1]` 범위의 `(H, W)` float32 배열인 결과가 들어 있으며,
`.binary(threshold)`는 불리언 엣지 마스크를 반환합니다. 바운딩 박스가 없으므로
`conf`, `iou`, `max_det`는 효과가 없습니다. 소스, 스트리밍, 결과 처리는
[예측](/docs/predict)을 참조합니다.

## 변형

LibreYOLO의 TEED 크기는 하나입니다. LibreYOLO 벤치마크 하네스는 이 계열을
측정하지 않았으므로 비교할 공개 수치는 없습니다.

## 검증

`val()`은 짝을 이룬 엣지 데이터셋을 기준으로 BSDS 방식 ODS 및 OIS F-measure를
보고합니다. 이미지는 파일 이름이 같은 엣지 맵과 나란히 있으며 선택적 유효성
마스크를 사용하여 패딩된 픽셀을 계산에서 제외할 수 있습니다. `imgsz`는 네트워크
다운샘플 스트라이드로 나누어떨어져야 하며, 그렇지 않으면 LibreYOLO가 명확한 오류를
발생시킵니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

엣지 내보내기는 고정 해상도, 배치 1 런타임 계약을 사용합니다. `dynamic`과 1이 아닌
`batch`는 거부되며 내보낸 그래프는 하나의 결합 확률 맵을 출력합니다. 내보낸
아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 파일이
체크포인트처럼 동작하며 동일한 `Results`를 반환합니다.

<code-tabs name="export" />

## 라이선스

<provenance-box>

LibreYOLO는 TEED 체크포인트를 게시하지 않습니다. LibreYOLO 조직 아래에는 아무것도
미러링되지 않습니다. 대신 `weights/convert_teed_weights.py`로 라이선스를 보유한
체크포인트를 변환합니다.

</provenance-box>

## 인용

<citation-block />

