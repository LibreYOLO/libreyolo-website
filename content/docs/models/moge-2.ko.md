---
title: MoGe-2
families:
  - moge2
seo_title: 'MoGe-2: 표면 노멀 예측, 검증 및 내보내기'
description: >-
  LibreYOLO에서 MoGe-2로 조밀한 표면 노멀을 예측합니다. 공식 ViT-S, ViT-B, ViT-L 체크포인트를 설치하고 예측,
  검증, 내보내기합니다.
lead: >-
  MoGe-2는 RGB 이미지 하나에서 조밀한 표면 노멀 필드를 예측하는 단일 순전파 단안 기하 모델입니다. LibreYOLO는 공식
  ViT-S, ViT-B, ViT-L 체크포인트를 통해 노멀 추정만 지원합니다.
keywords:
  - MoGe-2 사용법
  - MoGe 2
  - 표면 노멀 추정
  - 단안 기하
  - 노멀 맵
  - 조밀 예측
  - DINOv2
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normal = result.normal_map
        print(normal.array.shape)   # (H, W, 3) float32 단위 벡터
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMoGe2s-normal.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])   # 도
        print(metrics["metrics/median_angular_error"])
        print(metrics["metrics/within_11_25"])          # 픽셀 비율
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMoGe2s-normal.pt data=my-dataset.yaml imgsz=518
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
        model.export(format="tensorrt", imgsz=518, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMoGe2s-normal.pt format=onnx imgsz=518

        libreyolo export model=LibreMoGe2s-normal.pt format=tensorrt imgsz=518
        half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.array.shape)
source_hash: ddfacf6b7e9729f6
---

## 설치

MoGe-2에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 가중치를 자동으로 다운로드합니다. LibreYOLO는 공식 체크포인트에서 일치하는 크기를 직접 가져와 로컬에 캐시합니다.

<code-tabs name="predict" />

MoGe-2는 탐지 집합 대신 조밀 필드를 반환하므로 `result.boxes`는 비어 있고 `conf`, `iou`, `max_det`은 아무 효과가 없습니다. `result.normal_map`에 결과가 들어 있습니다. OpenCV 카메라 좌표계의 단위 벡터로 이루어진 `(H, W, 3)` 배열이며 `+x`는 오른쪽, `+y`는 아래쪽, `+z`는 장면 안쪽을 가리킵니다. 카메라를 향하는 표면은 `(0, 0, -1)`입니다. 이미지 목록을 예측하면 이미지마다 순전파를 한 번 실행합니다. 이 계열에는 스택 배치 고속 경로가 없습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

ViT-S, ViT-B, ViT-L의 세 인코더 크기가 별도 체크포인트로 제공되며 모두 같은 입력 해상도를 사용합니다. LibreYOLO 벤치마크 도구는 이 계열을 측정하지 않았으므로 크기를 비교할 수 있는 공개 정확도 수치가 없습니다. 연산 예산에 맞춰 크기를 선택합니다.

## 검증

`val()`은 페어링된 노멀 맵 데이터셋을 기준으로 각도 오차를 측정합니다. 이미지는 같은 기본 이름의 16비트 노멀 PNG 옆에 놓이며 선택적 유효성 마스크를 사용하면 패딩 픽셀과 유효하지 않은 픽셀을 계산에서 제외할 수 있습니다. 평균 및 중앙 각도 오차를 도 단위로 반환하고 11.25도, 22.5도, 30도 이내인 픽셀 비율도 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

노멀 내보내기는 고정 해상도, 배치 1 런타임 계약을 사용합니다. `dynamic`과 1이 아닌 `batch`는 거부되며 `imgsz`는 ViT 인코더의 패치 크기로 나누어떨어져야 합니다. LibreYOLO는 실행 전에 이를 검사합니다. 내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다.

<code-tabs name="export" />

## 라이선스

<provenance-box>

LibreYOLO는 이러한 체크포인트를 자체 조직으로 복사하지 않습니다. `LibreYOLO("LibreMoGe2s-normal.pt")`는 고정된 리비전의 공식 Hugging Face 저장소에서 일치하는 크기를 직접 다운로드하고 사용 전에 기록된 SHA-256 체크섬으로 파일을 검증합니다.

</provenance-box>

## 인용

<citation-block />
