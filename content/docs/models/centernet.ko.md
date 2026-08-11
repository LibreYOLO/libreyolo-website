---
title: CenterNet
families:
  - centernet
seo_title: 'CenterNet: LibreYOLO의 객체 탐지'
description: >-
  LibreYOLO에서 ResDCN-18 및 DLA-34 백본으로 CenterNet(Objects as Points)을 실행합니다. MIT로
  ONNX 예측, 검증, 내보내기를 지원하며 학습 경로는 없습니다.
lead: >-
  CenterNet은 객체를 바운딩 박스의 중심점으로 모델링하고 히트맵 피크에서 다른 모든 속성을 회귀하므로 앵커와 NMS 단계가 필요하지
  않습니다. LibreYOLO는 이를 추론 전용 탐지기로 제공합니다.
keywords:
  - CenterNet 사용법
  - Objects as Points
  - 키포인트 탐지
  - 앵커 프리 탐지기
  - ResDCN-18
  - DLA-34
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreCenterNetresdcn18.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: DLA-34
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetdla34.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCenterNetresdcn18.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")

        # ONNX 내보내기에는 opset 16 이상이 필요합니다. 변형 가능 컨볼루션
        # 업샘플링 단계가 opset 16에서 도입된 GridSample로 낮춰집니다.
        model.export(format="onnx", opset=18)
        model.export(format="tensorrt")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreCenterNetresdcn18.pt format=onnx opset=18
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreCenterNetresdcn18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 20aaef83cc95590d
---

## 설치

CenterNet에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에
포함됩니다.

```bash
pip install libreyolo
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열에서 동일하므로 다른 탐지기로 바꾸려면 한 줄만
수정하면 됩니다. `conf`와 `max_det`는 순위가 매겨진 히트맵 피크를 필터링합니다.
CenterNet의 top-k 피크 디코딩에는 바운딩 박스 IoU 억제 단계가 필요하지 않으므로
API 동등성을 위해 `iou`를 허용하지만 효과가 없습니다. 소스, 스트리밍, 결과 처리는
[예측](/docs/predict)을 참조합니다.

## 변형

백본은 두 가지입니다. `resdcn18`은 ResNet-18 트렁크와 변형 가능 컨볼루션
업샘플링을 결합하고, `dla34`는 DLA-34 트렁크와 반복적 심층 집계 업샘플링을
결합합니다. 둘 다 같은 3개 밀집 헤드(히트맵, 너비 및 높이, 오프셋)와 같은 입력
캔버스를 사용합니다.

## 검증

`val()`은 학습에 사용한 형식의 모든 데이터셋을 대상으로 측정한 정밀도, 재현율,
mAP 50, mAP 50-95를 포함하는 `metrics/` 키 딕셔너리를 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

ONNX 내보내기에는 opset 16 이상이 필요합니다. 두 백본의 변형 가능 컨볼루션
업샘플링 단계가 opset 16에서 도입된 ONNX `GridSample` 연산자로 낮춰집니다. 이전
opset을 요청하면 추적을 시작하기 전에 오류가 발생합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에서 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box>

ResDCN-18 그래프는 Microsoft의 MIT 라이선스
human-pose-estimation.pytorch에도 저작자를 표시하며, DLA-34 그래프는 Fisher Yu의
BSD-3-Clause DLA 구현에도 저작자를 표시합니다. LibreYOLO는 업스트림 프로젝트가
사용한 원본 DCNv2 확장을 포함하지 않습니다. 네이티브 실행은 torchvision의
BSD-3-Clause `deform_conv2d`를 대신 사용하며 내보내기 전용 이식 가능 구현은
LibreYOLO를 위해 별도로 작성했습니다.

</provenance-box>

## 인용

<citation-block />
