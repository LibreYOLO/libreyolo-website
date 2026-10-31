---
title: HRNet
families:
  - hrnet
seo_title: 'HRNet: LibreYOLO의 top-down 자세 추정'
description: >-
  LibreYOLO에서 HRNet으로 top-down COCO-17 자세를 추정합니다. MIT 라이선스의 W32 및 W48 체크포인트를
  설치하고 예측, 검증, 내보내기합니다.
lead: >-
  HRNet은 다운샘플링한 뒤 해상도를 복원하는 대신 반복적인 다중 스케일 융합을 통해 고해상도 특징 스트림을 유지하는 합성곱 네트워크입니다.
  LibreYOLO는 공식 top-down 자세 변형의 추론과 검증을 지원합니다.
keywords:
  - HRNet 사용법
  - 인체 자세 추정
  - top-down 자세
  - COCO-17 키포인트
  - 고해상도 네트워크
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 사람 소스를 지정하지 않으면 HRNet이 경량 LibreYOLO9t 탐지기와
        # 자동으로 페어링하고 해당 선택을 한 번 기록합니다.
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreHRNetw32-pose.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 사람 소스
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        # 탐지를 완전히 건너뛰고 전체 이미지를 사람 한 명으로 처리합니다.
        result = model(SAMPLE_IMAGE, cropped=True)

        # 또는 이미 실행한 탐지기의 박스를 HRNet에 전달합니다.
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        # 또는 기본 LibreYOLO9t 대신 특정 LibreYOLO 탐지기와 페어링합니다.
        result = model(SAMPLE_IMAGE, person_detector="rfdetr")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreHRNetw32-pose.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreHRNetw32-pose.pt format=onnx
    - label: 내보낸 파일 사용
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # 내보낸 그래프는 고정 캔버스 heatmap 헤드만 포함합니다. 이미 잘라내고
        # 정규화한 사람 crop 배치를 받아 원시 heatmap을 반환합니다. 사람 탐지,
        # crop 기하, heatmap 디코딩, OKS 억제는 그래프에 포함되지 않습니다.
        # LibreYOLO 밖에서 실행하려면 해당 디코딩 단계를 직접 다시 구현해야 합니다.
        session = ort.InferenceSession("LibreHRNetw32-pose.onnx")
        name = session.get_inputs()[0].name
        heatmaps = session.run(
            None, {name: np.zeros((1, 3, 256, 192), dtype=np.float32)}
        )[0]
source_hash: 5a5540fd54ee6f23
---

## 설치

HRNet에는 기본 패키지 외의 extra가 필요하지 않습니다.

```bash
pip install libreyolo
```

기본 사람 탐지기인 경량 LibreYOLO9t 체크포인트는 HRNet과 처음 페어링할 때 자동으로 다운로드됩니다.

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

HRNet은 top-down 자세 추정기이므로 자세 헤드가 실행되기 전에 사람 박스가 필요하고 모든 호출이 하나를 결정합니다. 별도 설정이 없으면 처음에 LibreYOLO9t 탐지기와 자동으로 페어링하고 해당 선택을 기록합니다. `cropped=True`는 탐지를 건너뛰고 전체 이미지를 사람 한 명으로 처리합니다. `person_boxes`는 이미 실행한 탐지기의 박스를 받고 `person_detector`는 `"auto"`, `"rfdetr"`, 모든 LibreYOLO 탐지 모델 또는 일반 callable을 받습니다. `flip_test=True`는 수평으로 뒤집은 crop에서도 모델을 실행해 두 heatmap의 평균을 구하는 HRNet 자체 테스트 시간 증강입니다. 일반 `augment=True`는 여기 정의되어 있지 않습니다. HRNet의 탐지기와 이미지별로 달라지는 사람 수는 스택 예측을 지원하지 않으므로 다중 이미지 소스는 순차적으로 실행됩니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

`w32`와 `w48` 두 크기 모두 고정 해상도의 사람 crop에서 표준 COCO-17 키포인트 집합을 예측합니다. 두 백본 중 `w48`이 더 넓습니다.

업스트림 모델 동물원은 각 크기마다 자체 사람 탐지기, flip-testing 설정, 공식 COCO 평가 프로토콜로 자세 정확도를 보고합니다. LibreYOLO의 기본 페어링은 다른 탐지기를 사용하므로 여기서 검증하면 업스트림 수치가 아니라 해당 조합을 측정합니다. 업스트림 수치와 일치시키려면 원래 평가에 사용한 것과 같은 사람 박스, 탐지기 점수, flip 설정이 필요합니다.

## 검증

`val()`은 COCO 방식 키포인트 OKS-AP를 실행하고 YOLO-pose `data.yaml` 또는 COCO 키포인트 JSON과 이미지 디렉터리를 받습니다. 기본 지표 백엔드는 faster-coco-eval이며 설치되지 않은 경우 `pycocotools`를 자동 사용합니다. `faster_coco_eval=False`는 `pycocotools` 경로를 강제합니다.

<code-tabs name="val" />

검증은 내부에서 HRNet 자체의 `predict()`를 실행하므로 모델을 만들거나 호출할 때 사용한 사람 탐지기를 그대로 사용합니다. 호출할 때마다 기본값을 다시 결정하게 두지 말고 명시적 `person_detector=`로 모델을 생성해 실행 간 소스를 고정합니다.

## 내보내기

<export-matrix />

HRNet 내보내기 계약은 ONNX, TorchScript, OpenVINO, TensorRT만 지원하며 다른 형식은 추적을 시작하기 전에 예외를 발생시킵니다. 모든 내보내기는 사람 crop을 받아 원시 heatmap을 반환하는 배치 1 FP32 고정 캔버스 heatmap 헤드만 포함합니다. 앞단의 affine crop 기하와 뒷단의 heatmap 디코딩, flip 복원, OKS 억제는 Python에 남으므로 전체 이미지 입력에서 키포인트 출력까지의 파이프라인에는 반대쪽에 LibreYOLO가 여전히 필요합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
