---
title: RTMDet
families:
  - rtmdet
seo_title: 'LibreYOLO의 RTMDet: 예측, 학습 및 내보내기'
description: >-
  LibreYOLO에서 RTMDet으로 객체 탐지를, RTMDet-Ins로 인스턴스 분할을 수행합니다. Apache-2.0 기반으로 설치,
  예측, 학습, 검증, 내보내기를 수행합니다.
lead: >-
  RTMDet은 앵커 없이 그리드 위치마다 하나의 점 기반 사전값에서 예측하는 단일 단계 탐지기입니다. 헤드 합성곱은 특징 레벨 전반에서
  공유됩니다. LibreYOLO는 객체 탐지와 RTMDet-Ins 인스턴스 분할을 지원합니다.
keywords:
  - RTMDet 사용법
  - 객체 탐지
  - 인스턴스 분할
  - RTMDet-Ins
  - 앵커 프리 탐지
  - mmdetection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRTMDets.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 인스턴스 분할
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 파일명의 -seg 접미사가 RTMDet-Ins 마스크 헤드를 선택하므로
        # 여기서는 task 인수가 필요하지 않습니다.
        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTMDets.pt data=my-dataset.yaml
    - label: 인스턴스 분할
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # 마스크
        print(metrics["metrics/mAP50-95(B)"])   # 박스
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, imgsz=640, batch=16, lr0=0.004,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreRTMDets.pt data=my-dataset.yaml imgsz=640
        epochs=300 batch=16 lr0=0.004
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRTMDets.pt format=onnx imgsz=640

        libreyolo export model=LibreRTMDets.pt format=tensorrt imgsz=640
        half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreRTMDets.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 2f5033bdc1c3c931
---

## 설치

RTMDet에는 기본 패키지 외의 extra가 필요하지 않습니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열이 반환하는 것과 같으므로 탐지기를 바꾸려면 한 줄만 변경하면 됩니다. `-seg` 파일명은 스스로 RTMDet-Ins 작업으로 해석되며 `result.masks`에 박스와 함께 인스턴스 마스크가 담깁니다. `conf`는 신뢰도 임곗값을, `iou`는 NMS 임곗값을 설정합니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

`t`부터 `x`까지 다섯 가지 크기가 공통 입력 해상도에서 동일한 구조를 공유합니다. 이 페이지에는 계열 벤치마크 표가 없습니다. 아래 표에서 체크포인트 파일 크기로 모델 크기를 비교합니다.

## 학습

<code-tabs name="train" />

탐지는 `train()`으로 학습합니다. QualityFocalLoss, GIoU, DynamicSoftLabelAssigner 구성 요소는 업스트림 mmdetection에서 이식했습니다. 순전파와 ONNX 내보내기는 비트 단위로 동등하며 후처리는 val2017 하위 집합에서 0.001 mAP 이내로 mmdet 출력과 일치합니다.

`train()` 자체의 docstring에 따라 아직 확인하지 않은 항목은 소규모 데이터셋 파인튜닝 수렴, 처음부터 학습했을 때 논문과의 동등성, 다중 GPU 동작, 캐시된 Mosaic 및 MixUp 처리량, 엄격한 업스트림 2단계 파이프라인 전환, norm 및 bias 매개변수의 감쇠를 0으로 만드는 매개변수별 weight decay 재정의입니다.

RTMDet-Ins에는 학습 경로가 없습니다. `-seg` 체크포인트나 `task="segment"`로 `train()`을 호출하면 `NotImplementedError`가 발생합니다. 인스턴스 분할은 추론과 검증만 지원합니다.

`train()`은 `pretrained` 인수도 받지만 메서드 내부에서 값을 읽지 않습니다. 학습은 항상 모델 생성 시 불러온 가중치에서 이어지므로 `pretrained=False`도 네트워크를 다시 초기화하지 않습니다.

그 외 기본 설정에서는 AdamW, `lr0=0.004`, `weight_decay=0.05`, cosine schedule의 1 epoch 워밍업으로 300 epoch를 실행하며 마지막 20 epoch에는 Mosaic와 MixUp을 끕니다.

데이터셋, 증강, 다중 GPU, 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 학습에 사용한 형식의 데이터셋을 대상으로 측정한 정밀도, 재현율, mAP 50, mAP 50-95를 포함하는 `metrics/` 키 사전을 반환합니다.

<code-tabs name="val" />

`-seg` 체크포인트에서는 일반 `metrics/mAP50-95` 키에 마스크 점수가 들어갑니다. 같은 실행에서 `(B)` 아래에 박스, `(M)` 아래에 마스크를 보고하므로 한 번의 패스로 둘 다 얻을 수 있습니다.

## 내보내기

<export-matrix />

탐지는 대부분 형식으로 내보낼 수 있지만 인스턴스 분할은 현재 어떤 형식으로도 내보낼 수 없습니다. 위 매트릭스가 이 구분을 반영합니다. 내보낸 탐지 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다. LibreYOLO가 설치되지 않은 기본 런타임에서 그래프를 실행할 수도 있지만 이 경우 전처리와 후처리를 직접 작성해야 합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
