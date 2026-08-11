---
title: RF-DETR
families:
  - rfdetr
seo_title: 'RF-DETR: MIT 기반 학습, 파인튜닝 및 내보내기'
description: >-
  LibreYOLO에서 RF-DETR로 탐지, 인스턴스 분할, 자세 추정, 회전 박스를 수행합니다. 모두 MIT 라이선스로 설치, 예측,
  학습, 검증, 내보내기합니다.
lead: >-
  조밀 그리드 대신 고정된 객체 집합을 예측하는 detection transformer이므로 추론 시 NMS가 필요하지 않습니다.
  LibreYOLO는 네 가지 작업을 지원합니다.
keywords:
  - RF-DETR 사용법
  - 실시간 detection transformer
  - DETR
  - 객체 탐지
  - 인스턴스 분할
  - 자세 추정
  - 회전 바운딩 박스
last_verified: 1.5.0
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: 'LibreRFDETRs, 512 px 동영상 객체 탐지.'
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRFDETRs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 동영상
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # 라이브러리가 받는 모든 소스: 파일, 폴더, URL, 웹캠 인덱스,
        # RTSP 스트림 또는 .streams 목록
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: 다중 GPU
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # val()은 객체가 아닌 일반 dict를 반환합니다.
        metrics = model.val(data="my-dataset.yaml", imgsz=512)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs.pt data=my-dataset.yaml imgsz=512
    - label: COCO 기준 평가
      language: bash
      code: |
        # 번들 COCO yaml에는 내장 다운로드 스크립트가 있으므로 데이터셋이 이미
        # 로컬에 있지 않다면 명시적 권한이 필요합니다.
        libreyolo val model=LibreRFDETRn.pt data=coco.yaml imgsz=384 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)

        # 모든 형식에서 받는 인수:
        #
        #   format    "onnx" | "torchscript" | "executorch" | "tensorrt"
        #             | "openvino" | "paddle" | "mnn" | "rknn" | "ncnn"
        #             | "tflite" | "coreml" | "coreai".
        #             "engine"은 tensorrt, "litert"는 tflite의 별칭입니다.
        #   imgsz     int 또는 (height, width). 기본값은 체크포인트의 기본 해상도입니다.
        #   batch     int, 기본값 1.
        #   half      bool, FP16으로 내보냅니다. 기본값 False.
        #   int8      bool, INT8로 내보냅니다. 기본값 False. `data`가 필요합니다.
        #   data      int8 보정에 사용하는 데이터셋 YAML 경로입니다.
        #   fraction  float, 사용할 보정 집합 비율입니다. 기본값 1.0.
        #   dynamic   bool, 동적 축입니다. 기본값 True.
        #   simplify  bool, ONNX 그래프 단순화를 실행합니다. 기본값 True.
        #   opset     int, ONNX opset입니다. 미지정 시 계열별로 선택합니다.
        #   device    str, 추적할 장치입니다. 기본값은 모델 장치입니다.
        #   output_path  str, 기본값은 체크포인트에서 파생한 이름입니다.
        #   verbose   bool, 기본값 False.
        #   allow_download_scripts  bool, 기본값 False. 다운로드해야 하는 데이터셋
        #             YAML의 내장 Python 실행을 허용합니다.
        #
        # 일부 형식은 RKNN 대상 플랫폼 같은 전용 인수를 추가로 받습니다.
        # 각 형식 페이지에서 이를 설명합니다.
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRFDETRs.pt format=onnx imgsz=512

        libreyolo export model=LibreRFDETRs.pt format=tensorrt imgsz=512
        half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreRFDETRs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
    - label: LibreYOLO 없이 실행
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # 그래프를 직접 실행하려면 전처리와 후처리를 직접 수행해야 합니다.

        # 연결하기 전에 시그니처를 확인합니다.

        session = ort.InferenceSession("LibreRFDETRs.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 512, 512),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 8c464aa759131694
---

## 설치

RF-DETR에는 백본용 `transformers`를 설치하는 전용 extra가 필요합니다.

```bash
pip install "libreyolo[rfdetr]"
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열이 반환하는 것과 같으므로 탐지기를 바꾸려면 한 줄만 변경하면 됩니다. `conf`와 `max_det`은 쿼리 선택을 필터링하며 조정할 NMS 단계가 없습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

네 가지 크기와 하나의 구조를 공유하는 네 가지 작업이 있습니다. 분할, 자세 추정, 회전 박스는 다른 헤드와 함께 탐지 디코더를 재사용하므로 같은 인수를 받습니다. 크기별 매개변수 수는 비슷하며 주로 입력 해상도가 다릅니다.

<benchmark-table task="detect" />

<va-embed />

## 학습

네 작업 모두 공개된 체크포인트에서 학습을 시작합니다. RF-DETR의 네이티브 학습기가 무시하는 인수에 `pretrained`가 포함되므로 여기서 `pretrained=False`를 전달해도 무작위로 초기화된 모델을 얻지 못합니다.

<code-tabs name="train" />

두 인수는 CNN 탐지기보다 더 중요합니다. transformer 탐지기는 YOLO 모델이 견디는 학습률에서 발산하므로 `lr0`를 `1e-4` 이하로 유지합니다. 변경할 이유가 없다면 `imgsz`를 체크포인트 기본 해상도로 유지합니다. 입력 크기는 백본 패치 크기와 창 개수의 곱으로 나누어떨어져야 합니다. LibreYOLO는 실행 전에 이를 검사하고 가장 가까운 유효 크기를 알려줍니다.

데이터셋, 증강, 다중 GPU, 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 학습에 사용한 형식의 데이터셋을 대상으로 측정한 정밀도, 재현율, mAP 50, mAP 50-95를 포함하는 `metrics/` 키 사전을 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다. LibreYOLO가 설치되지 않은 기본 런타임에서 그래프를 실행할 수도 있지만 이 경우 전처리와 후처리를 직접 작성해야 합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
