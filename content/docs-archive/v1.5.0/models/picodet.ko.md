---
title: PicoDet
families:
  - picodet
seo_title: 'LibreYOLO의 PicoDet: 예측, 학습 및 내보내기'
description: >-
  LibreYOLO에서 PicoDet으로 모바일 객체 탐지를 수행합니다. Apache-2.0 기반으로 설치, 예측, 학습, 검증, 내보내기를
  수행합니다.
lead: >-
  PicoDet은 모바일 및 엣지 CPU용으로 구축된 단일 단계 탐지기입니다. ESNet 백본, CSP-PAN 넥, 공유 Generalized
  Focal Loss 헤드를 사용합니다. LibreYOLO는 객체 탐지를 지원합니다.
keywords:
  - PicoDet 사용법
  - PP-PicoDet
  - 객체 탐지
  - 모바일 객체 탐지
  - 엣지 탐지 모델
  - ESNet
  - Generalized Focal Loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePICODETs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePICODETs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePICODETs.pt data=my-dataset.yaml
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, batch=16, lr0=0.01,
        )
    - label: CLI
      language: bash
      code: >
        # imgsz는 설정할 가치가 있습니다. CLI 기본값은 640이지만 s

        # 체크포인트의 기본 해상도는 320입니다.

        libreyolo train model=LibrePICODETs.pt data=my-dataset.yaml imgsz=320
        epochs=300 batch=16 lr0=0.01
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.export(format="onnx", imgsz=320)
        model.export(format="tensorrt", imgsz=320, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibrePICODETs.pt format=onnx imgsz=320

        libreyolo export model=LibrePICODETs.pt format=tensorrt imgsz=320
        half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibrePICODETs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 947aa47214abc4c0
---

## 설치

PicoDet에는 기본 패키지 외의 extra가 필요하지 않습니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열이 반환하는 것과 같으므로 탐지기를 바꾸려면 한 줄만 변경하면 됩니다. `conf`는 신뢰도 임곗값을, `iou`는 NMS 임곗값을 설정합니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

각각 고유한 고정 입력 해상도를 사용하는 세 가지 크기가 있습니다. `s`가 가장 작고 `l`이 가장 큽니다. 크기에 따라 해상도도 증가하므로 더 큰 체크포인트는 매개변수가 많을 뿐 아니라 이미지별 실행 비용도 더 큽니다.

<benchmark-table task="detect" />

<va-embed />

## 학습

<code-tabs name="train" />

손실 구성 요소와 할당기는 업스트림 레시피를 따릅니다. VFL, DFL, GIoU, SimOTA에 분류 품질 가중치와 동적 IoU VFL 대상을 사용합니다. 같은 체크포인트의 추론은 업스트림과 비트 단위로 동등합니다.

`train()` 자체의 docstring에 따라 아직 확인하지 않은 항목은 전체 데이터셋 수렴, 다중 GPU 동작, 수평 뒤집기 외의 증강입니다. 기본 320 해상도의 `s` 체크포인트도 라이브러리가 소규모 파인튜닝을 테스트하는 30이미지, 2클래스 fixture에서 LibreYOLO의 정확도 하한을 안정적으로 넘지 못했습니다. 해당 크기는 전체 COCO 규모에 더 적합합니다.

`train()`은 `pretrained` 인수도 받지만 메서드 내부에서 값을 읽지 않습니다. 학습은 항상 모델 생성 시 불러온 가중치에서 이어지므로 `pretrained=False`도 네트워크를 다시 초기화하지 않습니다. Python에서 `imgsz`를 설정하지 않으면 불러온 체크포인트의 기본 해상도를 사용하며 `s`는 320, `m`은 416, `l`은 640입니다. CLI는 항상 `imgsz`를 보내고 기본값은 640이므로 체크포인트에 맞게 설정합니다.

그 외 기본 설정에서는 SGD, `lr0=0.01`, 모멘텀 0.9, weight decay 4e-5, cosine schedule의 1 epoch 워밍업으로 300 epoch를 실행합니다. 수평 뒤집기만 증강으로 적용합니다.

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

<provenance-box>

LibreYOLO 이식은 PaddleDetection의 원래 PP-PicoDet을 PyTorch로 다시 구현한 Bo396543018/Picodet_Pytorch를 따릅니다. mmcv를 제거하고 모든 활성화를 정확히 일치시켜 Bo의 파이프라인으로 변환한 PaddlePaddle 체크포인트가 수치 드리프트 없이 불러와집니다. 두 소스 모두 논문 저자와 같은 Apache-2.0 약관을 적용합니다.

</provenance-box>

## 인용

<citation-block />
