---
title: YOLOX
families:
  - yolox
seo_title: 'YOLOX: Apache-2.0 기반 예측, 학습 및 내보내기'
description: 'LibreYOLO에서 YOLOX로 객체를 탐지합니다. Apache-2.0 기반으로 설치, 예측, 학습, 검증, 내보내기를 수행합니다.'
lead: >-
  YOLOX는 분리된 분류-회귀 헤드를 갖춘 앵커 프리 단일 단계 탐지기로 SimOTA 레이블 할당을 사용해 학습합니다. LibreYOLO는
  객체 탐지를 지원합니다.
keywords:
  - YOLOX 사용법
  - 객체 탐지
  - 앵커 프리 탐지
  - 분리 헤드
  - SimOTA
  - 실시간 객체 탐지
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLOXs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLOXs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLOXs.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLOXs.pt data=my-dataset.yaml
    - label: COCO 기준 평가
      language: bash
      code: |
        # 번들 COCO yaml에는 내장 다운로드 스크립트가 있으므로 데이터셋이 이미
        # 로컬에 있지 않다면 명시적 권한이 필요합니다.
        libreyolo val model=LibreYOLOXn.pt data=coco.yaml imgsz=416 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLOXs.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLOXs.pt format=tensorrt imgsz=640
        half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreYOLOXs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: f5ab735a29f85a95
---

## 설치

YOLOX에는 기본 패키지 외의 extra가 필요하지 않습니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열이 반환하는 것과 같으므로 탐지기를 바꾸려면 한 줄만 변경하면 됩니다. `conf`는 신뢰도 임곗값을, `iou`는 분리된 세 예측 스케일에 적용되는 NMS 임곗값을 설정합니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

여섯 가지 크기는 같은 CSP 백본과 PAFPN 넥을 공유합니다. 가장 작은 `n`과 `t`는 다른 네 모델보다 작은 고정 입력 해상도에서 실행됩니다. 아래 벤치마크 표에 각 모델의 정확한 수치가 나와 있습니다.

<benchmark-table task="detect" />

<va-embed />

## 학습

<code-tabs name="train" />

기본 설정에서는 SGD 모멘텀 0.9, 5 epoch 워밍업, `lr0=0.01`로 300 epoch를 실행하며 마지막 15 epoch에는 mosaic와 mixup 증강을 끕니다. `train()`은 `pretrained` 인수도 받지만 메서드 내부에서 값을 읽지 않습니다. 학습은 항상 모델을 생성할 때 불러온 가중치에서 이어지므로 `pretrained=False`도 네트워크를 다시 초기화하지 않습니다.

`imgsz`의 기본값은 불러온 체크포인트의 기본 해상도가 아니라 기본 학습 설정의 고정값입니다. 이는 특히 `n`과 `t` 체크포인트에 영향을 줍니다. 둘 중 하나를 `imgsz`를 명시하지 않고 이어 학습하면 공개 당시의 작은 크기가 아니라 더 큰 기본 크기로 전환됩니다.

데이터셋, 증강, 다중 GPU, 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 학습에 사용한 형식의 데이터셋을 대상으로 측정한 정밀도, 재현율, mAP 50, mAP 50-95를 포함하는 `metrics/` 키 사전을 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러옵니다. 따라서 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다. LibreYOLO가 설치되지 않은 기본 런타임에서 그래프를 실행할 수도 있지만 이 경우 전처리와 후처리를 직접 작성해야 합니다. CoreML 내보내기는 `nms=True`로 NMS를 그래프에 포함할 수 있습니다. 현재 이 플래그를 허용하는 계열은 YOLOX와 YOLOv9뿐입니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
