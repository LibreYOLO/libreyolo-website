---
title: YOLO-NAS
families:
  - yolonas
seo_title: 'YOLO-NAS: LibreYOLO의 예측, 학습 및 내보내기'
description: >-
  LibreYOLO에서 YOLO-NAS로 탐지와 자세 추정을 수행합니다. Deci.AI 가중치는 독점적이고 비상업적이며 LibreYOLO는
  이를 게시하지 않습니다.
lead: >-
  Deci.AI의 아키텍처 검색에서 나온 백본과 넥을 사용하고 양자화 인식 RepVGG 블록으로 구축된 합성곱 탐지기입니다. 가중치는
  Deci.AI 소유이며 비상업적 사용만 허용되고 LibreYOLO는 이를 게시하지 않습니다.
keywords:
  - YOLO-NAS 사용법
  - YOLONAS
  - Deci AI
  - SuperGradients
  - 객체 탐지
  - 자세 추정
  - 양자화 인식 탐지기
  - AutoNAC
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 디스크에 아직 없는 이름은 Deci CDN에서 가져옵니다. 다운로드 전에
        # Deci 라이선스 약관을 출력하며 파일을 받으면 약관에 동의한 것입니다.
        model = LibreYOLO("LibreYOLONASs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLONASs.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 자세 추정
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # -pose 접미사가 자세 헤드와 전용 가중치 집합을 선택합니다.
        model = LibreYOLO("LibreYOLONASs-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLONASs.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: 처음부터 학습
      language: python
      code: |
        from libreyolo import LibreYOLONAS

        # Deci 체크포인트를 전혀 사용하지 않습니다. 모델은 무작위 가중치에서
        # 시작하므로 실행 결과는 제공한 데이터에서만 파생됩니다.
        model = LibreYOLONAS(None, size="s")
        model.train(data="my-dataset.yaml", imgsz=640, batch=16)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLONASs.pt data=my-dataset.yaml
    - label: COCO 기준 평가
      language: bash
      code: |
        # 번들 COCO yaml에는 내장 다운로드 스크립트가 있으므로 데이터셋이 이미
        # 로컬에 있지 않다면 명시적 권한이 필요합니다.
        libreyolo val model=LibreYOLONASl.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLONASs.pt format=onnx imgsz=640
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreYOLONASs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 47c30d6e44024ce7
---

## 설치

YOLO-NAS에는 기본 패키지 외의 extra가 필요하지 않습니다.

```bash
pip install libreyolo
```

## 예측

디스크에 아직 없는 체크포인트 이름은 이러한 가중치를 하나도 호스팅하지 않는 LibreYOLO 조직이 아니라 Deci의 공개 CDN에서 가져옵니다. 전송을 시작하기 전에 라이브러리가 프로세스마다 한 번 Deci 라이선스 약관을 출력하며 다운로드 파일을 열기 전에 고정된 값과 SHA-256을 대조합니다. 해당 약관이 허용하는 범위는 [라이선스](#licensing)에 나와 있습니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열이 반환하는 것과 같으므로 탐지기를 바꾸려면 한 줄만 변경하면 됩니다. `conf`는 신뢰도 임곗값을, `iou`는 NMS 임곗값을 설정합니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

탐지와 자세 추정은 서로 다른 헤드 아래에서 같은 구조를 사용하며 같은 인수를 받습니다. 아래 표의 크기는 탐지용입니다. 자세 모델도 해당 크기들과 더 작은 크기 하나로 공개됩니다. 자세 헤드는 COCO 키포인트 집합을 예측합니다.

<benchmark-table task="detect" />

<va-embed />

## 학습

<code-tabs name="train" />

`epochs`, `lr0`, `amp`를 생략하면 작업별로 결정되므로 자세 실행은 탐지 실행과 다른 기본값에서 시작합니다. 기본 옵티마이저는 AdamW입니다. 클래스 수는 데이터셋 YAML에서 가져오며 첫 epoch 전에 이에 맞춰 헤드를 다시 구축합니다. 자세 헤드도 키포인트 수를 같은 방식으로 처리하므로 COCO 자세 체크포인트를 다른 크기의 스켈레톤에 파인튜닝할 수 있습니다.

파인튜닝은 Deci의 가중치에서 시작하며 여기에 Deci 라이선스가 적용됩니다. 무작위로 초기화한 모델의 학습은 Deci 체크포인트를 전혀 사용하지 않으며 위의 세 번째 스니펫이 이 경로입니다.

데이터셋, 증강, 다중 GPU, 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 학습에 사용한 형식의 데이터셋을 대상으로 측정한 정밀도, 재현율, mAP 50, mAP 50-95를 포함하는 `metrics/` 키 사전을 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다. LibreYOLO가 설치되지 않은 기본 런타임에서 그래프를 실행할 수도 있지만 이 경우 전처리와 후처리를 직접 작성해야 합니다. 각 형식은 서로 다른 extra를 설치하고 고유한 몇 가지 인수를 받습니다. 둘 다 해당 형식 페이지에 나와 있습니다.

내보내기는 같은 가중치를 다른 컨테이너에 복제하는 일입니다. Deci 체크포인트를 내보내도 가중치의 출처나 적용되는 라이선스는 바뀌지 않습니다.

<code-tabs name="export" />

## 체크포인트

나열할 체크포인트가 없습니다. Deci 라이선스는 재배포를 금지하므로 LibreYOLO 조직은 YOLO-NAS 가중치를 게시하지 않으며 다운로드는 다른 위치에서 처리됩니다. `LibreYOLONAS<size>.pt` 형식의 이름 또는 자세용 `LibreYOLONAS<size>-pose.pt`가 Deci 공개 CDN의 해당 객체에 매핑됩니다.

라이브러리가 SHA-256을 고정한 체크포인트만 이 방식으로 가져올 수 있습니다. 그 외 항목은 검증되지 않은 서드파티 pickle을 열지 않고 실패하며 직접 다운로드해 경로로 전달해야 합니다. 디스크에 이미 있는 파일은 다운로드나 체크섬 게이트 없이 해당 경로에서 불러옵니다. 로더가 인식하는 원래 이름의 Deci `.pth`도 여기에 포함됩니다.

## 라이선스

<provenance-box>

LibreYOLO는 이러한 가중치를 호스팅하거나 미러링하지 않습니다. 이 계열은 LibreYOLO Hugging Face 조직에 존재하지 않습니다. 모든 자동 다운로드는 대신 Deci의 공개 CDN으로 향하고 시작 전에 프로세스마다 한 번 Deci 약관을 출력하며 파일을 열기 전에 고정된 SHA-256과 대조합니다.

대안은 무작위로 초기화한 모델을 학습하는 것입니다. 구조는 업스트림에서 Apache-2.0이고 여기서는 MIT이므로 자체 데이터로 이렇게 학습한 모델은 어떤 Deci 체크포인트에서도 파생되지 않습니다.

</provenance-box>

## 인용

YOLO-NAS는 논문 없이 공개되었습니다. 아래 항목은 저자들이 배포 라이브러리인 SuperGradients를 대상으로 요청한 인용입니다.

<citation-block />
