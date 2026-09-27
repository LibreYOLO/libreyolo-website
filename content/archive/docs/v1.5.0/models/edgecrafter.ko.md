---
title: EdgeCrafter
families:
  - ec
seo_title: 'EdgeCrafter: LibreYOLO의 탐지, 자세 추정 및 분할'
description: >-
  LibreYOLO에서 EdgeCrafter로 탐지, 자세 추정, 인스턴스 분할을 수행합니다. MIT 라이선스 코드로 설치, 예측, 검증,
  내보내기합니다.
lead: >-
  엣지 하드웨어의 조밀 예측을 위한 소형 vision transformer로 업스트림에서는 ECDet, ECPose, ECSeg의 세 형제
  모델로 공개됩니다. LibreYOLO는 셋을 하나의 제품군으로 불러오며 체크포인트가 작업을 나타냅니다.
keywords:
  - EdgeCrafter 사용법
  - ECDet
  - ECPose
  - ECSeg
  - 소형 vision transformer
  - 객체 탐지
  - 자세 추정
  - 인스턴스 분할
  - 엣지 추론
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreECs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 자세 추정
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 파일명의 -pose 접미사가 키포인트 헤드를 선택하므로 여기서는
        # task 인수가 필요하지 않습니다.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.conf)
    - label: 인스턴스 분할
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=5e-4,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreECs.pt data=my-dataset.yaml epochs=50
        imgsz=640 batch=8 lr0=5e-4
    - label: 자세 추정
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yaml에 kpt_shape가 선언된 단일 클래스 키포인트 데이터셋과
        # 체크포인트의 기본 크기로 설정한 imgsz가 필요합니다.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="my-pose-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: 인스턴스 분할
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 폴리곤 레이블과 체크포인트의 기본 크기로 설정한 imgsz가 필요합니다.
        model = LibreYOLO("LibreECs-seg.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            lora=True,
        )
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs.pt data=my-dataset.yaml
    - label: 자세 추정
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        metrics = model.val(data="my-pose-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: 인스턴스 분할
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # 마스크
        print(metrics["metrics/mAP50-95(B)"])   # 박스
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-seg.pt format=onnx imgsz=640
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreECs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 39c6975fc16b3ff1
---

## 설치

EdgeCrafter에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

단, `lora=True`를 사용하는 어댑터 미세 조정에는 `lora` extra가 필요합니다.

```bash
pip install "libreyolo[lora]"
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

작업은 파일명에서 가져오므로 `-pose` 또는 `-seg` 체크포인트가 자체 헤드를 선택하며 task 인수를 받지 않습니다. 셋 모두 각 제품군이 반환하는 `Results` 객체를 반환하고 자세 추정에는 `result.keypoints`, 분할에는 `result.masks`가 추가됩니다. 자세 추정은 COCO 키포인트 17개를 가진 사람 클래스 하나를 다루며 모델을 구축할 때 개수가 고정됩니다. 박스 헤드가 없으므로 각 자세 박스는 자체 키포인트의 경계 범위이고 세 번째 키포인트 채널은 점별 점수가 아닌 상수입니다.

`conf`와 `max_det`은 쿼리 선택을 필터링합니다. 세 헤드 모두 NMS 단계 없이 쿼리 집합을 디코딩하므로 `iou`는 API 일관성을 위해 허용되지만 효과가 없습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

크기는 네 가지입니다. 모두 같은 입력 해상도에서 실행되므로 아래 표는 매개변수 수와 정확도로 구분합니다.

<benchmark-table task="detect" />

<va-embed />

업스트림은 ECDet, ECPose, ECSeg를 세 헤드가 있는 하나의 모델이 아니라 세 개의 별도 모델로 게시합니다. ECViT 백본과 하이브리드 인코더를 공유하고 헤드만 다르므로 LibreYOLO는 하나의 제품군으로 합치고 체크포인트 파일명이 작업을 나타내게 합니다. 따라서 크기 문자는 세 작업에서 같은 백본과 인코더를 뜻하고 예측, 검증, 내보내기는 어떤 모델을 불러와도 같은 인수를 받습니다.

## 학습

세 작업 모두 `train()`으로 학습하며 불러온 체크포인트에서 작업을 읽고 일치하는 학습기를 선택합니다.

<code-tabs name="train" />

탐지와 분할에서 확인한 항목은 크기별 계층 단위 업스트림 대비 1e-5 추론 동등성과 합성 입력에서 손실 및 단일 학습 단계 실행입니다. `train()` 자체의 docstring에 따라 확인하지 않은 항목은 전체 미세 조정 수렴, 다중 GPU 학습, 증강 중단 후 최고 모델 다시 불러오기 단계, Objects365에서 COCO로의 클래스 재매핑입니다. 자세 경로는 클래스, 키포인트 L1, OKS 비용에 헝가리안 매처와 대조적 키포인트 노이즈 제거를 사용하는 DETRPose 공개 레시피를 따르며 이 수렴도 처음부터 끝까지 확인하지 않았습니다.

기본 설정에서는 혼합 정밀도를 활성화하고 `lr0=5e-4`로 74 epoch를 실행하며 업스트림 레시피에 따라 AdamW, flat cosine 일정, EMA 0.9999, ImageNet 정규화 입력을 사용합니다. 자세 추정과 분할은 평가 앵커 그리드가 모델 생성 시 구축되므로 체크포인트의 기본 크기와 같은 `imgsz`가 필요하며 다른 값은 실행 전에 예외를 발생시킵니다. 자세 추정에는 `data.yaml`이 `kpt_shape`를 선언하고 키포인트 수가 헤드와 일치하는 단일 클래스 데이터셋도 필요합니다.

`lora=True`는 탐지에만 적용됩니다. 자세 추정과 분할에서는 `ValueError`가 발생합니다. Apple silicon에서 학습기는 실행을 GPU에 유지하고 PyTorch가 Metal에서 구현하지 않은 deformable attention 내부의 grid-sample 역전파 연산 하나만 CPU로 보냅니다.

데이터셋, 증강, 다중 GPU, 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 지표 이름으로 키가 지정된 사전을 반환하며 `verbose`가 활성화되어 있으면 클래스별 결과를 출력합니다.

<code-tabs name="val" />

자세 추정은 `metrics/keypoints_*` 아래에 키포인트 OKS 지표를 보고합니다. 분할은 일반 `metrics/mAP50-95` 키 아래에 마스크를 보고하고 한 번의 패스에서 `(B)` 아래의 박스와 `(M)` 아래의 마스크도 반복해 제공합니다.

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다. 자세 추정과 분할은 동적 형태가 아닌 고정 640x640 입력으로 내보내며 OpenVINO, Paddle, MNN, ExecuTorch, Core AI를 포함한 여러 탐지 대상도 고정 캔버스를 사용합니다. 각 형식이 받는 인수와 일부 형식이 추가하는 extra는 [내보내기](/docs/export)에 나와 있습니다.

<code-tabs name="export" />

## 체크포인트

이 제품군에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
