---
title: EdgeCrafter
families:
  - ec
seo_title: 'EdgeCrafter: LibreYOLO에서 탐지, 자세 인식 및 분할'
description: >-
  EdgeCrafter를 LibreYOLO에서 탐지, 자세 및 인스턴스 분할에 사용하십시오. MIT 라이선스 코드로 설치, 예측, 검증 및
  내보내기를 수행하십시오.
lead: >-
  엣지 하드웨어에서 밀도 예측을 위한 컴팩트 비전 트랜스포머로, 상위에 세 개의 형제 모델인 ECDet, ECPose, ECSeg로
  공개되었습니다. LibreYOLO는 세 가지 모델을 하나의 계열로 모두 로드하며, 작업은 체크포인트가 수행합니다.
keywords:
  - 엣지크래프터
  - ECDet
  - ECPose
  - ECSeg
  - 컴팩트 비전 트랜스포머
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
    - label: 자세
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 파일 이름의 -pose 접미사는 키포인트 헤드를 선택하므로, 아니요
        # 여기에 작업 인수가 필요합니다.
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
    - label: 자세
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yaml가 선언하는 단일 클래스 키포인트 데이터셋이 필요합니다
        # 체크포인트의 원래 크기에서 kpt_shape과 imgsz.
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

        # 체크포인트의 원래 크기에서 다각형 레이블과 이미지 크기가 필요합니다.
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
    - label: 자세
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
        print(metrics["metrics/mAP50-95(B)"])   # 상자들
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
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 경로를 지정하므로, 내보낸 아티팩트가 로드됩니다
        # 모든 체크포인트와 마찬가지로 같은 Results 객체를 반환합니다.
        model = LibreYOLO("LibreECs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 39c6975fc16b3ff1
---

## 설치

EdgeCrafter에는 선택적인 추가 항목이 필요 없습니다. 가져오는 모든 것은 기본 설치에 포함되어 있습니다.

```bash
pip install libreyolo
```

`lora=True`를 사용한 어댑터 파인튜닝은 예외이며, `lora` 추가가 필요합니다.

```bash
pip install "libreyolo[lora]"
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 다운로드되며 로컬에 캐시됩니다.

<code-tabs name="predict" />

작업은 파일 이름에서 결정되므로, `-pose` 또는 `-seg` 체크포인트는 자체 헤드를 선택하며 작업 인자를 필요로 하지 않습니다. 세 가지 모두가 모든 계열가 반환하는 `Results` 객체를 반환하며, 포즈의 경우 `result.keypoints`가, 세그멘테이션의 경우 `result.masks`가 추가됩니다. 포즈는 17개의 COCO 키포인트를 가진 한 클래스인 사람을 다루며, 모델이 구축될 때 수가 고정됩니다. 박스 헤드가 없기 때문에 각 포즈 박스는 자체 키포인트의 범위를 경계로 하며, 세 번째 키포인트 채널은 개별 점 점수가 아니라 상수입니다.

`conf`와 `max_det`는 쿼리 선택을 필터링합니다; `iou`는 API 일치를 위해 허용되지만 효과가 없습니다. 세 개의 헤드 모두 NMS 단계 없이 쿼리 집합을 디코딩하기 때문입니다. 소스, 스트리밍 및 결과 처리는 [prediction](/docs/predict)을 참조하십시오.

## 변형

네 가지 크기. 모두 동일한 입력 해상도로 실행되므로, 표는 매개변수 수와 정확도로 구분되어 있습니다.

<benchmark-table task="detect" />

<va-embed />

Upstream는 ECDet, ECPose, ECSeg를 세 개의 별도 모델로 게시하며, 세 개의 헤드를 가진 하나의 모델로 게시하지 않습니다. 이들은 ECViT 백본과 하이브리드 인코더를 공유하며 헤드만 다르기 때문에 LibreYOLO는 이를 하나의 계열로 합치고 체크포인트 파일 이름에 작업을 표시하도록 합니다. 따라서 사이즈 문자는 세 모델 모두 동일한 백본과 인코더를 의미하며, 어떤 모델을 불러오더라도 predict, validate, export는 동일한 인수를 사용합니다.

## 학습

세 가지 작업 모두 `train()`를 통해 학습되며, 이는 로드된 체크포인트에서 작업을 읽고 일치하는 트레이너를 선택합니다.

<code-tabs name="train" />

탐지 및 분할에 대해 확인된 사항: 업스트림 대비 추론 일관성(1e-5 기준), 레이어별 및 크기별 확인, 손실과 단일 학습 단계가 합성 입력에서 실행되는지 여부. `train()`의 자체 docstring에 따르면 확인되지 않은 사항: 전체 파인튜닝의 수렴, 다중 GPU 학습, stop-augmentation 최적 재로드 단계, Objects365를 COCO 클래스에 재매핑. 포즈 경로는 DETRPose의 공개 레시피를 따르며, 클래스, 키포인트 L1, OKS 비용에 대한 헝가리안 매처와 대조적 키포인트 디노이징을 사용하며, 수렴도 끝까지 확인되지 않음.

혼자 남겨진 상태에서, 트레이너는 업스트림 레시피를 따라 혼합 정밀도를 켠 상태로 `lr0=5e-4`에서 74 에폭을 수행합니다: AdamW, 평탄한 코사인 스케줄, EMA 0.9999, 그리고 ImageNet 정규화 입력. 포즈와 세그멘테이션은 모두 체크포인트의 기본 크기에서 `imgsz`를 필요로 합니다. 이는 평가 앵커 그리가 모델이 구축될 때 만들어지기 때문이며, 다른 값을 사용하면 실행 시작 전에 오류가 발생합니다. 포즈는 또한 `data.yaml`가 `kpt_shape`를 선언하는 단일 클래스 데이터셋을 필요로 하며, 키포인트 수가 헤드와 일치해야 합니다.

`lora=True`는 탐지에만 적용됩니다; 포즈와 세그멘테이션은 그 위에 `ValueError`를 발생시킵니다. Apple 실리콘에서는 트레이너가 GPU에서 실행을 유지하고 하나의 연산을 CPU로 전송하는데, 이는 변형 가능 주의(deformable attention) 내의 그리드 샘플 역전파로, PyTorch가 Metal에서는 구현하지 않습니다.

[training](/docs/train)에서 데이터셋, 증강, 멀티 GPU 및 로거에 대해 확인하십시오.

## 검증

`val()`는 메트릭 이름을 키로 하는 사전을 반환하며, `verbose`가 켜져 있으면 클래스별 결과를 출력합니다.

<code-tabs name="val" />

포즈는 `metrics/keypoints_*` 아래에서 주요 지점 OKS 지표를 보고합니다. 세그멘테이션은 일반 `metrics/mAP50-95` 키 아래에서 마스크를 보고하며, 한 번의 실행으로 두 뷰를 모두 반복하고, 박스는 `(B)` 아래에서, 마스크는 `(M)` 아래에서 보고합니다.

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사를 통해 `LibreYOLO()`로 다시 로드되므로, `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다. 포즈 및 세그멘테이션 내보내기는 동적 형태가 아닌 고정 640x640 입력에서 수행되며, 여러 탐지 대상도 고정 캔버스입니다. 여기에는 OpenVINO, Paddle, MNN, ExecuTorch 및 Core AI가 포함됩니다. [Export](/docs/export)는 각 형식이 허용하는 인수와 몇몇 형식이 추가하는 추가 항목을 나열합니다.

<code-tabs name="export" />

## 체크포인트

이 계열용으로 발행된 모든 무게 파일.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
