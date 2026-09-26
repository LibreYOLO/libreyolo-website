---
title: D-FINE
families:
  - dfine
seo_title: 'D-FINE: MIT 기반 파인튜닝, 검증 및 내보내기'
description: >-
  LibreYOLO에서 D-FINE으로 객체 탐지와 인스턴스 분할을 수행합니다. MIT 라이선스 코드로 설치, 예측, 파인튜닝, 검증,
  내보내기합니다.
lead: >-
  박스 회귀를 각 박스 가장자리의 확률 분포로 재정의하고 디코더 계층에 걸쳐 정제하는 detection transformer입니다.
  LibreYOLO는 객체 탐지와 인스턴스 분할을 지원합니다.
keywords:
  - D-FINE 사용법
  - detection transformer
  - 실시간 객체 탐지
  - 인스턴스 분할
  - fine-grained distribution refinement
  - DETR
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDFINEn.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 인스턴스 분할
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 파일명의 -seg 접미사가 마스크 헤드를 선택하므로 여기서는
        # task 인수가 필요하지 않습니다.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDFINEn.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: 인스턴스 분할
      language: bash
      code: |
        # 마스크 헤드를 포함해 공개된 분할 가중치에서 이어서 학습합니다.
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: 탐지 가중치에서 분할 전이
      language: bash
      code: |
        # 탐지 가중치에는 마스크 헤드가 없으므로 명시적인 전이입니다.
        # 헤드는 미학습 상태로 시작하고 학습한 뒤에만 유용합니다.
        # 여기서 task=segment를 요청하는 것이 전이를 승인합니다.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: 다중 GPU
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn.pt data=my-dataset.yaml
    - label: 인스턴스 분할
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # 마스크
        print(metrics["metrics/mAP50-95(B)"])   # 박스
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDFINEn.pt format=onnx imgsz=640

        libreyolo export model=LibreDFINEn.pt format=tensorrt imgsz=640
        half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreDFINEn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 0216631a26185524
---

## 설치

D-FINE에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

단, `lora=True`를 사용하는 어댑터 파인튜닝에는 `lora` extra가 필요합니다.

```bash
pip install "libreyolo[lora]"
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열이 반환하는 것과 같으므로 탐지기를 바꾸려면 한 줄만 변경하면 됩니다. `-seg` 파일명은 스스로 분할 작업으로 해석되며 `result.masks`에 박스와 함께 인스턴스 마스크가 담깁니다. `conf`와 `max_det`은 쿼리 선택을 필터링합니다. 디코더가 NMS 단계 없는 집합 예측기이므로 `iou`는 API 일관성을 위해 허용되지만 효과가 없습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

크기는 다섯 가지입니다. 모두 같은 입력 해상도에서 실행되므로 아래 표는 매개변수 수와 정확도로 구분합니다.

<benchmark-table task="detect" />

<va-embed />

분할은 탐지 백본, 인코더, 디코더를 재사용하고 마스크 헤드를 추가합니다. 따라서 `-seg` 체크포인트는 탐지 형제와 같은 인수를 받습니다. LibreYOLO의 RT-DETRv4 계열은 D-FINE 래퍼의 하위 클래스로 작성됩니다. 이 디코더 계열을 상속한 다음 마스크 헤드가 없으므로 작업 목록을 다시 탐지 전용으로 고정합니다.

## 학습

두 작업 모두 공개된 체크포인트에서 학습을 시작합니다.

<code-tabs name="train" />

기본 설정에서는 `amp=False`, 배치 16, `lr0=2e-4`로 132 epoch를 실행하며 50 epoch 동안 개선이 없으면 조기 중단합니다. 탐지 가중치는 분할 학습의 합법적인 시작점이지만 마스크 헤드가 미학습 상태로 시작하고 그렇지 않으면 의미 없는 마스크를 반환하므로 명시적인 전이로만 허용됩니다. CLI에서 `task=segment`를 전달하면 이를 승인합니다. Python 경로는 더 제한적입니다. `LibreYOLO()` 팩토리는 해당 인수를 받지 않으므로 `allow_detect_to_segment_transfer=True`로 `LibreDFINE`을 직접 생성해야 합니다. 직접 생성하면 다운로드하지 않으므로 가중치 파일이 이미 디스크에 있어야 합니다.

`lora=True`는 탐지에 적용됩니다. 분할 학습은 어댑터로 테스트하지 않은 마스크 헤드 때문에 이를 거부하고 대신 `freeze='backbone'`을 안내합니다. Apple silicon에서는 Integral의 binned matmul 역전파가 Metal 컴파일 오류를 일으키므로 학습 전체를 CPU로 이동합니다. MPS 추론은 영향을 받지 않습니다.

데이터셋, 증강, 다중 GPU, 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 지표 이름으로 키가 지정된 사전을 반환하며 `verbose`가 활성화되어 있으면 클래스별 결과를 출력합니다.

<code-tabs name="val" />

`-seg` 체크포인트에서는 일반 `metrics/mAP50-95` 키에 마스크 점수가 들어갑니다. 같은 실행에서 `(B)` 아래에 박스, `(M)` 아래에 마스크를 보고하므로 한 번의 패스로 둘 다 얻을 수 있습니다.

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다. OpenVINO, Paddle, MNN, Core AI 경로는 동적 형태가 아니라 고정 캔버스로 내보냅니다. 각 형식이 받는 인수와 일부 형식이 추가하는 extra는 [내보내기](/docs/export)에 나와 있습니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box>

분할 가중치에는 두 번째 업스트림이 있습니다. 마스크 디코더, 마스크 매칭, 마스크 손실은 역시 Apache-2.0인 ArgoHA/D-FINE-seg에서 가져왔으며 관리자가 저작자 표시와 함께 재사용을 승인했습니다.

</provenance-box>

## 인용

<citation-block />
