---
title: YOLOv7
families:
  - yolo7
seo_title: 'LibreYOLO의 YOLOv7: MIT 기반 예측, 학습 및 내보내기'
description: 'LibreYOLO에서 YOLOv7로 객체를 탐지합니다. MIT 라이선스의 코드와 가중치로 설치, 예측, 학습, 검증, 내보내기를 수행합니다.'
lead: >-
  YOLOv7은 최종 합성곱 전에 학습된 암묵 지식 오프셋을 추가하는 헤드를 갖춘 앵커 기반 단일 단계 탐지기입니다. LibreYOLO는
  공개된 단일 크기의 탐지를 지원합니다.
keywords:
  - YOLOv7 사용법
  - 객체 탐지
  - 앵커 기반 탐지
  - 암묵 지식
  - ImplicitA
  - 실시간 객체 탐지
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO7b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO7b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO7b.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO7b.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
    - label: 새 모델에서 워밍 스타트
      language: python
      code: |
        from libreyolo import LibreYOLO7

        # pretrained=True는 이 인스턴스를 무엇으로 생성했는지와 관계없이 항상
        # 공개된 LibreYOLO7b.pt 체크포인트를 불러옵니다. LibreYOLO()를 통하지 않고
        # 클래스를 직접 생성하면 아무 가중치도 불러오지 않은 상태로 시작합니다.
        model = LibreYOLO7(None, size="b")
        model.train(data="my-dataset.yaml", epochs=300, pretrained=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO7b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLO7b.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLO7b.pt format=tensorrt imgsz=640
        half=True
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreYOLO7b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 361e81de5614a571
---

## 설치

YOLOv7에는 기본 패키지 외의 extra가 필요하지 않습니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열이 반환하는 것과 같으므로 탐지기를 바꾸려면 한 줄만 변경하면 됩니다. `conf`는 신뢰도 임곗값을, `iou`는 앵커 기반 헤드를 디코딩한 뒤 적용하는 NMS 임곗값을 설정합니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

LibreYOLO는 `b` 크기 하나를 제공합니다. 업스트림은 YOLOv7 모델 하나만 공개하므로 선택할 크기가 없습니다.

## 학습

<code-tabs name="train" />

일부 다른 계열에서 같은 이름의 인수가 아무 작업도 하지 않는 것과 달리 여기서는 `pretrained`를 읽습니다. `True`를 전달하면 자동으로 다운로드되는 공개 `LibreYOLO7b.pt` 체크포인트에서 워밍 스타트하고, 다른 항목에는 경로나 이름을 전달합니다. 해당 공개 체크포인트는 80클래스 COCO입니다. 클래스 수가 다른 모델에서 이를 요청하면 먼저 80클래스로 다시 구축하여 불러온 다음 데이터셋의 클래스 수를 읽은 뒤 대상 헤드 수에 형태가 일치하는 모든 텐서를 한 번 전이합니다. `resume=True`는 `pretrained`와 함께 사용할 수 없습니다. 기본값 `None`을 유지하면 모델 생성 시 불러온 가중치에서 학습을 이어가거나 아무것도 불러오지 않았다면 무작위 초기화에서 시작합니다.

그 외 기본 설정에서는 SGD 모멘텀 0.937, 3 epoch 워밍업, `lr0=0.01`로 300 epoch를 실행합니다. YOLOX와 같은 SimOTA 할당과 마지막 15 epoch 무증강 단계를 앵커 기반 헤드에 맞게 적용합니다. 차이점이 하나 있습니다. YOLOX는 마지막 epoch에 L1 박스 회귀 정제를 추가하지만 v7은 이를 생략합니다. v7의 SimOTA 손실에는 정제할 원시 오프셋 L1 브랜치가 없기 때문입니다.

데이터셋, 증강, 다중 GPU, 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 학습에 사용한 형식의 데이터셋을 대상으로 측정한 정밀도, 재현율, mAP 50, mAP 50-95를 포함하는 `metrics/` 키 사전을 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러옵니다. 따라서 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다. LibreYOLO가 설치되지 않은 기본 런타임에서 그래프를 실행할 수도 있지만 이 경우 전처리와 후처리를 직접 작성해야 합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
