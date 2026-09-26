---
title: 빠른 시작
seo_title: LibreYOLO 빠른 시작
description: >-
  이미지에서 디텍터를 실행하고, 작은 데이터셋으로 파인튜닝한 다음, TorchScript나 ONNX로 내보내십시오. 모두 CPU에서 약
  10줄의 파이썬으로 가능합니다.
lead: >-
  LibreYOLO를 통한 가장 짧은 경로: 한 이미지로 예측하고, 작은 데이터셋으로 학습한 다음 결과를 내보냅니다. 여기의 모든 명령은
  CPU에서 실행됩니다.
keywords:
  - libreyolo 빠른 시작
  - libreyolo 튜토리얼
  - libreyolo 예측
  - libreyolo 학습
  - libreyolo 내보내기
  - yolo 파이썬 예제
last_verified: 1.5.0
meta:
  - label: 설치
    value: pip install libreyolo
    mono: true
  - label: 검문소
    value: LibreYOLO9t.pt
    mono: true
  - label: 하드웨어
    value: 이 페이지의 모든 것에는 CPU면 충분합니다
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 첫 사용 시 체크포인트를 다운로드한 후 weights/.에 캐시합니다
        model = LibreYOLO("LibreYOLO9t.pt")

        # 단일 이미지는 하나의 Results 객체를 반환합니다.
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=yolo9-t save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 비디오와 스트림
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # `stream=True`는 리스트를 만드는 대신 각 프레임마다 하나의 결과를 생성합니다.
        # 경로를 웹캠 인덱스, RTSP URL 또는 폴더로 교체하십시오.
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco8는 라이브러리와 함께 제공되는 8-이미지 데이터셋입니다. 다운로드합니다
        # 첫 사용 시 URL에서 가져오므로 스크립트를 실행할 필요가 없습니다.
        results = model.train(
            data="coco8.yaml",
            epochs=1,
            imgsz=640,
            batch=4,
            device="cpu",
        )

        print(results["save_dir"])
        print(results["best_checkpoint"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=yolo9-t data=coco8.yaml \
          epochs=1 imgsz=640 batch=4 device=cpu
    - label: 검증하다
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val()은 객체가 아니라 일반 딕셔너리를 반환합니다.
        metrics = model.val(data="coco8.yaml", device="cpu")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
  export:
    - label: TorchScript
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # export()는 작성한 경로를 반환합니다.
        path = model.export(format="torchscript")
        print(path)

        # 팩토리는 파일 접미사에 따라 경로를 지정하므로, 아티팩트가 다시 로드됩니다.
        # 체크포인트를 만들고 동일한 Results 객체를 반환합니다.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: ONNX
      language: bash
      code: |
        pip install "libreyolo[onnx]"
        libreyolo export model=yolo9-t format=onnx imgsz=640
source_hash: c11b6bdbf0b6fdf1
---

## 설치

```bash
pip install libreyolo
```

아래의 predict 및 train 섹션에 필요한 모든 것입니다. ONNX로 내보내기는 하나를 추가합니다. 전체 목록은 [install](/docs/install)을 참조하십시오.

## 예측

<code-tabs name="predict" />

`LibreYOLO()`는 팩토리입니다. 이 파일을 읽고, 가중치가 어느 계열에 속하는지 확인한 후 그 계열의 모델을 반환하므로, 다른 탐지기를 교체하는 것은 한 줄의 변경으로 가능합니다. 디렉토리를 지정하지 않고 `LibreYOLO9t.pt`을 전달하면, 작업 디렉토리를 기준으로 `weights/LibreYOLO9t.pt`를 찾고, 없으면 그곳에 다운로드합니다. 다운로드 규칙과 오프라인 사용 방법은 [체크포인트 및 가중치](/docs/weights)를 참조하십시오.

`save=True`는 `runs/detect/` 아래에서 주석이 달린 사본을 작성하며, 실행할 때마다 증가하는 `predict` 디렉토리에 저장합니다. 반환된 `Results`는 `boxes`를 포함하고, `names`는 클래스 인덱스를 해당 레이블에 매핑합니다. 단일 이미지 경로는 하나의 `Results`를 반환하고, 디렉토리, 이미지 목록 또는 `stream=True`는 그들의 목록이나 제너레이터를 반환합니다.

## 학습

<code-tabs name="train" />

`data`는 데이터셋 YAML입니다. `coco8.yaml`는 라이브러리와 함께 제공되므로, 해당 스니펫이 붙여넣기한 그대로 실행됩니다. 번들되지 않은 이름은 경로로 읽힙니다. 데이터셋은 `~/datasets` 아래에서 해결되거나, 해당 변수가 설정된 경우 `LIBREYOLO_DATASETS_DIR` 아래에서 해결됩니다.

실행은 `project/name`에 기록하며, 기본적으로 `runs/train` 아래의 디렉토리에 기록되고, 그 안에는 `weights/best.pt`와 `weights/last.pt`가 있습니다. `train()`는 `save_dir`, `best_checkpoint`, `last_checkpoint`, 에폭별 손실 및 에폭별 검증 지표를 포함하는 사전을 반환합니다. 학습된 체크포인트는 사전 학습된 것과 정확히 동일하게 `LibreYOLO()`를 통해 불러옵니다.

모든 계열이 학습 가능한 것은 아닙니다. 계열이 추론만 전달하는 경우, `train()`는 `NotImplementedError`를 올리고 그렇게 말합니다. [핵심 개념](/docs/concepts)은 어떤 지원 단계가 무엇을 의미하는지 설명합니다.

## 내보내기

<code-tabs name="export" />

TorchScript는 기본 설치 외에 아무 것도 필요하지 않습니다. 다른 대상은 각각 별도의 추가 사항이 있으며, 커버리지는 균일하지 않고 계열별 및 작업별로 이루어집니다: [내보내기 및 배포](/docs/export)를 참조하십시오.

모든 형식에서 허용되는 인수에는 `imgsz`(정수 또는 높이와 너비 쌍), `batch`(기본값 1), `half`, `int8`와 보정을 위한 `data` YAML, `dynamic`(기본값 True), `simplify`(기본값 True), `opset`, `device` 및 `output_path`가 포함됩니다. `output_path`가 생략되면 파일은 체크포인트에서 파생된 이름으로 `weights/` 아래에 작성됩니다.

## 다음에 어디로 갈지

- [작업, 계열, 크기 및 체크포인트 이름에 대한 핵심 개념](/docs/concepts).
- 자동 다운로드, 오프라인 사용 및 로딩 안전을 위한 [체크포인트 및 가중치](/docs/weights).
- 이미 업스트림 프로젝트에서 체크포인트가 있다면 [기존 가중치 가져오기](/docs/migrate)를 하십시오.
- 문제에 맞는 계열을 찾기 위한 [모든 모델](/docs/models).
- [학습](/docs/train), [예측](/docs/predict) 및 [내보내기](/docs/export) 전체 워크플로우를 위해.
