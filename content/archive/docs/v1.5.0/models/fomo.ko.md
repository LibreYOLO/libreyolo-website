---
title: FOMO
families:
  - fomo
seo_title: 'FOMO: LibreYOLO의 포인트 위치 추정, 학습, 내보내기'
description: >-
  LibreYOLO에서 많은 소형 객체를 세기 위한 초소형 포인트 위치 추정 탐지기 FOMO(Faster Objects, More
  Objects)를 실행합니다. 설치, 예측, 학습, 내보내기를 지원합니다.
lead: >-
  FOMO는 그리드 기반 포인트 위치 추정기입니다. 저해상도 그리드의 각 셀을 배경 또는 객체 중심으로 분류하며 바운딩 박스 회귀는 사용하지
  않습니다. LibreYOLO는 포인트 작업에서 이를 지원합니다.
keywords:
  - FOMO 사용법
  - Faster Objects More Objects
  - 포인트 위치 추정
  - 중심점 탐지
  - 소형 객체 탐지
  - 엣지 AI
  - MCU 객체 탐지
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # LibreFOMO 가중치는 자동으로 내려받지 않습니다(아래 체크포인트 참조).
        # 이미 로컬에 내려받은 체크포인트를 지정합니다.
        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for point in result.points:
            print(point.cls, point.conf, point.xy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=./LibreFOMOs-point.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=40, batch=32, lr0=3e-4,
        )
    - label: CLI
      language: bash
      code: >
        # imgsz를 전달해야 합니다. CLI 기본값은 640이며 s 체크포인트는

        # 네이티브 크기 96만 허용합니다.

        libreyolo train model=./LibreFOMOs-point.pt data=my-dataset.yaml
        imgsz=96 epochs=40 batch=32 lr0=3e-4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/grid_F1"])
        print(metrics["metrics/grid_precision"], metrics["metrics/grid_recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=./LibreFOMOs-point.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=./LibreFOMOs-point.pt format=onnx
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
source_hash: 03015f2bcd9fe99d
---

## 설치

FOMO에는 기본 패키지 외에 추가 항목이 필요하지 않습니다.

```bash
pip install libreyolo
```

## 예측

이 사이트의 다른 모든 계열과 달리 LibreFOMO 가중치는 자동으로 내려받지 않습니다.
`LibreYOLO("LibreFOMOs-point.pt")`는 디스크에서 해당 파일을 찾고 Hugging Face에서
가져오는 대신 파일 이름을 명시한 `ValueError`를 일으킵니다. 먼저
[LibreYOLO 조직](https://huggingface.co/LibreYOLO)에서 체크포인트를 내려받아 로컬
경로로 불러오거나 자체적으로 학습합니다(아래 학습 참조).

<code-tabs name="predict" />

결과에는 `boxes` 대신 `points` 페이로드가 들어 있습니다. 각 행은
`x, y, class, confidence`이며 `result.points.data` 또는 `.xy`, `.xyn`, `.cls`,
`.conf` 접근자로 사용할 수 있습니다. 억제할 바운딩 박스가 없으므로 설정할 `iou`
임계값도 없습니다. `predict(..., nms_radius=1)`은 두 탐지를 모두 유지하기 위해
얼마나 많은 그리드 셀 간격이 필요한지 제어하며, 로더가 인식하려면 파일 이름에
FOMO의 `-point` 작업 접미사가 있어야 합니다. 소스, 스트리밍, 결과 처리는
[예측](/docs/predict)을 참조합니다.

## 변형

`s`, `m`, `l` 크기 3개는 점차 넓어지는 MobileNetV2 방식 백본을 각각 더 큰 고정
입력 해상도에서 사용하며, 모두 단일 1x1 분류 헤드 뒤에 배치됩니다. 이 페이지에는
이 계열의 벤치마크 표가 없습니다. 아래 표의 체크포인트 파일 크기가 현재 공개된
크기별 지표 중 가장 명확합니다.

## 학습

<code-tabs name="train" />

`imgsz`는 자유롭게 선택할 수 없습니다. 기본값은 불러온 체크포인트의 네이티브
해상도이며 다른 값을 전달하면 예상 크기를 명시한 `ValueError`가 발생합니다. 해당
크기는 `s`가 96, `m`이 192, `l`이 224입니다. CLI의 `imgsz` 기본값은 640이므로
`libreyolo train` 명령에서 체크포인트에 맞는 값을 명시해야 합니다.

다른 값을 지정하지 않으면 트레이너는 Adam, 배치 32, `lr0=3e-4`, 가중치 감쇠
없이 40 에폭을 실행합니다. 일반적인 장면에서는 거의 모든 그리드 셀이 배경이므로
셀별 교차 엔트로피 손실에서 전경 클래스에 배경보다 100배 높은 가중치를 적용합니다.
EMA와 혼합 정밀도는 모두 기본적으로 비활성화되며, LibreYOLO의 다른 곳에서 사용하는
기하학적 또는 색상 증강도 적용하지 않습니다. 모자이크, mixup, HSV 지터, 뒤집기,
회전, 이동, 전단 값이 모두 0입니다.

공개된 LibreFOMO 체크포인트는 이 경로로 COCO에서 처음부터 학습했습니다.

데이터셋과 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 이 계열용 그리드 수준 검증기로 디스패치합니다. 다른 포인트 작업과 공유하는
포인트 일치 `metrics/precision`, `metrics/recall`, `metrics/mAP@` 키와 함께
신뢰도 임계값 및 `nms_radius` 값을 순회하며, 최상의 F1 조합을
`metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall`,
`metrics/grid_mean_distance`로 게시합니다. 해당 조합을 생성한 임계값과 반지름은
`decode/threshold`와 `decode/nms_radius`로 게시합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는
`.engine` 파일이 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다.
LibreYOLO를 설치하지 않고 런타임에서 그래프를 직접 실행할 수도 있지만 이 경우
전처리와 후처리를 직접 작성해야 합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에서 공개된 모든 가중치 파일입니다. 어떤 파일도 자동으로 내려받지 않습니다.
연결된 Hugging Face 페이지에서 원하는 파일을 가져와 로컬 경로를 `LibreYOLO()`에
전달합니다.

<checkpoint-table />

## 라이선스

<provenance-box>

연결할 FOMO 업스트림 코드 저장소는 없습니다. Edge Impulse는 블로그 게시물과 제품
문서에서 이 기법을 설명하지만 FOMO 학습 또는 추론 코드를 공개하지 않았습니다.
여기의 아키텍처와 학습은 해당 공개 설명을 바탕으로 한 LibreYOLO 자체 구현이며,
공개된 LibreFOMO 체크포인트는 COCO에서 처음부터 학습했습니다. 따라서 코드와 이러한
가중치는 모두 LibreYOLO 자체 MIT입니다. FOMO 이름과 해당 이름이 설명하는 기법은
계속 Edge Impulse에 속합니다.

</provenance-box>
