---
title: YOLOv9
families:
  - yolo9
seo_title: 'YOLOv9: MIT 기반 예측, 학습 및 내보내기'
description: >-
  NMS 없는 end-to-end 헤드와 stride-4 소형 객체 헤드를 포함한 YOLOv9를 LibreYOLO에서 실행합니다. 설치,
  예측, 학습, 검증, 내보내기를 수행합니다.
lead: >-
  단일 단계 합성곱 탐지기입니다. 한 번의 패스로 조밀한 박스 그리드에 점수를 매기고 NMS가 중복을 제거합니다. LibreYOLO는 세 가지
  변형을 제공하며 그중 하나에는 NMS 단계가 없습니다.
keywords:
  - YOLOv9 사용법
  - YOLO9
  - 객체 탐지
  - NMS 없는 탐지
  - end-to-end 탐지
  - 소형 객체 탐지
  - programmable gradient information
  - GELAN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: NMS 없이 실행
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 호출은 같고 체크포인트만 다릅니다. end-to-end 헤드가 자체 최고 점수
        # 예측을 반환하므로 NMS를 실행하지 않고 iou를 무시합니다.
        model = LibreYOLO("LibreYOLO9E2Es.pt")
        result = model(SAMPLE_IMAGE, conf=0.25, max_det=300)

        print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: 소형 객체
      language: python
      code: >
        from libreyolo import LibreYOLO9P2


        # stride-4 변형에는 자체 COCO 체크포인트가 없으므로 기본 탐지 체크포인트를

        # 지정합니다. 백본과 넥은 그대로 불러오고 stride-4 헤드 타워는 무작위 초기화합니다.

        model = LibreYOLO9P2(None, size="s")

        model.train(data="my-dataset.yaml", epochs=100,
        pretrained="LibreYOLO9s.pt")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=my-dataset.yaml
    - label: COCO 기준 평가
      language: bash
      code: |
        # 번들 COCO yaml에는 내장 다운로드 스크립트가 있으므로 데이터셋이 이미
        # 로컬에 있지 않다면 명시적 권한이 필요합니다.
        libreyolo val model=LibreYOLO9c.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: 그래프에 NMS 포함
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx nms=True \
          conf=0.25 iou=0.45 max_det=300
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreYOLO9s.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: eaa6023a4a0b9e71
---

## 설치

YOLOv9에는 기본 패키지 외의 extra가 필요하지 않습니다.

```bash
pip install libreyolo
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열이 반환하는 것과 같으므로 탐지기를 바꾸려면 한 줄만 변경하면 됩니다. 기본 모델과 stride-4 모델에서 `conf`는 신뢰도 임곗값을, `iou`는 NMS 임곗값을 설정합니다. end-to-end 모델은 NMS를 실행하지 않고 `iou`를 무시하므로 `conf`와 `max_det`이 출력을 결정합니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

세 변형은 백본을 공유합니다. 모두 탐지만 지원하고 같은 인수를 받습니다.

기본 모델은 세 특징 스케일에서 예측하고 NMS로 중복 박스를 제거합니다.

end-to-end 모델은 해당 헤드를 유지하고 옆에 일대일 매칭 브랜치를 추가합니다. 추론은 일대일 브랜치만 읽어 최고 점수 예측을 선택하므로 NMS를 실행하지 않습니다. 배포 런타임에 NMS 연산자가 없을 때 선택합니다.

stride-4 모델은 백본의 한 단계 더 위 레벨을 노출하고 넥을 해당 레벨까지 확장하여 세 스케일 대신 네 스케일에서 예측합니다. 추가 스케일은 픽셀을 적게 차지하는 객체를 위한 것입니다. 공개된 체크포인트 하나는 항공 이미지에서 학습했습니다. 기본 탐지 체크포인트를 전이할 수 있습니다. 백본과 넥은 그대로 불러오고 사전 학습한 세 헤드 타워를 한 슬롯 위로 옮기며 stride-4 타워는 무작위 초기화합니다.

<benchmark-table task="detect" />

<va-embed />

## 학습

<code-tabs name="train" />

`pretrained`가 실행의 시작점을 결정합니다. `True`를 전달하면 같은 모델과 크기의 공개 체크포인트를 불러오고 다른 항목에는 이름이나 경로를 전달합니다. 형태가 일치하지 않는 텐서는 거부하지 않고 건너뛰며 불러온 수를 기록합니다. 따라서 클래스 수가 다른 체크포인트도 시작점으로 사용할 수 있습니다.

stride-4 모델에는 자체 공개 COCO 체크포인트가 없으므로 여기서 `True`는 존재하지 않는 파일로 해석되어 다운로드가 실패합니다. 대신 기본 탐지 체크포인트의 이름을 지정합니다.

데이터셋, 증강, 다중 GPU, 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 학습에 사용한 형식의 데이터셋을 대상으로 측정한 정밀도, 재현율, mAP 50, mAP 50-95를 포함하는 `metrics/` 키 사전을 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

체크 표시는 세 변형 모두에 적용됩니다. 차이가 있는 경우 매트릭스는 셋 중 가장 약한 지원 수준을 표시합니다.

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다. LibreYOLO가 설치되지 않은 기본 런타임에서 그래프를 실행할 수도 있지만 이 경우 전처리와 후처리를 직접 작성해야 합니다.

기본 탐지 모델에서는 후처리 절반을 그래프로 옮길 수 있습니다. ONNX 내보내기에 `nms=True`를 사용하면 억제를 모델 안에 넣고 첫 출력은 행이 `x1, y1, x2, y2, score, class`인 고정 `(1, max_det, 6)` 텐서가 됩니다. 탐지 수 이후는 0으로 패딩합니다. 이 그래프는 배치 1이며 동적 축이 없습니다. end-to-end와 stride-4 모델은 이 플래그를 받지 않습니다.

각 형식은 서로 다른 extra를 설치하고 고유한 몇 가지 인수를 받습니다. 둘 다 해당 형식 페이지에 나와 있습니다.

<code-tabs name="export" />

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box>

여기 체크포인트 중 하나는 MIT가 아닙니다. VisDrone2019-DET에서 학습한 stride-4 모델은 해당 데이터셋의 CC BY-NC-SA 3.0 약관을 상속합니다. 비상업적 사용만 허용하고 파생물에 동일 조건을 적용해야 하며 나머지 계열의 허용적 라이선스 범위 밖입니다. COCO 클래스가 아닌 VisDrone 항공 클래스를 예측합니다. 라이브러리는 파일을 다운로드하기 전에 이 내용을 모두 출력합니다.

</provenance-box>

## 인용

<citation-block />
