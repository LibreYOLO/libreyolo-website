---
title: SSD
families:
  - ssd
seo_title: 'SSD(SSD300): LibreYOLO의 객체 탐지'
description: >-
  LibreYOLO에서 SSD300을 실행합니다. BSD-3-Clause로 예측, 검증, ONNX 내보내기를 지원하는 단일 순전파 VGG16
  탐지기이며 학습 경로는 없습니다.
lead: >-
  SSD(Single Shot MultiBox Detector)는 별도의 영역 제안 단계 없이 한 번의 순전파로 밀집 기본 바운딩 박스
  그리드의 모든 바운딩 박스와 클래스 점수를 예측합니다. LibreYOLO는 VGG16 기반 SSD300 체크포인트를 추론 전용 탐지기로
  제공합니다.
keywords:
  - SSD 사용법
  - SSD300
  - Single Shot MultiBox Detector
  - 객체 탐지
  - VGG16
  - 앵커 기반 탐지기
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSSD300.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSSD300.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSSD300.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")

        # 여기서는 imgsz를 의도적으로 생략합니다. SSD300은 체크포인트의
        # 네이티브 캔버스에서 추적하며 다른 값은 내보내기 전에 오류를 일으킵니다.
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSSD300.pt format=onnx
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreSSD300.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 3b3f9ea72291c4fa
---

## 설치

SSD에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에
포함됩니다.

```bash
pip install libreyolo
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열에서 동일하므로 다른 탐지기로 바꾸려면 한 줄만
수정하면 됩니다. SSD는 클래스별 점수와 함께 기본 바운딩 박스 그리드를 디코딩한 뒤
NMS를 실행하므로 이 라이브러리의 쿼리 기반 탐지기와 달리 `conf`, `iou`, `max_det`
모두 실제 효과가 있습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을
참조합니다.

## 변형

SSD는 고정 네이티브 캔버스에서 실행되는 VGG16 기반 SSD300 네트워크 체크포인트
하나를 제공합니다. 이 계열에는 선택할 크기나 스케일이 없으며 예측, 검증,
내보내기에서 모두 이 그래프 하나를 사용합니다.

가중치 파일은 계열 접두사 뒤에 유일한 크기 키인 `"300"`을 붙인
`LibreSSD300.pt`입니다. 이 파일의 클래스는 `LibreSSD`이므로 직접 생성할 때는 파일
이름을 따른 클래스 대신 `LibreSSD(size="300")`을 사용합니다.

## 검증

`val()`은 학습에 사용한 형식의 모든 데이터셋을 대상으로 측정한 정밀도, 재현율,
mAP 50, mAP 50-95를 포함하는 `metrics/` 키 딕셔너리를 반환합니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

SSD는 ONNX로만 내보내며 현재 이 계열의 다른 모든 형식은 차단됩니다. 내보내기는
항상 체크포인트의 네이티브 캔버스를 사용하고, 그래프는 NMS가 결합된 출력 대신 SSD의
원시 패킹 헤드를 노출하므로 내보낼 때 `nms=True`를 허용하지 않습니다. LibreYOLO
자체 백엔드는 그래프를 다시 불러온 뒤 디코딩과 억제 단계를 실행합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에서 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box>

LibreYOLO의 SSD300 코드는 논문 저자 자체 Caffe 릴리스를 포팅하지 않았습니다.
torchvision의 BSD-3-Clause SSD300 구현에서 파생되었으며, 위에서 업스트림 소스로
연결한 저장소가 해당 저장소입니다. 백본의 VGG16 가중치는 Karen Simonyan과 Andrew
Zisserman이 CC BY 4.0으로 공개한 Oxford의 완전 컨볼루션 reduced VGGNet까지 거슬러
올라갑니다.

</provenance-box>

## 인용

<citation-block />
