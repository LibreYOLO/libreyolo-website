---
title: PicoSAM3
families:
  - picosam3
seo_title: 'PicoSAM3: LibreYOLO의 바운딩 박스 프롬프트 엣지 분할'
description: >-
  LibreYOLO에서 엣지 센서의 바운딩 박스 프롬프트 영역 분할에 PicoSAM3를 사용합니다. Apache-2.0이 적용된 pico
  체크포인트를 설치하고 예측, 내보내기합니다.
lead: >-
  PicoSAM3는 SAM 2.1과 SAM 3에서 증류된 소형 CNN으로, Sony IMX500 같은 센서의 바운딩 박스 프롬프트 관심 영역
  분할을 위해 설계되었습니다. LibreYOLO는 LibreYOLO() 탐지기 팩토리와 별도의 전용 LibreSAM 팩토리로 이를 지원하며
  바운딩 박스 프롬프트만 사용할 수 있습니다.
keywords:
  - PicoSAM3 사용법
  - Segment Anything
  - 엣지 분할
  - 관심 영역 분할
  - 바운딩 박스 프롬프트
  - 센서 내 추론
  - IMX500
  - 지식 증류
last_verified: 1.5.0
snippets:
  predict:
    - label: 바운딩 박스 프롬프트
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # PicoSAM3는 "pico" 한 크기만 있으므로 다른 별칭이 필요하지 않습니다.
        model = LibreSAM("picosam3")

        # bboxes=만 지원합니다. [x1, y1, x2, y2] 또는 바운딩 박스 목록이며
        # 바운딩 박스마다 마스크 하나를 생성합니다. 각 바운딩 박스는 10% 확장하고
        # 정사각형으로 만든 뒤 이미지에 맞춰 자르고 CNN 실행 전 96x96으로 조정합니다.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
        print(result.masks.xy)      # 마스크별 폴리곤
        print(result.boxes.xyxy)    # 마스크에서 파생된 꼭 맞는 바운딩 박스
    - label: 한 번 인코딩하고 여러 번 프롬프트하기
      language: python
      code: |
        from libreyolo import LibrePicoSAM3, SAMPLE_IMAGE

        model = LibrePicoSAM3()

        # set_image()는 소스 이미지를 캐시합니다. PicoSAM3는 바운딩 박스마다 전체
        # CNN 순전파를 실행하므로 다른 SAM 계열처럼 인코더 단계를 절약하는 대신
        # 이미지 로드와 디코딩을 절약합니다.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(bboxes=[300, 200, 900, 700])
        b = model.predict(bboxes=[100, 100, 400, 400])
        model.reset_image()
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibrePicoSAM3

        model = LibrePicoSAM3()
        model.export(format="onnx", output_path="LibrePicoSAM3pico.onnx")

        # 이 계열이 허용하는 내보내기 인수는 opset(기본값 13)과
        # dynamic(기본값 True, 배치 축만)뿐입니다.
    - label: 내보낸 파일 사용하기
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # PicoSAM3는 원시 96x96 ROI CNN을 내보냅니다: roi_image -> mask_logits.

        # 탐지기 체크포인트처럼 export()가 LibreYOLO()로 다시 라우팅되지 않으므로

        # 여기서 재사용할 LibreYOLO 측 전처리나 후처리는 없습니다.

        session = ort.InferenceSession("LibrePicoSAM3pico.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 96, 96),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 5d60ff14fe61ba29
---

## 설치

PicoSAM3에는 `sam` extra가 필요합니다. 추론은 네이티브 비 `transformers` CNN에서
실행하지만 LibreYOLO 자체 가중치 내려받기는 여전히 `transformers`의 Hugging Face
도구를 사용합니다.

```bash
pip install "libreyolo[sam]"
```

## 예측

`LibreSAM(...)` 또는 계열별 `LibrePicoSAM3(...)`은 `LibreYOLO(...)`와 별도의
진입점입니다. 여기서는 프롬프트 없이는 순전파가 의미 없으므로 탐지기가 아니라
프롬프트 가능 분할기를 반환합니다. 이 계열에는 `libreyolo predict` CLI 명령이
없으므로 Python API를 사용합니다.

<code-tabs name="predict" />

PicoSAM3는 `bboxes=`만 허용합니다. 업스트림 모델에 다른 모드가 없으므로
`points=`, `labels=`, `masks=`, `text=`, `multimask=True`를 전달하거나 바운딩
박스를 생략하여 모든 항목을 분할하려 하면 명확한 `ValueError`가 발생합니다.
`conf`는 탐지 신뢰도가 아니라 예측한 마스크 품질(IoU)을 기준으로 필터링하며
`0.0`에서 `1.0` 사이여야 합니다. 모든 마스크에는 이름이 `"object"`인 클래스 ID
`0`이 지정됩니다. `train()`, `val()`, `track()`은 `NotImplementedError`를
일으킵니다. 포인트, 텍스트, 마스크 또는 모두 분할 프롬프트에는 LibreSAM2 또는
LibreSAM3를 사용합니다. 소스 유형은 [예측](/docs/predict)을 참조합니다.

## 변형

고정 96 px ROI 입력을 사용하는 pico 한 크기만 있습니다. PicoSAM3는 전체 이미지를
한 번 인코딩하는 대신 바운딩 박스마다 전체 CNN 순전파를 실행합니다.

## 내보내기

<export-matrix />

PicoSAM3는 SAM 계층에서 내보낼 수 있는 유일한 계열입니다. NMS나 마스크 후처리를
포함하지 않고 원시 96x96 ROI CNN을 `roi_image -> mask_logits` 형식의 ONNX로
내보냅니다. 다른 SAM 계열은 인코더와 디코더 분할에 정의된 런타임 내보내기 계약이
아직 없으므로 `export()`에서 `NotImplementedError`를 일으킵니다. 내보낸 PicoSAM3
그래프는 `LibreYOLO()`로 다시 불러오지 않습니다. 위에 표시된 것과 같은 10% 패딩
정사각형 ROI 전처리를 적용하고 `onnxruntime` 같은 런타임으로 직접 실행합니다.

<code-tabs name="export" />

## 체크포인트

이 계열에서 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box>

PicoSAM3는 SAM 2.1과 SAM 3를 교사 모델로 사용해 증류합니다. LibreYOLO는 이
계열에서 어느 교사 모델의 코드나 가중치도 포함하거나 재배포하지 않으며, 소형 학생
CNN과 변환된 체크포인트만 제공합니다.

</provenance-box>

## 인용

<citation-block />
