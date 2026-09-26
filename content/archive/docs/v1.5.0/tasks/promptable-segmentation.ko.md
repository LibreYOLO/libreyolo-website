---
title: 프롬프트 가능 분할
seo_title: LibreYOLO에서의 프롬프트 가능한 분할
description: >-
  LibreYOLO에서 포인트, 상자 또는 텍스트 개념을 객체 마스크로 변환하십시오. LibreSAM을 통해 SAM, SAM 2, SAM 3,
  EdgeTAM, MobileSAM 또는 PicoSAM3를 불러오십시오.
lead: >-
  프롬프트 가능한 분할은 클릭을 마스크로 바꿉니다: 객체를 가리키거나 그 주위에 상자를 그리면 모델이 그 윤곽을 반환합니다.
  LibreYOLO에서는 이것이 별도의 작업 키가 아니라 모델 계층이며, LibreSAM 팩토리를 통해 로드되며, 그 결과는 일반적인 분할
  결과입니다.
keywords:
  - 프롬프트 가능 분할
  - 인터랙티브 세분화
  - 파이썬으로 무엇이든 분할하기
  - 포인트 프롬프트
  - 박스 프롬프트
  - SAM 파이썬
  - 클릭에서 마스크
last_verified: 1.5.0
snippets:
  predict:
    - label: 포인트 및 박스 프롬프트
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # 한 점은 픽셀 단위의 [x, y]이고, 레이블은 1이면 양성, 0이면 음성입니다.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # 다각형
        print(result.boxes.xyxy)    # 마스크에서 유래한 빽빽한 상자들

        # 박스 프롬프트는 박스당 하나의 마스크를 제공합니다.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: 한 번 인코딩하고 여러 번 프롬프트하기
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # set_image는 무거운 이미지 인코더를 한 번 실행하고 캐시합니다.
        model.set_image(SAMPLE_IMAGE)
        first = model.predict(points=[640, 420], labels=[1])
        second = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
    - label: 모든 것을 분할하다
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # 프롬프트가 없다는 것은 이미지 전체에 걸쳐 점들의 격자를 의미합니다. 기본값
        # 한 변에 32개의 그리드는 약 1024번의 디코더 패스를 의미하며, 이는 CPU에서 느립니다.
        result = model.predict(SAMPLE_IMAGE, points_per_side=8)
        print(len(result.masks))
    - label: 모호성 마스크
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # 한 점은 소매, 셔츠, 또는 사람을 의미할 수 있습니다. 멀티마스크=True
        # 가장 좋은 것 대신 세 가지 전체 대 부분 마스크 모두를 반환합니다.
        result = model.predict(
            SAMPLE_IMAGE, points=[640, 420], labels=[1], multimask=True
        )
        print(len(result.masks))
source_hash: bb70ff24e6c0a767
---

## 정의

프롬프트 기반 세그멘테이션은 이미지와 공간 프롬프트를 받아 프롬프트가 가리키는 대상의 마스크를 반환합니다. 분류되는 것은 없습니다: 클래스 목록이 없으며, `result.boxes`는 자체적인 검출이 아니라 마스크에서 파생된 단단한 상자를 유지합니다. `result.masks`는 마스크 데이터를 가지고 있고 `result.masks.xy`는 그 폴리곤을 가지고 있습니다.

프롬프트는 인터페이스입니다. `points`는 `[x, y]` 픽셀 좌표로, 객체당 한 세트씩 있으며, `labels`는 각 점을 양수(1, 포함) 또는 음수(0, 제외)로 표시합니다. `bboxes`는 `[x1, y1, x2, y2]`로, 상자당 하나의 마스크가 있습니다. 점과 상자는 결합될 수 있으며, 이 경우 객체별로 쌍을 이루어야 하고 길이가 동일해야 합니다. 모든 프롬프트를 생략하면 이미지 위의 점 그리드로 세그먼트-모두 경로가 실행됩니다.

단일 지점은 원래부터 애매합니다. 소매를 클릭하는 것은 소매, 셔츠 또는 사람을 의미할 수 있으므로 `multimask=True`는 단일 최적 마스크 대신 프롬프트별로 세 가지 전체-부분 마스크를 모두 반환합니다. `conf`는 탐지 확신도가 아니라 모델이 예측한 IoU, 즉 마스크 품질 점수를 기준으로 필터링합니다.

LibreYOLO에는 `promptable` 작업 키가 없습니다. 계층은 `segment`로 등록되며, 이는 인스턴스 분할이 사용하는 동일한 키입니다. 이를 구분하는 것은 호출 형태이며, 그래서 자체 팩토리 `LibreSAM()`가 있으며, 이는 `LibreYOLO()`, `LibreOpenVocab()`, `LibreVLM()`의 형제입니다. 단일 `predict(image)` 시그니처로는 이러한 모델들이 구축된 루프를 표현할 수 없습니다: `set_image()`는 이미지 인코더를 한 번 실행하고 임베딩을 캐시하며, 이후 모든 `predict()` 호출에서 `source=None`는 오직 프롬프트 디코딩만 비용이 발생하고, `reset_image()`는 캐시를 지웁니다. 이미지 인코더는 주요 비용을 차지하며 이미지당 한 번 실행되므로, 동일한 이미지에서 두 번째 프롬프트는 이를 완전히 건너뜁니다.

## 모델들

여섯 계열이 별명을 통해 `LibreSAM`를 불러옵니다.

[SAM](/docs/models/sam)는 기본값이며, `base`, `large` 및 `huge` 크기에서도 사용됩니다. 또한 `b`, `l` 및 `h`로도 표기됩니다.

[SAM 2](/docs/models/sam-2), `sam2-tiny`, `sam2-small`, `sam2-base-plus` 및 `sam2-large`로서. LibreYOLO는 그 이미지 경로를 지원합니다.

[SAM 3](/docs/models/sam-3), `sam3`로서 텍스트 개념 프롬프트를 수락하는 유일한 계열입니다: `text="yellow school bus"`는 모든 일치하는 항목을 반환합니다. `text=`를 다른 계열에 전달하면 SAM 3을 언급하는 메시지가 나타납니다. 그 가중치는 LibreYOLO의 MIT 라이선스가 아닌 Meta에서 제공한 맞춤형 SAM 라이선스에서 나오며, 저장소는 제한되어 있습니다: 모델 페이지에서 조건에 동의하고 첫 다운로드 전에 `hf auth login`로 인증해야 합니다. 배포하기 전에 [SAM 3](/docs/models/sam-3)을 읽으십시오.

[EdgeTAM](/docs/models/edgetam), `edgetam`로서, SAM 2의 디바이스 내 버전입니다. LibreYOLO는 이미지 경로를 지원합니다.

[MobileSAM](/docs/models/mobilesam)은 `mobilesam`로서, SAM의 ViT-H 인코더를 증류된 TinyViT 인코더로 교체합니다.

[PicoSAM3](/docs/models/picosam3), `picosam3`로서, 엣지 센서에서 박스 프롬프트 영역을 위한 컴팩트 CNN입니다. 박스 프롬프트는 여기서 전체 계약입니다: 포인트, 텍스트, 마스크, 멀티마스크 및 모든 세그먼트는 모두 SAM 2 또는 SAM 3을 가리키는 메시지와 함께 올라갑니다.

이 계층의 추가 항목은 `transformers`를 통해 로드되는 네 계열을 포함합니다:

```bash
pip install "libreyolo[sam]"
```

MobileSAM과 PicoSAM3는 네이티브 LibreYOLO 포트이며 실행하기 위해 `transformers` 설치가 필요 없습니다.

## 예측

<code-tabs name="predict" />

`source`와 `set_image()`는 순서가 아니라 대안입니다: 이미지 하나를 `predict()`에 전달하여 원샷 호출을 하거나, 먼저 `set_image()`를 호출한 후 각 프롬프트마다 `predict(source=None)`를 호출합니다. `device=`를 `predict()`에 전달하면 그 호출과 이후 모든 호출에 대해 모델이 이동하며, 캐시된 임베딩은 무효화됩니다.

Segment-everything은 비싼 모드입니다. `points_per_side`는 기본적으로 32로 설정되어 있으며, 이는 대략 이미지에 대해 1024번의 디코더 패스를 의미합니다. CPU에서 상호작용이 필요한 경우에는 값을 낮추십시오. 이 모드에서 `conf`는 설정되지 않은 경우 계열의 그리드 임계값을 적용하는 반면, 프롬프트 경로에서는 설정되지 않은 `conf`가 모든 마스크를 유지합니다. `conf=0.0`를 전달하면 어떤 모드에서도 필터링을 비활성화할 수 있으며, `max_det`를 전달하면 반환되는 마스크 수를 제한할 수 있습니다.

마스크 프롬프트는 이 버전에서 지원되지 않으며, `masks=`는 무시되지 않고 오류를 발생시킵니다. `track()`도 모든 단계에서 오류를 발생시킵니다: 이들은 이미지 분할기이므로, 프레임마다 `predict()`를 실행하십시오. 소스와 결과 처리는 [예측](/docs/predict)을 참조하십시오.

## 학습

이 계층에는 LibreYOLO 안에서 학습하는 계열이 없습니다. `train()`가 제안합니다: 업스트림을 파인튜닝하고 결과로 나온 가중치를 불러옵니다.

## 검증

이 계층에는 검증기가 없으며, `val()`가 발생합니다. 프롬프트 가능한 마스크는 점수를 매길 고정 클래스 집합이 없기 때문에 일반적인 검출 및 분할 지표에는 기준이 없습니다. 프롬프트된 마스크를 점수화한다는 것은 사용자가 직접 제공한 참조 마스크와 관심 있는 프롬프트에 대해 비교하는 것을 의미합니다.

## 내보내기

내보내기은 전체 계층 범위에서는 제외되며 `export()`에서는 하나의 예외를 제외하고 적용됩니다. [PicoSAM3](/docs/models/picosam3)는 원시 96x96 영역 CNN을 `roi_image -> mask_logits`로 ONNX에 내보냅니다; 박스 자르기와 마스크를 이미지 좌표로 다시 조정하는 작업은 Python에서 유지됩니다. 다른 모든 계열은 PyTorch에서 `predict()`를 통해 실행됩니다. 라이브러리의 다른 곳에서 사용 가능한 형식은 [export](/docs/export)를 참조하십시오.
