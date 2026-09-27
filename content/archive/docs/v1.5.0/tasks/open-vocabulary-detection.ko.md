---
title: 오픈 보캐뷸러리 탐지
seo_title: LibreYOLO의 오픈 보카뷸러리 탐지
description: >-
  LibreYOLO에서 텍스트 어휘로부터 객체를 탐지합니다. LibreOpenVocab을 통해 Grounding DINO, OWLv2,
  OMDet-Turbo 또는 OV-DEIM을 불러오고 실행 시 클래스를 설정합니다.
lead: >-
  오픈 보캐블러리 탐지는 체크포인트의 고정된 클래스 목록을 호출 시 선택한 단어로 대체합니다. LibreYOLO에서는 별도의 작업이 아니며,
  LibreOpenVocab 팩토리를 통해 로드되는 별도의 모델 계층이 제공하는 탐지 작업입니다.
keywords:
  - 개방형 어휘 탐지
  - 제로샷 객체 탐지
  - 개방 집합 탐지
  - 접지 다이노 파이썬
  - 올브이2
  - 옴뎃 터보
  - 텍스트 프롬프트 탐지
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        print(result.names)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 어휘를 바꾸다
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")

        # set_classes는 지속적입니다: 다음 호출 때까지 유지됩니다.
        # 레이블은 소문자로 변환하고 관사를 제거한 후에는 고유해야 합니다.
        model.set_classes(["a red backpack", "traffic cone"])
        result = model.predict(SAMPLE_IMAGE)

        model.set_classes(["bicycle wheel"])
        result = model.predict(SAMPLE_IMAGE)
    - label: Grounding DINO 텍스트 임계값
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf는 박스 점수로 필터링하고, text_threshold는 디코딩된 문구로 필터링합니다
        # 토큰 점수. 설정하지 않으면 둘 다 기본값은 0.25입니다. 오직 그라운딩만
        # DINO는 text_threshold를 허용합니다; 다른 것들은 오류를 발생시킵니다.
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
source_hash: 17197cf4d80f3d6f
---

## 정의

개방형 어휘 탐지는 일반적인 탐지 `Results`: 박스, 신뢰도 및 클래스 인덱스를 반환하며, `result.names`는 해당 인덱스를 요청한 문자열로 다시 매핑합니다. 바뀌는 부분은 클래스 목록이 어디서 오는가입니다. 일반적인 탐지기는 고정된 범주의 집합을 기준으로 학습되며 그 밖의 범주는 절대 내보낼 수 없습니다. 이러한 모델은 추론 시 텍스트로 어휘를 사용하므로, `set_classes(["forklift", "safety cone"])`만으로도 그것을 클래스가 되게 할 수 있습니다.

LibreYOLO에는 `open-vocabulary` 작업 키가 없습니다. 이 모델들은 다른 탐지기와 마찬가지로 `SUPPORTED_TASKS = ("detect",)`를 선언합니다. 그들을 구분짓는 것은 로딩 경로입니다: 이들은 LibreYOLO 상태-딕셔너리 체크포인트가 아니라 Hugging Face 스냅샷이므로, `LibreYOLO()` 팩토리에는 포함되지 않고 대신 `LibreOpenVocab()`를 통해 구성됩니다. 그 팩토리는 `LibreSAM()`와 `LibreVLM()`의 형제이며 `LibreYOLO()`의 대체물이 아닙니다.

점수는 실제 탐지 점수이며, 사후에 파싱된 생성된 캡션이 아닙니다. 각 계열는 모든 프롬프트의 텍스트 임베딩과 비교하여 이미지 영역을 점수화합니다.

## 모델들

네 개의 가문이 이 계층을 구성하며, 모두 예측만 합니다. `LibreOpenVocab`를 통해 별칭으로 그 중 하나를 불러오십시오.

IDEA Research의 [Grounding DINO](/docs/models/grounding-dino), `t` 및 `b` 크기에서 제공됩니다. 이는 기본 계층이며, 디코딩된 문구의 토큰 점수에 대한 두 번째 컷오프인 `text_threshold`를 허용하는 유일한 계열입니다.

[OWLv2](/docs/models/owlv2), 구글 리서치에서 개발, `b16` 및 `l14` 크기로 제공됩니다. CLIP 스타일 인코더에서 생성된 텍스트 임베딩과 이미지 영역을 비교 평가합니다.

[OMDet-Turbo](/docs/models/omdet-turbo), Om AI Lab에서 나온 것으로, `t` 크기 하나입니다. 이것은 언어 작업 프롬프트에서 클래스 임베딩을 분리하며, 여기서 유일하게 자체 후처리 과정에서 겹치는 박스를 억제하는 계열이어서 `iou=`가 유지됩니다.

[OV-DEIM](/docs/models/ov-deim), `s`, `m` 및 `l` 크기에서, 번들로 제공된 MobileCLIP 텍스트 타워의 텍스트 임베딩과 디코더 쿼리를 매칭하는 DETR 스타일 디텍터입니다. 이는 일대일 매칭이며 Top-K 선택이 수행되므로 어디에서도 NMS가 실행되지 않습니다.

OV-DEIM의 가중치는 이 계층에서 제한된 경우입니다. 탐지기 가중치는 CC BY-NC 4.0, 비상업용입니다. 번들로 제공되는 텍스트 타워는 Apple의 머신러닝 연구 모델 라이선스 하에 있으며, 연구용으로만 사용 가능합니다. `l` 체크포인트는 Meta의 DINOv3 라이선스 하에 DINOv3-S 백본 파인튜닝을 추가합니다. 세 가지 라이선스 텍스트 모두 가중치 저장소 내에 포함되어 있으며, 라이브러리는 모델이 구축되기 전에 가중치를 확인할 때 동일한 요약을 기록합니다. 배포하기 전에 [OV-DEIM](/docs/models/ov-deim)를 읽으십시오.

그 단계에는 하나가 더 필요합니다:

```bash
pip install "libreyolo[openvocab]"
```

그것은 세 개의 래핑된 계열에 대한 `transformers` 및 `timm`과, OV-DEIM이 네이티브 포트로 필요로 하는 `huggingface_hub`, `safetensors`, `regex` 및 `ftfy` 패키지를 포함합니다.

두 번째 계층도 텍스트 어휘를 사용합니다: `LibreVLM()`는 [Qwen3-VL](/docs/models/qwen3-vl)과 [Florence-2](/docs/models/florence-2)와 같은 생성형 비전-언어 모델을 로드하고, 그들의 출력을 동일한 `Results`로 변환합니다. 그것은 `set_classes()` 표면을 공유합니다. 차이점은 박스를 생성하는 방식입니다: 이 페이지의 계열들은 점수를 직접 출력하는 판별 탐지기인 반면, VLM 계층은 박스를 생성합니다.

## 예측

<code-tabs name="predict" />

`set_classes()`는 비어 있지 않은 레이블 문자열 목록을 가져와 다음 호출될 때까지 유지합니다. 레이블은 소문자로 변환하고 앞부분의 관사를 제거한 후 고유해야 하므로 `"a bus"`와 `"bus"`는 하나의 어휘에서 공존할 수 없습니다. 여러 단어로 이루어진 구문도 다른 레이블과 마찬가지로 레이블이며, 각 계열은 토크나이즈하기 전에 목록을 자신의 텍스트 입력으로 변환하므로 `"traffic cone"`는 `"cone"`와 다른 쿼리입니다.

세 가지 예측 인자는 여기에서 네이티브 검출기에서와 다르게 작동합니다. `imgsz=`는 해당 계열에 대해 프로세서가 크기 조정을 소유하고 있으므로 거부됩니다. `augment=True`는 시험 시간 증강이 계층 범위를 벗어나므로 거부됩니다. `iou=`는 프로세서가 자체 억제를 실행하는 계열에만 적용됩니다; 억제되는 것이 없으면, 이를 통과하는 것은 경고를 발생시키고 무시됩니다.

설정하지 않으면 `conf`는 `predict()`의 일반적인 0.25 대신 로드된 계열 자체의 기본값을 사용하며, 그 기본값은 계층 전체에서 동일하지 않습니다. 동일한 이미지에서 두 계열를 비교할 때는 명시적으로 설정하십시오.

`track()`가 계층 전반에 걸쳐 증가합니다. 대신 프레임별로 `predict()`를 실행하십시오. 소스, 스트리밍 및 결과 처리에 대해서는 [prediction](/docs/predict)를 참조하십시오.

## 학습

이 계층에는 LibreYOLO 안에서 학습하는 계열이 없습니다. `train()`가 올립니다: 업스트림을 파인튜닝하고 결과 가중치를 로드합니다. `set_classes()`에 전달된 어휘만이 로드된 모델이 탐지하는 것을 변경하는 유일한 설정입니다.

## 검증

이 계층에는 검증기가 없으며, `val()`가 발생합니다. 오픈 보캐뷸러리 검증에는 전용 검증기가 필요합니다. 표준 검출 검증기는 이미지 텐서를 모델에 바로 공급하지만, 이러한 계열은 함께 구축된 텍스트 조건 입력이 필요하기 때문입니다.

## 내보내기

내보내기는 해당 등급의 범위를 벗어나며 `export()`가 발생합니다. 이 모델들은 PyTorch에서 `predict()`를 통해 실행됩니다.
