---
title: OV-DEIM
families:
  - ov_deim
seo_title: 'LibreYOLO의 OV-DEIM: 오픈 보캐뷸러리 탐지'
description: >-
  LibreYOLO에서 OV-DEIM으로 실시간 DETR 방식 오픈 보캐뷸러리 탐지를 수행합니다. openvocab extra를 설치하고 자유
  텍스트 어휘로 예측합니다.
lead: >-
  OV-DEIM은 디코더 쿼리를 포함된 MobileCLIP 텍스트 타워의 텍스트 임베딩과 일치시키는 DETR 방식 오픈 보캐뷸러리 객체
  탐지기입니다. LibreYOLO는 오픈 보캐뷸러리 탐지기 티어의 예측 전용 계열으로 이를 네이티브 이식합니다.
keywords:
  - OV-DEIM 사용법
  - DEIMv2
  - 오픈 보캐뷸러리 객체 탐지
  - 실시간 객체 탐지
  - 제로샷 탐지
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-s")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 어휘 교체
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-l")
        model.set_classes(["traffic light", "bicycle"])
        first = model.predict(SAMPLE_IMAGE, conf=0.3)

        # set_classes()를 두 번째로 호출하면 어휘를 완전히 교체하고 텍스트 타워로
        # 다시 임베딩합니다. 빈 결과는 오류가 아니라 유효한 결과입니다.
        model.set_classes(["giraffe"])
        second = model.predict(SAMPLE_IMAGE, conf=0.5)
        print(second.names, len(second))
source_hash: 0c295f555a9eb303
---

## 설치

OV-DEIM은 LibreYOLO의 오픈 보캐뷸러리 탐지기 티어를 통해 불러오며 `openvocab` extra가 필요합니다.

```bash
pip install "libreyolo[openvocab]"
```

이 티어의 다른 계열과 달리 OV-DEIM은 `transformers` 래퍼가 아닌 LibreYOLO 네이티브 이식입니다. 따라서 해당 모델의 `transformers` 모델 클래스는 없지만, 같은 extra가 예측 시 필요한 `huggingface_hub`, `safetensors`, `regex`, `ftfy` 패키지를 설치합니다.

## 예측

OV-DEIM은 LibreYOLO가 `LibreYOLO()`를 통해 불러오는 체크포인트가 아닙니다. 형제 팩토리인 `LibreOpenVocab`을 통해 불러오며 처음 사용할 때 Hugging Face 스냅샷을 다운로드해 `weights/` 아래에 캐시합니다.

<code-tabs name="predict" />

`set_classes()`는 지속되는 텍스트 어휘를 설정합니다. 다시 호출하면 목록을 완전히 교체할 수 있고, 호출하지 않으면 기본 COCO-80 레이블을 유지합니다. 빈 결과는 오류가 아니라 유효한 결과입니다. 각 디코더 쿼리는 포함된 MobileCLIP-B(LT) 텍스트 타워의 텍스트 임베딩과 코사인 유사도로 점수가 매겨집니다. 설정된 어휘마다 온라인으로 계산되고 어휘가 바뀔 때까지 캐시되므로 사전 계산된 임베딩 파일 없이 임의 프롬프트를 사용할 수 있습니다.

OV-DEIM에는 텍스트 토큰 임곗값이 없습니다. `conf`만 탐지를 필터링하며 `text_threshold`를 전달하면 예외가 발생합니다. 일치는 일대일 top-K 선택이므로 비최대 억제를 실행하지 않습니다. `iou`는 API 호환성을 위해 허용되지만 경고만 표시하고 아무 작업도 하지 않습니다. `imgsz`와 `augment=True`는 즉시 거부됩니다. 모델이 고정된 레터박스 입력을 직접 관리하며 테스트 시간 증강은 이 티어의 범위에 포함되지 않습니다. 이미지 하나에 `predict()`를 호출하면 목록이 아닌 `Results` 하나를 반환합니다. 디렉터리나 이미지 목록을 전달하거나 동영상 소스에 `stream=True`를 사용하면 여러 결과를 얻습니다. 이 계열에는 CLI 경로가 없습니다. `libreyolo predict`는 `LibreYOLO()`를 통해 `.pt` 체크포인트만 불러오므로 `LibreOpenVocab` 계열은 Python에서 실행합니다. 소스 유형과 스트리밍은 [예측](/docs/predict)을 참조합니다.

`predict()`를 호출할 때마다 포함된 MobileCLIP-B(LT) 텍스트 타워도 실행되어 현재 어휘를 임베딩합니다. 이에 따라 추가되는 약관은 라이선스를 참조합니다.

## 변형

체크포인트는 `s`, `m`, `l` 세 가지입니다. 크기를 지정하지 않으면 `s`가 이 티어의 기본값입니다. 이 티어의 다른 계열과 달리 OV-DEIM은 `transformers` 래퍼가 아닌 네이티브 이식입니다. LibreYOLO는 업스트림 코드와 같은 Apache-2.0 라이선스로 탐지기 모듈을 벤더링하고 DEIMv2 계열용으로 이미 구축된 DINOv3 백본 어댑터를 재사용합니다. `l` 체크포인트의 백본은 DINOv3-S 파인튜닝 모델이며 Meta의 DINOv3 License가 별도로 적용됩니다. 아직 이 계열의 정확도나 지연 시간 수치는 공개되지 않았습니다.

학습, 데이터셋 검증, 내보내기는 모두 이 티어의 범위에 포함되지 않습니다. `train()`, `val()`, `export()`는 조건 없이 `NotImplementedError`를 발생시킵니다. 공개된 체크포인트를 감싼 예측 전용 래퍼입니다.

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box>

OV-DEIM의 모든 예측 호출에는 세 가지 업스트림 라이선스가 추가됩니다. 탐지기 가중치에는 OV-DEIM의 CC BY-NC 4.0, 온라인 텍스트 타워에는 Apple의 Machine Learning Research Model license(연구 용도로만 사용), `l` 체크포인트에는 Meta의 DINOv3 License가 적용되는 DINOv3-S 백본 파인튜닝 모델이 포함됩니다. 세 라이선스 전문은 모두 LibreYOLO 가중치 저장소에 들어 있습니다.

</provenance-box>

## 인용

<citation-block />
