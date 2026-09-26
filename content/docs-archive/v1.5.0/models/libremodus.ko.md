---
title: LibreMODUS
families:
  - libremodus
seo_title: 'LibreYOLO의 LibreMODUS: any-to-any 이미지 분석'
description: >-
  LibreYOLO에서 LibreMODUS를 사용해 깊이, 노멀, 엣지, 탐지를 수행하고 any2any()로 이들을 조합합니다. 추론 전용이며
  가중치는 EPFL-VILAB에서 불러옵니다.
lead: >-
  LibreMODUS는 한 이미지에서 파생된 입력을 다른 출력으로 변환하는 any-to-any 모델인 MODUS 14B-A7B 체크포인트의
  추론 전용 통합입니다. RGB를 입력해 깊이를 출력하고, 깊이를 입력해 노멀을 출력하며, 이들 중 하나와 문구를 입력해 박스를 출력할 수
  있습니다. LibreYOLO는 표준 predict API로 네 가지 작업을 지원하며 any2any()로 더 폭넓은 작업을 제공합니다.
keywords:
  - LibreMODUS 사용법
  - MODUS 모델
  - any-to-any 이미지 분석
  - 깊이 추정
  - 표면 노멀
  - 엣지 탐지
  - 문구 기반 탐지
  - EPFL VILAB
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(size="14b-a7b", task="normal")
        result = model.predict("room.jpg")
        normals = result.normal_map.data

        model.set_task("edge")
        result = model.predict("room.jpg")
        edges = result.edges.data

        # 사용자 정의 어휘가 없으면 detect가 체크포인트의 COCO
        # 레이블 토큰을 연속된 COCO-80 클래스 ID로 디코딩합니다.
        model.set_task("detect")
        result = model.predict("street.jpg")
        print(result.boxes.xyxy)
    - label: 문구 그라운딩
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(task="detect")
        # set_classes()는 탐지를 문구 그라운딩으로 전환합니다. 각 문구는
        # 독립적으로 실행되고 같은 Boxes 계약을 통해 반환됩니다.
        model.set_classes(["red bus", "cyclist"])
        result = model.predict("street.jpg", conf=0.2)
        print(result.boxes.xyxy, result.boxes.cls)
    - label: any2any()
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS()

        # 이미지에서 파생된 입력(rgb, depth, normal, canny/edge)을 1~3개 사용하고
        # 선택적 보조 텍스트를 더해 하나의 대상으로 조합합니다.
        result = model.any2any(
            inputs={"rgb": "room.jpg"},
            target="normal",
            steps=10,
            cfg=2.0,
            seed=0,
        )
        normals = result.normal_map.data

        # any2any()를 통한 그라운딩에는 문구를 지정하는 텍스트 입력이 필요합니다.
        result = model.any2any(
            {"rgb": "street.jpg", "text": "red bus"},
            target="grounding",
        )
        print(result.boxes.xyxy)
source_hash: 7386886d4c36ea9a
---

## 설치

LibreMODUS에는 이 체크포인트가 요구하는 대형 모델 디스패치를 위해 `accelerate`를 가져오는 전용 extra가 필요합니다.

```bash
pip install "libreyolo[modus]"
```

LibreYOLO는 MODUS 가중치를 재배포하거나 미러링하지 않습니다. 기본적으로 `LibreMODUS` 모델을 불러오면 고정된 Hugging Face 리비전의 필수 파일을 `EPFL-VILAB/MODUS`에서 직접 다운로드합니다. 업스트림 호스팅 게이트가 일시적으로 열려 있어도 새로 다운로드할 때는 항상 인증된 개인 Hugging Face 계정이 필요합니다. 업스트림 약관을 검토하고 동의한 다음 인증합니다.

```bash
hf auth login
```

```python
from libreyolo import LibreMODUS

model = LibreMODUS(token="hf_...")
```

네트워크 요청을 완전히 피하려면 이미 보유한 스냅샷을 지정합니다.

```python
model = LibreMODUS(checkpoint_path="/models/MODUS")
```

해당 디렉터리에는 `model.safetensors`, `ae.safetensors`, `llm_config.json`, `vit_config.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`가 있어야 합니다. 체크포인트 약관이 허용하는 범위는 아래 라이선스를 참조합니다.

## 예측

<code-tabs name="predict" />

표준 작업 API는 네 가지 작업을 지원하며 각 작업은 하나의 MODUS 대상에 매핑됩니다. `depth`는 상대 깊이(`result.depth_map`), `normal`은 표면 노멀(`result.normal_map`), `edge`는 Canny 방식 엣지(`result.edges`)에 해당합니다. `detect`는 COCO-80 박스(`result.boxes`)에 해당하지만 `set_classes()`를 사용하면 문구 그라운딩으로 전환됩니다. `set_task()`를 사용하면 불러온 동일한 모델에서 작업을 전환할 수 있습니다. 공개된 레시피는 텍스트 가이던스 4.0과 이미지 가이던스 2.0으로 플로 샘플링 10단계를 사용합니다. 생성 시 `inference_steps=`, `inference_cfg=`, `inference_image_cfg=`로 재정의할 수 있습니다.

`any2any()`는 더 넓은 공개 분석 기능에 접근합니다. 이미지에서 파생된 입력(`rgb`, `depth`, `normal`, `canny`/`edge`)을 1~3개 사용하고 선택적 보조 텍스트를 더해 깊이, 노멀, 엣지, SAM 파생 엣지, COCO 탐지 또는 문구 그라운딩 중 하나로 조합합니다. 이미지에서 파생된 모든 입력은 정렬된 동일한 캔버스를 나타내야 합니다. LibreMODUS는 입력 크기를 개별적으로 조정하지 않고 너비와 높이가 일치하지 않으면 거부합니다. `chain=(...)`은 중간 대상을 생성해 동일한 컨텍스트에 다시 공급하며 체크포인트의 세 조건 학습 예산 안에서 작동합니다. `verify=N`(N >= 2)은 N개의 후보를 생성하고 제한된 자기 일관성 검사에서 가장 높은 점수를 받은 후보를 유지합니다. 이 점수는 `result.verification_score`로 제공됩니다.

`dtype="bf16"`(기본값)은 공개된 체크포인트 정밀도와 일치합니다. `dtype="fp8"`은 적용 가능한 디코더 트렁크 선형 가중치를 출력 채널별 스케일을 사용하는 E4M3 형식으로 저장하고, `~/.cache/libreyolo/modus/fp8`의 로컬 캐시로 한 번 변환한 다음 행렬 곱셈마다 입력 dtype으로 역양자화합니다. 따라서 활성화 수준의 정확도를 낮추는 대신 메모리를 절약합니다.

`train()`, `val()`, `export()`는 모두 예외를 발생시킵니다. LibreMODUS는 추론 전용이고 데이터셋 검증을 제공하지 않으며 ONNX, TensorRT 또는 TFLite 내보내기 경로가 없습니다. 배치 `predict()`와 테스트 시간 증강도 지원하지 않으며 각 호출은 이미지 하나를 처리합니다.

## 라이선스

<provenance-box>

LibreYOLO는 자체 Hugging Face 조직을 포함해 어디에도 MODUS 체크포인트를 호스팅하거나 미러링하지 않습니다. 체크포인트를 불러오면 항상 고정된 리비전을 EPFL-VILAB/MODUS에서 직접 가져오거나 `checkpoint_path`에 이미 저장된 스냅샷을 읽습니다.

</provenance-box>

## 인용

<citation-block />
