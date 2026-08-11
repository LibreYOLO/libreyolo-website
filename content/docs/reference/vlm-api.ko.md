---
title: 비전-언어 API
seo_title: 'LibreVLM API: 별칭, set_classes 및 채팅'
description: >-
  LibreVLM 팩토리, 모든 모델 별칭, 고정 set_classes 어휘, set_task, 채팅 이스케이프 해치 및 자신감이 자리
  표시자인 이유
lead: >-
  LibreVLM은 생성형 비전-언어 모델을 로드하고 객체 탐지기로 작동시킵니다. 클래스 목록은 고정된 헤드가 아닌 프롬프트이며, 모델은 다른
  모든 계열이 반환하는 동일한 결과를 반환합니다.
keywords:
  - LibreVLM
  - 비전 언어 모델 탐지
  - Qwen3-VL
  - LFM2-VL
  - InternVL3
  - SmolVLM2
  - Florence-2
  - libreyolo 채팅
last_verified: 1.5.0
verification: >-
  별칭은 libreyolo/models/vlm/__init__.py에서 읽고; 저장소, 크기 및 작업 목록은
  libreyolo/models/vlm/ 아래 계열 모듈과 libreyolo/models/sensenova/model.py에서 가져오며; 호출
  규칙과 예외는 libreyolo/models/vlm/base.py에서 가져오며, 모두 v1.5.0 기준입니다.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[vlm]'
  usage:
    - label: 열린 어휘 탐지
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        model.set_classes(["person", "skateboard"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
    - label: 자유 형식 질문하기
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        print(model.chat(SAMPLE_IMAGE, "How many people are in this image?"))
source_hash: 57ddac08bc4d4e05
---

## 설치

그 티어에는 `vlm` 추가가 필요합니다.

<code-tabs name="install" />

## 팩토리

```python
LibreVLM(model: str = "qwen3-vl-4b", **kwargs) -> LibreVLMModel
```

`model`는 경로가 아니라 별칭입니다. `**kwargs`는 계열 생성자에 도달하며, 이는 `device`, `names`(초기 어휘, load 후 `set_classes`를 호출하는 것과 동일), `prompt`(탐지 프롬프트 재정의) 및 `max_new_tokens`를 받습니다. 알 수 없는 별칭은 `ValueError`를 발생시켜 모든 별칭을 나열합니다.

<code-tabs name="usage" />

## 별칭

| 계열 | 별칭 | 크기 | 가중치 |
|---|---|---|---|
| Qwen3-VL | `qwen3-vl`, `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` | `2b`, `4b`, `8b` | `Qwen/Qwen3-VL-2B-Instruct`, `-4B-`, `-8B-` |
| LFM2-VL | `lfm2-vl`, `lfm2-vl-450m`, `lfm2-vl-1.6b` | `450m`, `1.6b` | `LiquidAI/LFM2.5-VL-450M`, `-1.6B` |
| InternVL3 | `internvl3`, `internvl3-1b`, `internvl3-2b`, `internvl3-8b` | `1b`, `2b`, `8b` | `OpenGVLab/InternVL3-1B-hf`, `-2B-hf`, `-8B-hf` |
| SmolVLM2 | `smolvlm2`, `smolvlm2-2.2b`, `smolvlm2-500m` | `2.2b`, `500m` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `SmolVLM2-500M-Video-Instruct` |
| Florence-2 | `florence-2`, `florence2`, `florence-2-base`, `florence-2-large` | `base`, `large` | `florence-community/Florence-2-base`, `-large` |
| Kosmos-2 | `kosmos-2`, `kosmos2` | `224` | `microsoft/kosmos-2-patch14-224` |
| LocateAnything | `locate-anything`, `locateanything`, `locate-anything-3b`, `locateanything-3b` | `3b` | `nvidia/LocateAnything-3B` |
| SenseNova-Vision | `sensenova-vision`, `sensenova-vision-7b`, `sensenovavision` | `7b` | `LibreYOLO/SenseNovaVision7b` |
| LibreMODUS | `libremodus`, `libremodus-14b-a7b`, `modus`, `modus-14b-a7b` | `14b-a7b` | 상위 스냅샷 고정 |

기본 별칭은 `qwen3-vl-4b`입니다. 각 패밀리의 기본 별칭에 대한 크기는 먼저 나열된 것입니다: `qwen3-vl`는 `4b`로 해결되고, `lfm2-vl`는 `450m`로, `internvl3`는 `2b`로, `smolvlm2`는 `2.2b`로, `florence-2`는 `base`로 해결됩니다.

`LibreVLM`, `LibreLFM2VL`, `LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`, `LibreFlorence2`, `LibreKosmos2`, `LibreLocateAnything` 및 `LibreMODUS`(또한 `LibreModus`로 표기됨)는 패키지 수준에서 내보내집니다.

## 작업

대부분의 패밀리는 `detect`만 제공합니다. 두 패밀리는 더 많은 서비스를 제공합니다:

| 패밀리 | 지원되는 작업 |
|---|---|
| LocateAnything | `detect`, `point` |
| SenseNova-Vision | `detect`, `segment`, `panoptic`, `pose`, `point`, `depth`, `ocr` |

작업이 체크포인트에 내장된 것이 아니라 프롬프트 기반이기 때문에, 로드된 모델에서 전환할 수 있습니다:

```python
model.set_task(task: str) -> LibreVLMModel
```

작업은 패밀리에서 지원되는 목록에 대해 검증되며, 이후 `predict()` 및 `track()` 호출에서도 유지되며, 모델이 반환되어 호출을 연결할 수 있습니다.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreVLMModel
```

오픈 보캐뷸러리를 설정합니다. 어떤 단어든 사용할 수 있으며, 모델이 고정 헤드에 제한되지 않고 프롬프트로 사용되기 때문입니다. 목록은 비어 있지 않아야 하고, 대소문자를 구분하지 않고 비교했을 때 항목이 고유해야 합니다. 단일 문자열을 전달하면 `TypeError` 오류가 발생하는데, 이는 문자열이 한 글자 클래스들로 나열되기 때문입니다. 어휘는 지속성을 갖습니다: 로드 후 한 번 설정하면 다시 설정될 때까지 유지됩니다.

## chat

```python
model.chat(image, prompt, max_new_tokens=None, color_format="auto") -> str
```

원시 다중 모달 생성: 이미지와 프롬프트 입력, 디코딩된 텍스트 출력, 그대로. 이것은 감지 편의를 위한 탈출 장치로, 자유 형식 질문, 계수, 또는 감지 래퍼가 다루지 못하는 출력 형식에 사용됩니다. `max_new_tokens`는 기본 클래스에서 1024인 계열 `MAX_NEW_TOKENS`로 대체됩니다. 디코딩은 약한 반복 패널티를 가진 탐욕적 방식입니다.

## 신뢰도

생성된 출력에는 보정된 박스별 신뢰도가 없습니다. 이 버전은 상수 자리 표시자를 할당하여 `predict`, 그리기 및 `track`가 작동하게 하며, 이는 `conf=` 필터링과 mAP를 의미 있는 것이 아닌 부드럽게 만듭니다. 또한 이것이 `val()`가 발생하는 이유입니다: 자리 표시자 점수로 COCO mAP를 계산하면 오도될 수 있습니다.

## 예측 및 추적

표준 예측(surface) 적용되며, `track()`가 작동하므로 VLM 탐지기는 다른 어떤 계열과 동일한 파이프라인에 들어갑니다. 두 가지 클래스 수준 정책은 컨볼루션 탐지기와 다릅니다: 테스트 시 증강(test-time augmentation)은 비활성화되어 있는데, 고정 해상도 생성기에는 다중 스케일 증강이 의미가 없기 때문이며, 배치 예측(batched predict)은 꺼져 있습니다. 이는 생성이 자기회귀적(autogressive)이고, 전처리가 스택 가능한 이미지 텐서가 아니라 텍스트와 이미지 인코딩을 반환하기 때문입니다.

## 지원되지 않음

`train()`, `val()` 및 `export()`는 `NotImplementedError`를 발생시킵니다. 업스트림(fine-tune upstream) 후 결과 가중치를 로드하십시오.

## 원격 코드

모든 배송된 패밀리 로드는 네이티브 모델 클래스를 통해 이루어지므로, LibreYOLO는 기본적으로 서드파티 저장소 코드를 실행하지 않습니다. 실제로 필요로 하는 패밀리는 명시적으로 옵트인하고 스냅샷 리비전을 고정해야 합니다; LocateAnything가 유일하게 그렇게 하며, 커밋 `c32291ca5e996f5a7a485845b4f57a233936bba0`에 고정되어 있습니다.

LibreMODUS는 체크포인트 스키마에 대한 명시적 예외입니다: 그 별칭은 LibreYOLO `.pt`가 아니라 상류 파일의 고정된 디렉토리를 참조하며, LibreYOLO는 여기에 v1.0 메타데이터를 추가하거나 재배포하지 않습니다.
