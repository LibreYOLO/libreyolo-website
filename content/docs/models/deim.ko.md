---
title: DEIM
families:
  - deim
seo_title: LibreYOLO의 DEIM과 DEIMv2
description: >-
  LibreYOLO에서 DEIM과 DEIMv2로 객체를 탐지합니다. 매개변수 50만 개 크기부터 설치, 예측, 학습, 검증, 내보내기를
  수행합니다.
lead: >-
  조밀 일대일 매칭으로 학습하는 detection transformer로 기반이 되는 DETR 레시피보다 훨씬 적은 epoch에 수렴합니다.
  LibreYOLO는 두 버전을 제공하며 불러온 체크포인트로 구분합니다.
keywords:
  - DEIM 사용법
  - DEIMv2
  - DINOv3
  - detection transformer
  - DETR
  - 객체 탐지
  - 실시간 탐지
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDEIMn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDEIMn.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 동영상
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 버전은 파일명에 포함되고 팩토리는 체크포인트에 따라 라우팅하므로
        # 두 버전을 같은 방식으로 불러옵니다.
        model = LibreYOLO("LibreDEIMv2pico.pt")

        # 라이브러리가 받는 모든 소스: 파일, 폴더, URL, 웹캠 인덱스,
        # RTSP 스트림 또는 .streams 목록
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # coco128.yaml은 처음 사용할 때 이미지 128개 샘플을 다운로드합니다.
        # 실제 실행에서는 `data`를 자체 데이터셋 YAML로 지정합니다.
        model.train(data="coco128.yaml", epochs=50, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 batch=8 lr0=1e-4
    - label: DEIMv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 미설정 시 epochs, batch, imgsz, lr0는 불러온 크기의 공개 레시피에서 가져옵니다.
        model = LibreYOLO("LibreDEIMv2pico.pt")
        model.train(data="coco128.yaml", epochs=50)
    - label: LoRA
      language: python
      code: |
        # lora extra가 필요합니다: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: 다중 GPU
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # val()은 객체가 아닌 일반 dict를 반환합니다.
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDEIMn.pt data=coco128.yaml
    - label: COCO 기준 평가
      language: bash
      code: |
        # coco-val-only.yaml은 val2017 이미지 5000개를 가져오고 학습 집합은 생략합니다.
        # 내장 다운로드 스크립트가 있으므로 데이터셋이 로컬에 없다면 명시적 권한이 필요합니다.
        libreyolo val model=LibreDEIMn.pt data=coco-val-only.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        # onnx extra가 필요합니다: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDEIMn.pt format=onnx
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreDEIMn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 6edaac5f05abaabe
---

## 설치

두 버전 모두 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

단, `lora=True`를 사용하는 어댑터 미세 조정에는 `lora` extra가 필요합니다.

```bash
pip install "libreyolo[lora]"
```

## 예측

처음 사용할 때 Hugging Face에서 가중치를 다운로드해 로컬에 캐시합니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 제품군이 반환하는 것과 같으므로 탐지기를 바꾸려면 한 줄만 변경하면 됩니다. `conf`와 `max_det`은 쿼리와 클래스의 top-k 디코딩을 필터링합니다. 조정할 NMS 단계가 없으며 `iou`는 허용되지만 사용되지 않습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

버전 1은 같은 입력 크기의 다섯 가지 크기를 제공합니다. 버전 2는 이 다섯 이름을 유지하고 더 작은 `atto`, `femto`, `pico` 세 가지를 추가합니다. 앞의 두 크기는 나머지보다 낮은 기본 입력 크기를 사용합니다. 따라서 다섯 크기 코드는 두 버전에 모두 존재하지만 서로 다른 모델을 나타내며 버전은 체크포인트 파일명에 기록됩니다.

<benchmark-table task="detect" />

<va-embed />

버전 1은 D-FINE 구조를 유지하고 분류 목적 함수를 조밀 일대일 레시피의 matchability-aware 손실로 교체합니다. 따라서 두 제품군은 거의 모든 state dict 키를 공유하며 체크포인트 메타데이터로 구분합니다. 버전 2는 해당 학습 계약을 유지하면서 백본을 혼합합니다. `s` 미만은 HGNetv2, `s` 이상은 공간 튜닝 어댑터가 있는 DINOv3 vision transformer를 사용합니다. 이 백본 때문에 네 체크포인트에 두 번째 라이선스가 적용되므로 배포 전에 [라이선스](#licensing)를 읽습니다.

## 학습

학습은 공개된 체크포인트에서 시작합니다. `pretrained`는 학습기에 전달되지 않습니다. 버전 1은 알 수 없는 키라고 경고하고 무시하며 버전 2는 제거합니다. 어느 버전도 무작위로 초기화된 모델을 제공하지 않습니다.

<code-tabs name="train" />

버전 1에서는 `lr0`를 직접 전달합니다. Python `train()` 시그니처 기본값은 공개 COCO 레시피의 `4e-4`이지만 제품군 학습 구성에는 미세 조정 기본값으로 `1e-4`가 들어 있으며 CLI에서 인수를 생략하면 이 낮은 값을 사용합니다. 구성에는 근거도 기록되어 있습니다. 실제 미세 조정에 사용하는 배치 크기의 소규모 데이터셋에서 COCO 학습률은 전이 성능을 측정 가능하게 낮췄습니다.

버전 2는 기본값을 자체적으로 결정합니다. `epochs`, `batch`, `imgsz`, `lr0`를 설정하지 않으면 불러온 크기의 공개 레시피에서 각각 읽으므로 작은 크기는 별도 설정 없이 자체 입력 해상도에서 학습하며 전달한 값은 레시피를 재정의합니다. `imgsz`에는 제약이 있습니다. 양의 32 배수여야 하며 그렇지 않으면 실행 전에 예외가 발생합니다.

데이터셋, 증강, 다중 GPU, 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 학습에 사용한 형식의 데이터셋을 대상으로 측정한 정밀도, 재현율, mAP 50, mAP 50-95를 포함하는 `metrics/` 키 사전을 반환합니다.

<code-tabs name="val" />

위 벤치마크 표의 행은 LibreYOLO 벤치마크 도구에서 나옵니다. 표 아래 참고 사항에는 수치를 만든 데이터셋과 실행 기록 링크가 있습니다.

## 내보내기

<export-matrix />

매트릭스는 두 버전을 한 페이지에서 다룹니다. 형식 지원이 다르면 셀은 둘 중 더 약한 수준을 표시하므로 불러온 버전에 대한 지원을 과장하지 않습니다.

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다.

<code-tabs name="export" />

## 체크포인트

이 제품군에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box>
S 이상 DEIMv2 크기 네 개는 DINOv3에서 백본을 가져오므로 가중치 저장소에 Apache-2.0과 Meta의 DINOv3 License가 모두 적용됩니다. LibreYOLO도 동일한 계약 아래 DINOv3 백본 소스를 제공합니다. S 미만의 모든 DEIMv2 크기를 포함한 나머지 제품군은 Apache-2.0만 적용됩니다.
</provenance-box>

## 인용

<citation-block />

DEIMv2는 별도 논문이며 [github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#5-citation)에 자체 인용 블록이 있습니다. 버전 2 체크포인트를 사용했다면 이를 인용합니다.
