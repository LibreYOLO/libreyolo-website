---
title: RT-DETR
families:
  - rtdetr
seo_title: 'LibreYOLO의 RT-DETR, RT-DETRv2 및 RT-DETRv4'
description: >-
  LibreYOLO에서 객체 검출을 위해 RT-DETR, RT-DETRv2 및 RT-DETRv4를 사용하고, RT-DETRv2에서는 회전
  바운딩 박스도 사용하십시오. Apache-2.0 가중치로 설치, 예측, 학습, 검증 및 내보내기를 수행하십시오.
lead: >-
  실시간 추론을 위해 구축된 검출 트랜스포머: 이는 촘촘한 그리드 대신 고정된 쿼리 세트를 디코딩하기 때문에 NMS를 실행하지 않습니다.
  LibreYOLO는 불러오는 체크포인트에 따라 구분되는 세 가지 버전을 제공하며, 버전 2는 방향이 있는 박스도 제공합니다.
keywords:
  - RT-DETR
  - RT-DETRv2
  - RT-DETRv4
  - 실시간 탐지 변환기
  - DETR
  - 객체 탐지
  - 방향성 바운딩 박스 탐지
  - OBB
  - 도타
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTDETRr18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRr18.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 비디오
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 버전은 파일 이름의 일부이며, 팩토리는 그 위에 경로를 지정합니다
        # 체크포인트, 그래서 세 개 모두 같은 방식으로 로드됩니다.
        model = LibreYOLO("LibreRTDETRv4s.pt")

        # 라이브러리가 허용하는 모든 소스: 파일, 폴더, URL, 웹캠 인덱스,
        # RTSP 스트림 또는 .streams 목록
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
    - label: 방향이 지정된 상자
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 버전 2만 해당. -obb 접미사는 작업과 체크포인트를 선택합니다.
        # 자체 텐서에서 방향이 지정된 것으로 인식되므로 작업 인수가 없습니다
        # 필요합니다. 이 가중치는 DOTA v1.0, 1024 px에서 15개의 항공 클래스입니다.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)     # (N, 5): cx, cy, w, h, 라디안
        print(obb.xyxyxyxy)  # 네 개의 모서리 점과 같은 행
        print(result.boxes.xyxy)  # 축에 정렬된 상자를 감싸는
    - label: '회전 바운딩 박스, CLI'
      language: bash
      code: >
        libreyolo predict model=LibreRTDETRv2n-obb.pt source=aerial.png
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # coco128.yaml는 처음 사용할 때 128이미지 샘플을 다운로드합니다. `data`를 가리킵니다
        # 실제 실행을 위해 자신의 데이터셋 YAML에서.
        model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 batch=4 lr0=1e-4
    - label: 로라
      language: python
      code: |
        # lora 추가가 필요합니다: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: 멀티 GPU
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # val()은 객체가 아니라 일반 딕셔너리를 반환합니다
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTDETRr18.pt data=coco128.yaml
    - label: COCO에 반대
      language: bash
      code: |
        # coco-val-only.yaml는 5000개의 val2017 이미지를 가져오고 건너뜁니다
        # 학습 세트. 내장된 다운로드 스크립트를 포함하고 있어, 따라서 필요합니다
        # 데이터셋이 이미 로컬에 있지 않는 한 명시적인 허가가 필요합니다.
        libreyolo val model=LibreRTDETRr18.pt data=coco-val-only.yaml \
          allow_download_scripts=True
    - label: 방향이 지정된 상자
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 회전된 IoU와 방향 검증이 일치하므로, 예측은
        # 올바른 위치에 있지만 잘못된 각도면 실패로 간주됩니다.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95(OBB)"])
        print(metrics["metrics/mAP50(OBB)"])
  export:
    - label: Python
      language: python
      code: |
        # onnx 추가가 필요합니다: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTDETRr18.pt format=onnx
    - label: 방향이 지정된 상자
      language: bash
      code: >
        # ONNX와 TorchScript는 해당 작업을 위해 검증된 대상입니다.

        # 고정된 1024 x 1024 캔버스에서 FP32, 배치 1로.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 경로를 지정하므로, 내보낸 아티팩트가 로드됩니다
        # 모든 체크포인트와 마찬가지로 같은 Results 객체를 반환합니다.
        model = LibreYOLO("LibreRTDETRr18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 8022a5a591922a90
---

## 설치

RT-DETR은 선택적인 추가 설치가 필요 없습니다. 가져오는 모든 것은 기본 설치에 포함되어 있으며, `rtdetr` 추가는 그것에 아무 것도 더하지 않는 안정적인 이름입니다.

```bash
pip install libreyolo
```

`lora=True`를 사용한 어댑터 파인튜닝은 예외이며, `lora` 추가가 필요합니다.

```bash
pip install "libreyolo[lora]"
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 다운로드되며 로컬에 캐시됩니다.

<code-tabs name="predict" />

반환된 `Results` 객체는 모든 계열가 반환하는 객체이므로, 다른 디텍터로 교체하는 것은 한 줄의 변경으로 가능합니다. `conf`와 `max_det`는 쿼리와 클래스에 대한 top-k 디코드를 필터링합니다; 튜닝할 NMS 단계는 없으며, `iou`는 허용되지만 사용되지 않습니다. 오리엔티드 체크포인트는 `result.obb`을 기본적으로 채우고, `result.boxes`에는 포함된 축 정렬 직사각형을 채웁니다. 출처, 스트리밍 및 결과 처리는 [prediction](/docs/predict)을 참조하십시오.

## 변형

세 가지 버전, 그 사이에 두 가지 작업이 있으며, 크기 코드는 단일 시리즈로 이어지지 않습니다. 버전 1은 크기 이름을 백본인 ResNet 또는 HGNetv2에 따라 지정합니다. 버전 2는 ResNet 이름만 재사용합니다. 버전 1에는 이미 두 가지 HGNetv2 크기가 포함되어 있고, 버전 2의 결과는 충분히 근접하여 LibreYOLO는 해당 크기에 대해 중복 가중치를 공개하지 않습니다. 버전 4는 단순한 문자 시리즈를 사용하며, 이는 버전 1의 HGNetv2 이름과 충돌하므로 크기 코드만으로 모델을 식별할 수 없습니다. 버전 정보는 체크포인트 파일 이름에 기록됩니다.

<benchmark-table task="detect" />

<va-embed />

버전 2는 버전 1의 아키텍처와 상태 딕트 레이아웃을 유지하면서 변형 가능한 어텐션 샘플링 방식을 변경하기 때문에, 두 버전은 형태(shape)로 구분되는 것이 아니라 체크포인트의 메타데이터로 구분됩니다. 버전 4는 다른 계열로, D-FINE의 아키텍처와 트레이너를 재사용하며, 그 가중치는 DINOv3 비전 기본 모델 교사(teacher)를 HGNetv2 학생(student)으로 증류(distill)하여 얻습니다. LibreYOLO에서 `LibreRTDETRv4`는 `LibreDFINE`의 하위 클래스이며, 마스크 헤드가 제거되어 오직 검출만 수행합니다.

### 버전 2의 회전 바운딩 박스

버전 2는 두 번째 작업을 수행하는 유일한 버전입니다. 지원되는 작업은 `detect`와 `obb`이며, 이 두 작업은 그래프나 크기 시리즈를 공유하지 않습니다. 검출은 640px의 ResNet 크기를 사용하며, 방향 검출은 1024px에서 HGNetv2 시리즈인 n, s, m, l, x를 사용하고 입력 크기는 계열 단위가 아닌 작업 단위로 결정됩니다. 체크포인트는 자체 텐서를 통해 방향성으로 인식되며, 5좌표 박스 헤드와 버전 2 샘플링 매개변수로 판단합니다. 따라서 `-obb` 가중치는 `task` 인수가 없어도 방향 그래프에 로드되며, 둘 사이의 불일치는 조용한 재해석이 아닌 심각한 오류로 처리됩니다.

출판된 파일은 `LibreRTDETRv2n-obb.pt`부터 `LibreRTDETRv2x-obb.pt`까지입니다. 이들은 공식 DOTA v1.0 단일 스케일 체크포인트를 LibreYOLO 형식으로 변환한 것이며, 비행기와 선박에서 항구와 헬리콥터까지 15개의 항공 클래스가 포함되어 있고, 클래스 이름이 체크포인트에 스탬프되어 있습니다. 탐지 측과 달리, 방향성 작업은 추론 전용입니다: 예측, 검증 및 내보내기 작업이며, 방향성 모델에서는 `train()`가 나타납니다. 추적 및 테스트 시 증강도 방향성 박스를 지원하지 않습니다. [방향성 탐지](/docs/tasks/oriented-detection)는 작업, 레이블 형식 및 평가 지표를 다룹니다.

## 학습

학습은 공개된 체크포인트에서 시작합니다. `pretrained`는 허용된 후 세 가지 버전 모두에서 제거되므로 `pretrained=False`는 무작위로 초기화된 모델을 제공하지 않습니다. 이 섹션의 모든 내용은 탐지에 관한 것입니다: 버전 2의 방향화된 작업은 추론만 가능하며, 탐지 가중치에서 이를 위한 전이 경로는 없습니다. 두 버전이 서로 다른 백본을 사용하기 때문입니다.

<code-tabs name="train" />

학습률은 올바르게 설정해야 하는 인수이며, 각 버전은 라이브러리 전체의 기본값이 아닌 자체 기본값을 갖습니다. Python `train()` 시그니처는 해당 버전의 학습 구성에서 이를 읽고, CLI는 `lr0`가 전달되지 않을 때 동일한 값을 해결합니다. 버전 1과 2는 또한 `lr_backbone`를 사용하며 원래 레시피에 따라 `lr0`의 1/20로 기본값이 설정됩니다. 버전 4는 D-FINE 트레이너를 통해 실행되며, 대신 `backbone_lr_mult`로 백본 매개변수 그룹을 스케일링합니다.

특별히 변경할 이유가 없다면 `imgsz`를 체크포인트의 원래 크기로 두십시오. 다른 크기에서도 검증과 예측은 가능하지만 한 가지 남는 문제가 있습니다: 토큰 수가 원래 크기와 일치하는 직사각형 크기는 여전히 잘못된 종횡비를 위해 만들어진 임베딩을 재사용합니다.

데이터셋, 증강, 멀티 GPU 및 로거에 대해서는 [training](/docs/train)를 참조하십시오.

## 검증

`val()`는 학습한 형식의 데이터셋에 대해 측정된 정밀도, 재현율, mAP 50 및 mAP 50-95를 포함하는 `metrics/` 키의 사전을 반환합니다.

<code-tabs name="val" />

위 벤치마크 표의 행들은 LibreYOLO 벤치마크 하니스에서 가져온 것이며, 해당 표 아래의 주석에는 어떤 데이터셋이 이를 생성했는지 기록되어 있고 실행 기록으로 연결됩니다.

지향된 검증은 동일한 호출을 통해 실행되며 동일한 키를 보고하며, 여기에 `(OBB)` 접미사로 반복된 네 개가 추가됩니다. 매칭은 둘러싸는 사각형의 IoU가 아니라 회전된 IoU를 사용하므로 각도 오류는 실패로 간주됩니다. `augment=True`는 이 작업에서 거부됩니다.

## 내보내기

<export-matrix />

행렬은 계보를 한 페이지로 다룹니다: 세 버전이 형식에 관해 일치하지 않을 경우, 셀에는 세 가지 중 가장 약한 것이 표시되므로 어느 버전을 로드하더라도 과장된 내용은 없습니다. 방향이 지정된 행은 버전 2에만 속합니다. ONNX와 TorchScript는 FP32, 배치 1, 고정된 1024x1024 캔버스에서 검증되었습니다; OpenVINO, TensorRT 및 ExecuTorch는 변환 및 재로드가 가능하지만 전체 쿼리 세트에서 원시 출력 일치성을 달성하지 못했기 때문에 상위 상자는 한 픽셀의 일부까지 일치하지만 하위는 차이가 있습니다.

내보낸 아티팩트는 파일 접미사를 통해 `LibreYOLO()`로 다시 로드되므로, `.onnx` 또는 `.engine` 파일은 체크포인트처럼 작동하며 동일한 `Results`를 반환합니다.

<code-tabs name="export" />

## 체크포인트

이 계열용으로 발행된 모든 무게 파일.

<checkpoint-table />

파일 이름에는 버전, 그 다음 크기, 그 다음 작업이 포함됩니다. 탐지 가중치는 `LibreRTDETR<size>.pt`, `LibreRTDETRv2<size>.pt`, `LibreRTDETRv4<size>.pt`이며, 모두 640 px입니다. 방향 가중치는 버전 2에만 존재하며 작업 접미사를 추가합니다. `LibreRTDETRv2n-obb.pt`부터 `LibreRTDETRv2x-obb.pt`까지이며, 모두 1024 px이고 COCO가 아닌 DOTA v1.0에서 학습되었습니다.

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />

위 블록은 저자들이 버전 1과 2 탐지를 위해 발표한 내용입니다. 버전 2의 방향성 가중치는 세 번째 업스트림인 Apache-2.0 RiO-DETR 저장소 [github.com/RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR)을 가지고 있으며, DOTA 체크포인트가 여기에서 나옵니다. 이를 사용했다면 해당 프로젝트를 인용하십시오. 버전 4는 다른 그룹의 별도 논문이며 [github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation)에 독자적인 인용 블록이 있습니다. 버전 4 체크포인트를 사용했다면 해당 논문을 인용하십시오.
