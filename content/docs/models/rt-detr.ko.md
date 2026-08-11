---
title: RT-DETR
families:
  - rtdetr
seo_title: 'LibreYOLO의 RT-DETR, RT-DETRv2, RT-DETRv4'
description: >-
  LibreYOLO에서 RT-DETR, RT-DETRv2, RT-DETRv4로 객체를 탐지하고 RT-DETRv2로 회전 박스도 탐지합니다.
  Apache-2.0 가중치로 설치, 예측, 학습, 검증, 내보내기합니다.
lead: >-
  실시간 추론용 detection transformer입니다. 조밀 그리드 대신 고정 쿼리 집합을 디코딩하므로 NMS를 실행하지 않습니다.
  LibreYOLO는 체크포인트로 구분되는 세 버전을 제공하며 버전 2는 회전 박스도 지원합니다.
keywords:
  - RT-DETR 사용법
  - RT-DETRv2
  - RT-DETRv4
  - 실시간 detection transformer
  - DETR
  - 객체 탐지
  - 회전 바운딩 박스 탐지
  - OBB
  - DOTA
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
    - label: 동영상
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 버전은 파일명에 포함되고 팩토리는 체크포인트에 따라 라우팅하므로
        # 세 버전을 같은 방식으로 불러옵니다.
        model = LibreYOLO("LibreRTDETRv4s.pt")

        # 라이브러리가 받는 모든 소스: 파일, 폴더, URL, 웹캠 인덱스,
        # RTSP 스트림 또는 .streams 목록
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
    - label: 회전 박스
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 버전 2 전용입니다. -obb 접미사가 작업을 선택하고 체크포인트 자체의
        # 텐서로 회전 모델임을 인식하므로 task 인수가 필요하지 않습니다.
        # 이 가중치는 DOTA v1.0의 항공 클래스 15개를 1024 px에서 처리합니다.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)     # (N, 5): cx, cy, w, h, 라디안
        print(obb.xyxyxyxy)  # 동일한 행을 네 모서리 점으로 표현
        print(result.boxes.xyxy)  # 둘러싸는 축 정렬 박스
    - label: 회전 박스 CLI
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

        # coco128.yaml은 처음 사용할 때 이미지 128개 샘플을 다운로드합니다.
        # 실제 실행에서는 `data`를 자체 데이터셋 YAML로 지정합니다.
        model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 batch=4 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        # lora extra가 필요합니다: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: 다중 GPU
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

        # val()은 객체가 아닌 일반 dict를 반환합니다.
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTDETRr18.pt data=coco128.yaml
    - label: COCO 기준 평가
      language: bash
      code: |
        # coco-val-only.yaml은 val2017 이미지 5000개를 가져오고 학습 집합은 생략합니다.
        # 내장 다운로드 스크립트가 있으므로 데이터셋이 로컬에 없다면 명시적 권한이 필요합니다.
        libreyolo val model=LibreRTDETRr18.pt data=coco-val-only.yaml \
          allow_download_scripts=True
    - label: 회전 박스
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 회전 검증은 rotated IoU로 일치시키므로 위치가 맞더라도 각도가 틀리면
        # 탐지를 놓친 것으로 계산합니다.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95(OBB)"])
        print(metrics["metrics/mAP50(OBB)"])
  export:
    - label: Python
      language: python
      code: |
        # onnx extra가 필요합니다: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTDETRr18.pt format=onnx
    - label: 회전 박스
      language: bash
      code: >
        # 회전 작업에서 검증된 대상은 ONNX와 TorchScript이며 FP32, 배치 1,

        # 고정된 1024x1024 캔버스를 사용합니다.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: 내보낸 파일 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 팩토리는 파일 접미사에 따라 라우팅하므로 내보낸 아티팩트도
        # 다른 체크포인트처럼 불러와 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreRTDETRr18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 8022a5a591922a90
---

## 설치

RT-DETR에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함되며 `rtdetr` extra는 아무것도 추가하지 않는 안정적인 이름입니다.

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

반환되는 `Results` 객체는 모든 제품군이 반환하는 것과 같으므로 탐지기를 바꾸려면 한 줄만 변경하면 됩니다. `conf`와 `max_det`은 쿼리와 클래스의 top-k 디코딩을 필터링합니다. 조정할 NMS 단계가 없으며 `iou`는 허용되지만 사용되지 않습니다. 회전 체크포인트는 `result.obb`를 네이티브로 채우고 둘러싸는 축 정렬 직사각형으로 `result.boxes`도 채웁니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

세 버전에 두 작업이 나뉘어 있으며 크기 코드는 하나의 연속된 계열이 아닙니다. 버전 1은 백본인 ResNet 또는 HGNetv2의 이름으로 크기를 정합니다. 버전 2는 ResNet 이름만 재사용합니다. 버전 1에 이미 두 HGNetv2 크기가 있고 버전 2의 결과도 충분히 비슷해 LibreYOLO는 중복 가중치를 게시하지 않습니다. 버전 4는 일반 문자 계열을 사용하며 버전 1의 HGNetv2 이름과 충돌하므로 크기 코드만으로 모델을 식별할 수 없습니다. 버전은 체크포인트 파일명에 기록됩니다.

<benchmark-table task="detect" />

<va-embed />

버전 2는 버전 1의 구조와 state dict 레이아웃을 유지하면서 deformable attention 샘플링 방식을 바꿉니다. 따라서 형태가 아니라 체크포인트 메타데이터로 두 버전을 구분합니다. 버전 4는 다른 계보입니다. D-FINE의 구조와 학습기를 재사용하고 DINOv3 vision foundation model teacher를 HGNetv2 student로 증류해 가중치를 만듭니다. LibreYOLO에서 `LibreRTDETRv4`는 마스크 헤드를 끈 `LibreDFINE` 하위 클래스이므로 탐지 전용입니다.

### 버전 2의 회전 박스

버전 2만 두 번째 작업을 제공합니다. 지원 작업은 `detect`와 `obb`이며 두 작업은 그래프나 크기 계열을 공유하지 않습니다. 탐지는 640 px에서 ResNet 크기를 사용하고 회전 탐지는 1024 px에서 n, s, m, l, x의 HGNetv2 계열을 사용합니다. 입력 크기는 제품군이 아니라 작업별로 결정됩니다. 체크포인트 자체 텐서의 5좌표 박스 헤드와 버전 2 샘플링 매개변수로 회전 모델을 인식하므로 `-obb` 가중치는 `task` 인수 없이 회전 그래프에 불러오며 둘이 불일치하면 조용히 재해석하지 않고 즉시 오류가 발생합니다.

공개 파일은 `LibreRTDETRv2n-obb.pt`부터 `LibreRTDETRv2x-obb.pt`까지입니다. 공식 DOTA v1.0 단일 스케일 체크포인트를 LibreYOLO 형식으로 변환한 것으로 plane과 ship부터 harbor와 helicopter까지 항공 클래스 15개를 포함하며 클래스 이름은 체크포인트에 기록됩니다. 탐지 쪽과 달리 회전 작업은 추론 전용입니다. 예측, 검증, 내보내기는 작동하지만 회전 모델에서 `train()`을 호출하면 예외가 발생합니다. 추적과 테스트 시간 증강도 회전 박스를 지원하지 않습니다. 작업, 레이블 형식, 지표는 [회전 탐지](/docs/tasks/oriented-detection)를 참조합니다.

## 학습

학습은 공개된 체크포인트에서 시작합니다. 세 버전 모두 `pretrained`를 받은 뒤 버리므로 `pretrained=False`도 무작위로 초기화된 모델을 제공하지 않습니다. 이 섹션은 모두 탐지에 관한 내용입니다. 버전 2의 회전 작업은 추론 전용이고 두 작업은 다른 백본을 사용하므로 탐지 가중치에서 회전 작업으로 전이하는 경로가 없습니다.

<code-tabs name="train" />

올바르게 설정해야 할 핵심 인수는 학습률이며 각 버전은 라이브러리 공통값이 아닌 자체 기본값을 갖습니다. Python `train()` 시그니처는 해당 버전 학습 구성에서 읽고 CLI도 `lr0`를 전달하지 않으면 같은 값을 결정합니다. 버전 1과 2는 `lr_backbone`도 받으며 원래 레시피에 따라 기본값을 `lr0`의 1/20로 설정합니다. 버전 4는 D-FINE 학습기를 거치며 대신 `backbone_lr_mult`로 백본 매개변수 그룹을 조정합니다.

변경할 이유가 없다면 `imgsz`를 체크포인트 기본 크기로 유지합니다. 다른 크기에서도 검증과 예측이 작동하지만 한 가지 잔여 문제가 있습니다. 토큰 수가 기본 크기와 같은 직사각형은 잘못된 종횡비용으로 만들어진 임베딩을 계속 재사용합니다.

데이터셋, 증강, 다중 GPU, 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 학습에 사용한 형식의 데이터셋을 대상으로 측정한 정밀도, 재현율, mAP 50, mAP 50-95를 포함하는 `metrics/` 키 사전을 반환합니다.

<code-tabs name="val" />

위 벤치마크 표의 행은 LibreYOLO 벤치마크 도구에서 나옵니다. 표 아래 참고 사항에는 수치를 만든 데이터셋과 실행 기록 링크가 있습니다.

회전 검증은 같은 호출을 사용하고 같은 키와 `(OBB)` 접미사가 붙은 네 키를 추가로 보고합니다. 둘러싸는 직사각형의 IoU가 아니라 rotated IoU로 일치시키므로 각도 오류는 누락으로 계산됩니다. 이 작업은 `augment=True`를 거부합니다.

## 내보내기

<export-matrix />

매트릭스는 세 버전의 계보를 한 페이지에서 다룹니다. 형식 지원이 다르면 셀은 셋 중 가장 약한 수준을 표시하므로 불러온 버전에 대한 지원을 과장하지 않습니다. 회전 행은 버전 2에만 해당합니다. 해당 작업에서 ONNX와 TorchScript는 FP32, 배치 1, 고정 1024x1024 캔버스로 검증되었습니다. OpenVINO, TensorRT, ExecuTorch는 변환하고 다시 불러올 수 있지만 전체 쿼리 집합에서 원시 출력 동등성을 충족하지 못했습니다. 최고 순위 박스는 픽셀 일부 범위로 일치하지만 꼬리 부분은 드리프트합니다.

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는 `.engine` 파일은 체크포인트처럼 동작하며 동일한 `Results`를 반환합니다.

<code-tabs name="export" />

## 체크포인트

이 제품군에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

파일명에는 버전, 크기, 작업 순서로 들어갑니다. 탐지 가중치는 `LibreRTDETR<size>.pt`, `LibreRTDETRv2<size>.pt`, `LibreRTDETRv4<size>.pt`이며 모두 640 px입니다. 회전 가중치는 버전 2에만 존재하고 작업 접미사를 추가한 `LibreRTDETRv2n-obb.pt`부터 `LibreRTDETRv2x-obb.pt`까지이며 모두 1024 px이고 COCO가 아닌 DOTA v1.0에서 학습했습니다.

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />

위 블록은 버전 1과 2의 탐지용으로 저자가 게시한 인용입니다. 버전 2 회전 가중치에는 세 번째 업스트림인 Apache-2.0 RiO-DETR 저장소 [github.com/RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR)이 있으며 DOTA 체크포인트는 여기에서 가져옵니다. 해당 체크포인트를 사용했다면 이 프로젝트를 인용합니다. 버전 4는 다른 그룹의 별도 논문이며 [github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation)에 자체 인용 블록이 있습니다. 버전 4 체크포인트를 사용했다면 이를 인용합니다.
