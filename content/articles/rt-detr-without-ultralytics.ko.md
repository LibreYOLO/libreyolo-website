---
title: "RT-DETR 상업적 사용: Ultralytics 없이 v1, v2, v4 실행하기"
description: "Apache-2.0 가중치를 사용하는 RT-DETR, RT-DETRv2, RT-DETRv4를 MIT 라이브러리에서 실행하고 파인튜닝합니다. 예측, 학습, 검증, 내보내기를 하나의 API로 처리합니다."
date: 2026-09-24
author: Xuban
tags: [LibreYOLO, rt-detr, rtdetr, rt-detrv4, object-detection, license, tutorial]
faq:
  - q: "RT-DETR을 상업적으로 무료로 사용할 수 있습니까?"
    a: "허용적 라이선스의 구현을 사용하면 가능합니다. RT-DETR 및 RT-DETRv2의 원본 코드(lyuwenyu/RT-DETR)와 RT-DETRv4(RT-DETRs/RT-DETRv4)는 Apache-2.0이며, LibreYOLO는 MIT 라이브러리에서 세 버전을 모두 Apache-2.0 가중치로 실행합니다. ultralytics 패키지의 RT-DETR 구현은 AGPL-3.0이며, 비공개 소스로 사용하려면 유료 Enterprise License가 대안입니다. 이는 일반적인 정보이며 법률 자문이 아닙니다."
  - q: "RT-DETRv4의 라이선스는 무엇입니까?"
    a: "Apache-2.0입니다. github.com/RT-DETRs/RT-DETRv4의 LICENSE 파일은 Apache-2.0입니다. RT-DETRv4는 학습 중에만 DINOv3 교사를 사용합니다. 공개된 학생 가중치에는 DINOv3 파라미터가 없으므로 Meta의 DINOv3 라이선스가 해당 가중치에 적용되지 않습니다."
  - q: "Ultralytics 없이 RT-DETR을 파인튜닝할 수 있습니까?"
    a: "가능합니다. LibreYOLO는 공개 COCO 가중치에서 시작해 model.train() 또는 명령줄의 libreyolo train으로 RT-DETR, RT-DETRv2, RT-DETRv4 탐지 체크포인트를 파인튜닝합니다. RT-DETRv2의 방향 상자 모델은 추론 전용입니다."
  - q: "LibreYOLO에는 어떤 RT-DETR 가중치가 포함되어 있습니까?"
    a: "RT-DETR은 r18, r34, r50, r50m, r101, l, x, RT-DETRv2는 r18, r34, r50, r50m, r101, RT-DETRv4는 s, m, l, x 모델이 있으며, 모두 640 px COCO 탐지 모델입니다. RT-DETRv2에는 1024 px에서 DOTA v1.0으로 학습한 n, s, m, l, x 방향 상자 모델도 있습니다. 모든 파일은 Apache-2.0입니다."
---

RT-DETR은 Baidu의 실시간 탐지 트랜스포머입니다. 조밀한 그리드 대신 고정된 쿼리 집합을 디코딩하므로 조정할 NMS 단계가 없습니다. RT-DETR을 검색하면 가장 먼저 찾는 구현 중 하나가 Python `ultralytics` 패키지에 포함된 구현입니다. 이 때문에 모델 자체에는 없던 라이선스 문제가 생깁니다.

## RT-DETR 라이선스는 저장소에 따라 다릅니다

라이선스는 모델이 아니라 저장소에 적용됩니다. RT-DETR 관련 저장소는 여러 개이며, 라이선스가 서로 다릅니다.

| 저장소 | 대상 | 라이선스 |
|:---|:---|:---|
| [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | RT-DETR 및 RT-DETRv2, PyTorch와 Paddle | Apache-2.0 |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | RT-DETRv4 | Apache-2.0 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | `ultralytics` 패키지 내부의 RT-DETR | AGPL-3.0 또는 Enterprise License |
| [LibreYOLO](https://github.com/LibreYOLO/libreyolo) | RT-DETR, RT-DETRv2 및 RT-DETRv4 | MIT 코드, Apache-2.0 가중치 |

Ultralytics 구현은 완성도가 높습니다. [RT-DETR 페이지](https://docs.ultralytics.com/models/rtdetr/)에는 사전 학습된 `rtdetr-l.pt` 및 `rtdetr-x.pt`와 학습, 검증, 추론, 내보내기가 소개되어 있습니다. 이 구현은 AGPL-3.0 라이선스 패키지에 포함되어 있으며, 프로젝트 전체를 오픈 소스화하지 않으려는 팀을 위해 Ultralytics가 Enterprise License를 판매합니다. 둘 다 적합하지 않다면 원 저자의 구현을 Apache-2.0으로 사용할 수 있습니다. [YOLO 라이선스 설명](/articles/yolo-licenses-explained)에서는 YOLO 계열 전반의 같은 패턴을 다루고, [상업적 라이선스 가이드](/articles/yolo-commercial-license)에서는 비공개 소스 제품에 AGPL-3.0이 어떤 의미인지 설명합니다.

## LibreYOLO로 RT-DETR 실행하기

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreRTDETRr18.pt")  # downloads from Hugging Face on first use
result = model(SAMPLE_IMAGE, save=True)

for box in result.boxes:
    print(box.cls, box.conf, box.xyxy)
```

같은 호출을 명령줄에서도 사용할 수 있습니다.

```bash
libreyolo predict model=LibreRTDETRr18.pt source=image.jpg save=True
```

`conf`와 `max_det`은 쿼리와 클래스에 대한 top-k 디코딩 결과를 필터링합니다. `iou`는 허용되지만 NMS가 없으므로 사용되지 않습니다. 반환되는 `Results` 객체는 모든 LibreYOLO 모델에서 동일하므로 탐지기를 한 줄만 바꿔 교체할 수 있습니다.

## 하나의 로더에서 RT-DETR, RT-DETRv2, RT-DETRv4 사용하기

파일 이름에 버전이 포함되며 `LibreYOLO()`는 체크포인트에 따라 해당 버전으로 연결하므로 세 버전을 모두 같은 방식으로 불러옵니다.

* **RT-DETR:** r18, r34, r50, r50m, r101(ResNet 백본), l, x(HGNetv2)의 `LibreRTDETR` 모델입니다.
* **RT-DETRv2:** r18, r34, r50, r50m, r101의 `LibreRTDETRv2` 모델입니다. 버전 1의 아키텍처를 유지하면서 변형 어텐션의 샘플링 방식을 변경합니다.
* **RT-DETRv4:** s, m, l, x의 `LibreRTDETRv4` 모델입니다. D-FINE의 아키텍처를 재사용하며, DINOv3 교사를 HGNetv2 학생 모델로 증류해 가중치를 만듭니다.

모든 탐지 가중치는 COCO로 학습하며 640 px에서 실행됩니다. 참고로 업스트림 README에는 COCO에서 RT-DETR-R18의 AP가 46.5, RT-DETR-L의 AP가 53.0이라고 나와 있으며, RT-DETRv4-S의 AP는 49.8, RT-DETRv4-X는 최대 57.0입니다. [RT-DETR 문서 페이지](/docs/models/rt-detr)에는 모든 체크포인트의 LibreYOLO 자체 벤치마크 표가 있습니다.

RT-DETRv2는 방향 상자도 지원합니다. `LibreRTDETRv2n-obb.pt`부터 `LibreRTDETRv2x-obb.pt`까지는 공식 DOTA v1.0 체크포인트로, 항공 이미지의 15개 클래스를 1024 px에서 처리하며 LibreYOLO 형식으로 변환되어 있습니다.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv2n-obb.pt")
result = model("aerial.png", save=True)
print(result.obb.xywhr)  # (N, 5): cx, cy, w, h, radians
```

[방향 탐지](/docs/tasks/oriented-detection) 문서에서 레이블 형식과 지표를 설명합니다.

## 자체 데이터로 RT-DETR 파인튜닝하기

학습은 공개된 체크포인트에서 시작합니다.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRr18.pt")
model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
```

실제 학습에서는 `data`에 자체 데이터셋 YAML을 지정합니다. 각 버전에는 고유한 학습률 기본값이 있으며, 버전 1과 2는 원본 레시피에 따라 백본 학습률을 `lr0`의 20분의 1로 설정합니다. CLI에서 다중 GPU를 사용하려면 `device=0,1`을 지정하고, [LoRA](/docs/train/lora) 어댑터 파인튜닝은 `lora` 추가 항목을 설치한 뒤 `lora=True`로 설정합니다. RT-DETRv2의 방향 상자 모델은 추론 전용입니다. 데이터셋, 증강, 로거는 [학습 문서](/docs/train)를 참조하십시오.

## 검증 및 내보내기

```python
metrics = model.val(data="coco128.yaml")
print(metrics["metrics/mAP50-95"])

path = model.export(format="onnx")  # needs: pip install "libreyolo[onnx]"
```

`val()`은 일반 Python 딕셔너리를 반환합니다. 내보내기 대상에는 ONNX, TorchScript, ExecuTorch, TensorRT, OpenVINO가 포함됩니다. [모델 페이지](/docs/models/rt-detr)의 내보내기 표에서 각 작업에 대해 검증된 형식을 확인할 수 있습니다. 내보낸 `.onnx` 또는 `.engine` 파일을 `LibreYOLO()`로 다시 불러오면 동일한 `Results`를 반환합니다. 형식별 가이드는 [내보내기 문서](/docs/export)를 참조하십시오.

## 원본 저장소를 사용하는 것이 적합한 경우

저자의 정확한 설정으로 논문 결과를 재현하거나, Paddle 버전이 필요하거나, RT-DETRv4의 DINOv3 교사 증류를 직접 실행하려면 업스트림 저장소를 사용하십시오. LibreYOLO는 공개된 v4 학생 모델을 파인튜닝하며 교사 모델을 실행하지는 않습니다. 프로젝트가 이미 AGPL-3.0을 따르거나 Ultralytics Enterprise License의 적용을 받는다면 라이선스 때문에 옮길 이유는 없습니다.

## 라이선스 참고 사항

LibreYOLO 코드는 MIT입니다. 모든 RT-DETR 가중치 파일은 Apache-2.0이며, 각 Hugging Face 가중치 저장소에는 가중치를 재배포할 때 보존하도록 Apache-2.0에서 요구하는 업스트림 LICENSE와 NOTICE가 포함되어 있습니다. 자체 데이터로 학습한 가중치는 학습한 사람의 소유입니다. RT-DETRv4는 학습 중에만 DINOv3를 교사로 사용하며, 공개된 학생 모델에는 DINOv3 파라미터가 없습니다. 이는 일반적인 정보이며 법률 자문이 아닙니다. 배포하기 전에 최신 LICENSE 파일을 확인하십시오.

## 사용해 보기

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv4s.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO는 MIT 라이선스이며 Linux, Mac, Windows에서 실행됩니다. 코드 변경 없이 GPU, Apple Silicon, 일반 CPU에서도 작동합니다.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [RT-DETR 문서](/docs/models/rt-detr)
