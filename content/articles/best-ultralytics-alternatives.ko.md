---
title: 2026년 Ultralytics 대안 가이드
description: "2026년 Ultralytics YOLO를 대체할 오픈 소스 도구를 실용적으로 비교합니다. 라이선스, 지원 작업, 배포 한계와 가장 완성도 높은 MIT 라이선스 YOLO 라이브러리 LibreYOLO의 역할을 살펴봅니다."
date: 2026-07-01
author: Xuban
tags: [LibreYOLO, ultralytics-alternative, object-detection, yolo, mit-license]
faq:
  - q: "Ultralytics YOLO는 상업적 사용이 무료인가요?"
    a: "AGPL-3.0 조건에서만 무료이며, 네트워크 서비스까지 포함해 이를 기반으로 만든 애플리케이션의 소스 코드를 공개해야 합니다. 폐쇄 소스 상업적 사용에는 Ultralytics의 유료 상업용 라이선스가 필요합니다. 허용적 라이선스(MIT, Apache-2.0)를 사용하는 대안은 이런 제약을 피할 수 있습니다."
  - q: "라이선스 제약이 가장 적은 Ultralytics 대안은 무엇인가요?"
    a: "허용적 라이선스로 어디서나 배포할 수 있는 스택을 원한다면 LibreYOLO(코드 MIT), RF-DETR(Apache-2.0), YOLOX(Apache-2.0)가 있습니다. 모두 AGPL 카피레프트를 피합니다."
  - q: "2026년에는 무엇이 YOLO를 대체하고 있나요?"
    a: "실시간 트랜스포머 탐지 모델이 점점 더 많이 사용됩니다. RT-DETR, RF-DETR, D-FINE 및 DEIM 계열이 여기에 해당합니다. 성능이 충분한 하드웨어에서는 비슷한 지연 시간으로 YOLO와 같거나 더 높은 정확도를 냅니다. YOLO 계열 CNN은 소형 엣지 디바이스에서 여전히 강점이 있습니다. 이런 트랜스포머 모델은 해당 환경에서 성능이 낮고 성숙한 NCNN 또는 CPU 경로가 부족하기 때문입니다."
  - q: "이 도구들을 Raspberry Pi나 NPU에서 실행할 수 있나요?"
    a: "내보내기 형식에 따라 다릅니다. YOLOX와 RTMDet 같은 CNN 탐지 모델은 NCNN으로 내보내 Raspberry Pi에서 실행할 수 있고, 일부 모델(YOLOX)은 Hailo NPU도 사용할 수 있습니다. RF-DETR 같은 트랜스포머 탐지 모델은 그렇지 않으며 GPU나 강력한 CPU가 필요합니다."
---

공개 사항: LibreYOLO는 저희 프로젝트이며 이 목록의 첫 번째에 있습니다. 여기에 소개한 다른 도구는 실제 업무에서 사용하는 것들이며, LibreYOLO보다 나은 점도 함께 설명합니다.

대부분의 팀이 다음 세 가지 이유 중 하나로 Ultralytics를 떠납니다.

- **라이선스.** Ultralytics의 YOLO 모델은 AGPL-3.0을 따릅니다(YOLOv5는 2023년부터, v8 이후 모델도 해당). 익숙한 상황입니다. 제품을 만들어 출시한 뒤 법무 검토나 고객이 모델 라이선스 문제를 제기하면, 애플리케이션 전체를 오픈 소스로 공개하거나 상업용 라이선스를 구매해야 합니다. AGPL의 카피레프트는 제품을 배포하거나 서비스로 제공하는 순간 코드에도 적용됩니다. 다른 도구를 찾는 흔한 이유입니다.
- **개방성.** 한 업체가 모델, 학습 코드, 사용 조건을 모두 소유합니다. 일부 팀은 허가를 구하지 않고도 기반을 구축할 수 있도록 가중치와 코드에 허용적 라이선스를 적용하기를 원합니다.
- **모델.** 작업에 따라 YOLO가 항상 최선의 아키텍처인 것은 아닙니다. RT-DETR, RF-DETR, D-FINE 같은 실시간 트랜스포머 탐지 모델은 같은 지연 시간에서 더 높은 정확도를 낼 수 있습니다. 단점은 여러 연구 저장소에 흩어져 있다는 점입니다. 아래에서 보듯 LibreYOLO는 이 세 모델을 모두 하나의 API로 실행하므로, 이는 특정 모델 하나를 위해 라이브러리를 떠날 이유가 아니라 라이브러리를 바꿀 이유입니다.

아래 도구는 세 가지 기준으로 평가합니다. 제품에 적용할 수 있는 라이선스인지, 현재 PyTorch에서 설치하고 실행할 수 있는지, 실제 배포하는 하드웨어용으로 내보낼 수 있는지입니다.

읽다 보면 반복되는 양상이 있습니다. 각 대안은 유용하지만 개별적으로 사용하기는 까다롭습니다. YOLOX는 설치가 쉽지 않고, MMDetection은 휠 버전 고정 문제에 발목이 잡히며, RT-DETR 계열은 지원 창구가 없는 연구 코드이고, Detectron2는 2021년 이후 동결되어 있습니다. LibreYOLO를 첫 번째에 둔 이유는 이들 대부분을 최신 스택과 익숙한 API로 묶어 MIT 라이선스 아래에서 유지 관리하기 때문입니다. 이 목록의 항목이라기보다 이들을 잇는 계층에 가깝습니다.

## 빠른 비교

| 도구 | 라이선스 | 지원 작업 | 적합한 용도 | 엣지 내보내기 |
| --- | --- | --- | --- | --- |
| **LibreYOLO** | MIT (코드) | 탐지, 분할, 자세 추정, 분류, 깊이 추정, 시선 추정, 추적 | 20개가 넘는 모델 계열을 하나의 API로 제공하는 MIT 허브 | ONNX, TensorRT, OpenVINO, NCNN, CoreML, TFLite |
| **RF-DETR** | Apache-2.0 (N/S/M/L) | 탐지, 분할, 자세 추정 (미리보기) | 프로덕션 탐지 및 분할 | ONNX, TFLite, TensorRT; NCNN 또는 Hailo는 지원하지 않음 |
| **Lightly / LightlyTrain** | MIT / AGPL-3.0 | 사전 학습, 증류 | 레이블이 없는 데이터, DINOv2/v3 증류 | 해당 없음 (탐지 모델이 아님) |
| **YOLOX** | Apache-2.0 | 탐지 | 앵커 프리 실시간 탐지 | 업스트림 설치가 까다롭고 LibreYOLO를 통해 실행 가능 |
| **MMDetection / OneDL 포크** | Apache-2.0 | 전체 (연구용) | 다양한 아키텍처, 논문 재현 | 내보내기 사용 |
| **Detectron2** | Apache-2.0 | 탐지, 분할, 키포인트 | Mask R-CNN 및 R-CNN 계열 | 수동 |
| **D-FINE / DEIM / DEIMv2** | Apache-2.0 | 탐지 | 최첨단 실시간 DETR | ONNX, TensorRT |
| **EdgeCrafter** | Apache-2.0 | 탐지, 분할, 자세 추정 | DINOv3에서 증류한 엣지용 소형 ViT | 연구 수준 |
| **RT-DETR (v1-v4)** | Apache-2.0 | 탐지 | 최첨단 실시간 탐지 | ONNX, TensorRT |

## 1. LibreYOLO

라이선스: MIT (코드). 가장 완성도 높은 MIT 라이선스 YOLO 라이브러리이자 바로 교체해 쓸 수 있는 Ultralytics 대안을 찾는 경우에 적합합니다.

**LibreYOLO는 모든 모델을 하나의 API로 실행하는 MIT 라이선스 YOLO 라이브러리입니다.** 저희 프로젝트이며, 나머지 목록을 보면 그 역할이 분명해집니다. 여러 대안을 실행하는 허브입니다. 위에 나온 까다로운 저장소인 YOLOX, RTMDet, RF-DETR, D-FINE, DEIM, RT-DETR 계열 거의 모두를 LibreYOLO가 익숙하고 유지 관리되는 하나의 API 뒤에 통합합니다. 설치 과정을 헤매거나 라이선스 문제를 겪지 않고 모델을 사용할 수 있습니다.

이 프로젝트에는 편리함을 넘어서는 이유가 있습니다. YOLO는 공개 연구로 시작했습니다. Joseph Redmon의 Darknet v1부터 v3까지 누구나 자유롭게 사용하고 기반을 만들 수 있었습니다. AGPL 시대에 그 길이 닫혔습니다. LibreYOLO는 창작자들이 의도한 방식대로 그 작업에 다시 접근할 수 있게 하려고 만들어졌습니다. 허용적 라이선스, 커뮤니티 유지 관리, 사용자 모르게 데이터를 외부로 전송하지 않는 방식입니다. 두 가지 이야기를 말씀드리겠습니다.

**첫째, Ultralytics YOLO의 MIT 라이선스 대안입니다.** 이미 익숙한 API를 유지하므로 YOLO 형식 데이터셋과 스크립트를 조금만 바꾸어 옮길 수 있습니다. 다만 AGPL 대신 MIT를 적용해 코드에 카피레프트가 확장되지 않고, 상업용 라이선스를 구매할 필요가 없으며, Ultralytics가 기본적으로 하듯 데이터를 외부로 전송하는 원격 측정도 없습니다. 라이선스 때문에 이 글을 읽고 있다면 이것이 핵심입니다. YOLO9와 RF-DETR은 충분히 검증된 프로덕션용 기본 모델이고, 더 넓은 모델 모음은 새롭고 실험용으로 명확히 표시되어 있습니다. 모든 모델이 현장에서 충분히 검증된 것처럼 가장하기보다 이 점을 분명히 알립니다.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # or LibreRFDETRl.pt, LibreDFINEl.pt, ...
results = model("image.jpg", save=True)
```

**둘째, Ultralytics보다 훨씬 다양한 범위를 지원합니다.** 하나의 API가 CNN 기반 YOLO, 트랜스포머 기반 DETR, ViT 백본, SAM 등을 포함해 20개가 넘는 모델 계열을 다룹니다. 한 업체의 제품군에만 한정되지 않습니다. 탐지 모델만 있는 것도 아닙니다.

- **작업:** 인스턴스 및 시맨틱 분할, 자세 추정, 분류, 회전 바운딩 박스를 지원합니다. 여기에 Ultralytics에는 전혀 없는 단안 깊이 추정(Depth Anything V2)과 시선 추정(L2CS)도 포함됩니다.
- **실용적인 기능:** 지속적인 ID를 사용하는 다중 객체 추적(ByteTrack 및 OC-SORT), 프레임 서브샘플링과 실시간 미리보기를 지원하는 비디오 추론, 드론 및 위성 이미지용 타일 추론, 드래그 앤 드롭 브라우저 UI(`libreyolo ui`), 학습을 시작하기 전에 데이터셋을 확인하는 `doctor` 명령을 제공합니다.
- **폭넓은 학습 기능:** 그래디언트 누적, 레이어 고정, 학습 재개, 테스트 시점 증강, LoRA/DoRA 파인튜닝, 멀티 GPU, TensorBoard/MLflow/Weights & Biases 로깅을 지원합니다.
- **내보내기:** 7개 형식(ONNX, TorchScript, TensorRT, OpenVINO, NCNN, CoreML, TFLite)을 지원하며 INT8/FP16 양자화, 내장 NMS, 모델 메타데이터 삽입 기능도 제공합니다.
- **어디서나 실행:** 경로, URL, PIL, NumPy, 텐서 또는 원시 바이트를 입력으로 받습니다. 코드를 바꾸지 않고 CUDA, Apple Silicon (MPS), 일반 CPU에서 실행됩니다.

성능이 좋은 탐지 모델의 저장소가 방치되었거나 설치하기 어렵다면 LibreYOLO가 최신 스택과 유지 관리되는 환경에서 해당 가중치를 실행합니다.

## 2. RF-DETR

라이선스: Apache-2.0 (Nano, Small, Medium, Large). 프로덕션 탐지 및 분할에 적합합니다.

Roboflow의 RF-DETR은 ICLR 2026에 발표되었으며, 이 목록에서 프로덕션 준비가 가장 잘 된 모델입니다. 설계가 탄탄하고 제품을 출시하는 팀이 개발했습니다. 실제 시스템에서 신뢰할 수 있는 탐지 또는 분할이 필요하다면 우선 고려할 만한 모델입니다. `rfdetr` 패키지와 Nano부터 Large까지의 가중치는 Apache-2.0을 따릅니다. 더 큰 XL 및 2XL 가중치에는 Roboflow의 PML 라이선스가 적용됩니다.

## 3. Lightly

레이블이 없는 데이터, 자기 지도 사전 학습, DINO 백본 증류에 적합합니다.

Lightly는 탐지 모델이 아닙니다. 레이블을 지정하지 않은 데이터에서 가치를 얻기 위한 도구이며, 서로 다른 라이선스가 적용되는 두 부분으로 구성됩니다.

- **[`lightly`](https://github.com/lightly-ai/lightly) (LightlySSL), MIT.** 손실 함수, 헤드, 증강, 메모리 뱅크, 20개가 넘는 방법(SimCLR, MoCo, BYOL, DINO, DINOv2, MAE 등)의 참조 구현을 제공하는 자기 지도 학습 구성 요소 프레임워크입니다. 학습 루프는 직접 작성하며, 레이블이 없는 이미지로 표현을 처음부터 사전 학습하는 데 필요한 구성 요소를 제공합니다.
- **[LightlyTrain](https://github.com/lightly-ai/lightly-train), AGPL-3.0 (상업용 및 무료 커뮤니티 옵션 제공).** 대부분의 사람이 말하는 바로 사용할 수 있는 도구이자 워크플로입니다. DINOv2 또는 DINOv3 같은 파운데이션 모델과 레이블이 없는 데이터를 지정하면 몇 줄의 코드만으로 레이블 없이 해당 백본을 배포 가능한 소형 모델(YOLO, RT-DETR, ViT 또는 사용자 지정 네트워크)로 증류합니다. 라이선스에 유의해야 합니다. AGPL-3.0은 사람들이 Ultralytics 대안을 찾게 만드는 카피레프트와 같습니다. 폐쇄 소스 제품을 출시할 계획이라면 조건을 읽거나 상업용 라이선스를 선택해야 합니다.

좋은 백본은 필요하지만 레이블이 없다면 Lightly를 사용해 보세요. 이 도구는 목록의 탐지 모델과 경쟁하기보다 함께 사용됩니다. 최신 모델도 이를 보여줍니다. RT-DETRv4는 시각 파운데이션 모델 증류를 학습에 포함하고, DEIMv2는 DINOv3 백본을 탐지 모델에 직접 통합합니다.

## 4. YOLOX

라이선스: Apache-2.0. 설치만 해결하면 앵커 프리 실시간 탐지에 적합합니다.

업스트림 저장소는 마지막 릴리스인 v0.3.0이 2022년 4월에 나온 뒤 동결되어 있고, 미해결 이슈가 700개가 넘습니다. 2026년 스택에 설치하기도 어렵습니다. `pyproject.toml`이 없고 `requirements.txt`는 `onnx-simplifier==0.4.10`을 고정하는데, Python 3.10 이후 버전용 사전 빌드 휠이 없습니다. 따라서 최신 인터프리터에서는 소스 빌드가 필요하고 cmake와 C++ 툴체인도 설치해야 합니다. NumPy 버전도 고정되어 있지 않아 NumPy 2.0과 오래된 pycocotools 및 torch 사이에 ABI 충돌이 생길 수 있습니다. 의존성 문제를 해결하면 핵심 탐지 기능은 실행되지만, 설치 과정에 비용이 듭니다.

모델 자체는 여전히 훌륭합니다. Megvii의 앵커 프리 실시간 탐지 모델이며 코드와 가중치 모두 Apache-2.0을 따릅니다. 더 새로운 탐지 모델이 앞서 나갔지만 여전히 합리적인 베이스라인입니다. 잘 설계된 저장소가 방치된 경우 코딩 에이전트가 이제 오후 한나절 만에 되살릴 수 있으므로, 설치 문제가 겉보기만큼 큰 장벽은 아닙니다. LibreYOLO도 최신 스택에서 YOLOX 가중치를 직접 실행합니다. 어느 쪽이든 이 아키텍처는 사용할 가치가 있습니다. 더 자세한 안내는 [여기](/articles/yolox-with-libreyolo)에 있습니다.

## 5. MMDetection 생태계

라이선스: 대부분 Apache-2.0. 다양한 아키텍처와 논문 재현에 적합합니다.

수년 동안 MMDetection은 거의 모든 탐지 논문의 공식 구현을 보유했습니다. Faster R-CNN, DINO, Grounding-DINO, RTMDet, Mask R-CNN과 수백 개의 설정이 포함되었습니다. 2026년 OpenMMLab은 mm 계열을 단계적으로 중단했습니다. [MMDetection](https://github.com/open-mmlab/mmdetection)의 마지막 릴리스는 2024년 1월의 v3.3.0이며, 이슈에는 답변이 없고 `mmcv` 버전 고정으로 스택이 제약됩니다. 사전 빌드 휠이 현재 PyTorch 및 CUDA 버전보다 뒤처져 있어 새로 설치하면 모든 항목을 다운그레이드하기 전까지 실패하는 경우가 많습니다. 이 문제는 [여기](/articles/rtmdet-without-mmdetection)에서 다뤘습니다.

네덜란드 회사 VBTI가 이 프로젝트를 계속 유지하고 있습니다. [OneDL 포크](https://github.com/VBTI-development/onedl-mmdetection)는 전체 스택을 `onedl-` 접두사 아래에 다시 배포합니다(`onedl-mmdetection`, `onedl-mmcv`, `onedl-mmengine` 등). 현재 PyTorch 2.x, CUDA, Python 3.10+에 맞춰 다시 빌드했으므로 버전 고정 문제를 해결합니다. Apache-2.0을 따르며 활발하게 유지 관리되고 있고, v3.5.1은 2026년 5월에 나왔습니다. 설치 작업 없이 MMDetection의 폭넓은 모델을 사용하려면 이 포크를 이용하세요. 소규모 팀이므로 의존하고 있다면 기여해 주세요.

관련 도구 두 가지도 있습니다.

- **[Detectron2](https://github.com/facebookresearch/detectron2)** (Meta, Apache-2.0)는 유지 관리 모드이며 2021년 v0.6 이후 태그가 붙은 릴리스가 없습니다. 하지만 안정적이며 인스턴스 및 판옵틱 분할을 위한 Mask R-CNN과 R-CNN 계열의 가장 깔끔한 구현으로 남아 있습니다.
- **[TorchVision](https://github.com/pytorch/vision)** (BSD-3-Clause, 활발히 유지 관리됨)은 Faster R-CNN, RetinaNet, FCOS, SSD, Mask R-CNN, Keypoint R-CNN을 제공합니다. 최신 YOLO나 DETR은 없고 학습 루프를 직접 작성해야 하지만, 별도 의존성이 없는 기준점입니다.

상업적 사용 시 주의할 점이 하나 있습니다. OpenMMLab의 YOLO 재구현 허브인 **MMYOLO**는 Apache-2.0이 아닌 GPL-3.0을 따르며, 2023년 8월 이후 동결되어 있습니다.

## 6. RT-DETR 계열

라이선스: 모두 Apache-2.0. 연구 수준 코드로 최첨단 실시간 탐지를 구현할 때 적합합니다.

현재 실시간 탐지의 최전선에는 YOLO가 아닌 서로 연관된 트랜스포머 탐지 모델들이 있습니다. 대부분 RT-DETR에서 파생되었고 백본 및 인코더 코드의 상당 부분을 공유합니다. 거의 모두 Apache-2.0을 따르므로 모델 간 전환도 쉽습니다. EdgeCrafter 한 가지 예외를 제외하면 대부분 탐지만 지원합니다. EdgeCrafter는 같은 증류 아이디어를 분할과 자세 추정으로 확장합니다. 공통적인 비용은 이들이 연구 저장소라는 점입니다. 설정 파일을 통한 학습, Ultralytics와 다른 사용 방식, 지원 창구 부재가 여기에 해당합니다. 모델 성능은 뛰어나며 ONNX 및 TensorRT 내보내기도 대체로 잘 지원됩니다.

- **[D-FINE](https://github.com/Peterande/D-FINE)** (USTC, ICLR 2025 Spotlight). 바운딩 박스 회귀를 자기 증류를 적용한 분포 정제로 재구성합니다. 지연 시간 대비 정확도가 높고(D-FINE-X는 T4에서 약 13 ms에 약 55.8 AP), N부터 X까지 크기가 있습니다. Hugging Face Transformers에도 통합되어 있어 자체 저장소 밖에서 불러오고 파인튜닝하기 가장 쉽습니다.
- **[DEIM 및 DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)** (Intellindust AI Lab). DEIM (CVPR 2025)은 D-FINE 또는 RT-DETRv2에 추가할 수 있는 학습 레시피(Dense O2O 매칭)로 학습 시간을 대략 절반으로 줄입니다. DEIMv2는 DINOv3 특징을 추가하고 모바일용 1M 미만 파라미터의 Atto 단계까지 크기를 줄였습니다. DEIMv2-S는 10M 미만 파라미터로 50 AP를 넘은 첫 번째 모델입니다. 활발하게 유지 관리되고 있으며 2026년까지 업데이트되었습니다.
- **[EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter)** (Intellindust AI Lab, 2026). 같은 연구소의 최신 모델이며 이 목록에서 탐지 범위를 넘어서는 유일한 모델입니다. 탐지(ECDet), 인스턴스 분할(ECSeg), 자세 추정(ECPose)을 각각 S/M/L/X 크기로 제공하고 모두 Apache-2.0을 따릅니다. DINOv3 사전 학습을 거친 대형 ViT를 작업별 교사 모델로 조정한 뒤, 엣지용 소형 학생 백본으로 증류합니다. 크기에 비해 성능이 뛰어납니다. ECDet-S는 COCO만으로 파라미터 10M 미만에서 51.7 AP를 기록하고, ECPose-X는 74.8 AP를 달성합니다(YOLO26Pose-X의 71.6보다 높음). 인스턴스 분할 성능은 RF-DETR과 견줄 만합니다. 출시된 지 얼마 안 되었으므로 연구용 릴리스로 보아야 하지만, 세 작업을 모두 지원하는 드문 소형 모델입니다.
- **[RT-DETR 및 RT-DETRv2](https://github.com/lyuwenyu/RT-DETR)** (Baidu). 원 논문 “DETRs beat YOLOs on real-time detection” (CVPR 2024)이자 이 계열의 기반입니다. NMS가 필요 없고 종단 간 방식이며 내보내기에도 적합합니다. v2는 2024년에 나온 bag-of-freebies 업데이트로, 같은 지연 시간에서 바로 교체할 수 있는 개선 버전입니다. 원본 Paddle과 표준 PyTorch 포트가 같은 저장소를 공유하며 HF Transformers에도 포함되어 있습니다.
- **[RT-DETRv3](https://github.com/clxia12/RT-DETRv3)** (Baidu, WACV 2025 Oral) 및 **[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)** (Peking University 및 Tsinghua, 2025년 말). v3는 PaddlePaddle만 지원합니다. v4는 현재 이 계열에서 가장 뛰어난 모델(X은 57 AP에 근접)이며 PyTorch를 사용합니다. 시각 파운데이션 모델을 경량 탐지 모델로 증류해 추론 비용을 늘리지 않습니다. 설정을 바꾸면 D-FINE, DEIM, RT-DETRv2도 재현할 수 있습니다.
- **[LW-DETR](https://github.com/Atten4Vis/LW-DETR)** (Baidu). YOLO를 대체하는 트랜스포머로 소개된 기본 ViT 기반 DETR입니다. tiny부터 xlarge까지 크기를 제공하며 ONNX 및 TensorRT로 내보낼 수 있습니다. 오픈 보캐뷸러리 모델인 OVLW-DETR의 기반이기도 합니다. 최근 활동은 줄어들었으므로 안정적인 연구용 릴리스로 보아야 합니다.

LibreYOLO는 이 모델 중 다수(D-FINE, DEIM, DEIMv2, RT-DETRv2, RT-DETRv4, RTMDet, EdgeCrafter)를 하나의 API로 이미 통합했으며, 가장 많이 검증된 모델은 YOLO9와 RF-DETR입니다. 참조 구현과 최신 연구 체크포인트는 위의 원본 저장소를 기준으로 확인할 수 있습니다.

## FAQ

**Ultralytics YOLO는 상업적 사용이 무료인가요?**
AGPL-3.0 조건에서만 무료이며, 네트워크 서비스까지 포함해 이를 기반으로 만든 애플리케이션의 소스 코드를 공개해야 합니다. 폐쇄 소스 상업적 사용에는 Ultralytics의 유료 상업용 라이선스가 필요합니다. 허용적 라이선스(MIT, Apache-2.0)를 사용하는 대안은 이런 제약을 피할 수 있습니다.

**라이선스 제약이 가장 적은 Ultralytics 대안은 무엇인가요?**
허용적 라이선스로 어디서나 배포할 수 있는 스택을 원한다면 LibreYOLO(코드 MIT), RF-DETR(Apache-2.0), YOLOX(Apache-2.0)가 있습니다. 모두 AGPL 카피레프트를 피합니다.

**2026년에는 무엇이 YOLO를 대체하고 있나요?**
실시간 트랜스포머 탐지 모델이 점점 더 많이 사용됩니다. RT-DETR, RF-DETR, D-FINE 및 DEIM 계열이 여기에 해당합니다. 성능이 충분한 하드웨어에서는 비슷한 지연 시간으로 YOLO와 같거나 더 높은 정확도를 냅니다. YOLO 계열 CNN은 소형 엣지 디바이스에서 여전히 강점이 있습니다. 이런 트랜스포머 모델은 해당 환경에서 성능이 낮고 성숙한 NCNN 또는 CPU 경로가 부족하기 때문입니다.

**이 도구들을 Raspberry Pi나 NPU에서 실행할 수 있나요?**
내보내기 형식에 따라 다릅니다. YOLOX와 RTMDet 같은 CNN 탐지 모델은 NCNN으로 내보내 Raspberry Pi에서 실행할 수 있고, 일부 모델(YOLOX)은 Hailo NPU도 사용할 수 있습니다. RF-DETR 같은 트랜스포머 탐지 모델은 그렇지 않으며 GPU나 강력한 CPU가 필요합니다.

## 사용해 보기

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO는 MIT 라이선스를 따르며 Linux, Mac, Windows에서 실행됩니다. 코드를 바꾸지 않고 GPU, Apple Silicon, 일반 CPU에서 사용할 수 있습니다. 하나의 API로 RF-DETR, D-FINE, DEIM, YOLOX, YOLO-NAS, RTMDet, RT-DETR, EdgeCrafter 등을 탐지, 분할, 자세 추정, 분류, 깊이 추정, 시선 추정, 추적에 사용할 수 있습니다.

GitHub에서 별을 눌러 주세요: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | 문서: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
