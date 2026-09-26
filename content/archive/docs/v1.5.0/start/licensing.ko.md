---
title: 라이선스
seo_title: 'LibreYOLO 라이선스: 코드 및 가중치'
description: >-
  LibreYOLO 자체 코드는 MIT 라이선스입니다. 외부에서 가져온 상위 코드와 공개된 체크포인트는 자체 라이선스를 가지며, 그 중 여러
  개는 비상업적입니다.
lead: >-
  LibreYOLO는 세 가지 별도로 라이선스된 항목을 포함하고 있습니다: 자체 코드, 모델 계열에 포함된 상위 코드, 그리고 사전 학습된
  체크포인트입니다. 이들은 종종 동일한 라이선스를 갖고 있지 않습니다.
keywords:
  - libreyolo 라이선스
  - 컴퓨터 비전 라이브러리와 함께
  - 비상업용 모델 가중치
  - 모델 체크포인트 라이선스
  - apache-2.0 객체 탐지
last_verified: 1.5.0
source_hash: 83536fea4dc4eaec
---

## LibreYOLO 자체 코드

이 라이브러리는 MIT 라이선스입니다. 여기에는 Python API, CLI, 트레이너, 검증기 및 익스포터, 데이터셋 로더, 그리고 `weights/` 아래의 변환 스크립트가 포함됩니다. 이를 상업적 또는 폐쇄 소스 제품에서 사용하되, 배포하는 복사본에 저작권 표시와 라이선스 문구를 유지하면 되며, 그 의무는 여기서 끝납니다.

허용 범위는 코드에서 끝납니다. [`LICENSE`](https://github.com/LibreYOLO/libreyolo/blob/release/LICENSE) 파일은 이를 명확히 설명합니다:

> 이 라이선스는 다양하며 모두 허용적인 것은 아닙니다: 일부 공개된 가중치는
> 비상업적이거나 기타 제한된 용도로, 이 MIT 라이선스는 확장되지 않습니다
> 그들에게. 모델을 선택한다는 것은 그 라이선스를 선택하는 것을 의미합니다.

## 계열당 업스트림 코드

대부분의 계열는 출판된 연구의 포트이며, 여러 공급업체의 업스트림 소스를 직접 포함합니다. 공급된 파일은 원래의 저작권 헤더와 원래 라이선스를 유지합니다. MIT는 이를 덮어쓰지 않으며, LibreYOLO는 누구의 작업도 재라이선스하지 않습니다. Apache-2.0과 BSD-3-Clause가 가장 자주 언급되는 두 가지입니다.

Apache-2.0은 DETR 라인과 많은 트랜스포머 작업을 포함합니다: Meta AI(FAIR)의 DETR, SenseTime의 Deformable DETR, Baidu의 LW-DETR, Leilei Wang과 공동 저자의 OV-DEIM, LibreYOLO가 Hugging Face Transformers에서 포팅한 SegFormer 구현, PaddlePaddle 저자들의 PP-OCRv5, ETH 취리히 컴퓨터 비전 연구소의 SwinIR, ByteDance Seed의 Depth Anything 3. 또한 Ross Wightman과 timm 기여자들이 개발한 timm에서 파생된 분류기들도 포함하며, 여기에는 ResNet, DeiT, EfficientNetV2, MobileNetV4, Swin이 있으며, 해당 모듈 이름은 timm과 동일하게 하여 ImageNet 텐서가 그대로 로드되도록 합니다.

BSD-3-Clause는 torchvision에서 파생된 모든 것을 포함합니다: Faster R-CNN, Mask R-CNN, FCOS, RetinaNet, SSD300, AlexNet, VGG, FCN 및 DeepLabv3.

MIT는 Megvii의 NAFNet, Xingyi Zhou의 CenterNet, 그리고 MultimediaTechLab에서 Kin-Yiu Wong과 Hao-Tang Tsui가 다시 발표한 YOLOv7을 포함하는 더 작은 그룹을 다룹니다. YOLOv1부터 YOLOv4까지의 계열는 Joseph Redmon의 Darknet 프로젝트, 그리고 YOLOv4의 경우 Alexey Bochkovskiy에 의해 만든 아키텍처를 재현합니다. Darknet은 퍼블릭 도메인이기 때문에 이들에 대해서는 아무런 의무가 없습니다.

하나의 번들된 서브트리는 오픈 소스 라이선스가 아닙니다. DEIMv2 계열는 Meta Platforms의 DINOv3 백본 코드를 DINOv3 라이선스 계약에 따라 제공하며, 이는 맞춤형 비-OSI 라이선스입니다. 해당 코드를 재배포하는 경우 계약서를 함께 제공해야 하며, 계약서는 ITAR, 군사 또는 전쟁 목적, 핵 산업, 첩보 활동, 무기 개발과 관련된 활동에 대한 사용을 금지하고 있습니다. 이러한 조건은 해당 서브트리에만 적용됩니다.

저장소의 두 파일이 전체 그림을 담고 있습니다. [`NOTICE`](https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE)은 모든 번들된 서드파티 서브트리를 경로, 라이선스 파일 및 업스트림 소스와 함께 나열합니다. [`THIRD_PARTY_NOTICES.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/THIRD_PARTY_NOTICES.txt)은 LibreYOLO가 파생된 업스트림 프로젝트를 나열하고 각 라이선스 텍스트를 전체 그대로 재현합니다.

## 체크포인트별 가중치

패키지 안에는 사전 학습된 가중치 파일이 포함되어 있지 않습니다. 공개된 체크포인트는 [LibreYOLO 조직](https://huggingface.co/LibreYOLO) 아래 Hugging Face에 있으며, 각 저장소는 해당 가중치가 나온 프로젝트를 반영하는 자체 `LICENSE`와 저작권 표시를 포함하고 있습니다.

그 저장소가 용어에 대한 권위 있는 출처입니다. 이 페이지도 아니고, 모델 페이지도 아니며, 소스 트리의 요약도 아닙니다. 파일이 어떻게 이름이 붙여지고 어디에서 다운로드되는지에 대해서는 [체크포인트 및 가중치](/docs/weights)를 참조하십시오.

라이선스는 계열마다 다르며, 한 계열 안의 파일들 사이에서도 다릅니다. 두 번째 경우의 예시 두 가지:

- YOLO9 COCO 체크포인트는 MIT 라이선스입니다. VisDrone2019-DET에서 학습된 `LibreYOLO9P2s-visdrone.pt`는 CC BY-NC-SA 3.0 라이선스로, 이는 비상업적입니다.
- RF-DETR 검출 체크포인트는 Apache-2.0입니다. 오리엔티드 박스 체크포인트는 CC BY 4.0입니다. 이는 Roboflow Universe 데이터셋에서 파인튜닝되었으며, 해당 데이터셋은 CC BY 4.0으로 공개되었고, 가중치에는 해당 데이터셋의 저작자 표시 요구 사항이 그대로 적용되기 때문입니다.

계열 간에는 범위가 더 넓게 나타나며, 여러 공개된 체크포인트는 상업 제품에서 사용할 수 없습니다:

- SegFormer는 두 레이어 간의 가장 명확한 구분입니다. 구현은 Hugging Face Transformers 코드의 Apache-2.0 포트입니다. 공개된 ADE20K 체크포인트는 NVIDIA의 공개를 NVIDIA 소스 코드 라이선스 하에 변환한 것으로, 재배포를 허용하지만 사용을 비상업적 연구나 평가로 제한하며, 그 제한을 파생 작품에도 적용합니다. 해당 체크포인트는 LibreYOLO의 관대한 조건에는 포함되지 않습니다.
- OV-DEIM 체크포인트는 업스트림 저자가 확인한 CC BY-NC 4.0입니다. 모든 예측은 또한 Apple의 MobileCLIP-B(LT) 텍스트 타워를 로드하며, 이 라이선스는 사용을 연구 목적으로 제한하며, 이는 체크포인트 자체 라이선스보다 더 엄격한 조건입니다.
- SenseNova-Vision 코드는 Apache-2.0이며, 그 가중치는 CC BY-NC 4.0입니다. 로더는 모든 자동 다운로드 전에 비상업적 고지를 출력합니다.

일부 계열는 LibreYOLO에서 호스팅되는 체크포인트가 전혀 없으며, 해당 페이지의 Weights 행에 그렇게 표시되어 있습니다. SAM 3는 Meta의 맞춤형 SAM 라이선스 하에서 Hugging Face에서 제한적으로 제공되며, Meta에서 직접 다운로드됩니다. MiDaS 릴리스 파일은 공식 URL에서 가져오고 해시 검증을 수행하며, 재호스팅되지 않습니다. Dome-DETR는 업스트림에서 링크되어 있는데, 이는 모델 카드에는 메타데이터에 라이선스가 없다고 나와 있지만 설명 문서에서는 Apache-2.0을 주장하고 사용을 학술 연구로 제한하고 있어 서로 일치하지 않기 때문입니다. TEED와 DexiNed 아키텍처는 MIT 라이선스이지만, 저자가 공개한 체크포인트는 BIPED에서 학습되었으며, BIPED 데이터셋의 이용 약관은 비상업적용으로 제한되어 있어 LibreYOLO는 이를 포함하거나 자동 다운로드하지 않습니다.

여러 torchvision 체크포인트는 자체 라이선스 파일을 포함하지 않습니다. LibreYOLO는 이를 공개 프로젝트에서 사용하는 라이선스를 따라 미러링하며, 각 모델 카드에 기초가 체크포인트별로 부여되는 것이 아니라 암시된 것임을 명시하고, pretrained 모델의 조건이 학습 데이터에서 유래할 수 있다는 torchvision의 경고를 반복합니다.

## 한 모델에 대한 용어 찾기

모델 페이지의 헤더에는 `Code X, weights Y` 형식의 **라이선스** 항목이 있으며, 이는 페이지의 라이선스 섹션으로 연결됩니다. 해당 섹션에는 원본 작품과 저자, 상위 라이선스, 상위 소스, LibreYOLO 코드 라이선스, 가중치, 그리고 약관이 허용하는 내용에 대한 해석이 나와 있습니다. 같은 페이지의 체크포인트 테이블에는 **가중치 라이선스** 열이 있으며, 게시된 파일마다 한 행씩 표시되므로, 서로 다른 조건을 가진 계열는 파일별로 조건을 보여줍니다.

모든 것은 라이브러리가 확인되는 동일한 데이터에서 렌더링되므로, 이 페이지가 그것을 표로 반복하지 않는 이유입니다. 수동으로 작성한 라이선스 매트릭스는 한 버전에서 잘못될 수 있으며, 여기서의 잘못은 비용이 많이 듭니다.

소스 트리에서, 번들 코드에 대한 등가는 `NOTICE`이고, 업스트림 프로젝트 및 그들의 라이선스 텍스트에 대한 등가는 `THIRD_PARTY_NOTICES.txt`이며, 게시된 체크포인트의 계열별 요약에 대한 등가는 [`weights/LICENSE_NOTICE.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/weights/LICENSE_NOTICE.txt)입니다.

그럼 다운로드하려는 정확한 파일의 Hugging Face 저장소를 확인하십시오. 그것은 권위 있으며, 문서 페이지가 변경되지 않아도 변경될 수 있습니다.

## 상업적 사용

코드는 거의 문제가 되지 않습니다. MIT, Apache-2.0 및 BSD-3-Clause는 모두 상업적 및 폐쇠형 소프트웨어 사용을 허용합니다. 각 라이선스는 재배포하는 복사본과 함께 라이선스 텍스트와 저작권 표시를 유지할 것을 요구하며, Apache-2.0은 또한 특허 라이선스를 부여하고, 어느 것도 자체 애플리케이션 코드에 조건을 부과하지 않습니다.

체크포인트는 제품이 막히는 곳입니다. 비상업용 체크포인트는 주변 코드가 아무리 허용적이어도 계속 비상업용이며, 파일을 변환한다고 해서 적용되는 조건이 변하지 않으며, 이것이 `weights/LICENSE_NOTICE.txt`가 직접 명시한 내용입니다. 제한된 체크포인트에서 생성된 ONNX 또는 TensorRT 아티팩트는 그 제한을 계승합니다.

라이선스가 NVIDIA 소스 코드 라이선스처럼 파생 작업에 제한을 적용하는 경우, 파인튜닝 또한 그 제한을 벗어나지 못합니다. 사용할 권리가 있는 데이터로 동일한 아키텍처를 처음부터 학습시키는 경우에는 예외입니다: 코드는 허용적이므로 직접 학습한 모델에는 사전 학습된 체크포인트의 조건이 적용되지 않습니다. SegFormer 페이지에서는 자체 가중치에 대해 이를 명시하고 있으니, 배포할 계열의 페이지에서 '해석(Interpretation)' 항목을 확인하십시오.

모델을 선택할 때 라이선스 문제를 결정하고, 배포할 때가 아니라 실제로 다운로드한 파일의 약관을 읽으십시오. 하나의 허용적인 체크포인트가 있는 계열에도 그 옆에 제한적인 체크포인트가 있을 수 있기 때문입니다.

## 법률 상담 아님

이 페이지에서는 관련된 라이선스에 대해 설명합니다. 이는 설명일 뿐 법적 조언이 아니며, 어떠한 보증도 생성하지 않습니다. 상업적으로 중요한 경우에는 직접 라이선스를 읽고 스스로 판단하십시오.
