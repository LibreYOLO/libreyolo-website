---
title: 라이선스
seo_title: 'LibreYOLO 라이선스: 코드 및 가중치'
description: >-
  LibreYOLO 자체 코드는 MIT 라이선스입니다. 포함된 업스트림 코드와 공개된 체크포인트는 각각 자체 라이선스를 가지며, 그중 일부는
  비상업적입니다.
lead: >-
  LibreYOLO는 세 가지 별도로 라이선스된 항목을 가지고 있습니다: 자체 코드, 모델 패밀리로 포함된 업스트림 코드, 사전 학습된
  체크포인트. 이들은 종종 동일한 라이선스가 아닙니다.
keywords:
  - libreyolo 라이선스
  - MIT 컴퓨터 비전 라이브러리
  - 비상업적 모델 가중치
  - 모델 체크포인트 라이선스
  - Apache-2.0 객체 탐지
last_verified: 1.5.0
source_hash: 83536fea4dc4eaec
---

## LibreYOLO 자체 코드

라이브러리는 MIT 라이선스입니다. 여기에는 Python API, CLI, 트레이너, 검증기 및 익스포터, 데이터셋 로더, `weights/` 하위의 변환 스크립트가 포함됩니다. 상업용 또는 폐쇄형 소프트웨어에서 사용할 수 있으며, 배포하는 모든 복사본에 저작권 라인과 라이선스 텍스트를 그대로 유지하면 의무는 거기서 끝납니다.

권한은 코드에서만 적용됩니다. [`LICENSE`](https://github.com/LibreYOLO/libreyolo/blob/release/LICENSE) 파일에서 명확하게 설명하고 있습니다:

> 해당 라이선스는 모두 허용적인 것은 아니며, 일부 공개된 가중치는
> 비상업적이거나 기타 제한이 있으며, 이 MIT 라이선스는 그들에게 적용되지 않습니다. 모델을 선택하는 것은 해당 모델의 라이선스를 선택하는 것입니다.
> 정의되지 않음

## 업스트림 코드는 패밀리별

대부분의 패밀리는 공개된 연구의 포트이며, 몇몇 공급업체의 업스트림 소스를 직접 사용합니다. 공급된 파일은 원래의 저작권 헤더와 원래 라이선스를 유지합니다. MIT는 그것을 덮어쓰지 않으며, LibreYOLO는 누구의 작업도 재라이선스하지 않습니다. Apache-2.0과 BSD-3-Clause가 가장 자주 언급되는 두 가지입니다.

Apache-2.0은 DETR 라인과 대부분의 트랜스포머 작업을 포함합니다: Meta AI(FAIR)의 DETR, SenseTime의 Deformable DETR, Baidu의 LW-DETR, Leilei Wang과 공동 저자의 OV-DEIM, Hugging Face Transformers에서 LibreYOLO가 포팅한 SegFormer 구현, PaddlePaddle 저자의 PP-OCRv5, ETH Zurich 컴퓨터 비전 실험실의 SwinIR, 그리고 ByteDance Seed의 Depth Anything 3이 포함됩니다. 또한 Ross Wightman과 timm 기여자들로부터 파생된 분류기들도 포함되며, 여기에는 ResNet, DeiT, EfficientNetV2, MobileNetV4, Swin이 있으며, 해당 모듈 이름은 timm과 동일하게 구성되어 ImageNet 텐서가 변경 없이 로드됩니다.

BSD-3-Clause는 torchvision에서 파생된 모든 것을 포함합니다: Faster R-CNN, Mask R-CNN, FCOS, RetinaNet, SSD300, AlexNet, VGG, FCN 및 DeepLabv3.

MIT는 Megvii의 NAFNet, Xingyi Zhou의 CenterNet, 그리고 MultimediaTechLab의 Kin-Yiu Wong과 Hao-Tang Tsui가 재배포한 YOLOv7을 포함한 더 작은 그룹을 다룹니다. YOLOv1부터 YOLOv4까지의 계열은 Joseph Redmon이 만든 Darknet 프로젝트의 아키텍처를 재현하며, YOLOv4의 경우 Alexey Bochkovskiy가 참여했습니다. Darknet은 퍼블릭 도메인이므로, 이에 따른 의무는 전혀 없습니다.

하나의 번들 서브트리는 오픈 소스 라이선스가 아닙니다. DEIMv2 패밀리는 Meta Platforms의 DINOv3 백본 코드를 DINOv3 라이선스 계약 하에 제공합니다. 이 계약은 맞춤형 비 OSI 라이선스입니다. 해당 코드를 재배포하려면, 그와 함께 계약서 사본을 배포해야 하며, 계약서는 ITAR, 군사 또는 전쟁 목적, 원자력 산업, 첩보 활동, 무기 개발에 사용을 금지합니다. 이러한 조건은 해당 서브트리에만 적용됩니다.

저장소의 두 파일에 전체 그림이 포함되어 있습니다. [`NOTICE`](https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE)에는 경로, 라이선스 파일, 업스트림 소스와 함께 번들된 모든 서드파티 서브트리가 나열되어 있습니다. [`THIRD_PARTY_NOTICES.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/THIRD_PARTY_NOTICES.txt)에는 LibreYOLO가 파생된 업스트림 프로젝트가 나열되어 있으며 각 라이선스 텍스트를 전체적으로 재현합니다.

## 체크포인트별 가중치

패키지 내에는 사전 학습된 가중치 파일이 포함되어 있지 않습니다. 공개된 체크포인트는 [LibreYOLO 조직](https://huggingface.co/LibreYOLO) 하의 Hugging Face에 있으며, 각 저장소에는 해당 가중치가 출처가 되는 프로젝트를 반영하는 `LICENSE`와 저작권 표시가 포함되어 있습니다.

해당 저장소가 약관의 권위 있는 출처입니다. 이 페이지도, 모델 페이지도, 소스 트리 요약도 아닙니다. 파일 이름과 다운로드 위치에 대해서는 [체크포인트 및 가중치](/docs/weights)를 참조하십시오.

라이선스는 계열마다 다르며, 하나의 계열 내 파일마다도 다릅니다. 두 번째 예제 두 가지:

- YOLO9 COCO 체크포인트는 MIT 라이선스입니다. VisDrone2019-DET로 학습된 `LibreYOLO9P2s-visdrone.pt`는 비상업적용 CC BY-NC-SA 3.0입니다.
- RF-DETR 검출 체크포인트는 Apache-2.0입니다. 방향 상자 체크포인트는 CC BY 4.0인데, 이는 CC BY 4.0 하에 게시된 Roboflow Universe 데이터셋으로 파인튜닝되었고, 가중치가 해당 데이터셋의 속성 요구 사항을 이어받기 때문입니다.

계열 간에는 범위가 더 넓어지며, 몇몇 공개된 체크포인트는 상업 제품에서 사용할 수 없습니다:

- SegFormer는 두 층 사이의 가장 명확한 구분입니다. 구현은 Hugging Face Transformers 코드의 Apache-2.0 포트입니다. 공개된 ADE20K 체크포인트는 NVIDIA의 소스 코드 라이선스 하에서 NVIDIA가 공개한 자료에서 변환된 것으로, 재배포는 허용되지만 사용은 비상업적 연구 또는 평가로 제한되며, 이러한 제한은 파생 작품에도 적용됩니다. 해당 체크포인트는 LibreYOLO의 관대한 이용 조건에는 포함되지 않습니다.
- OV-DEIM 체크포인트는 CC BY-NC 4.0이며 상류 저자가 확인한 것입니다. 모든 예측은 또한 Apple의 MobileCLIP-B(LT) 텍스트 타워를 로드하며, 그 라이선스는 사용을 연구 목적으로만 제한하고 있어 체크포인트 자체 라이선스보다 더 엄격합니다.
- SenseNova-Vision 코드는 Apache-2.0이며, 가중치는 CC BY-NC 4.0입니다. 로더는 모든 자동 다운로드 전에 비상업적 사용 알림을 출력합니다.

일부 계열들은 LibreYOLO에서 호스팅하는 체크포인트가 전혀 없으며, 해당 페이지의 Weights 행에 그렇게 명시되어 있습니다. SAM 3는 Meta의 맞춤 SAM 라이선스 하에 Hugging Face에서 접근 제한되어 있으며 Meta에서 직접 다운로드됩니다. MiDaS 릴리스 자산은 공식 URL에서 가져오고 해시 검증을 거치며, 재호스팅하지 않습니다. Dome-DETR은 모델 카드에 메타데이터 상에서 라이선스가 명시되어 있지 않지만, 설명에는 Apache-2.0을 명시하고 사용을 학술 연구로 제한한다고 되어 있어 상충되므로 상류 링크로 연결됩니다. TEED와 DexiNed 아키텍처는 MIT 라이선스이지만, 저자들이 공개한 체크포인트는 BIPED에서 학습되었으며, 그 데이터셋 사용 조건은 비상업적 사용으로 제한되어 있어 LibreYOLO에서는 이를 번들로 제공하거나 자동 다운로드하지 않습니다.

여러 torchvision 체크포인트 중 일부는 자체 라이선스 파일을 포함하고 있지 않습니다. LibreYOLO는 이를 출시 프로젝트에서 사용하는 라이선스를 기준으로 미러링하며, 각 모델 카드에 체크포인트별로 라이선스가 부여되기보다 암묵적으로 적용된다는 점을 명시하고, 사전 학습 모델의 조건이 학습 데이터에서 비롯될 수 있다는 torchvision 자체 경고를 반복합니다.

## 한 모델의 조건을 찾기

모델 페이지의 헤더에는 `Code X, weights Y` 형태의 **Licenses** 행이 있으며, 이는 페이지의 라이선스 섹션으로 연결됩니다. 해당 섹션에는 원본 작업과 저자, 상위 라이선스, 상위 소스, LibreYOLO 코드 라이선스, 가중치, 조건이 허용하는 사항에 대한 해석이 나열되어 있습니다. 동일 페이지의 체크포인트 테이블에는 **Weights license** 열이 있으며, 게시된 파일별로 한 행씩 있어, 조건이 혼합된 패밀리는 파일별로 표시됩니다.

이 모든 것은 라이브러리가 검증되는 동일한 데이터에서 렌더링되므로, 이 페이지에서는 이를 표로 반복하지 않습니다. 직접 입력한 라이선스 매트릭스는 한 릴리스 내에서 잘못될 수 있으며, 여기서 잘못되면 비용이 많이 듭니다.

소스 트리에서 동등한 항목은 번들 코드의 경우 `NOTICE`, 업스트림 프로젝트 및 그들의 라이선스 텍스트의 경우 `THIRD_PARTY_NOTICES.txt`, 그리고 게시된 체크포인트의 패밀리별 요약은 [`weights/LICENSE_NOTICE.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/weights/LICENSE_NOTICE.txt)입니다.

그런 다음 다운로드하려는 정확한 파일의 Hugging Face 저장소를 확인하십시오. 이는 권위 있는 자료이며, 문서 페이지가 변경되지 않아도 변경될 수 있습니다.

## 상업적 사용

코드는 문제가 되는 경우가 거의 없습니다. MIT, Apache-2.0 및 BSD-3-Clause 모두 상업적 및 폐쇄 소스 사용을 허용합니다. 각각은 배포하는 복사본과 함께 라이선스 텍스트와 저작권 표시를 유지하도록 요구하며, Apache-2.0은 특허 라이선스도 부여하고, 어느 것도 자신의 애플리케이션 코드에 조건을 부과하지 않습니다.

체크포인트는 제품이 막히는 지점입니다. 비상업적 체크포인트는 주변 코드가 얼마나 허용적이든 상관없이 비상업적 상태를 유지하며, 파일을 변환한다고 해서 적용되는 조건이 바뀌지 않습니다. 이는 `weights/LICENSE_NOTICE.txt`에서 직접 명시한 내용입니다. 제한된 체크포인트에서 만든 ONNX 또는 TensorRT 아티팩트는 해당 제한을 그대로 상속받습니다.

라이선스가 NVIDIA 소스 코드 라이선스처럼 파생 저작물에도 제한을 적용할 경우, 파인튜닝도 예외가 되지 않습니다. 데이터 사용 권한이 있는 데이터를 이용해 동일한 아키텍처를 처음부터 학습시키는 경우에는 다릅니다: 코드는 허용적이므로, 직접 학습한 모델은 당신의 것이며, 사전학습 체크포인트의 조건은 여기에 적용되지 않습니다. SegFormer 페이지는 자체 가중치에 대해 이를 명확히 설명하고 있습니다; 배포를 계획하는 해당 계열 페이지의 Interpretation 행을 읽으세요.

라이선스 문제는 모델을 선택할 때 결정하고, 실제로 다운로드한 파일의 조건을 읽으세요. 한 계열에 허용적 체크포인트가 하나 있다고 해서 제한된 체크포인트가 없다는 뜻은 아닙니다.

## 법적 조언이 아닙니다

이 페이지는 관련된 라이선스에 대해 설명합니다. 이는 설명일 뿐 법적 조언이 아니며, 어떠한 보증도 생성하지 않습니다. 상업적으로 중요한 경우, 직접 라이선스를 읽고 스스로 조언을 구하세요.
