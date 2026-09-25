---
title: "2026년 Edge AI NPU 기업 69곳: 칩, SDK, YOLO 지원 현황"
description: "칩, SDK, 컴파일 산출물, 양자화, 문서로 확인된 컴퓨터 비전 및 YOLO 지원을 기준으로 69개 엣지 AI 기업과 NPU 생태계를 정리한 출처 기반 가이드입니다."
date: 2026-08-15
author: Xuban
tags: [edge-ai, npu, computer-vision, yolo, hardware, ai-accelerators, amlogic, hailo, rockchip]
faq:
  - q: "엣지 AI NPU 기업은 몇 곳입니까?"
    a: "제품 공급업체, IP 라이선스 업체, 스마트 센서, GPU, 비공개 자동차 ASIC의 범위가 겹치므로 보편적인 기업 수는 없습니다. 이 비망라형 가이드는 2026년 8월 기준으로 관련 엣지 AI 칩, 가속기 플랫폼, 라이선스 가능한 NPU IP 또는 근거가 있는 주시 대상 실리콘을 보유한 기업과 플랫폼 생태계 69곳을 추적합니다."
  - q: "NPU란 무엇입니까?"
    a: "신경망 처리 장치(neural processing unit)는 합성곱과 행렬 곱셈 같은 신경망 연산에 특화된 하드웨어입니다. 업체마다 NPU, AIPU, BPU, KPU, DLA, HTP, TPU, MLA 등의 용어를 사용하며 컴파일된 모델은 대체로 업체 간에 이식할 수 없습니다."
  - q: "YOLO 모델을 공식 지원하는 NPU 기업은 어디입니까?"
    a: "Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Raspberry Pi의 공식 AI Camera 저장소를 통한 Sony IMX500, STMicroelectronics, Renesas, Lattice, Himax, Microchip 등은 적어도 한 YOLO 세대에 대해 자사 모델 동물원 항목, 튜토리얼 또는 검증 결과를 제공합니다. 정확한 세대, 작업, 칩 대상, SDK 버전은 여전히 확인해야 합니다."
  - q: "아무 ONNX 모델이나 모든 NPU에서 실행할 수 있습니까?"
    a: "아닙니다. ONNX는 교환 형식이지 하드웨어 호환성을 보장하지 않습니다. NPU 컴파일러는 그래프의 모든 연산자, 텐서 모양, 데이터 형식을 지원해야 합니다. 정적 모양, 그래프 분할, 사용자 지정 연산, CPU 폴백이 흔합니다."
  - q: "YOLO에 가장 적합한 NPU는 무엇입니까?"
    a: "모든 경우에 앞서는 제품은 없습니다. Hailo, Rockchip, Axelera AI, DEEPX, Amlogic은 문서로 확인되는 YOLO 지원 범위가 넓고, Qualcomm은 특히 다양한 기기에서 사용할 수 있습니다. 양자화 후 정확도, 전체 파이프라인 지연 시간, 지속 전력, 가격, 공급 상황, SDK 접근성, 운영체제 지원을 기준으로 선택하십시오."
  - q: "NPU TOPS 수치를 직접 비교할 수 없는 이유는 무엇입니까?"
    a: "업체마다 정밀도, 희소 연산, 곱셈 누산, 최대 활용률 가정을 다르게 계산할 수 있습니다. TOPS에는 메모리 트래픽, 미지원 연산자, 전처리, 후처리, 호스트 오버헤드가 반영되지 않습니다. 같은 모델, 정밀도, 해상도, 정확도, 전체 파이프라인을 비교해야 합니다."
  - q: "NPU 보정(calibration)이란 무엇입니까?"
    a: "보정은 보통 레이블이 없는 대표 배포 이미지를 모델에 입력해 컴파일러가 INT8 또는 다른 저정밀도 형식의 활성화 범위를 추정하게 하는 과정입니다. 무작위이거나 대표성이 없는 이미지로도 컴파일 산출물이 만들어질 수 있지만 작업 정확도는 낮아질 수 있습니다."
  - q: "LibreYOLO는 엣지 NPU를 지원합니까?"
    a: "LibreYOLO는 ONNX와 여러 런타임 형식으로 내보내며, 일부 Rockchip 모델에는 직접 RKNN 컴파일 경로를 제공하고 외부 Hailo 컴파일 절차도 문서화합니다. 그 밖의 모든 벤더 네이티브 산출물에는 해당 벤더 SDK가 필요합니다. 하드웨어에서 컴파일, 검증, 실행하기 전까지는 통합 후보라고 설명해야 합니다."
---

**단일한 NPU 시장은 없습니다. 이 비망라형 가이드는 배포 가능한 칩 또는 플랫폼을 보유한 업체 51곳, NPU IP 공급업체 9곳, 명확히 주시 대상으로 표시한 기업 9곳 등 기업과 플랫폼 생태계 69곳을 추적합니다.** 이들은 서로 호환되지 않는 여러 가속기 유형과 거의 그만큼 다양한 컴파일러 스택에 걸쳐 있습니다. 대부분 비슷한 절차를 약속합니다. 모델을 내보내고, 양자화하고, 컴파일한 뒤, 독점 산출물을 보드에 복사해 벤더 런타임을 호출합니다. 이 절차가 오후 만에 끝날지 엔지니어링 분기 하나가 걸릴지는 세부 사항에 달려 있습니다.

이 가이드는 2026년에 신뢰할 만한 컴퓨터 비전 하드웨어를 제공하는 기업을 정리합니다. 관련 칩, SDK와 컴파일러 이름, 배포 산출물, 공개 접근 수준, 벤더가 실제로 문서화한 모델을 기록합니다. 특히 Amlogic의 최신 ADLA 스택은 구형 A311D NPU 생태계와 실질적으로 다르므로 별도로 살펴봅니다.

> **조사 범위:** 출처는 2026년 8월 15일에 확인했습니다. 특정 모델에 관한 주장은 별도 표시가 없는 한 자사 제품 페이지, 문서, 모델 동물원, 저장소 또는 벤더 튜토리얼을 근거로 합니다. 광고된 TOPS는 벤더 수치이며 서로 직접 비교할 수 있는 독립 벤치마크 결과가 아닙니다. 이 글은 개발자 가이드이며 투자 조언이나 유료 순위가 아닙니다.

**빠른 탐색:** [기업 표](#npu-companies-and-platforms-at-a-glance) | [Amlogic 심층 분석](#amlogic-two-npu-generations-not-one) | [벤더 프로필](#the-strongest-public-deployment-ecosystems) | [컴파일 산출물 표](#what-you-actually-deploy-the-artifact-lock-in-table) | [도구 접근성과 제품 수명 주기](#tool-access-and-lifecycle-signals) | [LibreYOLO 로드맵](#what-this-means-for-libreyolo)

## 가장 중요한 발견

하드웨어는 파편화되어 있지만 배포 패턴은 놀라울 정도로 일관됩니다.

```text
PyTorch checkpoint
    -> ONNX or TFLite interchange graph
    -> representative calibration data
    -> vendor quantizer and graph compiler
    -> chip-specific binary or model package
    -> vendor runtime on the target
    -> application preprocessing, decode, NMS and rendering
```

[ONNX는 런타임, 코드 생성기, 하드웨어 구현을 명시적으로 허용](https://onnx.ai/onnx/repo-docs/IR.html)하지만 ONNX 파일은 전달 지점일 뿐입니다. 모든 ONNX 연산자가 모든 가속기에 매핑된다는 뜻은 아닙니다. 정적 입력 모양, 지원 연산자 제약, 양자화 규칙, 호스트 폴백에 따라 실제 NPU 실행 범위가 결정됩니다.

따라서 소프트웨어 스택도 칩의 일부입니다. 명목상 빠른 NPU라도 컴파일러가 제한적이거나 불안정하면 공개 툴체인, 모델 동물원, 시뮬레이터, 안정적인 런타임을 갖춘 더 작은 장치보다 배포 대상으로 불리할 수 있습니다.

## 이 가이드에서 "지원"의 의미

벤더 자료는 *지원*이라는 말을 매우 느슨하게 사용합니다. 이 글은 근거를 네 단계로 구분합니다.

| 근거 수준 | 입증하는 내용 | 입증하지 않는 내용 |
|---|---|---|
| **검증된 모델** | 벤더가 칩별 모델 항목, 결과 또는 호환성 표를 공개함 | 수정한 체크포인트도 동일한 정확도를 유지함 |
| **공식 예제** | 벤더가 이름이 명시된 모델의 종단 간 튜토리얼 또는 데모를 제공함 | 다른 크기, 작업, 세대도 컴파일됨 |
| **컴파일러 기능** | SDK가 프레임워크를 가져오거나 지원 연산자를 공개함 | 특정 YOLO 그래프가 종단 간 작동함 |
| **마케팅 또는 접근 제한** | 제품이 비전을 대상으로 하며 비공개 SDK가 존재함 | 공개적으로 재현 가능한 모델 호환성 |

이 구분은 중요합니다. "ONNX를 가져옵니다"는 "YOLO11 분할을 지원합니다"와 같지 않습니다. 모델을 분석하더라도 CPU로 폴백하거나, 출력 수치가 잘못된 채 컴파일되거나, 가속기 메모리를 초과하거나, 양자화 과정에서 정확도가 낮아질 수 있습니다.

## 한눈에 보는 NPU 기업 및 플랫폼

표에는 SoC, 개별 가속기, 스마트 센서, 인접한 GPU/FPGA 대상을 함께 포함했습니다. NPU, AIPU, BPU, KPU, DLA, HTP, TPU, MLA 등 업체별 명칭이 달라도 같은 배포 결정을 두고 경쟁하기 때문입니다.

### 임베디드 SoC, 스마트 센서, 산업용 프로세서

| 기업 | 관련 칩 또는 플랫폼 | SDK, 컴파일러, 산출물 | 공개적으로 문서화된 컴퓨터 비전 근거 | 접근성 |
|---|---|---|---|---|
| [Amlogic](https://github.com/Amlogic-NN/amlnn-toolkit) | A311D2, S928X, S905X5/S905D5, A311Y3, C308L/C302X/C302X2, T968D4, C305X2, A123X | AMLNN Toolkit, `libnnsdk.so`, 컴파일된 `.adla`; A311D용 별도 레거시 Acuity `.nb` 스택 | [모델 플레이그라운드](https://github.com/Amlogic-NN/amlnn-model-playground)는 A311D2, S905X5, A311Y3, C305X2, A123X에서 YOLOv5/6/7/8/10/11, YOLOX, YOLOE, YOLO-World, PP-YOLOE, 분할, 자세 추정, OBB를 문서화합니다. 여기에 나열한 다른 칩은 해당 지원 표 항목이 아니라 컴파일 대상입니다. | 공개 GitHub와 휠 제공, 호환 BSP/드라이버 필요 |
| [Rockchip](https://github.com/airockchip/rknn-toolkit2) | RK3562/3566/3568, RK3576, RK3588, RV1126B | RKNN-Toolkit2, RKNN Runtime, `.rknn` | [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)의 YOLOv5/6/7/8/10/11, YOLOX, YOLO-World, PP-YOLOE, 분할, 자세 추정, OBB | 공개 GitHub, 바이너리 휠 |
| [Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | Snapdragon 및 Dragonwing 플랫폼, QCS6490, QCS8550, Dragonwing IQ-9075 | QAIRT/QNN, SNPE, AI Hub; QNN 컨텍스트 바이너리 또는 DLC | [AI Hub 모델 컬렉션](https://github.com/qualcomm/ai-hub-models)에 YOLOv3/5/6/7/8/9/10/11/26, YOLOX, YOLO-World, YOLOR, RF-DETR, 탐지/분할/자세 모델 포함 | 문서는 공개, SDK/계정 요구사항은 다름 |
| [Texas Instruments](https://github.com/TexasInstruments/edgeai-tidl-tools) | AM62A, AM67A, AM68A, AM69A, TDA4x; 프리뷰 TDA54-Q1 | Processor SDK Edge AI, TIDL | [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo)는 현재 AM6xA/TDA4 경로용 최적화 탐지, 분할, 자세 추정, 분류 모델을 공개합니다. 프리뷰 TDA54-Q1용 대상 표는 아직 아닙니다. | 공개 GitHub 및 Processor SDK |
| [NXP](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | i.MX 8M Plus, i.MX 93, i.MX 95, 인수한 Kinara Ara-1/Ara240 | TIM-VX, Vela, Neutron Converter, Ara SDK를 포함한 eIQ | [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo)에 YOLOv4-tiny/v8, NanoDet, CenterNet, FastestDet, SSD Lite, YOLACT 산출물 또는 레시피 포함. YOLOv5 항목은 문서만 제공 | 공개 및 계정 제한 구성요소 혼재 |
| [STMicroelectronics](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) | Neural-ART 가속기를 탑재한 STM32N6x7 변형, STM32N657/647 포함 | STM32Cube AI Studio, ST Edge AI Core | 현재 서비스 저장소에 Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26, ST-YOLOX, 자세 추정, 분할 포함 | 도구와 저장소 공개 |
| [Infineon](https://documentation.infineon.com/psocedge/) | Cortex-M55와 Ethos-U55를 탑재한 PSOC Edge E83/E84; M33 측에는 별도 NNLite 가속기 | ModusToolbox, [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter), TFLite Micro, Arm Vela | 아키텍처 자료는 대표 커널로 MobileNetV1/V2를 들고 E84 AI Kit에 카메라가 포함되지만, PSOC Edge YOLO 검증 표는 자사 자료에서 찾지 못했습니다. | 공개 문서, SDK 구성요소, 현재 E84 평가 키트 |
| [Renesas](https://github.com/renesas-rz/rzv_drp-ai_tvm) | RZ/V2L, V2M, V2MA, V2H, V2N | DRP-AI Translator, DRP-AI TVM, RUHMI | [RZ/V2H 모델 목록](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md)은 YOLOv5/v8/v11/26 크기, YOLOX, 자세 추정, 분할 변형을 검증합니다. | 공개 GitHub 및 보드 SDK |
| [Sony](https://www.aitrios.sony-semicon.com/edge-ai-devices/imx500) | IMX500 지능형 비전 센서, Raspberry Pi AI Camera | Edge-MDT, Model Compression Toolkit, IMX500 packer; `.rpk` 패키지 | [Raspberry Pi IMX500 동물원](https://github.com/raspberrypi/imx500-models)에 YOLOv8n, YOLO11n, EfficientDet Lite, NanoDet+, SSD 포함 | Raspberry Pi 흐름 공개, Sony 도구 사용 약관 적용 |
| [Ambarella](https://www.ambarella.com/developer/model-garden/) | CV72/CV75, CV5/CV52, CV3-AD 계열 | Cooper Developer Platform, CVflow 컴파일러, 런타임 DAG | 공개 모델 정원에 YOLOX-S, RTMDet-nano, DeepLabV3+, TopFormer, OWL-ViT, LLaVA OneVision 명시 | 주로 파트너 전용 |
| [Synaptics](https://developer.synaptics.com/docs/sl/overview) | Astra SL1600/SL1680, SL2611/13/15/17/19; 별도 SR100 MCU 계열 | SL16xx의 SyNAP `.synap`; SL261x의 Torq/IREE `.vmfb`; 별도 Ethos-U55 기반 SR SDK | 공식 YOLOv8n/v8s SyNAP 벤치마크, SL261x Torq YOLOv8 예제. SR 근거는 별도 평가 필요 | 개발자 포털, 일부 다운로드 제한 |
| [MediaTek](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) | NP8, MDLA 5.3 기반 Genio 360/360P/420/520/720; 이전 NP6/MDLA 기반 Genio 510/700/1200 | NeuroPilot Converter, `ncc-tflite`, Neuron Runtime, `.dla`; 일부 최신 시스템의 ONNX Runtime 경로 | 공식 IoT AI Hub가 YOLOv5 및 YOLOv8 모델 페이지와 벤치마크, 분류/얼굴 모델을 게시합니다. | Yocto 문서 공개, 주요 NP8 번들 및 Android 자료는 직접 고객/NDA 접근 필요 |
| [Allwinner](https://docs.aw-ol.com/v853/en/npu/dev_npu/) | 1-TOPS Vivante NPU가 있는 V853 비전 SoC | Acuity/Pegasus 변환, `viplite` 런타임 | 공식 V853 자료에 YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet, 얼굴/사람 네트워크 명시 | 문서 공개, SDK 패키지 다운로드에 계정이 필요할 수 있음 |
| [Canaan/Kendryte](https://github.com/kendryte/nncase) | 현행 K230/K230D, 레거시 K210/K510 KPU | `nncase` 컴파일러와 런타임; `.kmodel` | K230 nncase 가이드에서 YOLOv5s 컴파일/시뮬레이션/런타임을 다룹니다. 공식 데모 카탈로그와 최신 [CanMV 변경 기록](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md)은 YOLOv8/11/26 작업을 문서화합니다. | 컴파일러/PyPI 공개, 칩 플러그인은 바이너리 |
| [Alif Semiconductor](https://alifsemi.com/support/kits/ensemble-e7appkit/) | Ethos-U55를 포함한 ML 가속기 2개가 있는 Ensemble E7; Ethos-U85 1개와 Ethos-U55 NPU 2개가 있는 [E8](https://alifsemi.com/ensemble-e8-series/) | Arm Vela, TFLite Micro 기반 배포 | Alif는 제품군 수준 INT8 YOLO-Fastest 얼굴 탐지 벤치마크를 공개하지만, 최신 YOLO의 폭넓은 칩별 표는 없습니다. | 공개 문서와 평가 키트 자료 |
| [Himax](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) | Cortex-M55, Ethos-U55가 있는 HX6538 WiseEye2 엣지 AI MCU | CMSIS-NN 및 참조 커널 폴백을 포함한 TFLite Micro, Arm Vela | 공식 [WiseEye2 예제](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2)에 YOLOv8n 탐지/자세 추정/분류, YOLO11n 탐지, 얼굴 메시, PeopleNet 포함 | 공개 GitHub와 문서, 파트너 보드 구매 가능 |
| [Analog Devices](https://www.analog.com/en/products/max78002.html) | 저전력 CNN 가속기가 있는 MAX78000/MAX78002 AI MCU | [`ai8x-training`](https://github.com/analogdevicesinc/ai8x-training), `ai8x-synthesis`/`izer`, MSDK; 생성된 C 코드와 가중치 | 공식 자료에서 얼굴 식별/탐지, RetinaNet, Visual Wake Words, 분류기, 동작 인식 지원. 자사 YOLO 배포는 찾지 못했습니다. | 공개 GitHub 도구, 문서, 평가 보드 |
| [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo) | RDK X3/X5/Ultra 및 최신 S100 계열 BPU 보드 | OpenExplorer/Algorithm Toolchain, `hbm_runtime`; X5 `.bin`, 최신 `rdk_s` `.hbm` | 현재 `rdk_x5` 브랜치는 RDK X5용 YOLOv5/v5u/v8/v9/v10/11/12/13/26, YOLOE, YOLO-World, 분할, 자세 추정, 분류, OCR, CLIP을 문서화합니다. X3와 S 시리즈에는 별도 브랜치가 있습니다. | 공개 GitHub, 일부 툴체인 패키지는 플랫폼별 제공 |
| [AXERA](https://github.com/AXERA-TECH/ax-samples) | AX650, AX637, AX630C, AX620Q, AX615 | Pulsar2 컴파일러, AXEngine; `.axmodel` | 현재 공식 샘플에서 YOLOv5/6/7/8/9/10/11/13/26, YOLOX, YOLO-World 지원. 작업과 대상 칩은 다릅니다. | 샘플과 문서 공개, 컴파일러 별도 배포 |
| [SOPHGO](https://github.com/sophgo/tpu-mlir) | BM1684/1684X/1688/1690, CV186X/CV18xx | TPU-MLIR, SOPHON 런타임; `.bmodel` 또는 `.cvimodel` | 공식 데모에서 YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, OBB, 분할, 얼굴, SAM, OCR 지원 | 오픈 소스 컴파일러와 벤더 런타임 |
| [Huawei Ascend](https://www.hiascend.com/en/software/cann) | Ascend 310/310P/310B, Atlas 200I/300I 엣지 제품 | CANN, ATC 컴파일러, AscendCL; `.om` | 공식 [Ascend ModelZoo](https://github.com/Ascend/modelzoo)에 YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN, 분할 모델 포함 | 문서 공개, CANN 패키지와 커널은 대상에 맞아야 함 |
| [Cambricon](https://github.com/Cambricon/magicmind_cloud) | 인용된 공개 표의 MLU370-X4/S4; 해당 표에는 없는 레거시 MLU270 | Neuware, MagicMind | 저장소의 MagicMind 1.7 표에 YOLOv3/4/5/7/8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN, 분할 모델 기재 | 과거 모델 예제 공개, 현재 SDK/컨테이너 접근 제한 |
| [Sunplus](https://sunplus.atlassian.net/wiki/spaces/C3/pages/1971126471/SP7350%2BSpecification) | 약 4.1~4.6 TOPS를 광고하는 SP7350/C3V | Vivante Acuity NPU Docker, SNNF 런타임, `.nb` | 공식 문서에서 YOLOv5와 사용자 지정 종단 간 [YOLOv8 배포](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) 제공 | 공개 소스/문서 및 보드 생태계 |
| [ESWIN Computing](https://www.eswincomputing.com/en/news/info/96.html) | EIC7700/EIC7700X, 듀얼 다이 [EIC7702/EIC7702X](https://www.eswincomputing.com/en/news/info/104.html) RISC-V SoC | ENNP: EsQuant, 최신 EsAAC 컴파일러, 시뮬레이터, ESSDK 런타임; `.model`; 이전 문서는 `ennc-compile` 사용 | Milk-V 호스팅 ENNP 문서에서 종단 간 [YOLOv3 튜토리얼](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3), MobileNetV2, ResNet 예제 제공 | 자사 및 보드 제공 문서 혼재, 지역별 다운로드 |
| [Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m55-mcus/m55m1-series/index.html) | Ethos-U55-256이 있는 M55M1 MCU | TFLite Micro, Arm Vela, NuEdgeWise/NuML | 공식 [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) 예제에서 YOLOv8-nano, YOLOX-nano, YOLO Fastest, SSD-MobileNet, 분류 모델 언급 | 대부분 MCU 툴체인 공개 |
| [Realtek](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | AmebaPro2 RTL8735B / AMB82-mini 카메라 플랫폼 | Arduino/FreeRTOS SDK, VoE, NeuralNetwork API; `.nb` | 패키지 모델에 YOLOv3-tiny, YOLOv4-tiny, YOLOv7-tiny, SCRFD, MobileFaceNet 포함 | 런타임 공개, [사용자 지정 변환기 접근](https://ameba-doc-arduino-sdk.readthedocs-hosted.com/en/latest/FAQ/offline_ai_model_conversion_steps.html)은 문의 필요 |
| [Telechips](https://docs.topst.ai/product/p/ai) | 8 TOPS를 광고하는 TCC7500 / TOPST AI 보드 | TC-NN-Toolkit / Enlight SDK; `.enlight` 중간 산출물과 컴파일된 배포 번들 | [공식 TOPST 프로젝트](https://docs.topst.ai/blog/31)에서 YOLOv8s를 배포했습니다. UFLD v1/v2 변환은 미지원 레이어에서 실패했고 분할/후처리 우회 방법만 제안했습니다. [YOLOv4](https://community.topst.ai/t/segmentation-fault/358)와 [다른 YOLOv8 절차](https://community.topst.ai/t/segmentation-fault-error/415)는 커뮤니티 지원 글에 등장합니다. | 보드 문서는 공개, 컴파일러/연산자 다운로드는 권한 제한이 잦음 |
| [T-Head](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | LicheePi 4A의 TH1520, 4-TOPS INT8 광고 | HHB 컴파일러, CSI-NN2/SHL; `hhb.bm`, 생성 코드/매개변수 | 공식 보드 업체 예제에서 NPU로 YOLOv5n/s, MobileNetV2 실행 | 예제 공개, 오래된 툴체인/유지보수 불확실성 |
| [SigmaStar](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | 최신 SSU9383CM 스마트 카메라 SoC, 구형 IPU 계열 | 최신 MI_IPU 런타임; [이전 SGS_IPU 툴체인](https://wx.comake.online/doc/doc/SigmaStarDocs-SSC9381G_9351_Pudding-ULS00V040-20210913/customer/development/dla/tools.html)은 `.sim`에서 `sgsimg.img` 생성 | [구형 공식 SDK 후처리기](https://wx.comake.online/doc/doc/Sigmastar_SDK_v1.2.2/module/User_Guide/Common/SigmaStar_Post_Processing_Module.html)는 SSD와 YOLOv1/2/3를 명시합니다. 해당 모델과 산출물이 SSU9383CM과 호환되는지는 검증되지 않았습니다. | 최신 실리콘이지만 공개 컴파일러/모델 근거는 세대별로 나뉨 |
| [HiSilicon](https://www.hisilicon.com/cn/products/smart-vision/machine-vision/hi3516cv610) | Hi3516CV610/DV500, Hi3519DV500, Hi3403V100 스마트 비전 SoC | NNN/SVP-NNN 보드 런타임을 사용하는 ATC-to-`.om` | 특히 Hi3403 및 Hi3591P 대상 HiSpark 표에 YOLOv3~YOLO11, 자세 추정, 분할, OBB, OCR, 깊이 추정 항목이 있습니다. 표의 모든 SoC에서 검증된 것은 아닙니다. | Ascend와 구별되는 활성 제품군. 저장소 모델은 비상업적 사용 전용으로 표시 |

### 전용 엣지 가속기 및 모듈

| 기업 | 관련 실리콘 | SDK 및 산출물 | 공개 모델 근거 | 접근성 |
|---|---|---|---|---|
| [Hailo](https://github.com/hailo-ai/hailo_model_zoo) | Hailo-8, Hailo-8L, Hailo-10H, Hailo-15 계열 | Dataflow Compiler, HailoRT; `.hef` | YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12, YOLO26, YOLOX, DAMO-YOLO, SSD, EfficientDet, 분할, 자세 추정, OBB와 세대별 모델 표 | 모델 동물원 공개, 컴파일러는 Developer Zone을 통해 제공 |
| [Axelera AI](https://docs.axelera.ai/sdk/reference/models/model-zoo/) | 판매 중인 Metis 제품, 발표된 Europa/Titania 제품군 | 공개 Voyager SDK; 알파 Pipeline Builder `.axm`/`.axe`, 기존 `.axmodel` 번들 | YOLOv3/5/7/8/9/10/11/26, YOLOX, YOLO-NAS, OBB, 자세 추정, 분할, 다수 비 YOLO 모델 검증 | SDK는 GitHub 공개, 고객 지원은 계정 필요 |
| [DEEPX](https://developer.deepx.ai/modelzoo/) | DX-M1, DX-M1M | DXNN SDK, DX-COM, DX-RT; `.dxnn` | 2026년 8월 15일 확인 당시 DX-COM 2.4.0/DX-RT 3.4.0 항목 354개. YOLOv3~YOLO11, YOLO26, YOLOX, SSD, EfficientDet, NanoDet, 분할, 자세 추정, OBB 포함 | 개발자 자료와 제품군 공개 |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | MX3 가속기 및 멀티칩 모듈 | MemryX SDK, Neural Compiler, 런타임; `.dfp` | 공식 예제와 릴리스 노트에서 탐지, 분할, 자세 추정, YOLOv10, YOLO11, YOLO26 지원 | 문서와 SDK 공개 |
| [Kneron](https://doc.kneron.com/docs/) | 최신 컴파일러 문서의 KL520, KL530, KL630, KL720, KL730; 일부 PLUS API에는 있으나 최신 컴파일러 대상 목록에는 없는 KL830 | Kneron PLUS, Model Toolchain; 문서화된 컴파일 대상에서 `.nef` | 공식 [YOLO 절차](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/)는 Tiny-YOLOv3 중심. 다른 자료에 YOLOv5 포함 | 문서 공개, 도구 패키지/다운로드는 칩별로 다름 |
| [SiMa.ai](https://docs.sima.ai/pages/palette/modelsdk.html) | MLSoC 및 양산 Modalix 50-TOPS 플랫폼 | Palette, ModelSDK, MLA Compiler, ModelExecutor | 공개 릴리스에서 YOLOv7/v8, YOLOX, 자세 추정/분할, DETR, Mask R-CNN, EfficientDet 검증 | 문서 공개, 제품 SDK는 상용 |
| [EdgeCortix](https://www.edgecortix.com/en/hardware) | M.2 및 PCIe 제품으로 제공되는 60-TOPS 광고 SAKURA-II | MERA 컴파일러 및 프레임워크 | 벤더는 비전에서 생성형 AI까지 지원한다고 설명하지만, 현재 YOLO 호환성 표는 충분히 구체적으로 공개하지 않음 | 상용 체험/주문 문의, 파트너 검증 중심 |
| [BrainChip](https://brainchip.com/metatf-dev-tools/) | 출하 중 AKD1500 코프로세서/M.2 모듈, 레거시 AKD1000, Akida 2 IP | MetaTF 패키지: `akida-models`, `quantizeml`, `cnn2snn`, `akida` 런타임 | Akida 2 모델 카드에 AkidaNet0.5 YOLOv2 탐지기, CenterNet, AkidaUNet, 얼굴 인식 기재. 해당 결과가 AKD1500 검증을 의미하지는 않음 | 핵심 Python 패키지 공개, 하드웨어 매핑은 대상별 확인 필요 |
| [Blaize](https://www.blaize.com/products/) | P1600, Pathfinder, Xplorer 제품 | Picasso SDK, NetDeploy, AI Studio | 비전이 목표 시장이지만 감사 가능한 공개 모델별 호환성 표를 찾지 못함 | 상용/접근 제한 |
| [Google Coral](https://github.com/google-coral/edgetpu) | 레거시 Edge TPU USB, PCIe, M.2, Dev Board 제품; 별도 활성 오픈 소스 [Coral NPU IP](https://github.com/google-coral/coralnpu) | 레거시 Edge TPU Compiler/`libedgetpu`/PyCoral; 신규 RISC-V Coral NPU는 IP, RTL/시뮬레이션, ELF 예제 제공 | 레거시 예제는 SSD MobileNet, 분류, DeepLab, MoveNet 중심. 신규 IP에는 공개 YOLO 배포 표가 없으며 Edge TPU 제품의 직접 후속품이 아님 | 레거시 핵심 저장소 보관 상태, 새 Coral NPU IP는 활발히 개발 중 |
| [Lattice Semiconductor](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E, Avant-X FPGA | sensAI Studio, Neural Network Compiler, 구성 가능한 가속기 IP | 최신 컴파일러 표에 대상별 모드의 YOLOv1, YOLOv5, YOLOv8, YOLO11, SSD, MobileNetV2-SSD, ResNet, ENet 기재 | 컴파일러/매뉴얼 공개, IP 약관은 다르며 Advanced CNN Accelerator는 상용 |
| [Mobilint](https://www.mobilint.com/sdk-qb) | 10-TOPS REGULUS, 80-TOPS ARIES MLA100 가속기 | qb SDK; INT8 컴파일러, `.mxq` | 공개 [모델 동물원](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md)에 YOLOv3/5/7/8/9/10/11/12/26 탐지 및 분할/자세 추정/OBB 변형 포함 | 문서와 모델 공개, SDK와 하드웨어는 상용 |
| [Rebellions](https://rebellions.ai/developers/) | 활성 ATOM+ CA22, ATOM-Max CA25; 단종 EoL ATOM CA02/ATOM+ CA12; ATOM-Lite CA21 도구 상태 불명확 | RBLN SDK 컴파일러/런타임/프로파일러; `.rbln` | 공식 지원 릴리스에서 YOLOv3, YOLOv5/6/7/8 계열과 최신 YOLOv8 튜토리얼 언급 | 문서 공개, 컴파일러 휠에는 포털 자격 증명 필요 |
| [FuriosaAI](https://furiosa.ai/warboy) | 비전 중심 Warboy Gen1, 별도 RNGD 세대 | Furiosa SDK; Warboy INT8 `.enf`, RNGD 별도 `.fxb` | Warboy 동물원에 SSD, YOLOv5M/L, YOLOv7-w6-pose 기재. RNGD 로드맵은 2024 Q4 YOLOv8m 지원 완료로 표시하지만 현재 상세 비전 표는 없음 | IAM/계정 접근, 두 세대를 구분해야 함 |

### 개발자가 NPU와 함께 비교하는 인접 플랫폼

| 기업 | 하드웨어 | 배포 스택 | 비교에 포함하는 이유 |
|---|---|---|---|
| [NVIDIA](https://github.com/NVIDIA-AI-IOT/deepstream_tools/tree/main/yolo_deepstream) | DLA가 있는 Jetson Orin GPU; DLA 없는 Jetson Thor GPU | JetPack, TensorRT, DeepStream; `.engine` | 성숙한 비전 배포 환경. 공식 DeepStream 도구에서 YOLOv4/v7/v8/v9/11과 YOLO11 OBB 구성을 문서화합니다. 많은 결과는 GPU를 사용하며, 지원되지 않는 Orin DLA 레이어는 GPU로 폴백할 수 있습니다. |
| [AMD](https://vitisai.docs.amd.com/en/6.2/) | Kria/레거시 DPU 대상, Versal AI Edge Gen1 VEK280/VE2802용 Vitis AI 6.2 GA, Gen2 VEK385; 별도 Ryzen AI 클라이언트 NPU | 레거시 `.xmodel`; Gen1 대상 바인딩 NPU 스냅샷; Gen2 컴파일 캐시 디렉터리 또는 프로덕션 `.rai` 패키지 | Vitis AI는 비전 자료를 제공하지만 레거시 DPU, Versal Gen1/Gen2, Ryzen AI는 별개의 컴파일 및 배포 계약입니다. |
| [Microchip](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox) | 구성 가능한 CoreVectorBlox 가속기 IP를 탑재한 PolarFire SoC FPGA | 공개 VectorBlox SDK 3.1, Libero; `.vnnx`, `.hex`, `.ucomp` 배포 파일 | 최신 [튜토리얼](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md)은 YOLOv5n, YOLOv8 탐지/분류/OBB/자세 추정/분할, YOLOv9t 등 지원. SDK 3.1은 현재 PolarFire SoC Video Kit 대상으로 한정됩니다. |
| [Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) | Core Ultra NPU, CPU, 통합/개별 GPU | OpenVINO IR 또는 컴파일 모델 캐시 | 공개 크로스-XPU 경로를 제공합니다. 모든 OpenVINO YOLO 예제가 NPU 검증을 거쳤다고 가정하지 말고 [검증 모델 표](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html)의 NPU 열을 확인하십시오. |
| [Apple](https://developer.apple.com/machine-learning/core-ml/) | A11 이후 A 시리즈 및 M 시리즈 칩의 Apple Neural Engine | Core ML, `coremltools`; `.mlpackage`/`.mlmodelc` | Core ML을 통한 접근성 높은 소비자 NPU입니다. 다만 CPU/GPU/Neural Engine의 실제 분할은 Apple이 추상화하며 YOLO 전용 하드웨어 컴파일러를 공개하지 않습니다. |
| [GreenWaves Technologies](https://github.com/GreenWaves-Technologies/nn_menu_gap9) | NE16 신경망 엔진을 탑재한 GAP9 | GAP SDK, NNTool, AutoTiler 생성 코드 | 공식 저장소에 MobileNet SSD, 얼굴 탐지, 분류, 전용 [YOLOX 사람 탐지기](https://github.com/GreenWaves-Technologies/yolox_people_detection) 포함. 전체 SDK는 자격을 갖춘 고객만 사용할 수 있습니다. |
| [Syntiant](https://www.syntiant.com/hardware) | 양산 중 NDP200, 샘플링 중 NDP250 Neural Decision Processor | Syntiant SDK 및 배포 패키지 | 초저전력 상시 동작 비전/센서 프로세서입니다. NDP200은 1 mW 미만, NDP250 이미지 인식은 30 mW 미만으로 문서화되어 있습니다. 광범위한 공개 YOLO 표는 찾지 못했습니다. |

## Amlogic: 하나가 아닌 두 NPU 세대

웹 자료가 서로 호환되지 않는 제품을 한데 묶는 경우가 많으므로 Amlogic은 자세히 설명할 필요가 있습니다. 이 회사에는 구형 VeriSilicon 경로와 새로운 독자 ADLA 경로가 있습니다. 두 경로는 컴파일러, 산출물, 런타임이 다릅니다.

| 세대 | 대표 칩 | NPU 및 툴체인 | 컴파일 산출물 | 실무 상태 |
|---|---|---|---|---|
| 레거시 | A311D, S905D3, 주로 Khadas VIM3/VIM3L에서 사용 | VeriSilicon Vivante VIPNano-QI, Acuity 툴킷, KSNN, 이전 `aml_npu_sdk` | `nbg_unify` 출력 디렉터리 안의 `.nb` | 기존 보드와 데모, 별도 레거시 통합 |
| 최신 ADLA2 | C308L/C302X, S928X, A311D2, T968D4, S905X5/S905D5, C302X2 | Amlogic ADLA2, AMLNN Toolkit, NNSDK2 | 대상별 `.adla`; W8A8 또는 W8A16 | 정수 중심 최신 스택 |
| 최신 ADLA3 | A311Y3, C305X2, A123X | Amlogic ADLA3, AMLNN Toolkit, NNSDK2 | 대상별 `.adla`; W4A8, W8A8, W4A16, W8A16, W16A16 | 기본 INT4, FP16, BF16 기능 추가 |

[A311D 데이터시트](https://dl.khadas.com/products/vim3/datasheet/a311d-datasheet.pdf)는 기존 5-TOPS INT8 NPU를 명시합니다. 구형 [Khadas VIM3 애플리케이션 페이지](https://docs.khadas.com/products/sbc/vim3/npu/npu-app)와 [NPU 개요](https://docs.khadas.com/products/sbc/vim3/npu/start)는 DenseNet CTC, MTCNN, RetinaFace, YOLOFace, YOLOv2, YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv7-tiny, YOLOv8n을 문서화합니다. 개요에는 YOLOv8n-pose도 있고 FaceNet은 지원 중단으로 표시됩니다. 이들은 실제 예제지만 현재 ADLA 칩이 아닌 이전 Acuity `.nb` 생태계의 근거입니다. A311D와 A311D2는 같지 않으며 C305X와 C305X2도 다릅니다.

두 번째 하드웨어 주의점도 있습니다. [Khadas VIM4 개정판 안내](https://docs.khadas.com/products/sbc/vim4/configurations/identify-version)에 따르면 최초 V12/A311D2 revision-B 보드에는 NPU가 없습니다. 광고된 3.2-TOPS NPU는 A311D2-N0D revision-C 부품을 사용하는 V13A 이후 보드에 포함됩니다. 테스트 장치를 고를 때 제품 이름만으로 판단할 수 없습니다.

### 현재 AMLNN 및 ADLA 스택

최신 [AMLNN Toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/7d3cc9a36179b756ce79c953c14d43acab1f2af5)은 모델 변환, 양자화, 컴파일, 추론, 프로파일링을 지원합니다. 고정된 [2026년 5월 사용자 가이드](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/02_Amlogic_NPU_User_Guide_V0.1.pdf)는 ONNX, 부동 소수점 또는 양자화 TFLite, TorchScript `.pt`, PyTorch 2 ExportedProgram/PT2 가져오기를 문서화합니다. 저장소 개요는 더 포괄적으로 표현하지만 상세 가이드에서 TensorFlow, Paddle, Keras 경로는 계획 중으로 표시됩니다. 컴파일러는 `.adla`를 출력합니다. Python 배포에는 AMLNN 또는 `amlnn_edge_toolkit_lite`를 사용하고, 네이티브 C/C++는 `nnsdk2.h`를 통해 `libnnsdk.so`를 링크합니다. 호스트 개발에서는 ADB와 `nnserver`를 사용할 수 있습니다. 런타임 대상은 Android, Buildroot, Yocto, 일부 Debian/Armbian 환경입니다.

현재 툴킷에 게시된 플랫폼 식별자는 다음과 같습니다.

| 대상 ID | 플랫폼 |
|---:|---|
| `001` | C308L / C302X |
| `002` | S928X |
| `003` | A311D2 |
| `004` | T968D4 |
| `005` | S905X5 / S905D5 |
| `006` | C302X2 |
| `007` | A311Y3 |
| `008` | 상세 문서와 모델 표의 C305X2 / A123X |

대상 필드는 장식이 아닙니다. Amlogic에서는 `.adla`, `nnsdk2.h`, `libnnsdk.so`, BSP, 컴파일러/런타임 세대가 서로 맞아야 합니다. 고정 [빠른 시작](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/01_Amlogic_NPU_Quick_Start_guide_V0.1.pdf)은 ADLA 드라이버 2.0.2 이상, NNSDK 3.0.0 이상, NNSDK2 1.0.0 이상을 요구합니다. [Android 배포 가이드](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/06_Amlogic_NPU_Android_Deployment_Guide_V0.1.pdf)는 라이브러리, 헤더, 드라이버, BSP의 관계를 구체적으로 설명합니다. `.adla`는 이식 가능한 모델이 아니라 ABI에 종속된 빌드 산출물로 취급해야 합니다.

양자화도 NPU 세대에 따라 달라집니다. ADLA2 대상 001~006은 W8A8, W8A16 컴파일을 문서화합니다. ADLA3 대상 007, 008은 [연산자 가이드](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/05_Amlogic_NPU_Support_Operator_List_V0.1.pdf)에 기본 INT4, FP16, BF16과 함께 W4A8, W4A16, W8A8, W8A16, W16A16 조합을 추가합니다. Amlogic은 대표 보정 샘플 200~500개를 권장합니다. 무작위 데이터 옵션은 정확도를 보존하는 양자화가 아니라 성능 테스트용이라고 명시합니다.

### Amlogic의 문서화된 모델 범위

[고정 Amlogic 모델 플레이그라운드](https://github.com/Amlogic-NN/amlnn-model-playground/blob/03ca63ef20c4f1560722de26ab910826884f334a/README.md)는 일반적인 ONNX 지원 주장보다 강한 근거입니다. 공개 지원/예제 표에는 A311D2, S905X5, A311Y3, C305X2, A123X가 명시되어 있습니다. 다른 대상 ID의 컴파일 허용 여부만으로 아래 전체 모델 목록이 그 칩에서 검증되었다고 볼 수는 없습니다.

| 작업 | 문서화된 모델 |
|---|---|
| 분류 | MobileNetV2, ResNet50-v2; ADLA3의 DINO |
| 객체 탐지 | PP-YOLOE, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv10, YOLO11, YOLOE, YOLO-World, YOLOX, QR 탐지 |
| 얼굴 및 제스처 | RetinaFace, Gesture Recognition |
| 분할 | DeepLabV3, PP-LiteSeg, YOLOv5-seg, YOLOv8-seg |
| 방향 탐지 | YOLOv8 OBB |
| 자세 추정 | BlazePose detector/landmark, YOLOv8 pose |
| OCR 및 음성 | LPRNet, PaddleOCR 변형, Whisper Tiny, 일부 최신 칩의 SenseVoice |
| 최신 트랜스포머 및 멀티모달 | DETR, MobileSAM, CLIP/MobileCLIP, 최신 플랫폼의 일부 저비트 LLM/VLM 예제 |

같은 저장소는 아래 W8A8 모델 런타임 수치를 공개합니다. 이는 NPU에서 신경망만 실행한 벤더 측정값이지 전체 카메라 파이프라인 결과는 아닙니다.

| 모델 및 입력 | S905X5 | A311D2 | A311Y3 |
|---|---:|---:|---:|
| YOLOv8n, 640 x 640 | 101.72 FPS | 95.14 FPS | 191.06 FPS |
| YOLOv8s, 640 x 640 | 42.33 FPS | 42.77 FPS | 83.08 FPS |
| YOLOv8m, 640 x 640 | 19.67 FPS | 19.82 FPS | 35.30 FPS |
| YOLOv8l, 640 x 640 | 10.53 FPS | 10.12 FPS | 18.37 FPS |
| YOLO11n, 640 x 640 | 41.14 FPS | 41.48 FPS | 62.24 FPS |

여기에는 특히 중요한 주의점이 있습니다. Amlogic 연산자 가이드에서 NMS는 ADLA2와 ADLA3 모두 소프트웨어에서 처리합니다. README는 전처리와 후처리를 제외한 NPU 모델 런타임 결과로 정의하지만 모든 메모리 전송이 포함되는지는 밝히지 않습니다. YOLO 정확도 행은 전체 검증 세트가 아니라 COCO 이미지 300장 부분집합을 사용하고, 분류 행은 ImageNet `val1000`을 사용합니다. 표에는 조건을 저장소가 설명하지 않은 A311Y3 YOLOv8n 지연 시간과 FPS 수치도 함께 제시됩니다. 클록, 전력 모드, 냉각, 정확한 SDK/BSP 버전, 테스트 시간은 공개되지 않았습니다. 이 수치는 한 벤더 스택을 이해하는 데 활용하고 다른 벤더와 순위를 매기는 데 사용하지 마십시오.

### Amlogic이 전략적으로 흥미로운 이유

Amlogic은 네 가지 이례적인 조건을 갖춥니다. 대규모 임베디드 SoC 보급 기반, 새롭게 공개된 컴파일러 스택, LibreYOLO 작업 영역과 상당 부분 겹치는 최신 모델 표, 주요 학습 라이브러리의 제한적인 일급 통합입니다. 최신 저장소도 2025년 말과 2026년 초에 공개되어 역사가 짧고, 실질적인 매뉴얼은 2026년 5~7월에 작성되었습니다. 따라서 선점 기회와 API 안정성 위험이 모두 실제로 존재합니다.

깔끔한 통합은 LibreYOLO의 결정적 ONNX 내보내기를 컴파일러 입력으로 사용하고, 저정밀도 빌드에 대표 보정 이미지를 요구하며, 메타데이터 매니페스트와 `.adla`를 생성한 뒤 NNSDK2 어댑터로 불러와야 합니다. 레거시 A311D에는 별도의 `.nb` 대상을 명확히 표시해야 합니다. `.nb`와 `.adla`를 하나의 백엔드로 제시하면 호환성 문제가 조용히 발생할 수 있습니다. Amlogic 저장소의 최상위 라이선스는 Apache-2.0이지만, 번들 휠, 공유 라이브러리, `nnserver`, Android AAR 바이너리의 재배포 권한은 LibreYOLO에서 미러링하기 전에 직접 확인해야 합니다.

## 공개 배포 생태계가 가장 강한 업체

다음 기업은 현재 구할 수 있는 하드웨어, 공개 기술 자료, 모델 이름이 명시된 근거를 가장 명확하게 조합해 제공합니다. 그렇다고 모두에게 더 빠른 것은 아닙니다. 하드웨어 구매 전에 호환성 주장을 평가하기 쉽다는 뜻입니다.

### Hailo

Hailo는 전용 비전 가속기 생태계의 대표 사례입니다. 모델은 Hailo Archive로 분석하고 대표 이미지를 사용해 최적화 및 양자화한 뒤 Hailo Executable Format(`.hef`) 파일로 컴파일합니다. 애플리케이션은 HailoRT로 HEF를 불러옵니다.

[Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)에는 탐지, 분할, 분류, 자세 추정 등의 레시피, 사전 학습 모델, 후처리 구성, 성능 자료가 있습니다. YOLO 범위는 YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 및 YOLO26과 YOLOX, DAMO-YOLO, SSD, EfficientDet을 포함합니다. 버전이 고정된 [Hailo-8 객체 탐지 표](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.19.0/docs/public_models/HAILO8/HAILO8_object_detection.rst)는 일반 제품 페이지보다 좋은 호환성 자료입니다. 독립 [애플리케이션](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/README.md)에서 배포 가능한 YOLO 파이프라인도 설명합니다.

중요한 버전 경계가 있습니다. Hailo-8 및 Hailo-8L은 구형 Model Zoo 2.x, Dataflow Compiler 3.x, HailoRT 4.x 계열을 사용하고 Hailo-10/Hailo-15는 최신 5.x 세대입니다. 모두 Hailo 이름을 사용한다고 레시피, 파서 동작, HEF를 서로 바꿔 쓸 수 있는 것은 아닙니다.

Hailo-15에는 별도의 애플리케이션 계층도 있습니다. Hailo-8/10H TAPPAS 스타일 호스트 경로가 아니라 Vision Processor Software Package와 Hailo Media Library를 사용합니다. 공개 참조 저장소 [`hailo-camera-apps`](https://github.com/hailo-ai/hailo-camera-apps)는 2026년 5월 3일 보관 처리되었고, [`hailo-apps-core`](https://github.com/hailo-ai/hailo-apps-core)는 별도의 공개 애플리케이션 프레임워크로 유지됩니다. 저장소 보관이 제품 단종의 증거는 아니지만, Hailo-15 프로젝트는 Developer Zone에서 현재 애플리케이션 패키지를 확인해야 합니다.

LibreYOLO의 [Hailo 배포 가이드](/docs/export/hailo)는 독점 컴파일러를 Python 의존성으로 묶을 수 없으므로 정적 ONNX 전달 지점에서 멈춥니다. 이는 적합한 그래프를 만들 수 있는 것과 테스트를 마친 HEF라고 주장할 수 있는 것 사이의 정확한 경계입니다.

### Rockchip

Rockchip은 특히 RK3588, RK3576, RK356x 보드를 통해 취미 및 상용 임베디드 Linux 시장에서 큰 비중을 차지합니다. 공개 [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)는 x86 Linux 호스트에서 모델을 변환하고 양자화하며, RKNN Runtime 또는 Toolkit-Lite가 보드에서 결과 `.rknn` 파일을 실행합니다.

[RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)는 칩, 모델 계열, 정밀도를 상세히 표시합니다. 현재 표에는 YOLOv5/6/7/8/10/11, YOLOX, PP-YOLOE, YOLO-World, YOLOv5/v8 분할 변형의 FP16 및 INT8 경로가 있습니다. YOLOv8 OBB와 YOLOv8 pose는 INT8 전용으로 표시됩니다. OCR, 얼굴, 그 밖의 분할 네트워크도 포함합니다.

LibreYOLO는 이미 보수적인 직접 [RKNN 내보내기 도구](/docs/export/rknn)를 제공합니다. RK3588에서 정확한 탐지 변형 4개를 검증하고, 컴파일러 시뮬레이터와 ONNX Runtime을 비교할 수 있으며, 성공적인 빌드를 올바른 예측과 동일시하지 않고 검증되지 않은 계열은 거부합니다. 이처럼 제한된 지원 주장은 향후 모든 NPU 백엔드의 좋은 본보기가 됩니다.

### Axelera AI

Axelera AI는 종종 "Accelera"로 잘못 표기되며 Metis AIPU를 개발하고 M.2, PCIe, 멀티칩 제품으로 판매합니다. 구매 가능한 제품군은 Metis이며 Europa와 Titania는 발표된 제품이므로 제품 상태를 한 범주로 합치면 안 됩니다. 공개 [Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk)는 GitHub에서 사용할 수 있고 모델 선택, 컴파일, 양자화, 실행, 애플리케이션 파이프라인을 처리합니다. 고객 지원에는 계정이 필요합니다.

공개 [Voyager Model Zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo/)는 시장에서 정보가 가장 많은 자료 중 하나입니다. YOLOv3, YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 n/s/m/l/x, YOLOX, YOLO-NAS, OBB, 자세 추정, 분할과 다양한 분류/밀집 예측 모델의 변형, 입력 크기, 정밀도, 정확도, 측정 성능을 나열합니다. 속도만이 아니라 양자화 손실과 정확도도 함께 공개하는 편이 최대 TOPS 수치만 공개하는 것보다 유용합니다.

API 세대가 바뀌면서 산출물 이름도 변했습니다. 알파 [Pipeline Builder 컴파일 절차](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/)는 `.axm` 모델을 만들고 이식 가능한 파이프라인을 `.axe`로 패키징할 수 있습니다. [기존 파이프라인 문서](https://docs.axelera.ai/sdk/reference/pipeline/model-formats/)는 모델 메타데이터와 매니페스트가 포함된 `.axmodel`을 설명합니다. 통합 도구는 확장자를 하나로 고정하지 말고 설치된 Voyager 세대를 확인해야 합니다.

### Qualcomm

Qualcomm은 배포 범위가 매우 넓지만 "Qualcomm NPU"에는 여러 인터페이스가 숨어 있습니다. 기기와 제품군에 따라 QAIRT/QNN을 통한 Hexagon HTP, 구형 SNPE 절차, Qualcomm AI Hub 컴파일 서비스, LiteRT, ONNX Runtime QNN 실행 제공자를 사용하게 됩니다.

공식 [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) 저장소는 모델 수준의 강한 근거를 제공합니다. YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 탐지/분할/자세 추정, YOLOX, YOLO-World, YOLOR, RF-DETR 및 많은 분류, 깊이 추정, 자세 모델의 최적화 패키지를 공개합니다. AI Hub는 기기별 프로파일링도 제공하는데, 한 HTP 세대에 컴파일된 모델이 다른 세대로 자동 이식되지 않기 때문입니다.

주요 통합 비용은 조합의 크기입니다. Android와 임베디드 Linux, QCS 주문 부품과 소비자 Snapdragon, HTP 아키텍처, QNN/QAIRT 버전, 양자화 방식 모두 중요합니다. Qualcomm의 현재 자료도 [Dragonwing IQ-9075](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075) 상태를 서로 다르게 설명합니다. 제품 페이지는 Active라고 표시하고 [EVK](https://www.qualcomm.com/developer/hardware/qualcomm-iq-9075-evaluation-kit-evk)로 평가할 수 있다고 하지만 [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program)은 여전히 IQ-9075를 Sampling으로 표시하며 2038년까지의 수명을 나열합니다. 주문/SKU 접두어는 QCS9075이고 Qualcomm은 50 및 100 dense INT8 TOPS 구성과 Ubuntu/Yocto 지원을 광고합니다. 정확한 주문 SKU의 양산 상태를 확인하고 LibreYOLO 통합에서도 "Qualcomm"이라는 일반 폴더를 만드는 대신 해당 SKU를 기록해야 합니다.

### DEEPX

DEEPX DX-M1/DX-M1M 가속기는 DXNN SDK를 사용합니다. DX-COM으로 모델을 컴파일하고 양자화하며 DX-RT로 실행합니다. 배포 산출물은 `.dxnn` 형식입니다. 회사는 [DX-AllSuite](https://github.com/DEEPX-AI/dx-all-suite), [DX-APP](https://github.com/DEEPX-AI/dx_app)의 애플리케이션 예제, 대형 [온라인 Model Zoo](https://developer.deepx.ai/modelzoo/)를 공개합니다.

2026년 8월 15일 확인 당시 공개 카탈로그는 DX-COM 2.4.0, DX-RT 3.4.0 기준으로 항목 354개를 표시했습니다. 문서화 범위에는 YOLOv3~YOLO11과 YOLO26, YOLOX, SSD, EfficientDet, NanoDet, DAMO-YOLO, YOLO 분할, 자세 추정, 방향 바운딩 박스가 포함됩니다. 컴파일러/런타임 경계와 배포 산출물 이름이 명확하고 공개 비전 지원 범위가 넓어 DEEPX는 특히 통합 후보로 적합합니다.

## 산업용 SoC는 하나가 아니라 여러 생태계

산업용 업체는 메이커 보드 SoC보다 제품 수명이 길고 카메라, 안전, 실시간 통합이 우수한 경우가 많습니다. 한 업체가 서로 관련 없는 NPU 아키텍처를 여럿 제공할 수 있어 AI 스택은 더 복잡해질 수 있습니다.

### Texas Instruments

TI의 최신 엣지 AI 프로세서는 AM62A, AM67A, AM68A, AM69A와 관련 TDA4 장치입니다. 소프트웨어 경로는 Processor SDK Linux, TIDL 컴파일러/런타임, [Edge AI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools), [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo)를 조합합니다. TIDL은 ONNX Runtime, TensorFlow Lite 등 애플리케이션 런타임과 통합하면서 지원하는 하위 그래프를 오프로딩할 수 있습니다.

TI는 프레임워크 가져오기 주장 이상의 자료를 공개합니다. 모델 동물원에는 객체 탐지, 분할, 자세 추정, 분류, 깊이 추정 등의 변환 모델과 성능 메타데이터가 있습니다. TI는 모델 동물원 산출물이 개발 시작점이지 자동으로 프로덕션에 바로 쓸 수 있는 자산은 아니라고 경고합니다. 이는 벤더 비교에서 자주 빠지는 중요한 주의사항입니다.

TI는 [TDA54-Q1](https://www.ti.com/product/TDA54-Q1)도 Preview로 표시하며, 해당 부품에 C7 NPU 최대 4개와 최대 400 TOPS를 명시합니다. 더 넓은 TDA5 제품군은 최대 1,200 TOPS로 광고합니다. 이는 신세대 제품에 대한 벤더 주장이지 TI가 대상별 표를 공개하기 전에 AM6xA/TDA4 TIDL 모델 검증을 TDA54-Q1에 그대로 적용할 근거는 아닙니다.

### NXP

NXP에서는 적어도 네 가지 백엔드 설명이 필요합니다.

| NXP 대상 | 가속기 | 컴파일러/런타임 경로 |
|---|---|---|
| i.MX 8M Plus | 2.3-TOPS VeriSilicon Vivante NPU | TIM-VX/VX delegate 기반 eIQ |
| i.MX 93 | Arm Ethos-U65 | 양자화 TFLite와 Arm Vela |
| i.MX 95 | NXP eIQ Neutron NPU | Neutron Converter와 eIQ 런타임 |
| Ara-1 / [Ara240](https://www.nxp.com/products/ARA240) | Kinara 기반 개별 NPU; Ara240은 최대 40 eTOPS 광고 | 광범위한 eIQ 생태계와 통합되는 Ara SDK |

[eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo)에는 YOLOv4-tiny, YOLOv8, NanoDet, CenterNet, FastestDet, SSD Lite, YOLACT 등의 산출물 또는 레시피가 있습니다. YOLOv5 항목은 문서만 제공합니다. 한 백엔드에서 검증된 항목은 네 가지 모두에 대한 근거가 아닙니다. NXP의 [i.MX 플랫폼 YOLO 내보내기 안내](https://community.nxp.com/t5/i-MX-Processors-Knowledge-Base/Exporting-YOLO-Models-for-NXP-i-MX-Platforms/ta-p/2381361)도 대상별 변환 문제를 보여 줍니다. NXP는 일체형 [eIQ Toolkit이 2025년 3분기 버전 1.17 이후 업데이트를 중단했다](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/eIQ-FAQ/ta-p/1099741)고 밝힙니다. 최신 절차는 eIQ Neutron SDK와 같은 독립 패키지를 사용합니다.

NXP는 [2025년 10월 Kinara 인수를 완료](https://media.nxp.com/news-releases/news-release-details/nxp-completes-acquisitions-aviva-links-and-kinara-advance/)했습니다. [Ara240 데이터시트](https://www.nxp.com/docs/en/fact-sheet/ARA240DNPUFS.pdf)는 벤더 광고 기준 최대 40 eTOPS와 YOLOv8n 초당 이미지 313장을 보고합니다. NXP 제품 자료에는 Ara SDK와 eIQ Toolkit이 모두 있지만 이것만으로 별도의 [i.MX 95 Neutron 경로](https://eiq.nxp.com/learning-hub/convQuant/neutron.html)와 산출물을 서로 바꿔 쓸 수 있다는 뜻은 아닙니다. [16 GB M.2 모듈](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT)은 판매 중이고 USB 옵션은 양산 전입니다. NXP는 eTOPS의 "e"가 "equivalent"를 뜻한다고 설명합니다. 이는 "effective"가 아니며 다른 업체의 dense INT8 TOPS 지표와 직접 비교할 수 없습니다.

### STMicroelectronics

[STM32N6x7 장치](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html)는 STM32N657/STM32N647을 포함해 MCU급 제품에 600-GOPS Neural-ART 가속기를 탑재합니다. N6x5 범용 제품군에는 이 가속기가 없습니다. [STM32Cube AI Studio](https://www.st.com/en/development-tools/stedgeai-cubeai.html)와 ST Edge AI Core가 가져온 모델을 분석하고 최적화하며 [STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services)는 배포, 최적화, 벤치마킹, 예제 절차를 제공합니다.

현재 [객체 탐지 개요](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/docs/README_OVERVIEW.md)는 분류, 자세 추정, 분할 서비스와 함께 Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26, ST-YOLOX를 문서화합니다. 이는 200-TOPS PCIe 카드와 같은 성능 등급이 아닙니다. 카메라 입력, 추론, 제어를 저전력과 메모리가 제한된 임베디드 환경에 함께 둘 수 있다는 점에서 의미가 있습니다.

### Renesas

Renesas RZ/V 프로세서에는 DRP-AI 가속기가 통합되어 있습니다. 소프트웨어는 DRP-AI Translator와 DRP-AI TVM에서 [RUHMI](https://www.renesas.com/en/software-tool/ruhmi-framework)로 발전해 왔으며, RUHMI는 EdgeCortix MERA 기술을 기반으로 합니다. 오픈 [RZ/V DRP-AI TVM 저장소](https://github.com/renesas-rz/rzv_drp-ai_tvm)와 [RZ/V2H 검증 목록](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md)은 분류 모델과 함께 YOLOv5, YOLOv8, YOLO11, YOLO26 n/s/m 변형, YOLOX, 자세 추정, 분할 네트워크를 명시합니다.

Renesas는 로보틱스 및 산업용 대상에 적합하지만 재현성을 위해 정확한 보드, DRP-AI 세대, Translator 버전, CPU 측 후처리를 기록해야 합니다.

## 스마트 센서 및 카메라 중심 프로세서

### Sony IMX500

Sony IMX500은 일반 애플리케이션 프로세서가 아닙니다. 이미지 센싱과 AI 처리를 쌓아 추론을 센서에서 수행하므로 카메라 밖으로 전송해야 하는 이미지 데이터가 줄어듭니다. Raspberry Pi AI Camera는 이 아키텍처를 특히 쉽게 사용할 수 있게 합니다.

공식 [Raspberry Pi IMX500 모델 저장소](https://github.com/raspberrypi/imx500-models)는 패키징된 YOLOv8n, YOLO11n, EfficientDet Lite0, NanoDet+, SSD MobileNetV2 FPN Lite 모델을 공개합니다. [AI Camera 문서](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)는 변환, 패키징, 센서 후처리를 설명합니다. 배포 단위는 일반 ONNX 파일이 아닌 RPK 패키지이며 센서 메모리와 연산자 제약이 주요 설계 한계입니다. Raspberry Pi [제품 페이지](https://www.raspberrypi.com/products/ai-camera/)에 따르면 이 카메라는 적어도 2028년 1월까지 생산됩니다.

### Ambarella

Ambarella CVflow 프로세서는 카메라, 드론, 자동차 비전 분야에 깊이 자리 잡았습니다. 최신 제품군은 카메라용 CV72/CV75, 고성능 비전 시스템용 CV5/CV52, 자동차용 CV3-AD를 포함합니다. Cooper 개발 플랫폼과 CVflow 컴파일/런타임 절차가 공개적으로 이름이 확인되는 소프트웨어 스택입니다.

[공개 개발자 모델 정원](https://www.ambarella.com/developer/model-garden/)에는 YOLOX, RTMDet, DeepLabV3+, TopFormer, OWL-ViT, 멀티모달 모델이 나옵니다. 그러나 상세 컴파일러 문서와 다운로드는 파트너 중심으로 제공됩니다. 따라서 Ambarella 통합은 공개 pip 패키지를 가정하지 말고 벤더 또는 OEM 관계에서 시작해야 합니다.

### Synaptics Astra

Synaptics에는 관련 대상이 세 가지 있습니다. SL1600/SL1680 Astra 시스템은 [SyNAP 툴킷](https://developer.synaptics.com/docs/synap/introduction)을 사용해 원본 모델과 YAML 설명에서 `model.synap` 패키지를 만듭니다. 최신 SL2611/13/15/17/19 장치는 `.vmfb`를 생성하는 IREE/MLIR 기반 흐름인 [Torq 플랫폼](https://developer.synaptics.com/docs/torq/introduction)을 사용합니다. [SR100 계열](https://developer.synaptics.com/docs/sr/introduction-sr)은 Cortex-M55와 Ethos-U55를 중심으로 한 별도 MCU 제품군이며 전용 SDK가 있습니다. 산출물과 API는 서로 바꿔 쓸 수 없습니다.

공식 [YOLO 벤치마크 튜토리얼](https://developer.synaptics.com/docs/sl/tutorials/vision/benchmark-yolo)은 매우 구체적입니다. YOLOv8을 TFLite로 내보내고 SyNAP에서 비대칭 UINT8 보정을 수행한 뒤 SL1680 대상으로 컴파일하고 `synap_cli`와 객체 탐지 애플리케이션을 실행합니다. 이는 SL1680의 YOLOv8n/v8s를 입증하고 SyNAP이 YOLO11까지 지원한다고 설명합니다. 별도의 [SL261x 객체 탐지 가이드](https://developer.synaptics.com/docs/sl/sl2600/getting-started/object-detection)는 Torq `.vmfb`에서 YOLOv8을 실행합니다. 이들은 대상별 공식 예제이지 세 계열 모두에서 모든 YOLO 그래프를 지원한다는 주장이 아닙니다.

### MediaTek Genio

최신 [IoT AI Hub](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html)가 이제 하드웨어/소프트웨어 표, 모델 패키지, 벤치마크 표를 공개하므로 MediaTek도 별도 항목으로 다룹니다. 과거 시장 조사는 이 자료를 감사할 수 없었습니다. Genio 360/360P/420/520/720은 MDLA 5.3 기반 NeuroPilot 8을 사용하고, Genio 510/700은 MDLA 3.0 기반 NP6, Genio 1200은 MDLA 2.0 기반 NP6를 사용합니다. MediaTek은 NeuroPilot 및 MDLA 세대, 연산자 집합, 컴파일러, 런타임이 각 SoC의 수명 동안 버전에 종속된다고 밝힙니다.

분석 AI 경로는 TFLite 중심입니다.

```text
PyTorch model
    -> NeuroPilot Converter and representative calibration
    -> quantized .tflite
    -> version-matched ncc-tflite compiler
    -> chip-generation-specific .dla
    -> Neuron Runtime on MDLA
```

최신 Genio 제품은 온라인 LiteRT 위임과 지원 운영체제의 ONNX Runtime CPU/NPU 경로도 제공합니다. 이는 오프라인 `.dla`와 다른 실행 모드입니다. [소프트웨어 아키텍처 가이드](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/software_architecture.html)와 [리소스 표](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html)가 차이를 명확히 보여 줍니다.

공식 [분석 모델 표](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical.html)는 여러 MDLA 세대의 YOLOv5s 및 YOLOv8s 벤치마크 항목과 변환 가이드를 공개합니다. AGPL-3.0 제한 때문에 사전 변환한 YOLO 산출물은 배포하지 않는다고 명시합니다. Quant8, 640 x 640 오프라인 측정은 Genio 720에서 각각 5.35 ms 및 8.04 ms입니다. [YOLOv8s 모델 페이지](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical/YOLOv8s.html)는 입력/출력 텐서, 백엔드별 시간, MediaTek 사용자 지정 연산자에 대한 경고를 게시합니다. 이는 벤더 벤치마크이지 종단 간 카메라 결과가 아닙니다.

접근성이 제약입니다. 공개 Yocto 문서는 상세하지만 현재 NP8 올인원 변환기/컴파일러 번들과 다수 Android 자료는 NDA 또는 직접 고객 전용으로 표시됩니다. 일반 개발자 계정은 MediaTek Online 고객 계정과 같은 권한을 부여하지 않습니다. 따라서 LibreYOLO는 Genio를 기술적으로 입증됐지만 재현 가능한 사용자 모델 컴파일 통합에는 파트너십이 필요한 대상으로 취급해야 합니다.

## 중국 중심 엣지 AI 생태계

가장 역량이 높고 접근하기 쉬운 저가형 비전 플랫폼 일부는 영어권 시장 목록에서 빠져 있습니다.

### D-Robotics 및 Horizon 기반 BPU 플랫폼

D-Robotics는 RDK X3, X5, Ultra, 최신 S100 계열 제품에 사용되는 BPU 가속기용 RDK 개발 보드 생태계를 운영합니다. [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo)의 최신 `rdk_x5` 브랜치는 RDK X5용 YOLOv5/v5u, YOLOv8/9/10/11/12/13/26, YOLOE, YOLO-World, 탐지, 분할, 자세 추정, 분류, OCR, CLIP 경로를 문서화합니다. X3는 별도 브랜치를 사용하고 최신 S 시리즈 배포는 기본 동물원의 [`rdk_s` 브랜치](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s)에 있습니다. 이전 [`rdk_model_zoo_s` 저장소](https://github.com/D-Robotics/rdk_model_zoo_s)는 과거 데모를 담고 있습니다. 따라서 최신 X5 목록은 모든 모델이 X3, Ultra, S100에서 검증되었다는 근거가 아닙니다.

OpenExplorer/Algorithm Toolchain은 ONNX 또는 Caffe를 가져와 PTQ를 수행하고 PyTorch 중심 QAT 절차를 지원합니다. 배포 방식은 플랫폼마다 다릅니다. 최신 RDK X5는 `libdnn`을 통한 `hbm_runtime`에서 `.bin`을 사용하고, 현재 주요 `rdk_s` 절차는 `libhbucp`를 통한 동명의 `hbm_runtime` API에서 `.hbm`을 사용합니다. X3에는 이전 브랜치와 추론 인터페이스가 유지됩니다. 이전 `rdk_model_zoo_s` 자료도 과거 `.bin`, `.hbm` 경로를 설명하므로 산출물은 대응 브랜치와 툴체인에 연결해야 합니다. 미지원 연산자는 CPU로 폴백할 수 있습니다. 공개 예제는 유용하지만 RDK OS, 보드 펌웨어, 툴체인, 모델 바이너리 간 버전 조합도 호환성 계약의 일부입니다.

### AXERA

AXERA AX650/AX630/AX620 계열은 Sipeed MaixCAM 제품군 같은 소형 AI 카메라와 보드에 사용됩니다. Pulsar2는 ONNX를 가져와 보정과 정밀도 분석을 수행하고 `.axmodel`을 출력하며 AXEngine이 산출물을 실행합니다.

[AXERA 공식 샘플](https://github.com/AXERA-TECH/ax-samples)은 YOLOv5/6/7/8/9/10/11/13/26, YOLOX, YOLO-World를 다루며 작업 및 칩 지원은 플랫폼마다 다릅니다. 지원 대상의 최신 YOLO26 예제에는 탐지, 자세 추정, 분할, OBB가 있습니다. [Pulsar2 변환 예제](https://pulsar2-docs.readthedocs.io/en/latest/appendix/model_convert_examples.html)는 정확한 텐서 이름, 출력 분할, 레이아웃 변환, 보정 아카이브, 대상 하드웨어 설정 등 실제 작업을 보여 줍니다.

### SOPHGO

컴파일러 자체가 오픈 소스라는 점에서 SOPHGO의 TPU-MLIR은 독립적인 통합에 매력적인 스택입니다. ONNX, PyTorch, TFLite, Caffe를 가져와 그래프를 변환 및 양자화하고 BM 대상의 `.bmodel` 또는 CV18xx 하드웨어의 `.cvimodel`을 생성합니다.

[TPU-MLIR 프로젝트](https://github.com/sophgo/tpu-mlir)는 대상에 따라 FP32, BF16, FP16, INT8 흐름을 지원합니다. [SOPHON 데모 모음](https://github.com/sophgo/sophon-demo/blob/release/README_EN.md)은 YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, OBB/분할 변형, SSD, CenterNet, RetinaFace, SAM/SAM2, OCR를 명시합니다. BM1684X의 데모가 BM1688 또는 CV18xx에서도 검증된 것은 아닙니다.

### Huawei Ascend

Huawei 엣지 추론 제품군에는 Atlas 200I 및 300I 제품의 Ascend 310/310P/310B 실리콘이 포함됩니다. CANN은 ATC 컴파일러, AscendCL 런타임, 칩별 연산자 커널을 제공합니다. ATC는 ONNX 등 원본 표현을 오프라인 `.om` 모델로 변환합니다.

현재 [Atlas 200I DK A2 문서](https://www.hiascend.com/en/hardware/developer-kit-a2/specification)와 CANN 샘플은 Ascend 310B용 YOLOv5 `.om` 경로를 제공합니다. 더 넓고 오래된 [Ascend ModelZoo](https://github.com/Ascend/modelzoo)에는 YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN, 자세 추정, 분할 모델이 있지만 모든 항목이 310B에서 재검증된 것처럼 제시하면 안 됩니다. CANN, 커널, 펌웨어, SoC가 서로 맞아야 합니다. Ascend310P용 `.om`은 범용 Ascend 산출물이 아닙니다.

### Cambricon

Cambricon MLU 가속기는 Neuware와 MagicMind 추론 엔진을 사용합니다. 공개 [MagicMind Cloud 저장소](https://github.com/Cambricon/magicmind_cloud)는 MagicMind 1.7, MLU370-X4/S4 표에 고정되어 있습니다. 이는 과거 호환성 근거이지 현재 SDK 또는 MLU270 검증 표는 아닙니다. 저장소는 PyTorch, ONNX, Caffe, Paddle 경로를 문서화하며, 과거 TensorFlow 예제가 남아 있더라도 1.7에서 TensorFlow 프레임워크 지원은 제거되었습니다. Cambricon의 최신 [3-series 개발자 포털](https://developer.cambricon.com/index/document/index/classid/3.html)에는 더 최신 SDK 릴리스가 있으므로 공개 예제 저장소와 현재 상용 스택을 같은 버전으로 설명하면 안 됩니다.

공식 [MagicMind 클라우드 모델 저장소](https://github.com/Cambricon/magicmind_cloud)는 매우 직접적인 MLU370-X4/S4 표를 제공합니다. YOLOv3, Tiny-YOLOv3, YOLOv4, YOLOv5, YOLOv7, YOLOv8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN, DeepLab, UNet 예제와 C++/Python 예제 여부를 표시합니다. 근거는 강하지만 Cambricon 채널을 통한 SDK 및 컨테이너 접근이 주된 장벽입니다.

### Canaan/Kendryte

Kendryte의 최신 K230/K230D와 레거시 K210/K510 KPU 칩은 저가형 보드 및 스마트 카메라 시장에서 중요합니다. [nncase 컴파일러](https://github.com/kendryte/nncase)는 ONNX 또는 TFLite를 가져오고 고정 소수점 변환에 보정 데이터를 사용하며 호스트에서 결과를 시뮬레이션한 뒤 `.kmodel`을 출력합니다.

공식 [K230 nncase 개발 가이드](https://github.com/kendryte/k230_docs/blob/main/en/01_software/board/ai/K230_nncase_Development_Guide.md)는 YOLOv5s의 컴파일, 시뮬레이션, 실행을 안내합니다. 별도 [AI 데모 카탈로그](https://github.com/kendryte/k230_docs/blob/main/en/02_applications/ai_demos/K230_AI_Demo_Introduction.md)에는 YOLOv8n 탐지, 분할, 자세 추정 산출물이 있고 최신 [CanMV 변경 기록](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md)에서는 최적화된 YOLOv8/11/26 분류, 탐지, 분할, OBB, 자세 추정 예제를 추가합니다. 실제 공개 벤치마크는 데이터시트의 [YOLOv5s 결과](https://github.com/kendryte/k230_docs/blob/main/en/00_hardware/K230_datasheet.md)입니다. 컴파일러와 바이너리 KPU 플러그인 버전은 보드 SDK에 맞아야 합니다.

### Allwinner

Allwinner V853은 카메라 중심 미디어 하드웨어와 1-TOPS Vivante NPU를 결합합니다. [공식 영어 NPU 가이드](https://docs.aw-ol.com/v853/en/npu/dev_npu/)는 Acuity/Pegasus 가져오기, 양자화, 검증, `viplite` 배포를 설명합니다. Allwinner의 모델 페이지와 [YOLOv5 가이드](https://docs.aw-ol.com/v853/npu/npu_yolov5/)에는 YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet, 얼굴/사람 네트워크가 나옵니다.

이는 실제 컴퓨터 비전 지원이지만 이전 Tina Linux/Vivante 툴체인과 계정이 필요한 SDK 다운로드에 묶여 있습니다. 이 제한을 명시하면서 시장 지도에 포함해야 합니다.

## 간과하기 쉬운 한국, 대만, 중국 플랫폼

영어 자료에서는 Rockchip에서 NVIDIA로 곧장 넘어가며, 문서화된 실리콘을 보유한 두 번째 계층을 놓치는 경우가 많습니다. 이 생태계 중 일부는 잘 알려진 서구 스타트업보다 현재 YOLO 표가 더 넓습니다.

### Mobilint

한국 가속기 업체 Mobilint는 저전력 REGULUS와 처리량이 높은 ARIES MLA100을 판매합니다. qb SDK는 PyTorch, TensorFlow, TFLite, ONNX, Keras 입력을 받고 INT8로 양자화해 컴파일된 `.mxq` 산출물을 생성합니다. 공개 [비전 표](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md)는 YOLOv3/v5/v7/v8/v9/v10/11/12/26 탐지와 YOLO 분할, 자세 추정, OBB 변형을 명시합니다. [ARIES 페이지](https://www.mobilint.com/aries/mla100)는 YOLO11s와 YOLO26m의 모델별 결과도 공개합니다. 강한 통합 근거지만 SDK와 하드웨어 접근은 상용입니다.

### Sunplus

Sunplus SP7350/C3V는 Vivante 기반 NPU 스택을 사용하지만 Allwinner나 구형 Amlogic과 다른 방식으로 패키징합니다. 공개 절차는 Acuity 변환, NPU Docker 환경, SNNF 런타임, `.nb` 출력을 조합합니다. 공식 [사용자 지정 YOLOv8 가이드](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350)는 프레임워크 지원만 주장하지 않고 변환과 배포를 안내합니다. 관련 IP를 사용한다는 사실만으로 `.nb`가 다른 Vivante SoC에서 이식되는 것은 아닙니다.

### ESWIN Computing

ESWIN의 RISC-V 계열에는 EIC7700/EIC7700X 및 듀얼 다이 EIC7702/EIC7702X가 포함됩니다. EIC7700은 Milk-V Megrez 같은 판매 보드에 사용됩니다. ENNP는 EsQuant, 최신 EsAAC 컴파일러, 골든 데이터 생성, 시뮬레이션, ESSDK 런타임을 포함하며 `.model` 출력을 생성합니다. 이전 ENNP 자료에는 `ennc-compile`이라는 이름이 사용됩니다. 최신 [EsAAC 페이지](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac)는 ONNX 입력을 문서화하므로 다른 프레임워크는 지원 IR로 내보내거나 변환해야 합니다. Milk-V 호스팅 [ENNP 개요](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/introduction)와 [YOLOv3 튜토리얼](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3)은 공개 종단 간 경로의 근거지만, 현재 폭넓은 모델/정확도 표를 입증하지는 않습니다.

### Nuvoton M55M1

Nuvoton M55M1은 Linux 애플리케이션 프로세서가 아니라 Cortex-M55와 Ethos-U55-256으로 구성된 MCU급 설계입니다. NuEdgeWise/NuML과 Arm Vela가 양자화 TFLite Micro 모델을 준비합니다. 공개 [NuEdgeWise 저장소](https://github.com/OpenNuvoton/NuEdgeWise)는 YOLOv8-nano, YOLOX-nano, YOLO Fastest v1.1, SSD-MobileNet FPNLite, 여러 분류기를 명시합니다. 이 소형 네트워크 예제는 장치의 메모리/전력 등급을 보여 줄 뿐 일반적인 640픽셀 YOLO 변형의 근거가 아닙니다.

### Rebellions 및 FuriosaAI

Rebellions ATOM 계열은 RBLN SDK와 컴파일된 `.rbln` 산출물을 사용합니다. 공식 [지원 릴리스](https://docs.rbln.ai/v0.8.2/supports/release_note.html)는 YOLOv3 tiny/full/SPP, YOLOv5 n~x, YOLOv6 n~l, YOLOv7 변형, YOLOv8 n~x를 명시합니다. 최신 [카드 지원 표](https://docs.rbln.ai/latest/supports/version_matrix.html)는 ATOM CA02와 ATOM+ CA12를 EoL, ATOM+ CA22와 ATOM-Max CA25를 Active로 표시합니다. ATOM-Lite CA21은 제품 페이지는 있지만 해당 SDK 표에는 없어 현재 툴체인 상태가 불분명합니다. 문서는 공개되어 있지만 컴파일러 휠에는 Rebellions Portal 자격 증명이 필요합니다.

FuriosaAI는 세대별로 나눠야 합니다. Warboy는 INT8 `.enf` 산출물을 사용하는 구형 컴퓨터 비전 가속기이며, 공식 [모델 동물원](https://developer.furiosa.ai/furiosa-models/latest/)에 SSD, YOLOv5M/L, YOLOv7-w6-pose가 있습니다. RNGD는 주로 LLM/VLM 작업을 겨냥한 별도 [`.fxb` 스택](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html)을 사용합니다. Furiosa 최신 [로드맵](https://developer.furiosa.ai/latest/en/overview/roadmap.html)은 YOLOv8m 비전 지원이 2024 Q4 완료됐다고 기록하지만 현재 공개 지원 모델과 성능 자료는 LLM/VLM 중심이며 상세 YOLOv8m 정확도/성능 표는 없습니다. 이는 출시된 세대의 근거이지 Warboy 호환성 근거가 아닙니다.

### Realtek, Telechips, T-Head, SigmaStar

이 네 플랫폼은 실제 제품이지만 사용자 모델 변환 경로가 더 좁거나 오래됐거나 접근 제한이 있습니다.

| 플랫폼 | 재현 가능한 공개 근거 | 유지해야 할 제한사항 |
|---|---|---|
| Realtek AmebaPro2 | [Arduino/FreeRTOS SDK](https://github.com/Ameba-AIoT/ameba-arduino-pro2)가 YOLOv3/4/7-tiny, SCRFD, MobileFaceNet `.nb` 모델을 패키징 | 사용자 지정 모델 변환을 위해 Realtek에 관심 등록/문의 필요 |
| Telechips TOPST TCC7500 | 공식 [2026년 프로젝트](https://docs.topst.ai/blog/31)에서 YOLOv8s를 변환 및 배포 | UFLD v1/v2 변환은 미지원 레이어에서 실패했으며 문서에는 분할/후처리 우회만 제안됩니다. 전체 컴파일러/연산자 자료는 이메일 허가가 필요한 경우가 많음 |
| T-Head TH1520 | Sipeed [애플리케이션 가이드](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html)에서 HHB와 CSI-NN2/SHL로 YOLOv5n/s 실행 | 공개 스택은 있지만 유지보수와 버전 명확성이 최신 선도 제품보다 뒤처짐 |
| SigmaStar SSU9383CM | 최신 문서에 MI_IPU 런타임 API가 있고 구형 SGS_IPU 자료에 `.sim`에서 `sgsimg.img` 생성 및 SSD/YOLOv1/2/3를 언급한 후처리기가 있음 | 구형 컴파일러 산출물이나 모델 경로가 SSU9383CM에 해당한다는 공개 근거 없음 |

### HiSilicon 스마트 비전은 Huawei Ascend와 다릅니다

HiSilicon Hi3516CV610/DV500, Hi3519DV500, Hi3403V100 카메라 SoC는 NNN/SVP-NNN 런타임을 통한 ATC-to-`.om` 절차를 제공합니다. 공식 [HiSpark 모델 동물원](https://gitee.com/HiSpark/modelzoo/blob/master/README.md)은 특히 Hi3403, Hi3591P에 대해 대상별 표에서 YOLOv3/4/5/6/7/8/9/10/11, YOLO11 분할/자세 추정, YOLOv8 OBB/World/분할, OCR, 깊이 추정, TinySAM을 명시합니다. 위에 적은 스마트 비전 SoC 모두에서 각 항목이 실행된다는 뜻은 아닙니다. 일부는 로드맵 항목이므로 출시된 예제와 계획된 지원을 구분해야 하며, 저장소는 제공 모델을 비상업적 사용 전용이라고 명시합니다. ATC/`.om`이라는 명칭이 같아도 CANN 기반 Ascend 제품군과 산출물/런타임 호환성을 입증하지 않습니다.

## 신흥 및 특수 가속기 업체

### MemryX

MemryX MX3 모듈은 데이터플로 아키텍처를 사용하며 여러 칩을 결합할 수 있습니다. [Neural Compiler](https://developer.memryx.com/tools/neural_compiler.html)는 ONNX, TensorFlow Lite, Keras, TensorFlow 모델을 받아 가속기 런타임에서 사용하는 DFP 패키지를 출력합니다. [런타임 API](https://developer.memryx.com/api/accelerator/accelerator.html)는 다중 모델 및 스트리밍 파이프라인을 지원합니다.

MemryX 문서와 예제에는 탐지, 분할, 자세 추정을 위한 최적화된 YOLO 전처리/후처리가 있습니다. [SDK 릴리스 노트](https://developer.memryx.com/release_notes.html)는 YOLOv10, YOLO11, YOLO26 지원을 명시합니다. 공개 기술 자료는 유용하지만 Axelera와 비교할 만한 단일 모델/정확도/장치 표는 없습니다.

### Kneron

Kneron KL520/KL530/KL630/KL720/KL730 제품은 소형 USB 및 임베디드 가속기를 아우릅니다. Kneron PLUS는 애플리케이션 런타임을 담당하며 최신 Model Toolchain 0.33.1은 해당 대상을 위한 변환, 양자화, 평가, 시뮬레이션, `.nef` 컴파일을 문서화합니다. KL830은 일부 [PLUS 런타임 API](https://doc.kneron.com/docs/plus_c/introduction/run_examples/)에 나오지만 최신 [컴파일러 대상 목록](https://doc.kneron.com/docs/toolchain/manual_1_overview/)에는 없습니다. 벤더 확인 없이 일반 `.nef` 컴파일 주장을 KL830에 적용하면 안 됩니다.

공식 [YOLO 예제](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/)는 Tiny-YOLOv3로 과정을 시연하고 다른 Kneron 자료에서는 YOLOv5 학습 및 배포를 다룹니다. 근거는 신뢰할 수 있지만 Hailo, Axelera, Rockchip, DEEPX의 최신 YOLO 범위보다는 좁습니다.

### SiMa.ai

SiMa.ai의 MLSoC와 [양산 Modalix 50-TOPS 플랫폼](https://sima.ai/press-release/sima-ai-next-gen-platform-for-physical-ai-in-production/)은 Palette 소프트웨어 환경을 사용합니다. ModelSDK, MLA 컴파일러, ModelExecutor가 가져오기, 양자화, 컴파일, 프로파일링, 실행을 다룹니다. 작업과 제품에 따라 INT8, INT16, BF16 흐름을 지원합니다.

회사 릴리스 노트에는 검증된 YOLOv7, YOLOv8 탐지/자세 추정/분할, YOLOX 및 YOLOX 분할, DETR, Mask R-CNN, EfficientDet 파이프라인이 나옵니다. 현재 한계도 숨기면 안 됩니다. [Palette SDK 2.1 노트](https://docs.sima.ai/v2.1.2/pages/release_notes/2.1.html)는 Python/PT2E 양자화 회귀로 QAT가 작동하지 않는다고 설명합니다. 문서화된 우회 방법은 SDK 2.0에서 주석 처리된 ONNX를 만든 뒤 2.1로 컴파일하는 것입니다. 도구 및 구매는 기업 중심입니다.

배포 경계도 문서화되어 있습니다. [ModelSDK 컴파일 절차](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html)는 컴파일된 `.tar.gz`를 출력하고 실행 가능한 ELF를 생성할 수 있습니다. [MPK Tool](https://docs.sima.ai/v2.1.1/pages/palette/mpk_tools.html)은 배포 가능한 `.mpk`를 패키징합니다. 이 출력은 MLSoC/Modalix 대상과 Palette 릴리스에 종속됩니다.

### EdgeCortix

SAKURA-II는 비전 및 생성형 AI용 60-TOPS 광고 가속기이며 MERA 소프트웨어 프레임워크로 컴파일합니다. EdgeCortix는 Renesas 최신 RUHMI 흐름에도 MERA 기술을 제공합니다.

[SAKURA-II 제품 자료](https://www.edgecortix.com/en/edgecortix-sakura-ii-accelerator-brief)는 하드웨어와 컴파일러를 확인하고 [M.2/PCIe 하드웨어](https://www.edgecortix.com/en/hardware)는 체험 또는 주문 문의를 통해 제공됩니다. 충분히 구체적이고 최신인 공개 YOLO 호환성 표는 찾지 못했습니다. 이는 구매 시 유용한 정보입니다. 하드웨어를 결정하기 전에 모델 검증을 요청해야 합니다.

### BrainChip

BrainChip Akida는 기존 밀집 텐서 NPU가 아니라 뉴로모픽 이벤트 도메인 프로세서입니다. [MetaTF 환경](https://brainchip.com/metatf-dev-tools/)은 모델 생성, 저비트 양자화, 변환, 시뮬레이션, 하드웨어 실행을 위한 설치형 Python 패키지로 구성됩니다. BrainChip은 현재 [출하 중 AKD1500 코프로세서/M.2 제품](https://brainchip.com/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/)을 판매하며 레거시 AKD1000 하드웨어, Akida 2 IP도 제공합니다.

최신 [Akida 2 모델 카드](https://brainchip.com/wp-content/uploads/2025/04/Akida-2-Model-Card-V1.1-Mar.25.pdf)는 AkidaNet0.5 YOLOv2 탐지기, CenterNet, AkidaUNet, 얼굴 인식 네트워크를 명시합니다. 다른 지원 자료에는 FOMO와 이벤트 기반 작업도 포함됩니다. Akida는 매우 낮은 비트 가중치와 활성화를 지원하지만 일반 INT8 ONNX 대상처럼 다루지 말고 아키텍처를 고려해 변환해야 할 수 있습니다. 명시적인 매핑/적합성 보고서가 없는 한 AKD1000 또는 Akida 2 모델 동물원 결과를 AKD1500에 적용하면 안 됩니다.

### Blaize

Blaize는 Picasso SDK, NetDeploy, AI Studio를 소프트웨어 경로로 제공하는 P1600 기반 Pathfinder 및 Xplorer 제품을 판매합니다. [제품 페이지](https://www.blaize.com/products/)는 비전과 엣지 AI를 명확한 목표 시장으로 제시하지만 최신 공개 모델별 호환성 표는 없습니다.

따라서 이 가이드에서는 커뮤니티 주장을 근거 없이 보태지 않고 Blaize를 상용 및 접근 제한 대상으로 분류합니다. 사용하려는 모델의 벤더 제공 컴파일 및 정확도 보고서를 선행 조건으로 삼아야 합니다.

## TinyML 및 엣지 비전

### Arm Ethos-U 및 Alif

Arm은 Ethos-U55, U65, U85 NPU IP를 칩 업체에 라이선스합니다. [Vela 컴파일러](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u)는 양자화 LiteRT/TFLite 그래프를 받아 지원 하위 그래프를 Ethos-U 사용자 지정 연산자로 변환합니다. 지원되지 않는 연산은 CPU에 남을 수 있으므로 "애플리케이션이 실행된다"와 "네트워크 전체가 NPU에서 실행된다"는 서로 다른 주장입니다.

실제 제품에는 Alif Ensemble E7의 Ethos-U55, NXP i.MX 93의 Ethos-U65, 최신 시스템의 Ethos-U85가 포함됩니다. Alif [E7 AI/ML AppKit](https://alifsemi.com/support/kits/ensemble-e7appkit/)은 ML 가속기 2개를 설명하고 E7의 TFLite-to-Vela 경로를 시연합니다. [Ensemble E8](https://alifsemi.com/ensemble-e8-series/)은 Ethos-U85 하나와 Ethos-U55 NPU 두 개를 결합합니다. Alif는 Cortex-M55와 Ethos-U55에서 192 x 192로 실행한 제품군 수준 [INT8 YOLO-Fastest 얼굴 탐지 벤치마크](https://alifsemi.com/faster-ai-mcu-inferencing-low-power-consumption/)도 공개하지만 최신 YOLO의 폭넓은 대상별 표는 없습니다. 이런 메모리 제약 시스템은 작은 분류/탐지 네트워크에 적합하며 TFLite로 표현할 수 있다는 이유만으로 임의의 640픽셀 탐지기가 들어가는 것은 아닙니다.

### Infineon PSOC Edge, Himax WiseEye2, Analog Devices MAX7800x

Infineon [PSOC Edge E83](https://documentation.infineon.com/psocedge/docs/xsk1761304020519) 및 E84 계열은 Cortex-M55와 Ethos-U55를 결합하고 저전력 M33 측에 별도 NNLite 블록을 더합니다. [E84 아키텍처 매뉴얼](https://documentation.infineon.com/psocedge/docs/bwb1750411526047)은 8비트 가중치, 8/16비트 활성화, 지원 연산자를 NPU 명령 스트림으로 변환하고 나머지를 CPU에 남기는 Vela 흐름을 설명합니다. [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter)는 TFLite, Keras, PyTorch 경로를 받습니다. 아키텍처 자료는 대상 벤치마크가 아닌 대표 커널로 MobileNetV1/V2를 듭니다. [E84 AI Kit](https://documentation.infineon.com/psocedge/docs/cci1762693051052)는 카메라를 포함하고 ModusToolbox를 사용하지만 공개 자료에는 아직 모델 이름이 명시된 YOLO 호환성 표가 없습니다. 모델 수준 근거가 발전 중인 프로그래밍 가능한 비전 플랫폼이며, 범용 탐지기 대상으로 검증된 것은 아닙니다.

Himax [HX6538 WiseEye2](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/)도 상시 동작 비전을 위해 Cortex-M55와 Ethos-U55를 결합합니다. 이 전력 등급에서는 공개 소프트웨어 자료가 특히 구체적입니다. [공식 Grove Vision AI Module V2 예제](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2)에 YOLOv8n 객체 탐지, 자세 추정, 성별 분류, YOLO11n 탐지, 얼굴 메시, PeopleNet이 있습니다. 통합 경로는 TFLite Micro와 Vela를 사용하고 필요한 경우 CMSIS-NN 및 참조 커널로 폴백합니다. 이는 의도적으로 작은 모델과 해상도이며 일반적인 서버급 YOLO 그래프가 적합하다는 근거가 아닙니다.

Analog Devices는 [MAX78000/MAX78002 CNN 가속기](https://www.analog.com/en/products/max78002.html)로 다른 경로를 택합니다. 공개 [`ai8x-synthesis`](https://github.com/analogdevicesinc/ai8x-synthesis) 도구는 학습된 네트워크를 양자화하고 이식 가능한 ONNX 런타임 패키지가 아니라 대상별 C 코드, 가중치, 가속기 구성을 생성합니다. 자사 근거에는 [얼굴 식별](https://www.analog.com/en/resources/app-notes/an-2616.html), [ADI 모델 동물원](https://ez.analog.com/dsp/software-and-development-tools/edgebench/a/docs-faqs/DF912/available-models-in-adi-model-zoo)의 TinierSSD QR 탐지, 이미지 분류기가 있습니다. 엣지 비전에 유용한 장치지만 최신 YOLO의 공식 표는 찾지 못했습니다.

### GreenWaves GAP9

GAP9은 RISC-V 연산과 NE16 신경망 엔진을 결합합니다. NNTool은 네트워크를 가져와 양자화하고 AutoTiler와 GAP SDK가 배포 코드를 생성합니다. 공개 [GAP9 신경망 메뉴](https://github.com/GreenWaves-Technologies/nn_menu_gap9)에는 MobileNet, EfficientNet, ResNet, MobileNet SSD, 얼굴 인식, 인식 애플리케이션이 있습니다. GreenWaves는 정확도 및 시뮬레이션 결과가 포함된 [YOLOX 사람 탐지 프로젝트](https://github.com/GreenWaves-Technologies/yolox_people_detection)도 공개합니다.

이는 투명성이 특히 높은 TinyML 근거지만 전체 SDK는 자격을 갖춘 고객에게만 제공되며 모델 크기도 일반적인 640 x 640 YOLO 변형보다 훨씬 작습니다.

### Lattice sensAI

Lattice sensAI는 고정 NPU의 FPGA 대안입니다. sensAI Studio와 Neural Network Compiler는 ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E, Avant-X 장치의 구성 가능한 가속기 IP에 지원 그래프를 매핑합니다. 출력은 이식 가능한 런타임 모델이 아니라 선택한 FPGA, 가속기 모드, 메모리 레이아웃, 비트스트림에 종속됩니다.

최신 [Neural Network Compiler 매뉴얼](https://www.latticesemi.com/-/media/LatticeSemi/Documents/UserManuals/MQ3/FPGA-UG-02052-8-0-Lattice-Neural-Network-Compiler-Software.ashx?document_id=52343)은 구체적인 토폴로지 표를 제공합니다. ECP5의 YOLOv1, CrossLink-NX 및 CertusPro-NX의 최적화/고급 모드 YOLOv5/YOLOv8, Advanced 모드 IP 전용 YOLO11을 표시합니다. FPGA 제품군별 미지원 셀이 나타난 SSD, MobileNetV2-SSD, ResNet, ENet, SqueezeDet 등도 있습니다. [Advanced CNN Accelerator](https://www.latticesemi.com/en/Products/DesignSoftwareAndIP/IntellectualProperty/IPCore/IPCores05/Advanced-CNN-Accelerator-IP)는 Avant-E/Avant-X/CertusPro-NX를 대상으로 하는 상용 IP입니다. 이는 컴파일러 표의 근거이지 비트스트림 하나에서 모든 토폴로지를 동시에 지원한다는 보장은 아닙니다.

### Microchip VectorBlox

Microchip 공개 [VectorBlox SDK 3.1](https://github.com/Microchip-Vectorblox/VectorBlox-SDK)은 PolarFire SoC FPGA의 구성 가능한 CoreVectorBlox IP를 위해 완전 양자화 INT8 TFLite 네트워크를 전처리하고 컴파일합니다. `tflite_preprocess`와 `vnnx_compile`은 `.vnnx`, `.hex`, `.ucomp` 배포 파일을 만들며 저장소에는 시뮬레이터와 대상 측 C 드라이버가 있습니다. 업스트림 TensorFlow, ONNX, OpenVINO 그래프도 문서화된 TFLite/INT8 준비 경로를 통과해야 합니다.

현재 [튜토리얼 표](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md)에는 YOLOv5n, YOLOv8n 탐지, 분류, OBB, 자세 추정, 분할, YOLOv9t, 희소/압축 YOLO 변형, MobileNet, EfficientNet-Lite0, ResNet18, MiDaS, FFNet, QuickSRNet이 포함됩니다. [C 후처리 가이드](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/C_Postprocessing.md)는 YOLOv2/3/4/5와 Ultralytics 탐지, 자세 추정, OBB 디코더를 별도로 명시합니다. SDK 3.1은 현재 PolarFire SoC Video Kit을 지원하고 비 SoC PolarFire Video Kit은 명시적으로 제외하므로 구형 보드 데모와 현재 대상 계약을 혼동하면 안 됩니다.

SDK는 소프트웨어와 파생물의 사용을 Microchip 제품으로 제한하는 [Microchip 전용 라이선스](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/LICENSE.md)로 공개 다운로드됩니다. Microchip은 요청 기반 무료 Libero Silver 및 CoreVectorBlox 라이선스를 제공합니다. 배포는 FPGA 시스템 계약에 여전히 종속됩니다. 가속기 구성, 비트스트림, 펌웨어, SDK 생성 네트워크 데이터, 메모리 레이아웃이 서로 일치해야 합니다. VectorBlox 컴파일이 성공했다고 임의의 PolarFire 설계에 모델 바이너리를 복사할 수 있는 것은 아닙니다.

### Syntiant

Syntiant NDP200은 일반 Linux SoC보다 낮은 전력 범위에서 상시 동작 비전, 오디오, 센서 추론을 대상으로 합니다. 현재 [하드웨어 표](https://www.syntiant.com/hardware)는 NDP200을 양산 중, NDP250을 샘플링 중으로 표시합니다. [NDP200 제품 설명서](https://www.syntiant.com/ndp200/)에는 CNN, RNN, 완전 연결 네트워크가 1 mW 미만으로 명시되어 있습니다. [NDP250 발표](https://www.syntiant.com/news/syntiant-unveils-ndp250-neural-decision-processor-with-next-gen-core-3-architecture)는 상시 동작 이미지 인식을 30 mW 미만으로 설명합니다. "1 mW 미만"을 두 제품 모두에 일반화하면 안 됩니다.

공개 자료는 광범위한 YOLO 호환성을 입증하지 않습니다. 따라서 일반 ONNX 내보내기를 가정하지 말고 벤더 지원 모델 평가를 통해 작은 상시 동작 분류기와 탐지기를 검토해야 합니다.

### Google Coral: 서로 무관한 두 세대

레거시 Edge TPU 제품은 완전 INT8 TFLite 모델과 Edge TPU Compiler, `libedgetpu`, PyCoral을 사용합니다. 주요 [Edge TPU](https://github.com/google-coral/edgetpu) 및 [PyCoral](https://github.com/google-coral/pycoral) 저장소는 보관 처리되었습니다. 이는 유지보수 위험이지만 공식 하드웨어 단종 공지는 아닙니다.

Google은 저전력 SoC 통합용 RISC-V 가속기 IP인 별도 활성 오픈 소스 [Coral NPU](https://github.com/google-coral/coralnpu)도 유지합니다. 현재 공개 프로젝트에는 RTL, 하드웨어 시뮬레이션, 툴체인, ELF 예제가 있습니다. 이는 Edge TPU의 직접적인 USB/M.2 후속품이 아니며 최신 YOLO 배포 표도 공개하지 않습니다.

## 인접 GPU, 클라이언트 NPU, 적응형 연산 관련 주의점

NVIDIA, AMD, Intel, Apple은 NPU 검색에 자주 등장하지만 서로 대체 가능한 한 종류의 가속기를 제공하지는 않습니다.

**NVIDIA Jetson:** Jetson Orin은 CUDA GPU와 전용 DLA 코어를 결합합니다. TensorRT는 DLA용 직렬화 엔진을 만들 수 있지만 GPU 폴백을 명시적으로 활성화한 경우에만 미지원 레이어가 GPU에서 실행됩니다. 최신 [DeepStream YOLO 표](https://github.com/NVIDIA-AI-IOT/deepstream_tools/blob/main/yolo_deepstream/README.md)에는 YOLOv4/v7/v8/v9/11이 있지만 대부분 GPU 대상이며 ONNX DLA 항목으로 명시된 것은 YOLOv8s INT8입니다. NVIDIA [DLA 제한사항](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html)은 연산자 제한을 설명합니다. Thor도 구분해야 합니다. NVIDIA [DriveOS 마이그레이션 가이드](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/NVIDIA_DriveOS_7.0.3_Migration_Guide.pdf)에 따르면 GPU 스케줄링 유연성을 위해 Thor에서 DLA 코어를 제거했습니다. 따라서 "Jetson에서 실행"만으로 어떤 가속기를 썼는지 알 수 없습니다.

**AMD Vitis AI:** 이전 Kria/DPU 세대는 `.xmodel` 그래프를 컴파일하고 VART로 실행했습니다. 최신 [Vitis AI 6.2](https://vitisai.docs.amd.com/en/6.2/)는 별도의 두 Versal AI Edge 경로에서 GA입니다. VEK280/VE2802의 [Gen1](https://vitisai.docs.amd.com/projects/gen1/en/latest/index.html)은 ONNX, AMD Quark, 대상 종속 스냅샷/하위 그래프 절차를 사용하고 VEK385의 [Gen2](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html)는 별도 컴파일/런타임 계약이 있습니다. Gen1은 YOLOv5, YOLOv7, YOLOv8, YOLOX 예제를 공개합니다. Ryzen AI는 네 번째로 별도인 클라이언트 NPU 스택입니다. 모두 Vitis 또는 AMD 이름을 사용한다고 레거시 DPU, Versal Gen1/Gen2, Ryzen AI 간에 산출물이나 검증 정보를 옮기지 마십시오.

**Intel OpenVINO:** OpenVINO는 하나의 API로 CPU, GPU, Core Ultra NPU를 대상으로 할 수 있습니다. [Intel NPU 플러그인 문서](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md)는 지원 NPU 세대를 밝히고 [검증 모델 표](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html)는 장치별 열을 제공합니다. YOLO 노트북은 애플리케이션 절차의 근거이지 해당 표에 표시되지 않았다면 NPU 검증의 근거가 아닙니다. 지원 연산, 모양, 정밀도, 설치된 NPU 드라이버가 그래프 전체의 실행 여부를 결정합니다. OpenVINO IR(`.xml` 및 `.bin`)은 재사용 가능한 소스 수준 IR이고 컴파일 모델 캐시는 장치 및 소프트웨어에 종속됩니다.

**Apple Core ML:** `coremltools`는 모델을 `.mlpackage`로 변환하고 Xcode가 이를 `.mlmodelc`로 컴파일합니다. Core ML은 CPU, GPU, Apple Neural Engine에 연산을 배치할 수 있지만 정확한 분할은 의도적으로 추상화합니다. 따라서 프로파일링으로 확인하지 않은 채 "YOLO 그래프 전체가 NPU에서 실행됐다"고 주장하기는 어렵습니다.

## 칩 업체 뒤의 NPU IP 기업

일부 기업은 보드나 패키지형 가속기를 직접 판매하지 않습니다. SoC 업체에 NPU 설계를 라이선스합니다. 같은 기반 아키텍처가 여러 칩 브랜드에 나타날 수 있지만 최종 SDK와 산출물은 라이선스를 받은 업체가 별도로 맞춤화할 수 있으므로 중요합니다.

| IP 공급업체 | NPU 계열 및 소프트웨어 | 중요한 이유 |
|---|---|---|
| [Arm](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) | Ethos-U55/U65/U85, Vela | NXP, Alif, Infineon, Himax 등 임베디드 제품에 탑재. 양자화 TFLite/TOSA 중심 흐름 |
| [Arm China](https://www.armchina.com/mountain?infoId=161&name=) | Zhouyi AIPU, Compass SDK | 공개 [모델 동물원](https://github.com/Arm-China/Model_zoo)에 YOLOv1-tiny/v2/v3/v4/v5, YOLOX, YOLOv8-seg 명시. 참조 모델이며 모든 라이선스 업체 칩의 근거는 아님 |
| [VeriSilicon](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico) | Vivante VIP 계열, Acuity SDK | 별도로 공급되는 A311D, i.MX 8M Plus 등 여러 SoC에 관련 Vivante NPU 계열이 사용됩니다. 각 칩 업체는 자체 통합을 제공합니다. |
| [Imagination Technologies](https://www.imaginationtech.com/products/open-access/) | PowerVR Series3NX AX3146/AX3386/AX3596, 과거 IMG Series4, Neural Compute SDK/IMG DNN | 소매 보드가 아닌 라이선스 NNA IP입니다. Open Access는 Series3NX의 1/5/10-TOPS 구성을 제공하고 공식 NC-SDK 프로그램은 PP-YOLOE, EfficientNet, HRNet을 명시하지만 모든 IP 구성을 포괄하지는 않습니다. |
| [CEVA](https://www.ceva-ip.com/product/ceva-neupro-studio/) | NeuPro-Nano/NeuPro-M, NeuPro Studio | 가져오기, 양자화, 압축, 그래프 컴파일, 시뮬레이션, C/C++ 생성을 지원하는 라이선스 NPU IP |
| [MIPS](https://mips.com/processor-solutions/arc-npx-family/) | ARC NPX6, [MetaWare MX](https://mips.com/processor-solutions/arc-metaware-mx/), 이전 Synopsys ARC | 확장형 NPU IP와 NN SDK는 GlobalFoundries가 인수해 [2026년 6월](https://mips.com/press-releases/gfmipsarcclose/) MIPS 포트폴리오에 편입. 소매 배포 대상 자체는 아님 |
| [Cadence](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/neuroweave-sdk.html) | Neo NPU, Tensilica DSP, NeuroWeave SDK | SoC 라이선스 업체가 사용하는 컴파일러/인터프리터 스택. 네트워크 및 양자화 지원은 라이선스 구성에 종속 |
| [Quadric](https://quadric.ai/npu-ip) | Chimera GPNPU IP, SDK | 다른 기업 칩에 라이선스되는 프로그래밍 가능한 NPU/DSP형 IP. 최종 대상은 라이선스 업체 실리콘 |
| [Expedera](https://www.expedera.com/products-overview/) | Origin NPU IP 및 소프트웨어 스택 | 라이선스 가능한 엣지 NPU 아키텍처이며 LibreYOLO가 바로 구매할 수 있는 보드는 아님 |

Imagination은 아키텍처 이름 옆에 접근성과 수명 주기를 표시해야 하는 이유를 보여 줍니다. [Open Access 프로그램](https://www.imaginationtech.com/products/open-access/)은 선불 IP 라이선스 비용 없이 실리콘 검증 Series3NX 구성 세 가지를 제공하지만, 회사 평가 이후 유료 지원/유지보수와 양산 로열티가 적용됩니다. [Neural Compute SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/)에는 컴파일, 최적화, 양자화, 런타임 도구가 있습니다. 공식 [Baidu PaddlePaddle 협업](https://www.imaginationtech.com/news/imagination-and-baidu-paddlepaddle-create-open-source-machine-learning-library-for-model-zoo/)은 PP-YOLOE 탐지, EfficientNet 분류, HRNet 분할을 명시합니다. 이 예제는 모든 Series3NX 또는 Series4 구성을 검증하지 않습니다. 현재 개별 [Series4 제품 페이지](https://www.imaginationtech.com/product/img-4nx-mc1/)에서 제품을 구매할 수 없다고 표시하므로 영업팀이 확인하기 전까지 Series4는 과거 제품으로 취급해야 합니다.

이 계층은 제품 계열이 겉보기에 비슷한 이유를 설명합니다. 하지만 산출물의 이식성을 만들지는 않습니다. Vivante 또는 Ethos IP가 관련되어 있어도 두 SoC의 메모리 맵, 드라이버, 컴파일러 버전, 연산자 집합, 벤더 런타임은 다를 수 있습니다.

## 실제 배포 대상: 산출물 종속성 표

컴파일 파이프라인의 마지막 파일에서 ONNX 이식성의 한계가 드러납니다. 이름과 확장자는 시간이 지나며 바뀌지만 실무 원칙은 같습니다. 네이티브 산출물은 대체로 다른 NPU 세대로 옮길 수 없고 다른 SDK 릴리스로 안정적으로 다시 만들 수도 없으므로 원본 모델과 보정 데이터를 보관하십시오.

| 벤더 스택 | 일반적인 컴파일 입력 | 장치에 전달되는 파일 | 종속 항목 |
|---|---|---|---|
| [Amlogic AMLNN](https://github.com/Amlogic-NN/amlnn-toolkit) | ONNX, TFLite, TorchScript, PT2 | `.adla`; 레거시 A311D는 `.nb` | 대상 ID, ADLA 세대, AMLNN 컴파일러 빌드, NNSDK2/런타임, ADLA 드라이버, BSP |
| [Rockchip RKNN](https://github.com/airockchip/rknn-toolkit2) | ONNX, TFLite, 프레임워크 내보내기 | `.rknn` | RKNN 툴킷/런타임 버전, Rockchip NPU 계열 |
| [Hailo](https://hailo.ai/developer-zone/documentation/) | 중간 HAR를 통한 ONNX 또는 TensorFlow | `.hef` | Hailo 아키텍처, 컴파일러/런타임 세대 |
| [Axelera Voyager](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) | ONNX 또는 지원 모델 동물원 정의 | 알파 Pipeline Builder의 `.axm`, `.axe` 파이프라인 패키지; 기존 `.axmodel` 및 매니페스트 | Voyager API 세대, Metis 대상 구성 |
| [DEEPX DXNN](https://github.com/DEEPX-AI/dx-all-suite) | ONNX, 지원 프레임워크 경로 | `.dxnn` | DX-COM/DX-RT 버전, DX-M 대상 |
| [Qualcomm QAIRT/QNN 또는 SNPE](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | ONNX, TFLite, 프레임워크 모델 | QNN 모델 라이브러리/컨텍스트 바이너리 또는 SNPE `.dlc` | HTP 아키텍처, SoC, 백엔드, 런타임 버전 |
| [MediaTek NeuroPilot](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) | 변환/양자화된 TFLite | 오프라인 Neuron Runtime용 `.dla`; 위임 모드용 `.tflite` | NP/MDLA 세대, OS 이미지, 컴파일러, 런타임 |
| [TI TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools) | ONNX 또는 TFLite | 가져온 산출물 디렉터리 및 애플리케이션 모델 파일 | TIDL 도구/런타임, 프로세서 세대 |
| [NXP eIQ](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | 대체로 TFLite 또는 ONNX | 대상별 Vela, TIM-VX, Neutron, Ara 출력 | 선택한 i.MX/Ara 가속기와 BSP. 포괄적인 "NXP"로 묶이지 않음 |
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | TFLite, ONNX, 지원 프레임워크 모델 | 생성된 네트워크 라이브러리/코드, 펌웨어 자산 | MCU/NPU 대상, 메모리 구성, 도구 버전 |
| [Infineon PSOC Edge](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) | 정수 양자화 TFLite/Keras/PyTorch 경로 | Ethos-U 명령 스트림이 포함된 Vela 최적화 TFLite 및 애플리케이션 펌웨어 | E83/E84 변형, Ethos-U 구성, Vela, ModusToolbox, BSP. NNLite는 별도 대상 |
| [Renesas DRP-AI](https://github.com/renesas-rz/rzv_drp-ai_tvm) | TVM 또는 Translator를 통한 ONNX/TFLite | 여러 바이너리 데이터 파일이 포함된 런타임 모델 디렉터리 | RZ/V 장치, Translator/런타임 세대 |
| [Sony IMX500](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | Edge-MDT 절차의 양자화/패킹된 네트워크 | `.rpk` 패키지 | 센서 펌웨어, 메모리 예산, packer 버전 |
| [Synaptics](https://developer.synaptics.com/) | TFLite/ONNX 및 지원 입력 | SyNAP의 `.synap`; Torq의 `.vmfb` | SL16xx와 SL261x 소프트웨어/하드웨어 세대 구분 |
| [Ambarella CVflow](https://www.ambarella.com/developer/) | 파트너 툴체인 입력 | 파트너 전용 배포 패키지. 정확한 산출물 계약은 공개되지 않음 | Cooper Developer Zone에서 CVflow 세대, SDK/BSP, 카메라 파이프라인 확인 |
| [D-Robotics BPU](https://github.com/D-Robotics/rdk_model_zoo) | ONNX 및 툴체인 지원 그래프 | X5 `.bin`; 최신 `rdk_s` `.hbm`; X3는 이전 플랫폼 흐름 | BPU 세대, 저장소 브랜치, 대응 OpenExplorer/런타임 릴리스 |
| [AXERA Pulsar2](https://github.com/AXERA-TECH/ax-samples) | ONNX | `.axmodel` | AX 칩 대상, Pulsar2, AXEngine 버전 |
| [SOPHGO TPU-MLIR](https://github.com/sophgo/tpu-mlir) | ONNX, TFLite, TorchScript 등 | BM의 `.bmodel`, CV18xx의 `.cvimodel` | BM/CV 칩 대상, SOPHON 런타임 세대 |
| [Huawei Ascend CANN](https://www.hiascend.com/en/software/cann) | ONNX 또는 지원 프레임워크 그래프 | `.om` | Ascend 칩, CANN/ATC, 장치 펌웨어/드라이버 |
| [Cambricon MagicMind](https://github.com/Cambricon/magicmind_cloud) | 프레임워크 그래프 또는 교환 모델 | 직렬화된 MagicMind 엔진/모델 | MLU 아키텍처, Neuware/MagicMind 버전 |
| [Canaan nncase](https://github.com/kendryte/nncase) | ONNX 또는 TFLite | `.kmodel` | KPU 세대, 대응 nncase 대상 플러그인 |
| [Mobilint qb](https://www.mobilint.com/sdk-qb) | ONNX 및 지원 프레임워크 모델 | `.mxq` | REGULUS/ARIES 대상, qb 컴파일러/런타임 |
| [Rebellions RBLN](https://docs.rbln.ai/latest/index.html) | PyTorch 2 또는 TensorFlow 지원 그래프 | `.rbln` | ATOM 세대, RBLN 컴파일러/런타임 |
| [FuriosaAI](https://developer.furiosa.ai/docs/latest/en/) | ONNX/TFLite 및 지원 프레임워크 경로 | Warboy `.enf`; RNGD `.fxb` | 완전히 다른 가속기 및 SDK 세대 |
| [Sunplus SNNF](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2004353133) | Acuity 지원 네트워크 | `.nb` | SP7350 NPU 드라이버, 런타임, BSP |
| [ESWIN ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) | 문서화된 최신 EsAAC 절차의 ONNX; 다른 프레임워크는 내보내기/변환 필요 | `.model` | 정확한 EIC7700 계열 대상, ENNP/ESSDK 릴리스 |
| [Nuvoton / Ethos-U](https://github.com/OpenNuvoton/NuEdgeWise) | 양자화 TFLite | Ethos-U 사용자 지정 연산을 포함하는 Vela 최적화 TFLite | M55M1 메모리 계획, Vela, 펌웨어 |
| [Himax WiseEye2](https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2) | 완전 정수 양자화 TFLite | 모델 플래시의 Vela 최적화 `.tflite`, `output.img` 같은 보드 펌웨어 | HX6538/WE2 구성, Vela, TFLite Micro, Ethos-U 드라이버, 폴백 커널 |
| [Analog Devices AI8X](https://github.com/analogdevicesinc/ai8x-synthesis) | PyTorch 체크포인트, 네트워크 YAML, 샘플 입력 | 장치별 C 소스/헤더, 가중치, 컴파일 펌웨어 | MAX78000/MAX78002 메모리/레이어 제한, synthesis 도구, MSDK 릴리스 |
| [Realtek AmebaPro2](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | 벤더 변환 서비스/입력 | `.nb` | RTL8735B 펌웨어, VoE/NeuralNetwork API 릴리스 |
| [Telechips Enlight](https://docs.topst.ai/product/p/ai) | Darknet, TensorFlow, ONNX, PyTorch 경로 | `.enlight` 중간 산출물 및 컴파일된 배포 번들 | TCC7500 NPU, TC-NN/Enlight 버전 |
| [T-Head HHB](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | ONNX 및 지원 프레임워크 그래프 | `hhb.bm`, 생성 매개변수/코드, 실행 파일 | TH1520 CSI-NN2/SHL 스택 |
| [SigmaStar MI_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | 최신 SSU9383CM 문서는 MI_IPU 런타임을 공개, 구형 자료는 TensorFlow/Caffe 계열 입력 사용 | 구형 `.sim`에서 `sgsimg.img`로 변환. 최신 공개 산출물은 미검증 | 정확한 SigmaStar IPU/SDK 세대. SSU9383CM과 레거시 컴파일러 호환성 미검증 |
| [HiSilicon smart vision](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) | ONNX 및 ATC 지원 그래프 | `.om` | 정확한 스마트 비전 SoC와 NNN/SVP-NNN 스택. Ascend 범용 이식성 없음 |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | ONNX, TFLite, Keras, TensorFlow | `.dfp` 데이터플로 패키지 | MX 하드웨어 구성, 런타임/컴파일러 릴리스 |
| [Kneron](https://doc.kneron.com/docs/) | ONNX 및 툴체인 구성 | `.nef`; KL730 NEFv2는 `.kne` 포함 가능 | 문서화된 컴파일 대상, 칩 세대, 펌웨어, PLUS 런타임. Toolchain 0.33.1에서 KL830 컴파일은 입증되지 않음 |
| [SiMa.ai Palette](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) | 지원 프레임워크 또는 ONNX 모델 | ModelSDK 컴파일 `.tar.gz`, 문서화된 실행 ELF, MPK Tool `.mpk` 패키지 | MLSoC/Modalix 대상, Palette 릴리스 |
| [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | ONNX 또는 네트워크 정의 | 직렬화 엔진, 보통 `.engine` 또는 `.plan` | GPU/DLA 아키텍처, TensorRT, CUDA, 종종 장치 |
| [AMD Vitis AI](https://vitisai.docs.amd.com/en/6.2/) | 양자화 ONNX/프레임워크 그래프 | 레거시 `.xmodel`; Gen1 NPU 스냅샷/하위 그래프; Gen2 컴파일 캐시 디렉터리 또는 프로덕션 `.rai` 패키지 | 정확한 NPU 세대/IP 구성, 플랫폼 이미지, Vitis/Vivado/PetaLinux 조합 |
| [Microchip VectorBlox](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) | 완전 양자화 INT8 TFLite | `.vnnx`, `.hex`, `.ucomp`와 대응 펌웨어/FPGA 설계 | CoreVectorBlox 구성, PolarFire SoC Video Kit, SDK 3.1, 비트스트림, 메모리 레이아웃 |
| [Intel OpenVINO](https://docs.openvino.ai/) | 프레임워크 모델 또는 ONNX | IR `.xml` 및 `.bin` 또는 장치별 컴파일 캐시 | IR은 재사용 가능, 컴파일 캐시는 장치/드라이버/OpenVINO에 종속 |
| [레거시 Google Edge TPU](https://coral.ai/docs/edgetpu/models-intro/) | 완전 INT8 TFLite | Edge TPU 컴파일 `_edgetpu.tflite` | Edge TPU 컴파일러/런타임 및 양자화 연산자 지원. 신규 Coral NPU IP와 무관 |
| [Google Coral NPU IP](https://github.com/google-coral/coralnpu) | 현재 공개 IP 프로젝트의 C/C++ 예제 | RTL/하드웨어 시뮬레이션 및 향후 SoC 통합용 ELF 예제. 공개 YOLO 컴파일 산출물 없음 | 정확한 라이선스/통합 SoC 구현, 발전 중인 오픈 소스 툴체인 |
| [Lattice sensAI](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | 지원 네트워크 설명 | FPGA 비트스트림, 가속기 구성, 가중치 | FPGA 제품군, sensAI IP 모드, 메모리 구현 |
| [Imagination NC-SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) | Caffe, TensorFlow, ONNX | 고객용 컴파일 산출물 파일명 공개되지 않음 | 라이선스 NNA 구성, SoC 통합, NC-SDK/DDK, 라이선스 업체 런타임 |

이 표는 확장자만 정리한 것이 아닙니다. 컴파일러 버전, 대상 식별자, 드라이버, 펌웨어, BSP는 모델 재현 정보의 일부입니다. 향후 LibreYOLO 내보내기 도구는 산출물 옆에 기계 판독 가능한 매니페스트를 기록해야 합니다.

## 도구 접근성과 제품 수명 주기 신호

하드웨어 구매 가능성과 컴파일러 접근성은 별개입니다. 보드는 쉽게 구할 수 있지만 현재 컴파일러에는 고객 계약이 필요할 수 있고, SDK는 공개되어도 대상 실리콘을 구하기 어려울 수 있습니다. 다음은 보편적인 수명 순위가 아니라 구체적인 1차 자료 신호입니다.

| 플랫폼 | 문서로 확인된 신호 | 실무상 의미 |
|---|---|---|
| Amlogic ADLA | Apache-2.0 공개 저장소와 다운로드 가능한 휠. 호환 보드 드라이버/BSP에는 Amlogic 또는 보드 업체가 필요할 수 있음 | 컴파일러 평가가 쉽지만 바이너리 재배포와 전체 호환성 표를 확인해야 함 |
| MediaTek Genio | 공개 IoT AI Hub 및 Yocto 가이드. NP8 올인원 도구와 Android 문서는 직접 고객/NDA로 표시 | 모델 근거는 확인 가능하지만 사용자 모델 자동화에는 파트너십이 필요할 수 있음 |
| Hailo | 공개 모델 동물원과 런타임. Dataflow Compiler는 Developer Zone 배포. 공개 `hailo-camera-apps`는 2026년 5월 3일 보관 처리 | 모델 검색은 개방적이지만 재현 가능한 컴파일에는 올바른 계정/버전이 필요. 보관 상태는 Hailo-15 단종 근거가 아니며 최신 비전 프로세서 애플리케이션 패키지를 확인해야 함 |
| Axelera AI | 공개 문서 및 [공개 Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk), 고객 지원은 계정 필요 | 드문 자체 서비스형 가속기 SDK지만 제품 지원과 조달은 상용 |
| Qualcomm Dragonwing | [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program)은 IQ-9075를 2038년까지 수명인 Sampling, QCS6490을 2036년 7월까지로 표시하지만 IQ-9075 제품 페이지는 Active라고 표기 | 산업 계획에 좋은 신호지만 1차 자료의 상태 충돌을 SKU별로 확인해야 하며 해당 기간 내 AI SDK의 ABI 안정성을 보장하지 않음 |
| Rebellions | 최신 [지원 표](https://docs.rbln.ai/latest/supports/version_matrix.html)는 CA02/CA12를 EoL, CA22/CA25를 Active로 표시. CA21은 없음 | ATOM 계열 내에서도 카드별 조달 및 SDK 호환성이 다름 |
| NVIDIA Jetson | [공식 모듈 수명 주기 표](https://developer.nvidia.com/embedded/lifecycle)에 Orin 모듈은 2032년 1월까지, AGX Orin Industrial은 2033년 7월까지로 기재. 개발 키트에는 수명 약정 없음 | 개발 키트 공급이 아니라 모듈을 기준으로 양산 설계 |
| Raspberry Pi용 Sony IMX500 | [AI Camera 제품 설명서](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf)는 적어도 2028년 1월까지 생산한다고 명시 | 해당 카메라 모듈의 최소 수명이며 모든 IMX500 제품에 적용되지 않음 |
| Amlogic A311Y3 | [2026년 출시 페이지](https://www.amlogic.com/News/index248.html)는 최소 10년 제품 수명 보장을 광고 | 유망한 신제품 신호지만 제품 프로그램에 적용할 때 계약과 SKU 세부사항 확인 필요 |
| Google Coral | 레거시 Edge TPU/PyCoral 저장소는 보관/읽기 전용, 별도 [Coral NPU IP](https://github.com/google-coral/coralnpu) 저장소는 활성 상태 | 레거시 소프트웨어 유지보수 위험이지 공식 하드웨어 단종 공지는 아님. 무관한 오픈 소스 IP 프로젝트의 수명과도 별개 |

## TOPS는 구매 기준이 아닙니다

최대 TOPS는 한 벤더 및 아키텍처 안에서는 유용할 수 있지만 다음 조건이 모두 맞지 않는다면 벤더 간 비교는 대체로 유효하지 않습니다.

- INT8, INT4, FP16, 혼합 모드 등 수치 정밀도
- 곱셈 누산 한 번을 연산 하나로 셌는지 둘로 셌는지
- 밀집 연산과 구조적 희소 연산 구분
- 입력 해상도, 배치 크기, 모델 그래프
- 양자화 후 정확도
- NPU 단독 지연 시간과 애플리케이션 전체 파이프라인 지연 시간
- 메모리 전송과 호스트 폴백
- 열 상태와 순간 성능이 아닌 지속 성능

탐지기는 파이프라인입니다. 이미지 수집, 크기 조정/레터박스, 정규화, 장치 전송, 신경망 추론, 디코드, NMS, 좌표 변환, 애플리케이션 로직이 포함됩니다. 벤더는 흔히 그중 신경망 추론 부분만 공개합니다. [MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/)는 홍보용 연산량을 따로 비교하지 않고 시나리오, 품질 기준, 시스템 수준 측정 규칙을 정의하므로 유용합니다.

YOLO 배포에서 신뢰할 수 있는 최소 벤치마크 기록은 다음과 같습니다.

```text
exact checkpoint + checksum
source and compiled model format
compiler, runtime, driver and firmware versions
target board and clock/power mode
input resolution and batch
precision and calibration dataset
accuracy before and after compilation
warmup and sample count
preprocess, inference and postprocess latency
power measurement boundary
```

이 항목이 없으면 FPS 두 수치는 대개 서로 다른 것을 측정합니다.

## 컴퓨터 비전용 NPU 선택

제품 페이지에서 가장 큰 숫자를 고르지 말고 다음 순서로 선택하십시오.

1. **그래프 호환성을 입증합니다.** 이름이 비슷한 모델 동물원 체크포인트가 아니라 내보낸 정확한 모델을 컴파일합니다.
2. **정확도를 측정합니다.** 보정 및 양자화 후 실제 작업 데이터셋에서 컴파일 산출물을 검증합니다.
3. **종단 간 성능을 측정합니다.** 입력 변환, 전송, 디코드, NMS를 포함합니다.
4. **소프트웨어 접근성을 확인합니다.** 컴파일러가 공개인지, 등록이 필요한지, 상용 계약 후에만 쓸 수 있는지 파악합니다.
5. **버전 조합을 고정합니다.** 컴파일러, 런타임, 드라이버, 펌웨어, 대상 식별자를 기록합니다.
6. **운영체제 지원의 실제 범위를 확인합니다.** Android, Debian, Yocto, Buildroot, RTOS, Windows 지원은 서로 대체되지 않습니다.
7. **공급과 수명 주기를 확인합니다.** 모듈, 드라이버, 장기 지원이 불확실하면 기술적으로 뛰어난 칩도 양산에 부적합할 수 있습니다.
8. **그다음 비용, 전력, 성능을 비교합니다.** 두 대상에서 같은 모델이 올바르게 동작한 뒤에야 측정값을 비교할 수 있습니다.

### 사용 사례별 실용적인 시작점

| 사용 사례 | 우선 평가할 플랫폼 | 이유 |
|---|---|---|
| Raspberry Pi 전용 가속기 | Hailo-8L/8, Sony IMX500 | 공식 Raspberry Pi 제품, 소프트웨어 이미지, 구체적인 개발 흐름 |
| 범용 Linux PCIe, M.2, USB 추가 장치 | DEEPX, MemryX, Hailo, Axelera 제품 | 구할 수 있는 가속기 폼팩터. 호스트/드라이버 호환성 확인 필요 |
| 저가형 Linux SBC 또는 카메라 | Rockchip RK3588/RK3576, 지원 보드/BSP를 구할 수 있는 Amlogic ADLA, AXERA, Canaan K230, Sunplus SP7350 | 통합 미디어 파이프라인과 공개 모델/컴파일러 근거. A311D2 NPU에는 VIM4 V13A 이상인지 확인 |
| 모바일 및 대량 생산 Android | Qualcomm QNN/AI Hub, LiteRT, Apple Core ML | 대규모 배포 기반과 유지보수되는 모바일 런타임 |
| 산업용 Linux 및 로보틱스 | TI TDA4/AM6xA, NXP i.MX, Renesas RZ/V, Hailo | 장기 수명 임베디드 제품과 카메라/I/O 생태계 |
| 초소형 엣지 또는 MCU | STM32N6x7, Infineon PSOC Edge, Himax WiseEye2, Analog Devices MAX7800x, Nuvoton M55M1, Alif/Arm Ethos-U, GAP9, Syntiant | 전력과 메모리 제약에 맞춘 도구, 모델 크기 한계 |
| 구성형 FPGA 비전 | Microchip VectorBlox, Lattice sensAI, AMD Vitis AI 대상 | 유연한 하드웨어 파이프라인. 가속기 구성, 비트스트림, 모델 컴파일러를 버전 관리해야 함 |
| 고처리량 PCIe 비전 | Axelera Metis, Hailo, DEEPX, Mobilint, MemryX, Rebellions, SiMa.ai | 전용 가속기 제품 및 다중 스트림 지향 |
| 중국 중심 제품 생태계 | Rockchip, AXERA, D-Robotics, SOPHGO, Huawei Ascend, HiSilicon, Cambricon | 강력한 지역 보드, 툴체인, 모델 예제 |

이 표는 조사 기반 후보 목록이지 성능 순위가 아닙니다. 조달, 지역별 공급, 정확한 모델에 따라 순서가 바뀔 수 있습니다.

## LibreYOLO에 주는 의미

확장 가능한 설계는 서로 무관한 내보내기 도구 수십 개가 아닙니다. 엄격한 교환 계약 하나와 소규모 벤더 컴파일러/런타임 어댑터가 필요합니다.

```text
LibreYOLO model
    -> deterministic static ONNX
    -> VendorCompiler.compile(model, target, calibration, precision)
    -> native artifact + manifest
    -> VendorRuntime.load() / infer()
    -> shared task-specific decode and Results objects
```

모든 네이티브 산출물에 다음 내용을 담은 보조 매니페스트가 있어야 합니다.

- 벤더, 칩 대상, 보드 대상
- 컴파일러, 런타임, 드라이버, 펌웨어 버전
- 원본 체크포인트 및 ONNX 체크섬
- 입력 이름, 모양, 레이아웃, 색 공간, 정규화, dtype
- 양자화 정밀도, 공개된 경우 스케일, 보정 데이터 해시
- 출력 텐서 이름, 모양, 의미
- 탐지 작업, 클래스 이름, 디코드/NMS가 칩 또는 호스트에서 실행되는지 여부
- 기록된 수치적 동등성 및 작업 정확도 결과

LibreYOLO는 이미 이식 가능한 [ONNX 내보내기](/docs/export/onnx), [양자화 도구](/docs/export/quantization), 의도적으로 제한한 직접 [RKNN 내보내기](/docs/export/rknn), 정직한 외부 컴파일러 [Hailo 절차](/docs/export/hailo)를 제공합니다. 이 가이드 기준으로 Amlogic ADLA는 강력한 다음 통합 후보입니다. 툴킷이 공개되어 있고 모델 범위가 LibreYOLO와 많이 겹치며 구형 A311D 경로를 명확히 분리할 수 있습니다.

Amlogic 다음 우선순위는 컴파일러 접근성만이 아니라 사용자의 하드웨어와 정확도 검증 가능성을 기준으로 정해야 합니다. DEEPX, Sony IMX500, D-Robotics, AXERA, SOPHGO, Qualcomm은 기술적으로 매력적입니다. TI, NXP, ST, Renesas는 산업 파트너가 보드와 장기 CI 접근성을 제공할 수 있을 때 특히 가치가 높습니다.

## 주시 목록: 공개 통합 계약이 없는 실제 실리콘

기업을 완전히 제외하면 시장 지도가 오해를 부를 수 있습니다. 반대로 "지원"으로 포함하면 더 문제가 될 수 있습니다. 이 업체들은 관련 실리콘을 판매, 샘플 제공, 발표했지만 공개 근거만으로 LibreYOLO 백엔드에 필요한 재현 가능한 최신 컴파일러, 산출물, 모델 경로를 입증하지 못했습니다.

| 기업 | 확인된 사실 | 주시 목록에 남는 이유 |
|---|---|---|
| [Samsung](https://semiconductor.samsung.com/processor/automotive-processor/exynos-auto-v920/) | Exynos Auto V920은 최대 23.1 TOPS를 광고하는 듀얼 코어 NPU를 포함 | Samsung은 [Neural SDK를 타사 개발자에게 더 이상 제공하지 않는다](https://developer.samsung.com/neural/overview.html)고 밝혔습니다. Samsung ONE/Circle만으로 Exynos NPU 접근이 입증되지 않음 |
| [Novatek](https://www.novatek.com.tw/en-global/Milestone/aboutus_milestones) | 최신 기업 연혁에 Edge AI 및 Edge Vision/Imaging AI SoC가 등장하며, 2020년 기록에는 MobileNet, SSD, YOLOv3 언급 | 최신 부품 번호/컴파일러/산출물/연산자/모델 표가 공개되지 않음. 과거 YOLOv3는 현재 SDK 근거가 아님 |
| [Nextchip](https://www.nextchip.com/en/adas/adas.php?idx=5) | aiMotive aiWare NPU가 있는 APACHE5/NVS2900, 최신 [APACHE6/NVS3000](https://www.nextchip.com/en/adas/adas.php?idx=7) 자동차 프로세서. APACHE6 페이지 자체가 현재 12와 8 TOPS를 모두 표시 | aiWare Studio와 런타임은 있지만 공개 산출물, 양자화 가이드, 연산자 목록, YOLO 표를 찾지 못함. 상충하는 벤더 수치를 임의로 조정하면 안 됨 |
| [Black Sesame Technologies](https://bst.ai/en.html) | Huashan A1000/A2000, Wudang C 시리즈 자동차/엣지 AI 실리콘 | 공개 제품 정보는 있으나 셀프 서비스 컴파일러, 산출물 계약, 재현 가능한 모델 동물원은 없음 |
| [Ingenic](https://en.ingenic.com.cn/products-detail/id-19.html) | 저비트 NPU 주장이 있는 T41/T40 카메라 SoC. Magik AI는 PTQ/QAT 및 그래프 컴파일을 설명 | 현재 컴파일러 다운로드, 산출물 계약, 벤더 YOLO 목록을 찾지 못함 |
| [Fullhan](https://fullhan.com/en/index.php?a=type&c=article&tid=9) | 여러 IPC SoC가 0.5~2 TOPS NPU 광고, [MC6880 NVR 계열](https://fullhan.com/en/index.php?a=type&c=article&tid=48)은 4 TOPS 광고 | 독립 재현에 필요한 공개 NN 컴파일러/런타임/프레임워크/모델 문서 없음 |
| [Goke Microelectronics](http://www.gokemicro.com/News/info.aspx?itemid=437) | GK7606V1/GK7206V1/GK7203V1 스마트 카메라 부품의 1차 자료에서 통합 NPU 성능을 광고하지만 게시 시점에 구형 HTTP로만 제공 | 공개 변환기, 런타임 계약, 모델 이름 표가 없음. OEM 채널 지향 |
| [Chengheng Micro](https://en.chenghengmicro.com/) | CH37은 64 INT8 TOPS와 FP16/FP32/FP64 모드를 광고. 회사의 [2026년 연혁](https://chenghengmicro.com/about.html)은 소량 생산 시작을 기록 | 공개 SDK, 컴파일러 계약, 모델 검증 이름을 찾지 못함. 재현 가능한 배포 근거가 아니라 초기 상용/비 셀프 서비스 플랫폼으로 취급 |
| [Bouffalo Lab](https://github.com/bouffalolab/bouffalo_sdk) | BL808에는 BLAI-100 NPU가 있으며 최신 공식 SDK 저장소도 존재 | 유지보수되는 SDK에 최신 공식 BLAI 변환/예제 경로가 없음. 남은 흐름은 레거시 또는 커뮤니티 근거 |

이 주시 목록은 근거 중심으로 구성했습니다. 업체가 최신 툴체인 또는 평가 경로, 대상별 산출물 계약, 종단 간 절차가 있는 모델 하나 이상을 공개하면 배포 대상 표로 옮길 수 있습니다.

## 의도적으로 주장하지 않은 내용

이 글은 이름이 나온 모든 모델이 모든 업스트림 저장소에서 작동한다고 주장하지 않습니다. 벤더 모델 동물원의 모델은 수정되거나 출력 노드가 다르게 분할되거나 사용자 지정 후처리와 짝을 이루는 경우가 많습니다.

다음 사실도 지원 주장으로 바꾸지 않습니다.

- 컴파일러가 ONNX를 가져옴.
- 칩이 Android NNAPI 드라이버를 광고함.
- 유통업체가 보드가 "YOLO 호환"이라고 말함.
- 커뮤니티 저장소에서 한 모델 변형을 실행함.
- 오류 없이 모델이 컴파일됨.
- 벤더가 정확도 결과 없이 NPU 단독 FPS를 보고함.

Google Tensor 휴대전화 가속기와 다수 OEM 전용 맞춤 ASIC도 실제 제품이지만 Google은 Tensor를 위 플랫폼과 비슷한 범용 셀프 서비스 컴파일 대상으로 공개하지 않습니다. 기업이 존재하거나 NPU 블록 다이어그램이 있거나 Android 가속이 된다는 사실만으로 LibreYOLO 통합 주장을 만들 수는 없습니다.

Google TPU, AWS Inferentia/Trainium, Microsoft Maia, 데이터센터 전용 AI 카드 같은 클라우드 가속기도 범위 밖입니다. 이 가이드는 컴퓨터 비전 개발자가 엣지에서 실제 배포할 수 있는 하드웨어를 다룹니다.

모델 라이선스는 별도 계층입니다. 칩 벤더가 YOLO 변환 레시피를 게시해도 업스트림 가중치, 학습 코드, 컴파일된 파생물의 사용 또는 재배포 권리를 부여하지는 않습니다. 하드웨어 지원, SDK 라이선스, 모델 라이선스를 각각 실제 제품 기준으로 확인해야 합니다.

## 조사 방법 및 정정

각 기업에 대해 다음 네 가지 1차 출처를 조사했습니다.

1. 실리콘을 식별하는 최신 제품 페이지 또는 데이터시트.
2. 실제 배포 절차를 설명하는 컴파일러/런타임 문서.
3. 비전 모델 이름을 명시하는 모델 동물원, 호환성 표, 종단 간 튜토리얼.
4. 개발자가 도구를 실제로 얻을 수 있는지 보여 주는 제품 수명 주기/접근성 신호.

자사 GitHub 조직은 벤더 자료로 간주합니다. 칩 회사가 같은 자료를 공개하지 않는 경우 보드 업체 문서는 생태계 근거라고 표시했습니다. 제3자 성능 주장은 비교 표에서 제외했습니다.

완성된 초안은 별도 벤더 그룹 점검과 표 간 교차 점검을 거쳤습니다. 이 점검에서 대상 범위, 산출물 이름, 정밀도, 모델과 후처리기 근거, 발표와 출시 상태, 수명 주기 표기, 벤치마크 분모, 모든 기업 수를 다시 확인했습니다. 1차 자료끼리 충돌하면 더 편한 수치 하나를 고르지 않고 충돌을 드러냈습니다.

이 시장은 빠르게 바뀝니다. 관련 기업에서 칩, SDK, 모델 목록, 접근성 상태가 잘못되었다면 정확한 공개 문서 URL과 버전을 LibreYOLO에 보내 주십시오. 1차 근거로 뒷받침된 정정은 이 글을 대신해야 하며 문서화되지 않은 마케팅 주장은 그렇지 않습니다.

## FAQ

### 엣지 AI NPU 기업은 몇 곳입니까?

제품 공급업체, IP 라이선스 업체, 스마트 센서, GPU, 비공개 자동차 ASIC의 범위가 겹치므로 보편적인 기업 수는 없습니다. 이 비망라형 가이드는 2026년 8월 기준으로 관련 엣지 AI 칩, 가속기 플랫폼, 라이선스 가능한 NPU IP 또는 근거가 있는 주시 대상 실리콘을 보유한 기업과 플랫폼 생태계 69곳을 추적합니다.

### NPU란 무엇입니까?

신경망 처리 장치(neural processing unit)는 합성곱과 행렬 곱셈 같은 신경망 연산에 특화된 하드웨어입니다. 업체마다 NPU, AIPU, BPU, KPU, DLA, HTP, TPU, MLA 등의 용어를 사용하며 컴파일된 모델은 대체로 업체 간에 이식할 수 없습니다.

### YOLO 모델을 공식 지원하는 NPU 기업은 어디입니까?

Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Raspberry Pi의 공식 AI Camera 저장소를 통한 Sony IMX500, STMicroelectronics, Renesas, Lattice, Himax, Microchip 등은 적어도 한 YOLO 세대에 대해 자사 모델 동물원 항목, 튜토리얼 또는 검증 결과를 제공합니다. 정확한 세대, 작업, 칩 대상, SDK 버전은 여전히 확인해야 합니다.

### 아무 ONNX 모델이나 모든 NPU에서 실행할 수 있습니까?

아닙니다. ONNX는 교환 형식이지 하드웨어 호환성을 보장하지 않습니다. NPU 컴파일러는 그래프의 모든 연산자, 텐서 모양, 데이터 형식을 지원해야 합니다. 정적 모양, 그래프 분할, 사용자 지정 연산, CPU 폴백이 흔합니다.

### YOLO에 가장 적합한 NPU는 무엇입니까?

모든 경우에 앞서는 제품은 없습니다. Hailo, Rockchip, Axelera AI, DEEPX, Amlogic은 문서로 확인되는 YOLO 지원 범위가 넓고, Qualcomm은 특히 다양한 기기에서 사용할 수 있습니다. 양자화 후 정확도, 전체 파이프라인 지연 시간, 지속 전력, 가격, 공급 상황, SDK 접근성, 운영체제 지원을 기준으로 선택하십시오.

### NPU TOPS 수치를 직접 비교할 수 없는 이유는 무엇입니까?

업체마다 정밀도, 희소 연산, 곱셈 누산, 최대 활용률 가정을 다르게 계산할 수 있습니다. TOPS에는 메모리 트래픽, 미지원 연산자, 전처리, 후처리, 호스트 오버헤드가 반영되지 않습니다. 같은 모델, 정밀도, 해상도, 정확도, 전체 파이프라인을 비교해야 합니다.

### NPU 보정이란 무엇입니까?

보정은 보통 레이블이 없는 대표 배포 이미지를 모델에 입력해 컴파일러가 INT8 또는 다른 저정밀도 형식의 활성화 범위를 추정하게 하는 과정입니다. 무작위이거나 대표성이 없는 이미지로도 컴파일 산출물이 만들어질 수 있지만 작업 정확도는 낮아질 수 있습니다.

### LibreYOLO는 엣지 NPU를 지원합니까?

LibreYOLO는 ONNX와 여러 런타임 형식으로 내보내며, 일부 Rockchip 모델에는 직접 RKNN 컴파일 경로를 제공하고 외부 Hailo 컴파일 절차도 문서화합니다. 그 밖의 모든 벤더 네이티브 산출물에는 해당 벤더 SDK가 필요합니다. 하드웨어에서 컴파일, 검증, 실행하기 전까지는 통합 후보라고 설명해야 합니다.
