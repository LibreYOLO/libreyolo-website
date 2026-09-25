---
title: "2026 年 69 家 Edge AI NPU 厂商：芯片、SDK 与 YOLO 部署"
description: "基于一手资料梳理 69 家边缘 AI 厂商及 NPU 生态，涵盖芯片、SDK、部署产物、量化，以及有文档记录的计算机视觉和 YOLO 支持。"
date: 2026-08-15
author: Xuban
tags: [edge-ai, npu, computer-vision, yolo, hardware, ai-accelerators, amlogic, hailo, rockchip]
faq:
  - q: "有多少家 Edge AI NPU 公司？"
    a: "没有统一的公司总数，因为产品厂商、IP 授权商、智能传感器、GPU 和私有汽车 ASIC 之间存在重叠。本指南并非穷尽性清单，截至 2026 年 8 月，共跟踪了 69 家拥有相关边缘 AI 芯片、加速平台、可授权 NPU IP，或具有可信候选芯片的公司及平台生态。"
  - q: "什么是 NPU？"
    a: "神经网络处理器（NPU）是用于卷积、矩阵乘法等神经网络运算的专用硬件。厂商使用的名称包括 NPU、AIPU、BPU、KPU、DLA、HTP、TPU 和 MLA；编译后的模型通常无法在不同厂商之间移植。"
  - q: "哪些 NPU 公司官方支持 YOLO 模型？"
    a: "Amlogic、Hailo、Axelera AI、Rockchip、高通、MediaTek、DEEPX、D-Robotics、AXERA、SOPHGO、Huawei、Cambricon、通过树莓派官方 AI Camera 仓库提供支持的 Sony IMX500、STMicroelectronics、Renesas、Lattice、Himax、Microchip 等厂商，都有至少一个 YOLO 代际的一手模型库条目、教程或验证结果。具体代际、任务、芯片目标和 SDK 版本仍然很重要。"
  - q: "任何 ONNX 模型都能在任何 NPU 上运行吗？"
    a: "不能。ONNX 是模型交换格式，不是硬件兼容性保证。NPU 编译器必须支持计算图中的每个算子、张量形状和数据类型。静态形状、计算图切分、自定义算子和 CPU 回退都很常见。"
  - q: "哪款 NPU 最适合 YOLO？"
    a: "没有适用于所有场景的赢家。Hailo、Rockchip、Axelera AI、DEEPX 和 Amlogic 的 YOLO 文档覆盖较充分，而高通的设备覆盖面很广。选择时应考虑量化后的精度、完整流水线延迟、持续功耗、价格、供货情况、SDK 获取方式和操作系统支持。"
  - q: "为什么 NPU 的 TOPS 数字不能直接比较？"
    a: "厂商统计时可能采用不同精度、稀疏运算、乘加计数方式和峰值利用率假设。TOPS 不包含内存流量、不支持的算子、预处理、后处理和主机开销。应在相同模型、精度、分辨率、精度指标和完整流水线上进行比较。"
  - q: "什么是 NPU 校准？"
    a: "校准是将有代表性、通常没有标签的部署图像输入模型，让编译器估算 INT8 或其他低精度格式的激活值范围。随机或缺乏代表性的图像仍可能生成编译产物，却会损害任务精度。"
  - q: "LibreYOLO 支持边缘 NPU 吗？"
    a: "LibreYOLO 可导出 ONNX 和多种运行时格式；部分 Rockchip 模型可直接通过 RKNN 编译器路径处理，并提供外部 Hailo 编译流程的文档。其他厂商原生格式仍需其厂商 SDK；在完成编译、验证并在硬件上运行之前，应将它们描述为集成候选方案。"
---

**NPU 市场并非单一市场。本指南并非穷尽性清单，共跟踪 69 家公司及平台生态：51 家提供可部署芯片或平台的厂商、9 家 NPU IP 供应商，以及 9 家明确标注的候选厂商。** 它们横跨多个彼此不兼容的加速器类别，编译器技术栈也几乎各不相同。大多数厂商承诺的流程都类似：导出模型、量化、编译、把专有产物复制到开发板，再调用厂商运行时。具体细节决定这条流程耗时是一个下午还是一个工程季度。

本指南梳理了 2026 年提供可信计算机视觉硬件的公司，记录相关芯片、SDK 和编译器名称、部署产物、公开程度，以及厂商实际记录的模型。本文也特别介绍 Amlogic：其较新的 ADLA 技术栈与较早的 A311D NPU 生态有实质区别。

> **研究范围：**资料核查截至 2026 年 8 月 15 日。除非另有标注，具名模型的说法均来自一手产品页面、文档、模型库、仓库或厂商教程。宣传的 TOPS 是厂商数据，不能视作可独立比较的基准测试结果。这是一份开发者指南，不构成投资建议，也不是付费排名。

**快速导航：**[公司表格](#npu-companies-and-platforms-at-a-glance) | [Amlogic 深入介绍](#amlogic-two-npu-generations-not-one) | [厂商简介](#the-strongest-public-deployment-ecosystems) | [编译产物表](#what-you-actually-deploy-the-artifact-lock-in-table) | [工具获取与生命周期](#tool-access-and-lifecycle-signals) | [LibreYOLO 路线图](#what-this-means-for-libreyolo)

## 最重要的发现

硬件市场高度碎片化，但部署模式却出奇一致：

```text
PyTorch checkpoint
    -> ONNX or TFLite interchange graph
    -> representative calibration data
    -> vendor quantizer and graph compiler
    -> chip-specific binary or model package
    -> vendor runtime on the target
    -> application preprocessing, decode, NMS and rendering
```

[ONNX 明确支持运行时、代码生成器和硬件实现](https://onnx.ai/onnx/repo-docs/IR.html)，但 ONNX 文件只是交接格式。这不代表每个 ONNX 算子都能被每种加速器支持并下沉执行。静态输入形状、算子支持限制、量化规则和主机回退共同决定了哪些计算真正运行在 NPU 上。

因此，软件栈也是芯片的一部分。标称速度很快但编译器受限或脆弱的 NPU，可能不如工具链公开、拥有模型库和模拟器、运行时稳定的小型设备适合作为部署目标。

## 本指南中「支持」的含义

厂商资料对「支持」一词的使用很宽泛。本文区分四种证据等级：

| 证据等级 | 能证明什么 | 不能证明什么 |
|---|---|---|
| **已验证模型** | 厂商发布了特定芯片的模型条目、结果或兼容性表 | 你修改后的检查点仍能保持相同精度 |
| **官方示例** | 厂商提供了针对具名模型的端到端教程或演示 | 其他尺寸、任务或代际也能编译 |
| **编译器能力** | SDK 可导入某个框架或公开受支持算子 | 某个具体 YOLO 计算图可以端到端运行 |
| **营销宣传或受限获取** | 产品面向视觉任务，且存在私有 SDK | 模型兼容性可公开复现 |

这个区别很重要。「可以导入 ONNX」不等于「支持 YOLO11 分割」。模型可能可以解析，却回退到 CPU；也可能编译后输出数值不正确、超出加速器内存，或在量化过程中损失精度。

## NPU 公司与平台一览

下表有意纳入 SoC、独立加速器、智能传感器，以及相邻的 GPU/FPGA 目标。即使厂商使用 NPU、AIPU、BPU、KPU、DLA、HTP、TPU 或 MLA 等不同名称，它们仍会参与同一个部署选型。

### 嵌入式 SoC、智能传感器和工业处理器

| 公司 | 相关芯片或平台 | SDK、编译器和产物 | 有公开文档记录的计算机视觉证据 | 获取方式 |
|---|---|---|---|---|
| [Amlogic](https://github.com/Amlogic-NN/amlnn-toolkit) | A311D2、S928X、S905X5/S905D5、A311Y3、C308L/C302X/C302X2、T968D4、C305X2 和 A123X | AMLNN Toolkit 和 `libnnsdk.so`，编译产物为 `.adla`；A311D 另有旧版 Acuity `.nb` 技术栈 | [模型 playground](https://github.com/Amlogic-NN/amlnn-model-playground) 记录了 A311D2、S905X5、A311Y3、C305X2 和 A123X 上的 YOLOv5/6/7/8/10/11、YOLOX、YOLOE、YOLO-World、PP-YOLOE、分割、姿态和 OBB；此处列出的其他芯片是编译器目标，并未列入该支持矩阵 | GitHub 和 wheel 公开；仍需兼容的 BSP/驱动 |
| [Rockchip](https://github.com/airockchip/rknn-toolkit2) | RK3562/3566/3568、RK3576、RK3588、RV1126B | RKNN-Toolkit2、RKNN Runtime、`.rknn` | [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) 收录 YOLOv5/6/7/8/10/11、YOLOX、YOLO-World、PP-YOLOE、分割、姿态和 OBB | GitHub 公开；提供二进制 wheel |
| [Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | Snapdragon 和 Dragonwing 平台、QCS6490、QCS8550 和 Dragonwing IQ-9075 | QAIRT/QNN、SNPE 和 AI Hub；QNN context 二进制文件或 DLC | [AI Hub 模型集合](https://github.com/qualcomm/ai-hub-models) 收录 YOLOv3/5/6/7/8/9/10/11/26、YOLOX、YOLO-World、YOLOR、RF-DETR，以及检测/分割/姿态模型 | 文档公开；SDK 和账号要求因情况而异 |
| [Texas Instruments](https://github.com/TexasInstruments/edgeai-tidl-tools) | AM62A、AM67A、AM68A、AM69A 和 TDA4x；预览版 TDA54-Q1 | Processor SDK Edge AI 和 TIDL | [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo) 发布了适用于当前 AM6xA/TDA4 路径的优化检测、分割、姿态和分类模型；目前还没有针对预览版 TDA54-Q1 的目标矩阵 | GitHub 公开，并提供 Processor SDK |
| [NXP](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | i.MX 8M Plus、i.MX 93、i.MX 95，以及收购而来的 Kinara Ara-1/Ara240 | eIQ，含 TIM-VX、Vela、Neutron Converter 和 Ara SDK | [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) 包含 YOLOv4-tiny/v8、NanoDet、CenterNet、FastestDet、SSD Lite 和 YOLACT 的产物或配方；YOLOv5 条目仅有文档 | 部分公开，部分需账号 |
| [STMicroelectronics](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) | STM32N6x7 系列，包括带 Neural-ART 加速器的 STM32N657/647 | STM32Cube AI Studio 和 ST Edge AI Core | 当前服务仓库列有 Tiny YOLOv2、YOLOv5u、YOLOv8、YOLO11、YOLO26 和 ST-YOLOX，以及姿态和分割 | 工具和仓库公开 |
| [Infineon](https://documentation.infineon.com/psocedge/) | PSOC Edge E83/E84，采用 Cortex-M55 加 Ethos-U55；M33 侧另有 NNLite 加速器 | ModusToolbox、[DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter)、TFLite Micro 和 Arm Vela | 架构资料将 MobileNetV1/V2 列为代表性内核，E84 AI Kit 配有摄像头，但未找到一手 PSOC Edge YOLO 验证矩阵 | 文档和 SDK 组件公开，当前 E84 评估套件可获取 |
| [Renesas](https://github.com/renesas-rz/rzv_drp-ai_tvm) | RZ/V2L、V2M、V2MA、V2H 和 V2N | DRP-AI Translator、DRP-AI TVM 和 RUHMI | [RZ/V2H 模型列表](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) 验证了 YOLOv5/v8/v11/26 多种尺寸、YOLOX，以及姿态和分割变体 | GitHub 公开，并提供开发板 SDK |
| [Sony](https://www.aitrios.sony-semicon.com/edge-ai-devices/imx500) | IMX500 智能视觉传感器和 Raspberry Pi AI Camera | Edge-MDT、Model Compression Toolkit 和 IMX500 packer；`.rpk` 包 | [Raspberry Pi IMX500 模型库](https://github.com/raspberrypi/imx500-models) 收录 YOLOv8n、YOLO11n、EfficientDet Lite、NanoDet+ 和 SSD | 树莓派流程公开；Sony 工具条款另行适用 |
| [Ambarella](https://www.ambarella.com/developer/model-garden/) | CV72/CV75、CV5/CV52 和 CV3-AD 系列 | Cooper Developer Platform、CVflow 编译器和运行时 DAG | 公开模型库列有 YOLOX-S、RTMDet-nano、DeepLabV3+、TopFormer、OWL-ViT 和 LLaVA OneVision | 主要需合作伙伴资格 |
| [Synaptics](https://developer.synaptics.com/docs/sl/overview) | Astra SL1600/SL1680；SL2611/13/15/17/19；另有 SR100 MCU 系列 | SL16xx 使用 SyNAP `.synap`；SL261x 使用 Torq/IREE `.vmfb`；SR SDK 面向 Ethos-U55 | 官方 YOLOv8n/v8s SyNAP 基准测试，以及 SL261x Torq YOLOv8 示例；SR 证据需单独评估 | 开发者门户；部分下载受限 |
| [MediaTek](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) | Genio 360/360P/420/520/720 使用 NP8 和 MDLA 5.3；Genio 510/700/1200 使用旧版 NP6/MDLA | NeuroPilot Converter、`ncc-tflite`、Neuron Runtime 和 `.dla`；部分较新系统支持 ONNX Runtime 路径 | 官方 IoT AI Hub 发布 YOLOv5、YOLOv8 模型页面和基准测试，以及分类和人脸模型 | Yocto 文档公开；关键 NP8 工具包和 Android 资料需直客/NDA 权限 |
| [Allwinner](https://docs.aw-ol.com/v853/en/npu/dev_npu/) | 带 1-TOPS Vivante NPU 的 V853 视觉 SoC | Acuity/Pegasus 转换和 `viplite` 运行时 | 官方 V853 资料列有 YOLOv2/3/4/4-tiny/5/5s、RetinaNet、MobileNet、ResNet 以及人脸/人员网络 | 文档公开；SDK 包可能需要账号 |
| [Canaan/Kendryte](https://github.com/kendryte/nncase) | 当前 K230/K230D；旧款 K210/K510 KPU | `nncase` 编译器和运行时；`.kmodel` | K230 nncase 指南讲解 YOLOv5s 编译、模拟和运行；官方演示目录及当前 [CanMV 更新日志](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md)记录了 YOLOv8/11/26 任务 | 编译器/PyPI 公开；芯片插件是二进制文件 |
| [Alif Semiconductor](https://alifsemi.com/support/kits/ensemble-e7appkit/) | Ensemble E7 含两个 ML 加速器，其中有 Ethos-U55；[E8](https://alifsemi.com/ensemble-e8-series/) 含一个 Ethos-U85 和两个 Ethos-U55 NPU | Arm Vela 和基于 TFLite Micro 的部署 | Alif 发布了系列级 INT8 YOLO-Fastest 人脸检测基准，但没有覆盖多个现代 YOLO 的芯片级矩阵 | 文档和评估套件资源公开 |
| [Himax](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) | HX6538 WiseEye2 端侧 AI MCU，采用 Cortex-M55 和 Ethos-U55 | TFLite Micro 加 Arm Vela，提供 CMSIS-NN 和参考内核回退 | 官方 [WiseEye2 示例](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2)包含 YOLOv8n 检测、姿态和分类，YOLO11n 检测、人脸网格和 PeopleNet | GitHub、文档公开，合作伙伴开发板可购买 |
| [Analog Devices](https://www.analog.com/en/products/max78002.html) | MAX78000 和 MAX78002 AI MCU，集成低功耗 CNN 加速器 | [`ai8x-training`](https://github.com/analogdevicesinc/ai8x-training)、`ai8x-synthesis`/`izer` 和 MSDK；生成 C 代码和权重 | 官方资料涉及人脸识别/检测、RetinaNet、Visual Wake Words、分类器和动作识别；未找到一手 YOLO 部署资料 | GitHub 工具、文档和评估板公开 |
| [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo) | RDK X3/X5/Ultra 和较新的 S100 系列 BPU 开发板 | OpenExplorer/Algorithm Toolchain 和 `hbm_runtime`；X5 为 `.bin`，当前 `rdk_s` 为 `.hbm` | 当前 `rdk_x5` 分支记录了 RDK X5 上的 YOLOv5/v5u/v8/v9/v10/11/12/13/26、YOLOE、YOLO-World、分割、姿态、分类、OCR 和 CLIP；X3 与 S 系列使用各自分支 | GitHub 公开；部分工具包因平台而异 |
| [AXERA](https://github.com/AXERA-TECH/ax-samples) | AX650、AX637、AX630C、AX620Q 和 AX615 | Pulsar2 编译器和 AXEngine；`.axmodel` | 当前官方示例覆盖 YOLOv5/6/7/8/9/10/11/13/26、YOLOX 和 YOLO-World；任务及芯片目标因型号而异 | 示例和文档公开；编译器单独分发 |
| [SOPHGO](https://github.com/sophgo/tpu-mlir) | BM1684/1684X/1688/1690 和 CV186X/CV18xx | TPU-MLIR 和 SOPHON 运行时；`.bmodel` 或 `.cvimodel` | 官方演示覆盖 YOLOv3/4/5/7/8/9/10/11/12/26、YOLOX、PP-YOLOE、YOLO-World、OBB、分割、人脸、SAM 和 OCR | 开源编译器及厂商运行时 |
| [Huawei Ascend](https://www.hiascend.com/en/software/cann) | Ascend 310/310P/310B 和 Atlas 200I/300I 边缘产品 | CANN、ATC 编译器和 AscendCL；`.om` | 官方 [Ascend ModelZoo](https://github.com/Ascend/modelzoo)包含 YOLOv2/3/4/5、YOLOX、YOLOR、SSD、RetinaNet、Mask R-CNN 和分割模型 | 文档公开；CANN 包和内核必须匹配目标 |
| [Cambricon](https://github.com/Cambricon/magicmind_cloud) | 引用的公开矩阵针对 MLU370-X4/S4；MLU270 是未被该矩阵覆盖的旧硬件 | Neuware 和 MagicMind | 仓库中的 MagicMind 1.7 矩阵列有 YOLOv3/4/5/7/8、PP-YOLOE、SSD、RetinaFace、RetinaNet、Mask R-CNN 和分割模型 | 历史模型示例公开；当前 SDK/容器访问受限 |
| [Sunplus](https://sunplus.atlassian.net/wiki/spaces/C3/pages/1971126471/SP7350%2BSpecification) | SP7350/C3V，宣传值约为 4.1 至 4.6 TOPS | Vivante Acuity NPU Docker、SNNF 运行时和 `.nb` | 官方文档提供 YOLOv5 和自定义端到端 [YOLOv8 部署](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) | 公开源码/文档及开发板生态 |
| [ESWIN Computing](https://www.eswincomputing.com/en/news/info/96.html) | EIC7700/EIC7700X 和双晶粒 [EIC7702/EIC7702X](https://www.eswincomputing.com/en/news/info/104.html) RISC-V SoC | ENNP：EsQuant、当前 EsAAC 编译器、模拟器和 ESSDK 运行时；`.model`；旧文档使用 `ennc-compile` | Milk-V 托管的 ENNP 文档包含端到端 [YOLOv3 教程](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3)、MobileNetV2 和 ResNet 示例 | 一手和开发板托管文档并存；下载受地区限制 |
| [Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m55-mcus/m55m1-series/index.html) | 带 Ethos-U55-256 的 M55M1 MCU | TFLite Micro、Arm Vela 和 NuEdgeWise/NuML | 官方 [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise)示例列出 YOLOv8-nano、YOLOX-nano、YOLO Fastest、SSD-MobileNet 和分类模型 | MCU 工具链大多公开 |
| [Realtek](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | AmebaPro2 RTL8735B / AMB82-mini 摄像头平台 | Arduino/FreeRTOS SDK、VoE 和 NeuralNetwork API；`.nb` | 打包模型包括 YOLOv3-tiny、YOLOv4-tiny、YOLOv7-tiny、SCRFD 和 MobileFaceNet | 运行时公开；[自定义转换器](https://ameba-doc-arduino-sdk.readthedocs-hosted.com/en/latest/FAQ/offline_ai_model_conversion_steps.html)需联系厂商 |
| [Telechips](https://docs.topst.ai/product/p/ai) | TCC7500 / TOPST AI 开发板，宣传值 8 TOPS | TC-NN-Toolkit / Enlight SDK；`.enlight` 中间格式和编译部署包 | 一个[官方 TOPST 项目](https://docs.topst.ai/blog/31)部署了 YOLOv8s；UFLD v1/v2 转换因不支持的层而失败，文档仅提出拆分/后处理绕行方案。[YOLOv4](https://community.topst.ai/t/segmentation-fault/358)和[另一个 YOLOv8 流程](https://community.topst.ai/t/segmentation-fault-error/415)见于社区支持帖子 | 开发板文档公开；编译器/算子下载常需权限 |
| [T-Head](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | LicheePi 4A 上的 TH1520，宣传值为 4-TOPS INT8 | HHB 编译器和 CSI-NN2/SHL；`hhb.bm` 加生成代码/参数 | 官方开发板厂商示例在 NPU 上运行 YOLOv5n/s 和 MobileNetV2 | 示例公开；工具链老旧，维护情况不确定 |
| [SigmaStar](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | 当前 SSU9383CM 智能摄像头 SoC 和旧 IPU 系列 | 当前 MI_IPU 运行时；旧版 [SGS_IPU 工具链](https://wx.comake.online/doc/doc/SigmaStarDocs-SSC9381G_9351_Pudding-ULS00V040-20210913/customer/development/dla/tools.html)生成 `.sim` 到 `sgsimg.img` | [较早的官方 SDK 后处理器](https://wx.comake.online/doc/doc/Sigmastar_SDK_v1.2.2/module/User_Guide/Common/SigmaStar_Post_Processing_Module.html)列有 SSD 和 YOLOv1/2/3；这些模型和产物是否兼容 SSU9383CM 尚未验证 | 当前芯片可用，但公开引用的编译器/模型证据跨代 |
| [HiSilicon](https://www.hisilicon.com/cn/products/smart-vision/machine-vision/hi3516cv610) | Hi3516CV610/DV500、Hi3519DV500 和 Hi3403V100 智能视觉 SoC | ATC 转换为 `.om`，搭配 NNN/SVP-NNN 开发板运行时 | 特定目标的 HiSpark 矩阵，尤其针对 Hi3403 和 Hi3591P，列有 YOLOv3 至 YOLO11、姿态、分割、OBB、OCR 和深度条目；这不代表此处所有 SoC 都经过验证 | 仍在维护，且独立于 Ascend；仓库模型标记为仅限非商业用途 |

### 独立边缘加速器和模块

| 公司 | 相关芯片 | SDK 和产物 | 有公开文档记录的模型证据 | 获取方式 |
|---|---|---|---|---|
| [Hailo](https://github.com/hailo-ai/hailo_model_zoo) | Hailo-8、Hailo-8L、Hailo-10H 和 Hailo-15 系列 | Dataflow Compiler 和 HailoRT；`.hef` | YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 和 YOLO26，以及 YOLOX、DAMO-YOLO、SSD、EfficientDet、分割、姿态和 OBB；模型表按代际区分 | Model Zoo 公开；编译器通过 Developer Zone 获取 |
| [Axelera AI](https://docs.axelera.ai/sdk/reference/models/model-zoo/) | 已出货的 Metis 产品；已发布但未出货的 Europa 和 Titania 产品 | 公开的 Voyager SDK；alpha Pipeline Builder 使用 `.axm`/`.axe`，经典流程使用 `.axmodel` 包 | 已验证 YOLOv3/5/7/8/9/10/11/26、YOLOX、YOLO-NAS、OBB、姿态、分割和许多非 YOLO 模型 | SDK 在 GitHub 公开；客户支持需账号 |
| [DEEPX](https://developer.deepx.ai/modelzoo/) | DX-M1 和 DX-M1M | DXNN SDK、DX-COM 和 DX-RT；`.dxnn` | 截至 2026 年 8 月 15 日，Model Zoo 在 DX-COM 2.4.0/DX-RT 3.4.0 下列出 354 个条目，包括 YOLOv3 至 YOLO11 和 YOLO26、YOLOX、SSD、EfficientDet、NanoDet、分割、姿态和 OBB | 开发资源和套件公开 |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | MX3 加速器和多芯片模块 | MemryX SDK、Neural Compiler 和运行时；`.dfp` | 官方示例和发行说明涵盖检测、分割和姿态，包括 YOLOv10、YOLO11 和 YOLO26 | 文档和 SDK 公开 |
| [Kneron](https://doc.kneron.com/docs/) | KL520、KL530、KL630、KL720 和 KL730 有当前编译器文档；KL830 出现在部分 PLUS API 中，但不在当前编译器目标列表中 | Kneron PLUS 和 Model Toolchain；文档所列编译目标使用 `.nef` | 官方 [YOLO 流程](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/)以 Tiny-YOLOv3 为主；其他资料涉及 YOLOv5 | 文档公开；工具包/下载因芯片而异 |
| [SiMa.ai](https://docs.sima.ai/pages/palette/modelsdk.html) | MLSoC 和量产 Modalix 50-TOPS 平台 | Palette、ModelSDK、MLA Compiler 和 ModelExecutor | 公开发行资料验证 YOLOv7/v8、YOLOX、姿态/分割、DETR、Mask R-CNN 和 EfficientDet | 文档公开；产品 SDK 为商业软件 |
| [EdgeCortix](https://www.edgecortix.com/en/hardware) | SAKURA-II，宣传值为 60-TOPS，提供 M.2 和 PCIe 产品 | MERA 编译器和框架 | 厂商称支持从视觉到生成式 AI，但未发布足够具体的当前 YOLO 兼容矩阵 | 商业试用/订购咨询；验证由合作伙伴主导 |
| [BrainChip](https://brainchip.com/metatf-dev-tools/) | 已出货的 AKD1500 协处理器/M.2 模块；旧款 AKD1000 平台；Akida 2 IP | MetaTF 包：`akida-models`、`quantizeml`、`cnn2snn` 和 `akida` 运行时 | Akida 2 模型卡列出 AkidaNet0.5 YOLOv2 检测器、CenterNet、AkidaUNet 和人脸识别；这些结果不能自动视为 AKD1500 验证 | 核心 Python 包公开；硬件映射仍取决于具体目标 |
| [Blaize](https://www.blaize.com/products/) | P1600、Pathfinder 和 Xplorer 产品 | Picasso SDK、NetDeploy 和 AI Studio | 视觉是其目标市场，但没有找到可审计的公开具名模型兼容矩阵 | 商业/受限获取 |
| [Google Coral](https://github.com/google-coral/edgetpu) | 旧款 Edge TPU USB、PCIe、M.2 和 Dev Board 产品；另有独立且仍活跃的开源 [Coral NPU IP](https://github.com/google-coral/coralnpu) | 旧版 Edge TPU Compiler/`libedgetpu`/PyCoral；新 RISC-V Coral NPU 公开 IP、RTL/模拟和 ELF 示例 | 旧示例以 SSD MobileNet、分类、DeepLab 和 MoveNet 为主；新 IP 没有公开 YOLO 部署矩阵，也不是 Edge TPU 产品的直接替代品 | 旧核心仓库已归档；新 Coral NPU IP 持续开发 |
| [Lattice Semiconductor](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | ECP5、iCE40 UltraPlus、CrossLink-NX、CertusPro-NX、Avant-E 和 Avant-X FPGA | sensAI Studio、Neural Network Compiler 和可配置加速器 IP | 当前编译器表按目标模式列出 YOLOv1、YOLOv5、YOLOv8 和 YOLO11，以及 SSD、MobileNetV2-SSD、ResNet 和 ENet | 编译器/手册公开；IP 条款因产品而异，Advanced CNN Accelerator 为商业 IP |
| [Mobilint](https://www.mobilint.com/sdk-qb) | REGULUS 10-TOPS 和 ARIES MLA100 80-TOPS 加速器 | qb SDK；INT8 编译器和 `.mxq` | 公开[模型库](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md)列有 YOLOv3/5/7/8/9/10/11/12/26 检测，以及分割、姿态和 OBB 变体 | 文档/模型公开；SDK 和硬件为商业产品 |
| [Rebellions](https://rebellions.ai/developers/) | 当前 ATOM+ CA22 和 ATOM-Max CA25；已 EoL 的 ATOM CA02/ATOM+ CA12；ATOM-Lite CA21 工具链状态不明确 | RBLN SDK 编译器/运行时/分析器；`.rbln` | 官方支持公告列出 YOLOv3、YOLOv5/6/7/8 系列及当前 YOLOv8 教程 | 文档公开；编译器 wheel 需门户凭据 |
| [FuriosaAI](https://furiosa.ai/warboy) | 面向视觉的 Warboy 第一代；独立的 RNGD 系列 | Furiosa SDK；Warboy 使用 INT8 `.enf`，RNGD 使用独立 `.fxb` | Warboy 模型库记录 SSD、YOLOv5M/L 和 YOLOv7-w6-pose；RNGD 路线图标记 YOLOv8m 支持于 2024 年第四季度完成，但没有详细的当前公开视觉矩阵 | 需要 IAM/账号访问；两代产品应分开看待 |

### 开发者也会拿来与 NPU 比较的相邻平台

| 公司 | 硬件 | 部署技术栈 | 纳入比较的原因 |
|---|---|---|---|
| [NVIDIA](https://github.com/NVIDIA-AI-IOT/deepstream_tools/tree/main/yolo_deepstream) | Jetson Orin GPU 加 DLA；Jetson Thor GPU 不含 DLA | JetPack、TensorRT 和 DeepStream；`.engine` | 视觉部署生态成熟；官方 DeepStream 工具记录 YOLOv4/v7/v8/v9/11，包括 YOLO11 OBB 配置。许多结果使用 GPU，不受支持的 Orin DLA 层可能回退到 GPU。 |
| [AMD](https://vitisai.docs.amd.com/en/6.2/) | Kria/旧 DPU 目标；Vitis AI 6.2 GA 面向 Versal AI Edge Gen1 VEK280/VE2802 和 Gen2 VEK385；另有 Ryzen AI 客户端 NPU | 旧版 `.xmodel`；Gen1 绑定目标的 NPU snapshot；Gen2 编译缓存目录或生产 `.rai` 包 | Vitis AI 发布视觉资料，但旧 DPU、Versal Gen1、Versal Gen2 和 Ryzen AI 是不同的编译与部署契约。 |
| [Microchip](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox) | 带可配置 CoreVectorBlox 加速器 IP 的 PolarFire SoC FPGA | 公开的 VectorBlox SDK 3.1 和 Libero；`.vnnx`、`.hex` 和 `.ucomp` 部署产物 | 当前[教程](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md)涵盖 YOLOv5n、YOLOv8 检测/分类/OBB/姿态/分割、YOLOv9t 等；SDK 3.1 目前针对 PolarFire SoC Video Kit。 |
| [Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) | Core Ultra NPU、CPU 和集成/独立 GPU | OpenVINO IR 或编译模型缓存 | OpenVINO 提供公开的跨 XPU 路径；应查看[已验证模型矩阵](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html)的 NPU 栏，不要假设每个 OpenVINO YOLO 示例都通过了 NPU 验证。 |
| [Apple](https://developer.apple.com/machine-learning/core-ml/) | A11 及更新 A 系列芯片和 M 系列芯片中的 Apple Neural Engine | Core ML 和 `coremltools`；`.mlpackage`/`.mlmodelc` | 通过 Core ML 可轻松使用消费级 NPU，但 Apple 会抽象具体的 CPU/GPU/Neural Engine 分配，不提供面向 YOLO 的专用硬件编译器。 |
| [GreenWaves Technologies](https://github.com/GreenWaves-Technologies/nn_menu_gap9) | 带 NE16 神经引擎的 GAP9 | GAP SDK、NNTool 和 AutoTiler 生成的代码 | 官方仓库包括 MobileNet SSD、人脸检测、分类和专用 [YOLOX 行人检测器](https://github.com/GreenWaves-Technologies/yolox_people_detection)。完整 SDK 仅对合格客户开放。 |
| [Syntiant](https://www.syntiant.com/hardware) | 量产中的 NDP200 和送样中的 NDP250 Neural Decision Processor | Syntiant SDK 和部署包 | 超低功耗常开视觉/传感器处理器：资料称 NDP200 低于 1 mW，NDP250 图像识别低于 30 mW。未找到广泛公开的 YOLO 矩阵。 |

## Amlogic：两代 NPU，而非同一套技术

Amlogic 值得深入介绍，因为网络资料常将不兼容的产品混为一谈。该公司有较旧的 VeriSilicon 路径和较新的专有 ADLA 路径。两者使用不同编译器、产物和运行时。

| 代际 | 代表芯片 | NPU 和工具链 | 编译产物 | 实际情况 |
|---|---|---|---|---|
| 旧版 | A311D 和 S905D3，常见于 Khadas VIM3/VIM3L | VeriSilicon Vivante VIPNano-QI、Acuity 工具包、KSNN 和旧版 `aml_npu_sdk` | `nbg_unify` 输出目录中的 `.nb` | 已有开发板和演示；需单独集成旧版技术栈 |
| 当前 ADLA2 | C308L/C302X、S928X、A311D2、T968D4、S905X5/S905D5 和 C302X2 | Amlogic ADLA2、AMLNN Toolkit 和 NNSDK2 | 芯片目标专用 `.adla`；W8A8 或 W8A16 | 当前技术栈侧重整数精度 |
| 当前 ADLA3 | A311Y3、C305X2 和 A123X | Amlogic ADLA3、AMLNN Toolkit 和 NNSDK2 | 芯片目标专用 `.adla`；W4A8、W8A8、W4A16、W8A16 或 W16A16 | 增加原生 INT4、FP16 和 BF16 能力 |

[A311D 数据手册](https://dl.khadas.com/products/vim3/datasheet/a311d-datasheet.pdf)将原始 NPU 标为 5-TOPS INT8。[较早的 Khadas VIM3 应用页面](https://docs.khadas.com/products/sbc/vim3/npu/npu-app)和 [NPU 概览](https://docs.khadas.com/products/sbc/vim3/npu/start)记录了 DenseNet CTC、MTCNN、RetinaFace、YOLOFace、YOLOv2、YOLOv3、YOLOv3-tiny、YOLOv4、YOLOv7-tiny 和 YOLOv8n；概览还记录了 YOLOv8n-pose，并标记 FaceNet 已弃用。这些示例确实存在，但它们证明的是旧 Acuity `.nb` 生态，而非当前 ADLA 芯片。A311D 不等于 A311D2，C305X 不等于 C305X2。

还有第二个硬件陷阱。[Khadas VIM4 修订版指南](https://docs.khadas.com/products/sbc/vim4/configurations/identify-version)指出，最初的 V12/A311D2 Rev.B 开发板没有 NPU；宣传的 3.2-TOPS NPU 出现在采用 A311D2-N0D Rev.C 芯片的 V13A 及后续开发板上。仅凭产品名称不足以选定测试设备。

### 当前 AMLNN 和 ADLA 技术栈

当前 [AMLNN Toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/7d3cc9a36179b756ce79c953c14d43acab1f2af5)涵盖模型转换、量化、编译、推理和分析。其固定版本的[2026 年 5 月用户指南](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/02_Amlogic_NPU_User_Guide_V0.1.pdf)记录了 ONNX、浮点或量化 TFLite、TorchScript `.pt`，以及 PyTorch 2 ExportedProgram/PT2 导入器。尽管仓库概览用语更宽泛，详细指南仍将 TensorFlow、Paddle 和 Keras 路径标记为计划中。编译器生成 `.adla`。Python 部署使用 AMLNN 或 `amlnn_edge_toolkit_lite`；原生 C/C++ 通过 `nnsdk2.h` 链接 `libnnsdk.so`。主机端开发可通过 ADB 使用 `nnserver`；运行时还支持 Android、Buildroot、Yocto 和部分 Debian/Armbian 环境。

工具包当前公布的平台标识如下：

| 目标 ID | 平台 |
|---:|---|
| `001` | C308L / C302X |
| `002` | S928X |
| `003` | A311D2 |
| `004` | T968D4 |
| `005` | S905X5 / S905D5 |
| `006` | C302X2 |
| `007` | A311Y3 |
| `008` | 详细文档和模型矩阵中为 C305X2 / A123X |

目标字段并非装饰信息。Amlogic 要求 `.adla`、`nnsdk2.h`、`libnnsdk.so`、BSP 以及编译器/运行时代际彼此匹配。固定版本的[快速入门](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/01_Amlogic_NPU_Quick_Start_guide_V0.1.pdf)要求 ADLA 驱动 2.0.2 或更新版本、NNSDK 3.0.0 或更新版本，以及 NNSDK2 1.0.0 或更新版本；[Android 部署指南](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/06_Amlogic_NPU_Android_Deployment_Guide_V0.1.pdf)具体说明了库、头文件、驱动和 BSP 之间的关系。应将 `.adla` 视为与 ABI 绑定的构建产物，而非可移植模型。

量化也取决于 NPU 代际。ADLA2 目标 001 至 006 的文档记录了 W8A8 和 W8A16 编译。ADLA3 目标 007 和 008 增加了 W4A8、W4A16、W8A8、W8A16 和 W16A16 组合，并在[算子指南](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/05_Amlogic_NPU_Support_Operator_List_V0.1.pdf)中记录了原生 INT4、FP16 和 BF16 支持。Amlogic 建议使用 200 至 500 张有代表性的校准样本。其随机数据选项明确用于性能测试，而非保持精度的量化。

### Amlogic 有文档记录的模型覆盖

[固定版本的 Amlogic model playground](https://github.com/Amlogic-NN/amlnn-model-playground/blob/03ca63ef20c4f1560722de26ab910826884f334a/README.md)比笼统声称支持 ONNX 更有力。其公开支持/示例矩阵列出了 A311D2、S905X5、A311Y3、C305X2 和 A123X。编译器接受其他目标 ID，并不能证明整个模型列表都在这些目标上通过验证。

| 任务 | 有文档记录的模型 |
|---|---|
| 分类 | MobileNetV2 和 ResNet50-v2；ADLA3 上的 DINO |
| 目标检测 | PP-YOLOE、YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv10、YOLO11、YOLOE、YOLO-World、YOLOX 和二维码检测 |
| 人脸和手势 | RetinaFace 和 Gesture Recognition |
| 分割 | DeepLabV3、PP-LiteSeg、YOLOv5-seg 和 YOLOv8-seg |
| 旋转目标检测 | YOLOv8 OBB |
| 姿态 | BlazePose detector/landmark 和 YOLOv8 pose |
| OCR 和语音 | LPRNet、PaddleOCR 变体、Whisper Tiny 和部分较新芯片上的 SenseVoice |
| 新一代 Transformer 和多模态任务 | DETR、MobileSAM、CLIP/MobileCLIP，以及较新平台上的部分低比特 LLM/VLM 示例 |

同一仓库还公布了以下 W8A8 模型运行时数据。这些是神经网络在 NPU 上的厂商测量结果，并非完整摄像头流水线。

| 模型和输入 | S905X5 | A311D2 | A311Y3 |
|---|---:|---:|---:|
| YOLOv8n，640 x 640 | 101.72 FPS | 95.14 FPS | 191.06 FPS |
| YOLOv8s，640 x 640 | 42.33 FPS | 42.77 FPS | 83.08 FPS |
| YOLOv8m，640 x 640 | 19.67 FPS | 19.82 FPS | 35.30 FPS |
| YOLOv8l，640 x 640 | 10.53 FPS | 10.12 FPS | 18.37 FPS |
| YOLO11n，640 x 640 | 41.14 FPS | 41.48 FPS | 62.24 FPS |

需要特别注意其中的限制。Amlogic 的算子指南指出，ADLA2 和 ADLA3 都在软件中执行 NMS。README 将这些数据定义为 NPU 原生模型运行结果，不包含预处理和后处理；是否包含所有内存传输并未披露。YOLO 精度数据使用 300 张 COCO 子集图像，而非完整验证集；分类数据则使用 ImageNet `val1000`。表中 A311Y3 的 YOLOv8n 延迟与 FPS 表给出的数据不同，仓库未说明测试条件差异。时钟频率、功耗模式、散热、具体 SDK/BSP 版本和测试时长均未披露。应使用这些数据了解单一厂商技术栈，而非与其他厂商横向排名。

### Amlogic 为什么值得关注

Amlogic 同时具备四个少见特点：嵌入式 SoC 覆盖面大、编译器技术栈刚刚公开、当前模型矩阵与 LibreYOLO 的任务范围高度重叠，而且主流训练库中原生集成较少。当前仓库也很新，分别于 2025 年底和 2026 年初公开，实质性手册发布于 2026 年 5 月至 7 月。这带来了先发机会，也意味着 API 稳定性存在风险。

一次干净的集成应使用 LibreYOLO 的确定性 ONNX 导出作为编译器输入，低精度构建要求提供有代表性的校准图像，生成 `.adla` 和元数据清单，再通过 NNSDK2 适配器加载。旧款 A311D 应使用明确区分的 `.nb` 目标，因为把 `.nb` 和 `.adla` 当作同一个后端会造成静默兼容故障。Amlogic 仓库采用 Apache-2.0 顶层许可；在 LibreYOLO 镜像其中的 wheel、共享库、`nnserver` 和 Android AAR 二进制文件之前，应直接确认再分发权利。

## 公开部署生态较完善的厂商

以下公司目前较清楚地兼具可获取的硬件、公开技术资料和具名模型证据。这并不代表它们在所有场景下都更快，而是其兼容性声明更容易在购买硬件前评估。

### Hailo

Hailo 是专用视觉加速器生态的典型代表。模型先解析为 Hailo Archive，再使用有代表性的图像进行优化和量化，最后编译为 Hailo Executable Format（`.hef`）文件。应用通过 HailoRT 加载该 HEF。

[Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)包含检测、分割、分类、姿态等任务的配方、预训练模型、后处理配置和性能数据。YOLO 覆盖包括 YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 和 YOLO26，以及 YOLOX、DAMO-YOLO、SSD 和 EfficientDet。固定版本的 [Hailo-8 目标检测表](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.19.0/docs/public_models/HAILO8/HAILO8_object_detection.rst)比笼统的产品页面更适合用作兼容性依据；[独立应用](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/README.md)则记录了可部署的 YOLO 流水线。

这里有一个重要的版本界线。Hailo-8 和 Hailo-8L 使用较旧的 Model Zoo 2.x、Dataflow Compiler 3.x 和 HailoRT 4.x 系列；Hailo-10 和 Hailo-15 使用当前 5.x 代际。即使设备都带有 Hailo 名称，配方、解析器行为和 HEF 也不能互换。

Hailo-15 还有独立的应用层。它使用 Vision Processor Software Package 和 Hailo Media Library，而不是 Hailo-8/10H 的 TAPPAS 风格主机路径。公开参考仓库 [`hailo-camera-apps`](https://github.com/hailo-ai/hailo-camera-apps)于 2026 年 5 月 3 日归档；[`hailo-apps-core`](https://github.com/hailo-ai/hailo-apps-core)则是另一个仍公开的应用框架。仓库归档不能证明产品已 EOL，但 Hailo-15 项目应通过 Developer Zone 确认当前的应用包。

LibreYOLO 的 [Hailo 部署指南](/docs/export/hailo)有意只到静态 ONNX 交接为止，因为专有编译器不能作为 Python 依赖打包。这是生成适用计算图与声称已测试 HEF 之间应保持的界线。

### Rockchip

Rockchip 的嵌入式 Linux 开发板在爱好者和商业项目中都很常见，尤其是 RK3588、RK3576 和 RK356x。公开的 [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)可在 x86 Linux 主机上转换和量化模型，再由开发板上的 RKNN Runtime 或 Toolkit-Lite 执行生成的 `.rknn` 文件。

[RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)对芯片、模型系列和精度的说明相当具体。当前表格列出 YOLOv5/6/7/8/10/11、YOLOX、PP-YOLOE、YOLO-World 和 YOLOv5/v8 分割变体的 FP16、INT8 路径；YOLOv8 OBB 和 YOLOv8 pose 仅列出 INT8。它还涵盖 OCR、人脸和其他分割网络。

LibreYOLO 已提供保守的直接 [RKNN 导出器](/docs/export/rknn)。它在 RK3588 上验证了四种确切的检测变体，可比较编译器模拟器和 ONNX Runtime，并拒绝未验证的模型系列，不会把成功构建等同于预测正确。这种范围明确的支持声明可作为未来 NPU 后端的参考。

### Axelera AI

Axelera AI 常被误拼为「Accelera」，其产品是 Metis AIPU，提供 M.2、PCIe 和多芯片产品。Metis 是目前可公开订购的系列；Europa 和 Titania 是已发布的产品规划，因此不能把它们都描述成相同的供货状态。[Voyager SDK 在 GitHub 公开](https://github.com/axelera-ai-hub/voyager-sdk)，涵盖模型选择、编译、量化、执行和应用流水线；客户支持仍需账号。

公开的 [Voyager Model Zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo/)是市场上信息最充分的模型库之一。它列出 YOLOv3、YOLOv5、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLO11、YOLO26 n/s/m/l/x、YOLOX、YOLO-NAS、OBB、姿态和分割，以及大量分类和稠密预测模型，并提供模型变体、输入尺寸、精度、准确率和实测性能。将量化损失和精度与速度一并公布，比只公布峰值 TOPS 更有用。

产物命名随 API 代际发生过变化。alpha 版 [Pipeline Builder 编译流程](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/)会生成 `.axm` 模型，并可将可移植流水线打包为 `.axe`。[经典流水线文档](https://docs.axelera.ai/sdk/reference/pipeline/model-formats/)则描述 `.axmodel`、模型元数据和清单。集成时应检测安装的 Voyager 代际，而不是硬编码一种扩展名。

### Qualcomm

高通的设备覆盖范围很广，但「Qualcomm NPU」背后有多种接口。开发者会根据设备和产品类别使用 QAIRT/QNN 下的 Hexagon HTP、较旧的 SNPE 流程、Qualcomm AI Hub 编译服务、LiteRT，或 ONNX Runtime 的 QNN 执行提供程序。

官方 [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models)仓库提供了较强的模型级证据。它发布了 YOLOv3、YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLO11、YOLO26 检测/分割/姿态、YOLOX、YOLO-World、YOLOR、RF-DETR，以及大量分类、深度和姿态模型的优化包。AI Hub 还提供特定设备上的性能分析，这很重要，因为针对一种 HTP 代际编译的模型不会自动适用于另一代。

主要集成成本来自庞大的组合矩阵：Android 与嵌入式 Linux、QCS 订购型号与消费级 Snapdragon、HTP 架构、QNN/QAIRT 版本和量化方案都很重要。高通当前不同页面对 [Dragonwing IQ-9075](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075)状态的说法不一致：产品页标为 Active，其[评估套件](https://www.qualcomm.com/developer/hardware/qualcomm-iq-9075-evaluation-kit-evk)可用于评估，而[产品生命周期计划](https://www.qualcomm.com/internet-of-things/products/product-longevity-program)仍标为 Sampling，并列出至 2038 年的生命周期。订购 SKU 前缀为 QCS9075；高通宣传 50 和 100-dense-INT8-TOPS 配置，并支持 Ubuntu/Yocto。应针对确切订购 SKU 确认量产状态；LibreYOLO 集成也应记录 SKU，而非生成名为「Qualcomm」的通用目录。

### DEEPX

DEEPX 的 DX-M1/DX-M1M 加速器使用 DXNN SDK。DX-COM 负责模型编译和量化，DX-RT 执行模型，部署产物格式为 `.dxnn`。公司公开了 [DX-AllSuite](https://github.com/DEEPX-AI/dx-all-suite)、[DX-APP](https://github.com/DEEPX-AI/dx_app)应用示例，以及大型[在线 Model Zoo](https://developer.deepx.ai/modelzoo/)。

截至 2026 年 8 月 15 日，公开目录在 DX-COM 2.4.0 和 DX-RT 3.4.0 下报告 354 个条目。文档覆盖 YOLOv3 至 YOLO11 和 YOLO26、YOLOX、SSD、EfficientDet、NanoDet、DAMO-YOLO、YOLO 分割、姿态和旋转框。DEEPX 是较可行的集成候选，原因是编译器/运行时边界和部署产物名称都很清楚，且公开视觉模型覆盖面广。

## 工业 SoC 是多个生态，不是一套生态

工业厂商的产品生命周期通常较长，在摄像头、安全和实时集成方面也优于面向创客的开发板 SoC。其 AI 技术栈可能更复杂，因为同一厂商可能同时提供几种互不相关的 NPU 架构。

### Texas Instruments

TI 当前的边缘 AI 处理器包括 AM62A、AM67A、AM68A、AM69A 和相关 TDA4 设备。软件路径由 Processor SDK Linux、TIDL 编译器/运行时、[Edge AI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools)和 [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo)组成。TIDL 可与 ONNX Runtime、TensorFlow Lite 等应用运行时集成，同时卸载受支持的子图。

TI 提供的远不止框架导入声明：其模型库包含目标检测、分割、姿态、分类、深度等任务的转换模型和性能元数据。TI 也提醒，模型库产物是开发起点，不会自动成为可用于生产的资产。很多厂商对比资料会漏掉这个有价值的限制说明。

TI 还将 [TDA54-Q1](https://www.ti.com/product/TDA54-Q1)列为 Preview，该器件最多有四个 C7 NPU，标称可达 400 TOPS；更广泛的 TDA5 系列宣传值最高为 1,200 TOPS。这些是新一代产品的厂商数据，不代表在 TI 发布针对 TDA54-Q1 的目标矩阵之前，可以把 AM6xA/TDA4 的 TIDL 模型验证结论转移过去。

### NXP

NXP 至少需要区分四种后端：

| NXP 目标 | 加速器 | 编译器/运行时路径 |
|---|---|---|
| i.MX 8M Plus | 2.3-TOPS VeriSilicon Vivante NPU | eIQ 搭配 TIM-VX/VX delegate |
| i.MX 93 | Arm Ethos-U65 | 量化 TFLite 加 Arm Vela |
| i.MX 95 | NXP eIQ Neutron NPU | Neutron Converter 和 eIQ 运行时 |
| Ara-1 / [Ara240](https://www.nxp.com/products/ARA240) | 源自 Kinara 的独立 NPU；Ara240 宣传最高 40 eTOPS | Ara SDK，并纳入更广泛的 eIQ 生态 |

[eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo)包含 YOLOv4-tiny、YOLOv8、NanoDet、CenterNet、FastestDet、SSD Lite、YOLACT 等模型的产物或配方；其中 YOLOv5 条目仅为文档。某个后端上的验证条目不能证明其他三个后端也适用。NXP 自己的[i.MX 平台 YOLO 导出指南](https://community.nxp.com/t5/i-MX-Processors-Knowledge-Base/Exporting-YOLO-Models-for-NXP-i-MX-Platforms/ta-p/2381361)就说明了按目标转换模型的复杂性。NXP 表示，整合式 [eIQ Toolkit 在 2025 年第三季度发布 1.17 版后将不再更新](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/eIQ-FAQ/ta-p/1099741)；当前工作流使用 eIQ Neutron SDK 等独立软件包。

NXP 于[2025 年 10 月完成对 Kinara 的收购](https://media.nxp.com/news-releases/news-release-details/nxp-completes-acquisitions-aviva-links-and-kinara-advance/)。[Ara240 规格表](https://www.nxp.com/docs/en/fact-sheet/ARA240DNPUFS.pdf)标称最高 40 eTOPS，并报告 YOLOv8n 每秒处理 313 张图像。NXP 的产品信息同时列出 Ara SDK 和 eIQ Toolkit，但这不能证明它们与独立的 [i.MX 95 Neutron 路径](https://eiq.nxp.com/learning-hub/convQuant/neutron.html)之间的产物可以互换。[16 GB M.2 模块](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT)处于 Active 状态，USB 选项仍为预生产。NXP 将 eTOPS 中的「e」定义为「equivalent」，即等效，而非「effective」，即有效；这不是其他厂商的 dense-INT8 TOPS 指标，不能直接比较。

### STMicroelectronics

[STM32N6x7 设备](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html)，包括 STM32N657 和 STM32N647，将 600-GOPS Neural-ART 加速器带入 MCU 级产品；通用型 N6x5 产品线不含该加速器。[STM32Cube AI Studio](https://www.st.com/en/development-tools/stedgeai-cubeai.html)和 ST Edge AI Core 会分析并优化导入模型，[STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services)则提供部署、优化、基准测试和示例工作流。

当前[目标检测概览](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/docs/README_OVERVIEW.md)记录了 Tiny YOLOv2、YOLOv5u、YOLOv8、YOLO11、YOLO26 和 ST-YOLOX，以及分类、姿态和分割服务。这与 200-TOPS PCIe 卡不属于同一性能级别。它的价值在于摄像头接入、推理和控制都能在严格的嵌入式功耗和内存范围内完成。

### Renesas

Renesas RZ/V 处理器集成 DRP-AI 加速器。软件从 DRP-AI Translator 和 DRP-AI TVM 演进到由 EdgeCortix MERA 技术驱动的 [RUHMI](https://www.renesas.com/en/software-tool/ruhmi-framework)。开源的 [RZ/V DRP-AI TVM 仓库](https://github.com/renesas-rz/rzv_drp-ai_tvm)和 [RZ/V2H 验证列表](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md)列出 YOLOv5、YOLOv8、YOLO11 和 YOLO26 的 n/s/m 变体、YOLOX、姿态和分割网络，以及分类模型。

Renesas 是可信的机器人和工业目标，但为了保证可复现性，仍需记录具体开发板、DRP-AI 代际、转换器版本和 CPU 侧后处理。

## 智能传感器和摄像头优先的处理器

### Sony IMX500

Sony IMX500 不是普通应用处理器。它将图像传感和 AI 处理堆叠起来，让推理可在传感器内完成，从而减少离开摄像头的图像数据量。Raspberry Pi AI Camera 让这种架构更易于使用。

官方 [Raspberry Pi IMX500 模型仓库](https://github.com/raspberrypi/imx500-models)发布了打包好的 YOLOv8n、YOLO11n、EfficientDet Lite0、NanoDet+ 和 SSD MobileNetV2 FPN Lite 模型。[AI Camera 文档](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)说明了转换、打包和传感器端后处理。部署单元是 RPK 包，而不是通用 ONNX 文件；传感器内存和算子限制是关键设计边界。Raspberry Pi 的[产品页面](https://www.raspberrypi.com/products/ai-camera/)称该摄像头至少会生产至 2028 年 1 月。

### Ambarella

Ambarella 的 CVflow 处理器广泛应用于摄像头、无人机和汽车视觉。当前产品系列包括摄像头用 CV72/CV75、高性能视觉系统用 CV5/CV52，以及汽车用 CV3-AD。Cooper 开发者平台和 CVflow 编译/运行时流程构成公开命名的软件栈。

[公开开发者模型库](https://www.ambarella.com/developer/model-garden/)列出 YOLOX、RTMDet、DeepLabV3+、TopFormer、OWL-ViT 和多模态模型。不过，详细编译器文档和下载仍面向合作伙伴。因此，Ambarella 集成应从厂商或 OEM 合作关系开始，不能假设有公开 pip 包可用。

### Synaptics Astra

Synaptics 有三个相关目标。SL1600/SL1680 Astra 系统使用 [SyNAP toolkit](https://developer.synaptics.com/docs/synap/introduction)，将源模型和 YAML 描述转换为 `model.synap` 包。较新的 SL2611/13/15/17/19 设备使用基于 IREE/MLIR 的 [Torq 平台](https://developer.synaptics.com/docs/torq/introduction)，产物为 `.vmfb`。[SR100 系列](https://developer.synaptics.com/docs/sr/introduction-sr)则是采用 Cortex-M55 和 Ethos-U55、使用独立 SDK 的 MCU 系列。这些产物和 API 不能互换。

官方 [YOLO 基准教程](https://developer.synaptics.com/docs/sl/tutorials/vision/benchmark-yolo)提供了相当具体的流程：将 YOLOv8 导出到 TFLite，在 SyNAP 中进行非对称 UINT8 校准，为 SL1680 编译，再运行 `synap_cli` 和目标检测应用。教程证明 SL1680 支持 YOLOv8n/v8s，并说明 SyNAP 支持至 YOLO11。另一个 [SL261x 目标检测指南](https://developer.synaptics.com/docs/sl/sl2600/getting-started/object-detection)通过 Torq `.vmfb` 运行 YOLOv8。这些是特定目标的官方示例，并不表示三个系列的所有 YOLO 计算图都受支持。

### MediaTek Genio

MediaTek 值得单独介绍，因为当前 [IoT AI Hub](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html)现在公开了硬件/软件矩阵、模型包和基准测试表，补上了早期市场调研无法核验的信息。Genio 360/360P/420/520/720 使用 NeuroPilot 8 和 MDLA 5.3；Genio 510/700 使用 NP6 和 MDLA 3.0；Genio 1200 使用 NP6 和 MDLA 2.0。MediaTek 指出，每款 SoC 的 NeuroPilot 和 MDLA 代际、算子集合、编译器和运行时版本在产品生命周期内相互绑定。

分析型 AI 路径以 TFLite 为中心：

```text
PyTorch model
    -> NeuroPilot Converter and representative calibration
    -> quantized .tflite
    -> version-matched ncc-tflite compiler
    -> chip-generation-specific .dla
    -> Neuron Runtime on MDLA
```

较新的 Genio 产品还支持在线 LiteRT delegate，并可在受支持的操作系统上使用 ONNX Runtime CPU/NPU 路径。这些模式不同于离线 `.dla`。[软件架构指南](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/software_architecture.html)和[资源矩阵](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html)明确说明了这种区别。

官方[分析型模型表](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical.html)发布了 YOLOv5s 和 YOLOv8s 的基准条目，以及多个 MDLA 代际的转换指南。由于 AGPL-3.0 限制，该页面明确不分发预转换的 YOLO 产物。其 Quant8、640 x 640 离线测量在 Genio 720 上分别为 5.35 ms 和 8.04 ms。[YOLOv8s 模型页面](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical/YOLOv8s.html)发布了输入/输出张量、按后端区分的耗时，以及有关 MediaTek 自定义算子的警告。这些是厂商基准，不代表端到端摄像头结果。

获取权限是限制因素。公开的 Yocto 文档很详细，但当前 NP8 一体化转换器/编译器包和大量 Android 资料标为 NDA 或直客资源。普通开发者账号与 MediaTek Online 客户账号的权限不同。因此，LibreYOLO 应将 Genio 视为技术证据充分、但可复现的自带模型编译集成依赖合作关系的平台。

## 面向中国市场的边缘 AI 生态

一些性能突出、价格低廉且易获取的视觉平台，在英文市场清单中并不常见。

### D-Robotics 和源自 Horizon 的 BPU 平台

D-Robotics 围绕 RDK 开发板生态维护 BPU 加速器，应用于 RDK X3、X5、Ultra 和较新的 S100 系列产品。其 [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo)当前 `rdk_x5` 分支记录了 RDK X5 上 YOLOv5/v5u、YOLOv8/9/10/11/12/13/26、YOLOE、YOLO-World，以及检测、分割、姿态、分类、OCR 和 CLIP 路径。X3 使用独立分支；当前 S 系列交付位于主模型库的 [`rdk_s` 分支](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s)；旧版 [`rdk_model_zoo_s 仓库`](https://github.com/D-Robotics/rdk_model_zoo_s)则保留历史演示。因此，现代 X5 列表不能证明所有模型都在 X3、Ultra 或 S100 上通过验证。

OpenExplorer/Algorithm Toolchain 可导入 ONNX 或 Caffe 进行 PTQ，也支持面向 PyTorch 的 QAT 流程。部署因平台而异：当前 RDK X5 通过 `libdnn` 下的 `hbm_runtime` 使用 `.bin`；当前主线 `rdk_s` 流程通过 `libhbucp` 下同名的 `hbm_runtime` API 使用 `.hbm`；X3 仍保留较旧的分支和推理接口。旧 `rdk_model_zoo_s` 资料也描述过历史 `.bin` 和 `.hbm` 路径，因此这些产物必须与对应分支和工具链配套。不支持的算子可能回退至 CPU。公开示例值得参考，但 RDK OS、开发板固件、工具链和模型二进制文件之间的版本配对仍是兼容性契约的一部分。

### AXERA

AXERA AX650/AX630/AX620 系列常见于 Sipeed MaixCAM 等紧凑型 AI 摄像头和开发板。Pulsar2 导入 ONNX，执行校准和精度分析并生成 `.axmodel`；AXEngine 负责运行该产物。

[AXERA 官方示例](https://github.com/AXERA-TECH/ax-samples)覆盖 YOLOv5/6/7/8/9/10/11/13/26、YOLOX 和 YOLO-World，任务和芯片支持因平台而异。在受支持的目标上，当前 YOLO26 示例包括检测、姿态、分割和 OBB。[Pulsar2 转换示例](https://pulsar2-docs.readthedocs.io/en/latest/appendix/model_convert_examples.html)展现了实际工作：精确的张量名称、输出切分、布局转换、校准归档和目标硬件配置。

### SOPHGO

SOPHGO 的 TPU-MLIR 是较适合独立集成的编译器技术栈之一，因为编译器本身为开源软件。它可导入 ONNX、PyTorch、TFLite 和 Caffe，降低并量化计算图，为 BM 目标生成 `.bmodel`，为 CV18xx 硬件生成 `.cvimodel`。

[TPU-MLIR 项目](https://github.com/sophgo/tpu-mlir)按目标支持 FP32、BF16、FP16 和 INT8 流程。[SOPHON 演示集合](https://github.com/sophgo/sophon-demo/blob/release/README_EN.md)列出 YOLOv3/4/5/7/8/9/10/11/12/26、YOLOX、PP-YOLOE、YOLO-World、OBB/分割变体、SSD、CenterNet、RetinaFace、SAM/SAM2 和 OCR。与往常一样，在 BM1684X 上运行的演示不能证明相同产物也适用于 BM1688 或 CV18xx。

### Huawei Ascend

Huawei 的边缘推理产品包括 Atlas 200I 和 300I 产品中的 Ascend 310/310P/310B 芯片。CANN 提供 ATC 编译器、AscendCL 运行时和芯片专用算子内核；ATC 将 ONNX 等源表示转换为离线 `.om` 模型。

当前 [Atlas 200I DK A2 文档](https://www.hiascend.com/en/hardware/developer-kit-a2/specification)和 CANN 示例提供适用于 Ascend 310B 的 YOLOv5 `.om` 路径。较早且范围更广的 [Ascend ModelZoo](https://github.com/Ascend/modelzoo)包含 YOLOv2/3/4/5、YOLOX、YOLOR、SSD、RetinaNet、Mask R-CNN、姿态和分割模型，但不应将其描述成每个条目都已在 310B 上重新验证。CANN、内核、固件和 SoC 必须匹配。面向 Ascend310P 的 `.om` 不是通用 Ascend 产物。

### Cambricon

Cambricon MLU 加速器使用 Neuware 和 MagicMind 推理引擎。公开的 [MagicMind Cloud 仓库](https://github.com/Cambricon/magicmind_cloud)固定于 MagicMind 1.7 和 MLU370-X4/S4 矩阵；它提供的是历史兼容性证据，不代表当前 SDK 或 MLU270 的验证情况。仓库记录了 PyTorch、ONNX、Caffe 和 Paddle 路径；1.7 版移除了 TensorFlow 框架支持，尽管仍有历史 TensorFlow 示例。Cambricon 当前的[第 3 系列开发者门户](https://developer.cambricon.com/index/document/index/classid/3.html)列出后续整体 SDK 版本，因此不能把公开示例仓库和当前商业技术栈说成同一版本。

官方 [MagicMind Cloud 模型仓库](https://github.com/Cambricon/magicmind_cloud)发布了相当直接的 MLU370-X4/S4 矩阵，列有 YOLOv3、Tiny-YOLOv3、YOLOv4、YOLOv5、YOLOv7、YOLOv8、PP-YOLOE、SSD、RetinaFace、RetinaNet、Mask R-CNN、DeepLab 和 UNet 示例，也说明是否提供 C++ 或 Python 示例。证据充分；主要障碍是通过 Cambricon 渠道获取 SDK 和容器。

### Canaan/Kendryte

Kendryte 当前的 K230/K230D 和旧款 K210/K510 KPU 芯片在低成本开发板和智能摄像头领域很重要。[nncase 编译器](https://github.com/kendryte/nncase)可导入 ONNX 或 TFLite，使用校准数据进行定点转换，在主机上模拟结果，并生成 `.kmodel`。

官方 [K230 nncase 开发指南](https://github.com/kendryte/k230_docs/blob/main/en/01_software/board/ai/K230_nncase_Development_Guide.md)介绍 YOLOv5s 的编译、模拟和运行流程。独立的 [AI 演示目录](https://github.com/kendryte/k230_docs/blob/main/en/02_applications/ai_demos/K230_AI_Demo_Introduction.md)包含 YOLOv8n 检测、分割和姿态产物；当前 [CanMV 更新日志](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md)增加了优化后的 YOLOv8/11/26 分类、检测、分割、OBB 和姿态示例。数据手册的 [YOLOv5s 结果](https://github.com/kendryte/k230_docs/blob/main/en/00_hardware/K230_datasheet.md)是实际公布的基准；编译器与二进制 KPU 插件版本必须和开发板 SDK 匹配。

### Allwinner

Allwinner V853 将面向摄像头的媒体硬件与 1-TOPS Vivante NPU 集成在一起。[官方英文 NPU 指南](https://docs.aw-ol.com/v853/en/npu/dev_npu/)说明如何通过 `viplite` 导入 Acuity/Pegasus、量化、验证和部署。Allwinner 自己的模型页面及 [YOLOv5 指南](https://docs.aw-ol.com/v853/npu/npu_yolov5/)列出 YOLOv2/3/4/4-tiny/5/5s、RetinaNet、MobileNet、ResNet 和人脸/人员网络。

这是真实的计算机视觉支持，但仍绑定较旧的 Tina Linux/Vivante 工具链，SDK 下载需要账号。市场图谱应纳入该平台，同时明确这些限制。

## 常被忽略的韩国、台湾和中国平台

英文资料经常从 Rockchip 直接跳到 NVIDIA，遗漏另一批真实且有文档支持的芯片。其中一些生态当前 YOLO 矩阵的覆盖范围甚至超过知名度更高的西方初创公司。

### Mobilint

韩国加速器厂商 Mobilint 销售低功耗 REGULUS 和高吞吐量 ARIES MLA100。其 qb SDK 接受 PyTorch、TensorFlow、TFLite、ONNX 和 Keras 相关输入，量化至 INT8 并生成编译产物 `.mxq`。公开的[视觉矩阵](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md)列出 YOLOv3/v5/v7/v8/v9/v10/11/12/26 检测，以及 YOLO 分割、姿态和 OBB 变体。[ARIES 产品页](https://www.mobilint.com/aries/mla100)甚至发布了 YOLO11s 和 YOLO26m 的具体结果。这是有力的集成证据，但 SDK/硬件仍需商业渠道获取。

### Sunplus

Sunplus SP7350/C3V 使用源自 Vivante 的 NPU 技术栈，但其打包方式与 Allwinner 或旧版 Amlogic 不同。公开流程包括 Acuity 转换、NPU Docker 环境、SNNF 运行时和 `.nb` 输出。官方[自定义 YOLOv8 指南](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350)讲解了转换和部署，而不只是宣称支持某个框架。采用相关 IP 不代表其 `.nb` 能移植到其他 Vivante SoC。

### ESWIN Computing

ESWIN 的 RISC-V 系列包括 EIC7700/EIC7700X 和双晶粒 EIC7702/EIC7702X；采用 EIC7700 的 Milk-V Megrez 等开发板已出货。ENNP 包括 EsQuant、当前的 EsAAC 编译器、黄金数据生成、模拟和 ESSDK 运行时，编译产物为 `.model`；较旧 ENNP 资料使用 `ennc-compile` 名称。当前 [EsAAC 页面](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac)记录了 ONNX 输入，因此其他框架需先导出或转换为受支持的中间表示。Milk-V 托管的 [ENNP 概览](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/introduction)和 [YOLOv3 教程](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3)证明存在公开端到端路径，但没有广泛的当前模型/精度矩阵。

### Nuvoton M55M1

Nuvoton M55M1 属于 MCU 级设计，采用 Cortex-M55 加 Ethos-U55-256，而非 Linux 应用处理器。NuEdgeWise/NuML 和 Arm Vela 负责准备量化 TFLite Micro 模型。公开的 [NuEdgeWise 仓库](https://github.com/OpenNuvoton/NuEdgeWise)列出 YOLOv8-nano、YOLOX-nano、YOLO Fastest v1.1、SSD-MobileNet FPNLite 和多个分类器。这些小型网络示例体现了设备的内存/功耗等级，并不证明其支持标准 640 像素 YOLO 变体。

### Rebellions 和 FuriosaAI

Rebellions 的 ATOM 系列使用 RBLN SDK 和编译产物 `.rbln`。其[官方支持版本说明](https://docs.rbln.ai/v0.8.2/supports/release_note.html)列出 YOLOv3 tiny/full/SPP、YOLOv5 n 至 x、YOLOv6 n 至 l、YOLOv7 变体和 YOLOv8 n 至 x。当前[加速卡支持矩阵](https://docs.rbln.ai/latest/supports/version_matrix.html)将 ATOM CA02 和 ATOM+ CA12 标为 EoL，ATOM+ CA22 和 ATOM-Max CA25 标为 Active。ATOM-Lite CA21 虽有产品页面，却未出现在该 SDK 矩阵中，其当前工具链状态不明确。文档公开，但编译器 wheel 需要 Rebellions Portal 凭据。

FuriosaAI 必须按代际区分。Warboy 是较旧的计算机视觉加速器，使用 INT8 `.enf` 产物，官方[模型库](https://developer.furiosa.ai/furiosa-models/latest/)包含 SSD、YOLOv5M/L 和 YOLOv7-w6-pose。RNGD 使用完全不同的 [`.fxb` 技术栈](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html)，主要面向 LLM/VLM 工作负载。Furiosa 当前[路线图](https://developer.furiosa.ai/latest/en/overview/roadmap.html)记录了 YOLOv8m 视觉支持于 2024 年第四季度完成，但当前公开支持模型和性能资料仍以 LLM/VLM 为主，未提供详细的 YOLOv8m 精度/性能矩阵。这是已发布代际的证据，不能据此声称 Warboy 兼容。

### Realtek、Telechips、T-Head 和 SigmaStar

以下四个平台真实存在，但其自带模型部署路径范围较窄、版本较旧或获取受限：

| 平台 | 可复现的公开证据 | 需要保留的限制 |
|---|---|---|
| Realtek AmebaPro2 | [Arduino/FreeRTOS SDK](https://github.com/Ameba-AIoT/ameba-arduino-pro2)打包了 YOLOv3/4/7-tiny、SCRFD 和 MobileFaceNet `.nb` 模型 | 自定义模型转换需登记意向/联系 Realtek |
| Telechips TOPST TCC7500 | [2026 年官方项目](https://docs.topst.ai/blog/31)完成了 YOLOv8s 转换和部署 | UFLD v1/v2 转换因不支持的层而失败；文章只提出拆分/后处理绕行方案。完整编译器/算子资源通常需要邮件申请权限 |
| T-Head TH1520 | Sipeed 的[应用指南](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html)使用 HHB 和 CSI-NN2/SHL 运行 YOLOv5n/s | 技术栈公开，但维护情况和版本说明落后于当前领先平台 |
| SigmaStar SSU9383CM | 当前文档说明 MI_IPU 运行时 API；旧 SGS_IPU 资料记录 `.sim` 到 `sgsimg.img`，旧后处理器列出 SSD 和 YOLOv1/2/3 | 没有公开证据证明旧编译产物或具名模型路径适用于 SSU9383CM |

### HiSilicon 智能视觉不等于 Huawei Ascend

HiSilicon Hi3516CV610/DV500、Hi3519DV500 和 Hi3403V100 摄像头 SoC 通过 NNN/SVP-NNN 运行时提供 ATC 到 `.om` 的工作流。官方 [HiSpark 模型库](https://gitee.com/HiSpark/modelzoo/blob/master/README.md)针对特定目标的矩阵列出 YOLOv3/4/5/6/7/8/9/10/11、YOLO11 分割/姿态、YOLOv8 OBB/World/分割、OCR、深度和 TinySAM，尤其涉及 Hi3403 和 Hi3591P。它并未证明以上所有智能视觉 SoC 都能运行每一个条目。部分条目属于路线图，因此必须区分已发布示例和计划支持；仓库也声明其提供的 ModelZoo 模型仅限非商业用途。ATC/`.om` 名称相同，并不能证明它与基于 CANN 的 Ascend 产品线在产物或运行时方面兼容。

## 新兴和专业加速器公司

### MemryX

MemryX MX3 模块采用数据流架构，可组合多颗芯片。[Neural Compiler](https://developer.memryx.com/tools/neural_compiler.html)接受 ONNX、TensorFlow Lite、Keras 和 TensorFlow 模型，生成由加速器运行时使用的 DFP 包。[运行时 API](https://developer.memryx.com/api/accelerator/accelerator.html)支持多模型和流式处理流水线。

MemryX 文档和示例包含检测、分割和姿态任务的优化 YOLO 预处理/后处理。[SDK 发行说明](https://developer.memryx.com/release_notes.html)明确列出 YOLOv10、YOLO11 和 YOLO26 支持。公开资料在技术上很有用，但缺少类似 Axelera 那种简单的模型/精度/设备表。

### Kneron

Kneron KL520/KL530/KL630/KL720/KL730 产品涵盖小型 USB 和嵌入式加速器。Kneron PLUS 面向应用运行时；当前 Model Toolchain 0.33.1 记录了适用于这些目标的转换、量化、评估、模拟和 `.nef` 编译。KL830 出现在部分 [PLUS 运行时 API](https://doc.kneron.com/docs/plus_c/introduction/run_examples/)中，但不在当前[编译器目标列表](https://doc.kneron.com/docs/toolchain/manual_1_overview/)中；未经厂商确认，不能把通用 `.nef` 编译声明应用于 KL830。

官方 [YOLO 示例](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/)以 Tiny-YOLOv3 演示流程，其他 Kneron 资料涉及 YOLOv5 训练和部署。证据可信，但覆盖范围不如 Hailo、Axelera、Rockchip 或 DEEPX 的现代 YOLO 矩阵广。

### SiMa.ai

SiMa.ai 的 MLSoC 和[量产 Modalix 50-TOPS 平台](https://sima.ai/press-release/sima-ai-next-gen-platform-for-physical-ai-in-production/)使用 Palette 软件环境。ModelSDK、MLA 编译器和 ModelExecutor 覆盖导入、量化、编译、性能分析和执行。根据工作负载与产品不同，工具支持 INT8、INT16 和 BF16 流程。

公司发行说明列出了已验证的 YOLOv7、YOLOv8 检测/姿态/分割、YOLOX 和 YOLOX 分割、DETR、Mask R-CNN 和 EfficientDet 流水线。当前有一个限制不应隐瞒：[Palette SDK 2.1 说明](https://docs.sima.ai/v2.1.2/pages/release_notes/2.1.html)指出，由于 Python/PT2E 量化回归问题，QAT 不可用。文档中的绕行办法是在 SDK 2.0 中生成带标注的 ONNX，再使用 2.1 编译。获取和采购仍以企业客户为主。

部署边界也有文档说明：[ModelSDK 编译流程](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html)生成编译后的 `.tar.gz`，还可生成可执行 ELF；[MPK Tool](https://docs.sima.ai/v2.1.1/pages/palette/mpk_tools.html)会打包可部署的 `.mpk`。这些输出与 MLSoC/Modalix 目标及 Palette 版本绑定。

### EdgeCortix

SAKURA-II 是面向视觉和生成式 AI 的 60-TOPS 宣传型加速器，通过 MERA 软件框架编译。EdgeCortix 也向 Renesas 较新的 RUHMI 流程提供 MERA 技术。

[SAKURA-II 产品资料](https://www.edgecortix.com/en/edgecortix-sakura-ii-accelerator-brief)说明了硬件和编译器，[M.2/PCIe 硬件](https://www.edgecortix.com/en/hardware)通过试用或订购咨询提供。没有找到足够具体且当前有效的公开 YOLO 兼容矩阵。这对采购很有参考价值：确定硬件前，应要求厂商验证目标模型。

### BrainChip

BrainChip 的 Akida 是神经形态事件域处理器，不是传统的稠密张量 NPU。[MetaTF 环境](https://brainchip.com/metatf-dev-tools/)由可安装 Python 包构成，可用于模型创建、低比特量化、转换、模拟和硬件执行。BrainChip 目前销售[已出货的 AKD1500 M.2 协处理器](https://brainchip.com/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/)，也有旧款 AKD1000 硬件和 Akida 2 IP。

当前 [Akida 2 模型卡](https://brainchip.com/wp-content/uploads/2025/04/Akida-2-Model-Card-V1.1-Mar.25.pdf)列出 AkidaNet0.5 YOLOv2 检测器、CenterNet、AkidaUNet 和人脸识别网络。其他支持资料包括 FOMO 和事件型工作负载。Akida 支持特别低比特的权重和激活值，但成功部署可能需要结合架构特点进行转换，不能将它当作另一个通用 INT8 ONNX 目标。除非映射/适配报告明确验证，否则不能把 AKD1000 或 Akida 2 模型库的结果归到 AKD1500 上。

### Blaize

Blaize 销售基于 P1600 的 Pathfinder 和 Xplorer 产品，通过 Picasso SDK、NetDeploy 和 AI Studio 提供软件路径。[产品页面](https://www.blaize.com/products/)明确面向视觉和边缘 AI，但没有当前公开的逐模型兼容矩阵。

因此，本指南将 Blaize 标为商业/受限获取，而不采用社区说法来填补空白。开始前应要求厂商提供针对目标模型的编译和精度报告。

## TinyML 和端侧视觉

### Arm Ethos-U 和 Alif

Arm 将 Ethos-U55、U65 和 U85 NPU IP 授权给芯片公司。[Vela 编译器](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u)接收量化 LiteRT/TFLite 计算图，并将受支持的子图改写为 Ethos-U 自定义算子。不受支持的算子可能留在 CPU 上运行，因此「应用可以运行」和「整个网络都运行在 NPU 上」是两种不同的声明。

实际产品包括 Alif Ensemble E7 中的 Ethos-U55、NXP i.MX 93 中的 Ethos-U65，以及较新系统中的 Ethos-U85。Alif 的 [E7 AI/ML AppKit](https://alifsemi.com/support/kits/ensemble-e7appkit/)记录了两个 ML 加速器，并演示 E7 的 TFLite 到 Vela 路径；[Ensemble E8](https://alifsemi.com/ensemble-e8-series/)结合一个 Ethos-U85 和两个 Ethos-U55 NPU。Alif 还发布了系列级 [INT8 YOLO-Fastest 人脸检测基准](https://alifsemi.com/faster-ai-mcu-inferencing-low-power-consumption/)，在 Cortex-M55 加 Ethos-U55 上以 192 x 192 运行，但没有广泛的现代 YOLO 目标级矩阵。这类内存受限系统适合小型分类和检测网络，不能仅因为模型可以表示为 TFLite，就推断任意 640 像素检测器都适用。

### Infineon PSOC Edge、Himax WiseEye2 和 Analog Devices MAX7800x

Infineon [PSOC Edge E83](https://documentation.infineon.com/psocedge/docs/xsk1761304020519)和 E84 系列采用 Cortex-M55 加 Ethos-U55，并在低功耗 M33 侧增加独立 NNLite 模块。[E84 架构手册](https://documentation.infineon.com/psocedge/docs/bwb1750411526047)记录了 8 位权重、8 或 16 位激活值，以及 Vela 流程：受支持算子转成 NPU 命令流，不受支持的工作留在 CPU 上。[DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter)接受 TFLite、Keras 和 PyTorch 路径；架构资料将 MobileNetV1/V2 列为代表性内核，而非目标基准。[E84 AI Kit](https://documentation.infineon.com/psocedge/docs/cci1762693051052)带有摄像头并使用 ModusToolbox，但目前公开资料尚未提供具名 YOLO 兼容矩阵。

Himax 的 [HX6538 WiseEye2](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/)也将 Cortex-M55 和 Ethos-U55 结合起来，用于常开视觉。对于这一功耗级别，其公开软件资料相当具体：[官方 Grove Vision AI Module V2 示例](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2)包含 YOLOv8n 目标检测、姿态和性别分类、YOLO11n 检测、人脸网格和 PeopleNet。统一流程使用 TFLite Micro 和 Vela，必要时再回退到 CMSIS-NN 和参考内核。这些都是有意设计的小型模型和分辨率，并不证明常规服务器规模的 YOLO 计算图也能运行。

Analog Devices 通过 [MAX78000/MAX78002 CNN 加速器](https://www.analog.com/en/products/max78002.html)采用另一种方案。公开的 [`ai8x-synthesis`](https://github.com/analogdevicesinc/ai8x-synthesis)工具会量化已训练网络并生成目标专用 C 代码、权重和加速器配置，而不是可移植的 ONNX 运行时包。一手证据包括[人脸识别](https://www.analog.com/en/resources/app-notes/an-2616.html)、ADI 模型库中的 TinierSSD 二维码检测（见[模型库](https://ez.analog.com/dsp/software-and-development-tools/edgebench/a/docs-faqs/DF912/available-models-in-adi-model-zoo)），以及图像分类器。这些设备适用于端侧视觉，但未找到当前官方现代 YOLO 矩阵。

### GreenWaves GAP9

GAP9 将 RISC-V 计算与 NE16 神经引擎结合起来。NNTool 导入并量化网络，AutoTiler 和 GAP SDK 则生成可部署代码。公开的 [GAP9 神经网络目录](https://github.com/GreenWaves-Technologies/nn_menu_gap9)包含 MobileNet、EfficientNet、ResNet、MobileNet SSD、人脸和识别应用。GreenWaves 还发布了一个 [YOLOX 行人检测项目](https://github.com/GreenWaves-Technologies/yolox_people_detection)，并提供精度和模拟结果。

这是透明度较高的 TinyML 证据，但完整 SDK 只提供给合格客户，且这些模型远小于主流的 640 x 640 YOLO 变体。

### Lattice sensAI

Lattice FPGA 的 sensAI 工具链将神经网络编译为 FPGA 配置、权重和运行时资源。目标硬件可能是 ECP5、CrossLink-NX、CertusPro-NX、Avant-E 或 Avant-X，具体取决于编译器/IP 模式和开发板。输出与所选 FPGA、加速器模式、内存布局和 bitstream 绑定，不是可移植的运行时模型。

当前 [Neural Network Compiler 手册](https://www.latticesemi.com/-/media/LatticeSemi/Documents/UserManuals/MQ3/FPGA-UG-02052-8-0-Lattice-Neural-Network-Compiler-Software.ashx?document_id=52343)包含很直接的拓扑表：ECP5 支持 YOLOv1；优化/高级 CrossLink-NX 和 CertusPro-NX 模式列有 YOLOv5 和 YOLOv8；YOLO11 仅支持 Advanced 模式 IP。表中也列有 SSD、MobileNetV2-SSD、ResNet、ENet、SqueezeDet 等视觉网络，并按 FPGA 系列显示不支持的单元格。[Advanced CNN Accelerator](https://www.latticesemi.com/en/Products/DesignSoftwareAndIP/IntellectualProperty/IPCore/IPCores05/Advanced-CNN-Accelerator-IP)面向 Avant-E/Avant-X/CertusPro-NX，是商业 IP。这是编译器表中的证据，不代表同一 bitstream 能同时支持所有列出的网络拓扑。

### Microchip VectorBlox

Microchip 公开的 [VectorBlox SDK 3.1](https://github.com/Microchip-Vectorblox/VectorBlox-SDK)可为 PolarFire SoC FPGA 中可配置的 CoreVectorBlox IP 预处理并编译完全量化的 INT8 TFLite 网络。`tflite_preprocess` 和 `vnnx_compile` 会生成 `.vnnx`、`.hex` 和 `.ucomp` 部署产物；仓库还提供模拟器和目标端 C 驱动。上游 TensorFlow、ONNX 和 OpenVINO 计算图仍需走文档化的 TFLite/INT8 准备流程。

当前[教程矩阵](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md)涵盖 YOLOv5n、YOLOv8n 检测、分类、OBB、姿态和分割，YOLOv9t、稀疏/压缩 YOLO 变体、MobileNet、EfficientNet-Lite0、ResNet18、MiDaS、FFNet 和 QuickSRNet。[C 后处理指南](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/C_Postprocessing.md)另外列出 YOLOv2/3/4/5 及 Ultralytics 检测、姿态和 OBB 解码器。SDK 3.1 当前支持 PolarFire SoC Video Kit，明确不支持非 SoC PolarFire Video Kit，因此不能把旧开发板演示误当成当前目标契约。

SDK 可在 GitHub 公开下载，但采用[针对 Microchip 的许可](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/LICENSE.md)，限制软件及衍生品只能用于 Microchip 产品。Microchip 提供免费但需申请的 Libero Silver 和 CoreVectorBlox 许可。部署仍受 FPGA 系统契约约束：加速器配置、bitstream、固件、SDK 生成的网络数据和内存布局必须一致。VectorBlox 编译成功不会让模型变成可复制到任意 PolarFire 设计中的二进制文件。

### Syntiant

Syntiant NDP200 面向常开视觉、音频和传感器推理，功耗范围低于常规 Linux SoC。当前[硬件表](https://www.syntiant.com/hardware)将 NDP200 列为量产，NDP250 列为送样。[NDP200 产品简报](https://www.syntiant.com/ndp200/)记录了低于 1 mW 的 CNN、RNN 和全连接网络支持；[NDP250 公告](https://www.syntiant.com/news/syntiant-unveils-ndp250-neural-decision-processor-with-next-gen-core-3-architecture)则称其常开图像识别低于 30 mW。不能将「亚毫瓦」概括用于这两款芯片。

公开资料不能证明广泛兼容 YOLO，因此应通过厂商支持的模型评估来测试小型常开分类器和检测器，而不是假设任意 ONNX 导出都能使用。

### Google Coral：两种互不相关的代际

旧款 Edge TPU 产品使用 Edge TPU Compiler、`libedgetpu` 和 PyCoral，配合完全 INT8 量化的 TFLite 模型。其主要 [Edge TPU](https://github.com/google-coral/edgetpu)和 [PyCoral](https://github.com/google-coral/pycoral)仓库已归档。这确实带来维护风险，但不等于正式的硬件 EOL 通知。

Google 还在维护一个独立且活跃的开源 [Coral NPU](https://github.com/google-coral/coralnpu)：一种可集成到低功耗 SoC 的 RISC-V 加速器 IP。公开项目目前提供 RTL、硬件模拟、工具链和 ELF 示例。它不是 Edge TPU 的 USB/M.2 直接替代品，也没有公布当前 YOLO 部署矩阵。

## 相邻 GPU、客户端 NPU 和可重构计算的陷阱

NVIDIA、AMD、Intel 和 Apple 经常出现在 NPU 搜索结果中，但它们并未提供同一种可互换的加速器。

**NVIDIA Jetson：**Jetson Orin 结合 CUDA GPU 和专用 DLA 核心。TensorRT 可为 DLA 构建序列化引擎，但只有显式启用 GPU 回退时，不受支持的层才会转到 GPU。当前 [DeepStream YOLO 表](https://github.com/NVIDIA-AI-IOT/deepstream_tools/blob/main/yolo_deepstream/README.md)涵盖 YOLOv4/v7/v8/v9/11，但大多数条目针对 GPU；其明确记录的 ONNX DLA 条目是 INT8 YOLOv8s。[NVIDIA DLA 限制](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html)说明了算子限制。Thor 又是另一种情况：NVIDIA 的 [DriveOS 迁移指南](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/NVIDIA_DriveOS_7.0.3_Migration_Guide.pdf)称 Thor 移除了 DLA 核心，改用 GPU 调度灵活性。因此，「在 Jetson 上运行」并不能说明使用了哪种加速器。

**AMD Vitis AI：**较早的 Kria/DPU 代际将 `.xmodel` 计算图编译后通过 VART 执行。当前 [Vitis AI 6.2](https://vitisai.docs.amd.com/en/6.2/)已面向两种独立的 Versal AI Edge 路径 GA：[Gen1](https://vitisai.docs.amd.com/projects/gen1/en/latest/index.html)在 VEK280/VE2802 上使用 ONNX、AMD Quark 和绑定目标的 snapshot/子图流程；[Gen2](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html)在 VEK385 上有独立的编译和运行时契约。Gen1 发布了 YOLOv5、YOLOv7、YOLOv8 和 YOLOX 示例。Ryzen AI 是第四套独立的客户端 NPU 技术栈。不能仅因它们都带有 Vitis 或 AMD 名称，就在旧 DPU、Versal Gen1、Versal Gen2 和 Ryzen AI 之间迁移产物或验证结果。

**Intel OpenVINO：**OpenVINO 可通过同一 API 面向 CPU、GPU 和 Core Ultra NPU。[Intel NPU 插件文档](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md)说明受支持的 NPU 代际，[已验证模型矩阵](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html)则按设备列出结果。除非矩阵明确说明 NPU 验证，否则 YOLO notebook 只能证明应用流程，不能证明 NPU 验证。算子、形状、精度和已安装 NPU 驱动仍决定完整计算图能否运行于 NPU。OpenVINO IR（`.xml` 加 `.bin`）是可复用的源级 IR；编译模型缓存则取决于设备和软件。

**Apple Core ML：**`coremltools` 可将模型转换为 `.mlpackage`，再由 Xcode 编译成 `.mlmodelc`。Core ML 可在 CPU、GPU 和 Apple Neural Engine 之间调度，但有意抽象了具体分配情况。Apple 因而是优秀的部署平台；除非性能分析证据能确认，否则不宜声称「整个 YOLO 计算图都运行在 NPU 上」。

## NPU 芯片厂商背后的 IP 公司

有些公司不直接销售开发板或封装加速器，而是将 NPU 设计授权给 SoC 厂商。同一架构可能出现在不同芯片品牌下，但最终 SDK 和产物仍可能由被授权方定制。

| IP 供应商 | NPU 系列和软件 | 相关性 |
|---|---|---|
| [Arm](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) | Ethos-U55/U65/U85 和 Vela | 应用于 NXP、Alif、Infineon、Himax 等嵌入式产品；面向量化 TFLite/TOSA 的流程 |
| [Arm China](https://www.armchina.com/mountain?infoId=161&name=) | Zhouyi AIPU 和 Compass SDK | 公开[模型库](https://github.com/Arm-China/Model_zoo)列有 YOLOv1-tiny/v2/v3/v4/v5、YOLOX 和 YOLOv8-seg；参考模型不代表每家被授权芯片都适用 |
| [VeriSilicon](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico) | Vivante VIP 系列和 Acuity SDK | 相关 Vivante NPU 系列出现在多个 SoC 中，包括分别采购的 A311D 和 i.MX 8M Plus 路径；各芯片厂商仍会提供各自集成方案 |
| [Imagination Technologies](https://www.imaginationtech.com/products/open-access/) | PowerVR Series3NX AX3146/AX3386/AX3596；历史 IMG Series4；Neural Compute SDK/IMG DNN | 可授权的 NNA IP，而非零售开发板；Open Access 提供 1/5/10-TOPS Series3NX 配置；官方 NC-SDK 项目列出 PP-YOLOE、EfficientNet 和 HRNet，但未覆盖每种 IP 配置 |
| [CEVA](https://www.ceva-ip.com/product/ceva-neupro-studio/) | NeuPro-Nano/NeuPro-M 和 NeuPro Studio | 可授权 NPU IP，配套导入、量化、压缩、计算图编译、模拟和 C/C++ 代码生成 |
| [MIPS](https://mips.com/processor-solutions/arc-npx-family/) | ARC NPX6 和 [MetaWare MX](https://mips.com/processor-solutions/arc-metaware-mx/)，原属 Synopsys ARC | 可扩展 NPU IP 和 NN SDK，于 [2026 年 6 月](https://mips.com/press-releases/gfmipsarcclose/)被 GlobalFoundries 收购并纳入 MIPS 产品组合；本身不是零售部署目标 |
| [Cadence](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/neuroweave-sdk.html) | Neo NPU、Tensilica DSP 和 NeuroWeave SDK | SoC 被授权方使用的编译器/解释器技术栈；网络和量化支持取决于授权配置 |
| [Quadric](https://quadric.ai/npu-ip) | Chimera GPNPU IP 和 SDK | 可编程 NPU/DSP 类 IP，授权给其他公司的芯片；最终目标是被授权方的硅片 |
| [Expedera](https://www.expedera.com/products-overview/) | Origin NPU IP 和软件栈 | 可授权的边缘 NPU 架构，不是 LibreYOLO 可直接购买的开发板 |

Imagination 说明了为什么架构名称旁边还应列出获取方式和生命周期。其 [Open Access 计划](https://www.imaginationtech.com/products/open-access/)提供三种经硅片验证的 Series3NX 配置，无需预付 IP 许可费，但需经过公司评估，并支付支持/维护费用及量产版税。[Neural Compute SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/)提供编译、优化、量化和运行时工具；官方 [Baidu PaddlePaddle 合作](https://www.imaginationtech.com/news/imagination-and-baidu-paddlepaddle-create-open-source-machine-learning-library-for-model-zoo/)列出了 PP-YOLOE 检测、EfficientNet 分类和 HRNet 分割示例。这些示例不能验证每种 Series3NX 或 Series4 配置；目前各个 [Series4 产品页面](https://www.imaginationtech.com/product/img-4nx-mc1/)都标为不可获取，因此除非销售渠道确认，否则应将 Series4 视为历史产品。

这一层解释了架构之间为何相似，却不会带来产物可移植性。采用相关 Vivante 或 Ethos IP 的两款 SoC，仍可能有不同内存映射、驱动、编译器版本、算子集合和厂商运行时。

## 实际部署的产物：格式绑定表

编译流程中的最后一个文件，是 ONNX 表面可移植性终结的地方。名称和扩展名会变化，但实践规则稳定：保留原始模型和校准数据，因为原生产物通常无法迁移到其他 NPU 代际，也不一定能用另一个 SDK 版本可靠重建。

| 厂商技术栈 | 常见编译器输入 | 设备实际接收的内容 | 绑定条件 |
|---|---|---|---|
| [Amlogic AMLNN](https://github.com/Amlogic-NN/amlnn-toolkit) | ONNX、TFLite、TorchScript 或 PT2 | `.adla`；旧 A311D 使用 `.nb` | 目标 ID、ADLA 代际、AMLNN 编译器构建、NNSDK2/运行时、ADLA 驱动和 BSP |
| [Rockchip RKNN](https://github.com/airockchip/rknn-toolkit2) | ONNX、TFLite 或框架导出模型 | `.rknn` | RKNN 工具包/运行时版本和 Rockchip NPU 系列 |
| [Hailo](https://hailo.ai/developer-zone/documentation/) | ONNX 或经中间 HAR 格式转换的 TensorFlow | `.hef` | Hailo 架构和编译器/运行时代际 |
| [Axelera Voyager](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) | ONNX 或受支持的模型库定义 | alpha Pipeline Builder 使用 `.axm`，`.axe` 流水线包；经典流程使用 `.axmodel` 和清单 | Voyager API 代际和 Metis 目标配置 |
| [DEEPX DXNN](https://github.com/DEEPX-AI/dx-all-suite) | ONNX 和受支持的框架路径 | `.dxnn` | DX-COM/DX-RT 版本和 DX-M 目标 |
| [Qualcomm QAIRT/QNN 或 SNPE](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | ONNX、TFLite 或框架模型 | QNN 模型库/context 二进制文件，或 SNPE `.dlc` | HTP 架构、SoC、后端和运行时版本 |
| [MediaTek NeuroPilot](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) | 转换/量化后的 TFLite | 离线 Neuron Runtime 使用 `.dla`；delegate 模式使用 `.tflite` | NP/MDLA 代际、操作系统镜像、编译器和运行时 |
| [TI TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools) | ONNX 或 TFLite | 导入产物目录和应用模型文件 | TIDL 工具/运行时和处理器代际 |
| [NXP eIQ](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | 通常是 TFLite 或 ONNX | Vela、TIM-VX、Neutron 或 Ara 后端产物 | 选定的 i.MX/Ara 加速器和 BSP，而非笼统的「NXP」 |
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | TFLite、ONNX 或受支持的框架模型 | 生成的网络库/代码和固件资产 | MCU/NPU 目标、内存配置和工具版本 |
| [Infineon PSOC Edge](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) | 整数量化 TFLite/Keras/PyTorch 路径 | 经 Vela 优化、带 Ethos-U 命令流的 TFLite 和应用固件 | E83/E84 型号、Ethos-U 配置、Vela、ModusToolbox 和 BSP；NNLite 是独立目标 |
| [Renesas DRP-AI](https://github.com/renesas-rz/rzv_drp-ai_tvm) | 通过 TVM 或 Translator 处理 ONNX/TFLite | 含多个二进制数据文件的运行时模型目录 | RZ/V 设备以及转换器/运行时代际 |
| [Sony IMX500](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | Edge-MDT 流程生成的量化/打包网络 | `.rpk` 包 | 传感器固件、内存预算和 packer 版本 |
| [Synaptics](https://developer.synaptics.com/) | TFLite/ONNX 和支持的导入格式 | SyNAP 使用 `.synap`；Torq 使用 `.vmfb` | SL16xx 与 SL261x 软件/硬件代际 |
| [Ambarella CVflow](https://www.ambarella.com/developer/) | 合作伙伴工具链输入 | 仅合作伙伴可获取的部署包；产物契约未公开说明 | 通过 Cooper Developer Zone 确认 CVflow 代际、SDK/BSP 和摄像头流水线 |
| [D-Robotics BPU](https://github.com/D-Robotics/rdk_model_zoo) | ONNX 和工具链支持的计算图 | X5 `.bin`；当前 `rdk_s` `.hbm`；X3 使用较旧平台流程 | BPU 代际、仓库分支和对应的 OpenExplorer/运行时发行版 |
| [AXERA Pulsar2](https://github.com/AXERA-TECH/ax-samples) | ONNX | `.axmodel` | AX 芯片目标、Pulsar2 和 AXEngine 版本 |
| [SOPHGO TPU-MLIR](https://github.com/sophgo/tpu-mlir) | ONNX、TFLite、TorchScript 等 | BM 使用 `.bmodel`，CV18xx 使用 `.cvimodel` | BM/CV 芯片目标和 SOPHON 运行时代际 |
| [Huawei Ascend CANN](https://www.hiascend.com/en/software/cann) | ONNX 或受支持的框架计算图 | `.om` | Ascend 芯片、CANN/ATC 和设备固件/驱动 |
| [Cambricon MagicMind](https://github.com/Cambricon/magicmind_cloud) | 框架计算图或交换模型 | 序列化 MagicMind engine/model | MLU 架构和 Neuware/MagicMind 版本 |
| [Canaan nncase](https://github.com/kendryte/nncase) | ONNX 或 TFLite | `.kmodel` | KPU 代际和匹配的 nncase 目标插件 |
| [Mobilint qb](https://www.mobilint.com/sdk-qb) | ONNX 和受支持的框架模型 | `.mxq` | REGULUS/ARIES 目标和 qb 编译器/运行时 |
| [Rebellions RBLN](https://docs.rbln.ai/latest/index.html) | PyTorch 2 或 TensorFlow 支持的计算图 | `.rbln` | ATOM 代际和 RBLN 编译器/运行时 |
| [FuriosaAI](https://developer.furiosa.ai/docs/latest/en/) | ONNX/TFLite 和受支持的框架路径 | Warboy `.enf`；RNGD `.fxb` | 完全不同的加速器和 SDK 代际 |
| [Sunplus SNNF](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2004353133) | Acuity 支持的网络 | `.nb` | SP7350 NPU 驱动、运行时和 BSP |
| [ESWIN ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) | 当前 EsAAC 流程文档记录 ONNX 输入；其他框架需先导出/转换 | `.model` | 确切的 EIC7700 系列目标和 ENNP/ESSDK 版本 |
| [Nuvoton / Ethos-U](https://github.com/OpenNuvoton/NuEdgeWise) | 量化 TFLite | 带 Ethos-U 自定义算子、经 Vela 优化的 TFLite | M55M1 内存规划、Vela 和固件 |
| [Himax WiseEye2](https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2) | 完全整数量化的 TFLite | 模型闪存中的 Vela 优化 `.tflite`，以及 `output.img` 等开发板固件 | HX6538/WE2 配置、Vela、TFLite Micro、Ethos-U 驱动和回退内核 |
| [Analog Devices AI8X](https://github.com/analogdevicesinc/ai8x-synthesis) | PyTorch 检查点、网络 YAML 和样例输入 | 生成的设备专用 C 源码/头文件、权重和编译固件 | MAX78000/MAX78002 内存/层限制、综合工具和 MSDK 版本 |
| [Realtek AmebaPro2](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | 厂商转换服务/输入 | `.nb` | RTL8735B 固件和 VoE/NeuralNetwork API 版本 |
| [Telechips Enlight](https://docs.topst.ai/product/p/ai) | Darknet、TensorFlow、ONNX 或 PyTorch 路径 | `.enlight` 中间格式和编译部署包 | TCC7500 NPU 及 TC-NN/Enlight 版本 |
| [T-Head HHB](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | ONNX 和受支持的框架计算图 | `hhb.bm`、生成的参数/代码和可执行文件 | TH1520 CSI-NN2/SHL 技术栈 |
| [SigmaStar MI_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | 当前 SSU9383CM 文档公开 MI_IPU 运行时；旧资料使用 TensorFlow/Caffe 系列输入 | 旧版 `.sim` 转为 `sgsimg.img`；当前公开产物尚未验证 | 确切 SigmaStar IPU 和 SDK 代际；旧编译器与 SSU9383CM 的兼容性尚未验证 |
| [HiSilicon smart vision](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) | ONNX 和 ATC 支持的计算图 | `.om` | 确切智能视觉 SoC 和 NNN/SVP-NNN 技术栈；不具备通用 Ascend 可移植性 |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | ONNX、TFLite、Keras 或 TensorFlow | `.dfp` 数据流包 | MX 硬件配置和运行时/编译器发行版 |
| [Kneron](https://doc.kneron.com/docs/) | ONNX 加工具链配置 | `.nef`；KL730 NEFv2 可包含 `.kne` | 文档记录的编译目标、芯片代际、固件和 PLUS 运行时；Toolchain 0.33.1 未证明支持 KL830 编译 |
| [SiMa.ai Palette](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) | 受支持的框架或 ONNX 模型 | ModelSDK 编译后的 `.tar.gz`、文档记录的可执行 ELF，或 MPK Tool `.mpk` 包 | MLSoC/Modalix 目标和 Palette 发行版 |
| [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | ONNX 或网络定义 | 序列化引擎，常见扩展名为 `.engine` 或 `.plan` | GPU/DLA 架构、TensorRT、CUDA，且通常还绑定设备 |
| [AMD Vitis AI](https://vitisai.docs.amd.com/en/6.2/) | 量化 ONNX/框架计算图 | 旧版 `.xmodel`；Gen1 NPU snapshot/子图；Gen2 编译缓存目录或生产 `.rai` 包 | 确切 NPU 代际/IP 配置、平台镜像和 Vitis/Vivado/PetaLinux 矩阵 |
| [Microchip VectorBlox](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) | 完全量化的 INT8 TFLite | `.vnnx`、`.hex`、`.ucomp`，以及匹配固件/FPGA 设计 | CoreVectorBlox 配置、PolarFire SoC Video Kit、SDK 3.1、bitstream 和内存布局 |
| [Intel OpenVINO](https://docs.openvino.ai/) | 框架模型或 ONNX | IR `.xml` 加 `.bin`，或设备专用编译缓存 | IR 可复用；编译缓存取决于设备、驱动和 OpenVINO 版本 |
| [旧版 Google Edge TPU](https://coral.ai/docs/edgetpu/models-intro/) | 完全 INT8 量化的 TFLite | 经 Edge TPU 编译的 `_edgetpu.tflite` | Edge TPU 编译器/运行时和受支持的量化算子；与新 Coral NPU IP 无关 |
| [Google Coral NPU IP](https://github.com/google-coral/coralnpu) | 当前公开 IP 项目的 C/C++ 示例 | RTL/硬件模拟用 ELF 示例及未来 SoC 集成；没有公开 YOLO 编译产物 | 确切授权/集成的 SoC 实现及仍在演进的开源工具链 |
| [Lattice sensAI](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | 受支持的网络描述 | FPGA bitstream、加速器配置和权重 | FPGA 系列、sensAI IP 模式和内存实现 |
| [Imagination NC-SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) | Caffe、TensorFlow 或 ONNX | 面向客户的编译产物文件名未公开指定 | 授权的 NNA 配置、SoC 集成、NC-SDK/DDK 和被授权方运行时 |

这张表不只是扩展名速查表。编译器版本、目标标识、驱动、固件和 BSP 都属于模型可复现身份的一部分。未来 LibreYOLO 导出器应将这些信息写入每个产物旁的机器可读清单。

## 工具获取和生命周期信号

硬件是否容易购买，与编译器是否能获取，是两回事。开发板可能容易买到，但当前编译器需要客户协议；SDK 也可能公开，却面向难以采购的芯片。以下是一手资料中的具体信号，不是统一的生命周期排名。

| 平台 | 有文档记录的信号 | 实际含义 |
|---|---|---|
| Amlogic ADLA | Apache-2.0 仓库公开并可下载 wheel；兼容开发板驱动/BSP 仍可能需通过 Amlogic 或开发板厂商获取 | 编译器易于评估，但应确认二进制再分发许可和完整兼容矩阵 |
| MediaTek Genio | IoT AI Hub 和 Yocto 指南公开；NP8 一体化工具及 Android 文档标记为直客/NDA | 模型证据可审计，自带模型自动化可能需要合作关系 |
| Hailo | Model Zoo 和运行时公开；Dataflow Compiler 通过 Developer Zone 分发；公开的 `hailo-camera-apps` 于 2026 年 5 月 3 日归档 | 模型发现范围广，但可复现编译需要匹配的账号/版本；归档不能证明 Hailo-15 EOL，应确认当前视觉处理器应用包 |
| Axelera AI | 文档和[公开 Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk)；客户支持需账号 | 少见的可自助使用加速器 SDK，但产品支持和采购仍属商业流程 |
| Qualcomm Dragonwing | [产品生命周期计划](https://www.qualcomm.com/internet-of-things/products/product-longevity-program)将 IQ-9075 列为 Sampling、生命周期至 2038 年，将 QCS6490 列至 2036 年 7 月；IQ-9075 产品页却标为 Active | 工业规划信号较强，但一手资料的状态冲突需按 SKU 确认，也不保证一套 AI SDK 的 ABI 稳定至整个周期 |
| Rebellions | 当前[支持矩阵](https://docs.rbln.ai/latest/supports/version_matrix.html)将 CA02/CA12 标为 EoL，CA22/CA25 标为 Active；未列 CA21 | 即使同属 ATOM 系列，采购和 SDK 兼容性也因加速卡而异 |
| NVIDIA Jetson | [官方模块生命周期表](https://developer.nvidia.com/embedded/lifecycle)列出 Orin 模块至 2032 年 1 月、AGX Orin Industrial 至 2033 年 7 月；开发套件没有生命周期承诺 | 生产系统应围绕模块设计，而非开发套件供货情况 |
| 树莓派上的 Sony IMX500 | [AI Camera 产品简报](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf)称至少生产至 2028 年 1 月 | 这是该摄像头模块的最低期限，不适用于所有 IMX500 产品 |
| Amlogic A311Y3 | [2026 年发布页面](https://www.amlogic.com/News/index248.html)宣传至少十年的产品生命周期保证 | 新平台信号积极，但产品项目仍需确认合同和 SKU 细节 |
| Google Coral | 旧版 Edge TPU 和 PyCoral 仓库已归档/只读；独立的 [Coral NPU IP](https://github.com/google-coral/coralnpu)仓库仍活跃 | 这是旧软件维护风险，本身不是正式硬件 EOL 通知；也不能说明无关开源 IP 项目的生命周期 |

## 为什么 TOPS 不是购买指南

在同一厂商和架构内，峰值 TOPS 可能有参考价值；跨厂商比较通常无效，除非以下各项都相同：

- 数值精度，包括 INT8、INT4、FP16 或混合模式
- 一次乘加按一次还是两次运算计数
- 稠密计算还是结构化稀疏计算
- 输入分辨率、批大小和模型计算图
- 量化后的精度
- 仅 NPU 延迟还是完整应用流水线延迟
- 内存传输和主机回退
- 温度状态，以及持续运行而非瞬时突发的性能

检测器是一条流水线：图像采集、缩放/letterbox、归一化、设备传输、神经网络推理、解码、NMS、坐标缩放和应用逻辑。厂商经常只公布中间的神经网络推理部分。[MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/)很有价值，因为它定义了场景、质量目标和系统级测量规则，而不是孤立比较宣传数据。

YOLO 部署至少应记录以下基准信息：

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

缺少这些字段时，两组 FPS 往往衡量的不是同一件事。

## 如何为计算机视觉选择 NPU

按以下顺序选择，不要只看产品页上最大的数字。

1. **验证计算图兼容性。** 编译准确导出的模型，而不是名字相似的模型库检查点。
2. **测量精度。** 在校准和量化后，用真实任务数据集验证编译产物。
3. **端到端测量。** 纳入输入转换、数据传输、解码和 NMS。
4. **检查软件获取方式。** 确认编译器是公开、需要注册，还是只在商业协议后提供。
5. **固定版本矩阵。** 记录编译器、运行时、驱动、固件和目标标识。
6. **核实操作系统支持。** Android、Debian、Yocto、Buildroot、RTOS 和 Windows 支持并不等价。
7. **检查供货和生命周期。** 如果模块、驱动或长期支持情况不确定，技术上很出色的芯片也可能不适合生产。
8. **最后比较成本、功耗和性能。** 只有两个目标上的同一模型都能正确运行后，这些测量才有意义。

### 按使用场景选择起点

| 使用场景 | 可优先评估的平台 | 原因 |
|---|---|---|
| 树莓派专用加速器 | Hailo-8L/8 和 Sony IMX500 | 树莓派官方产品、软件镜像和具体开发流程 |
| 通用 Linux PCIe、M.2 或 USB 扩展卡 | DEEPX、MemryX、Hailo 和 Axelera 产品 | 可获取的加速器外形规格，但需确认主机/驱动兼容性 |
| 低成本 Linux SBC 或摄像头 | Rockchip RK3588/RK3576、有受支持开发板/BSP 时的 Amlogic ADLA、AXERA、Canaan K230 或 Sunplus SP7350 | 集成媒体流水线和公开模型/编译器证据；确认 VIM4 V13A 及以上版本才有 A311D2 NPU |
| 移动设备和大规模 Android 部署 | Qualcomm QNN/AI Hub、LiteRT、Apple Core ML | 部署基数大，移动运行时仍在维护 |
| 工业 Linux 和机器人 | TI TDA4/AM6xA、NXP i.MX、Renesas RZ/V、Hailo | 长生命周期嵌入式产品及摄像头/I/O 生态 |
| 极小型端侧设备或 MCU | STM32N6x7、Infineon PSOC Edge、Himax WiseEye2、Analog Devices MAX7800x、Nuvoton M55M1、Alif/Arm Ethos-U、GAP9 和 Syntiant | 面向严格功耗/内存范围的工具和模型尺寸限制 |
| 可配置 FPGA 视觉 | Microchip VectorBlox、Lattice sensAI 和 AMD Vitis AI 目标 | 硬件流水线灵活，但加速器配置、bitstream 和模型编译器构成同一个版本化系统 |
| 高吞吐量 PCIe 视觉 | Axelera Metis、Hailo、DEEPX、Mobilint、MemryX、Rebellions 或 SiMa.ai | 专用加速器产品及多路流定位 |
| 以中国为中心的产品生态 | Rockchip、AXERA、D-Robotics、SOPHGO、Huawei Ascend、HiSilicon 和 Cambricon | 区域开发板、工具链和模型示例丰富 |

这张表是调研候选清单，不是性能排名。采购渠道、地区供货情况和具体模型都可能改变选型顺序。

## 这对 LibreYOLO 意味着什么

可扩展的设计不是构建数十个互不相关的导出器，而是建立统一、严格的交换契约，以及小型厂商编译器和运行时适配器：

```text
LibreYOLO model
    -> deterministic static ONNX
    -> VendorCompiler.compile(model, target, calibration, precision)
    -> native artifact + manifest
    -> VendorRuntime.load() / infer()
    -> shared task-specific decode and Results objects
```

每个原生产物旁都应有清单，记录：

- 厂商、芯片目标和开发板目标
- 编译器、运行时、驱动和固件版本
- 源检查点和 ONNX 校验和
- 输入名称、形状、布局、色彩空间、归一化和数据类型
- 量化精度、可获取时的 scale，以及校准数据哈希
- 输出张量名称、形状和语义
- 检测任务、类别名称，以及解码/NMS 在芯片上还是主机上执行
- 数值一致性和任务精度结果

LibreYOLO 已提供可移植的 [ONNX 导出](/docs/export/onnx)、[量化工具](/docs/export/quantization)、直接但有意限制范围的 [RKNN 导出](/docs/export/rknn)，以及如实说明外部编译器要求的 [Hailo 流程](/docs/export/hailo)。按本指南采用的标准，Amlogic ADLA 是很适合下一步集成的候选：工具包公开，模型覆盖与 LibreYOLO 高度重叠，而且可明确区分旧版 A311D 路径。

Amlogic 之后的优先级应取决于用户硬件和运行精度验证的能力，而不应只看编译器是否可用。DEEPX、Sony IMX500、D-Robotics、AXERA、SOPHGO 和 Qualcomm 在技术上很有吸引力；如果工业合作伙伴能提供开发板和长期 CI 访问权限，TI、NXP、ST 和 Renesas 也会尤其有价值。

## 候选清单：芯片真实存在，但没有公开的集成契约

完全排除一家公司会让市场图谱产生误导；把它纳入「已支持」则可能更糟。这些厂商已经销售、送样或宣布了相关芯片，但公开证据还不能证明它们提供可复现的当前编译器、产物和具名模型路径，因而不适合作为 LibreYOLO 后端。

| 公司 | 已确认的情况 | 仍列为候选的原因 |
|---|---|---|
| [Samsung](https://semiconductor.samsung.com/processor/automotive-processor/exynos-auto-v920/) | Exynos Auto V920 含双核 NPU，宣传最高 23.1 TOPS | Samsung 表示其 [Neural SDK 已不再提供给第三方开发者](https://developer.samsung.com/neural/overview.html)；Samsung ONE/Circle 不能证明开发者可以访问 Exynos NPU |
| [Novatek](https://www.novatek.com.tw/en-global/Milestone/aboutus_milestones) | 当前公司里程碑提到 Edge AI 和 Edge Vision/Imaging AI SoC；2020 年里程碑提及 MobileNet、SSD 和 YOLOv3 | 没有当前公开的型号/编译器/产物/算子/模型矩阵；历史 YOLOv3 信息不能证明当前 SDK 支持 |
| [Nextchip](https://www.nextchip.com/en/adas/adas.php?idx=5) | APACHE5/NVS2900 和当前 [APACHE6/NVS3000](https://www.nextchip.com/en/adas/adas.php?idx=7)汽车处理器，带 aiMotive aiWare NPU；APACHE6 页面当前同时标出 12 和 8 TOPS | 存在 aiWare Studio 和运行时，但未找到公开产物、量化指南、算子列表或 YOLO 矩阵；不应暗中消解厂商数据之间的冲突 |
| [Black Sesame Technologies](https://bst.ai/en.html) | Huashan A1000/A2000 和 Wudang C 系列汽车/边缘 AI 芯片 | 有公开产品信息，但没有自助式编译器、产物契约或可复现模型库 |
| [Ingenic](https://en.ingenic.com.cn/products-detail/id-19.html) | T41/T40 摄像头 SoC 宣传低比特 NPU；Magik AI 介绍 PTQ/QAT 和计算图编译 | 未找到可下载的当前编译器、产物契约或准确的厂商 YOLO 列表 |
| [Fullhan](https://fullhan.com/en/index.php?a=type&c=article&tid=9) | 多款 IPC SoC 宣传 0.5 至 2 TOPS NPU，[MC6880 NVR 系列](https://fullhan.com/en/index.php?a=type&c=article&tid=48)宣传 4 TOPS | 没有足够公开的 NN 编译器/运行时/框架/模型文档来支持独立复现 |
| [Goke Microelectronics](http://www.gokemicro.com/News/info.aspx?itemid=437) | GK7606V1/GK7206V1/GK7203V1 智能摄像头芯片的一手页面宣传集成 NPU 性能，但在发布本文时仅通过旧版 HTTP 提供 | 没有公开转换器、运行时契约或具名模型矩阵；产品渠道面向 OEM |
| [Chengheng Micro](https://en.chenghengmicro.com/) | CH37 宣传 64 INT8 TOPS 和 FP16/FP32/FP64 模式；公司[2026 年大事记](https://chenghengmicro.com/about.html)称已进入小批量生产 | 未找到公开 SDK、编译器契约或具名模型验证；应视为商业化早期、无法自助获取的平台，而非可复现部署证据 |
| [Bouffalo Lab](https://github.com/bouffalolab/bouffalo_sdk) | BL808 包含 BLAI-100 NPU，且有当前官方 SDK 仓库 | 维护中的 SDK 没有公开当前官方 BLAI 转换/示例路径；现存流程属于历史或社区证据 |

这份候选清单有意以证据为依据。厂商若发布当前工具链或评估路径、目标专用产物契约，以及至少一个具名模型的端到端配方，就可以进入可部署平台表格。

## 本文有意未作的声明

本文没有说每个具名模型都能从任意上游仓库直接运行。厂商模型库中的模型经常经过修改、在不同输出节点切分，或搭配自定义后处理。

以下说法也不能单独构成支持声明：

- 编译器可以导入 ONNX。
- 芯片宣传带有 Android NNAPI 驱动。
- 经销商说开发板「兼容 YOLO」。
- 某社区仓库可以运行某个模型的一个分支。
- 模型编译没有报错。
- 厂商报告了 NPU 单独运行的 FPS，却没有精度结果。

Google Tensor 手机加速器和许多仅面向 OEM 的定制 ASIC 也真实存在，但 Google 并未开放 Tensor，无法像上文平台那样作为通用自助编译目标。公司存在、NPU 框图或 Android 加速功能，都不足以凭空作出 LibreYOLO 集成声明。

同样，Google TPU、AWS Inferentia/Trainium、Microsoft Maia 和仅用于数据中心的 AI 卡等云端加速器不在本文范围内。本指南关注计算机视觉开发者实际可能在边缘设备上部署的硬件。

模型许可是另一层问题。芯片厂商发布 YOLO 转换配方，并不会授予使用或再分发上游权重、训练代码或编译衍生物的权利。硬件支持、SDK 许可和模型许可都必须针对预期产品分别核查。

## 研究方法和更正

针对每家公司，研究都查找了四类一手资料：

1. 说明芯片的当前产品页面或数据手册。
2. 解释实际部署路径的编译器和运行时文档。
3. 列出视觉模型名称的模型库、兼容性表或端到端教程。
4. 说明开发者能否实际获取工具的生命周期或访问信号。

厂商的一手 GitHub 组织视为厂商来源。如果芯片公司自己没有发布相同资料，则会将开发板厂商文档标为生态证据。对比表排除了第三方性能声明。

随后，草稿经过了独立的厂商分组和跨表复核。这些检查重新验证了目标范围、产物名称、精度、模型与后处理证据、已发布与出货状态、生命周期标签、基准测试分母和所有实体数量。如果两个一手页面互相矛盾，本文会展示冲突，而不是暗中选取更方便的数据。

这个市场变化很快。如果你在其中一家厂商工作，发现芯片、SDK、模型清单或访问状态有误，请向 LibreYOLO 提供准确的公开文档 URL 和版本。应以一手证据支持的更正替换本文；没有文档的营销说法不应加入。

## 常见问题

### 有多少家 Edge AI NPU 公司？

没有统一的公司总数，因为产品厂商、IP 授权商、智能传感器、GPU 和私有汽车 ASIC 之间存在重叠。本指南并非穷尽性清单，截至 2026 年 8 月，共跟踪了 69 家拥有相关边缘 AI 芯片、加速平台、可授权 NPU IP，或具有可信候选芯片的公司及平台生态。

### 什么是 NPU？

神经网络处理器（NPU）是用于卷积、矩阵乘法等神经网络运算的专用硬件。厂商使用的名称包括 NPU、AIPU、BPU、KPU、DLA、HTP、TPU 和 MLA；编译后的模型通常无法在不同厂商之间移植。

### 哪些 NPU 公司官方支持 YOLO 模型？

Amlogic、Hailo、Axelera AI、Rockchip、高通、MediaTek、DEEPX、D-Robotics、AXERA、SOPHGO、Huawei、Cambricon、通过树莓派官方 AI Camera 仓库提供支持的 Sony IMX500、STMicroelectronics、Renesas、Lattice、Himax、Microchip 等厂商，都有至少一个 YOLO 代际的一手模型库条目、教程或验证结果。具体代际、任务、芯片目标和 SDK 版本仍然很重要。

### 任何 ONNX 模型都能在任何 NPU 上运行吗？

不能。ONNX 是模型交换格式，不是硬件兼容性保证。NPU 编译器必须支持计算图中的每个算子、张量形状和数据类型。静态形状、计算图切分、自定义算子和 CPU 回退都很常见。

### 哪款 NPU 最适合 YOLO？

没有适用于所有场景的赢家。Hailo、Rockchip、Axelera AI、DEEPX 和 Amlogic 的 YOLO 文档覆盖较充分，而高通的设备覆盖面很广。选择时应考虑量化后的精度、完整流水线延迟、持续功耗、价格、供货情况、SDK 获取方式和操作系统支持。

### 为什么 NPU 的 TOPS 数字不能直接比较？

厂商统计时可能采用不同精度、稀疏运算、乘加计数方式和峰值利用率假设。TOPS 不包含内存流量、不支持的算子、预处理、后处理和主机开销。应在相同模型、精度、分辨率、精度指标和完整流水线上进行比较。

### 什么是 NPU 校准？

校准是将有代表性、通常没有标签的部署图像输入模型，让编译器估算 INT8 或其他低精度格式的激活值范围。随机或缺乏代表性的图像仍可能生成编译产物，却会损害任务精度。

### LibreYOLO 支持边缘 NPU 吗？

LibreYOLO 可导出 ONNX 和多种运行时格式；部分 Rockchip 模型可直接通过 RKNN 编译器路径处理，并提供外部 Hailo 编译流程的文档。其他厂商原生格式仍需其厂商 SDK；在完成编译、验证并在硬件上运行之前，应将它们描述为集成候选方案。
