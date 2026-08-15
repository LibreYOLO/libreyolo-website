---
title: "69 Edge AI NPU Companies: Chips, SDKs & YOLO (2026)"
description: "A source-backed guide to 69 edge AI companies and NPU ecosystems: chips, SDKs, artifacts, quantization, and documented computer vision and YOLO support."
date: 2026-08-15
author: Xuban
tags: [edge-ai, npu, computer-vision, yolo, hardware, ai-accelerators, amlogic, hailo, rockchip]
faq:
  - q: "How many edge AI NPU companies are there?"
    a: "There is no universal company total because product vendors, IP licensors, smart sensors, GPUs and private automotive ASICs overlap. This non-exhaustive guide tracks 69 companies and platform ecosystems with relevant edge AI chips, accelerator platforms, licensable NPU IP or credible watchlist silicon as of August 2026."
  - q: "What is an NPU?"
    a: "A neural processing unit is specialized hardware for neural-network operations such as convolutions and matrix multiplication. Vendor terms include NPU, AIPU, BPU, KPU, DLA, HTP, TPU and MLA, and their compiled models are generally not portable between companies."
  - q: "Which NPU companies officially support YOLO models?"
    a: "Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 through Raspberry Pi's official AI Camera repository, STMicroelectronics, Renesas, Lattice, Himax, Microchip and several others have first-party model-zoo entries, tutorials or validated results for at least one YOLO generation. The exact generation, task, chip target and SDK version still matter."
  - q: "Can any ONNX model run on any NPU?"
    a: "No. ONNX is an interchange format, not a hardware compatibility guarantee. An NPU compiler must support every operator, tensor shape and data type in the graph. Static shapes, graph cuts, custom operations and CPU fallback are common."
  - q: "What is the best NPU for YOLO?"
    a: "There is no universal winner. Hailo, Rockchip, Axelera AI, DEEPX and Amlogic have strong documented YOLO coverage, while Qualcomm has exceptional device reach. Choose using accuracy after quantization, full-pipeline latency, sustained power, price, availability, SDK access and operating-system support."
  - q: "Why can NPU TOPS numbers not be compared directly?"
    a: "Vendors may count different precisions, sparse operations, multiply-accumulates and peak-utilization assumptions. TOPS omits memory traffic, unsupported operators, preprocessing, postprocessing and host overhead. Compare the same model, precision, resolution, accuracy and full pipeline."
  - q: "What is NPU calibration?"
    a: "Calibration runs representative, usually unlabeled deployment images through the model so the compiler can estimate activation ranges for INT8 or another low-precision format. Random or unrepresentative images may still produce a compiled artifact while damaging task accuracy."
  - q: "Does LibreYOLO support edge NPUs?"
    a: "LibreYOLO exports ONNX and several runtime formats, has a direct RKNN compiler path for selected Rockchip models, and documents the external Hailo compilation flow. Every other vendor-native artifact still requires its vendor SDK and should be described as an integration candidate until it is compiled, validated and run on hardware."
---

**There is no single NPU market. This non-exhaustive guide tracks 69 companies and platform ecosystems: 51 vendors with deployable chips or platforms, nine NPU IP suppliers, and nine clearly labeled watchlist companies.** They span several incompatible classes of accelerator and almost as many compiler stacks. Most promise the same path: export a model, quantize it, compile it, copy a proprietary artifact to a board, and call a vendor runtime. The details decide whether that path takes an afternoon or an engineering quarter.

This guide maps the companies that provide credible computer-vision hardware in 2026. It records the relevant chips, SDK and compiler names, deployment artifacts, public access level, and models the vendor actually documents. It also gives special attention to Amlogic, whose newer ADLA stack is materially different from the older A311D NPU ecosystem.

> **Research scope:** Sources were checked on August 15, 2026. Named-model claims come from first-party product pages, documentation, model zoos, repositories or vendor tutorials unless explicitly labeled otherwise. Advertised TOPS are vendor figures, not independently comparable benchmark results. This is a developer guide, not investment advice or a paid ranking.

**Quick navigation:** [company tables](#npu-companies-and-platforms-at-a-glance) | [Amlogic deep dive](#amlogic-two-npu-generations-not-one) | [vendor profiles](#the-strongest-public-deployment-ecosystems) | [compiled artifact table](#what-you-actually-deploy-the-artifact-lock-in-table) | [access and lifecycle](#tool-access-and-lifecycle-signals) | [LibreYOLO roadmap](#what-this-means-for-libreyolo)

## The most important finding

The hardware is fragmented, but the deployment pattern is remarkably consistent:

```text
PyTorch checkpoint
    -> ONNX or TFLite interchange graph
    -> representative calibration data
    -> vendor quantizer and graph compiler
    -> chip-specific binary or model package
    -> vendor runtime on the target
    -> application preprocessing, decode, NMS and rendering
```

[ONNX explicitly allows runtimes, code generators and hardware implementations](https://onnx.ai/onnx/repo-docs/IR.html), but an ONNX file is only the handoff. It does not mean that every ONNX operator will lower to every accelerator. Static input shapes, supported-operator constraints, quantization rules and host fallback determine what really runs on the NPU.

The software stack is therefore part of the chip. A nominally fast NPU with a gated or brittle compiler can be a worse deployment target than a smaller device with a public toolchain, model zoo, simulator and stable runtime.

## What "supported" means in this guide

Vendor literature uses the word *support* very loosely. This article separates four levels of evidence:

| Evidence level | What it proves | What it does not prove |
|---|---|---|
| **Validated model** | The vendor publishes a chip-specific model entry, result or compatibility table | Your modified checkpoint retains the same accuracy |
| **Official example** | The vendor provides an end-to-end tutorial or demo for a named model | Other sizes, tasks or generations compile |
| **Compiler capability** | The SDK imports a framework or exposes supported operators | A particular YOLO graph works end to end |
| **Marketing or gated** | The product is intended for vision and a private SDK exists | Publicly reproducible model compatibility |

This distinction matters. "Imports ONNX" is not equivalent to "supports YOLO11 segmentation." A model can parse but fall back to a CPU, compile with incorrect output numerics, overflow accelerator memory, or lose accuracy during quantization.

## NPU companies and platforms at a glance

The tables intentionally mix SoCs, discrete accelerators, smart sensors and adjacent GPU/FPGA targets. They compete for the same deployment decision even when the vendors use names such as NPU, AIPU, BPU, KPU, DLA, HTP, TPU or MLA.

### Embedded SoCs, smart sensors and industrial processors

| Company | Relevant chips or platforms | SDK, compiler and artifact | Publicly documented computer-vision evidence | Access |
|---|---|---|---|---|
| [Amlogic](https://github.com/Amlogic-NN/amlnn-toolkit) | A311D2, S928X, S905X5/S905D5, A311Y3, C308L/C302X/C302X2, T968D4, C305X2 and A123X | AMLNN Toolkit and `libnnsdk.so`, compiled `.adla`; separate legacy Acuity `.nb` stack for A311D | The [model playground](https://github.com/Amlogic-NN/amlnn-model-playground) documents YOLOv5/6/7/8/10/11, YOLOX, YOLOE, YOLO-World, PP-YOLOE, segmentation, pose and OBB on A311D2, S905X5, A311Y3, C305X2 and A123X; the other chips listed here are compiler targets, not entries in that support matrix | Public GitHub and wheels; compatible BSP/driver still required |
| [Rockchip](https://github.com/airockchip/rknn-toolkit2) | RK3562/3566/3568, RK3576, RK3588, RV1126B | RKNN-Toolkit2, RKNN Runtime, `.rknn` | YOLOv5/6/7/8/10/11, YOLOX, YOLO-World, PP-YOLOE, segmentation, pose and OBB in the [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) | Public GitHub; binary wheels |
| [Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | Snapdragon and Dragonwing platforms, QCS6490, QCS8550 and Dragonwing IQ-9075 | QAIRT/QNN, SNPE and AI Hub; QNN context binaries or DLC | The [AI Hub model collection](https://github.com/qualcomm/ai-hub-models) includes YOLOv3/5/6/7/8/9/10/11/26, YOLOX, YOLO-World, YOLOR, RF-DETR and detection/segmentation/pose models | Docs public; SDK/account requirements vary |
| [Texas Instruments](https://github.com/TexasInstruments/edgeai-tidl-tools) | AM62A, AM67A, AM68A, AM69A and TDA4x; preview TDA54-Q1 | Processor SDK Edge AI and TIDL | The [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo) publishes optimized detection, segmentation, pose and classification models for current AM6xA/TDA4 paths; this is not yet a target matrix for preview TDA54-Q1 | Public GitHub plus Processor SDK |
| [NXP](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | i.MX 8M Plus, i.MX 93, i.MX 95 and acquired Kinara Ara-1/Ara240 | eIQ with TIM-VX, Vela, Neutron Converter and Ara SDK | The [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) includes YOLOv4-tiny/v8, NanoDet, CenterNet, FastestDet, SSD Lite and YOLACT assets or recipes; its YOLOv5 entry is documentation-only | Mixed public and account-gated components |
| [STMicroelectronics](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) | STM32N6x7 variants, including STM32N657/647, with Neural-ART accelerator | STM32Cube AI Studio and ST Edge AI Core | Current services repository lists Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 and ST-YOLOX, plus pose and segmentation | Public tools and repositories |
| [Infineon](https://documentation.infineon.com/psocedge/) | PSOC Edge E83/E84 with Cortex-M55 plus Ethos-U55; separate NNLite accelerator on the M33 side | ModusToolbox, [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter), TFLite Micro and Arm Vela | Architecture material names MobileNetV1/V2 as representative kernels and the E84 AI kit includes a camera, but no first-party PSOC Edge YOLO validation matrix was found | Public documentation, SDK components and current E84 evaluation kits |
| [Renesas](https://github.com/renesas-rz/rzv_drp-ai_tvm) | RZ/V2L, V2M, V2MA, V2H and V2N | DRP-AI Translator, DRP-AI TVM and RUHMI | The [RZ/V2H model list](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) validates YOLOv5/v8/v11/26 sizes, YOLOX, pose and segmentation variants | Public GitHub plus board SDK |
| [Sony](https://www.aitrios.sony-semicon.com/edge-ai-devices/imx500) | IMX500 intelligent vision sensor and Raspberry Pi AI Camera | Edge-MDT, Model Compression Toolkit and IMX500 packer; `.rpk` package | The [Raspberry Pi IMX500 zoo](https://github.com/raspberrypi/imx500-models) includes YOLOv8n, YOLO11n, EfficientDet Lite, NanoDet+ and SSD | Public Raspberry Pi flow; Sony tooling terms apply |
| [Ambarella](https://www.ambarella.com/developer/model-garden/) | CV72/CV75, CV5/CV52 and CV3-AD families | Cooper Developer Platform, CVflow compiler and runtime DAGs | Public model garden names YOLOX-S, RTMDet-nano, DeepLabV3+, TopFormer, OWL-ViT and LLaVA OneVision | Primarily partner-gated |
| [Synaptics](https://developer.synaptics.com/docs/sl/overview) | Astra SL1600/SL1680; SL2611/13/15/17/19; separate SR100 MCU family | SyNAP `.synap` on SL16xx; Torq/IREE `.vmfb` on SL261x; separate Ethos-U55-oriented SR SDK | Official YOLOv8n/v8s SyNAP benchmark and SL261x Torq YOLOv8 example; SR evidence must be evaluated separately | Developer portal; some downloads gated |
| [MediaTek](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) | Genio 360/360P/420/520/720 with NP8 and MDLA 5.3; Genio 510/700/1200 with older NP6/MDLA | NeuroPilot Converter, `ncc-tflite`, Neuron Runtime and `.dla`; ONNX Runtime path on selected newer systems | The official IoT AI Hub publishes model pages and benchmarks for YOLOv5 and YOLOv8, plus classification and face models | Public Yocto docs; key NP8 bundles and Android material require direct-customer/NDA access |
| [Allwinner](https://docs.aw-ol.com/v853/en/npu/dev_npu/) | V853 vision SoC with 1-TOPS Vivante NPU | Acuity/Pegasus conversion and `viplite` runtime | Official V853 material names YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet and face/person networks | Public docs; SDK package download may require account |
| [Canaan/Kendryte](https://github.com/kendryte/nncase) | Current K230/K230D; legacy K210/K510 KPU | `nncase` compiler and runtime; `.kmodel` | The K230 nncase guide covers YOLOv5s compilation/simulation/runtime; the official demo catalog and current [CanMV changelog](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) document YOLOv8/11/26 tasks | Public compiler/PyPI; chip plug-in is binary |
| [Alif Semiconductor](https://alifsemi.com/support/kits/ensemble-e7appkit/) | Ensemble E7 with two ML accelerators including Ethos-U55; [E8](https://alifsemi.com/ensemble-e8-series/) with one Ethos-U85 and two Ethos-U55 NPUs | Arm Vela and TFLite Micro-based deployment | Alif publishes a family-level INT8 YOLO-Fastest face-detector benchmark, but no broad target-by-target modern-YOLO matrix | Public docs and evaluation kit resources |
| [Himax](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) | HX6538 WiseEye2 endpoint AI MCU with Cortex-M55 and Ethos-U55 | TFLite Micro plus Arm Vela, with CMSIS-NN and reference-kernel fallback | The official [WiseEye2 examples](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) include YOLOv8n detection, pose and classification, YOLO11n detection, face mesh and PeopleNet | Public GitHub, documentation and purchasable partner board |
| [Analog Devices](https://www.analog.com/en/products/max78002.html) | MAX78000 and MAX78002 AI MCUs with low-power CNN accelerators | [`ai8x-training`](https://github.com/analogdevicesinc/ai8x-training), `ai8x-synthesis`/`izer` and MSDK; generated C and weights | Official material covers face identification/detection, RetinaNet, Visual Wake Words, classifiers and action recognition; no first-party YOLO deployment was found | Public GitHub tools, documentation and evaluation boards |
| [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo) | RDK X3/X5/Ultra and newer S100-series BPU boards | OpenExplorer/Algorithm Toolchain and `hbm_runtime`; X5 `.bin`, current `rdk_s` `.hbm` | The current `rdk_x5` branch documents YOLOv5/v5u/v8/v9/v10/11/12/13/26, YOLOE, YOLO-World, segmentation, pose, classification, OCR and CLIP for RDK X5; X3 and S-series use separate branches | Public GitHub; some toolchain packages are platform-specific |
| [AXERA](https://github.com/AXERA-TECH/ax-samples) | AX650, AX637, AX630C, AX620Q and AX615 | Pulsar2 compiler and AXEngine; `.axmodel` | Current official samples cover YOLOv5/6/7/8/9/10/11/13/26, YOLOX and YOLO-World; tasks and targets vary by chip | Public samples and docs; compiler distributed separately |
| [SOPHGO](https://github.com/sophgo/tpu-mlir) | BM1684/1684X/1688/1690 and CV186X/CV18xx | TPU-MLIR and SOPHON runtime; `.bmodel` or `.cvimodel` | Official demos cover YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, OBB, segmentation, face, SAM and OCR | Open-source compiler plus vendor runtime |
| [Huawei Ascend](https://www.hiascend.com/en/software/cann) | Ascend 310/310P/310B and Atlas 200I/300I edge products | CANN, ATC compiler and AscendCL; `.om` | The official [Ascend ModelZoo](https://github.com/Ascend/modelzoo) includes YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN and segmentation models | Public docs; CANN packages and kernels must match target |
| [Cambricon](https://github.com/Cambricon/magicmind_cloud) | MLU370-X4/S4 in the cited public matrix; MLU270 is legacy hardware not covered by that matrix | Neuware and MagicMind | The repository's MagicMind 1.7 matrix lists YOLOv3/4/5/7/8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN and segmentation models | Historical model examples public; current SDK/container access gated |
| [Sunplus](https://sunplus.atlassian.net/wiki/spaces/C3/pages/1971126471/SP7350%2BSpecification) | SP7350/C3V, approximately 4.1 to 4.6 advertised TOPS | Vivante Acuity NPU Docker, SNNF runtime and `.nb` | Official documentation provides YOLOv5 and a custom end-to-end [YOLOv8 deployment](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) | Public source/docs and board ecosystem |
| [ESWIN Computing](https://www.eswincomputing.com/en/news/info/96.html) | EIC7700/EIC7700X and dual-die [EIC7702/EIC7702X](https://www.eswincomputing.com/en/news/info/104.html) RISC-V SoCs | ENNP: EsQuant, current EsAAC compiler, simulator and ESSDK runtime; `.model`; older docs used `ennc-compile` | Milk-V-hosted ENNP documentation includes an end-to-end [YOLOv3 tutorial](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3), MobileNetV2 and ResNet examples | Mixed first-party and board-hosted docs; regional downloads |
| [Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m55-mcus/m55m1-series/index.html) | M55M1 MCU with Ethos-U55-256 | TFLite Micro, Arm Vela and NuEdgeWise/NuML | Official [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) examples name YOLOv8-nano, YOLOX-nano, YOLO Fastest, SSD-MobileNet and classification models | Mostly public MCU toolchain |
| [Realtek](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | AmebaPro2 RTL8735B / AMB82-mini camera platform | Arduino/FreeRTOS SDK, VoE and NeuralNetwork API; `.nb` | Packaged models include YOLOv3-tiny, YOLOv4-tiny, YOLOv7-tiny, SCRFD and MobileFaceNet | Runtime public; [custom converter access](https://ameba-doc-arduino-sdk.readthedocs-hosted.com/en/latest/FAQ/offline_ai_model_conversion_steps.html) requires contact |
| [Telechips](https://docs.topst.ai/product/p/ai) | TCC7500 / TOPST AI board, advertised 8 TOPS | TC-NN-Toolkit / Enlight SDK; `.enlight` intermediate and compiled deployment bundle | An [official TOPST project](https://docs.topst.ai/blog/31) deployed YOLOv8s; its UFLD v1/v2 conversions failed on unsupported layers and only a split/postprocessing workaround was proposed. [YOLOv4](https://community.topst.ai/t/segmentation-fault/358) and [another YOLOv8 flow](https://community.topst.ai/t/segmentation-fault-error/415) appear in community support threads | Board docs public; compiler/operator downloads often permission-gated |
| [T-Head](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | TH1520, advertised 4-TOPS INT8, on LicheePi 4A | HHB compiler and CSI-NN2/SHL; `hhb.bm` plus generated code/parameters | Official board-vendor examples run YOLOv5n/s and MobileNetV2 on the NPU | Public examples; aging toolchain/maintenance uncertainty |
| [SigmaStar](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | Current SSU9383CM smart-camera SoC and older IPU families | Current MI_IPU runtime; an [older SGS_IPU toolchain](https://wx.comake.online/doc/doc/SigmaStarDocs-SSC9381G_9351_Pudding-ULS00V040-20210913/customer/development/dla/tools.html) produced `.sim` to `sgsimg.img` | An [older official SDK postprocessor](https://wx.comake.online/doc/doc/Sigmastar_SDK_v1.2.2/module/User_Guide/Common/SigmaStar_Post_Processing_Module.html) names SSD and YOLOv1/2/3; public compatibility of those models and artifacts with SSU9383CM is unverified | Current silicon, but publicly cited compiler/model evidence is generation-split |
| [HiSilicon](https://www.hisilicon.com/cn/products/smart-vision/machine-vision/hi3516cv610) | Hi3516CV610/DV500, Hi3519DV500 and Hi3403V100 smart-vision SoCs | ATC-to-`.om` with NNN/SVP-NNN board runtime | The target-specific HiSpark matrix, notably for Hi3403 and Hi3591P, names YOLOv3 through YOLO11, pose, segmentation, OBB, OCR and depth entries; it is not validation across every SoC in this row | Active and distinct from Ascend; repository models are labeled non-commercial-use only |

### Dedicated edge accelerators and modules

| Company | Relevant silicon | SDK and artifact | Publicly documented model evidence | Access |
|---|---|---|---|---|
| [Hailo](https://github.com/hailo-ai/hailo_model_zoo) | Hailo-8, Hailo-8L, Hailo-10H and Hailo-15 family | Dataflow Compiler and HailoRT; `.hef` | YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 and YOLO26, plus YOLOX, DAMO-YOLO, SSD, EfficientDet, segmentation, pose and OBB, with generation-specific model tables | Model Zoo public; compiler through Developer Zone |
| [Axelera AI](https://docs.axelera.ai/sdk/reference/models/model-zoo/) | Shipping Metis products; announced Europa and Titania portfolio products | Public Voyager SDK; alpha Pipeline Builder `.axm`/`.axe`, classic `.axmodel` bundles | Validated YOLOv3/5/7/8/9/10/11/26, YOLOX, YOLO-NAS, OBB, pose, segmentation and many non-YOLO models | SDK public on GitHub; customer support account-gated |
| [DEEPX](https://developer.deepx.ai/modelzoo/) | DX-M1 and DX-M1M | DXNN SDK, DX-COM and DX-RT; `.dxnn` | Model Zoo listed 354 entries under DX-COM 2.4.0/DX-RT 3.4.0 when checked on August 15, 2026, including YOLOv3 through YOLO11 and YOLO26, YOLOX, SSD, EfficientDet, NanoDet, segmentation, pose and OBB | Public developer resources and suite |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | MX3 accelerator and multi-chip modules | MemryX SDK, Neural Compiler and runtime; `.dfp` | Official examples and release notes cover detection, segmentation and pose, including YOLOv10, YOLO11 and YOLO26 | Public documentation and SDK |
| [Kneron](https://doc.kneron.com/docs/) | KL520, KL530, KL630, KL720 and KL730 have current compiler docs; KL830 appears in some PLUS APIs but not the current compiler target list | Kneron PLUS and Model Toolchain; `.nef` on documented compiler targets | Official [YOLO workflow](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) centers on Tiny-YOLOv3; other materials cover YOLOv5 | Public docs; tool packages/downloads vary by chip |
| [SiMa.ai](https://docs.sima.ai/pages/palette/modelsdk.html) | MLSoC and production Modalix 50-TOPS platform | Palette, ModelSDK, MLA Compiler and ModelExecutor | Public releases validate YOLOv7/v8, YOLOX, pose/segmentation, DETR, Mask R-CNN and EfficientDet | Documentation public; product SDK is commercial |
| [EdgeCortix](https://www.edgecortix.com/en/hardware) | SAKURA-II, advertised 60-TOPS accelerator in M.2 and PCIe products | MERA compiler and framework | Vendor describes vision-to-generative-AI support, but does not publish a sufficiently precise current YOLO compatibility matrix | Commercial trial/order inquiry; partner-led validation |
| [BrainChip](https://brainchip.com/metatf-dev-tools/) | Shipping AKD1500 co-processor/M.2 module; legacy AKD1000 platforms; Akida 2 IP | MetaTF packages: `akida-models`, `quantizeml`, `cnn2snn` and `akida` runtime | The Akida 2 model card names an AkidaNet0.5 YOLOv2 detector, CenterNet, AkidaUNet and face recognition; those results are not automatically AKD1500 validation | Core Python packages public; hardware mapping remains target-specific |
| [Blaize](https://www.blaize.com/products/) | P1600, Pathfinder and Xplorer products | Picasso SDK, NetDeploy and AI Studio | Vision is a target market, but no auditable public named-model compatibility matrix was found | Commercial/gated |
| [Google Coral](https://github.com/google-coral/edgetpu) | Legacy Edge TPU USB, PCIe, M.2 and Dev Board products; separate active open-source [Coral NPU IP](https://github.com/google-coral/coralnpu) | Legacy Edge TPU Compiler/`libedgetpu`/PyCoral; new RISC-V Coral NPU exposes IP, RTL/simulation and ELF examples | Legacy examples emphasize SSD MobileNet, classification, DeepLab and MoveNet; the new IP has no public YOLO deployment matrix and is not a drop-in Edge TPU product successor | Legacy core repos archived; new Coral NPU IP actively developed |
| [Lattice Semiconductor](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E and Avant-X FPGAs | sensAI Studio, Neural Network Compiler and configurable accelerator IP | The current compiler table lists YOLOv1, YOLOv5, YOLOv8 and YOLO11 in target-specific modes, plus SSD, MobileNetV2-SSD, ResNet and ENet | Compiler/manuals public; IP terms vary, and Advanced CNN Accelerator is commercial |
| [Mobilint](https://www.mobilint.com/sdk-qb) | REGULUS 10-TOPS and ARIES MLA100 80-TOPS accelerators | qb SDK; INT8 compiler and `.mxq` | Public [model zoo](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) lists YOLOv3/5/7/8/9/10/11/12/26 detection plus segmentation, pose and OBB variants | Public docs/models; SDK and hardware are commercial |
| [Rebellions](https://rebellions.ai/developers/) | Active ATOM+ CA22 and ATOM-Max CA25; EoL ATOM CA02/ATOM+ CA12; ATOM-Lite CA21 tool status unclear | RBLN SDK compiler/runtime/profiler; `.rbln` | Official support releases name YOLOv3, YOLOv5/6/7/8 families and current YOLOv8 tutorials | Docs public; compiler wheel requires portal credentials |
| [FuriosaAI](https://furiosa.ai/warboy) | Vision-oriented Warboy Gen1; separate RNGD generation | Furiosa SDK; INT8 `.enf` on Warboy, separate `.fxb` on RNGD | Warboy zoo documents SSD, YOLOv5M/L and YOLOv7-w6-pose; RNGD's roadmap marks YOLOv8m support completed in 2024 Q4 but exposes no detailed current public vision matrix | IAM/account access; keep the two generations separate |

### Adjacent platforms developers compare with NPUs

| Company | Hardware | Deployment stack | Why it belongs in the comparison |
|---|---|---|---|
| [NVIDIA](https://github.com/NVIDIA-AI-IOT/deepstream_tools/tree/main/yolo_deepstream) | Jetson Orin GPU plus DLA; Jetson Thor GPU without DLA | JetPack, TensorRT and DeepStream; `.engine` | Extremely mature vision deployment; official DeepStream tools document YOLOv4/v7/v8/v9/11, including a YOLO11 OBB configuration. Many results use the GPU, and unsupported Orin DLA layers can fall back to GPU. |
| [AMD](https://vitisai.docs.amd.com/en/6.2/) | Kria/legacy DPU targets; Vitis AI 6.2 GA for Versal AI Edge Gen1 VEK280/VE2802 and Gen2 VEK385; separate Ryzen AI client NPUs | Legacy `.xmodel`; Gen1 target-bound NPU snapshot; Gen2 compiled cache directory or production `.rai` package | Vitis AI publishes vision material, but legacy DPU, Versal Gen1, Versal Gen2 and Ryzen AI are separate compilation and deployment contracts. |
| [Microchip](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox) | PolarFire SoC FPGA with configurable CoreVectorBlox accelerator IP | Public VectorBlox SDK 3.1 and Libero; `.vnnx`, `.hex` and `.ucomp` deployment assets | Current [tutorials](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) cover YOLOv5n, YOLOv8 detection/classification/OBB/pose/segmentation, YOLOv9t and other vision models; SDK 3.1 is currently scoped to the PolarFire SoC Video Kit. |
| [Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) | Core Ultra NPU, CPU and integrated/discrete GPU | OpenVINO IR or compiled model cache | OpenVINO provides a public cross-XPU path; use the [verified-model matrix](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html) NPU columns rather than assuming every OpenVINO YOLO example is NPU-validated. |
| [Apple](https://developer.apple.com/machine-learning/core-ml/) | Apple Neural Engine in A11-and-newer A-series chips and M-series chips | Core ML and `coremltools`; `.mlpackage`/`.mlmodelc` | A highly accessible consumer NPU through Core ML, but Apple abstracts the exact CPU/GPU/Neural Engine partition rather than exposing a YOLO-specific hardware compiler. |
| [GreenWaves Technologies](https://github.com/GreenWaves-Technologies/nn_menu_gap9) | GAP9 with NE16 neural engine | GAP SDK, NNTool and AutoTiler-generated code | Official repositories include MobileNet SSD, face detection, classification and a dedicated [YOLOX people detector](https://github.com/GreenWaves-Technologies/yolox_people_detection). The full SDK is limited to qualified customers. |
| [Syntiant](https://www.syntiant.com/hardware) | Mass-production NDP200 and sampling NDP250 Neural Decision Processors | Syntiant SDK and deployment packages | Ultra-low-power always-on vision/sensor processors: NDP200 is documented below 1 mW, while NDP250 image recognition is documented below 30 mW. No broad public YOLO matrix was found. |

## Amlogic: two NPU generations, not one

Amlogic deserves a deeper explanation because information on the web often combines incompatible products. The company has an older VeriSilicon-based path and a newer proprietary ADLA path. They do not use the same compiler, artifact or runtime.

| Generation | Representative chips | NPU and toolchain | Compiled artifact | Practical status |
|---|---|---|---|---|
| Legacy | A311D and S905D3, commonly seen on Khadas VIM3/VIM3L | VeriSilicon Vivante VIPNano-QI, Acuity toolkit, KSNN and older `aml_npu_sdk` | `.nb` inside an `nbg_unify` output directory | Existing boards and demos; separate legacy integration |
| Current ADLA2 | C308L/C302X, S928X, A311D2, T968D4, S905X5/S905D5 and C302X2 | Amlogic ADLA2, AMLNN Toolkit and NNSDK2 | Target-specific `.adla`; W8A8 or W8A16 | Integer-focused current stack |
| Current ADLA3 | A311Y3, C305X2 and A123X | Amlogic ADLA3, AMLNN Toolkit and NNSDK2 | Target-specific `.adla`; W4A8, W8A8, W4A16, W8A16 or W16A16 | Adds native INT4, FP16 and BF16 capabilities |

The [A311D datasheet](https://dl.khadas.com/products/vim3/datasheet/a311d-datasheet.pdf) identifies the original 5-TOPS INT8 NPU. The older [Khadas VIM3 application page](https://docs.khadas.com/products/sbc/vim3/npu/npu-app) and [NPU overview](https://docs.khadas.com/products/sbc/vim3/npu/start) document DenseNet CTC, MTCNN, RetinaFace, YOLOFace, YOLOv2, YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv7-tiny and YOLOv8n; the overview additionally documents YOLOv8n-pose and marks FaceNet deprecated. Those examples are real, but they are evidence for the old Acuity `.nb` ecosystem, not for current ADLA chips. A311D is not A311D2, and C305X is not C305X2.

There is a second hardware trap. The [Khadas VIM4 revision guide](https://docs.khadas.com/products/sbc/vim4/configurations/identify-version) says the original V12/A311D2 revision-B board has no NPU; the advertised 3.2-TOPS NPU appears on V13A and later boards using the A311D2-N0D revision-C part. A product name alone is not enough to select a test device.

### The current AMLNN and ADLA stack

The current [AMLNN Toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/7d3cc9a36179b756ce79c953c14d43acab1f2af5) covers model conversion, quantization, compilation, inference and profiling. Its pinned [May 2026 user guide](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/02_Amlogic_NPU_User_Guide_V0.1.pdf) documents importers for ONNX, floating-point or quantized TFLite, TorchScript `.pt`, and PyTorch 2 ExportedProgram/PT2. TensorFlow, Paddle and Keras paths are marked as planned in the detailed guide even though the repository overview uses broader wording. The compiler emits `.adla`. Python deployment uses AMLNN or `amlnn_edge_toolkit_lite`; native C/C++ links `libnnsdk.so` through `nnsdk2.h`. Host-side development can use ADB with `nnserver`; the runtime also targets Android, Buildroot, Yocto and selected Debian/Armbian environments.

The toolkit's published platform identifiers currently include:

| Target ID | Platform |
|---:|---|
| `001` | C308L / C302X |
| `002` | S928X |
| `003` | A311D2 |
| `004` | T968D4 |
| `005` | S905X5 / S905D5 |
| `006` | C302X2 |
| `007` | A311Y3 |
| `008` | C305X2 / A123X in the detailed documents and model matrix |

That target field is not cosmetic. Amlogic requires the `.adla`, `nnsdk2.h`, `libnnsdk.so`, BSP and compiler/runtime generation to match. The pinned [Quick Start](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/01_Amlogic_NPU_Quick_Start_guide_V0.1.pdf) calls for ADLA driver 2.0.2 or newer, NNSDK 3.0.0 or newer, and NNSDK2 1.0.0 or newer; the [Android deployment guide](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/06_Amlogic_NPU_Android_Deployment_Guide_V0.1.pdf) makes the library, header, driver and BSP relationship concrete. Treat `.adla` as an ABI-bound build artifact, not a portable model.

Quantization also depends on the NPU generation. ADLA2 targets 001 through 006 document W8A8 and W8A16 compilation. ADLA3 targets 007 and 008 add W4A8, W4A16, W8A8, W8A16 and W16A16 combinations, with native INT4, FP16 and BF16 support in the [operator guide](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/05_Amlogic_NPU_Support_Operator_List_V0.1.pdf). Amlogic recommends 200 to 500 representative calibration samples. Its random-data option is explicitly for performance testing, not accuracy-preserving quantization.

### Amlogic's documented model coverage

The [pinned Amlogic model playground](https://github.com/Amlogic-NN/amlnn-model-playground/blob/03ca63ef20c4f1560722de26ab910826884f334a/README.md) is much stronger evidence than a generic claim of ONNX support. Its published support/example matrix names A311D2, S905X5, A311Y3, C305X2 and A123X. Compiler acceptance of the other target IDs does not prove that this entire model list was validated on them.

| Task | Documented models |
|---|---|
| Classification | MobileNetV2 and ResNet50-v2; DINO on ADLA3 |
| Object detection | PP-YOLOE, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv10, YOLO11, YOLOE, YOLO-World, YOLOX and QR detection |
| Face and gesture | RetinaFace and Gesture Recognition |
| Segmentation | DeepLabV3, PP-LiteSeg, YOLOv5-seg and YOLOv8-seg |
| Oriented detection | YOLOv8 OBB |
| Pose | BlazePose detector/landmark and YOLOv8 pose |
| OCR and speech | LPRNet, PaddleOCR variants, Whisper Tiny and SenseVoice on selected newer chips |
| Newer transformer and multimodal work | DETR, MobileSAM, CLIP/MobileCLIP and selected low-bit LLM/VLM examples on newer platforms |

The same repository publishes the following W8A8 model-runtime figures. These are vendor measurements of the neural network on the NPU, not complete camera pipelines.

| Model and input | S905X5 | A311D2 | A311Y3 |
|---|---:|---:|---:|
| YOLOv8n, 640 x 640 | 101.72 FPS | 95.14 FPS | 191.06 FPS |
| YOLOv8s, 640 x 640 | 42.33 FPS | 42.77 FPS | 83.08 FPS |
| YOLOv8m, 640 x 640 | 19.67 FPS | 19.82 FPS | 35.30 FPS |
| YOLOv8l, 640 x 640 | 10.53 FPS | 10.12 FPS | 18.37 FPS |
| YOLO11n, 640 x 640 | 41.14 FPS | 41.48 FPS | 62.24 FPS |

The caveats are unusually important. Amlogic's operator guide places NMS in software on both ADLA2 and ADLA3. The README defines these as native NPU model-runtime results and excludes preprocessing and postprocessing; whether every memory transfer is included is not disclosed. The YOLO accuracy rows use 300-image COCO subsets rather than the complete validation set, while the classification rows use ImageNet `val1000`. The tables also report a different A311Y3 YOLOv8n latency from the FPS table under conditions the repository does not reconcile. Clocks, power mode, cooling, thermal steady state, exact SDK/BSP version and test duration are not disclosed. Use the figures to understand one vendor stack, not to rank it against another.

### Why Amlogic is strategically interesting

Amlogic combines four unusual properties: a large embedded SoC footprint, a newly public compiler stack, a current model matrix that overlaps heavily with LibreYOLO's task space, and relatively little first-class integration in mainstream training libraries. The current repositories are also young: they appeared publicly in late 2025 and early 2026, with substantive manuals dated May through July 2026. That creates a genuine first-mover opportunity and a corresponding API-stability risk.

A clean integration should use LibreYOLO's deterministic ONNX export as compiler input, require representative calibration images for low-precision builds, produce `.adla` plus a metadata manifest, and load through an NNSDK2 adapter. Legacy A311D should use a visibly different `.nb` target because presenting `.nb` and `.adla` as one backend would create silent compatibility failures. Amlogic's repositories carry Apache-2.0 top-level licenses, but redistribution rights for bundled wheels, shared libraries, `nnserver` and Android AAR binaries should be confirmed directly before LibreYOLO mirrors them.

## The strongest public deployment ecosystems

The following companies currently provide the clearest combination of obtainable hardware, public technical material and named-model evidence. That does not make them universally faster. It makes their compatibility claims easier to evaluate before buying hardware.

### Hailo

Hailo is the reference example of a dedicated vision accelerator ecosystem. Models are parsed into a Hailo Archive, optimized and quantized using representative images, then compiled into a Hailo Executable Format (`.hef`) file. Applications load that HEF through HailoRT.

The [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) contains recipes, pretrained models, postprocessing configurations and performance data for detection, segmentation, classification, pose and other tasks. Its YOLO coverage includes YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 and YOLO26 alongside YOLOX, DAMO-YOLO, SSD and EfficientDet. The version-pinned [Hailo-8 object-detection table](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.19.0/docs/public_models/HAILO8/HAILO8_object_detection.rst) is a better compatibility source than a generic product page, while the [standalone application](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/README.md) documents deployable YOLO pipelines.

There is an important version boundary. Hailo-8 and Hailo-8L use the older Model Zoo 2.x, Dataflow Compiler 3.x and HailoRT 4.x line, while Hailo-10 and Hailo-15 use the current 5.x generation. Recipes, parser behavior and HEFs are not interchangeable just because every device carries the Hailo name.

Hailo-15 also has a distinct application layer. It uses the Vision Processor Software Package and Hailo Media Library rather than the Hailo-8/10H TAPPAS-style host path. The public [`hailo-camera-apps`](https://github.com/hailo-ai/hailo-camera-apps) reference repository was archived on May 3, 2026, while [`hailo-apps-core`](https://github.com/hailo-ai/hailo-apps-core) remains a separate public application framework. A repository archive is not proof of product EOL, but Hailo-15 projects should confirm the current application package in the Developer Zone.

LibreYOLO's [Hailo deployment guide](/docs/export/hailo) deliberately stops at a static ONNX handoff because the proprietary compiler cannot be bundled as a Python dependency. That is the honest boundary between producing a suitable graph and claiming a tested HEF.

### Rockchip

Rockchip has one of the largest hobbyist and commercial embedded Linux footprints, particularly through RK3588, RK3576 and RK356x boards. The public [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) converts and quantizes models on an x86 Linux host, and RKNN Runtime or Toolkit-Lite executes the resulting `.rknn` file on the board.

The [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) is unusually explicit about chips, model families and precision. Its current table lists FP16 and INT8 paths for YOLOv5/6/7/8/10/11, YOLOX, PP-YOLOE, YOLO-World and YOLOv5/v8 segmentation variants; YOLOv8 OBB and YOLOv8 pose are listed as INT8-only. It also covers OCR, face and other segmentation networks.

LibreYOLO already has a conservative direct [RKNN exporter](/docs/export/rknn). It validates four exact detection variants on RK3588, can compare the compiler simulator with ONNX Runtime, and rejects unvalidated families rather than equating a successful build with correct predictions. That narrow support claim is a useful model for every future NPU backend.

### Axelera AI

Axelera AI, often misspelled "Accelera," builds the Metis AIPU and sells it in M.2, PCIe and multi-chip products. Metis is the publicly orderable family; Europa and Titania are announced portfolio products, so availability should not be flattened into one product status. The [Voyager SDK is public on GitHub](https://github.com/axelera-ai-hub/voyager-sdk) and handles model selection, compilation, quantization, execution and application pipelines; customer support remains account-gated.

The public [Voyager Model Zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo/) is among the most informative in the market. It lists model variant, input size, precision, accuracy and measured performance for YOLOv3, YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 n/s/m/l/x, YOLOX, YOLO-NAS, OBB, pose and segmentation, together with many classification and dense-prediction models. Publishing quantization loss and accuracy next to speed is more useful than publishing a peak TOPS number alone.

Artifact naming changed with the API generation. The alpha [Pipeline Builder compilation flow](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) emits an `.axm` model and can package a portable pipeline as `.axe`. The [classic pipeline documentation](https://docs.axelera.ai/sdk/reference/pipeline/model-formats/) describes `.axmodel` plus model metadata and a manifest. An integration must detect the installed Voyager generation instead of hard-coding one extension.

### Qualcomm

Qualcomm offers enormous deployment reach, but "Qualcomm NPU" hides several interfaces. Developers encounter the Hexagon HTP through QAIRT/QNN, the older SNPE flow, Qualcomm AI Hub compilation services, LiteRT or ONNX Runtime's QNN execution provider depending on the device and product class.

The official [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) repository is strong model-level evidence. It publishes optimized packages for YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 detection/segmentation/pose, YOLOX, YOLO-World, YOLOR, RF-DETR and many classification, depth and pose models. AI Hub also exposes device-specific profiling, which is important because a model compiled for one HTP generation is not automatically portable to another.

The main integration cost is matrix size: Android versus embedded Linux, QCS ordering parts versus consumer Snapdragon, HTP architecture, QNN/QAIRT version and quantization scheme all matter. Qualcomm's own current pages disagree on [Dragonwing IQ-9075](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075) status: the product page labels it Active and its [EVK](https://www.qualcomm.com/developer/hardware/qualcomm-iq-9075-evaluation-kit-evk) is evaluable, while the [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) still labels IQ-9075 Sampling and lists longevity through 2038. The ordering/SKU prefix is QCS9075; Qualcomm advertises 50- and 100-dense-INT8-TOPS configurations plus Ubuntu/Yocto support. Production status should be confirmed for the exact ordering SKU, and a LibreYOLO integration should record that SKU rather than producing a generic folder called "Qualcomm."

### DEEPX

DEEPX's DX-M1/DX-M1M accelerators use the DXNN SDK. DX-COM compiles and quantizes the model, DX-RT executes it, and the deployed artifact uses the `.dxnn` format. The company publishes the [DX-AllSuite](https://github.com/DEEPX-AI/dx-all-suite), application examples in [DX-APP](https://github.com/DEEPX-AI/dx_app), and a large [online Model Zoo](https://developer.deepx.ai/modelzoo/).

The public catalog reported 354 entries under DX-COM 2.4.0 and DX-RT 3.4.0 when checked on August 15, 2026. Its documented coverage includes YOLOv3 through YOLO11 and YOLO26, YOLOX, SSD, EfficientDet, NanoDet, DAMO-YOLO, YOLO segmentation, pose and oriented boxes. DEEPX is a particularly plausible integration target because the compiler/runtime boundary and deployment artifact are clearly named and the public vision coverage is broad.

## Industrial SoCs are several ecosystems, not one

Industrial vendors often have longer product lifecycles and better camera, safety and real-time integration than maker-board SoCs. Their AI stacks can be more complicated because one vendor may ship several unrelated NPU architectures at once.

### Texas Instruments

TI's current edge-AI processors include AM62A, AM67A, AM68A, AM69A and related TDA4 devices. The software path combines Processor SDK Linux, the TIDL compiler/runtime, [Edge AI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) and the [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo). TIDL can integrate with ONNX Runtime, TensorFlow Lite and other application runtimes while offloading supported subgraphs.

TI publishes much more than a framework-import claim: the zoo contains converted models and performance metadata for object detection, segmentation, pose, classification, depth and other tasks. TI also warns that model-zoo artifacts are development starting points rather than automatically production-ready assets. That is a valuable caveat often missing from vendor comparisons.

TI also lists [TDA54-Q1](https://www.ti.com/product/TDA54-Q1) as Preview, with up to four C7 NPUs and up to 400 TOPS on that part; the broader TDA5 family is advertised up to 1,200 TOPS. Those are vendor product claims for a new generation, not permission to transfer AM6xA/TDA4 TIDL model validation to TDA54-Q1 before TI publishes a target-specific matrix.

### NXP

NXP requires at least four backend descriptions:

| NXP target | Accelerator | Compiler/runtime path |
|---|---|---|
| i.MX 8M Plus | 2.3-TOPS VeriSilicon Vivante NPU | eIQ with TIM-VX/VX delegate |
| i.MX 93 | Arm Ethos-U65 | Quantized TFLite plus Arm Vela |
| i.MX 95 | NXP eIQ Neutron NPU | Neutron Converter and eIQ runtime |
| Ara-1 / [Ara240](https://www.nxp.com/products/ARA240) | Kinara-derived discrete NPU; Ara240 advertises up to 40 eTOPS | Ara SDK, with integration into the broader eIQ story |

The [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) contains assets or recipes for YOLOv4-tiny, YOLOv8, NanoDet, CenterNet, FastestDet, SSD Lite, YOLACT and other models; its YOLOv5 entry was added as documentation only. An entry validated for one backend is not evidence for all four. NXP's own [YOLO export guidance for i.MX platforms](https://community.nxp.com/t5/i-MX-Processors-Knowledge-Base/Exporting-YOLO-Models-for-NXP-i-MX-Platforms/ta-p/2381361) illustrates this target-specific conversion problem. NXP says the monolithic [eIQ Toolkit stopped receiving updates after version 1.17 in Q3 2025](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/eIQ-FAQ/ta-p/1099741); current workflows use standalone packages such as eIQ Neutron SDK.

NXP [completed its acquisition of Kinara in October 2025](https://media.nxp.com/news-releases/news-release-details/nxp-completes-acquisitions-aviva-links-and-kinara-advance/). The [Ara240 fact sheet](https://www.nxp.com/docs/en/fact-sheet/ARA240DNPUFS.pdf) reaches up to 40 vendor-advertised eTOPS and reports 313 images per second for YOLOv8n. NXP lists both Ara SDK and eIQ Toolkit on the product surface, but that does not establish artifact interchange with the separate [i.MX 95 Neutron path](https://eiq.nxp.com/learning-hub/convQuant/neutron.html). The [16 GB M.2 module](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT) is active while the USB option is preproduction. NXP defines the "e" in eTOPS as "equivalent," not "effective"; it is not another vendor's dense-INT8 TOPS metric and should not be compared directly.

### STMicroelectronics

[STM32N6x7 devices](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html), including STM32N657 and STM32N647, bring a 600-GOPS Neural-ART accelerator into an MCU-class product; the N6x5 general-purpose line does not include that accelerator. [STM32Cube AI Studio](https://www.st.com/en/development-tools/stedgeai-cubeai.html) and ST Edge AI Core analyze and optimize imported models, while [STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) provides deployment, optimization, benchmarking and example workflows.

The current [object-detection overview](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/docs/README_OVERVIEW.md) documents Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 and ST-YOLOX alongside classification, pose and segmentation services. This is not the same performance class as a 200-TOPS PCIe card. It is interesting because camera ingest, inference and control can live in a deeply embedded power and memory envelope.

### Renesas

Renesas RZ/V processors integrate DRP-AI accelerators. The software has evolved from DRP-AI Translator and DRP-AI TVM toward [RUHMI](https://www.renesas.com/en/software-tool/ruhmi-framework), powered by EdgeCortix MERA technology. The open [RZ/V DRP-AI TVM repository](https://github.com/renesas-rz/rzv_drp-ai_tvm) and the [RZ/V2H validation list](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) name YOLOv5, YOLOv8, YOLO11 and YOLO26 n/s/m variants, YOLOX, pose and segmentation networks alongside classification models.

Renesas is a credible robotics and industrial target, but the exact board, DRP-AI generation, translator version and CPU-side postprocessing need to be recorded for reproducibility.

## Smart sensors and camera-first processors

### Sony IMX500

Sony's IMX500 is not a normal application processor. It stacks image sensing and AI processing so inference can occur in the sensor, reducing the amount of image data that must leave the camera. The Raspberry Pi AI Camera makes that architecture unusually accessible.

The official [Raspberry Pi IMX500 model repository](https://github.com/raspberrypi/imx500-models) publishes packaged YOLOv8n, YOLO11n, EfficientDet Lite0, NanoDet+ and SSD MobileNetV2 FPN Lite models. The [AI Camera documentation](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) explains conversion, packaging and on-sensor postprocessing. The deployment unit is an RPK package rather than a generic ONNX file, and sensor memory plus operator constraints are central design limits. Raspberry Pi's [product page](https://www.raspberrypi.com/products/ai-camera/) states that this camera will remain in production until at least January 2028.

### Ambarella

Ambarella's CVflow processors are deeply established in cameras, drones and automotive vision. Current product families include CV72/CV75 for cameras, CV5/CV52 for higher-performance vision systems and CV3-AD automotive devices. The Cooper developer platform and CVflow compilation/runtime flow form the publicly named software stack.

The [public developer model garden](https://www.ambarella.com/developer/model-garden/) names YOLOX, RTMDet, DeepLabV3+, TopFormer, OWL-ViT and multimodal models. However, detailed compiler documentation and downloads remain partner-oriented. An Ambarella integration should therefore begin with a vendor or OEM relationship, not an assumption that a public pip package is available.

### Synaptics Astra

Synaptics has three relevant targets. SL1600/SL1680 Astra systems use the [SyNAP toolkit](https://developer.synaptics.com/docs/synap/introduction), which converts a source model plus a YAML description into a `model.synap` package. Newer SL2611/13/15/17/19 devices use the [Torq platform](https://developer.synaptics.com/docs/torq/introduction), an IREE/MLIR-based flow that produces `.vmfb`. The [SR100 series](https://developer.synaptics.com/docs/sr/introduction-sr) is a separate MCU family built around Cortex-M55 and Ethos-U55 with its own SDK. These artifacts and APIs are not interchangeable.

The official [YOLO benchmark tutorial](https://developer.synaptics.com/docs/sl/tutorials/vision/benchmark-yolo) is unusually concrete: it exports YOLOv8 to TFLite, performs asymmetric UINT8 calibration in SyNAP, compiles for SL1680, then runs `synap_cli` and the object-detection application. It proves YOLOv8n/v8s on SL1680 and describes SyNAP support through YOLO11. A separate [SL261x object-detection guide](https://developer.synaptics.com/docs/sl/sl2600/getting-started/object-detection) runs YOLOv8 from a Torq `.vmfb`. These are target-specific official examples, not a claim that every YOLO graph is supported on all three families.

### MediaTek Genio

MediaTek deserves a full entry because its current [IoT AI Hub](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) now exposes the hardware/software matrix, model packages and benchmark tables that older market surveys could not audit. Genio 360/360P/420/520/720 use NeuroPilot 8 with MDLA 5.3; Genio 510/700 use NP6 with MDLA 3.0; Genio 1200 uses NP6 with MDLA 2.0. MediaTek states that the NeuroPilot and MDLA generation, operator set, compiler and runtime are version-bound for the life of each SoC.

The analytical AI path is TFLite-centered:

```text
PyTorch model
    -> NeuroPilot Converter and representative calibration
    -> quantized .tflite
    -> version-matched ncc-tflite compiler
    -> chip-generation-specific .dla
    -> Neuron Runtime on MDLA
```

Newer Genio products also expose online LiteRT delegation and an ONNX Runtime CPU/NPU path on supported operating systems. Those are different execution modes from an offline `.dla`. The [software architecture guide](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/software_architecture.html) and [resource matrix](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) make the distinction explicit.

The official [analytical model table](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical.html) publishes YOLOv5s and YOLOv8s benchmark entries and conversion guides across multiple MDLA generations. It explicitly does not distribute preconverted YOLO artifacts because of AGPL-3.0 restrictions. Its Quant8, 640 x 640 offline measurements include 5.35 ms and 8.04 ms respectively on Genio 720. The [YOLOv8s model page](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical/YOLOv8s.html) publishes input/output tensors, backend-specific times and warnings about MediaTek custom operations. These figures are vendor benchmarks, not end-to-end camera results.

Access is the limiting factor. Public Yocto documentation is detailed, but the current NP8 all-in-one converter/compiler bundle and much Android material are listed as NDA or direct-customer resources. A normal developer account does not grant the same access as a MediaTek Online customer account. LibreYOLO should therefore treat Genio as technically proven but partnership-dependent for a reproducible bring-your-own-model compiler integration.

## China-focused edge AI ecosystems

Several of the most capable and accessible low-cost vision platforms are poorly represented in English-language market lists.

### D-Robotics and Horizon-derived BPU platforms

D-Robotics maintains the RDK developer-board ecosystem around BPU accelerators used in RDK X3, X5, Ultra and newer S100-series products. The current `rdk_x5` branch of its [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo) documents YOLOv5/v5u, YOLOv8/9/10/11/12/13/26, YOLOE, YOLO-World, detection, segmentation, pose, classification, OCR and CLIP paths for RDK X5. X3 uses a separate branch, while the current S-series delivery lives in the main zoo's [`rdk_s` branch](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s); the older [`rdk_model_zoo_s` repository](https://github.com/D-Robotics/rdk_model_zoo_s) contains historical demos. The modern X5 list is therefore not evidence that every model was validated on X3, Ultra or S100.

OpenExplorer/Algorithm Toolchain imports ONNX or Caffe for PTQ and supports PyTorch-oriented QAT flows. Deployment is platform-specific: current RDK X5 uses `.bin` through `hbm_runtime` over `libdnn`, while the current primary `rdk_s` flow uses `.hbm` through a same-named `hbm_runtime` API over `libhbucp`; X3 retains its older branch and inference interfaces. Older `rdk_model_zoo_s` material also describes historical `.bin` and `.hbm` paths, so those artifacts must remain paired with their matching branch and toolchain. Unsupported operators may fall back to CPU. The public examples are good, but version pairing between RDK OS, board firmware, toolchain and model binary remains part of the compatibility contract.

### AXERA

AXERA's AX650/AX630/AX620 families appear in compact AI cameras and boards such as Sipeed's MaixCAM line. Pulsar2 imports ONNX, performs calibration and precision analysis, and emits `.axmodel`; AXEngine runs the artifact.

The [official AXERA samples](https://github.com/AXERA-TECH/ax-samples) cover YOLOv5/6/7/8/9/10/11/13/26, YOLOX and YOLO-World, with task and chip support varying by platform. On supported targets, the current YOLO26 examples include detection, pose, segmentation and OBB. The [Pulsar2 conversion examples](https://pulsar2-docs.readthedocs.io/en/latest/appendix/model_convert_examples.html) expose the real work: exact tensor names, output cuts, layout transforms, calibration archives and target hardware settings.

### SOPHGO

SOPHGO's TPU-MLIR is one of the more attractive compiler stacks for independent integration because the compiler itself is open source. It imports ONNX, PyTorch, TFLite and Caffe, lowers and quantizes graphs, and produces `.bmodel` for BM targets or `.cvimodel` for CV18xx hardware.

The [TPU-MLIR project](https://github.com/sophgo/tpu-mlir) supports FP32, BF16, FP16 and INT8 flows depending on target. The [SOPHON demo collection](https://github.com/sophgo/sophon-demo/blob/release/README_EN.md) names YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, OBB/segmentation variants, SSD, CenterNet, RetinaFace, SAM/SAM2 and OCR. As always, a demo on BM1684X does not validate the same artifact on BM1688 or CV18xx.

### Huawei Ascend

Huawei's edge inference line includes Ascend 310/310P/310B silicon in Atlas 200I and 300I products. CANN provides the ATC compiler, AscendCL runtime and chip-specific operator kernels; ATC turns ONNX and other source representations into an offline `.om` model.

The current [Atlas 200I DK A2 documentation](https://www.hiascend.com/en/hardware/developer-kit-a2/specification) and CANN samples provide a YOLOv5 `.om` route for Ascend 310B. The broader, older [Ascend ModelZoo](https://github.com/Ascend/modelzoo) contains YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN, pose and segmentation models, but it should not be presented as if every entry were revalidated on 310B. CANN, kernels, firmware and SoC must match. An `.om` for Ascend310P is not a generic Ascend artifact.

### Cambricon

Cambricon's MLU accelerators use Neuware and the MagicMind inference engine. The public [MagicMind Cloud repository](https://github.com/Cambricon/magicmind_cloud) is pinned to MagicMind 1.7 and an MLU370-X4/S4 matrix; it is historical compatibility evidence, not a current-SDK or MLU270 validation matrix. That repository documents PyTorch, ONNX, Caffe and Paddle paths; TensorFlow framework support was removed in 1.7 even though historical TensorFlow examples remain. Cambricon's current [3-series developer portal](https://developer.cambricon.com/index/document/index/classid/3.html) lists later overall SDK releases, so the public example repository and the current commercial stack must not be presented as one version.

The official [MagicMind cloud model repository](https://github.com/Cambricon/magicmind_cloud) publishes an unusually direct MLU370-X4/S4 matrix. It marks YOLOv3, Tiny-YOLOv3, YOLOv4, YOLOv5, YOLOv7, YOLOv8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN, DeepLab and UNet examples, including whether C++ or Python examples exist. The evidence is strong; the main barrier is SDK and container access through Cambricon channels.

### Canaan/Kendryte

Kendryte's current K230/K230D and legacy K210/K510 KPU chips are important at the low-cost board and smart-camera end of the market. The [nncase compiler](https://github.com/kendryte/nncase) imports ONNX or TFLite, uses calibration data for fixed-point conversion, simulates the result on a host, and emits `.kmodel`.

The official [K230 nncase development guide](https://github.com/kendryte/k230_docs/blob/main/en/01_software/board/ai/K230_nncase_Development_Guide.md) walks through compiling, simulating and running YOLOv5s. The separate [AI demo catalog](https://github.com/kendryte/k230_docs/blob/main/en/02_applications/ai_demos/K230_AI_Demo_Introduction.md) includes YOLOv8n detection, segmentation and pose artifacts, while the current [CanMV changelog](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) adds optimized YOLOv8/11/26 classification, detection, segmentation, OBB and pose examples. The datasheet's [YOLOv5s result](https://github.com/kendryte/k230_docs/blob/main/en/00_hardware/K230_datasheet.md) is the actual published benchmark; the compiler and binary KPU plug-in versions must match the board SDK.

### Allwinner

Allwinner's V853 combines camera-oriented media hardware with a 1-TOPS Vivante NPU. The [official English NPU guide](https://docs.aw-ol.com/v853/en/npu/dev_npu/) describes Acuity/Pegasus import, quantization, verification and deployment through `viplite`. Allwinner's own model pages and [YOLOv5 guide](https://docs.aw-ol.com/v853/npu/npu_yolov5/) name YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet and face/person networks.

This is real computer-vision support, but it remains tied to the older Tina Linux/Vivante toolchain and account-dependent SDK downloads. It belongs in the market map with that limitation visible.

## Overlooked Korean, Taiwanese and Chinese platforms

English-language lists often jump from Rockchip to NVIDIA and miss a second tier of real, documented silicon. Some of these ecosystems have broader current YOLO matrices than better-known Western startups.

### Mobilint

Korean accelerator vendor Mobilint sells the low-power REGULUS and higher-throughput ARIES MLA100. Its qb SDK accepts PyTorch, TensorFlow, TFLite, ONNX and Keras-oriented inputs, quantizes to INT8 and emits a compiled `.mxq` artifact. The public [vision matrix](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) names YOLOv3/v5/v7/v8/v9/v10/11/12/26 detection plus YOLO segmentation, pose and OBB variants. The [ARIES page](https://www.mobilint.com/aries/mla100) even publishes model-specific YOLO11s and YOLO26m results. That is strong integration evidence, though SDK/hardware access remains commercial.

### Sunplus

Sunplus SP7350/C3V uses a Vivante-derived NPU stack but packages it differently from Allwinner or legacy Amlogic. Its public flow combines Acuity conversion, an NPU Docker environment, SNNF runtime and `.nb` output. The official [custom YOLOv8 guide](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) walks through conversion and deployment rather than merely claiming framework support. Related IP does not make its `.nb` portable to another Vivante-based SoC.

### ESWIN Computing

ESWIN's RISC-V family includes EIC7700/EIC7700X and the dual-die EIC7702/EIC7702X, with EIC7700 appearing in shipping boards such as Milk-V Megrez. ENNP contains EsQuant, the current EsAAC compiler, golden-data generation, simulation and the ESSDK runtime, with compiled `.model` output; older ENNP material used the `ennc-compile` name. The current [EsAAC page](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) documents ONNX input, so other frameworks need export or conversion to that supported IR. The Milk-V-hosted [ENNP overview](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/introduction) and [YOLOv3 tutorial](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3) establish a public end-to-end path, but not a broad current model/accuracy matrix.

### Nuvoton M55M1

Nuvoton's M55M1 is an MCU-class Cortex-M55 plus Ethos-U55-256 design rather than a Linux application processor. NuEdgeWise/NuML and Arm Vela prepare quantized TFLite Micro models. The public [NuEdgeWise repository](https://github.com/OpenNuvoton/NuEdgeWise) names YOLOv8-nano, YOLOX-nano, YOLO Fastest v1.1, SSD-MobileNet FPNLite and multiple classifiers. These small-network examples are credible evidence for the device's memory/power class, not evidence for standard 640-pixel YOLO variants.

### Rebellions and FuriosaAI

Rebellions' ATOM line uses the RBLN SDK and compiled `.rbln` artifacts. Its [official support release](https://docs.rbln.ai/v0.8.2/supports/release_note.html) names YOLOv3 tiny/full/SPP, YOLOv5 n through x, YOLOv6 n through l, YOLOv7 variants and YOLOv8 n through x. The current [card support matrix](https://docs.rbln.ai/latest/supports/version_matrix.html) marks ATOM CA02 and ATOM+ CA12 EoL, while ATOM+ CA22 and ATOM-Max CA25 are Active. ATOM-Lite CA21 has a product page but is absent from that SDK matrix, so its current toolchain status is unclear. Documentation is public, but the compiler wheel requires Rebellions Portal credentials.

FuriosaAI must be split by generation. Warboy is the older computer-vision accelerator, with INT8 `.enf` artifacts and an [official model zoo](https://developer.furiosa.ai/furiosa-models/latest/) containing SSD, YOLOv5M/L and YOLOv7-w6-pose. RNGD uses a different [`.fxb` stack](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html) aimed primarily at LLM/VLM workloads. Furiosa's current [roadmap](https://developer.furiosa.ai/latest/en/overview/roadmap.html) records YOLOv8m vision support as completed in 2024 Q4, but its current public supported-model and performance surfaces remain LLM/VLM-centered and do not expose a detailed YOLOv8m accuracy/performance matrix. That is released-generation evidence, not a Warboy compatibility claim.

### Realtek, Telechips, T-Head and SigmaStar

These four platforms are real but have narrower, older or gated bring-your-own-model surfaces:

| Platform | Reproducible public evidence | Limitation to preserve |
|---|---|---|
| Realtek AmebaPro2 | [Arduino/FreeRTOS SDK](https://github.com/Ameba-AIoT/ameba-arduino-pro2) packages YOLOv3/4/7-tiny, SCRFD and MobileFaceNet `.nb` models | Custom model conversion requires registering interest/contacting Realtek |
| Telechips TOPST TCC7500 | Official [2026 project](https://docs.topst.ai/blog/31) converted and deployed YOLOv8s | UFLD v1/v2 conversion failed on unsupported layers; the article only proposes a split/postprocessing workaround. Full compiler/operator resources commonly require emailed permission |
| T-Head TH1520 | Sipeed's [application guide](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) runs YOLOv5n/s with HHB and CSI-NN2/SHL | Public stack exists, but maintenance and version clarity lag current leaders |
| SigmaStar SSU9383CM | Current docs establish MI_IPU runtime APIs; older SGS_IPU material documents `.sim` to `sgsimg.img` and an older postprocessor names SSD and YOLOv1/2/3 | No public proof that the older compiler artifacts or named-model paths apply to SSU9383CM |

### HiSilicon smart vision is not Huawei Ascend

HiSilicon's Hi3516CV610/DV500, Hi3519DV500 and Hi3403V100 camera SoCs expose an ATC-to-`.om` workflow through NNN/SVP-NNN runtimes. The official [HiSpark model zoo](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) names YOLOv3/4/5/6/7/8/9/10/11, YOLO11 segmentation/pose, YOLOv8 OBB/World/segmentation, OCR, depth and TinySAM in a target-specific matrix, notably for Hi3403 and Hi3591P. It does not establish that every entry runs on every smart-vision SoC listed above. Some entries are roadmap items, so released examples and planned support must stay separate, and the repository states that its provided ModelZoo models are for non-commercial use only. Shared ATC/`.om` terminology does not prove artifact or runtime compatibility with the CANN-based Ascend product line.

## Emerging and specialized accelerator companies

### MemryX

MemryX MX3 modules use a dataflow architecture and can combine multiple chips. The [Neural Compiler](https://developer.memryx.com/tools/neural_compiler.html) accepts ONNX, TensorFlow Lite, Keras and TensorFlow models and emits a DFP package consumed by the accelerator runtime. Its [runtime APIs](https://developer.memryx.com/api/accelerator/accelerator.html) support multi-model and streaming pipelines.

MemryX documentation and examples include optimized YOLO preprocessing and postprocessing for detection, segmentation and pose. The [SDK release notes](https://developer.memryx.com/release_notes.html) explicitly name YOLOv10, YOLO11 and YOLO26 support. The public material is technically useful, although it lacks one simple model/accuracy/device table comparable to Axelera's.

### Kneron

Kneron's KL520/KL530/KL630/KL720/KL730 products span small USB and embedded accelerators. Kneron PLUS covers the application runtime, while the current Model Toolchain 0.33.1 documents conversion, quantization, evaluation, simulation and `.nef` compilation for those targets. KL830 appears in some [PLUS runtime APIs](https://doc.kneron.com/docs/plus_c/introduction/run_examples/), but not in the current [compiler target list](https://doc.kneron.com/docs/toolchain/manual_1_overview/); a generic `.nef` compilation claim should not be transferred to KL830 without vendor confirmation.

The official [YOLO example](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) demonstrates the process with Tiny-YOLOv3, and other Kneron material covers YOLOv5 training and deployment. The evidence is credible but narrower than the broad modern-YOLO matrices from Hailo, Axelera, Rockchip or DEEPX.

### SiMa.ai

SiMa.ai's MLSoC and [production Modalix 50-TOPS platform](https://sima.ai/press-release/sima-ai-next-gen-platform-for-physical-ai-in-production/) use the Palette software environment. ModelSDK, the MLA compiler and ModelExecutor cover import, quantization, compilation, profiling and execution. The tools support INT8, INT16 and BF16-oriented flows depending on the workload and product.

The company's release notes name validated YOLOv7, YOLOv8 detection/pose/segmentation, YOLOX and YOLOX segmentation, DETR, Mask R-CNN and EfficientDet pipelines. One current limitation should not be hidden: [Palette SDK 2.1 notes](https://docs.sima.ai/v2.1.2/pages/release_notes/2.1.html) say QAT is not functional because of a Python/PT2E quantization regression. The documented workaround is to create the annotated ONNX in SDK 2.0 and compile it with 2.1. Access and procurement remain enterprise-oriented.

The deployment boundary is also documented: the [ModelSDK compilation flow](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) emits a compiled `.tar.gz` and can produce an executable ELF, while [MPK Tool](https://docs.sima.ai/v2.1.1/pages/palette/mpk_tools.html) packages a deployable `.mpk`. These outputs remain bound to the MLSoC/Modalix target and Palette release.

### EdgeCortix

SAKURA-II is a 60-TOPS advertised accelerator for vision and generative AI, compiled through the MERA software framework. EdgeCortix also supplies MERA technology to Renesas' newer RUHMI flow.

The [SAKURA-II product material](https://www.edgecortix.com/en/edgecortix-sakura-ii-accelerator-brief) establishes the hardware and compiler, and [M.2/PCIe hardware](https://www.edgecortix.com/en/hardware) is offered through trial or order inquiry. No sufficiently precise, current public YOLO compatibility matrix was found. That absence is useful purchasing information: model validation should be requested before committing to hardware.

### BrainChip

BrainChip's Akida is a neuromorphic event-domain processor rather than a conventional dense-tensor NPU. Its [MetaTF environment](https://brainchip.com/metatf-dev-tools/) consists of installable Python packages for model creation, low-bit quantization, conversion, simulation and hardware execution. BrainChip now sells a [shipping AKD1500 co-processor in M.2 form](https://brainchip.com/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/), alongside legacy AKD1000 hardware and Akida 2 IP.

The current [Akida 2 model card](https://brainchip.com/wp-content/uploads/2025/04/Akida-2-Model-Card-V1.1-Mar.25.pdf) names an AkidaNet0.5 YOLOv2 detector, CenterNet, AkidaUNet and face-recognition networks. Other support material includes FOMO and event-based workloads. Akida supports unusually low-bit weights and activations, but successful deployment may require architecture-aware conversion rather than treating it as another generic INT8 ONNX target. An AKD1000 or Akida 2 model-zoo result should not be attributed to AKD1500 until an explicit map/fit report validates it.

### Blaize

Blaize sells P1600-based Pathfinder and Xplorer products, with Picasso SDK, NetDeploy and AI Studio providing the software path. The [product pages](https://www.blaize.com/products/) clearly target vision and edge AI, but a current, public, model-by-model compatibility matrix is not available.

This guide therefore classifies Blaize as commercial and gated rather than filling the gap with community claims. A vendor-supplied compilation and accuracy report for the intended model should be a prerequisite.

## TinyML and endpoint vision

### Arm Ethos-U and Alif

Arm licenses Ethos-U55, U65 and U85 NPU IP to chip companies. The [Vela compiler](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) takes quantized LiteRT/TFLite graphs and rewrites supported subgraphs into Ethos-U custom operations. Unsupported operations may remain on the CPU, which means "the application runs" and "the whole network runs on the NPU" are different claims.

Real implementations include Ethos-U55 in Alif Ensemble E7, Ethos-U65 in NXP i.MX 93 and Ethos-U85 in newer systems. Alif's [E7 AI/ML AppKit](https://alifsemi.com/support/kits/ensemble-e7appkit/) documents two ML accelerators and demonstrates the E7 TFLite-to-Vela route; the [Ensemble E8](https://alifsemi.com/ensemble-e8-series/) combines one Ethos-U85 with two Ethos-U55 NPUs. Alif also publishes a family-level [INT8 YOLO-Fastest face-detector benchmark](https://alifsemi.com/faster-ai-mcu-inferencing-low-power-consumption/) at 192 by 192 on Cortex-M55 plus Ethos-U55, but not a broad target-by-target modern-YOLO matrix. These memory-constrained systems suit small classification and detection networks, not arbitrary 640-pixel detectors merely because they are expressible in TFLite.

### Infineon PSOC Edge, Himax WiseEye2 and Analog Devices MAX7800x

Infineon's [PSOC Edge E83](https://documentation.infineon.com/psocedge/docs/xsk1761304020519) and E84 families pair Cortex-M55 with Ethos-U55 and add a separate NNLite block on the low-power M33 side. The [E84 architecture manual](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) documents 8-bit weights, 8- or 16-bit activations, and a Vela flow in which supported operators become an NPU command stream while unsupported work remains on the CPU. The [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter) accepts TFLite, Keras and PyTorch paths, while the architecture material names MobileNetV1/V2 as representative kernels rather than target benchmarks. The [E84 AI Kit](https://documentation.infineon.com/psocedge/docs/cci1762693051052) includes a camera and uses ModusToolbox, but the public material does not yet provide a named YOLO compatibility matrix. This is a real programmable vision platform with maturing model-level evidence, not a validated general-purpose detector target.

Himax's [HX6538 WiseEye2](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) also combines Cortex-M55 and Ethos-U55 for always-on vision. Its public software story is unusually concrete for this power class: the [official Grove Vision AI Module V2 examples](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) include YOLOv8n object detection, pose and gender classification, YOLO11n detection, face mesh and PeopleNet. The unified path uses TFLite Micro and Vela, then falls back through CMSIS-NN and reference kernels when necessary. These are deliberately tiny models and resolutions, not evidence that a conventional server-size YOLO graph fits.

Analog Devices takes a different route with the [MAX78000/MAX78002 CNN accelerators](https://www.analog.com/en/products/max78002.html). The public [`ai8x-synthesis`](https://github.com/analogdevicesinc/ai8x-synthesis) tool quantizes a trained network and generates target-specific C code, weights and accelerator configuration rather than a portable ONNX runtime package. First-party evidence includes [face identification](https://www.analog.com/en/resources/app-notes/an-2616.html), TinierSSD QR detection in the [ADI model zoo](https://ez.analog.com/dsp/software-and-development-tools/edgebench/a/docs-faqs/DF912/available-models-in-adi-model-zoo), and image classifiers. The devices are relevant to endpoint vision, but no current official modern-YOLO matrix was found.

### GreenWaves GAP9

GAP9 combines RISC-V compute with the NE16 neural engine. NNTool imports and quantizes networks, while AutoTiler and the GAP SDK generate deployable code. The public [GAP9 neural-network menu](https://github.com/GreenWaves-Technologies/nn_menu_gap9) contains MobileNet, EfficientNet, ResNet, MobileNet SSD, face and recognition applications. GreenWaves also publishes a [YOLOX people-detection project](https://github.com/GreenWaves-Technologies/yolox_people_detection) with accuracy and simulation results.

This is unusually transparent TinyML evidence, but the complete SDK is available only to qualified customers and the models are substantially smaller than mainstream 640 by 640 YOLO variants.

### Lattice sensAI

Lattice sensAI is an FPGA-based alternative to a fixed NPU. sensAI Studio and the Neural Network Compiler map a supported graph onto configurable accelerator IP for ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E and Avant-X devices. The output is tied to the selected FPGA, accelerator mode, memory layout and bitstream rather than being a portable runtime model.

The current [Neural Network Compiler manual](https://www.latticesemi.com/-/media/LatticeSemi/Documents/UserManuals/MQ3/FPGA-UG-02052-8-0-Lattice-Neural-Network-Compiler-Software.ashx?document_id=52343) includes an unusually direct topology table. It lists YOLOv1 for ECP5, YOLOv5 and YOLOv8 for optimized/advanced CrossLink-NX and CertusPro-NX modes, and YOLO11 only with Advanced mode IP. It also lists SSD, MobileNetV2-SSD, ResNet, ENet, SqueezeDet and other vision networks, with unsupported cells visible by FPGA family. The [Advanced CNN Accelerator](https://www.latticesemi.com/en/Products/DesignSoftwareAndIP/IntellectualProperty/IPCore/IPCores05/Advanced-CNN-Accelerator-IP) targets Avant-E/Avant-X/CertusPro-NX and is commercial IP. This is compiler-table evidence, not a promise that one bitstream supports every listed topology simultaneously.

### Microchip VectorBlox

Microchip's public [VectorBlox SDK 3.1](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) preprocesses and compiles fully quantized INT8 TFLite networks for configurable CoreVectorBlox IP in PolarFire SoC FPGAs. `tflite_preprocess` and `vnnx_compile` produce `.vnnx`, `.hex` and `.ucomp` deployment assets, and the repository includes a simulator and target-side C driver. Upstream TensorFlow, ONNX and OpenVINO graphs still have to pass through the documented TFLite/INT8 preparation path.

The current [tutorial matrix](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) includes YOLOv5n, YOLOv8n detection, classification, OBB, pose and segmentation, YOLOv9t, sparse/compressed YOLO variants, MobileNet, EfficientNet-Lite0, ResNet18, MiDaS, FFNet and QuickSRNet. The [C postprocessing guide](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/C_Postprocessing.md) separately names YOLOv2/3/4/5 and Ultralytics detection, pose and OBB decoders. SDK 3.1 currently supports the PolarFire SoC Video Kit and explicitly not the non-SoC PolarFire Video Kit, so older board demos must not be mistaken for the current target contract.

The SDK is publicly downloadable under a [Microchip-specific license](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/LICENSE.md) that restricts the software and derivatives to Microchip products. Microchip offers free but request-based Libero Silver and CoreVectorBlox licenses. Deployment is still an FPGA system contract: accelerator configuration, bitstream, firmware, SDK-generated network data and memory layout must agree. A successful VectorBlox compile does not turn the model into a binary that can be copied to an arbitrary PolarFire design.

### Syntiant

Syntiant's NDP200 targets always-on vision, audio and sensor inference below the power envelope of normal Linux SoCs. The current [hardware table](https://www.syntiant.com/hardware) lists NDP200 as mass production and NDP250 as sampling. The [NDP200 product brief](https://www.syntiant.com/ndp200/) documents CNN, RNN and fully connected network support below 1 mW; the [NDP250 announcement](https://www.syntiant.com/news/syntiant-unveils-ndp250-neural-decision-processor-with-next-gen-core-3-architecture) instead describes always-on image recognition below 30 mW. "Sub-milliwatt" should not be generalized across both parts.

The public material does not establish broad YOLO compatibility, so the product should be evaluated for small always-on classifiers and detectors through a vendor-supported model assessment rather than a generic ONNX export assumption.

### Google Coral: two unrelated generations

The legacy Edge TPU products use the Edge TPU Compiler, `libedgetpu` and PyCoral with fully INT8 TFLite models. Their main [Edge TPU](https://github.com/google-coral/edgetpu) and [PyCoral](https://github.com/google-coral/pycoral) repositories are archived. That is a real maintenance risk, but it is not a formal hardware EOL notice.

Google also maintains a separate, active, open-source [Coral NPU](https://github.com/google-coral/coralnpu): RISC-V accelerator IP for integration into low-power SoCs. The public project currently exposes RTL, hardware simulation, a toolchain and ELF examples. It is not a drop-in USB/M.2 successor to Edge TPU and does not publish a current YOLO deployment matrix.

## Adjacent GPU, client-NPU and adaptive-compute traps

NVIDIA, AMD, Intel and Apple often appear in NPU searches, but they do not expose one interchangeable class of accelerator.

**NVIDIA Jetson:** Jetson Orin combines a CUDA GPU with dedicated DLA cores. TensorRT can build a serialized engine for DLA, but unsupported layers run on the GPU only when GPU fallback is explicitly enabled. The current [DeepStream YOLO table](https://github.com/NVIDIA-AI-IOT/deepstream_tools/blob/main/yolo_deepstream/README.md) covers YOLOv4/v7/v8/v9/11, but most entries are GPU targets; its explicit ONNX DLA entry is YOLOv8s INT8. NVIDIA's [DLA restrictions](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html) explain the operator limits. Thor is another distinction: NVIDIA's [DriveOS migration guide](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/NVIDIA_DriveOS_7.0.3_Migration_Guide.pdf) says DLA cores were removed from Thor in favor of GPU scheduling flexibility. "Runs on Jetson" therefore does not identify the accelerator.

**AMD Vitis AI:** The older Kria/DPU generation compiled `.xmodel` graphs executed through VART. Current [Vitis AI 6.2](https://vitisai.docs.amd.com/en/6.2/) is GA for two separate Versal AI Edge paths: [Gen1](https://vitisai.docs.amd.com/projects/gen1/en/latest/index.html) on VEK280/VE2802 uses ONNX, AMD Quark and a target-bound snapshot/subgraph flow; [Gen2](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html) on VEK385 has its own compilation and runtime contract. Gen1 publishes YOLOv5, YOLOv7, YOLOv8 and YOLOX examples. Ryzen AI is a fourth, separate client-NPU stack. Do not transfer artifacts or validation among legacy DPU, Versal Gen1, Versal Gen2 and Ryzen AI merely because all use the Vitis or AMD name.

**Intel OpenVINO:** OpenVINO can target CPU, GPU and Core Ultra NPU through one API. The [Intel NPU plug-in documentation](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) identifies supported NPU generations, while the [verified-model matrix](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html) exposes device-specific columns. A YOLO notebook proves an application recipe, not NPU validation unless the matrix says so. Supported operations, shapes, precision and the installed NPU driver still determine whether the complete graph runs there. OpenVINO IR (`.xml` plus `.bin`) is reusable source-level IR; a compiled-model cache is device and software specific.

**Apple Core ML:** `coremltools` converts models into `.mlpackage`, which Xcode compiles to `.mlmodelc`. Core ML can schedule work across CPU, GPU and Apple Neural Engine, but it intentionally abstracts the exact partition. That makes Apple an excellent deployment platform and a poor candidate for claims such as "the whole YOLO graph ran on the NPU" unless profiling evidence establishes it.

## The NPU IP companies behind the chip companies

Some companies do not sell a board or packaged accelerator at all. They license NPU designs to SoC vendors. They matter because the same underlying architecture can appear under several chip brands, but the final SDK and artifact may still be customized by the licensee.

| IP supplier | NPU family and software | Why it matters |
|---|---|---|
| [Arm](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) | Ethos-U55/U65/U85 and Vela | Appears in NXP, Alif, Infineon, Himax and other embedded products; quantized TFLite/TOSA-oriented flow |
| [Arm China](https://www.armchina.com/mountain?infoId=161&name=) | Zhouyi AIPU and Compass SDK | Public [model zoo](https://github.com/Arm-China/Model_zoo) names YOLOv1-tiny/v2/v3/v4/v5, YOLOX and YOLOv8-seg; entries are reference models, not proof for every licensee chip |
| [VeriSilicon](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico) | Vivante VIP family and Acuity SDK | Related Vivante NPU families appear in multiple SoCs, including the separately sourced A311D and i.MX 8M Plus paths; each chip vendor still supplies its own integration |
| [Imagination Technologies](https://www.imaginationtech.com/products/open-access/) | PowerVR Series3NX AX3146/AX3386/AX3596; historical IMG Series4; Neural Compute SDK/IMG DNN | Licensable NNA IP rather than a retail board; Open Access exposes 1/5/10-TOPS Series3NX configurations, while an official NC-SDK program names PP-YOLOE, EfficientNet and HRNet but not every IP configuration |
| [CEVA](https://www.ceva-ip.com/product/ceva-neupro-studio/) | NeuPro-Nano/NeuPro-M and NeuPro Studio | Licensable NPU IP with import, quantization, compression, graph compilation, simulation and C/C++ generation |
| [MIPS](https://mips.com/processor-solutions/arc-npx-family/) | ARC NPX6 and [MetaWare MX](https://mips.com/processor-solutions/arc-metaware-mx/), formerly Synopsys ARC | Scalable NPU IP and NN SDK acquired by GlobalFoundries and placed in the MIPS portfolio in [June 2026](https://mips.com/press-releases/gfmipsarcclose/); not itself a retail deployment target |
| [Cadence](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/neuroweave-sdk.html) | Neo NPU, Tensilica DSPs and NeuroWeave SDK | Compiler/interpreter stack used by SoC licensees, with network and quantization support tied to licensed configurations |
| [Quadric](https://quadric.ai/npu-ip) | Chimera GPNPU IP and SDK | Programmable NPU/DSP-style IP licensed into other companies' chips; the final target is the licensee's silicon |
| [Expedera](https://www.expedera.com/products-overview/) | Origin NPU IP and software stack | Licensable edge NPU architecture, not a directly purchasable LibreYOLO board |

Imagination illustrates why access and lifecycle labels belong beside architecture names. Its [Open Access program](https://www.imaginationtech.com/products/open-access/) offers three silicon-proven Series3NX configurations with no upfront IP license fee, but only after company evaluation and with paid support/maintenance plus production royalties. The [Neural Compute SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) supplies compilation, optimization, quantization and runtime tooling; an official [Baidu PaddlePaddle collaboration](https://www.imaginationtech.com/news/imagination-and-baidu-paddlepaddle-create-open-source-machine-learning-library-for-model-zoo/) names PP-YOLOE detection, EfficientNet classification and HRNet segmentation. Those examples do not validate every Series3NX or Series4 configuration, and current individual [Series4 product pages](https://www.imaginationtech.com/product/img-4nx-mc1/) label the products unavailable, so Series4 should be treated as historical unless sales confirms otherwise.

This layer explains apparent family resemblances. It does not create artifact portability. Two SoCs containing related Vivante or Ethos IP can have different memory maps, drivers, compiler versions, operator sets and vendor runtimes.

## What you actually deploy: the artifact lock-in table

The last file in the compiler pipeline is where apparent ONNX portability ends. Names and extensions change over time, but the practical rule is stable: keep the original model and calibration data because the native artifact usually cannot be moved to another NPU generation or reliably rebuilt with another SDK release.

| Vendor stack | Common compiler input | What reaches the device | What it is bound to |
|---|---|---|---|
| [Amlogic AMLNN](https://github.com/Amlogic-NN/amlnn-toolkit) | ONNX, TFLite, TorchScript or PT2 | `.adla`; legacy A311D uses `.nb` | Target ID, ADLA generation, AMLNN compiler build, NNSDK2/runtime, ADLA driver and BSP |
| [Rockchip RKNN](https://github.com/airockchip/rknn-toolkit2) | ONNX, TFLite or framework export | `.rknn` | RKNN toolkit/runtime version and Rockchip NPU family |
| [Hailo](https://hailo.ai/developer-zone/documentation/) | ONNX or TensorFlow through an intermediate HAR | `.hef` | Hailo architecture and compiler/runtime generation |
| [Axelera Voyager](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) | ONNX or supported zoo definition | `.axm` in alpha Pipeline Builder, `.axe` pipeline package; classic `.axmodel` plus manifest | Voyager API generation and Metis target configuration |
| [DEEPX DXNN](https://github.com/DEEPX-AI/dx-all-suite) | ONNX and supported framework paths | `.dxnn` | DX-COM/DX-RT version and DX-M target |
| [Qualcomm QAIRT/QNN or SNPE](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | ONNX, TFLite or framework model | QNN model library/context binary, or SNPE `.dlc` | HTP architecture, SoC, backend and runtime version |
| [MediaTek NeuroPilot](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) | Converted/quantized TFLite | `.dla` for offline Neuron Runtime; `.tflite` for delegated mode | NP/MDLA generation, OS image, compiler and runtime |
| [TI TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools) | ONNX or TFLite | Imported-artifacts directory plus application model files | TIDL tools/runtime and processor generation |
| [NXP eIQ](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | Usually TFLite or ONNX | Backend-specific Vela, TIM-VX, Neutron or Ara output | The selected i.MX/Ara accelerator and BSP, not "NXP" generically |
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | TFLite, ONNX or supported framework model | Generated network library/code and firmware assets | MCU/NPU target, memory configuration and tool version |
| [Infineon PSOC Edge](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) | Integer-quantized TFLite/Keras/PyTorch path | Vela-optimized TFLite with Ethos-U command stream plus application firmware | E83/E84 variant, Ethos-U configuration, Vela, ModusToolbox and BSP; NNLite is a separate target |
| [Renesas DRP-AI](https://github.com/renesas-rz/rzv_drp-ai_tvm) | ONNX/TFLite via TVM or Translator | Runtime-model directory containing multiple binary data files | RZ/V device and translator/runtime generation |
| [Sony IMX500](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | Quantized/packed network from Edge-MDT flow | `.rpk` package | Sensor firmware, memory budget and packer version |
| [Synaptics](https://developer.synaptics.com/) | TFLite/ONNX and supported imports | `.synap` on SyNAP; `.vmfb` on Torq | SL16xx versus SL261x software/hardware generation |
| [Ambarella CVflow](https://www.ambarella.com/developer/) | Partner toolchain input | Partner-only deployment package; exact artifact contract is not publicly documented | Confirm the CVflow generation, SDK/BSP and camera pipeline through the Cooper Developer Zone |
| [D-Robotics BPU](https://github.com/D-Robotics/rdk_model_zoo) | ONNX and toolchain-supported graph | X5 `.bin`; current `rdk_s` `.hbm`; X3 uses an older platform flow | BPU generation, repository branch and matching OpenExplorer/runtime release |
| [AXERA Pulsar2](https://github.com/AXERA-TECH/ax-samples) | ONNX | `.axmodel` | AX chip target, Pulsar2 and AXEngine version |
| [SOPHGO TPU-MLIR](https://github.com/sophgo/tpu-mlir) | ONNX, TFLite, TorchScript and others | `.bmodel` on BM, `.cvimodel` on CV18xx | BM/CV chip target and SOPHON runtime generation |
| [Huawei Ascend CANN](https://www.hiascend.com/en/software/cann) | ONNX or supported framework graph | `.om` | Ascend chip, CANN/ATC and device firmware/driver |
| [Cambricon MagicMind](https://github.com/Cambricon/magicmind_cloud) | Framework graph or exchange model | Serialized MagicMind engine/model | MLU architecture and Neuware/MagicMind version |
| [Canaan nncase](https://github.com/kendryte/nncase) | ONNX or TFLite | `.kmodel` | KPU generation and matching nncase target plug-in |
| [Mobilint qb](https://www.mobilint.com/sdk-qb) | ONNX and supported framework models | `.mxq` | REGULUS/ARIES target and qb compiler/runtime |
| [Rebellions RBLN](https://docs.rbln.ai/latest/index.html) | PyTorch 2 or TensorFlow-supported graph | `.rbln` | ATOM generation and RBLN compiler/runtime |
| [FuriosaAI](https://developer.furiosa.ai/docs/latest/en/) | ONNX/TFLite and supported framework path | Warboy `.enf`; RNGD `.fxb` | Completely different accelerator and SDK generations |
| [Sunplus SNNF](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2004353133) | Acuity-supported network | `.nb` | SP7350 NPU driver, runtime and BSP |
| [ESWIN ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) | ONNX in the documented current EsAAC flow; other frameworks require export/conversion | `.model` | Exact EIC7700-family target and ENNP/ESSDK release |
| [Nuvoton / Ethos-U](https://github.com/OpenNuvoton/NuEdgeWise) | Quantized TFLite | Vela-optimized TFLite with Ethos-U custom operations | M55M1 memory plan, Vela and firmware |
| [Himax WiseEye2](https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2) | Fully integer-quantized TFLite | Vela-optimized `.tflite` in model flash plus board firmware such as `output.img` | HX6538/WE2 configuration, Vela, TFLite Micro, Ethos-U driver and fallback kernels |
| [Analog Devices AI8X](https://github.com/analogdevicesinc/ai8x-synthesis) | PyTorch checkpoint, network YAML and sample input | Generated device-specific C source/headers, weights and compiled firmware | MAX78000 versus MAX78002 memory/layer limits, synthesis tool and MSDK release |
| [Realtek AmebaPro2](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Vendor conversion service/input | `.nb` | RTL8735B firmware and VoE/NeuralNetwork API release |
| [Telechips Enlight](https://docs.topst.ai/product/p/ai) | Darknet, TensorFlow, ONNX or PyTorch path | `.enlight` intermediate plus compiled deployment bundle | TCC7500 NPU and TC-NN/Enlight versions |
| [T-Head HHB](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | ONNX and supported framework graph | `hhb.bm`, generated parameters/code and executable | TH1520 CSI-NN2/SHL stack |
| [SigmaStar MI_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | Current SSU9383CM docs expose the MI_IPU runtime; older material uses TensorFlow/Caffe-family inputs | Older `.sim` converted to `sgsimg.img`; current public artifact unverified | Exact SigmaStar IPU and SDK generation; legacy compiler compatibility with SSU9383CM is unverified |
| [HiSilicon smart vision](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) | ONNX and ATC-supported graph | `.om` | Exact smart-vision SoC and NNN/SVP-NNN stack; not generic Ascend portability |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | ONNX, TFLite, Keras or TensorFlow | `.dfp` dataflow package | MX hardware configuration and runtime/compiler release |
| [Kneron](https://doc.kneron.com/docs/) | ONNX plus toolchain configuration | `.nef`; KL730 NEFv2 can contain `.kne` | Documented compiler target, chip generation, firmware and PLUS runtime; KL830 compilation is not established by Toolchain 0.33.1 |
| [SiMa.ai Palette](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) | Supported framework or ONNX model | ModelSDK compiled `.tar.gz`, documented executable ELF, or MPK Tool `.mpk` package | MLSoC/Modalix target and Palette release |
| [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | ONNX or network definition | Serialized engine, commonly `.engine` or `.plan` | GPU/DLA architecture, TensorRT, CUDA and often device |
| [AMD Vitis AI](https://vitisai.docs.amd.com/en/6.2/) | Quantized ONNX/framework graph | Legacy `.xmodel`; Gen1 NPU snapshot/subgraphs; Gen2 compiled cache directory or production `.rai` package | Exact NPU generation/IP configuration, platform image and Vitis/Vivado/PetaLinux matrix |
| [Microchip VectorBlox](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) | Fully quantized INT8 TFLite | `.vnnx`, `.hex` and `.ucomp` plus matching firmware/FPGA design | CoreVectorBlox configuration, PolarFire SoC Video Kit, SDK 3.1, bitstream and memory layout |
| [Intel OpenVINO](https://docs.openvino.ai/) | Framework model or ONNX | IR `.xml` plus `.bin`, or device-specific compiled cache | IR is reusable; compiled cache is device, driver and OpenVINO specific |
| [Legacy Google Edge TPU](https://coral.ai/docs/edgetpu/models-intro/) | Fully INT8 TFLite | Edge-TPU-compiled `_edgetpu.tflite` | Edge TPU compiler/runtime and supported quantized operators; unrelated to the new Coral NPU IP |
| [Google Coral NPU IP](https://github.com/google-coral/coralnpu) | C/C++ examples for the current public IP project | ELF examples for RTL/hardware simulation and eventual SoC integration; no public YOLO compiler artifact | Exact licensed/integrated SoC implementation and evolving open-source toolchain |
| [Lattice sensAI](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | Supported network description | FPGA bitstream, accelerator configuration and weights | FPGA family, sensAI IP mode and memory implementation |
| [Imagination NC-SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) | Caffe, TensorFlow or ONNX | Customer-facing compiled artifact filename is not publicly specified | Licensed NNA configuration, SoC integration, NC-SDK/DDK and licensee runtime |

This table is not merely an extension cheat sheet. The compiler version, target identifier, driver, firmware and BSP are part of the model's reproducible identity. A future LibreYOLO exporter should write them into a machine-readable manifest beside every artifact.

## Tool access and lifecycle signals

Hardware availability and compiler access are independent. A board may be easy to buy while its current compiler requires a customer agreement; a public SDK may target silicon that is hard to source. The following are concrete first-party signals, not a universal longevity ranking.

| Platform | Documented signal | What it means in practice |
|---|---|---|
| Amlogic ADLA | Public Apache-2.0 repositories and downloadable wheels; compatible board drivers/BSP may still require Amlogic or a board vendor | Easy compiler evaluation, but binary redistribution and the full compatibility matrix should be confirmed |
| MediaTek Genio | Public IoT AI Hub and Yocto guides; NP8 all-in-one tools and Android documents are marked direct-customer/NDA | Model evidence is auditable, while bring-your-own-model automation may require a partnership |
| Hailo | Public model zoo and runtime; Dataflow Compiler distributed through the Developer Zone; public `hailo-camera-apps` was archived May 3, 2026 | Broad model discovery is open, but reproducible compilation needs the correct account/version; archive status does not establish Hailo-15 EOL and its current vision-processor application package should be confirmed |
| Axelera AI | Public documentation and [public Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk); customer support requires an account | A rare self-service accelerator SDK, while product support and procurement remain commercial |
| Qualcomm Dragonwing | The [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) lists IQ-9075 as Sampling with longevity through 2038 and QCS6490 through July 2036, while the IQ-9075 product page says Active | Strong industrial planning signal, but the first-party status conflict requires SKU-level confirmation and does not guarantee one AI SDK remains ABI-stable for the whole period |
| Rebellions | The current [support matrix](https://docs.rbln.ai/latest/supports/version_matrix.html) marks CA02 and CA12 EoL, and CA22/CA25 Active; CA21 is absent | Procurement and SDK compatibility are card-specific even inside the ATOM family |
| NVIDIA Jetson | [Official module lifecycle table](https://developer.nvidia.com/embedded/lifecycle) lists Orin modules through January 2032 and AGX Orin Industrial through July 2033; developer kits have no lifecycle commitment | Design production systems around modules, not development-kit availability |
| Sony IMX500 on Raspberry Pi | The [AI Camera product brief](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf) says production until at least January 2028 | A concrete minimum for that camera module, not for every IMX500 product |
| Amlogic A311Y3 | The [2026 launch page](https://www.amlogic.com/News/index248.html) advertises a minimum ten-year product-longevity guarantee | Promising new-platform signal that still needs contract and SKU details for a product program |
| Google Coral | The legacy Edge TPU and PyCoral repositories are archived/read-only; the separate [Coral NPU IP](https://github.com/google-coral/coralnpu) repository is active | Legacy software maintenance risk, not by itself a formal hardware EOL notice; it says nothing about lifecycle of the unrelated open-source IP project |

## Why TOPS is not a buying guide

Peak TOPS can be useful within one vendor and architecture, but cross-vendor comparisons are usually invalid unless all of the following match:

- Numeric precision, including INT8, INT4, FP16 or a mixed mode
- Whether one multiply-accumulate counts as one or two operations
- Dense versus structured-sparse arithmetic
- Input resolution, batch size and model graph
- Accuracy after quantization
- NPU-only latency versus the complete application pipeline
- Memory transfers and host fallback
- Thermal state and sustained, not burst, operation

A detector is a pipeline: image acquisition, resize/letterbox, normalization, device transfer, neural inference, decode, NMS, coordinate scaling and application logic. Vendors frequently publish only the neural-inference middle. [MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/) is valuable because it defines scenarios, quality targets and measured system-level rules instead of comparing marketing arithmetic in isolation.

For YOLO deployments, the minimum credible benchmark record is:

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

Without those fields, two FPS numbers usually measure different things.

## How to choose an NPU for computer vision

Choose in this order, not by the largest number printed on a product page.

1. **Prove graph compatibility.** Compile the exact exported model, not a similarly named zoo checkpoint.
2. **Measure accuracy.** Validate the compiled artifact on the real task dataset after calibration and quantization.
3. **Measure end to end.** Include input conversion, transfers, decode and NMS.
4. **Check software access.** Determine whether the compiler is public, registration-only or available only after a commercial agreement.
5. **Freeze the version matrix.** Record compiler, runtime, driver, firmware and target identifiers.
6. **Check operating-system reality.** Android, Debian, Yocto, Buildroot, RTOS and Windows support are not interchangeable.
7. **Check supply and lifecycle.** A technically excellent chip is a poor production choice if modules, drivers or long-term support are uncertain.
8. **Then compare cost, power and performance.** Those measurements become meaningful only after the same model is correct on both targets.

### Practical starting points by use case

| Use case | Platforms worth evaluating first | Reason |
|---|---|---|
| Raspberry Pi-specific accelerator | Hailo-8L/8 and Sony IMX500 | Official Raspberry Pi products, software images and concrete developer flows |
| General Linux PCIe, M.2 or USB add-on | DEEPX, MemryX, Hailo and Axelera products | Obtainable accelerator form factors, subject to host/driver compatibility |
| Low-cost Linux SBC or camera | Rockchip RK3588/RK3576, Amlogic ADLA where a supported board/BSP is obtainable, AXERA, Canaan K230 or Sunplus SP7350 | Integrated media pipelines and public model/compiler evidence; verify VIM4 V13A+ for the A311D2 NPU |
| Mobile and high-volume Android | Qualcomm QNN/AI Hub, LiteRT, Apple Core ML | Large deployed base and maintained mobile runtimes |
| Industrial Linux and robotics | TI TDA4/AM6xA, NXP i.MX, Renesas RZ/V, Hailo | Long-life embedded products and camera/IO ecosystems |
| Very small endpoint or MCU | STM32N6x7, Infineon PSOC Edge, Himax WiseEye2, Analog Devices MAX7800x, Nuvoton M55M1, Alif/Arm Ethos-U, GAP9 and Syntiant | Tight power and memory envelopes with purpose-built tooling and model-size constraints |
| Configurable FPGA vision | Microchip VectorBlox, Lattice sensAI and AMD Vitis AI targets | Flexible hardware pipelines, but the accelerator configuration, bitstream and model compiler are one versioned system |
| High-throughput PCIe vision | Axelera Metis, Hailo, DEEPX, Mobilint, MemryX, Rebellions or SiMa.ai | Dedicated accelerator products and multi-stream positioning |
| China-centered product ecosystem | Rockchip, AXERA, D-Robotics, SOPHGO, Huawei Ascend, HiSilicon and Cambricon | Strong regional boards, toolchains and model examples |

This table is a research shortlist, not a performance ranking. Procurement, regional availability and the exact model can reverse the order.

## What this means for LibreYOLO

The scalable design is not dozens of unrelated exporters. It is one strict interchange contract plus small vendor compiler and runtime adapters:

```text
LibreYOLO model
    -> deterministic static ONNX
    -> VendorCompiler.compile(model, target, calibration, precision)
    -> native artifact + manifest
    -> VendorRuntime.load() / infer()
    -> shared task-specific decode and Results objects
```

Every native artifact should have a sidecar manifest containing:

- Vendor, chip target and board target
- Compiler, runtime, driver and firmware versions
- Source checkpoint and ONNX checksums
- Input names, shapes, layout, colorspace, normalization and dtype
- Quantization precision, scales where exposed, and calibration-data hash
- Output tensor names, shapes and semantic meaning
- Detection task, class names and whether decode/NMS is on-chip or host-side
- Recorded numerical parity and task-accuracy results

LibreYOLO already provides portable [ONNX export](/docs/export/onnx), [quantization tooling](/docs/export/quantization), a direct but deliberately narrow [RKNN export](/docs/export/rknn), and an honest external-compiler [Hailo workflow](/docs/export/hailo). Under the criteria used in this guide, Amlogic ADLA is a strong next integration candidate: the toolkit is public, the model coverage overlaps LibreYOLO heavily, and the older A311D path can be kept explicitly separate.

The priority after Amlogic should be based on user hardware and the ability to run accuracy validation, not only compiler availability. DEEPX, Sony IMX500, D-Robotics, AXERA, SOPHGO and Qualcomm are technically attractive; TI, NXP, ST and Renesas become especially valuable when industrial partners can provide boards and long-term CI access.

## Watchlist: real silicon without a public integration contract

Excluding a company entirely can make a market map misleading. Including it as "supported" can be worse. These vendors sell, sample or have announced relevant silicon, but the public evidence does not yet establish a reproducible current compiler, artifact and named-model path suitable for a LibreYOLO backend.

| Company | What is real | Why it remains a watchlist entry |
|---|---|---|
| [Samsung](https://semiconductor.samsung.com/processor/automotive-processor/exynos-auto-v920/) | Exynos Auto V920 includes a dual-core NPU advertised at up to 23.1 TOPS | Samsung says its [Neural SDK is no longer provided to third-party developers](https://developer.samsung.com/neural/overview.html); Samsung ONE/Circle does not prove access to the Exynos NPU |
| [Novatek](https://www.novatek.com.tw/en-global/Milestone/aboutus_milestones) | Current corporate milestones name Edge AI and Edge Vision/Imaging AI SoCs; a 2020 milestone mentioned MobileNet, SSD and YOLOv3 | No current public part-number/compiler/artifact/operator/model matrix; historical YOLOv3 is not current SDK proof |
| [Nextchip](https://www.nextchip.com/en/adas/adas.php?idx=5) | APACHE5/NVS2900 and current [APACHE6/NVS3000](https://www.nextchip.com/en/adas/adas.php?idx=7) automotive processors with aiMotive aiWare NPU; the APACHE6 page itself currently states both 12 and 8 TOPS | aiWare Studio and a runtime exist, but no public artifact, quantization guide, operator list or YOLO matrix was found; the conflicting vendor figures should not be silently resolved |
| [Black Sesame Technologies](https://bst.ai/en.html) | Huashan A1000/A2000 and Wudang C-series automotive/edge AI silicon | Public product information exists, but not a self-service compiler, artifact contract or reproducible model zoo |
| [Ingenic](https://en.ingenic.com.cn/products-detail/id-19.html) | T41/T40 camera SoCs with low-bit NPU claims; Magik AI describes PTQ/QAT and graph compilation | No downloadable current compiler, artifact contract or exact vendor YOLO list was located |
| [Fullhan](https://fullhan.com/en/index.php?a=type&c=article&tid=9) | Multiple IPC SoCs advertise 0.5 to 2 TOPS NPUs, while the [MC6880 NVR family](https://fullhan.com/en/index.php?a=type&c=article&tid=48) advertises 4 TOPS | No public NN compiler/runtime/framework/model documentation sufficient for independent reproduction |
| [Goke Microelectronics](http://www.gokemicro.com/News/info.aspx?itemid=437) | A first-party page for GK7606V1/GK7206V1/GK7203V1 smart-camera parts advertises integrated NPU performance, but is served only over legacy HTTP at publication time | No public converter, runtime contract or named-model matrix; the route is OEM-channel oriented |
| [Chengheng Micro](https://en.chenghengmicro.com/) | CH37 advertises 64 INT8 TOPS and FP16/FP32/FP64 modes; the company's [2026 timeline](https://chenghengmicro.com/about.html) says it entered small-batch production | No public SDK, compiler contract or named-model validation was found; treat it as an early-commercial, non-self-service platform rather than reproducible deployment evidence |
| [Bouffalo Lab](https://github.com/bouffalolab/bouffalo_sdk) | BL808 includes the BLAI-100 NPU and current official SDK repositories exist | The maintained SDK does not expose a current official BLAI conversion/example path; surviving flows are legacy or community evidence |

This watchlist is intentionally evidence-driven. A vendor can move into the deployable tables by publishing a current toolchain or evaluation path, a target-specific artifact contract, and at least one named model with an end-to-end recipe.

## What was deliberately not claimed

This article does not say that every named model works from every upstream repository. Vendor zoo models are frequently modified, split at different output nodes or paired with custom postprocessing.

It also does not turn these statements into support claims:

- A compiler imports ONNX.
- A chip advertises an Android NNAPI driver.
- A distributor says the board is "YOLO compatible."
- A community repository runs one fork of one model.
- A model compiles without an error.
- A vendor reports NPU-only FPS without an accuracy result.

Google Tensor phone accelerators and numerous OEM-only custom ASICs are also real, but Google does not expose Tensor as a general self-service compiler target comparable to the platforms above. Company existence, an NPU block diagram or Android acceleration is not enough to invent a LibreYOLO integration claim.

Likewise, cloud accelerators such as Google TPU, AWS Inferentia/Trainium, Microsoft Maia and data-center-only AI cards are outside scope. This guide is about hardware a computer-vision developer can plausibly deploy at the edge.

Model licensing is a separate layer. A chip vendor publishing a YOLO conversion recipe does not grant rights to use or redistribute the upstream weights, training code or compiled derivative. Hardware support, SDK licensing and model licensing must each be checked for the intended product.

## Research methodology and corrections

For every company, the research looked for four primary-source surfaces:

1. A current product page or datasheet identifying the silicon.
2. Compiler and runtime documentation explaining the real deployment path.
3. A model zoo, compatibility table or end-to-end tutorial naming vision models.
4. A lifecycle or access signal showing whether developers can actually obtain the tools.

First-party GitHub organizations count as vendor sources. Board-vendor documentation is labeled as ecosystem evidence when the chip company itself does not publish the same material. Third-party performance claims were excluded from the comparison tables.

The completed draft was then re-audited in separate vendor-cohort and cross-table passes. Those checks revalidated target scope, artifact names, precision, model-versus-postprocessor evidence, announced-versus-shipping status, lifecycle labels, benchmark denominators and every entity count. When two first-party pages conflict, this guide exposes the conflict instead of silently selecting the more convenient value.

This market changes quickly. If you work for one of these companies and a chip, SDK, model list or access status is wrong, send LibreYOLO the exact public documentation URL and version. Corrections supported by primary evidence should replace this text; undocumented marketing assertions should not.

## FAQ

### How many edge AI NPU companies are there?

There is no universal company total because product vendors, IP licensors, smart sensors, GPUs and private automotive ASICs overlap. This non-exhaustive guide tracks 69 companies and platform ecosystems with relevant edge AI chips, accelerator platforms, licensable NPU IP or credible watchlist silicon as of August 2026.

### What is an NPU?

A neural processing unit is specialized hardware for neural-network operations such as convolutions and matrix multiplication. Vendor terms include NPU, AIPU, BPU, KPU, DLA, HTP, TPU and MLA, and their compiled models are generally not portable between companies.

### Which NPU companies officially support YOLO models?

Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 through Raspberry Pi's official AI Camera repository, STMicroelectronics, Renesas, Lattice, Himax, Microchip and several others have first-party model-zoo entries, tutorials or validated results for at least one YOLO generation. The exact generation, task, chip target and SDK version still matter.

### Can any ONNX model run on any NPU?

No. ONNX is an interchange format, not a hardware compatibility guarantee. An NPU compiler must support every operator, tensor shape and data type in the graph. Static shapes, graph cuts, custom operations and CPU fallback are common.

### What is the best NPU for YOLO?

There is no universal winner. Hailo, Rockchip, Axelera AI, DEEPX and Amlogic have strong documented YOLO coverage, while Qualcomm has exceptional device reach. Choose using accuracy after quantization, full-pipeline latency, sustained power, price, availability, SDK access and operating-system support.

### Why can NPU TOPS numbers not be compared directly?

Vendors may count different precisions, sparse operations, multiply-accumulates and peak-utilization assumptions. TOPS omits memory traffic, unsupported operators, preprocessing, postprocessing and host overhead. Compare the same model, precision, resolution, accuracy and full pipeline.

### What is NPU calibration?

Calibration runs representative, usually unlabeled deployment images through the model so the compiler can estimate activation ranges for INT8 or another low-precision format. Random or unrepresentative images may still produce a compiled artifact while damaging task accuracy.

### Does LibreYOLO support edge NPUs?

LibreYOLO exports ONNX and several runtime formats, has a direct RKNN compiler path for selected Rockchip models, and documents the external Hailo compilation flow. Every other vendor-native artifact still requires its vendor SDK and should be described as an integration candidate until it is compiled, validated and run on hardware.
