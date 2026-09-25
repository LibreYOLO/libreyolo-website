---
title: "69 Edge-AI-NPU-Unternehmen: Chips, SDKs und YOLO für Computer Vision (2026)"
description: "Ein quellenbasierter Überblick über 69 Edge-AI-Unternehmen und NPU-Ökosysteme: Chips, SDKs, Artefakte, Quantisierung sowie dokumentierte Computer-Vision- und YOLO-Unterstützung."
date: 2026-08-15
author: Xuban
tags: [edge-ai, npu, computer-vision, yolo, hardware, ai-accelerators, amlogic, hailo, rockchip]
faq:
  - q: "Wie viele Edge-AI-NPU-Unternehmen gibt es?"
    a: "Eine allgemeingültige Zahl gibt es nicht, da Produkthersteller, IP-Lizenzgeber, intelligente Sensoren, GPUs und proprietäre Automotive-ASICs überlappen. Dieser nicht abschließende Leitfaden erfasst mit Stand August 2026 insgesamt 69 Unternehmen und Plattform-Ökosysteme mit relevanten Edge-AI-Chips, Beschleunigerplattformen, lizenzierbarer NPU-IP oder glaubwürdigen Kandidaten auf der Beobachtungsliste."
  - q: "Was ist eine NPU?"
    a: "Eine Neural Processing Unit ist spezialisierte Hardware für Operationen neuronaler Netze wie Faltungen und Matrixmultiplikationen. Hersteller verwenden unter anderem die Bezeichnungen NPU, AIPU, BPU, KPU, DLA, HTP, TPU und MLA. Kompilierte Modelle lassen sich in der Regel nicht zwischen Unternehmen übertragen."
  - q: "Welche NPU-Unternehmen unterstützen YOLO-Modelle offiziell?"
    a: "Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 über das offizielle AI-Camera-Repository von Raspberry Pi, STMicroelectronics, Renesas, Lattice, Himax, Microchip und mehrere weitere Unternehmen haben eigene Model-Zoo-Einträge, Tutorials oder validierte Ergebnisse für mindestens eine YOLO-Generation. Die konkrete Generation, Aufgabe, Chipziel und SDK-Version sind weiterhin entscheidend."
  - q: "Lässt sich jedes ONNX-Modell auf jeder NPU ausführen?"
    a: "Nein. ONNX ist ein Austauschformat und keine Garantie für Hardwarekompatibilität. Ein NPU-Compiler muss jeden Operator, jede Tensorform und jeden Datentyp im Graphen unterstützen. Statische Formen, Graph-Aufteilungen, benutzerdefinierte Operationen und ein CPU-Fallback sind häufig."
  - q: "Welche NPU eignet sich am besten für YOLO?"
    a: "Einen allgemeingültigen Sieger gibt es nicht. Hailo, Rockchip, Axelera AI, DEEPX und Amlogic bieten eine umfangreich dokumentierte YOLO-Unterstützung, während Qualcomm eine außergewöhnlich große Gerätebasis hat. Entscheide anhand der Accuracy nach der Quantisierung, der Latenz der gesamten Pipeline, der Dauerleistung, des Preises, der Verfügbarkeit, des SDK-Zugangs und der Betriebssystemunterstützung."
  - q: "Warum lassen sich NPU-TOPS-Werte nicht direkt vergleichen?"
    a: "Hersteller können unterschiedliche Präzisionen, Sparse-Operationen, Multiply-Accumulates und Annahmen zur Spitzenauslastung zählen. TOPS berücksichtigen weder Speicherverkehr noch nicht unterstützte Operatoren, Vorverarbeitung, Nachverarbeitung oder Host-Overhead. Vergleiche dasselbe Modell, dieselbe Präzision, Auflösung, Accuracy und vollständige Pipeline."
  - q: "Was ist NPU-Kalibrierung?"
    a: "Bei der Kalibrierung werden repräsentative, meist nicht annotierte Einsatzbilder durch das Modell geführt, damit der Compiler Aktivierungsbereiche für INT8 oder ein anderes Format mit niedriger Präzision schätzen kann. Zufällige oder nicht repräsentative Bilder können zwar ein kompiliertes Artefakt ergeben, aber die Accuracy für die Aufgabe verschlechtern."
  - q: "Unterstützt LibreYOLO Edge-NPUs?"
    a: "LibreYOLO exportiert ONNX und mehrere Runtime-Formate, bietet für ausgewählte Rockchip-Modelle einen direkten RKNN-Compilerpfad und dokumentiert den externen Hailo-Kompilierungsablauf. Für jedes andere herstellerspezifische Artefakt ist weiterhin das SDK des Herstellers nötig. Bis es kompiliert, validiert und auf Hardware ausgeführt wurde, sollte es als Integrationskandidat bezeichnet werden."
---

**Es gibt keinen einheitlichen NPU-Markt. Dieser nicht abschließende Leitfaden erfasst 69 Unternehmen und Plattform-Ökosysteme: 51 Anbieter mit auslieferbaren Chips oder Plattformen, neun NPU-IP-Anbieter und neun klar gekennzeichnete Unternehmen auf der Beobachtungsliste.** Sie decken mehrere inkompatible Beschleunigerklassen und fast ebenso viele Compiler-Stacks ab. Die meisten versprechen denselben Ablauf: Modell exportieren, quantisieren, kompilieren, ein proprietäres Artefakt auf ein Board kopieren und eine Hersteller-Runtime aufrufen. Die Details entscheiden, ob das einen Nachmittag oder ein Quartal Entwicklungsarbeit kostet.

Dieser Leitfaden ordnet die Unternehmen ein, die 2026 glaubwürdige Computer-Vision-Hardware anbieten. Er erfasst relevante Chips, SDK- und Compilernamen, Deployment-Artefakte, öffentlichen Zugangsstatus und die vom Anbieter tatsächlich dokumentierten Modelle. Besonderes Augenmerk gilt außerdem Amlogic, dessen neuerer ADLA-Stack sich wesentlich vom älteren A311D-NPU-Ökosystem unterscheidet.

> **Umfang der Recherche:** Der Quellenstand ist August 15, 2026. Aussagen zu namentlich genannten Modellen beruhen, sofern nicht anders gekennzeichnet, auf Produktseiten, Dokumentation, Model Zoos, Repositories oder Tutorials der Hersteller. Beworbene TOPS-Werte stammen von den Herstellern und sind keine unabhängig vergleichbaren Benchmark-Ergebnisse. Dies ist ein Entwicklerleitfaden, keine Anlageberatung und kein bezahltes Ranking.

**Schnellnavigation:** [Unternehmenstabellen](#npu-companies-and-platforms-at-a-glance) | [Amlogic im Detail](#amlogic-two-npu-generations-not-one) | [Anbieterprofile](#the-strongest-public-deployment-ecosystems) | [Tabelle kompilierter Artefakte](#what-you-actually-deploy-the-artifact-lock-in-table) | [Zugang und Lebenszyklus](#tool-access-and-lifecycle-signals) | [LibreYOLO-Roadmap](#what-this-means-for-libreyolo)

## Wichtigste Erkenntnis

Die Hardware ist fragmentiert, das Deployment-Muster dagegen bemerkenswert einheitlich:

```text
PyTorch checkpoint
    -> ONNX or TFLite interchange graph
    -> representative calibration data
    -> vendor quantizer and graph compiler
    -> chip-specific binary or model package
    -> vendor runtime on the target
    -> application preprocessing, decode, NMS and rendering
```

[ONNX unterstützt ausdrücklich Runtimes, Codegeneratoren und Hardwareimplementierungen](https://onnx.ai/onnx/repo-docs/IR.html), doch eine ONNX-Datei ist nur die Übergabe. Daraus folgt nicht, dass sich jeder ONNX-Operator auf jeden Beschleuniger abbilden lässt. Statische Eingabeformen, Einschränkungen bei unterstützten Operatoren, Quantisierungsregeln und Host-Fallback bestimmen, was tatsächlich auf der NPU läuft.

Der Software-Stack gehört daher zum Chip. Eine nominell schnelle NPU mit einem eingeschränkten oder fragilen Compiler kann ein schlechteres Deployment-Ziel sein als ein kleineres Gerät mit öffentlicher Toolchain, Model Zoo, Simulator und stabiler Runtime.

## Was „unterstützt“ in diesem Leitfaden bedeutet

In Herstellertexten wird der Begriff *Unterstützung* sehr großzügig verwendet. Dieser Artikel unterscheidet vier Evidenzstufen:

| Evidenzstufe | Was sie belegt | Was sie nicht belegt |
|---|---|---|
| **Validiertes Modell** | Der Hersteller veröffentlicht einen chipspezifischen Modelleintag, ein Ergebnis oder eine Kompatibilitätstabelle | Dein modifizierter Checkpoint behält dieselbe Accuracy |
| **Offizielles Beispiel** | Der Hersteller stellt ein End-to-End-Tutorial oder eine Demo für ein benanntes Modell bereit | Andere Größen, Aufgaben oder Generationen lassen sich kompilieren |
| **Compilerfunktion** | Das SDK importiert ein Framework oder stellt unterstützte Operatoren bereit | Ein bestimmter YOLO-Graph funktioniert durchgängig |
| **Marketing oder eingeschränkter Zugang** | Das Produkt ist für Vision vorgesehen und ein privates SDK existiert | Öffentlich reproduzierbare Modellkompatibilität |

Diese Unterscheidung ist wichtig. „Importiert ONNX“ ist nicht gleichbedeutend mit „unterstützt YOLO11-Segmentierung“. Ein Modell kann eingelesen werden, dann aber auf eine CPU zurückfallen, mit falschen Ausgabewerten kompiliert werden, den Beschleunigerspeicher überlasten oder bei der Quantisierung Accuracy verlieren.

## NPU-Unternehmen und Plattformen im Überblick

Die Tabellen führen bewusst SoCs, diskrete Beschleuniger, intelligente Sensoren und angrenzende GPU-/FPGA-Ziele gemeinsam auf. Sie konkurrieren um dieselbe Deployment-Entscheidung, auch wenn Hersteller Bezeichnungen wie NPU, AIPU, BPU, KPU, DLA, HTP, TPU oder MLA verwenden.

### Eingebettete SoCs, intelligente Sensoren und Industrieprozessoren

| Unternehmen | Relevante Chips oder Plattformen | SDK, Compiler und Artefakt | Öffentlich dokumentierte Computer-Vision-Evidenz | Zugang |
|---|---|---|---|---|
| [Amlogic](https://github.com/Amlogic-NN/amlnn-toolkit) | A311D2, S928X, S905X5/S905D5, A311Y3, C308L/C302X/C302X2, T968D4, C305X2 und A123X | AMLNN Toolkit und `libnnsdk.so`, kompiliertes `.adla`; separater älterer Acuity-`.nb`-Stack für A311D | Der [Model Playground](https://github.com/Amlogic-NN/amlnn-model-playground) dokumentiert YOLOv5/6/7/8/10/11, YOLOX, YOLOE, YOLO-World, PP-YOLOE, Segmentierung, Pose und OBB auf A311D2, S905X5, A311Y3, C305X2 und A123X. Die übrigen aufgeführten Chips sind Compilerziele, keine Einträge in dieser Support-Matrix. | Öffentliches GitHub und Wheels; kompatibles BSP/Driver weiterhin erforderlich |
| [Rockchip](https://github.com/airockchip/rknn-toolkit2) | RK3562/3566/3568, RK3576, RK3588, RV1126B | RKNN-Toolkit2, RKNN Runtime, `.rknn` | YOLOv5/6/7/8/10/11, YOLOX, YOLO-World, PP-YOLOE, Segmentierung, Pose und OBB im [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) | Öffentliches GitHub; binäre Wheels |
| [Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | Snapdragon- und Dragonwing-Plattformen, QCS6490, QCS8550 und Dragonwing IQ-9075 | QAIRT/QNN, SNPE und AI Hub; QNN-Kontext-Binärdateien oder DLC | Die [AI-Hub-Modellsammlung](https://github.com/qualcomm/ai-hub-models) umfasst YOLOv3/5/6/7/8/9/10/11/26, YOLOX, YOLO-World, YOLOR, RF-DETR sowie Modelle für Erkennung, Segmentierung und Pose. | Dokumentation öffentlich; SDK- und Kontoanforderungen variieren |
| [Texas Instruments](https://github.com/TexasInstruments/edgeai-tidl-tools) | AM62A, AM67A, AM68A, AM69A und TDA4x; Vorschau TDA54-Q1 | Processor SDK Edge AI und TIDL | Der [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo) veröffentlicht optimierte Modelle für Erkennung, Segmentierung, Pose und Klassifikation auf aktuellen AM6xA-/TDA4-Pfaden. Dies ist noch keine Zielmatrix für das TDA54-Q1 in der Vorschau. | Öffentliches GitHub plus Processor SDK |
| [NXP](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | i.MX 8M Plus, i.MX 93, i.MX 95 und übernommene Kinara Ara-1/Ara240 | eIQ mit TIM-VX, Vela, Neutron Converter und Ara SDK | Der [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) enthält YOLOv4-tiny/v8, NanoDet, CenterNet, FastestDet, SSD Lite und YOLACT als Assets oder Rezepte; der YOLOv5-Eintrag besteht nur aus Dokumentation. | Mischung aus öffentlichen und kontogeschützten Komponenten |
| [STMicroelectronics](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) | STM32N6x7-Varianten, einschließlich STM32N657/647, mit Neural-ART-Beschleuniger | STM32Cube AI Studio und ST Edge AI Core | Das aktuelle Services-Repository führt Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 und ST-YOLOX sowie Pose und Segmentierung auf. | Öffentliche Tools und Repositories |
| [Infineon](https://documentation.infineon.com/psocedge/) | PSOC Edge E83/E84 mit Cortex-M55 plus Ethos-U55; separater NNLite-Beschleuniger auf der M33-Seite | ModusToolbox, [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter), TFLite Micro und Arm Vela | In Architekturunterlagen werden MobileNetV1/V2 als repräsentative Kernels genannt, und das E84-AI-Kit enthält eine Kamera. Eine YOLO-Validierungsmatrix für PSOC Edge von Infineon selbst wurde jedoch nicht gefunden. | Öffentliche Dokumentation, SDK-Komponenten und aktuelle E84-Evaluierungskits |
| [Renesas](https://github.com/renesas-rz/rzv_drp-ai_tvm) | RZ/V2L, V2M, V2MA, V2H und V2N | DRP-AI Translator, DRP-AI TVM und RUHMI | Die [Modellliste für RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) validiert YOLOv5/v8/v11/26-Größen, YOLOX sowie Pose- und Segmentierungsvarianten. | Öffentliches GitHub plus Board-SDK |
| [Sony](https://www.aitrios.sony-semicon.com/edge-ai-devices/imx500) | Intelligenter Bildsensor IMX500 und Raspberry Pi AI Camera | Edge-MDT, Model Compression Toolkit und IMX500-Packer; `.rpk`-Paket | Der [Raspberry-Pi-IMX500-Zoo](https://github.com/raspberrypi/imx500-models) umfasst YOLOv8n, YOLO11n, EfficientDet Lite, NanoDet+ und SSD. | Öffentlicher Raspberry-Pi-Ablauf; Sony-Tooling-Bedingungen gelten |
| [Ambarella](https://www.ambarella.com/developer/model-garden/) | CV72/CV75-, CV5/CV52- und CV3-AD-Familien | Cooper Developer Platform, CVflow-Compiler und Runtime-DAGs | Der öffentliche Model Garden nennt YOLOX-S, RTMDet-nano, DeepLabV3+, TopFormer, OWL-ViT und LLaVA OneVision. | Vorwiegend nur für Partner zugänglich |
| [Synaptics](https://developer.synaptics.com/docs/sl/overview) | Astra SL1600/SL1680; SL2611/13/15/17/19; separate SR100-MCU-Familie | SyNAP `.synap` auf SL16xx; Torq/IREE `.vmfb` auf SL261x; separates SR-SDK mit Ausrichtung auf Ethos-U55 | Offizieller YOLOv8n/v8s-SyNAP-Benchmark und SL261x-Torq-YOLOv8-Beispiel; SR-Evidenz muss separat bewertet werden. | Entwicklerportal; einige Downloads zugangsbeschränkt |
| [MediaTek](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) | Genio 360/360P/420/520/720 mit NP8 und MDLA 5.3; Genio 510/700/1200 mit älterem NP6/MDLA | NeuroPilot Converter, `ncc-tflite`, Neuron Runtime und `.dla`; ONNX-Runtime-Pfad auf ausgewählten neueren Systemen | Der offizielle IoT AI Hub veröffentlicht Modellseiten und Benchmarks für YOLOv5 und YOLOv8 sowie Klassifikations- und Gesichtsmodelle. | Öffentliche Yocto-Dokumentation; wichtige NP8-Bundles und Android-Material erfordern direkten Kundenzugang/NDA |
| [Allwinner](https://docs.aw-ol.com/v853/en/npu/dev_npu/) | V853-Vision-SoC mit Vivante-NPU mit 1-TOPS | Acuity/Pegasus-Konvertierung und `viplite`-Runtime | Die offiziellen V853-Unterlagen nennen YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet sowie Gesichts- und Personennetzwerke. | Öffentliche Dokumentation; SDK-Paketdownload kann ein Konto erfordern |
| [Canaan/Kendryte](https://github.com/kendryte/nncase) | Aktuelle K230/K230D; ältere K210/K510-KPU | `nncase`-Compiler und Runtime; `.kmodel` | Der nncase-Leitfaden für K230 behandelt YOLOv5s-Kompilierung, Simulation und Runtime. Der offizielle Demo-Katalog und das aktuelle [CanMV-Changelog](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) dokumentieren Aufgaben mit YOLOv8/11/26. | Öffentlicher Compiler/PyPI; Chip-Plug-in binär |
| [Alif Semiconductor](https://alifsemi.com/support/kits/ensemble-e7appkit/) | Ensemble E7 mit zwei ML-Beschleunigern, darunter Ethos-U55; [E8](https://alifsemi.com/ensemble-e8-series/) mit einem Ethos-U85 und zwei Ethos-U55-NPUs | Deployment auf Basis von Arm Vela und TFLite Micro | Alif veröffentlicht einen INT8-Benchmark für einen YOLO-Fastest-Gesichtsdetektor auf Familienebene, aber keine breite, zielgenaue Matrix für moderne YOLO-Modelle. | Öffentliche Dokumentation und Ressourcen für Evaluierungskits |
| [Himax](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) | HX6538 WiseEye2 Endpoint-AI-MCU mit Cortex-M55 und Ethos-U55 | TFLite Micro plus Arm Vela, mit CMSIS-NN und Reference-Kernel-Fallback | Die offiziellen [WiseEye2-Beispiele](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) enthalten YOLOv8n für Erkennung, Pose und Klassifikation, YOLO11n-Erkennung, Face Mesh und PeopleNet. | Öffentliches GitHub, Dokumentation und erhältlich über Partner-Boards |
| [Analog Devices](https://www.analog.com/en/products/max78002.html) | AI-MCUs MAX78000 und MAX78002 mit stromsparenden CNN-Beschleunigern | [`ai8x-training`](https://github.com/analogdevicesinc/ai8x-training), `ai8x-synthesis`/`izer` und MSDK; generierter C-Code und Gewichte | Offizielle Unterlagen behandeln Gesichtserkennung/-identifikation, RetinaNet, Visual Wake Words, Klassifikatoren und Action Recognition. Eine YOLO-Bereitstellung von Analog Devices selbst wurde nicht gefunden. | Öffentliche GitHub-Tools, Dokumentation und Evaluierungsboards |
| [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo) | RDK-X3/X5/Ultra und neuere BPU-Boards der S100-Serie | OpenExplorer/Algorithm Toolchain und `hbm_runtime`; X5 `.bin`, aktuelles `rdk_s` `.hbm` | Der aktuelle Branch `rdk_x5` dokumentiert für RDK X5 YOLOv5/v5u/v8/v9/v10/11/12/13/26, YOLOE, YOLO-World, Segmentierung, Pose, Klassifikation, OCR und CLIP. X3 und S-Serie verwenden eigene Branches. | Öffentliches GitHub; einige Toolchain-Pakete sind plattformspezifisch |
| [AXERA](https://github.com/AXERA-TECH/ax-samples) | AX650, AX637, AX630C, AX620Q und AX615 | Pulsar2-Compiler und AXEngine; `.axmodel` | Aktuelle offizielle Beispiele decken YOLOv5/6/7/8/9/10/11/13/26, YOLOX und YOLO-World ab. Aufgaben und Ziele variieren je nach Chip. | Öffentliche Beispiele und Dokumentation; Compiler separat verteilt |
| [SOPHGO](https://github.com/sophgo/tpu-mlir) | BM1684/1684X/1688/1690 und CV186X/CV18xx | TPU-MLIR und SOPHON-Runtime; `.bmodel` oder `.cvimodel` | Offizielle Demos decken YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, OBB, Segmentierung, Gesicht, SAM und OCR ab. | Open-Source-Compiler plus Hersteller-Runtime |
| [Huawei Ascend](https://www.hiascend.com/en/software/cann) | Ascend 310/310P/310B und Atlas-200I/300I-Edge-Produkte | CANN, ATC-Compiler und AscendCL; `.om` | Der offizielle [Ascend ModelZoo](https://github.com/Ascend/modelzoo) umfasst YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN und Segmentierungsmodelle. | Öffentliche Dokumentation; CANN-Pakete und Kernels müssen zum Ziel passen |
| [Cambricon](https://github.com/Cambricon/magicmind_cloud) | MLU370-X4/S4 in der zitierten öffentlichen Matrix; MLU270 ist ältere Hardware und nicht darin enthalten | Neuware und MagicMind | Die MagicMind-1.7-Matrix des Repositorys führt YOLOv3/4/5/7/8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN und Segmentierungsmodelle auf. | Historische Modellbeispiele öffentlich; aktueller SDK-/Containerzugang beschränkt |
| [Sunplus](https://sunplus.atlassian.net/wiki/spaces/C3/pages/1971126471/SP7350%2BSpecification) | SP7350/C3V mit beworbenen etwa 4.1 bis 4.6 TOPS | Vivante-Acuity-NPU-Docker, SNNF-Runtime und `.nb` | Die offizielle Dokumentation bietet YOLOv5 und ein angepasstes End-to-End-[YOLOv8-Deployment](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350). | Öffentliche Quellen/Dokumentation und Board-Ökosystem |
| [ESWIN Computing](https://www.eswincomputing.com/en/news/info/96.html) | EIC7700/EIC7700X und Dual-Die-[EIC7702/EIC7702X](https://www.eswincomputing.com/en/news/info/104.html)-RISC-V-SoCs | ENNP: EsQuant, aktueller EsAAC-Compiler, Simulator und ESSDK-Runtime; `.model`; ältere Dokumentation nutzte `ennc-compile` | Die von Milk-V bereitgestellte ENNP-Dokumentation enthält ein End-to-End-[YOLOv3-Tutorial](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3) sowie Beispiele für MobileNetV2 und ResNet. | Mischung aus Hersteller- und Board-Dokumentation; regionale Downloads |
| [Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m55-mcus/m55m1-series/index.html) | M55M1-MCU mit Ethos-U55-256 | TFLite Micro, Arm Vela und NuEdgeWise/NuML | Die offiziellen [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise)-Beispiele nennen YOLOv8-nano, YOLOX-nano, YOLO Fastest, SSD-MobileNet und Klassifikationsmodelle. | MCU-Toolchain überwiegend öffentlich |
| [Realtek](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | AmebaPro2 RTL8735B / AMB82-mini-Kameraplattform | Arduino-/FreeRTOS-SDK, VoE und NeuralNetwork API; `.nb` | Enthaltene Modelle sind YOLOv3-tiny, YOLOv4-tiny, YOLOv7-tiny, SCRFD und MobileFaceNet. | Runtime öffentlich; Zugang zum [benutzerdefinierten Converter](https://ameba-doc-arduino-sdk.readthedocs-hosted.com/en/latest/FAQ/offline_ai_model_conversion_steps.html) nur auf Anfrage |
| [Telechips](https://docs.topst.ai/product/p/ai) | TCC7500 / TOPST-AI-Board, beworben mit 8 TOPS | TC-NN-Toolkit / Enlight SDK; `.enlight`-Zwischenformat und kompiliertes Deployment-Bundle | Ein [offizielles TOPST-Projekt](https://docs.topst.ai/blog/31) hat YOLOv8s bereitgestellt. Die Konvertierung von UFLD v1/v2 scheiterte an nicht unterstützten Layers; vorgeschlagen wurde nur ein Workaround mit Aufteilung und Nachverarbeitung. [YOLOv4](https://community.topst.ai/t/segmentation-fault/358) und ein [weiterer YOLOv8-Ablauf](https://community.topst.ai/t/segmentation-fault-error/415) tauchen in Community-Support-Threads auf. | Board-Dokumentation öffentlich; Compiler-/Operator-Downloads oft genehmigungspflichtig |
| [T-Head](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | TH1520, beworben mit 4-TOPS-INT8 auf LicheePi 4A | HHB-Compiler und CSI-NN2/SHL; `hhb.bm` plus generierter Code und Parameter | Offizielle Beispiele des Board-Anbieters führen YOLOv5n/s und MobileNetV2 auf der NPU aus. | Öffentliche Beispiele; Unsicherheit wegen Alter und Wartungsstatus der Toolchain |
| [SigmaStar](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | Aktueller Smart-Kamera-SoC SSU9383CM und ältere IPU-Familien | Aktuelle MI_IPU-Runtime; eine [ältere SGS_IPU-Toolchain](https://wx.comake.online/doc/doc/SigmaStarDocs-SSC9381G_9351_Pudding-ULS00V040-20210913/customer/development/dla/tools.html) erzeugte `.sim`, daraus `sgsimg.img` | Ein [älterer offizieller SDK-Postprozessor](https://wx.comake.online/doc/doc/Sigmastar_SDK_v1.2.2/module/User_Guide/Common/SigmaStar_Post_Processing_Module.html) nennt SSD und YOLOv1/2/3. Die öffentliche Kompatibilität dieser Modelle und Artefakte mit SSU9383CM ist nicht verifiziert. | Aktueller Chip, doch öffentlich belegte Compiler-/Modellpfade stammen aus unterschiedlichen Generationen |
| [HiSilicon](https://www.hisilicon.com/cn/products/smart-vision/machine-vision/hi3516cv610) | Hi3516CV610/DV500, Hi3519DV500 und Hi3403V100 Smart-Vision-SoCs | ATC zu `.om` mit NNN-/SVP-NNN-Board-Runtime | Die zielgenaue HiSpark-Matrix, insbesondere für Hi3403 und Hi3591P, nennt YOLOv3 bis YOLO11, Pose, Segmentierung, OBB, OCR und Tiefenmodelle. Sie belegt keine Validierung auf jedem SoC dieser Tabellenzeile. | Aktiv und von Ascend getrennt; Repository-Modelle sind als nicht kommerziell gekennzeichnet |

### Dedizierte Edge-Beschleuniger und Module

| Unternehmen | Relevante Chips | SDK und Artefakt | Öffentlich dokumentierte Modellevidenz | Zugang |
|---|---|---|---|---|
| [Hailo](https://github.com/hailo-ai/hailo_model_zoo) | Hailo-8, Hailo-8L, Hailo-10H und Hailo-15-Familie | Dataflow Compiler und HailoRT; `.hef` | YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 und YOLO26 sowie YOLOX, DAMO-YOLO, SSD, EfficientDet, Segmentierung, Pose und OBB, jeweils mit generationsspezifischen Modelltabelle | Model Zoo öffentlich; Compiler über Developer Zone |
| [Axelera AI](https://docs.axelera.ai/sdk/reference/models/model-zoo/) | Ausgelieferte Metis-Produkte; angekündigte Produktlinien Europa und Titania | Öffentlicher Voyager SDK; Alpha-Pipeline-Builder `.axm`/`.axe`, klassisches `.axmodel`-Bundle | Validierte YOLOv3/5/7/8/9/10/11/26, YOLOX, YOLO-NAS, OBB, Pose, Segmentierung und zahlreiche Nicht-YOLO-Modelle | SDK öffentlich auf GitHub; Kundensupport kontogeschützt |
| [DEEPX](https://developer.deepx.ai/modelzoo/) | DX-M1 und DX-M1M | DXNN SDK, DX-COM und DX-RT; `.dxnn` | Der Model Zoo führte laut Prüfung mit Stichtag August 15, 2026 unter DX-COM 2.4.0/DX-RT 3.4.0 insgesamt 354 Einträge, darunter YOLOv3 bis YOLO11 und YOLO26, YOLOX, SSD, EfficientDet, NanoDet, Segmentierung, Pose und OBB. | Öffentliche Entwicklerressourcen und Suite |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | MX3-Beschleuniger und Mehrchip-Module | MemryX SDK, Neural Compiler und Runtime; `.dfp` | Offizielle Beispiele und Release Notes decken Erkennung, Segmentierung und Pose ab, darunter YOLOv10, YOLO11 und YOLO26. | Öffentliche Dokumentation und SDK |
| [Kneron](https://doc.kneron.com/docs/) | KL520, KL530, KL630, KL720 und KL730 mit aktueller Compiler-Dokumentation; KL830 erscheint in einigen PLUS-APIs, aber nicht in der aktuellen Compiler-Zielliste | Kneron PLUS und Model Toolchain; `.nef` auf den dokumentierten Compilerzielen | Das offizielle [YOLO-Verfahren](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) konzentriert sich auf Tiny-YOLOv3; weiteres Material behandelt YOLOv5. | Öffentliche Dokumentation; Toolpakete/Downloads variieren je nach Chip |
| [SiMa.ai](https://docs.sima.ai/pages/palette/modelsdk.html) | MLSoC und produktive Modalix-Plattform mit 50 TOPS | Palette, ModelSDK, MLA Compiler und ModelExecutor | Öffentliche Releases validieren YOLOv7/v8, YOLOX, Pose-/Segmentierung, DETR, Mask R-CNN und EfficientDet. | Dokumentation öffentlich; Produkt-SDK kommerziell |
| [EdgeCortix](https://www.edgecortix.com/en/hardware) | SAKURA-II, beworbener 60-TOPS-Beschleuniger als M.2- und PCIe-Produkt | MERA-Compiler und Framework | Der Hersteller beschreibt Unterstützung von Vision bis generativer KI, veröffentlicht aber keine hinreichend genaue aktuelle YOLO-Kompatibilitätsmatrix. | Kommerzielle Teststellung/Bestellanfrage; Validierung über Partner |
| [BrainChip](https://brainchip.com/metatf-dev-tools/) | Ausgelieferter AKD1500-Coprozessor/M.2-Modul; ältere AKD1000-Plattformen; Akida-2-IP | MetaTF-Pakete: `akida-models`, `quantizeml`, `cnn2snn` und `akida`-Runtime | Die Akida-2-Modellkarte nennt einen AkidaNet0.5-YOLOv2-Detektor, CenterNet, AkidaUNet und Gesichtserkennung. Diese Ergebnisse belegen nicht automatisch AKD1500. | Zentrale Python-Pakete öffentlich; Hardwarezuordnung zielspezifisch |
| [Blaize](https://www.blaize.com/products/) | P1600-, Pathfinder- und Xplorer-Produkte | Picasso SDK, NetDeploy und AI Studio | Vision ist ein Zielmarkt, aber es wurde keine prüfbare öffentliche Kompatibilitätsmatrix mit konkreten Modellen gefunden. | Kommerziell/zugangsbeschränkt |
| [Google Coral](https://github.com/google-coral/edgetpu) | Ältere Edge-TPU-USB-, PCIe-, M.2- und Dev-Board-Produkte; separate aktive Open-Source-[Coral-NPU-IP](https://github.com/google-coral/coralnpu) | Älterer Edge TPU Compiler/`libedgetpu`/PyCoral; die neue RISC-V-Coral-NPU stellt IP, RTL/Simulation und ELF-Beispiele bereit | Ältere Beispiele konzentrieren sich auf SSD MobileNet, Klassifikation, DeepLab und MoveNet. Für die neue IP gibt es keine öffentliche YOLO-Deployment-Matrix; sie ist kein direkter Nachfolger des Edge TPU. | Zentrale ältere Repositories archiviert; neue Coral-NPU-IP aktiv entwickelt |
| [Lattice Semiconductor](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E und Avant-X-FPGAs | sensAI Studio, Neural Network Compiler und konfigurierbare Beschleuniger-IP | Die aktuelle Compilertabelle führt YOLOv1, YOLOv5, YOLOv8 und YOLO11 in zielabhängigen Modi sowie SSD, MobileNetV2-SSD, ResNet und ENet auf. | Compiler/Handbücher öffentlich; IP-Bedingungen variieren, Advanced CNN Accelerator ist kommerziell |
| [Mobilint](https://www.mobilint.com/sdk-qb) | REGULUS mit 10 TOPS und ARIES MLA100 mit 80 TOPS | qb SDK; INT8-Compiler und `.mxq` | Der öffentliche [Model Zoo](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) führt YOLOv3/5/7/8/9/10/11/12/26 für Erkennung sowie Varianten für Segmentierung, Pose und OBB auf. | Öffentliche Dokumentation/Modelle; SDK und Hardware kommerziell |
| [Rebellions](https://rebellions.ai/developers/) | Aktive ATOM+ CA22 und ATOM-Max CA25; EoL ATOM CA02/ATOM+ CA12; Toolstatus von ATOM-Lite CA21 unklar | RBLN SDK Compiler/Runtime/Profiler; `.rbln` | Offizielle Support-Releases nennen YOLOv3, YOLOv5/6/7/8-Familien und aktuelle YOLOv8-Tutorials. | Dokumentation öffentlich; Compiler-Wheel erfordert Portalzugang |
| [FuriosaAI](https://furiosa.ai/warboy) | Auf Vision ausgelegtes Warboy Gen1; separate RNGD-Generation | Furiosa SDK; INT8-`.enf` auf Warboy, separates `.fxb` auf RNGD | Der Warboy-Zoo dokumentiert SSD, YOLOv5M/L und YOLOv7-w6-pose. Die RNGD-Roadmap kennzeichnet YOLOv8m-Unterstützung für 2024 Q4 als abgeschlossen, bietet aber keine detaillierte aktuelle öffentliche Vision-Matrix. | IAM-/Kontozugang; beide Generationen getrennt betrachten |

### Angrenzende Plattformen, die Entwickler mit NPUs vergleichen

| Unternehmen | Hardware | Deployment-Stack | Warum die Plattform in den Vergleich gehört |
|---|---|---|---|
| [NVIDIA](https://github.com/NVIDIA-AI-IOT/deepstream_tools/tree/main/yolo_deepstream) | Jetson-Orin-GPU plus DLA; Jetson Thor GPU ohne DLA | JetPack, TensorRT und DeepStream; `.engine` | Äußerst ausgereiftes Vision-Deployment. Offizielle DeepStream-Tools dokumentieren YOLOv4/v7/v8/v9/11 einschließlich einer YOLO11-OBB-Konfiguration. Viele Ergebnisse nutzen die GPU, und nicht unterstützte Orin-DLA-Layers können auf die GPU zurückfallen. |
| [AMD](https://vitisai.docs.amd.com/en/6.2/) | Kria-/ältere DPU-Ziele; Vitis AI 6.2 GA für Versal AI Edge Gen1 VEK280/VE2802 und Gen2 VEK385; separate Ryzen-AI-Client-NPUs | Älteres `.xmodel`; zielgebundener NPU-Snapshot für Gen1; kompilierter Cache-Ordner oder Produktionspaket `.rai` für Gen2 | Vitis AI veröffentlicht Vision-Material, doch ältere DPU, Versal Gen1, Versal Gen2 und Ryzen AI haben getrennte Compile- und Deployment-Verträge. |
| [Microchip](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox) | PolarFire-SoC-FPGA mit konfigurierbarer CoreVectorBlox-Beschleuniger-IP | Öffentlicher VectorBlox SDK 3.1 und Libero; Deployment-Assets `.vnnx`, `.hex` und `.ucomp` | Aktuelle [Tutorials](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) decken YOLOv5n, YOLOv8-Erkennung/Klassifikation/OBB/Pose/Segmentierung, YOLOv9t und weitere Vision-Modelle ab. SDK 3.1 ist derzeit auf das PolarFire SoC Video Kit beschränkt. |
| [Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) | Core-Ultra-NPU, CPU und integrierte/diskrete GPU | OpenVINO IR oder kompilierter Modellcache | OpenVINO bietet einen öffentlichen Cross-XPU-Pfad. Verwende die NPU-Spalten der [Modellkompatibilitätsmatrix](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html), statt anzunehmen, dass jedes OpenVINO-YOLO-Beispiel auf einer NPU validiert ist. |
| [Apple](https://developer.apple.com/machine-learning/core-ml/) | Apple Neural Engine in A11- und neueren A-Serien- sowie M-Serien-Chips | Core ML und `coremltools`; `.mlpackage`/`.mlmodelc` | Eine sehr zugängliche Consumer-NPU über Core ML, die CPU-/GPU-/Neural-Engine-Aufteilung aber abstrahiert, statt einen YOLO-spezifischen Hardwarecompiler bereitzustellen. |
| [GreenWaves Technologies](https://github.com/GreenWaves-Technologies/nn_menu_gap9) | GAP9 mit NE16 Neural Engine | GAP SDK, NNTool und AutoTiler-generierter Code | Offizielle Repositories umfassen MobileNet SSD, Gesichtserkennung, Klassifikation und einen dedizierten [YOLOX-Personendetektor](https://github.com/GreenWaves-Technologies/yolox_people_detection). Das vollständige SDK ist qualifizierten Kunden vorbehalten. |
| [Syntiant](https://www.syntiant.com/hardware) | NDP200 Neural Decision Processor in Massenproduktion und NDP250 in Bemusterung | Syntiant SDK und Deployment-Pakete | Extrem stromsparende Always-on-Vision-/Sensorprozessoren: Für NDP200 sind weniger als 1 mW dokumentiert, für NDP250-Bilderkennung weniger als 30 mW. Eine breite öffentliche YOLO-Matrix wurde nicht gefunden. |

## Amlogic: zwei NPU-Generationen, nicht eine

Amlogic verdient eine genauere Erklärung, weil Webquellen häufig inkompatible Produkte zusammenfassen. Das Unternehmen hat einen älteren Pfad auf VeriSilicon-Basis und einen neueren proprietären ADLA-Pfad. Sie verwenden weder denselben Compiler noch dasselbe Artefakt oder dieselbe Runtime.

| Generation | Repräsentative Chips | NPU und Toolchain | Kompiliertes Artefakt | Praktischer Status |
|---|---|---|---|---|
| Legacy | A311D und S905D3, häufig auf Khadas VIM3/VIM3L | VeriSilicon Vivante VIPNano-QI, Acuity Toolkit, KSNN und älteres `aml_npu_sdk` | `.nb` in einem `nbg_unify`-Ausgabeverzeichnis | Vorhandene Boards und Demos; separate Legacy-Integration |
| Aktuell ADLA2 | C308L/C302X, S928X, A311D2, T968D4, S905X5/S905D5 und C302X2 | Amlogic ADLA2, AMLNN Toolkit und NNSDK2 | Zielabhängiges `.adla`; W8A8 oder W8A16 | Aktueller, auf Integerpräzision ausgerichteter Stack |
| Aktuell ADLA3 | A311Y3, C305X2 und A123X | Amlogic ADLA3, AMLNN Toolkit und NNSDK2 | Zielabhängiges `.adla`; W4A8, W8A8, W4A16, W8A16 oder W16A16 | Ergänzt native Fähigkeiten für INT4, FP16 und BF16 |

Das [A311D-Datenblatt](https://dl.khadas.com/products/vim3/datasheet/a311d-datasheet.pdf) weist die ursprüngliche 5-TOPS-INT8-NPU aus. Die ältere [Khadas-VIM3-Anwendungsseite](https://docs.khadas.com/products/sbc/vim3/npu/npu-app) und der [NPU-Überblick](https://docs.khadas.com/products/sbc/vim3/npu/start) dokumentieren DenseNet CTC, MTCNN, RetinaFace, YOLOFace, YOLOv2, YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv7-tiny und YOLOv8n. Der Überblick dokumentiert zusätzlich YOLOv8n-pose und kennzeichnet FaceNet als veraltet. Diese Beispiele sind echt, belegen aber das alte Acuity-`.nb`-Ökosystem und nicht die aktuellen ADLA-Chips. A311D ist nicht A311D2, und C305X ist nicht C305X2.

Es gibt noch eine zweite Hardware-Falle. Der [Revisionsleitfaden für Khadas VIM4](https://docs.khadas.com/products/sbc/vim4/configurations/identify-version) besagt, dass das ursprüngliche Board V12/A311D2 Revision B keine NPU hat. Die beworbene NPU mit 3.2 TOPS gibt es erst auf V13A und neueren Boards mit Bauteil A311D2-N0D Revision C. Der Produktname allein genügt nicht, um ein Testgerät auszuwählen.

### Der aktuelle AMLNN- und ADLA-Stack

Das aktuelle [AMLNN Toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/7d3cc9a36179b756ce79c953c14d43acab1f2af5) deckt Modellkonvertierung, Quantisierung, Kompilierung, Inferenz und Profiling ab. Sein festgeschriebener [Nutzerleitfaden von May 2026](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/02_Amlogic_NPU_User_Guide_V0.1.pdf) dokumentiert Importe für ONNX, Fließkomma- oder quantisiertes TFLite, TorchScript `.pt` und PyTorch-2-ExportedProgram/PT2. Die detaillierte Anleitung führt TensorFlow-, Paddle- und Keras-Pfade als geplant auf, obwohl die Repository-Übersicht eine breitere Formulierung verwendet. Der Compiler erzeugt `.adla`. Python-Deployment verwendet AMLNN oder `amlnn_edge_toolkit_lite`; natives C/C++ bindet `libnnsdk.so` über `nnsdk2.h` ein. Die Entwicklung auf dem Host kann ADB mit `nnserver` nutzen. Die Runtime zielt außerdem auf Android, Buildroot, Yocto und ausgewählte Debian-/Armbian-Umgebungen.

Die veröffentlichten Plattformkennungen des Toolkits sind derzeit:

| Ziel-ID | Plattform |
|---:|---|
| `001` | C308L / C302X |
| `002` | S928X |
| `003` | A311D2 |
| `004` | T968D4 |
| `005` | S905X5 / S905D5 |
| `006` | C302X2 |
| `007` | A311Y3 |
| `008` | C305X2 / A123X in den detaillierten Dokumenten und der Modellmatrix |

Das Zielfeld ist nicht kosmetisch. Amlogic verlangt, dass `.adla`, `nnsdk2.h`, `libnnsdk.so`, BSP und Compiler-/Runtime-Generation zusammenpassen. Der festgeschriebene [Quick Start](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/01_Amlogic_NPU_Quick_Start_guide_V0.1.pdf) verlangt ADLA Driver 2.0.2 oder neuer, NNSDK 3.0.0 oder neuer und NNSDK2 1.0.0 oder neuer. Der [Android-Deployment-Leitfaden](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/06_Amlogic_NPU_Android_Deployment_Guide_V0.1.pdf) macht die Abhängigkeiten zwischen Bibliothek, Header, Driver und BSP konkret. Behandle `.adla` als ABI-gebundenes Build-Artefakt, nicht als portables Modell.

Die Quantisierung hängt ebenfalls von der NPU-Generation ab. Für ADLA2-Ziele 001 bis 006 ist die Kompilierung mit W8A8 und W8A16 dokumentiert. ADLA3-Ziele 007 und 008 ergänzen Kombinationen aus W4A8, W4A16, W8A8, W8A16 und W16A16 sowie native INT4-, FP16- und BF16-Unterstützung im [Operatorleitfaden](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/05_Amlogic_NPU_Support_Operator_List_V0.1.pdf). Amlogic empfiehlt 200 bis 500 repräsentative Kalibrierungsbeispiele. Die Option für Zufallsdaten ist ausdrücklich für Leistungstests vorgesehen, nicht für eine accuracy-erhaltende Quantisierung.

### Dokumentierte Modellabdeckung von Amlogic

Der [festgeschriebene Amlogic Model Playground](https://github.com/Amlogic-NN/amlnn-model-playground/blob/03ca63ef20c4f1560722de26ab910826884f334a/README.md) ist ein deutlich besserer Beleg als eine allgemeine Behauptung zur ONNX-Unterstützung. Seine veröffentlichte Support-/Beispielmatrix nennt A311D2, S905X5, A311Y3, C305X2 und A123X. Die Compilerunterstützung der anderen Ziel-IDs belegt nicht, dass die gesamte Modellliste auf ihnen validiert wurde.

| Aufgabe | Dokumentierte Modelle |
|---|---|
| Klassifikation | MobileNetV2 und ResNet50-v2; DINO auf ADLA3 |
| Objekterkennung | PP-YOLOE, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv10, YOLO11, YOLOE, YOLO-World, YOLOX und QR-Erkennung |
| Gesicht und Gesten | RetinaFace und Gesture Recognition |
| Segmentierung | DeepLabV3, PP-LiteSeg, YOLOv5-seg und YOLOv8-seg |
| Orientierte Erkennung | YOLOv8 OBB |
| Pose | BlazePose-Detektor/Landmark und YOLOv8 pose |
| OCR und Sprache | LPRNet, PaddleOCR-Varianten, Whisper Tiny und SenseVoice auf ausgewählten neueren Chips |
| Neuere Transformer- und multimodale Anwendungen | DETR, MobileSAM, CLIP/MobileCLIP und ausgewählte Low-Bit-LLM-/VLM-Beispiele auf neueren Plattformen |

Dasselbe Repository veröffentlicht folgende W8A8-Laufzeitwerte. Es handelt sich um Herstellerwerte für das neuronale Netz auf der NPU, nicht um vollständige Kamerapipelines.

| Modell und Eingabe | S905X5 | A311D2 | A311Y3 |
|---|---:|---:|---:|
| YOLOv8n, 640 x 640 | 101.72 FPS | 95.14 FPS | 191.06 FPS |
| YOLOv8s, 640 x 640 | 42.33 FPS | 42.77 FPS | 83.08 FPS |
| YOLOv8m, 640 x 640 | 19.67 FPS | 19.82 FPS | 35.30 FPS |
| YOLOv8l, 640 x 640 | 10.53 FPS | 10.12 FPS | 18.37 FPS |
| YOLO11n, 640 x 640 | 41.14 FPS | 41.48 FPS | 62.24 FPS |

Die Einschränkungen sind hier besonders wichtig. Amlogics Operatorleitfaden führt NMS sowohl auf ADLA2 als auch ADLA3 als Software aus. Laut README sind dies native NPU-Modelllaufzeitwerte ohne Vor- und Nachverarbeitung. Ob jeder Speichertransfer enthalten ist, wird nicht offengelegt. Für die YOLO-Accuracy-Zeilen wurden COCO-Teilmengen mit 300 Bildern statt des vollständigen Validierungssatzes verwendet. Die Klassifikationszeilen verwenden ImageNet `val1000`. Die Tabellen nennen außerdem für YOLOv8n auf A311Y3 eine andere Latenz als die FPS-Tabelle, ohne die Bedingungen abzugleichen. Takt, Leistungsmodus, Kühlung, thermischer Dauerzustand, genaue SDK-/BSP-Version und Testdauer sind nicht angegeben. Nutze die Werte, um einen Hersteller-Stack einzuordnen, nicht um ihn mit anderen zu vergleichen.

### Warum Amlogic strategisch interessant ist

Amlogic vereint vier ungewöhnliche Eigenschaften: eine große Verbreitung eingebetteter SoCs, einen neu öffentlich verfügbaren Compiler-Stack, eine aktuelle Modellmatrix mit großer Überschneidung zu den Aufgaben von LibreYOLO und eine bislang geringe erstklassige Integration in verbreitete Trainingsbibliotheken. Die aktuellen Repositories sind noch jung: Sie wurden Ende 2025 und Anfang 2026 öffentlich, umfangreiche Handbücher stammen von May through July 2026. Daraus ergeben sich sowohl eine echte Chance für frühe Integration als auch ein Risiko durch instabile APIs.

Eine saubere Integration sollte den deterministischen ONNX-Export von LibreYOLO als Compiler-Eingabe verwenden, repräsentative Kalibrierungsbilder für Builds mit niedriger Präzision verlangen, `.adla` samt Metadatenmanifest erstellen und über einen NNSDK2-Adapter laden. Für das ältere A311D sollte ein klar unterscheidbares `.nb`-Ziel verwendet werden, da eine gemeinsame Backend-Bezeichnung für `.nb` und `.adla` stille Kompatibilitätsfehler verursachen würde. Die obersten Lizenzen der Amlogic-Repositories sind Apache-2.0. Die Weiterverteilungsrechte für enthaltene Wheels, Shared Libraries, `nnserver` und Android-AAR-Binärdateien sollten jedoch direkt bestätigt werden, bevor LibreYOLO sie spiegelt.

## Die stärksten öffentlichen Deployment-Ökosysteme

Die folgenden Unternehmen bieten derzeit die klarste Kombination aus erhältlicher Hardware, öffentlichen technischen Unterlagen und Evidenz für benannte Modelle. Das macht sie nicht grundsätzlich schneller. Ihre Kompatibilitätsaussagen lassen sich vor dem Hardwarekauf leichter bewerten.

### Hailo

Hailo ist ein Referenzbeispiel für ein Ökosystem dedizierter Vision-Beschleuniger. Modelle werden in ein Hailo Archive eingelesen, optimiert und mit repräsentativen Bildern quantisiert. Anschließend kompiliert man sie in eine Datei im Hailo Executable Format (`.hef`). Anwendungen laden dieses HEF über HailoRT.

Der [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) enthält Rezepte, vortrainierte Modelle, Nachverarbeitungskonfigurationen und Leistungsdaten für Erkennung, Segmentierung, Klassifikation, Pose und weitere Aufgaben. Die YOLO-Abdeckung umfasst YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 und YOLO26 sowie YOLOX, DAMO-YOLO, SSD und EfficientDet. Die versionsgebundene [Tabelle zur Objekterkennung auf Hailo-8](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.19.0/docs/public_models/HAILO8/HAILO8_object_detection.rst) ist eine bessere Kompatibilitätsquelle als eine allgemeine Produktseite. Die [Standalone-Anwendung](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/README.md) dokumentiert ausführbare YOLO-Pipelines.

Es gibt eine wichtige Versionsgrenze. Hailo-8 und Hailo-8L verwenden die ältere Generation Model Zoo 2.x, Dataflow Compiler 3.x und HailoRT 4.x. Hailo-10 und Hailo-15 verwenden die aktuelle Generation 5.x. Rezepte, Parserverhalten und HEFs sind nicht austauschbar, nur weil auf jedem Gerät der Name Hailo steht.

Hailo-15 besitzt außerdem eine eigene Anwendungsschicht. Dort kommt das Vision Processor Software Package mit Hailo Media Library statt des TAPPAS-ähnlichen Host-Pfads von Hailo-8/10H zum Einsatz. Das öffentliche Referenz-Repository [`hailo-camera-apps`](https://github.com/hailo-ai/hailo-camera-apps) wurde am May 3, 2026 archiviert. [`hailo-apps-core`](https://github.com/hailo-ai/hailo-apps-core) bleibt ein separates öffentliches Anwendungs-Framework. Die Archivierung eines Repositorys belegt nicht das Produktende. Für Hailo-15-Projekte sollte das aktuelle Anwendungspaket in der Developer Zone bestätigt werden.

LibreYOLOs [Hailo-Deployment-Leitfaden](/docs/export/hailo) endet bewusst bei der statischen ONNX-Übergabe, da der proprietäre Compiler nicht als Python-Abhängigkeit gebündelt werden kann. Das ist die ehrliche Grenze zwischen der Erstellung eines geeigneten Graphen und der Behauptung, ein HEF getestet zu haben.

### Rockchip

Rockchip hat eine der größten Verbreitungen im Embedded-Linux-Hobbybereich und im kommerziellen Bereich, insbesondere über RK3588-, RK3576- und RK356x-Boards. Das öffentliche [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) konvertiert und quantisiert Modelle auf einem x86-Linux-Host. RKNN Runtime oder Toolkit-Lite führt die resultierende `.rknn`-Datei auf dem Board aus.

Der [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) ist ungewöhnlich genau bei Chips, Modellfamilien und Präzision. Die aktuelle Tabelle führt FP16- und INT8-Pfade für YOLOv5/6/7/8/10/11, YOLOX, PP-YOLOE, YOLO-World sowie YOLOv5-/v8-Segmentierungsvarianten auf. YOLOv8 OBB und YOLOv8 pose sind nur mit INT8 aufgeführt. Der Zoo deckt auch OCR, Gesichtsmodelle und andere Segmentierungsnetzwerke ab.

LibreYOLO bietet bereits einen vorsichtigen direkten [RKNN-Exporter](/docs/export/rknn). Er validiert vier genau definierte Erkennungsvarianten auf RK3588, kann den Compilersimulator mit ONNX Runtime vergleichen und lehnt nicht validierte Familien ab, statt einen erfolgreichen Build mit korrekten Vorhersagen gleichzusetzen. Dieser enge Supportumfang ist ein sinnvolles Modell für jedes künftige NPU-Backend.

### Axelera AI

Axelera AI, oft fälschlich „Accelera“ geschrieben, entwickelt die Metis AIPU und verkauft sie in M.2-, PCIe- und Mehrchip-Produkten. Metis ist die öffentlich bestellbare Familie; Europa und Titania sind angekündigte Produkte im Portfolio. Ihr Verfügbarkeitsstatus sollte nicht zu einem einheitlichen Produktstatus zusammengezogen werden. Der [Voyager SDK ist öffentlich auf GitHub verfügbar](https://github.com/axelera-ai-hub/voyager-sdk) und übernimmt Modellauswahl, Kompilierung, Quantisierung, Ausführung und Anwendungspipelines. Kundensupport ist weiterhin kontogeschützt.

Der öffentliche [Voyager Model Zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo/) gehört zu den informativsten des Marktes. Er führt Modellvariante, Eingabegröße, Präzision, Accuracy und gemessene Leistung für YOLOv3, YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 n/s/m/l/x, YOLOX, YOLO-NAS, OBB, Pose und Segmentierung sowie viele Klassifikations- und dichte Vorhersagemodelle auf. Quantisierungsverlust und Accuracy neben der Geschwindigkeit zu veröffentlichen ist nützlicher, als nur einen Spitzen-TOPS-Wert anzugeben.

Die Artefaktbezeichnung änderte sich mit der API-Generation. Der Alpha-Kompilierungsablauf des [Pipeline Builders](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) erzeugt ein `.axm`-Modell und kann eine portable Pipeline als `.axe` verpacken. Die [klassische Pipeline-Dokumentation](https://docs.axelera.ai/sdk/reference/pipeline/model-formats/) beschreibt `.axmodel` samt Modellmetadaten und Manifest. Eine Integration muss die installierte Voyager-Generation erkennen, statt eine einzelne Dateiendung fest einzutragen.

### Qualcomm

Qualcomm bietet eine enorme Deployment-Reichweite, doch unter „Qualcomm-NPU“ verbergen sich mehrere Schnittstellen. Entwickler begegnen Hexagon HTP über QAIRT/QNN, dem älteren SNPE-Ablauf, Qualcomm AI Hub-Kompilierungsdiensten, LiteRT oder dem QNN Execution Provider von ONNX Runtime, je nach Gerät und Produktklasse.

Das offizielle Repository [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) liefert starke Evidenz auf Modellebene. Es veröffentlicht optimierte Pakete für YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 für Erkennung/Segmentierung/Pose, YOLOX, YOLO-World, YOLOR, RF-DETR und zahlreiche Klassifikations-, Tiefen- und Posemodelle. AI Hub bietet auch gerätespezifisches Profiling. Das ist wichtig, weil ein für eine HTP-Generation kompiliertes Modell nicht automatisch auf eine andere übertragbar ist.

Der größte Integrationsaufwand entsteht durch die Breite der Matrix: Android gegenüber Embedded Linux, QCS-Bestellvarianten gegenüber Consumer-Snapdragon, HTP-Architektur, QNN-/QAIRT-Version und Quantisierungsschema sind entscheidend. Die aktuellen eigenen Seiten von Qualcomm widersprechen sich außerdem beim Status von [Dragonwing IQ-9075](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075): Die Produktseite kennzeichnet ihn als Active und das [EVK](https://www.qualcomm.com/developer/hardware/qualcomm-iq-9075-evaluation-kit-evk) ist evaluierbar, während das [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) den IQ-9075 weiterhin als Sampling kennzeichnet und eine Langlebigkeit bis 2038 nennt. Das Bestell-/SKU-Präfix ist QCS9075. Qualcomm bewirbt Konfigurationen mit 50 und 100 dichten INT8-TOPS sowie Unterstützung für Ubuntu/Yocto. Der Produktionsstatus sollte für die genaue Bestell-SKU bestätigt werden. Eine LibreYOLO-Integration sollte diese SKU dokumentieren, statt einen allgemeinen Ordner „Qualcomm“ zu erzeugen.

### DEEPX

Die Beschleuniger DX-M1/DX-M1M von DEEPX verwenden das DXNN SDK. DX-COM kompiliert und quantisiert das Modell, DX-RT führt es aus, und das Deployment-Artefakt nutzt das Format `.dxnn`. Das Unternehmen veröffentlicht die [DX-AllSuite](https://github.com/DEEPX-AI/dx-all-suite), Anwendungsbeispiele in [DX-APP](https://github.com/DEEPX-AI/dx_app) und einen umfangreichen [Online Model Zoo](https://developer.deepx.ai/modelzoo/).

Bei der Prüfung mit Stichtag August 15, 2026 meldete der öffentliche Katalog 354 Einträge unter DX-COM 2.4.0 und DX-RT 3.4.0. Die dokumentierte Abdeckung umfasst YOLOv3 bis YOLO11 und YOLO26, YOLOX, SSD, EfficientDet, NanoDet, DAMO-YOLO, YOLO-Segmentierung, Pose und orientierte Boxen. DEEPX ist ein besonders plausibles Integrationsziel, weil die Compiler-/Runtime-Grenze und das Deployment-Artefakt klar benannt sind und die öffentliche Vision-Abdeckung umfangreich ist.

## Industrielle SoCs sind mehrere Ökosysteme, nicht eines

Industrieanbieter bieten oft längere Produktlebenszyklen sowie bessere Kamera-, Sicherheits- und Echtzeitintegration als SoCs für Maker-Boards. Ihre KI-Stacks können komplizierter sein, weil ein Anbieter unter Umständen mehrere voneinander unabhängige NPU-Architekturen gleichzeitig ausliefert.

### Texas Instruments

Zu den aktuellen Edge-AI-Prozessoren von TI gehören AM62A, AM67A, AM68A, AM69A und verwandte TDA4-Geräte. Der Softwarepfad kombiniert Processor SDK Linux, TIDL-Compiler/Runtime, [Edge AI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) und den [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo). TIDL kann sich in ONNX Runtime, TensorFlow Lite und andere Anwendungs-Runtimes integrieren und unterstützte Teilgraphen auslagern.

TI veröffentlicht mehr als nur die Aussage, ein Framework importieren zu können: Der Zoo enthält konvertierte Modelle und Leistungsmetadaten für Objekterkennung, Segmentierung, Pose, Klassifikation, Tiefe und weitere Aufgaben. TI weist auch darauf hin, dass Modell-Zoo-Artefakte Ausgangspunkte für die Entwicklung und nicht automatisch produktionsreife Assets sind. Dieser wertvolle Vorbehalt fehlt in vielen Herstellervergleichen.

TI führt außerdem [TDA54-Q1](https://www.ti.com/product/TDA54-Q1) als Preview mit bis zu vier C7-NPUs und bis zu 400 TOPS bei diesem Bauteil. Für die breitere TDA5-Familie werden bis zu 1.200 TOPS beworben. Das sind Herstellerangaben zu einer neuen Generation und keine Grundlage, die TIDL-Modellvalidierung für AM6xA/TDA4 auf TDA54-Q1 zu übertragen, bevor TI eine zielgenaue Matrix veröffentlicht.

### NXP

NXP benötigt mindestens vier Backend-Beschreibungen:

| NXP-Ziel | Beschleuniger | Compiler-/Runtime-Pfad |
|---|---|---|
| i.MX 8M Plus | VeriSilicon-Vivante-NPU mit 2.3 TOPS | eIQ mit TIM-VX/VX-Delegate |
| i.MX 93 | Arm Ethos-U65 | Quantisiertes TFLite plus Arm Vela |
| i.MX 95 | NXP eIQ Neutron NPU | Neutron Converter und eIQ Runtime |
| Ara-1 / [Ara240](https://www.nxp.com/products/ARA240) | Von Kinara abgeleitete diskrete NPU; Ara240 bewirbt bis zu 40 eTOPS | Ara SDK mit Integration in das umfassendere eIQ-Angebot |

Der [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) enthält Assets oder Rezepte für YOLOv4-tiny, YOLOv8, NanoDet, CenterNet, FastestDet, SSD Lite, YOLACT und andere Modelle. Der YOLOv5-Eintrag wurde nur als Dokumentation ergänzt. Ein Eintrag, der für ein Backend validiert ist, belegt nicht die Eignung für alle vier. NXPs eigene [YOLO-Exportanleitung für i.MX-Plattformen](https://community.nxp.com/t5/i-MX-Processors-Knowledge-Base/Exporting-YOLO-Models-for-NXP-i-MX-Platforms/ta-p/2381361) veranschaulicht die zielabhängige Konvertierung. NXP zufolge erhielt das monolithische [eIQ Toolkit nach Version 1.17 im Q3 2025 keine Updates mehr](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/eIQ-FAQ/ta-p/1099741). Aktuelle Abläufe verwenden eigenständige Pakete wie das eIQ Neutron SDK.

NXP [schloss die Übernahme von Kinara in October 2025 ab](https://media.nxp.com/news-releases/news-release-details/nxp-completes-acquisitions-aviva-links-and-kinara-advance/). Das [Ara240-Datenblatt](https://www.nxp.com/docs/en/fact-sheet/ARA240DNPUFS.pdf) nennt bis zu 40 vom Hersteller beworbene eTOPS und 313 Bilder pro Sekunde bei YOLOv8n. NXP listet Ara SDK und eIQ Toolkit auf der Produktseite, was aber keinen Austausch von Artefakten mit dem separaten [Neutron-Pfad von i.MX 95](https://eiq.nxp.com/learning-hub/convQuant/neutron.html) belegt. Das [M.2-Modul mit 16 GB](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT) ist aktiv, die USB-Option befindet sich in der Vorproduktion. NXP definiert das „e“ in eTOPS als „equivalent“, nicht als „effective“. Das ist keine dichte INT8-TOPS-Kennzahl eines anderen Herstellers und sollte nicht direkt verglichen werden.

### STMicroelectronics

[STM32N6x7-Geräte](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html), darunter STM32N657 und STM32N647, integrieren einen Neural-ART-Beschleuniger mit 600 GOPS in ein MCU-Produkt. Die universellen Produktlinien N6x5 enthalten diesen Beschleuniger nicht. [STM32Cube AI Studio](https://www.st.com/en/development-tools/stedgeai-cubeai.html) und ST Edge AI Core analysieren und optimieren importierte Modelle. [STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) stellt Deployment-, Optimierungs-, Benchmark- und Beispielabläufe bereit.

Die aktuelle [Übersicht zur Objekterkennung](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/docs/README_OVERVIEW.md) dokumentiert Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 und ST-YOLOX sowie Klassifikations-, Pose- und Segmentierungsdienste. Das ist nicht dieselbe Leistungsklasse wie eine PCIe-Karte mit 200 TOPS. Interessant ist die Plattform, weil Kameraeingang, Inferenz und Steuerung in ein enges Energie- und Speicherbudget eingebetteter Systeme passen.

### Renesas

Renesas-RZ/V-Prozessoren integrieren DRP-AI-Beschleuniger. Die Software entwickelte sich vom DRP-AI Translator und DRP-AI TVM in Richtung [RUHMI](https://www.renesas.com/en/software-tool/ruhmi-framework), das auf EdgeCortix-MERA-Technologie basiert. Das offene [RZ/V-DRP-AI-TVM-Repository](https://github.com/renesas-rz/rzv_drp-ai_tvm) und die [Validierungsliste für RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) nennen YOLOv5, YOLOv8, YOLO11 und YOLO26 in den Größen n/s/m, YOLOX sowie Pose- und Segmentierungsnetzwerke neben Klassifikationsmodellen.

Renesas ist ein glaubwürdiges Ziel für Robotik und Industrie. Für Reproduzierbarkeit müssen jedoch das genaue Board, die DRP-AI-Generation, die Translator-Version und die CPU-seitige Nachverarbeitung dokumentiert werden.

## Intelligente Sensoren und kamerazentrierte Prozessoren

### Sony IMX500

Sony IMX500 ist kein normaler Anwendungsprozessor. Bildsensor und KI-Verarbeitung sind gestapelt, sodass die Inferenz im Sensor stattfinden kann und weniger Bilddaten die Kamera verlassen müssen. Die Raspberry Pi AI Camera macht diese Architektur ungewöhnlich zugänglich.

Das offizielle [Raspberry-Pi-IMX500-Modellrepository](https://github.com/raspberrypi/imx500-models) veröffentlicht paketierte Modelle für YOLOv8n, YOLO11n, EfficientDet Lite0, NanoDet+ und SSD MobileNetV2 FPN Lite. Die [Dokumentation der AI Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) beschreibt Konvertierung, Paketierung und Nachverarbeitung auf dem Sensor. Die Deployment-Einheit ist ein RPK-Paket statt einer allgemeinen ONNX-Datei. Sensor-Speicher und Operatorbeschränkungen sind zentrale Designgrenzen. Auf der [Produktseite von Raspberry Pi](https://www.raspberrypi.com/products/ai-camera/) steht, dass diese Kamera mindestens bis January 2028 produziert wird.

### Ambarella

Ambarellas CVflow-Prozessoren sind in Kameras, Drohnen und Automotive Vision fest etabliert. Zu den aktuellen Produktfamilien gehören CV72/CV75 für Kameras, CV5/CV52 für leistungsfähigere Vision-Systeme und CV3-AD für Automotive. Die Cooper Developer Platform sowie der CVflow-Kompilierungs- und Runtime-Ablauf bilden den öffentlich benannten Software-Stack.

Der [öffentliche Entwickler-Model-Garden](https://www.ambarella.com/developer/model-garden/) nennt YOLOX, RTMDet, DeepLabV3+, TopFormer, OWL-ViT und multimodale Modelle. Detaillierte Compiler-Dokumentation und Downloads bleiben jedoch partnerorientiert. Eine Ambarella-Integration sollte daher mit einer Beziehung zum Hersteller oder OEM beginnen und nicht mit der Annahme, ein öffentliches pip-Paket sei erhältlich.

### Synaptics Astra

Für Synaptics sind drei Ziele relevant. Astra-Systeme SL1600/SL1680 verwenden das [SyNAP Toolkit](https://developer.synaptics.com/docs/synap/introduction), das ein Quellmodell samt YAML-Beschreibung in ein `model.synap`-Paket umwandelt. Neuere Geräte SL2611/13/15/17/19 verwenden die [Torq-Plattform](https://developer.synaptics.com/docs/torq/introduction), einen IREE-/MLIR-basierten Ablauf, der `.vmfb` erzeugt. Die [SR100-Serie](https://developer.synaptics.com/docs/sr/introduction-sr) ist eine separate MCU-Familie auf Basis von Cortex-M55 und Ethos-U55 mit eigenem SDK. Diese Artefakte und APIs sind nicht austauschbar.

Das offizielle [YOLO-Benchmark-Tutorial](https://developer.synaptics.com/docs/sl/tutorials/vision/benchmark-yolo) ist ungewöhnlich konkret: Es exportiert YOLOv8 nach TFLite, führt asymmetrische UINT8-Kalibrierung in SyNAP aus, kompiliert für SL1680 und startet anschließend `synap_cli` sowie die Objekterkennungsanwendung. Das belegt YOLOv8n/v8s auf SL1680 und beschreibt SyNAP-Unterstützung bis YOLO11. Ein separater [Leitfaden zur Objekterkennung auf SL261x](https://developer.synaptics.com/docs/sl/sl2600/getting-started/object-detection) führt YOLOv8 mit Torq-`.vmfb` aus. Dies sind offizielle Beispiele für bestimmte Ziele und keine Behauptung, jeder YOLO-Graph werde auf allen drei Familien unterstützt.

### MediaTek Genio

MediaTek verdient einen ausführlichen Eintrag, denn der aktuelle [IoT AI Hub](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) stellt jetzt Hardware-/Softwarematrix, Modellpakete und Benchmarktabellen bereit, die ältere Marktübersichten nicht prüfen konnten. Genio 360/360P/420/520/720 verwenden NeuroPilot 8 mit MDLA 5.3. Genio 510/700 verwenden NP6 mit MDLA 3.0, Genio 1200 NP6 mit MDLA 2.0. MediaTek erklärt, dass NeuroPilot- und MDLA-Generation, Operatorsatz, Compiler und Runtime für die Lebensdauer jedes SoC versionsgebunden sind.

Der analytische KI-Pfad ist auf TFLite ausgerichtet:

```text
PyTorch model
    -> NeuroPilot Converter and representative calibration
    -> quantized .tflite
    -> version-matched ncc-tflite compiler
    -> chip-generation-specific .dla
    -> Neuron Runtime on MDLA
```

Neuere Genio-Produkte bieten außerdem Online-Delegation über LiteRT und auf unterstützten Betriebssystemen einen ONNX-Runtime-CPU-/NPU-Pfad. Diese Ausführungsmodi unterscheiden sich von einem Offline-`.dla`. Der [Softwarearchitektur-Leitfaden](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/software_architecture.html) und die [Ressourcenmatrix](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) machen den Unterschied ausdrücklich.

Die offizielle [Tabelle analytischer Modelle](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical.html) veröffentlicht YOLOv5s- und YOLOv8s-Benchmarkeinträge und Konvertierungsleitfäden für mehrere MDLA-Generationen. Vorab konvertierte YOLO-Artefakte werden wegen der Beschränkungen von AGPL-3.0 ausdrücklich nicht verteilt. Die Quant8-Offline-Messungen bei 640 x 640 nennen 5.35 ms beziehungsweise 8.04 ms auf Genio 720. Die [Modellseite zu YOLOv8s](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical/YOLOv8s.html) veröffentlicht Ein-/Ausgabe-Tensoren, Backend-spezifische Zeiten und Warnungen zu benutzerdefinierten MediaTek-Operationen. Diese Werte sind Hersteller-Benchmarks und keine Ergebnisse vollständiger Kamerapipelines.

Der Zugang ist der begrenzende Faktor. Die öffentliche Yocto-Dokumentation ist detailliert, doch das aktuelle NP8-Komplettpaket für Converter und Compiler sowie viele Android-Unterlagen sind als NDA- oder Direktkundenressourcen gekennzeichnet. Ein normales Entwicklerkonto bietet nicht denselben Zugang wie ein MediaTek-Online-Kundenkonto. LibreYOLO sollte Genio deshalb als technisch belegt, aber für eine reproduzierbare Compilerintegration mit eigenen Modellen als partnerschaftsabhängig behandeln.

## Auf China ausgerichtete Edge-AI-Ökosysteme

Einige der leistungsfähigsten und zugänglichsten kostengünstigen Vision-Plattformen sind in englischsprachigen Marktübersichten unterrepräsentiert.

### D-Robotics und von Horizon abgeleitete BPU-Plattformen

D-Robotics unterhält das Entwickler-Board-Ökosystem RDK für BPU-Beschleuniger in RDK X3, X5, Ultra und neueren Produkten der S100-Serie. Der aktuelle Branch `rdk_x5` des [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo) dokumentiert für RDK X5 YOLOv5/v5u, YOLOv8/9/10/11/12/13/26, YOLOE, YOLO-World sowie Pfade für Erkennung, Segmentierung, Pose, Klassifikation, OCR und CLIP. X3 verwendet einen eigenen Branch, während die aktuelle S-Serie im Haupt-Zoo im Branch [`rdk_s`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s) bereitgestellt wird. Das ältere Repository [`rdk_model_zoo_s`](https://github.com/D-Robotics/rdk_model_zoo_s) enthält historische Demos. Die moderne X5-Liste belegt daher keine Validierung aller Modelle auf X3, Ultra oder S100.

OpenExplorer/Algorithm Toolchain importiert ONNX oder Caffe für PTQ und unterstützt QAT-Abläufe mit PyTorch. Das Deployment ist plattformspezifisch: Aktuelles RDK X5 verwendet `.bin` über `hbm_runtime` und `libdnn`. Der aktuelle primäre `rdk_s`-Pfad verwendet `.hbm` über eine gleichnamige `hbm_runtime`-API und `libhbucp`. X3 behält seinen älteren Branch und seine Inferenzschnittstellen. Älteres Material in `rdk_model_zoo_s` beschreibt ebenfalls historische `.bin`- und `.hbm`-Pfade. Diese Artefakte müssen jeweils mit dem zugehörigen Branch und der passenden Toolchain gepaart bleiben. Nicht unterstützte Operatoren können auf die CPU zurückfallen. Die öffentlichen Beispiele sind hilfreich, doch die Versionskompatibilität von RDK OS, Board-Firmware, Toolchain und Modell-Binärdatei gehört weiterhin zum Kompatibilitätsvertrag.

### AXERA

Die AX650-/AX630-/AX620-Familien von AXERA kommen in kompakten KI-Kameras und Boards wie der MaixCAM-Reihe von Sipeed vor. Pulsar2 importiert ONNX, führt Kalibrierung und Präzisionsanalyse aus und erzeugt `.axmodel`. AXEngine führt das Artefakt aus.

Die [offiziellen AXERA-Beispiele](https://github.com/AXERA-TECH/ax-samples) decken YOLOv5/6/7/8/9/10/11/13/26, YOLOX und YOLO-World ab. Aufgaben- und Chipunterstützung variieren je nach Plattform. Für unterstützte Ziele enthalten die aktuellen YOLO26-Beispiele Erkennung, Pose, Segmentierung und OBB. Die [Pulsar2-Konvertierungsbeispiele](https://pulsar2-docs.readthedocs.io/en/latest/appendix/model_convert_examples.html) zeigen den tatsächlichen Aufwand: genaue Tensor-Namen, Ausgabeaufteilungen, Layout-Transformationen, Kalibrierungsarchive und Zieleinstellungen für die Hardware.

### SOPHGO

TPU-MLIR von SOPHGO ist einer der attraktiveren Compiler-Stacks für unabhängige Integration, da der Compiler selbst Open Source ist. Er importiert ONNX, PyTorch, TFLite und Caffe, senkt Graphen ab und quantisiert sie. Für BM-Ziele erzeugt er `.bmodel`, für CV18xx-Hardware `.cvimodel`.

Das [TPU-MLIR-Projekt](https://github.com/sophgo/tpu-mlir) unterstützt abhängig vom Ziel Abläufe mit FP32, BF16, FP16 und INT8. Die [SOPHON-Demo-Sammlung](https://github.com/sophgo/sophon-demo/blob/release/README_EN.md) nennt YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, OBB-/Segmentierungsvarianten, SSD, CenterNet, RetinaFace, SAM/SAM2 und OCR. Wie immer validiert eine Demo für BM1684X nicht dasselbe Artefakt auf BM1688 oder CV18xx.

### Huawei Ascend

Zu Huaweis Edge-Inferenzlinie gehören Ascend-310/310P/310B-Chips in Atlas-200I- und -300I-Produkten. CANN stellt ATC-Compiler, AscendCL-Runtime und chipspezifische Operator-Kernels bereit. ATC wandelt ONNX und andere Quellrepräsentationen in ein Offline-Modell `.om` um.

Die aktuelle [Dokumentation zum Atlas 200I DK A2](https://www.hiascend.com/en/hardware/developer-kit-a2/specification) und CANN-Beispiele stellen einen YOLOv5-`.om`-Pfad für Ascend 310B bereit. Der breitere, ältere [Ascend ModelZoo](https://github.com/Ascend/modelzoo) enthält YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN, Pose- und Segmentierungsmodelle. Diese Einträge sollten jedoch nicht so dargestellt werden, als seien alle auf 310B erneut validiert worden. CANN, Kernels, Firmware und SoC müssen zusammenpassen. Ein `.om` für Ascend310P ist kein allgemeines Ascend-Artefakt.

### Cambricon

Cambricons MLU-Beschleuniger verwenden Neuware und die Inferenz-Engine MagicMind. Das öffentliche [MagicMind-Cloud-Repository](https://github.com/Cambricon/magicmind_cloud) ist auf MagicMind 1.7 und eine MLU370-X4/S4-Matrix festgelegt. Es ist historische Evidenz zur Kompatibilität, keine Matrix für ein aktuelles SDK oder MLU270. Das Repository dokumentiert PyTorch-, ONNX-, Caffe- und Paddle-Pfade. TensorFlow-Framework-Unterstützung wurde in 1.7 entfernt, obwohl historische TensorFlow-Beispiele verbleiben. Cambricons aktuelles [3-Series-Entwicklerportal](https://developer.cambricon.com/index/document/index/classid/3.html) führt spätere SDK-Gesamtversionen auf. Öffentliches Beispielrepository und aktueller kommerzieller Stack dürfen daher nicht als eine Version dargestellt werden.

Das offizielle [MagicMind-Cloud-Modellrepository](https://github.com/Cambricon/magicmind_cloud) veröffentlicht eine ungewöhnlich direkte MLU370-X4/S4-Matrix. Sie führt Beispiele zu YOLOv3, Tiny-YOLOv3, YOLOv4, YOLOv5, YOLOv7, YOLOv8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN, DeepLab und UNet auf und nennt jeweils, ob C++- oder Python-Beispiele verfügbar sind. Die Evidenz ist stark; die größte Hürde ist der SDK- und Containerzugang über Cambricon-Kanäle.

### Canaan/Kendryte

Kendrytes aktuelle K230/K230D- und älteren K210/K510-KPU-Chips sind am kostengünstigen Board- und Smart-Kamera-Ende des Marktes relevant. Der [nncase-Compiler](https://github.com/kendryte/nncase) importiert ONNX oder TFLite, verwendet Kalibrierungsdaten für die Festkommaumwandlung, simuliert das Ergebnis auf einem Host und erstellt `.kmodel`.

Der offizielle [K230-nncase-Entwicklungsleitfaden](https://github.com/kendryte/k230_docs/blob/main/en/01_software/board/ai/K230_nncase_Development_Guide.md) führt durch Kompilierung, Simulation und Ausführung von YOLOv5s. Der separate [KI-Demokatalog](https://github.com/kendryte/k230_docs/blob/main/en/02_applications/ai_demos/K230_AI_Demo_Introduction.md) enthält YOLOv8n-Artefakte für Erkennung, Segmentierung und Pose. Das aktuelle [CanMV-Changelog](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) ergänzt optimierte Beispiele für Klassifikation, Erkennung, Segmentierung, OBB und Pose mit YOLOv8/11/26. Das YOLOv5s-Ergebnis im Datenblatt ist der tatsächlich veröffentlichte [Benchmark](https://github.com/kendryte/k230_docs/blob/main/en/00_hardware/K230_datasheet.md). Versionen des Compilers und des binären KPU-Plug-ins müssen zum Board-SDK passen.

### Allwinner

Allwinner V853 kombiniert medienorientierte Kamerahardware mit einer Vivante-NPU mit 1 TOPS. Der [offizielle englische NPU-Leitfaden](https://docs.aw-ol.com/v853/en/npu/dev_npu/) beschreibt Import, Quantisierung, Verifizierung und Deployment über Acuity/Pegasus und `viplite`. Die eigenen Modellseiten von Allwinner und der [YOLOv5-Leitfaden](https://docs.aw-ol.com/v853/npu/npu_yolov5/) nennen YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet sowie Gesichts- und Personennetzwerke.

Das ist eine echte Unterstützung für Computer Vision, sie bleibt aber an die ältere Tina-Linux-/Vivante-Toolchain und SDK-Downloads mit Kontoanforderung gebunden. Mit sichtbarer Einschränkung gehört sie in die Marktübersicht.

## Weniger beachtete Plattformen aus Korea, Taiwan und China

Englischsprachige Listen springen häufig von Rockchip zu NVIDIA und übersehen eine zweite Gruppe realer, dokumentierter Chips. Einige dieser Ökosysteme haben aktuell breitere YOLO-Matrizen als bekanntere westliche Start-ups.

### Mobilint

Der koreanische Beschleunigeranbieter Mobilint verkauft den stromsparenden REGULUS und den leistungsstärkeren ARIES MLA100. Sein qb SDK akzeptiert PyTorch, TensorFlow, TFLite, ONNX und Keras-Eingaben, quantisiert nach INT8 und erzeugt ein kompiliertes `.mxq`-Artefakt. Die öffentliche [Vision-Matrix](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) nennt YOLOv3/v5/v7/v8/v9/v10/11/12/26 für Erkennung sowie YOLO-Segmentierungs-, Pose- und OBB-Varianten. Die [ARIES-Seite](https://www.mobilint.com/aries/mla100) veröffentlicht sogar Ergebnisse speziell für YOLO11s und YOLO26m. Das ist ein starker Integrationsbeleg, auch wenn Zugang zu SDK und Hardware kommerziell bleibt.

### Sunplus

Sunplus SP7350/C3V verwendet einen von Vivante abgeleiteten NPU-Stack, verpackt ihn aber anders als Allwinner oder das ältere Amlogic. Der öffentliche Ablauf kombiniert Acuity-Konvertierung, eine NPU-Docker-Umgebung, SNNF-Runtime und `.nb`-Ausgabe. Der offizielle [YOLOv8-Leitfaden für angepasste Modelle](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) führt durch Konvertierung und Deployment, statt lediglich Framework-Unterstützung zu behaupten. Verwandte IP macht dessen `.nb` nicht auf einem anderen Vivante-basierten SoC portabel.

### ESWIN Computing

Zu ESWINs RISC-V-Familie gehören EIC7700/EIC7700X und das Dual-Die-Modell EIC7702/EIC7702X. EIC7700 steckt in ausgelieferten Boards wie Milk-V Megrez. ENNP umfasst EsQuant, den aktuellen EsAAC-Compiler, Erzeugung von Golden Data, Simulation und ESSDK-Runtime sowie die kompilierte Ausgabe `.model`. Älteres ENNP-Material verwendete den Namen `ennc-compile`. Die aktuelle [EsAAC-Seite](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) dokumentiert ONNX als Eingabe. Für andere Frameworks muss daher exportiert oder in diese unterstützte IR konvertiert werden. Die von Milk-V bereitgestellte [ENNP-Übersicht](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/introduction) und das [YOLOv3-Tutorial](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3) belegen einen öffentlichen End-to-End-Pfad, aber keine aktuelle, umfassende Modell-/Accuracy-Matrix.

### Nuvoton M55M1

Nuvoton M55M1 kombiniert die Cortex-M55-Klasse mit Ethos-U55-256 und ist ein MCU-Design, kein Linux-Anwendungsprozessor. NuEdgeWise/NuML und Arm Vela bereiten quantisierte TFLite-Micro-Modelle vor. Das öffentliche [NuEdgeWise-Repository](https://github.com/OpenNuvoton/NuEdgeWise) nennt YOLOv8-nano, YOLOX-nano, YOLO Fastest v1.1, SSD-MobileNet FPNLite und mehrere Klassifikatoren. Diese Beispiele für kleine Netzwerke belegen die Speicher-/Leistungsklasse, nicht Standard-YOLO-Varianten mit 640 Pixeln.

### Rebellions und FuriosaAI

Die ATOM-Reihe von Rebellions verwendet das RBLN SDK und kompilierte `.rbln`-Artefakte. Das [offizielle Support-Release](https://docs.rbln.ai/v0.8.2/supports/release_note.html) nennt YOLOv3 tiny/full/SPP, YOLOv5 n bis x, YOLOv6 n bis l, YOLOv7-Varianten und YOLOv8 n bis x. Die aktuelle [Kartensupport-Matrix](https://docs.rbln.ai/latest/supports/version_matrix.html) kennzeichnet ATOM CA02 und ATOM+ CA12 als EoL sowie ATOM+ CA22 und ATOM-Max CA25 als Active. ATOM-Lite CA21 hat eine Produktseite, fehlt aber in der SDK-Matrix. Der aktuelle Toolchainstatus ist daher unklar. Die Dokumentation ist öffentlich, das Compiler-Wheel erfordert aber Anmeldedaten für das Rebellions Portal.

FuriosaAI muss nach Generationen getrennt werden. Warboy ist der ältere Computer-Vision-Beschleuniger mit INT8-`.enf`-Artefakten und einem [offiziellen Model Zoo](https://developer.furiosa.ai/furiosa-models/latest/) mit SSD, YOLOv5M/L und YOLOv7-w6-pose. RNGD nutzt einen anderen [`.fxb`-Stack](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html), der hauptsächlich auf LLM-/VLM-Arbeitslasten ausgelegt ist. Furiosas aktuelle [Roadmap](https://developer.furiosa.ai/latest/en/overview/roadmap.html) führt Vision-Unterstützung für YOLOv8m als in 2024 Q4 abgeschlossen auf. Die aktuellen öffentlichen Modell- und Leistungsseiten konzentrieren sich jedoch auf LLM/VLM und enthalten keine detaillierte Accuracy-/Leistungsmatrix für YOLOv8m. Das belegt einen abgeschlossenen Release, nicht Kompatibilität mit Warboy.

### Realtek, Telechips, T-Head und SigmaStar

Diese vier Plattformen sind real, bieten aber engere, ältere oder zugangsbeschränkte Möglichkeiten, eigene Modelle einzubringen:

| Plattform | Reproduzierbare öffentliche Evidenz | Zu erhaltende Einschränkung |
|---|---|---|
| Realtek AmebaPro2 | Das [Arduino-/FreeRTOS-SDK](https://github.com/Ameba-AIoT/ameba-arduino-pro2) enthält `.nb`-Modelle für YOLOv3/4/7-tiny, SCRFD und MobileFaceNet. | Benutzerdefinierte Modellkonvertierung erfordert eine Interessenbekundung oder Kontakt zu Realtek. |
| Telechips TOPST TCC7500 | Das [offizielle Projekt von 2026](https://docs.topst.ai/blog/31) konvertierte YOLOv8s und führte es aus. | Die UFLD-v1/v2-Konvertierung scheiterte an nicht unterstützten Layers. Der Artikel schlägt lediglich einen Workaround durch Aufteilung/Nachverarbeitung vor. Vollständige Compiler-/Operatorressourcen erfordern häufig eine Genehmigung per E-Mail. |
| T-Head TH1520 | Der [Anwendungsleitfaden von Sipeed](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) führt YOLOv5n/s mit HHB und CSI-NN2/SHL aus. | Ein öffentlicher Stack ist vorhanden, Wartung und Versionsklarheit hinken den aktuellen Marktführern jedoch hinterher. |
| SigmaStar SSU9383CM | Die aktuelle Dokumentation belegt MI_IPU-Runtime-APIs. Älteres SGS_IPU-Material dokumentiert `.sim` zu `sgsimg.img`, und ein älterer Postprozessor nennt SSD sowie YOLOv1/2/3. | Es gibt keinen öffentlichen Beleg dafür, dass ältere Compilerartefakte oder Modellpfade für SSU9383CM gelten. |

### HiSilicon Smart Vision ist nicht Huawei Ascend

HiSilicon-Kamera-SoCs Hi3516CV610/DV500, Hi3519DV500 und Hi3403V100 bieten einen ATC-zu-`.om`-Ablauf über NNN-/SVP-NNN-Runtimes. Der offizielle [HiSpark Model Zoo](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) nennt in einer zielbezogenen Matrix, insbesondere für Hi3403 und Hi3591P, YOLOv3/4/5/6/7/8/9/10/11, YOLO11-Segmentierung/Pose, YOLOv8 OBB/World/Segmentierung, OCR, Tiefe und TinySAM. Daraus folgt nicht, dass alle Einträge auf jedem oben genannten Smart-Vision-SoC laufen. Einige Einträge sind Roadmap-Punkte. Ausgelieferte Beispiele und geplante Unterstützung müssen daher getrennt bleiben. Im Repository steht außerdem, dass die bereitgestellten Model-Zoo-Modelle nur für nicht kommerzielle Nutzung bestimmt sind. Gleiche Begriffe wie ATC/`.om` belegen keine Kompatibilität von Artefakten oder Runtimes mit der CANN-basierten Ascend-Produktlinie.

## Neue und spezialisierte Beschleunigerunternehmen

### MemryX

MemryX-MX3-Module verwenden eine Datenflussarchitektur und lassen sich zu Systemen mit mehreren Chips kombinieren. Der [Neural Compiler](https://developer.memryx.com/tools/neural_compiler.html) akzeptiert ONNX-, TensorFlow-Lite-, Keras- und TensorFlow-Modelle und erzeugt ein DFP-Paket, das die Beschleuniger-Runtime verwendet. Die [Runtime-APIs](https://developer.memryx.com/api/accelerator/accelerator.html) unterstützen Pipelines mit mehreren Modellen und Streaming.

Die MemryX-Dokumentation und Beispiele enthalten optimierte YOLO-Vor- und Nachverarbeitung für Erkennung, Segmentierung und Pose. Die [SDK-Release-Notes](https://developer.memryx.com/release_notes.html) nennen ausdrücklich Unterstützung für YOLOv10, YOLO11 und YOLO26. Das öffentliche Material ist technisch hilfreich, bietet aber keine einzelne übersichtliche Tabelle für Modell, Accuracy und Gerät wie bei Axelera.

### Kneron

Knerons Produkte KL520/KL530/KL630/KL720/KL730 reichen von kleinen USB- bis zu eingebetteten Beschleunigern. Kneron PLUS deckt die Anwendungs-Runtime ab. Die aktuelle Model Toolchain 0.33.1 dokumentiert Konvertierung, Quantisierung, Evaluierung, Simulation und `.nef`-Kompilierung für diese Ziele. KL830 erscheint in einigen [PLUS-Runtime-APIs](https://doc.kneron.com/docs/plus_c/introduction/run_examples/), aber nicht in der aktuellen [Compiler-Zielliste](https://doc.kneron.com/docs/toolchain/manual_1_overview/). Eine allgemeine Aussage über `.nef`-Kompilierung sollte daher ohne Herstellerbestätigung nicht auf KL830 übertragen werden.

Das offizielle [YOLO-Beispiel](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) demonstriert den Ablauf mit Tiny-YOLOv3. Anderes Material von Kneron behandelt Training und Deployment mit YOLOv5. Die Belege sind glaubwürdig, aber enger gefasst als die umfangreichen modernen YOLO-Matrizen von Hailo, Axelera, Rockchip oder DEEPX.

### SiMa.ai

SiMa.ai MLSoC und die [produktive Modalix-Plattform mit 50 TOPS](https://sima.ai/press-release/sima-ai-next-gen-platform-for-physical-ai-in-production/) verwenden die Palette-Softwareumgebung. ModelSDK, MLA-Compiler und ModelExecutor decken Import, Quantisierung, Kompilierung, Profiling und Ausführung ab. Je nach Arbeitslast und Produkt unterstützen die Tools Abläufe mit INT8, INT16 und BF16.

In den Release Notes des Unternehmens werden validierte Pipelines mit YOLOv7, YOLOv8 für Erkennung/Pose/Segmentierung, YOLOX, YOLOX-Segmentierung, DETR, Mask R-CNN und EfficientDet genannt. Eine aktuelle Einschränkung sollte nicht verschwiegen werden: Laut [Palette SDK 2.1 Notes](https://docs.sima.ai/v2.1.2/pages/release_notes/2.1.html) funktioniert QAT wegen einer Python-/PT2E-Quantisierungsregression nicht. Als Workaround soll die annotierte ONNX-Datei mit SDK 2.0 erstellt und mit 2.1 kompiliert werden. Zugang und Beschaffung bleiben auf Unternehmen ausgerichtet.

Die Deployment-Grenze ist ebenfalls dokumentiert: Der [ModelSDK-Kompilierungsablauf](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) erzeugt ein kompiliertes `.tar.gz` und kann eine ausführbare ELF-Datei erstellen. Das [MPK Tool](https://docs.sima.ai/v2.1.1/pages/palette/mpk_tools.html) verpackt ein auslieferbares `.mpk`. Diese Ausgaben bleiben an das MLSoC-/Modalix-Ziel und die Palette-Version gebunden.

### EdgeCortix

SAKURA-II ist ein mit 60 TOPS beworbener Beschleuniger für Vision und generative KI, der über das MERA-Software-Framework kompiliert wird. EdgeCortix liefert außerdem MERA-Technologie für Renesas' neueren RUHMI-Ablauf.

Die [SAKURA-II-Produktunterlagen](https://www.edgecortix.com/en/edgecortix-sakura-ii-accelerator-brief) belegen Hardware und Compiler. [M.2-/PCIe-Hardware](https://www.edgecortix.com/en/hardware) wird auf Anfrage zur Teststellung oder Bestellung angeboten. Eine hinreichend genaue, aktuelle öffentliche YOLO-Kompatibilitätsmatrix wurde nicht gefunden. Das ist bei der Beschaffung nützlich: Vor einer Hardwarezusage sollte eine Modellvalidierung angefragt werden.

### BrainChip

BrainChip Akida ist ein neuromorpher Ereignisprozessor und keine konventionelle Dense-Tensor-NPU. Die [MetaTF-Umgebung](https://brainchip.com/metatf-dev-tools/) besteht aus installierbaren Python-Paketen zum Erstellen von Modellen, Quantisieren auf niedrige Bitbreiten, Konvertieren, Simulieren und Ausführen auf Hardware. BrainChip verkauft inzwischen einen [ausgelieferten AKD1500-Coprozessor im M.2-Format](https://brainchip.com/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/) sowie ältere AKD1000-Hardware und Akida-2-IP.

Die aktuelle [Akida-2-Modellkarte](https://brainchip.com/wp-content/uploads/2025/04/Akida-2-Model-Card-V1.1-Mar.25.pdf) nennt einen AkidaNet0.5-YOLOv2-Detektor, CenterNet, AkidaUNet und Gesichtserkennungsnetzwerke. Weiteres Supportmaterial umfasst FOMO und ereignisbasierte Arbeitslasten. Akida unterstützt ungewöhnlich niedrige Bitbreiten bei Gewichten und Aktivierungen. Für ein erfolgreiches Deployment kann jedoch eine architekturbewusste Konvertierung nötig sein, statt Akida wie ein weiteres allgemeines INT8-ONNX-Ziel zu behandeln. Ein Modellergebnis für AKD1000 oder Akida 2 darf AKD1500 erst zugeschrieben werden, wenn ein expliziter Zuordnungs-/Fit-Bericht dies validiert.

### Blaize

Blaize verkauft Pathfinder- und Xplorer-Produkte auf Basis von P1600. Picasso SDK, NetDeploy und AI Studio bilden den Softwarepfad. Die [Produktseiten](https://www.blaize.com/products/) richten sich klar an Vision und Edge AI, stellen aber keine aktuelle öffentliche Kompatibilitätsmatrix für einzelne Modelle bereit.

Dieser Leitfaden ordnet Blaize daher als kommerziell und zugangsbeschränkt ein, statt die Lücke mit Community-Aussagen zu füllen. Ein vom Hersteller bereitgestellter Kompilierungs- und Accuracy-Bericht für das vorgesehene Modell sollte Voraussetzung sein.

## TinyML und Vision am Endgerät

### Arm Ethos-U und Alif

Arm lizenziert NPU-IP Ethos-U55, U65 und U85 an Chipunternehmen. Der [Vela-Compiler](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) übernimmt quantisierte LiteRT-/TFLite-Graphen und wandelt unterstützte Teilgraphen in benutzerdefinierte Ethos-U-Operationen um. Nicht unterstützte Operationen können auf der CPU verbleiben. „Die Anwendung läuft“ und „das ganze Netzwerk läuft auf der NPU“ sind unterschiedliche Aussagen.

Reale Implementierungen sind Ethos-U55 in Alif Ensemble E7, Ethos-U65 in NXP i.MX 93 und Ethos-U85 in neueren Systemen. Alifs [E7 AI/ML AppKit](https://alifsemi.com/support/kits/ensemble-e7appkit/) dokumentiert zwei ML-Beschleuniger und zeigt den E7-TFLite-zu-Vela-Pfad. [Ensemble E8](https://alifsemi.com/ensemble-e8-series/) kombiniert einen Ethos-U85 mit zwei Ethos-U55-NPUs. Alif veröffentlicht außerdem einen INT8-[Benchmark für einen YOLO-Fastest-Gesichtsdetektor](https://alifsemi.com/faster-ai-mcu-inferencing-low-power-consumption/) auf Familienebene bei 192 mal 192 auf Cortex-M55 plus Ethos-U55, aber keine breite, zielgenaue Matrix für moderne YOLO-Modelle. Diese speicherbegrenzten Systeme eignen sich für kleine Klassifikations- und Erkennungsnetzwerke, nicht für beliebige Detektoren mit 640 Pixeln, nur weil sie sich mit TFLite darstellen lassen.

### Infineon PSOC Edge, Himax WiseEye2 und Analog Devices MAX7800x

Infineons Familien [PSOC Edge E83](https://documentation.infineon.com/psocedge/docs/xsk1761304020519) und E84 kombinieren Cortex-M55 mit Ethos-U55 und ergänzen einen separaten NNLite-Block auf der stromsparenden M33-Seite. Das [E84-Architekturhandbuch](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) dokumentiert 8-Bit-Gewichte, 8- oder 16-Bit-Aktivierungen und einen Vela-Ablauf. Dabei werden unterstützte Operatoren zu einem NPU-Befehlsstrom und nicht unterstützte Arbeit bleibt auf der CPU. Der [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter) akzeptiert TFLite-, Keras- und PyTorch-Pfade. Die Architekturunterlagen nennen MobileNetV1/V2 als repräsentative Kernels, nicht als Ziel-Benchmarks. Das [E84 AI Kit](https://documentation.infineon.com/psocedge/docs/cci1762693051052) enthält eine Kamera und verwendet ModusToolbox. Öffentliches Material bietet jedoch noch keine YOLO-Kompatibilitätsmatrix mit benannten Modellen. Das ist eine programmierbare Vision-Plattform mit einer sich entwickelnden Modellevidenz, kein validiertes allgemeines Detektorziel.

Himax [HX6538 WiseEye2](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) kombiniert für Always-on-Vision ebenfalls Cortex-M55 und Ethos-U55. Die öffentliche Softwaredokumentation ist für diese Leistungsklasse ungewöhnlich konkret: Die [offiziellen Beispiele für Grove Vision AI Module V2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) umfassen YOLOv8n-Objekterkennung, Pose und Geschlechterklassifikation, YOLO11n-Erkennung, Face Mesh und PeopleNet. Der einheitliche Pfad verwendet TFLite Micro und Vela und fällt bei Bedarf auf CMSIS-NN und Reference-Kernels zurück. Es handelt sich ausdrücklich um kleine Modelle und Auflösungen, nicht um einen Nachweis, dass ein herkömmlicher YOLO-Graph im Serverformat passt.

Analog Devices verfolgt mit den [CNN-Beschleunigern MAX78000/MAX78002](https://www.analog.com/en/products/max78002.html) einen anderen Weg. Das öffentliche Tool [`ai8x-synthesis`](https://github.com/analogdevicesinc/ai8x-synthesis) quantisiert ein trainiertes Netzwerk und erzeugt zielspezifischen C-Code, Gewichte und Beschleunigerkonfiguration statt eines portablen ONNX-Runtime-Pakets. Evidenz des Herstellers umfasst [Gesichtsidentifikation](https://www.analog.com/en/resources/app-notes/an-2616.html), TinierSSD zur QR-Erkennung im [ADI Model Zoo](https://ez.analog.com/dsp/software-and-development-tools/edgebench/a/docs-faqs/DF912/available-models-in-adi-model-zoo) und Bildklassifikatoren. Die Geräte sind für Vision am Endgerät relevant, aber eine aktuelle offizielle Matrix für moderne YOLO-Modelle wurde nicht gefunden.

### GreenWaves GAP9

GAP9 kombiniert RISC-V-Rechenleistung mit der NE16 Neural Engine. NNTool importiert und quantisiert Netzwerke; AutoTiler und GAP SDK erzeugen daraus ausführbaren Code. Das öffentliche [GAP9-Netzwerkmenü](https://github.com/GreenWaves-Technologies/nn_menu_gap9) enthält MobileNet, EfficientNet, ResNet, MobileNet SSD sowie Anwendungen für Gesichts- und Wiedererkennung. GreenWaves veröffentlicht außerdem ein [YOLOX-Projekt zur Personenerkennung](https://github.com/GreenWaves-Technologies/yolox_people_detection) mit Accuracy- und Simulationsergebnissen.

Das ist ungewöhnlich transparente TinyML-Evidenz. Das vollständige SDK erhalten aber nur qualifizierte Kunden, und die Modelle sind deutlich kleiner als gängige YOLO-Varianten mit 640 mal 640 Pixeln.

### Lattice sensAI

Lattice sensAI ist eine FPGA-basierte Alternative zu einer festen NPU. sensAI Studio und der Neural Network Compiler bilden einen unterstützten Graphen auf konfigurierbare Beschleuniger-IP für ECP5-, iCE40-UltraPlus-, CrossLink-NX-, CertusPro-NX-, Avant-E- und Avant-X-Geräte ab. Die Ausgabe ist an das gewählte FPGA, den Beschleunigermodus, das Speicherlayout und das Bitstream gebunden und nicht als Runtime-Modell portabel.

Das aktuelle [Handbuch zum Neural Network Compiler](https://www.latticesemi.com/-/media/LatticeSemi/Documents/UserManuals/MQ3/FPGA-UG-02052-8-0-Lattice-Neural-Network-Compiler-Software.ashx?document_id=52343) enthält eine ungewöhnlich direkte Topologietabelle. Sie führt YOLOv1 für ECP5, YOLOv5 und YOLOv8 für optimierte/erweiterte CrossLink-NX- und CertusPro-NX-Modi sowie YOLO11 nur mit Advanced-Mode-IP auf. Außerdem sind SSD, MobileNetV2-SSD, ResNet, ENet, SqueezeDet und weitere Vision-Netzwerke mit nicht unterstützten Zellen für einzelne FPGA-Familien aufgeführt. Der [Advanced CNN Accelerator](https://www.latticesemi.com/en/Products/DesignSoftwareAndIP/IntellectualProperty/IPCore/IPCores05/Advanced-CNN-Accelerator-IP) zielt auf Avant-E/Avant-X/CertusPro-NX und ist kommerzielle IP. Das ist ein Beleg aus einer Compilertabelle, kein Versprechen, ein einzelnes Bitstream unterstütze alle aufgeführten Topologien gleichzeitig.

### Microchip VectorBlox

Microchips öffentlicher [VectorBlox SDK 3.1](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) bereitet vollständig quantisierte INT8-TFLite-Netzwerke für konfigurierbare CoreVectorBlox-IP in PolarFire-SoC-FPGAs vor und kompiliert sie. `tflite_preprocess` und `vnnx_compile` erzeugen die Deployment-Assets `.vnnx`, `.hex` und `.ucomp`. Das Repository enthält auch einen Simulator und einen C-Treiber für das Zielsystem. TensorFlow-, ONNX- und OpenVINO-Graphen müssen zuvor den dokumentierten TFLite-/INT8-Aufbereitungspfad durchlaufen.

Die aktuelle [Tutorialmatrix](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) enthält YOLOv5n, YOLOv8n für Erkennung, Klassifikation, OBB, Pose und Segmentierung, YOLOv9t, dünn besetzte/komprimierte YOLO-Varianten, MobileNet, EfficientNet-Lite0, ResNet18, MiDaS, FFNet und QuickSRNet. Der [C-Leitfaden zur Nachverarbeitung](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/C_Postprocessing.md) nennt separat YOLOv2/3/4/5 sowie Ultralytics-Decoder für Erkennung, Pose und OBB. SDK 3.1 unterstützt derzeit das PolarFire SoC Video Kit ausdrücklich nicht aber das PolarFire Video Kit ohne SoC. Ältere Board-Demos dürfen daher nicht mit dem aktuellen Zielvertrag verwechselt werden.

Das SDK steht öffentlich unter einer [Microchip-spezifischen Lizenz](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/LICENSE.md) zum Download bereit, die Software und abgeleitete Werke auf Microchip-Produkte beschränkt. Microchip bietet kostenlose, aber antragsbasierte Lizenzen für Libero Silver und CoreVectorBlox. Das Deployment bleibt ein FPGA-Systemvertrag: Beschleunigerkonfiguration, Bitstream, Firmware, vom SDK erzeugte Netzwerkdaten und Speicherlayout müssen zusammenpassen. Eine erfolgreiche VectorBlox-Kompilierung macht das Modell nicht zu einer Binärdatei, die auf ein beliebiges PolarFire-Design kopiert werden kann.

### Syntiant

Syntiants NDP200 zielt auf Always-on-Inferenz für Vision, Audio und Sensoren unterhalb des Leistungsbudgets gewöhnlicher Linux-SoCs. Die aktuelle [Hardwaretabelle](https://www.syntiant.com/hardware) führt NDP200 als Massenproduktion und NDP250 als Sampling. Das [NDP200-Produktbriefing](https://www.syntiant.com/ndp200/) dokumentiert CNN-, RNN- und vollständig verbundene Netzwerke bei weniger als 1 mW. Die [Ankündigung von NDP250](https://www.syntiant.com/news/syntiant-unveils-ndp250-neural-decision-processor-with-next-gen-core-3-architecture) beschreibt dagegen Always-on-Bilderkennung bei weniger als 30 mW. „Sub-Milliwatt“ sollte nicht auf beide Bauteile verallgemeinert werden.

Öffentliches Material belegt keine breite YOLO-Kompatibilität. Das Produkt sollte deshalb über eine vom Hersteller unterstützte Modellbewertung für kleine Always-on-Klassifikatoren und -Detektoren geprüft werden, statt eine allgemeine ONNX-Exportannahme zugrunde zu legen.

### Google Coral: zwei nicht verwandte Generationen

Die älteren Edge-TPU-Produkte verwenden den Edge TPU Compiler, `libedgetpu` und PyCoral mit vollständig INT8-quantisierten TFLite-Modellen. Die zentralen Repositories [Edge TPU](https://github.com/google-coral/edgetpu) und [PyCoral](https://github.com/google-coral/pycoral) sind archiviert. Das ist ein echtes Wartungsrisiko, aber keine formelle Mitteilung zum Produktende der Hardware.

Google pflegt daneben eine separate, aktive, offene [Coral NPU](https://github.com/google-coral/coralnpu): RISC-V-Beschleuniger-IP zur Integration in stromsparende SoCs. Das öffentliche Projekt stellt derzeit RTL, Hardwaresimulation, Toolchain und ELF-Beispiele bereit. Es ist kein direkter USB-/M.2-Nachfolger der Edge TPU und veröffentlicht keine aktuelle YOLO-Deployment-Matrix.

## Fallstricke bei angrenzenden GPUs, Client-NPUs und adaptiven Beschleunigern

NVIDIA, AMD, Intel und Apple tauchen häufig in NPU-Suchen auf, stellen aber keine austauschbare Beschleunigerklasse dar.

**NVIDIA Jetson:** Jetson Orin kombiniert eine CUDA-GPU mit dedizierten DLA-Kernen. TensorRT kann eine serialisierte Engine für DLA erstellen. Nicht unterstützte Layers laufen auf der GPU, sofern GPU-Fallback ausdrücklich aktiviert ist. Die aktuelle [DeepStream-YOLO-Tabelle](https://github.com/NVIDIA-AI-IOT/deepstream_tools/blob/main/yolo_deepstream/README.md) deckt YOLOv4/v7/v8/v9/11 ab, doch die meisten Einträge zielen auf GPUs. Der ausdrückliche ONNX-DLA-Eintrag ist YOLOv8s INT8. NVIDIAs [DLA-Einschränkungen](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html) erklären die Operatorgrenzen. Thor ist ein weiterer Sonderfall: Laut NVIDIAs [DriveOS-Migrationsleitfaden](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/NVIDIA_DriveOS_7.0.3_Migration_Guide.pdf) wurden DLA-Kerne in Thor zugunsten flexibler GPU-Planung entfernt. „Läuft auf Jetson“ sagt daher nicht, welcher Beschleuniger verwendet wird.

**AMD Vitis AI:** Die ältere Kria-/DPU-Generation kompilierte `.xmodel`-Graphen, die mit VART ausgeführt wurden. Das aktuelle [Vitis AI 6.2](https://vitisai.docs.amd.com/en/6.2/) ist GA für zwei getrennte Versal-AI-Edge-Pfade: [Gen1](https://vitisai.docs.amd.com/projects/gen1/en/latest/index.html) auf VEK280/VE2802 nutzt ONNX, AMD Quark und einen zielgebundenen Snapshot-/Teilgraph-Ablauf. [Gen2](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html) auf VEK385 hat einen eigenen Compile- und Runtime-Vertrag. Gen1 veröffentlicht Beispiele für YOLOv5, YOLOv7, YOLOv8 und YOLOX. Ryzen AI ist ein vierter, separater Client-NPU-Stack. Artefakte und Validierung dürfen nicht zwischen älterer DPU, Versal Gen1, Versal Gen2 und Ryzen AI übertragen werden, nur weil überall der Name Vitis oder AMD vorkommt.

**Intel OpenVINO:** OpenVINO kann über eine API CPU, GPU und Core-Ultra-NPU ansprechen. Die [Dokumentation des Intel-NPU-Plug-ins](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) nennt unterstützte NPU-Generationen. Die [Matrix verifizierter Modelle](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html) enthält gerätespezifische Spalten. Ein YOLO-Notebook belegt ein Anwendungsrezept, aber keine NPU-Validierung, sofern die Matrix dies nicht angibt. Unterstützte Operatoren, Formen, Präzision und installierter NPU-Treiber bestimmen weiterhin, ob der vollständige Graph darauf läuft. OpenVINO IR (`.xml` plus `.bin`) ist wiederverwendbare Quell-IR; ein Cache kompilierter Modelle ist geräte- und softwareabhängig.

**Apple Core ML:** `coremltools` konvertiert Modelle nach `.mlpackage`, das Xcode zu `.mlmodelc` kompiliert. Core ML kann Arbeit zwischen CPU, GPU und Apple Neural Engine verteilen, abstrahiert aber bewusst die genaue Aufteilung. Apple ist damit eine ausgezeichnete Deployment-Plattform. Aussagen wie „der gesamte YOLO-Graph lief auf der NPU“ sind ungeeignet, sofern Profilingdaten dies nicht belegen.

## Die NPU-IP-Unternehmen hinter den Chipunternehmen

Einige Unternehmen verkaufen weder ein Board noch einen fertigen Beschleuniger. Sie lizenzieren NPU-Designs an SoC-Anbieter. Sie sind wichtig, weil dieselbe zugrundeliegende Architektur unter mehreren Chipmarken auftauchen kann, der endgültige SDK- und Artefaktpfad vom Lizenznehmer aber dennoch angepasst sein kann.

| IP-Anbieter | NPU-Familie und Software | Bedeutung |
|---|---|---|
| [Arm](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) | Ethos-U55/U65/U85 und Vela | Kommt in NXP-, Alif-, Infineon-, Himax- und anderen Embedded-Produkten vor; quantisierter, auf TFLite/TOSA ausgerichteter Pfad |
| [Arm China](https://www.armchina.com/mountain?infoId=161&name=) | Zhouyi AIPU und Compass SDK | Der öffentliche [Model Zoo](https://github.com/Arm-China/Model_zoo) nennt YOLOv1-tiny/v2/v3/v4/v5, YOLOX und YOLOv8-seg. Einträge sind Referenzmodelle, kein Beleg für jeden Chip eines Lizenznehmers. |
| [VeriSilicon](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico) | Vivante-VIP-Familie und Acuity SDK | Verwandte Vivante-NPU-Familien erscheinen in mehreren SoCs, darunter die separat bezogenen A311D- und i.MX-8M-Plus-Pfade. Jeder Chipanbieter liefert weiterhin eine eigene Integration. |
| [Imagination Technologies](https://www.imaginationtech.com/products/open-access/) | PowerVR Series3NX AX3146/AX3386/AX3596; historische IMG Series4; Neural Compute SDK/IMG DNN | Lizenzierbare NNA-IP statt eines Verkaufsboards. Open Access bietet Series3NX-Konfigurationen mit 1/5/10 TOPS. Ein offizielles NC-SDK-Programm nennt PP-YOLOE, EfficientNet und HRNet, aber nicht jede IP-Konfiguration. |
| [CEVA](https://www.ceva-ip.com/product/ceva-neupro-studio/) | NeuPro-Nano/NeuPro-M und NeuPro Studio | Lizenzierbare NPU-IP mit Import, Quantisierung, Komprimierung, Graph-Kompilierung, Simulation und C-/C++-Codegenerierung |
| [MIPS](https://mips.com/processor-solutions/arc-npx-family/) | ARC NPX6 und [MetaWare MX](https://mips.com/processor-solutions/arc-metaware-mx/), ehemals Synopsys ARC | Skalierbare NPU-IP und NN-SDK, in [June 2026](https://mips.com/press-releases/gfmipsarcclose/) von GlobalFoundries übernommen und ins MIPS-Portfolio überführt; selbst kein Verkaufs-Deploymentziel |
| [Cadence](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/neuroweave-sdk.html) | Neo NPU, Tensilica DSPs und NeuroWeave SDK | Compiler-/Interpreter-Stack für SoC-Lizenznehmer; Netzwerk- und Quantisierungsunterstützung hängt von der lizenzierten Konfiguration ab |
| [Quadric](https://quadric.ai/npu-ip) | Chimera GPNPU-IP und SDK | Programmierbare NPU-/DSP-artige IP zur Lizenzierung in Chips anderer Unternehmen; das endgültige Ziel ist deren Silizium |
| [Expedera](https://www.expedera.com/products-overview/) | Origin NPU IP und Software-Stack | Lizenzierbare Edge-NPU-Architektur, kein direkt käufliches LibreYOLO-Board |

Imagination zeigt, warum Angaben zu Zugang und Lebenszyklus neben Architekturnamen stehen sollten. Das [Open-Access-Programm](https://www.imaginationtech.com/products/open-access/) bietet drei siliziumerprobte Series3NX-Konfigurationen ohne anfängliche IP-Lizenzgebühr, jedoch erst nach Unternehmensprüfung und mit kostenpflichtigem Support/Wartung sowie Produktionsroyalties. Das [Neural Compute SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) stellt Kompilierungs-, Optimierungs-, Quantisierungs- und Runtime-Tools bereit. Eine offizielle [Zusammenarbeit mit Baidu PaddlePaddle](https://www.imaginationtech.com/news/imagination-and-baidu-paddlepaddle-create-open-source-machine-learning-library-for-model-zoo/) nennt PP-YOLOE-Erkennung, EfficientNet-Klassifikation und HRNet-Segmentierung. Diese Beispiele validieren nicht jede Series3NX- oder Series4-Konfiguration. Aktuelle einzelne [Series4-Produktseiten](https://www.imaginationtech.com/product/img-4nx-mc1/) kennzeichnen die Produkte als nicht verfügbar. Series4 sollte daher als historisch gelten, sofern der Vertrieb nichts anderes bestätigt.

Diese Schicht erklärt Ähnlichkeiten zwischen Produktfamilien. Sie schafft keine Portabilität von Artefakten. Zwei SoCs mit verwandter Vivante- oder Ethos-IP können unterschiedliche Speicherzuordnungen, Treiber, Compilerversionen, Operatorsätze und Hersteller-Runtimes haben.

## Was tatsächlich ausgerollt wird: Tabelle zur Artefaktbindung

Am Ende der Compilerpipeline endet die scheinbare ONNX-Portabilität. Namen und Dateiendungen ändern sich, die praktische Regel bleibt: Bewahre das Originalmodell und die Kalibrierungsdaten auf, denn das native Artefakt lässt sich in der Regel weder auf eine andere NPU-Generation verschieben noch zuverlässig mit einem anderen SDK-Release neu erstellen.

| Hersteller-Stack | Typische Compiler-Eingabe | Was auf dem Gerät landet | Woran es gebunden ist |
|---|---|---|---|
| [Amlogic AMLNN](https://github.com/Amlogic-NN/amlnn-toolkit) | ONNX, TFLite, TorchScript oder PT2 | `.adla`; älteres A311D verwendet `.nb` | Ziel-ID, ADLA-Generation, AMLNN-Compilerbuild, NNSDK2/Runtime, ADLA Driver und BSP |
| [Rockchip RKNN](https://github.com/airockchip/rknn-toolkit2) | ONNX, TFLite oder Framework-Export | `.rknn` | RKNN-Toolkit-/Runtime-Version und Rockchip-NPU-Familie |
| [Hailo](https://hailo.ai/developer-zone/documentation/) | ONNX oder TensorFlow über ein Zwischenformat HAR | `.hef` | Hailo-Architektur sowie Compiler-/Runtime-Generation |
| [Axelera Voyager](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) | ONNX oder unterstützte Zoo-Definition | `.axm` im Alpha-Pipeline-Builder, `.axe`-Pipeline-Paket; klassisches `.axmodel` samt Manifest | Voyager-API-Generation und Metis-Zielkonfiguration |
| [DEEPX DXNN](https://github.com/DEEPX-AI/dx-all-suite) | ONNX und unterstützte Framework-Pfade | `.dxnn` | DX-COM-/DX-RT-Version und DX-M-Ziel |
| [Qualcomm QAIRT/QNN oder SNPE](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | ONNX, TFLite oder Framework-Modell | QNN-Modellbibliothek/Kontext-Binärdatei oder SNPE-`.dlc` | HTP-Architektur, SoC, Backend und Runtime-Version |
| [MediaTek NeuroPilot](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) | Konvertiertes/quantisiertes TFLite | `.dla` für Offline-Neuron-Runtime; `.tflite` für delegierten Modus | NP-/MDLA-Generation, Betriebssystemimage, Compiler und Runtime |
| [TI TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools) | ONNX oder TFLite | Verzeichnis mit importierten Artefakten plus Anwendungsmodelldateien | TIDL-Tools/Runtime und Prozessorgeneration |
| [NXP eIQ](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | Üblicherweise TFLite oder ONNX | Backend-spezifische Vela-, TIM-VX-, Neutron- oder Ara-Ausgabe | Das ausgewählte i.MX-/Ara-Modell und BSP, nicht pauschal „NXP“ |
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | TFLite, ONNX oder unterstütztes Framework-Modell | Generierte Netzwerkbibliothek/-code und Firmware-Assets | MCU-/NPU-Ziel, Speicherkonfiguration und Toolversion |
| [Infineon PSOC Edge](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) | Integer-quantisierter TFLite-/Keras-/PyTorch-Pfad | Vela-optimiertes TFLite mit Ethos-U-Befehlsstrom plus Anwendungsfirmware | E83-/E84-Variante, Ethos-U-Konfiguration, Vela, ModusToolbox und BSP; NNLite ist ein separates Ziel |
| [Renesas DRP-AI](https://github.com/renesas-rz/rzv_drp-ai_tvm) | ONNX/TFLite über TVM oder Translator | Runtime-Modellverzeichnis mit mehreren Binärdateien | RZ/V-Gerät und Translator-/Runtime-Generation |
| [Sony IMX500](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | Quantisiertes/gepacktes Netzwerk aus dem Edge-MDT-Ablauf | `.rpk`-Paket | Sensorfirmware, Speicherbudget und Packer-Version |
| [Synaptics](https://developer.synaptics.com/) | TFLite/ONNX und unterstützte Importe | `.synap` für SyNAP; `.vmfb` für Torq | SL16xx gegenüber SL261x, Software-/Hardwaregeneration |
| [Ambarella CVflow](https://www.ambarella.com/developer/) | Eingabe über Partner-Toolchain | Deployment-Paket nur für Partner; genauer Artefaktvertrag nicht öffentlich dokumentiert | CVflow-Generation, SDK/BSP und Kamerapipeline über die Cooper Developer Zone bestätigen |
| [D-Robotics BPU](https://github.com/D-Robotics/rdk_model_zoo) | ONNX und von der Toolchain unterstützter Graph | X5 `.bin`; aktuelles `rdk_s` `.hbm`; X3 mit älterem Plattformablauf | BPU-Generation, Repository-Branch und passendes OpenExplorer-/Runtime-Release |
| [AXERA Pulsar2](https://github.com/AXERA-TECH/ax-samples) | ONNX | `.axmodel` | AX-Chipziel, Pulsar2 und AXEngine-Version |
| [SOPHGO TPU-MLIR](https://github.com/sophgo/tpu-mlir) | ONNX, TFLite, TorchScript und andere | `.bmodel` auf BM, `.cvimodel` auf CV18xx | BM-/CV-Chipziel und SOPHON-Runtime-Generation |
| [Huawei Ascend CANN](https://www.hiascend.com/en/software/cann) | ONNX oder unterstützter Framework-Graph | `.om` | Ascend-Chip, CANN/ATC und Gerätefirmware/-treiber |
| [Cambricon MagicMind](https://github.com/Cambricon/magicmind_cloud) | Framework-Graph oder Austauschmodell | Serialisierte MagicMind-Engine/Modell | MLU-Architektur und Neuware-/MagicMind-Version |
| [Canaan nncase](https://github.com/kendryte/nncase) | ONNX oder TFLite | `.kmodel` | KPU-Generation und passendes nncase-Ziel-Plug-in |
| [Mobilint qb](https://www.mobilint.com/sdk-qb) | ONNX und unterstützte Framework-Modelle | `.mxq` | REGULUS-/ARIES-Ziel und qb-Compiler/Runtime |
| [Rebellions RBLN](https://docs.rbln.ai/latest/index.html) | PyTorch 2 oder TensorFlow-unterstützter Graph | `.rbln` | ATOM-Generation und RBLN-Compiler/Runtime |
| [FuriosaAI](https://developer.furiosa.ai/docs/latest/en/) | ONNX/TFLite und unterstützter Framework-Pfad | Warboy `.enf`; RNGD `.fxb` | Völlig unterschiedliche Beschleuniger- und SDK-Generationen |
| [Sunplus SNNF](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2004353133) | Von Acuity unterstütztes Netzwerk | `.nb` | SP7350-NPU-Treiber, Runtime und BSP |
| [ESWIN ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) | ONNX im dokumentierten aktuellen EsAAC-Pfad; andere Frameworks erfordern Export/Konvertierung | `.model` | Exaktes EIC7700-Familienziel und ENNP-/ESSDK-Release |
| [Nuvoton / Ethos-U](https://github.com/OpenNuvoton/NuEdgeWise) | Quantisiertes TFLite | Vela-optimiertes TFLite mit benutzerdefinierten Ethos-U-Operationen | M55M1-Speicherplan, Vela und Firmware |
| [Himax WiseEye2](https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2) | Vollständig integer-quantisiertes TFLite | Vela-optimiertes `.tflite` im Modellspeicher plus Board-Firmware wie `output.img` | HX6538-/WE2-Konfiguration, Vela, TFLite Micro, Ethos-U-Treiber und Fallback-Kernels |
| [Analog Devices AI8X](https://github.com/analogdevicesinc/ai8x-synthesis) | PyTorch-Checkpoint, Netzwerk-YAML und Beispieleingabe | Gerätespezifischer generierter C-Quellcode/Header, Gewichte und kompilierte Firmware | Speicher-/Layergrenzen von MAX78000 gegenüber MAX78002, Synthese-Tool und MSDK-Release |
| [Realtek AmebaPro2](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Eingabe für Hersteller-Konvertierungsdienst | `.nb` | RTL8735B-Firmware und VoE-/NeuralNetwork-API-Release |
| [Telechips Enlight](https://docs.topst.ai/product/p/ai) | Darknet-, TensorFlow-, ONNX- oder PyTorch-Pfad | `.enlight`-Zwischenformat plus kompiliertes Deployment-Bundle | TCC7500-NPU und TC-NN-/Enlight-Versionen |
| [T-Head HHB](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | ONNX und unterstützter Framework-Graph | `hhb.bm`, generierte Parameter/Code und ausführbare Datei | TH1520-CSI-NN2-/SHL-Stack |
| [SigmaStar MI_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | Aktuelle SSU9383CM-Dokumentation zeigt die MI_IPU-Runtime; älteres Material verwendet Eingaben aus TensorFlow-/Caffe-Familien | Älteres `.sim` wird zu `sgsimg.img`; aktuelles öffentliches Artefakt nicht verifiziert | Exakte SigmaStar-IPU und SDK-Generation; Kompatibilität des Legacy-Compilers mit SSU9383CM nicht verifiziert |
| [HiSilicon Smart Vision](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) | ONNX und von ATC unterstützter Graph | `.om` | Exaktes Smart-Vision-SoC und NNN-/SVP-NNN-Stack; keine allgemeine Ascend-Portabilität |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | ONNX, TFLite, Keras oder TensorFlow | `.dfp`-Datenflusspaket | MX-Hardwarekonfiguration und Compiler-/Runtime-Version |
| [Kneron](https://doc.kneron.com/docs/) | ONNX plus Toolchain-Konfiguration | `.nef`; KL730 NEFv2 kann `.kne` enthalten | Dokumentiertes Compilerziel, Chipgeneration, Firmware und PLUS Runtime; KL830-Kompilierung ist mit Toolchain 0.33.1 nicht belegt |
| [SiMa.ai Palette](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) | Unterstütztes Framework oder ONNX-Modell | Kompiliertes ModelSDK-`.tar.gz`, dokumentierte ausführbare ELF oder MPK-Tool-`.mpk`-Paket | MLSoC-/Modalix-Ziel und Palette-Release |
| [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | ONNX oder Netzwerkdefinition | Serialisierte Engine, meist `.engine` oder `.plan` | GPU-/DLA-Architektur, TensorRT, CUDA und oft das Gerät |
| [AMD Vitis AI](https://vitisai.docs.amd.com/en/6.2/) | Quantisierter ONNX-/Framework-Graph | Älteres `.xmodel`; NPU-Snapshot/Teilgraphen für Gen1; kompilierter Cache-Ordner oder Produktionspaket `.rai` für Gen2 | Exakte NPU-Generation/IP-Konfiguration, Plattformimage und Vitis-/Vivado-/PetaLinux-Matrix |
| [Microchip VectorBlox](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) | Vollständig quantisiertes INT8-TFLite | `.vnnx`, `.hex` und `.ucomp` samt passender Firmware/FPGA-Design | CoreVectorBlox-Konfiguration, PolarFire SoC Video Kit, SDK 3.1, Bitstream und Speicherlayout |
| [Intel OpenVINO](https://docs.openvino.ai/) | Framework-Modell oder ONNX | IR `.xml` plus `.bin` oder gerätespezifischer kompilierter Cache | IR ist wiederverwendbar; kompilierter Cache ist geräte-, treiber- und OpenVINO-spezifisch |
| [Älteres Google Edge TPU](https://coral.ai/docs/edgetpu/models-intro/) | Vollständig INT8-quantisiertes TFLite | Edge-TPU-kompiliertes `_edgetpu.tflite` | Edge-TPU-Compiler/Runtime und unterstützte quantisierte Operatoren; unabhängig von der neuen Coral-NPU-IP |
| [Google Coral NPU IP](https://github.com/google-coral/coralnpu) | C-/C++-Beispiele für das aktuelle öffentliche IP-Projekt | ELF-Beispiele für RTL-/Hardwaresimulation und spätere SoC-Integration; kein öffentlicher YOLO-Compiler-Artefaktpfad | Exakte lizenzierte/integrierte SoC-Implementierung und sich entwickelnde Open-Source-Toolchain |
| [Lattice sensAI](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | Unterstützte Netzwerkbeschreibung | FPGA-Bitstream, Beschleunigerkonfiguration und Gewichte | FPGA-Familie, sensAI-IP-Modus und Speicherimplementierung |
| [Imagination NC-SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) | Caffe, TensorFlow oder ONNX | Kundenorientierter Dateiname des kompilierten Artefakts nicht öffentlich angegeben | Lizenzierte NNA-Konfiguration, SoC-Integration, NC-SDK/DDK und Laufzeitumgebung des Lizenznehmers |

Diese Tabelle ist mehr als eine Übersicht über Dateiendungen. Compilerversion, Zielkennung, Treiber, Firmware und BSP gehören zur reproduzierbaren Identität eines Modells. Ein künftiger LibreYOLO-Exporter sollte sie zusammen mit jedem Artefakt in einem maschinenlesbaren Manifest festhalten.

## Toolzugang und Lebenszyklussignale

Hardwareverfügbarkeit und Compilerzugang sind unabhängig voneinander. Ein Board kann leicht zu kaufen sein, obwohl sein aktueller Compiler eine Kundenvereinbarung erfordert. Ein öffentliches SDK kann wiederum Silizium adressieren, das schwer zu beschaffen ist. Im Folgenden stehen konkrete Erstquellensignale, kein allgemeingültiges Ranking zur Lebensdauer.

| Plattform | Dokumentiertes Signal | Praktische Bedeutung |
|---|---|---|
| Amlogic ADLA | Öffentliche Apache-2.0-Repositories und herunterladbare Wheels; kompatible Board-Treiber/BSP können weiterhin Amlogic oder einen Boardanbieter erfordern | Compiler lässt sich leicht evaluieren, doch Weiterverteilung der Binärdateien und vollständige Kompatibilitätsmatrix sollten bestätigt werden |
| MediaTek Genio | Öffentlicher IoT AI Hub und Yocto-Leitfäden; NP8-Kompletttools und Android-Dokumente als Direktkunden-/NDA-Material gekennzeichnet | Modellevidenz ist prüfbar, Automatisierung für eigene Modelle kann aber eine Partnerschaft erfordern |
| Hailo | Öffentlicher Model Zoo und Runtime; Dataflow Compiler über Developer Zone verteilt; öffentliches `hailo-camera-apps` am May 3, 2026 archiviert | Breite Modellsuche ist offen, reproduzierbare Kompilierung erfordert das passende Konto und die passende Version. Der Archivstatus belegt weder das Produktende von Hailo-15 noch den Status des aktuellen Vision-Processor-Anwendungspakets; letzterer sollte bestätigt werden. |
| Axelera AI | Öffentliche Dokumentation und [öffentlicher Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk); Kundensupport erfordert ein Konto | Seltenes Self-Service-SDK für Beschleuniger, während Produktsupport und Beschaffung kommerziell bleiben |
| Qualcomm Dragonwing | Das [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) führt IQ-9075 als Sampling mit Laufzeit bis 2038 und QCS6490 bis July 2036. Die IQ-9075-Produktseite nennt dagegen Active. | Starkes Signal für industrielle Planung, aber der Statuskonflikt erfordert Bestätigung auf SKU-Ebene. Er garantiert nicht, dass ein KI-SDK während des gesamten Zeitraums ABI-stabil bleibt. |
| Rebellions | Die aktuelle [Support-Matrix](https://docs.rbln.ai/latest/supports/version_matrix.html) kennzeichnet CA02/CA12 als EoL und CA22/CA25 als Active; CA21 fehlt. | Beschaffung und SDK-Kompatibilität sind selbst innerhalb der ATOM-Familie von der Karte abhängig |
| NVIDIA Jetson | Die [offizielle Modullebenszyklustabelle](https://developer.nvidia.com/embedded/lifecycle) führt Orin-Module bis January 2032 und AGX Orin Industrial bis July 2033; für Developer Kits gibt es keine Lebenszykluszusage. | Produktionssysteme sollten auf Modulen statt auf der Verfügbarkeit von Entwicklungskits beruhen |
| Sony IMX500 auf Raspberry Pi | Das [Produktbriefing der AI Camera](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf) nennt Produktion mindestens bis January 2028. | Konkrete Mindestangabe für dieses Kameramodul, nicht für jedes IMX500-Produkt |
| Amlogic A311Y3 | Die [Einführungsseite von 2026](https://www.amlogic.com/News/index248.html) bewirbt eine Mindestgarantie von zehn Jahren Produktlebensdauer. | Vielversprechendes Signal für die neue Plattform; Vertrags- und SKU-Details für ein Produktprogramm müssen dennoch geklärt werden |
| Google Coral | Die älteren Edge-TPU- und PyCoral-Repositories sind archiviert/schreibgeschützt. Das separate Repository zur [Coral-NPU-IP](https://github.com/google-coral/coralnpu) ist aktiv. | Wartungsrisiko der älteren Software, aber allein kein formeller Hardware-EOL-Hinweis. Es sagt nichts über die Lebensdauer des unabhängigen Open-Source-IP-Projekts aus. |

## Warum TOPS keine Kaufempfehlung sind

Spitzen-TOPS können innerhalb eines Anbieters und einer Architektur nützlich sein. Herstellerübergreifende Vergleiche sind meist ungültig, wenn nicht alle folgenden Punkte übereinstimmen:

- Numerische Präzision, etwa INT8, INT4, FP16 oder ein gemischter Modus
- Ob ein Multiply-Accumulate als eine oder zwei Operationen gezählt wird
- Dichte oder strukturiert dünn besetzte Arithmetik
- Eingabeauflösung, Batch-Größe und Modellgraph
- Accuracy nach der Quantisierung
- Nur NPU-Latenz oder Latenz der vollständigen Anwendungspipeline
- Speichertransfers und Host-Fallback
- Thermischer Zustand sowie Dauerbetrieb statt Spitzenlast

Ein Detektor ist eine Pipeline: Bilderfassung, Größenänderung/Letterboxing, Normalisierung, Geräteübertragung, neuronale Inferenz, Decode, NMS, Koordinatenskalierung und Anwendungslogik. Hersteller veröffentlichen oft nur den mittleren Inferenzschritt. [MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/) ist wertvoll, weil es Szenarien, Qualitätsziele und Regeln für Messungen auf Systemebene definiert, statt Marketing-Arithmetik isoliert zu vergleichen.

Für YOLO-Deployments sollte der minimale glaubwürdige Benchmarkbericht Folgendes enthalten:

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

Ohne diese Angaben messen zwei FPS-Zahlen meist unterschiedliche Dinge.

## Eine NPU für Computer Vision auswählen

Entscheide in dieser Reihenfolge und nicht nach dem größten Wert auf einer Produktseite.

1. **Graph-Kompatibilität belegen.** Kompiliere genau das exportierte Modell, nicht einen ähnlich benannten Checkpoint aus einem Zoo.
2. **Accuracy messen.** Validiere das kompilierte Artefakt nach Kalibrierung und Quantisierung auf dem echten Aufgabedatensatz.
3. **Ablauf vollständig messen.** Berücksichtige Eingabekonvertierung, Transfers, Decode und NMS.
4. **Softwarezugang prüfen.** Kläre, ob der Compiler öffentlich, nur nach Registrierung oder nur über eine kommerzielle Vereinbarung erhältlich ist.
5. **Versionsmatrix fixieren.** Halte Compiler, Runtime, Treiber, Firmware und Zielkennungen fest.
6. **Betriebssystemrealität prüfen.** Android, Debian, Yocto, Buildroot, RTOS und Windows sind nicht austauschbar.
7. **Lieferung und Lebenszyklus prüfen.** Ein technisch hervorragender Chip eignet sich schlecht für die Produktion, wenn Module, Treiber oder Langzeitsupport unsicher sind.
8. **Erst dann Kosten, Leistung und Performance vergleichen.** Diese Messungen sind erst aussagekräftig, wenn dasselbe Modell auf beiden Zielen korrekt läuft.

### Praktische erste Auswahl nach Anwendungsfall

| Anwendungsfall | Plattformen, die sich zuerst lohnen | Grund |
|---|---|---|
| Beschleuniger speziell für Raspberry Pi | Hailo-8L/8 und Sony IMX500 | Offizielle Raspberry-Pi-Produkte, Softwareimages und konkrete Entwicklungsabläufe |
| Allgemeine Linux-Erweiterung per PCIe, M.2 oder USB | DEEPX, MemryX, Hailo und Axelera-Produkte | Erhältliche Beschleuniger-Formfaktoren, abhängig von Host-/Treiberkompatibilität |
| Kostengünstiger Linux-SBC oder Kamera | Rockchip RK3588/RK3576, Amlogic ADLA mit verfügbarem unterstütztem Board/BSP, AXERA, Canaan K230 oder Sunplus SP7350 | Integrierte Medienpipelines und öffentliche Evidenz zu Modellen/Compiler; für die A311D2-NPU VIM4 V13A+ prüfen |
| Mobilgeräte und Android mit hohem Volumen | Qualcomm QNN/AI Hub, LiteRT, Apple Core ML | Große installierte Basis und gepflegte mobile Runtimes |
| Industrielles Linux und Robotik | TI TDA4/AM6xA, NXP i.MX, Renesas RZ/V, Hailo | Eingebettete Produkte mit langer Lebensdauer sowie Kamera-/I/O-Ökosysteme |
| Sehr kleines Endgerät oder MCU | STM32N6x7, Infineon PSOC Edge, Himax WiseEye2, Analog Devices MAX7800x, Nuvoton M55M1, Alif/Arm Ethos-U, GAP9 und Syntiant | Enge Energie- und Speicherbudgets mit spezialisierten Tools und Grenzen bei der Modellgröße |
| Konfigurierbare FPGA-Vision | Microchip VectorBlox, Lattice sensAI und AMD-Vitis-AI-Ziele | Flexible Hardwarepipelines, aber Beschleunigerkonfiguration, Bitstream und Modellcompiler bilden ein versionsgebundenes System |
| PCIe-Vision mit hohem Durchsatz | Axelera Metis, Hailo, DEEPX, Mobilint, MemryX, Rebellions oder SiMa.ai | Dedizierte Beschleuniger und Ausrichtung auf mehrere parallele Streams |
| Chinaorientiertes Produktökosystem | Rockchip, AXERA, D-Robotics, SOPHGO, Huawei Ascend, HiSilicon und Cambricon | Starke regionale Boards, Toolchains und Modellbeispiele |

Diese Tabelle ist eine Recherche-Auswahlliste, kein Leistungsranking. Beschaffung, regionale Verfügbarkeit und das genaue Modell können die Reihenfolge ändern.

## Was das für LibreYOLO bedeutet

Die skalierbare Architektur besteht nicht aus Dutzenden unabhängigen Exportern. Sie braucht einen strikten Austauschvertrag und kleine Compiler- und Runtime-Adapter der Anbieter:

```text
LibreYOLO model
    -> deterministic static ONNX
    -> VendorCompiler.compile(model, target, calibration, precision)
    -> native artifact + manifest
    -> VendorRuntime.load() / infer()
    -> shared task-specific decode and Results objects
```

Jedes native Artefakt sollte ein Manifest mit folgenden Angaben erhalten:

- Anbieter, Chipziel und Boardziel
- Versionen von Compiler, Runtime, Treiber und Firmware
- Checksummen des Quell-Checkpoints und ONNX
- Eingabenamen, Formen, Layout, Farbraum, Normalisierung und Datentyp
- Quantisierungspräzision, verfügbare Skalen und Hash der Kalibrierungsdaten
- Ausgabetensor-Namen und -Formen sowie ihre semantische Bedeutung
- Erkennungsaufgabe, Klassennamen und Angabe, ob Decode/NMS auf dem Chip oder Host läuft
- Gemessene numerische Parität und Ergebnisse zur Accuracy der Aufgabe

LibreYOLO bietet bereits portablen [ONNX-Export](/docs/export/onnx), [Quantisierungstools](/docs/export/quantization), einen direkten, bewusst eng gefassten [RKNN-Export](/docs/export/rknn) und einen ehrlichen Ablauf mit externem Compiler für [Hailo](/docs/export/hailo). Nach den Kriterien dieses Leitfadens ist Amlogic ADLA ein starker nächster Integrationskandidat: Das Toolkit ist öffentlich, die Modellabdeckung überschneidet sich stark mit LibreYOLO, und der ältere A311D-Pfad kann ausdrücklich getrennt bleiben.

Die Priorität nach Amlogic sollte sich an der Hardware der Nutzer und an der Möglichkeit zur Accuracy-Validierung orientieren, nicht allein an der Compilerverfügbarkeit. DEEPX, Sony IMX500, D-Robotics, AXERA, SOPHGO und Qualcomm sind technisch attraktiv. TI, NXP, ST und Renesas werden besonders wertvoll, wenn Industriepartner Boards und dauerhaften CI-Zugang bereitstellen können.

## Beobachtungsliste: echtes Silizium ohne öffentlichen Integrationsvertrag

Ein Unternehmen ganz auszuschließen kann eine Marktübersicht verzerren. Es als „unterstützt“ einzubeziehen kann schlimmer sein. Diese Anbieter verkaufen, bemustern oder haben relevantes Silizium angekündigt. Die öffentlichen Belege weisen jedoch noch keinen reproduzierbaren aktuellen Compiler-, Artefakt- und Modellpfad mit konkreten Modellnamen nach, der sich für ein LibreYOLO-Backend eignet.

| Unternehmen | Was real ist | Warum es auf der Beobachtungsliste bleibt |
|---|---|---|
| [Samsung](https://semiconductor.samsung.com/processor/automotive-processor/exynos-auto-v920/) | Exynos Auto V920 enthält eine Dual-Core-NPU mit beworbenen bis zu 23.1 TOPS. | Samsung stellt sein [Neural SDK Drittentwicklern nicht mehr bereit](https://developer.samsung.com/neural/overview.html). Samsung ONE/Circle belegt keinen Zugang zur Exynos-NPU. |
| [Novatek](https://www.novatek.com.tw/en-global/Milestone/aboutus_milestones) | Aktuelle Unternehmensmeilensteine nennen Edge-AI- und Edge-Vision-/Imaging-AI-SoCs; ein Meilenstein von 2020 erwähnte MobileNet, SSD und YOLOv3. | Keine aktuelle öffentliche Matrix für Bauteilnummern, Compiler, Artefakte, Operatoren und Modelle. Historisches YOLOv3 ist kein Nachweis eines aktuellen SDK. |
| [Nextchip](https://www.nextchip.com/en/adas/adas.php?idx=5) | Automotive-Prozessoren APACHE5/NVS2900 und aktueller [APACHE6/NVS3000](https://www.nextchip.com/en/adas/adas.php?idx=7) mit aiMotive-aiWare-NPU; die APACHE6-Seite nennt aktuell sowohl 12 als auch 8 TOPS. | aiWare Studio und eine Runtime existieren, aber es wurde kein öffentliches Artefakt, Quantisierungsleitfaden, Operatorsatz oder YOLO-Matrix gefunden. Die widersprüchlichen Herstellerwerte sollten nicht stillschweigend aufgelöst werden. |
| [Black Sesame Technologies](https://bst.ai/en.html) | Huashan A1000/A2000 und Wudang-C-Serien für Automotive-/Edge-AI | Öffentliche Produktinformationen sind vorhanden, aber keine Self-Service-Toolchain, kein Artefaktvertrag und kein reproduzierbarer Model Zoo |
| [Ingenic](https://en.ingenic.com.cn/products-detail/id-19.html) | T41/T40-Kamera-SoCs mit NPU-Aussagen für niedrige Bitbreiten; Magik AI beschreibt PTQ/QAT und Graph-Kompilierung. | Es wurde kein aktueller herunterladbarer Compiler, Artefaktvertrag oder genaue YOLO-Liste des Anbieters gefunden. |
| [Fullhan](https://fullhan.com/en/index.php?a=type&c=article&tid=9) | Mehrere IPC-SoCs bewerben NPU-Leistungen von 0.5 bis 2 TOPS; die [MC6880-NVR-Familie](https://fullhan.com/en/index.php?a=type&c=article&tid=48) nennt 4 TOPS. | Keine öffentlichen Unterlagen zu NN-Compiler, Runtime, Framework oder Modellen, die eine unabhängige Reproduktion erlauben |
| [Goke Microelectronics](http://www.gokemicro.com/News/info.aspx?itemid=437) | Eine eigene Seite zu Smart-Kamera-Bauteilen GK7606V1/GK7206V1/GK7203V1 bewirbt integrierte NPU-Leistung, wurde zum Zeitpunkt der Veröffentlichung aber nur über veraltetes HTTP ausgeliefert. | Kein öffentlicher Converter, Runtime-Vertrag oder Modellmatrix mit konkreten Namen; der Zugang erfolgt über OEM-Kanäle. |
| [Chengheng Micro](https://en.chenghengmicro.com/) | CH37 bewirbt 64 INT8 TOPS und FP16/FP32/FP64-Modi; laut [Zeitstrahl des Unternehmens von 2026](https://chenghengmicro.com/about.html) begann die Kleinserienproduktion. | Es wurden kein öffentliches SDK, Compilervertrag oder benannte Modellvalidierung gefunden. Als frühe kommerzielle Plattform ohne Self-Service statt als reproduzierbarer Deployment-Nachweis behandeln. |
| [Bouffalo Lab](https://github.com/bouffalolab/bouffalo_sdk) | BL808 enthält die BLAI-100-NPU und es gibt aktuelle offizielle SDK-Repositories. | Das gepflegte SDK bietet keinen aktuellen offiziellen BLAI-Konvertierungs-/Beispielpfad; übrig gebliebene Abläufe beruhen auf alten oder Community-Belegen. |

Diese Beobachtungsliste folgt bewusst der Evidenz. Ein Anbieter kann in die Tabellen für auslieferbare Produkte wechseln, wenn er eine aktuelle Toolchain oder Evaluierungsmöglichkeit, einen zielbezogenen Artefaktvertrag und mindestens ein benanntes Modell mit End-to-End-Rezept veröffentlicht.

## Was bewusst nicht behauptet wird

Dieser Artikel behauptet nicht, dass jedes genannte Modell mit jedem Upstream-Repository funktioniert. Modelle in Vendor Zoos sind häufig angepasst, an anderen Ausgabeknoten aufgeteilt oder mit eigener Nachverarbeitung kombiniert.

Auch die folgenden Aussagen werden nicht als Support-Nachweis gewertet:

- Ein Compiler importiert ONNX.
- Ein Chip bewirbt einen Android-NNAPI-Treiber.
- Ein Distributor nennt ein Board „YOLO-kompatibel“.
- Ein Community-Repository führt einen Fork eines einzelnen Modells aus.
- Ein Modell lässt sich fehlerfrei kompilieren.
- Ein Hersteller meldet NPU-FPS ohne Accuracy-Ergebnis.

Google-Tensor-Beschleuniger in Smartphones und zahlreiche proprietäre OEM-ASICs existieren ebenfalls. Google bietet Tensor jedoch nicht als allgemeines Self-Service-Compilerziel vergleichbar mit den obigen Plattformen an. Die Existenz eines Unternehmens, ein NPU-Blockdiagramm oder Android-Beschleunigung reichen nicht aus, um eine LibreYOLO-Integrationsaussage zu erfinden.

Cloud-Beschleuniger wie Google TPU, AWS Inferentia/Trainium, Microsoft Maia und ausschließlich für Rechenzentren vorgesehene KI-Karten liegen ebenfalls außerhalb des Umfangs. Dieser Leitfaden behandelt Hardware, die ein Entwickler für Computer Vision plausibel am Edge einsetzen kann.

Die Modelllizenzierung ist eine separate Ebene. Wenn ein Chipanbieter ein YOLO-Konvertierungsrezept veröffentlicht, erteilt das keine Rechte an den Upstream-Gewichten, dem Trainingscode oder einem kompilierten Derivat. Für das vorgesehene Produkt müssen Hardwareunterstützung, SDK-Lizenz und Modelllizenz jeweils geprüft werden.

## Forschungsmethodik und Korrekturen

Für jedes Unternehmen wurden vier Primärquellenbereiche gesucht:

1. Eine aktuelle Produktseite oder ein Datenblatt, das den Chip benennt.
2. Compiler- und Runtime-Dokumentation zum tatsächlichen Deploymentpfad.
3. Ein Model Zoo, eine Kompatibilitätstabelle oder ein End-to-End-Tutorial mit Vision-Modellen beim Namen.
4. Ein Signal zu Lebenszyklus oder Zugang, das zeigt, ob Entwickler die Tools tatsächlich erhalten können.

GitHub-Organisationen des Herstellers gelten als Herstellerquellen. Dokumentation des Boardanbieters wird als Ökosystembeleg gekennzeichnet, wenn der Chiphersteller das Material nicht selbst veröffentlicht. Leistungsangaben Dritter wurden aus den Vergleichstabellen ausgeschlossen.

Der fertige Entwurf wurde anschließend in getrennten Anbietergruppen- und tabellenübergreifenden Prüfungen erneut auditiert. Dabei wurden Zielumfang, Artefaktnamen, Präzision, Modell- gegenüber Postprozessor-Evidenz, angekündigter gegenüber ausgeliefertem Status, Lebenszyklusangaben, Benchmark-Nenner und sämtliche Unternehmenszahlen nochmals geprüft. Wenn sich zwei Primärquellen widersprechen, macht der Leitfaden den Konflikt sichtbar, statt stillschweigend den bequemeren Wert auszuwählen.

Dieser Markt verändert sich schnell. Wenn du für eines dieser Unternehmen arbeitest und ein Chip, SDK, eine Modellliste oder ein Zugangsstatus falsch ist, sende LibreYOLO die genaue URL zur öffentlichen Dokumentation samt Version. Korrekturen mit Primärquellenbeleg sollten diesen Text ersetzen; undokumentierte Marketingaussagen nicht.

## FAQ

### Wie viele Edge-AI-NPU-Unternehmen gibt es?

Eine allgemeingültige Zahl gibt es nicht, da Produkthersteller, IP-Lizenzgeber, intelligente Sensoren, GPUs und proprietäre Automotive-ASICs überlappen. Dieser nicht abschließende Leitfaden erfasst mit Stand August 2026 insgesamt 69 Unternehmen und Plattform-Ökosysteme mit relevanten Edge-AI-Chips, Beschleunigerplattformen, lizenzierbarer NPU-IP oder glaubwürdigen Kandidaten auf der Beobachtungsliste.

### Was ist eine NPU?

Eine Neural Processing Unit ist spezialisierte Hardware für Operationen neuronaler Netze wie Faltungen und Matrixmultiplikationen. Hersteller verwenden unter anderem die Bezeichnungen NPU, AIPU, BPU, KPU, DLA, HTP, TPU und MLA. Kompilierte Modelle lassen sich in der Regel nicht zwischen Unternehmen übertragen.

### Welche NPU-Unternehmen unterstützen YOLO-Modelle offiziell?

Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 über das offizielle AI-Camera-Repository von Raspberry Pi, STMicroelectronics, Renesas, Lattice, Himax, Microchip und mehrere weitere Unternehmen haben eigene Model-Zoo-Einträge, Tutorials oder validierte Ergebnisse für mindestens eine YOLO-Generation. Die konkrete Generation, Aufgabe, Chipziel und SDK-Version sind weiterhin entscheidend.

### Lässt sich jedes ONNX-Modell auf jeder NPU ausführen?

Nein. ONNX ist ein Austauschformat und keine Garantie für Hardwarekompatibilität. Ein NPU-Compiler muss jeden Operator, jede Tensorform und jeden Datentyp im Graphen unterstützen. Statische Formen, Graph-Aufteilungen, benutzerdefinierte Operationen und ein CPU-Fallback sind häufig.

### Welche NPU eignet sich am besten für YOLO?

Einen allgemeingültigen Sieger gibt es nicht. Hailo, Rockchip, Axelera AI, DEEPX und Amlogic bieten eine umfangreich dokumentierte YOLO-Unterstützung, während Qualcomm eine außergewöhnlich große Gerätebasis hat. Entscheide anhand der Accuracy nach der Quantisierung, der Latenz der gesamten Pipeline, der Dauerleistung, des Preises, der Verfügbarkeit, des SDK-Zugangs und der Betriebssystemunterstützung.

### Warum lassen sich NPU-TOPS-Werte nicht direkt vergleichen?

Hersteller können unterschiedliche Präzisionen, Sparse-Operationen, Multiply-Accumulates und Annahmen zur Spitzenauslastung zählen. TOPS berücksichtigen weder Speicherverkehr noch nicht unterstützte Operatoren, Vorverarbeitung, Nachverarbeitung oder Host-Overhead. Vergleiche dasselbe Modell, dieselbe Präzision, Auflösung, Accuracy und vollständige Pipeline.

### Was ist NPU-Kalibrierung?

Bei der Kalibrierung werden repräsentative, meist nicht annotierte Einsatzbilder durch das Modell geführt, damit der Compiler Aktivierungsbereiche für INT8 oder ein anderes Format mit niedriger Präzision schätzen kann. Zufällige oder nicht repräsentative Bilder können zwar ein kompiliertes Artefakt ergeben, aber die Accuracy für die Aufgabe verschlechtern.

### Unterstützt LibreYOLO Edge-NPUs?

LibreYOLO exportiert ONNX und mehrere Runtime-Formate, bietet für ausgewählte Rockchip-Modelle einen direkten RKNN-Compilerpfad und dokumentiert den externen Hailo-Kompilierungsablauf. Für jedes andere herstellerspezifische Artefakt ist weiterhin das SDK des Herstellers nötig. Bis es kompiliert, validiert und auf Hardware ausgeführt wurde, sollte es als Integrationskandidat bezeichnet werden.
