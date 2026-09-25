---
title: "69 empresas de NPU para Edge AI: chips, SDK y YOLO en visión artificial (2026)"
description: "Guía documentada de 69 empresas y ecosistemas de NPU para Edge AI: chips, SDK, artefactos, cuantización y compatibilidad documentada con visión artificial y YOLO."
date: 2026-08-15
author: Xuban
tags: [edge-ai, npu, computer-vision, yolo, hardware, ai-accelerators, amlogic, hailo, rockchip]
faq:
  - q: "¿Cuántas empresas de NPU para Edge AI existen?"
    a: "No hay un total universal de empresas porque se solapan los proveedores de productos, los licenciantes de IP, los sensores inteligentes, las GPU y los ASIC automotrices privados. Esta guía no exhaustiva sigue a 69 empresas y ecosistemas de plataforma con chips Edge AI pertinentes, plataformas aceleradoras, IP de NPU licenciable o silicio relevante en observación, a fecha de agosto de 2026."
  - q: "¿Qué es una NPU?"
    a: "Una unidad de procesamiento neuronal es hardware especializado para operaciones de redes neuronales, como convoluciones y multiplicación de matrices. Los proveedores usan términos como NPU, AIPU, BPU, KPU, DLA, HTP, TPU y MLA, y los modelos compilados generalmente no son portables entre empresas."
  - q: "¿Qué empresas de NPU ofrecen compatibilidad oficial con modelos YOLO?"
    a: "Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 mediante el repositorio oficial de Raspberry Pi AI Camera, STMicroelectronics, Renesas, Lattice, Himax, Microchip y otras empresas tienen entradas de primera parte en model zoos, tutoriales o resultados validados para al menos una generación de YOLO. La generación, tarea, chip de destino y versión del SDK concretos siguen siendo importantes."
  - q: "¿Se puede ejecutar cualquier modelo ONNX en cualquier NPU?"
    a: "No. ONNX es un formato de intercambio, no una garantía de compatibilidad con el hardware. El compilador de NPU debe admitir todos los operadores, las formas de tensores y los tipos de datos del grafo. Son habituales las formas estáticas, los cortes de grafo, las operaciones personalizadas y la ejecución alternativa en CPU."
  - q: "¿Cuál es la mejor NPU para YOLO?"
    a: "No hay un ganador universal. Hailo, Rockchip, Axelera AI, DEEPX y Amlogic tienen una amplia compatibilidad documentada con YOLO, mientras que Qualcomm ofrece un alcance excepcional de dispositivos. Elige según la precisión tras la cuantización, la latencia de toda la canalización, la potencia sostenida, el precio, la disponibilidad, el acceso al SDK y la compatibilidad con el sistema operativo."
  - q: "¿Por qué no se pueden comparar directamente las cifras TOPS de las NPU?"
    a: "Los proveedores pueden contabilizar precisiones, operaciones dispersas, operaciones multiply-accumulate y supuestos de utilización máxima distintos. TOPS no refleja el tráfico de memoria, los operadores no compatibles, el preprocesamiento, el postprocesamiento ni la carga del host. Compara el mismo modelo, precisión, resolución, precisión y canalización completa."
  - q: "¿Qué es la calibración de una NPU?"
    a: "La calibración ejecuta imágenes representativas, normalmente sin etiquetar, del despliegue a través del modelo para que el compilador estime los rangos de activación de INT8 u otro formato de baja precisión. Las imágenes aleatorias o poco representativas pueden generar un artefacto compilado y, aun así, perjudicar la precisión de la tarea."
  - q: "¿LibreYOLO es compatible con NPU edge?"
    a: "LibreYOLO exporta a ONNX y varios formatos de runtime, tiene una ruta directa al compilador RKNN para algunos modelos Rockchip y documenta el flujo externo de compilación para Hailo. Cualquier otro artefacto nativo de proveedor sigue necesitando su SDK y debe describirse como candidato a integración hasta que se compile, valide y ejecute en hardware."
---

**No existe un único mercado de NPU. Esta guía no exhaustiva sigue a 69 empresas y ecosistemas de plataforma: 51 proveedores con chips o plataformas desplegables, nueve proveedores de IP de NPU y nueve empresas claramente identificadas como candidatas en observación.** Abarcan varias clases incompatibles de aceleradores y casi tantos stacks de compilador. La mayoría promete el mismo flujo: exportar un modelo, cuantizarlo, compilarlo, copiar un artefacto propietario a una placa y llamar al runtime del proveedor. Los detalles determinan si ese proceso lleva una tarde o un trimestre de ingeniería.

Esta guía cartografía las empresas que ofrecen hardware creíble para visión artificial en 2026. Registra los chips pertinentes, los nombres de SDK y compiladores, los artefactos de despliegue, el nivel de acceso público y los modelos que el proveedor documenta realmente. También presta especial atención a Amlogic, cuyo stack ADLA más reciente difiere sustancialmente del antiguo ecosistema NPU A311D.

> **Alcance de la investigación:** Las fuentes se revisaron el 15 de agosto de 2026. Las afirmaciones sobre modelos concretos proceden de páginas de producto, documentación, model zoos, repositorios o tutoriales del propio proveedor, salvo que se indique expresamente lo contrario. Los TOPS anunciados son cifras de los proveedores, no resultados de benchmarks comparables e independientes. Esta es una guía para desarrolladores, no asesoramiento de inversión ni una clasificación de pago.

**Navegación rápida:** [tablas de empresas](#npu-companies-and-platforms-at-a-glance) | [análisis de Amlogic](#amlogic-two-npu-generations-not-one) | [perfiles de proveedores](#the-strongest-public-deployment-ecosystems) | [tabla de artefactos compilados](#what-you-actually-deploy-the-artifact-lock-in-table) | [acceso y ciclo de vida](#tool-access-and-lifecycle-signals) | [hoja de ruta de LibreYOLO](#what-this-means-for-libreyolo)

## El hallazgo más importante

El hardware está fragmentado, pero el patrón de despliegue es sorprendentemente uniforme:

```text
PyTorch checkpoint
    -> ONNX or TFLite interchange graph
    -> representative calibration data
    -> vendor quantizer and graph compiler
    -> chip-specific binary or model package
    -> vendor runtime on the target
    -> application preprocessing, decode, NMS and rendering
```

[ONNX permite expresamente runtimes, generadores de código e implementaciones de hardware](https://onnx.ai/onnx/repo-docs/IR.html), pero un archivo ONNX solo es el punto de entrega. No significa que todos los operadores ONNX puedan traducirse para todos los aceleradores. Las formas de entrada estáticas, las restricciones de operadores compatibles, las reglas de cuantización y la ejecución alternativa en el host determinan qué se ejecuta realmente en la NPU.

Por tanto, el stack de software forma parte del chip. Una NPU nominalmente rápida con un compilador restringido o poco fiable puede ser un destino de despliegue peor que un dispositivo más pequeño con una cadena de herramientas pública, un model zoo, un simulador y un runtime estable.

## Qué significa «compatible» en esta guía

La documentación de los proveedores usa la palabra *compatibilidad* de forma muy imprecisa. Este artículo distingue cuatro niveles de evidencia:

| Nivel de evidencia | Qué demuestra | Qué no demuestra |
|---|---|---|
| **Modelo validado** | El proveedor publica una entrada de modelo, un resultado o una tabla de compatibilidad específica del chip | Que tu checkpoint modificado conserve la misma precisión |
| **Ejemplo oficial** | El proveedor ofrece un tutorial o una demostración integral para un modelo concreto | Que se compilen otros tamaños, tareas o generaciones |
| **Capacidad del compilador** | El SDK importa un framework o expone operadores compatibles | Que un grafo YOLO concreto funcione de principio a fin |
| **Marketing o acceso restringido** | El producto está pensado para visión y existe un SDK privado | Compatibilidad del modelo reproducible públicamente |

La distinción importa. «Importa ONNX» no equivale a «es compatible con la segmentación de YOLO11». Un modelo puede analizarse y luego ejecutarse en CPU, compilarse con valores numéricos de salida incorrectos, desbordar la memoria del acelerador o perder precisión durante la cuantización.

## Empresas y plataformas NPU de un vistazo

Las tablas mezclan deliberadamente SoC, aceleradores discretos, sensores inteligentes y plataformas adyacentes de GPU/FPGA. Compiten por la misma decisión de despliegue aunque los proveedores utilicen términos como NPU, AIPU, BPU, KPU, DLA, HTP, TPU o MLA.

### SoC embebidos, sensores inteligentes y procesadores industriales

| Empresa | Chips o plataformas pertinentes | SDK, compilador y artefacto | Evidencia de visión artificial documentada públicamente | Acceso |
|---|---|---|---|---|
| [Amlogic](https://github.com/Amlogic-NN/amlnn-toolkit) | A311D2, S928X, S905X5/S905D5, A311Y3, C308L/C302X/C302X2, T968D4, C305X2 y A123X | AMLNN Toolkit y `libnnsdk.so`, `.adla` compilado; stack Acuity `.nb` heredado por separado para A311D | El [entorno de pruebas de modelos](https://github.com/Amlogic-NN/amlnn-model-playground) documenta YOLOv5/6/7/8/10/11, YOLOX, YOLOE, YOLO-World, PP-YOLOE, segmentación, pose y OBB en A311D2, S905X5, A311Y3, C305X2 y A123X; los demás chips de esta fila son destinos del compilador, no entradas de esa matriz de compatibilidad | GitHub y wheels públicos; se requiere un BSP/controlador compatible |
| [Rockchip](https://github.com/airockchip/rknn-toolkit2) | RK3562/3566/3568, RK3576, RK3588, RV1126B | RKNN-Toolkit2, RKNN Runtime, `.rknn` | YOLOv5/6/7/8/10/11, YOLOX, YOLO-World, PP-YOLOE, segmentación, pose y OBB en el [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) | GitHub público; wheels binarios |
| [Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | Plataformas Snapdragon y Dragonwing, QCS6490, QCS8550 y Dragonwing IQ-9075 | QAIRT/QNN, SNPE y AI Hub; binarios de contexto QNN o DLC | La [colección de modelos AI Hub](https://github.com/qualcomm/ai-hub-models) incluye YOLOv3/5/6/7/8/9/10/11/26, YOLOX, YOLO-World, YOLOR, RF-DETR y modelos de detección/segmentación/pose | Documentación pública; los requisitos de SDK/cuenta varían |
| [Texas Instruments](https://github.com/TexasInstruments/edgeai-tidl-tools) | AM62A, AM67A, AM68A, AM69A y TDA4x; TDA54-Q1 en vista previa | Processor SDK Edge AI y TIDL | El [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo) publica modelos optimizados de detección, segmentación, pose y clasificación para las rutas actuales AM6xA/TDA4; todavía no es una matriz de destinos para TDA54-Q1 en vista previa | GitHub público y Processor SDK |
| [NXP](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | i.MX 8M Plus, i.MX 93, i.MX 95 y Kinara Ara-1/Ara240 adquiridos | eIQ con TIM-VX, Vela, Neutron Converter y Ara SDK | El [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) incluye recursos o recetas para YOLOv4-tiny/v8, NanoDet, CenterNet, FastestDet, SSD Lite y YOLACT; la entrada YOLOv5 solo contiene documentación | Algunos componentes son públicos y otros requieren cuenta |
| [STMicroelectronics](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) | Variantes STM32N6x7, entre ellas STM32N657/647, con acelerador Neural-ART | STM32Cube AI Studio y ST Edge AI Core | El repositorio actual de servicios incluye Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 y ST-YOLOX, además de pose y segmentación | Herramientas y repositorios públicos |
| [Infineon](https://documentation.infineon.com/psocedge/) | PSOC Edge E83/E84 con Cortex-M55 y Ethos-U55; acelerador NNLite separado en el lado M33 | ModusToolbox, [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter), TFLite Micro y Arm Vela | La documentación de arquitectura menciona MobileNetV1/V2 como kernels representativos y el kit E84 AI incluye cámara, pero no se encontró una matriz de validación YOLO de primera parte para PSOC Edge | Documentación pública, componentes de SDK y kits de evaluación E84 actuales |
| [Renesas](https://github.com/renesas-rz/rzv_drp-ai_tvm) | RZ/V2L, V2M, V2MA, V2H y V2N | DRP-AI Translator, DRP-AI TVM y RUHMI | La [lista de modelos RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) valida tamaños de YOLOv5/v8/v11/26, YOLOX y variantes de pose y segmentación | GitHub público y SDK de placa |
| [Sony](https://www.aitrios.sony-semicon.com/edge-ai-devices/imx500) | Sensor de visión inteligente IMX500 y Raspberry Pi AI Camera | Edge-MDT, Model Compression Toolkit y empaquetador IMX500; paquete `.rpk` | El [zoo de Raspberry Pi IMX500](https://github.com/raspberrypi/imx500-models) incluye YOLOv8n, YOLO11n, EfficientDet Lite, NanoDet+ y SSD | Flujo de Raspberry Pi público; se aplican las condiciones de las herramientas de Sony |
| [Ambarella](https://www.ambarella.com/developer/model-garden/) | Familias CV72/CV75, CV5/CV52 y CV3-AD | Cooper Developer Platform, compilador CVflow y DAG de runtime | El model garden público menciona YOLOX-S, RTMDet-nano, DeepLabV3+, TopFormer, OWL-ViT y LLaVA OneVision | Principalmente acceso restringido a socios |
| [Synaptics](https://developer.synaptics.com/docs/sl/overview) | Astra SL1600/SL1680; SL2611/13/15/17/19; familia MCU SR100 separada | SyNAP `.synap` en SL16xx; Torq/IREE `.vmfb` en SL261x; SDK SR separado orientado a Ethos-U55 | Benchmark oficial SyNAP YOLOv8n/v8s y ejemplo Torq YOLOv8 en SL261x; la evidencia de SR debe evaluarse por separado | Portal para desarrolladores; algunas descargas restringidas |
| [MediaTek](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) | Genio 360/360P/420/520/720 con NP8 y MDLA 5.3; Genio 510/700/1200 con NP6/MDLA anteriores | NeuroPilot Converter, `ncc-tflite`, Neuron Runtime y `.dla`; ruta ONNX Runtime en algunos sistemas nuevos | El IoT AI Hub oficial publica páginas de modelos y benchmarks de YOLOv5 y YOLOv8, además de modelos de clasificación y rostros | Documentación Yocto pública; paquetes NP8 clave y material Android requieren acceso de cliente directo/NDA |
| [Allwinner](https://docs.aw-ol.com/v853/en/npu/dev_npu/) | SoC de visión V853 con NPU Vivante de 1-TOPS | Conversión Acuity/Pegasus y runtime `viplite` | La documentación oficial de V853 menciona YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet y redes de rostros/personas | Documentación pública; la descarga del paquete SDK puede requerir cuenta |
| [Canaan/Kendryte](https://github.com/kendryte/nncase) | K230/K230D actuales; KPU K210/K510 heredados | Compilador y runtime `nncase`; `.kmodel` | La guía nncase de K230 cubre compilación/simulación/runtime de YOLOv5s; el catálogo oficial de demostraciones y el [registro de cambios actual de CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) documentan tareas YOLOv8/11/26 | Compilador/PyPI públicos; el complemento del chip es binario |
| [Alif Semiconductor](https://alifsemi.com/support/kits/ensemble-e7appkit/) | Ensemble E7 con dos aceleradores ML, incluido Ethos-U55; [E8](https://alifsemi.com/ensemble-e8-series/) con un Ethos-U85 y dos NPU Ethos-U55 | Despliegue basado en Arm Vela y TFLite Micro | Alif publica un benchmark INT8 a nivel de familia para un detector facial YOLO-Fastest, pero no una matriz amplia de YOLO moderno por destino | Documentación pública y recursos para kits de evaluación |
| [Himax](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) | MCU de IA de borde HX6538 WiseEye2 con Cortex-M55 y Ethos-U55 | TFLite Micro más Arm Vela, con CMSIS-NN y kernels de referencia como alternativa | Los ejemplos oficiales de [WiseEye2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) incluyen detección, pose y clasificación YOLOv8n, detección YOLO11n, face mesh y PeopleNet | GitHub y documentación públicos, y placa asociada disponible para compra |
| [Analog Devices](https://www.analog.com/en/products/max78002.html) | MCU de IA MAX78000 y MAX78002 con aceleradores CNN de bajo consumo | [`ai8x-training`](https://github.com/analogdevicesinc/ai8x-training), `ai8x-synthesis`/`izer` y MSDK; C y pesos generados | El material oficial cubre identificación/detección facial, RetinaNet, Visual Wake Words, clasificadores y reconocimiento de acciones; no se encontró un despliegue YOLO de primera parte | Herramientas GitHub, documentación y placas de evaluación públicas |
| [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo) | Placas BPU RDK X3/X5/Ultra y nuevas series S100 | OpenExplorer/Algorithm Toolchain y `hbm_runtime`; X5 `.bin`, `rdk_s` actual `.hbm` | La rama actual `rdk_x5` documenta YOLOv5/v5u/v8/v9/v10/11/12/13/26, YOLOE, YOLO-World, segmentación, pose, clasificación, OCR y CLIP para RDK X5; X3 y la serie S usan ramas distintas | GitHub público; algunos paquetes de la cadena de herramientas dependen de la plataforma |
| [AXERA](https://github.com/AXERA-TECH/ax-samples) | AX650, AX637, AX630C, AX620Q y AX615 | Compilador Pulsar2 y AXEngine; `.axmodel` | Los ejemplos oficiales actuales cubren YOLOv5/6/7/8/9/10/11/13/26, YOLOX y YOLO-World; las tareas y destinos varían según el chip | Ejemplos y documentación públicos; el compilador se distribuye por separado |
| [SOPHGO](https://github.com/sophgo/tpu-mlir) | BM1684/1684X/1688/1690 y CV186X/CV18xx | TPU-MLIR y runtime SOPHON; `.bmodel` o `.cvimodel` | Las demostraciones oficiales cubren YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, OBB, segmentación, rostros, SAM y OCR | Compilador de código abierto más runtime del proveedor |
| [Huawei Ascend](https://www.hiascend.com/en/software/cann) | Ascend 310/310P/310B y productos edge Atlas 200I/300I | CANN, compilador ATC y AscendCL; `.om` | El [Ascend ModelZoo](https://github.com/Ascend/modelzoo) oficial incluye YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN y modelos de segmentación | Documentación pública; los paquetes y kernels CANN deben coincidir con el destino |
| [Cambricon](https://github.com/Cambricon/magicmind_cloud) | MLU370-X4/S4 en la matriz pública citada; MLU270 es hardware heredado no cubierto por esa matriz | Neuware y MagicMind | La matriz MagicMind 1.7 del repositorio enumera YOLOv3/4/5/7/8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN y modelos de segmentación | Ejemplos de modelos históricos públicos; acceso al SDK/contenedor actual restringido |
| [Sunplus](https://sunplus.atlassian.net/wiki/spaces/C3/pages/1971126471/SP7350%2BSpecification) | SP7350/C3V, aproximadamente 4.1 a 4.6 TOPS anunciados | Docker NPU Vivante Acuity, runtime SNNF y `.nb` | La documentación oficial ofrece YOLOv5 y un despliegue [YOLOv8 integral personalizado](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) | Código/documentación públicos y ecosistema de placas |
| [ESWIN Computing](https://www.eswincomputing.com/en/news/info/96.html) | SoC RISC-V EIC7700/EIC7700X y [EIC7702/EIC7702X](https://www.eswincomputing.com/en/news/info/104.html) de doble chip | ENNP: EsQuant, compilador EsAAC actual, simulador y runtime ESSDK; `.model`; la documentación anterior usaba `ennc-compile` | La documentación ENNP alojada por Milk-V incluye un [tutorial integral de YOLOv3](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3) y ejemplos de MobileNetV2 y ResNet | Documentación de primera parte y de placas; descargas regionales |
| [Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m55-mcus/m55m1-series/index.html) | MCU M55M1 con Ethos-U55-256 | TFLite Micro, Arm Vela y NuEdgeWise/NuML | Los ejemplos oficiales de [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) mencionan YOLOv8-nano, YOLOX-nano, YOLO Fastest, SSD-MobileNet y modelos de clasificación | Cadena de herramientas MCU principalmente pública |
| [Realtek](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Plataforma de cámara AmebaPro2 RTL8735B / AMB82-mini | SDK Arduino/FreeRTOS, VoE y API NeuralNetwork; `.nb` | Los modelos empaquetados incluyen YOLOv3-tiny, YOLOv4-tiny, YOLOv7-tiny, SCRFD y MobileFaceNet | Runtime público; el [acceso al conversor personalizado](https://ameba-doc-arduino-sdk.readthedocs-hosted.com/en/latest/FAQ/offline_ai_model_conversion_steps.html) requiere contacto |
| [Telechips](https://docs.topst.ai/product/p/ai) | Placa TCC7500 / TOPST AI, 8 TOPS anunciados | TC-NN-Toolkit / Enlight SDK; `.enlight` intermedio y paquete compilado para despliegue | Un [proyecto oficial TOPST](https://docs.topst.ai/blog/31) desplegó YOLOv8s; las conversiones UFLD v1/v2 fallaron por capas no compatibles y solo se propuso una solución alternativa de división/postprocesamiento. [YOLOv4](https://community.topst.ai/t/segmentation-fault/358) y [otro flujo YOLOv8](https://community.topst.ai/t/segmentation-fault-error/415) aparecen en hilos de soporte comunitarios | Documentación de la placa pública; las descargas del compilador/operadores suelen requerir permisos |
| [T-Head](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | TH1520, 4-TOPS INT8 anunciado, en LicheePi 4A | Compilador HHB y CSI-NN2/SHL; `hhb.bm` más código/parámetros generados | Los ejemplos oficiales del proveedor de la placa ejecutan YOLOv5n/s y MobileNetV2 en la NPU | Ejemplos públicos; incertidumbre por antigüedad/mantenimiento de la cadena de herramientas |
| [SigmaStar](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | SoC actual de cámara inteligente SSU9383CM y familias IPU anteriores | Runtime MI_IPU actual; una cadena SGS_IPU [anterior](https://wx.comake.online/doc/doc/SigmaStarDocs-SSC9381G_9351_Pudding-ULS00V040-20210913/customer/development/dla/tools.html) generaba `.sim` y luego `sgsimg.img` | Un [postprocesador de SDK oficial anterior](https://wx.comake.online/doc/doc/Sigmastar_SDK_v1.2.2/module/User_Guide/Common/SigmaStar_Post_Processing_Module.html) menciona SSD y YOLOv1/2/3; no está verificada la compatibilidad pública de esos modelos con SSU9383CM | Silicio actual, pero la evidencia pública de compilador/modelos corresponde a generaciones distintas |
| [HiSilicon](https://www.hisilicon.com/cn/products/smart-vision/machine-vision/hi3516cv610) | SoC de visión inteligente Hi3516CV610/DV500, Hi3519DV500 y Hi3403V100 | ATC a `.om` con runtime de placa NNN/SVP-NNN | La matriz HiSpark específica del destino, en particular para Hi3403 y Hi3591P, menciona YOLOv3 a YOLO11, pose, segmentación, OBB, OCR y entradas de profundidad; no valida todos los SoC de esta fila | Activo y distinto de Ascend; los modelos del repositorio se etiquetan solo para uso no comercial |

### Aceleradores edge dedicados y módulos

| Empresa | Silicio pertinente | SDK y artefacto | Evidencia pública de modelos | Acceso |
|---|---|---|---|---|
| [Hailo](https://github.com/hailo-ai/hailo_model_zoo) | Hailo-8, Hailo-8L, Hailo-10H y familia Hailo-15 | Dataflow Compiler y HailoRT; `.hef` | YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 y YOLO26, además de YOLOX, DAMO-YOLO, SSD, EfficientDet, segmentación, pose y OBB, con tablas de modelos específicas por generación | Model Zoo público; compilador mediante Developer Zone |
| [Axelera AI](https://docs.axelera.ai/sdk/reference/models/model-zoo/) | Productos Metis disponibles; productos Europa y Titania anunciados | Voyager SDK público; Pipeline Builder alfa `.axm`/`.axe`, paquetes `.axmodel` clásicos | YOLOv3/5/7/8/9/10/11/26 validados, YOLOX, YOLO-NAS, OBB, pose, segmentación y muchos modelos no YOLO | SDK público en GitHub; soporte a clientes requiere cuenta |
| [DEEPX](https://developer.deepx.ai/modelzoo/) | DX-M1 y DX-M1M | DXNN SDK, DX-COM y DX-RT; `.dxnn` | El Model Zoo tenía 354 entradas con DX-COM 2.4.0/DX-RT 3.4.0 al revisarlo el 15 de agosto de 2026, incluidas YOLOv3 a YOLO11 y YOLO26, YOLOX, SSD, EfficientDet, NanoDet, segmentación, pose y OBB | Recursos y suite para desarrolladores públicos |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | Acelerador MX3 y módulos multichip | MemryX SDK, Neural Compiler y runtime; `.dfp` | Los ejemplos y notas oficiales de versión cubren detección, segmentación y pose, incluidos YOLOv10, YOLO11 y YOLO26 | Documentación y SDK públicos |
| [Kneron](https://doc.kneron.com/docs/) | KL520, KL530, KL630, KL720 y KL730 tienen documentación actual del compilador; KL830 aparece en algunas API PLUS, pero no en la lista actual de destinos del compilador | Kneron PLUS y Model Toolchain; `.nef` en destinos documentados del compilador | El [flujo YOLO oficial](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) se centra en Tiny-YOLOv3; otros materiales cubren YOLOv5 | Documentación pública; los paquetes/herramientas varían según el chip |
| [SiMa.ai](https://docs.sima.ai/pages/palette/modelsdk.html) | MLSoC y plataforma Modalix de 50-TOPS en producción | Palette, ModelSDK, MLA Compiler y ModelExecutor | Lanzamientos públicos validan YOLOv7/v8, YOLOX, pose/segmentación, DETR, Mask R-CNN y EfficientDet | Documentación pública; SDK de producto comercial |
| [EdgeCortix](https://www.edgecortix.com/en/hardware) | SAKURA-II, acelerador anunciado de 60-TOPS en productos M.2 y PCIe | Compilador MERA y framework | El proveedor anuncia compatibilidad desde visión hasta IA generativa, pero no publica una matriz actual suficientemente precisa de compatibilidad YOLO | Solicitud comercial de prueba/compra; validación dirigida por socios |
| [BrainChip](https://brainchip.com/metatf-dev-tools/) | Coprocesador/módulo M.2 AKD1500 disponible; plataformas AKD1000 heredadas; IP Akida 2 | Paquetes MetaTF: `akida-models`, `quantizeml`, `cnn2snn` y runtime `akida` | La ficha de Akida 2 menciona un detector YOLOv2 AkidaNet0.5, CenterNet, AkidaUNet y reconocimiento facial; esos resultados no validan automáticamente AKD1500 | Paquetes Python básicos públicos; la asignación al hardware depende del destino |
| [Blaize](https://www.blaize.com/products/) | Productos P1600, Pathfinder y Xplorer | Picasso SDK, NetDeploy y AI Studio | La visión es un mercado objetivo, pero no se encontró una matriz pública auditable de compatibilidad de modelos concretos | Comercial/restringido |
| [Google Coral](https://github.com/google-coral/edgetpu) | Productos heredados Edge TPU USB, PCIe, M.2 y Dev Board; IP [Coral NPU](https://github.com/google-coral/coralnpu) nueva y separada | Edge TPU Compiler/`libedgetpu`/PyCoral heredados; la nueva Coral NPU RISC-V expone IP, RTL/simulación y ejemplos ELF | Los ejemplos heredados se centran en SSD MobileNet, clasificación, DeepLab y MoveNet; la nueva IP no tiene matriz pública de despliegue YOLO y no es una sucesora directa del producto Edge TPU | Repositorios principales heredados archivados; IP Coral NPU nueva en desarrollo activo |
| [Lattice Semiconductor](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | FPGA ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E y Avant-X | sensAI Studio, Neural Network Compiler e IP aceleradora configurable | La tabla actual del compilador incluye YOLOv1, YOLOv5, YOLOv8 y YOLO11 en modos específicos por destino, además de SSD, MobileNetV2-SSD, ResNet y ENet | Compilador/manuales públicos; las condiciones de IP varían y Advanced CNN Accelerator es comercial |
| [Mobilint](https://www.mobilint.com/sdk-qb) | Aceleradores REGULUS de 10-TOPS y ARIES MLA100 de 80-TOPS | qb SDK; compilador INT8 y `.mxq` | El [model zoo público](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) enumera detección YOLOv3/5/7/8/9/10/11/12/26 y variantes de segmentación, pose y OBB | Documentación/modelos públicos; SDK y hardware comerciales |
| [Rebellions](https://rebellions.ai/developers/) | ATOM+ CA22 y ATOM-Max CA25 activos; ATOM CA02/ATOM+ CA12 EoL; estado de herramientas ATOM-Lite CA21 incierto | Compilador/runtime/profiler RBLN; `.rbln` | Los lanzamientos oficiales mencionan YOLOv3, familias YOLOv5/6/7/8 y tutoriales actuales YOLOv8 | Documentación pública; wheel del compilador requiere credenciales del portal |
| [FuriosaAI](https://furiosa.ai/warboy) | Warboy Gen1 orientado a visión; generación RNGD separada | Furiosa SDK; `.enf` INT8 en Warboy, `.fxb` separado en RNGD | El zoo Warboy documenta SSD, YOLOv5M/L y YOLOv7-w6-pose; la hoja de ruta de RNGD marca compatibilidad con YOLOv8m completada en 2024 Q4, pero no expone una matriz pública actual detallada de visión | Acceso por IAM/cuenta; mantener separadas las dos generaciones |

### Plataformas adyacentes que se comparan con las NPU

| Empresa | Hardware | Stack de despliegue | Por qué se incluye en la comparación |
|---|---|---|---|
| [NVIDIA](https://github.com/NVIDIA-AI-IOT/deepstream_tools/tree/main/yolo_deepstream) | GPU Jetson Orin más DLA; GPU Jetson Thor sin DLA | JetPack, TensorRT y DeepStream; `.engine` | Despliegue de visión muy maduro; las herramientas DeepStream oficiales documentan YOLOv4/v7/v8/v9/11, incluida una configuración OBB YOLO11. Muchos resultados usan la GPU y las capas de Orin DLA no compatibles pueden ejecutarse en GPU como alternativa. |
| [AMD](https://vitisai.docs.amd.com/en/6.2/) | Destinos Kria/DPU heredados; Vitis AI 6.2 GA para Versal AI Edge Gen1 VEK280/VE2802 y Gen2 VEK385; NPU cliente Ryzen AI separadas | `.xmodel` heredado; snapshot NPU vinculado al destino en Gen1; directorio de caché compilada o paquete de producción `.rai` en Gen2 | Vitis AI publica material de visión, pero DPU heredada, Versal Gen1, Versal Gen2 y Ryzen AI son contratos distintos de compilación y despliegue. |
| [Microchip](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox) | FPGA PolarFire SoC con IP aceleradora CoreVectorBlox configurable | VectorBlox SDK 3.1 público y Libero; recursos `.vnnx`, `.hex` y `.ucomp` | Los [tutoriales actuales](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) cubren YOLOv5n, detección/clasificación/OBB/pose/segmentación YOLOv8, YOLOv9t y otros modelos de visión; SDK 3.1 actualmente se limita a PolarFire SoC Video Kit. |
| [Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) | NPU Core Ultra, CPU y GPU integrada/discreta | OpenVINO IR o caché de modelo compilado | OpenVINO ofrece una ruta pública entre XPU; consulta las columnas NPU de la [matriz de modelos verificados](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html), en vez de asumir que todos los ejemplos YOLO de OpenVINO se validaron en NPU. |
| [Apple](https://developer.apple.com/machine-learning/core-ml/) | Apple Neural Engine en chips A-series desde A11 y chips M-series | Core ML y `coremltools`; `.mlpackage`/`.mlmodelc` | NPU de consumo muy accesible mediante Core ML, pero Apple abstrae el particionado exacto entre CPU/GPU/Neural Engine en lugar de exponer un compilador de hardware específico para YOLO. |
| [GreenWaves Technologies](https://github.com/GreenWaves-Technologies/nn_menu_gap9) | GAP9 con motor neuronal NE16 | GAP SDK, NNTool y código generado por AutoTiler | Los repositorios oficiales incluyen MobileNet SSD, detección facial, clasificación y un detector de personas [YOLOX específico](https://github.com/GreenWaves-Technologies/yolox_people_detection). El SDK completo se limita a clientes cualificados. |
| [Syntiant](https://www.syntiant.com/hardware) | NDP200 en producción masiva y procesadores NDP250 Neural Decision en fase de muestras | SDK Syntiant y paquetes de despliegue | Procesadores de visión/sensores de consumo ultrabajo siempre activos: se documenta NDP200 por debajo de 1 mW, mientras que el reconocimiento de imágenes de NDP250 se documenta por debajo de 30 mW. No se encontró una matriz pública amplia de YOLO. |

## Amlogic: dos generaciones de NPU, no una

Amlogic merece una explicación detallada porque la información web suele mezclar productos incompatibles. La empresa tiene una ruta antigua basada en VeriSilicon y una ruta ADLA propietaria más reciente. No usan el mismo compilador, artefacto ni runtime.

| Generación | Chips representativos | NPU y cadena de herramientas | Artefacto compilado | Estado práctico |
|---|---|---|---|---|
| Heredada | A311D y S905D3, habituales en Khadas VIM3/VIM3L | VeriSilicon Vivante VIPNano-QI, toolkit Acuity, KSNN y `aml_npu_sdk` anterior | `.nb` dentro de un directorio de salida `nbg_unify` | Placas y demostraciones existentes; integración heredada separada |
| ADLA2 actual | C308L/C302X, S928X, A311D2, T968D4, S905X5/S905D5 y C302X2 | Amlogic ADLA2, AMLNN Toolkit y NNSDK2 | `.adla` específico del destino; W8A8 o W8A16 | Stack actual centrado en enteros |
| ADLA3 actual | A311Y3, C305X2 y A123X | Amlogic ADLA3, AMLNN Toolkit y NNSDK2 | `.adla` específico del destino; W4A8, W8A8, W4A16, W8A16 o W16A16 | Añade capacidades nativas INT4, FP16 y BF16 |

La [ficha técnica A311D](https://dl.khadas.com/products/vim3/datasheet/a311d-datasheet.pdf) identifica la NPU original de 5-TOPS INT8. La antigua [página de aplicaciones Khadas VIM3](https://docs.khadas.com/products/sbc/vim3/npu/npu-app) y el [resumen de NPU](https://docs.khadas.com/products/sbc/vim3/npu/start) documentan DenseNet CTC, MTCNN, RetinaFace, YOLOFace, YOLOv2, YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv7-tiny y YOLOv8n; el resumen también documenta YOLOv8n-pose y marca FaceNet como obsoleto. Son ejemplos reales, pero evidencian el antiguo ecosistema Acuity `.nb`, no los chips ADLA actuales. A311D no es A311D2, y C305X no es C305X2.

Hay otra trampa de hardware. La [guía de revisiones Khadas VIM4](https://docs.khadas.com/products/sbc/vim4/configurations/identify-version) indica que la placa original V12/A311D2 revisión B no tiene NPU; la NPU anunciada de 3.2-TOPS aparece en las placas V13A y posteriores con el componente A311D2-N0D revisión C. El nombre del producto no basta para elegir un dispositivo de prueba.

### El stack AMLNN y ADLA actual

El [AMLNN Toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/7d3cc9a36179b756ce79c953c14d43acab1f2af5) actual cubre conversión de modelos, cuantización, compilación, inferencia y profiling. Su [guía de usuario fijada en mayo de 2026](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/02_Amlogic_NPU_User_Guide_V0.1.pdf) documenta importadores para ONNX, TFLite de punto flotante o cuantizado, TorchScript `.pt` y PyTorch 2 ExportedProgram/PT2. La guía detallada marca TensorFlow, Paddle y Keras como previstos, aunque el resumen del repositorio use términos más amplios. El compilador genera `.adla`. El despliegue Python usa AMLNN o `amlnn_edge_toolkit_lite`; el código C/C++ nativo enlaza `libnnsdk.so` mediante `nnsdk2.h`. El desarrollo en host puede usar ADB con `nnserver`; el runtime también tiene como destino Android, Buildroot, Yocto y algunos entornos Debian/Armbian.

Los identificadores de plataforma publicados actualmente por el toolkit incluyen:

| ID de destino | Plataforma |
|---:|---|
| `001` | C308L / C302X |
| `002` | S928X |
| `003` | A311D2 |
| `004` | T968D4 |
| `005` | S905X5 / S905D5 |
| `006` | C302X2 |
| `007` | A311Y3 |
| `008` | C305X2 / A123X en la documentación detallada y la matriz de modelos |

Ese campo de destino no es decorativo. Amlogic exige que `.adla`, `nnsdk2.h`, `libnnsdk.so`, BSP y generación del compilador/runtime coincidan. El [Inicio rápido](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/01_Amlogic_NPU_Quick_Start_guide_V0.1.pdf) fijado requiere el controlador ADLA 2.0.2 o posterior, NNSDK 3.0.0 o posterior y NNSDK2 1.0.0 o posterior; la [guía de despliegue Android](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/06_Amlogic_NPU_Android_Deployment_Guide_V0.1.pdf) concreta la relación entre biblioteca, cabecera, controlador y BSP. Trata `.adla` como un artefacto de compilación ligado a una ABI, no como un modelo portable.

La cuantización también depende de la generación de NPU. Los destinos ADLA2 001 a 006 documentan compilación W8A8 y W8A16. Los destinos ADLA3 007 y 008 añaden combinaciones W4A8, W4A16, W8A8, W8A16 y W16A16, con compatibilidad nativa con INT4, FP16 y BF16 en la [guía de operadores](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/05_Amlogic_NPU_Support_Operator_List_V0.1.pdf). Amlogic recomienda entre 200 y 500 muestras de calibración representativas. La opción de datos aleatorios es explícitamente para pruebas de rendimiento, no para una cuantización que preserve la precisión.

### Compatibilidad de modelos documentada por Amlogic

El [entorno de pruebas de modelos Amlogic fijado](https://github.com/Amlogic-NN/amlnn-model-playground/blob/03ca63ef20c4f1560722de26ab910826884f334a/README.md) es una evidencia mucho más sólida que una afirmación genérica de compatibilidad con ONNX. Su matriz publicada de compatibilidad/ejemplos menciona A311D2, S905X5, A311Y3, C305X2 y A123X. Que el compilador acepte los demás ID de destino no demuestra que se hayan validado en ellos todos estos modelos.

| Tarea | Modelos documentados |
|---|---|
| Clasificación | MobileNetV2 y ResNet50-v2; DINO en ADLA3 |
| Detección de objetos | PP-YOLOE, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv10, YOLO11, YOLOE, YOLO-World, YOLOX y detección de QR |
| Rostros y gestos | RetinaFace y Gesture Recognition |
| Segmentación | DeepLabV3, PP-LiteSeg, YOLOv5-seg y YOLOv8-seg |
| Detección orientada | YOLOv8 OBB |
| Pose | Detector/landmark BlazePose y pose YOLOv8 |
| OCR y voz | LPRNet, variantes PaddleOCR, Whisper Tiny y SenseVoice en algunos chips más recientes |
| Transformers recientes y trabajo multimodal | DETR, MobileSAM, CLIP/MobileCLIP y ejemplos LLM/VLM seleccionados de pocos bits en plataformas más recientes |

El mismo repositorio publica las siguientes cifras W8A8 de runtime de modelos. Son mediciones del proveedor de la red neuronal en la NPU, no de canalizaciones de cámara completas.

| Modelo y entrada | S905X5 | A311D2 | A311Y3 |
|---|---:|---:|---:|
| YOLOv8n, 640 x 640 | 101.72 FPS | 95.14 FPS | 191.06 FPS |
| YOLOv8s, 640 x 640 | 42.33 FPS | 42.77 FPS | 83.08 FPS |
| YOLOv8m, 640 x 640 | 19.67 FPS | 19.82 FPS | 35.30 FPS |
| YOLOv8l, 640 x 640 | 10.53 FPS | 10.12 FPS | 18.37 FPS |
| YOLO11n, 640 x 640 | 41.14 FPS | 41.48 FPS | 62.24 FPS |

Las salvedades son especialmente importantes. La guía de operadores de Amlogic sitúa NMS en software tanto en ADLA2 como en ADLA3. El README define estas cifras como resultados de runtime del modelo en la NPU nativa y excluye preprocesamiento y postprocesamiento; no especifica si incluye todas las transferencias de memoria. Las filas de precisión YOLO usan subconjuntos COCO de 300 imágenes, no el conjunto completo de validación, mientras que las filas de clasificación usan ImageNet `val1000`. Las tablas también ofrecen una latencia YOLOv8n de A311Y3 distinta de la tabla FPS, en condiciones que el repositorio no concilia. No se especifican relojes, modo de potencia, refrigeración, estado térmico estable, versiones exactas de SDK/BSP ni duración de las pruebas. Usa las cifras para entender un stack de proveedor, no para compararlo con otro.

### Por qué Amlogic es estratégicamente interesante

Amlogic combina cuatro características poco habituales: una amplia presencia de SoC embebidos, un compilador recién hecho público, una matriz actual de modelos que coincide mucho con las tareas de LibreYOLO y una integración de primera clase relativamente escasa en las bibliotecas de entrenamiento más conocidas. Los repositorios actuales también son recientes: aparecieron públicamente a finales de 2025 y principios de 2026, con manuales sustanciales fechados entre mayo y julio de 2026. Esto abre una oportunidad real para ser pioneros y conlleva un riesgo de estabilidad de API.

Una integración limpia debería usar la exportación ONNX determinista de LibreYOLO como entrada del compilador, exigir imágenes de calibración representativas para compilaciones de baja precisión, generar `.adla` junto con un manifiesto de metadatos y cargarlo mediante un adaptador NNSDK2. A311D heredado debería usar un destino `.nb` claramente distinto, porque presentar `.nb` y `.adla` como un único backend produciría fallos de compatibilidad silenciosos. Los repositorios de Amlogic tienen licencias Apache-2.0 en su nivel superior, pero conviene confirmar directamente los derechos de redistribución de wheels, bibliotecas compartidas, `nnserver` y binarios Android AAR incluidos antes de que LibreYOLO los replique.

## Los ecosistemas públicos de despliegue más sólidos

Las siguientes empresas ofrecen actualmente la combinación más clara de hardware obtenible, material técnico público y evidencia de modelos concretos. Eso no las convierte en las más rápidas para todos los casos. Significa que sus afirmaciones de compatibilidad son más fáciles de evaluar antes de comprar hardware.

### Hailo

Hailo es el ejemplo de referencia de un ecosistema dedicado a aceleradores de visión. Los modelos se analizan en un Hailo Archive, se optimizan y cuantizan con imágenes representativas y luego se compilan en un archivo Hailo Executable Format (`.hef`). Las aplicaciones cargan ese HEF mediante HailoRT.

El [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) contiene recetas, modelos preentrenados, configuraciones de postprocesamiento y datos de rendimiento para detección, segmentación, clasificación, pose y otras tareas. Su compatibilidad YOLO incluye YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 y YOLO26, además de YOLOX, DAMO-YOLO, SSD y EfficientDet. La [tabla de detección de objetos Hailo-8 fijada a una versión](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.19.0/docs/public_models/HAILO8/HAILO8_object_detection.rst) es una fuente de compatibilidad mejor que una página de producto genérica, mientras que la [aplicación independiente](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/README.md) documenta canalizaciones YOLO desplegables.

Hay un límite importante entre versiones. Hailo-8 y Hailo-8L usan las ramas anteriores Model Zoo 2.x, Dataflow Compiler 3.x y HailoRT 4.x, mientras que Hailo-10 y Hailo-15 usan la generación actual 5.x. Las recetas, el comportamiento del analizador y los HEF no son intercambiables solo porque todos los dispositivos lleven el nombre Hailo.

Hailo-15 también tiene una capa de aplicación distinta. Usa Vision Processor Software Package y Hailo Media Library en lugar de la ruta de host tipo TAPPAS de Hailo-8/10H. El repositorio de referencia público [`hailo-camera-apps`](https://github.com/hailo-ai/hailo-camera-apps) se archivó el 3 de mayo de 2026, mientras que [`hailo-apps-core`](https://github.com/hailo-ai/hailo-apps-core) sigue siendo un framework de aplicaciones público aparte. El archivo de un repositorio no demuestra el fin de vida del producto, pero los proyectos Hailo-15 deberían confirmar el paquete de aplicaciones actual en Developer Zone.

La [guía de despliegue Hailo](/docs/export/hailo) de LibreYOLO se detiene deliberadamente en la entrega de un ONNX estático, porque el compilador propietario no puede incluirse como dependencia Python. Ese es el límite honesto entre producir un grafo adecuado y afirmar que un HEF está probado.

### Rockchip

Rockchip tiene una de las mayores presencias de Linux embebido comercial y para aficionados, especialmente mediante placas RK3588, RK3576 y RK356x. El [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) público convierte y cuantiza modelos en un host Linux x86, y RKNN Runtime o Toolkit-Lite ejecuta el archivo `.rknn` resultante en la placa.

El [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) es excepcionalmente explícito sobre chips, familias de modelos y precisión. Su tabla actual enumera rutas FP16 e INT8 para YOLOv5/6/7/8/10/11, YOLOX, PP-YOLOE, YOLO-World y variantes de segmentación YOLOv5/v8; YOLOv8 OBB y YOLOv8 pose figuran solo como INT8. También cubre OCR, rostros y otras redes de segmentación.

LibreYOLO ya tiene un exportador [RKNN](/docs/export/rknn) directo y prudente. Valida cuatro variantes exactas de detección en RK3588, puede comparar el simulador del compilador con ONNX Runtime y rechaza familias no validadas, en vez de equiparar una compilación correcta con predicciones correctas. Esa afirmación limitada de compatibilidad es un buen modelo para todos los futuros backends NPU.

### Axelera AI

Axelera AI, que suele escribirse mal como «Accelera», desarrolla la AIPU Metis y la vende en productos M.2, PCIe y multichip. Metis es la familia que se puede pedir públicamente; Europa y Titania son productos de cartera anunciados, así que no conviene reducirlos a un único estado de disponibilidad. El [Voyager SDK es público en GitHub](https://github.com/axelera-ai-hub/voyager-sdk) y gestiona selección de modelos, compilación, cuantización, ejecución y canalizaciones de aplicación; el soporte a clientes sigue restringido a cuentas.

El [Voyager Model Zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo/) público es uno de los más informativos del mercado. Enumera variante del modelo, tamaño de entrada, precisión, exactitud y rendimiento medido para YOLOv3, YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 n/s/m/l/x, YOLOX, YOLO-NAS, OBB, pose y segmentación, junto con muchos modelos de clasificación y predicción densa. Publicar la pérdida por cuantización y la precisión junto a la velocidad aporta más que publicar solo los TOPS máximos.

La nomenclatura de artefactos cambió con la generación de la API. El flujo alfa de compilación [Pipeline Builder](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) genera un modelo `.axm` y puede empaquetar una canalización portable como `.axe`. La [documentación de la canalización clásica](https://docs.axelera.ai/sdk/reference/pipeline/model-formats/) describe `.axmodel` junto con metadatos del modelo y un manifiesto. La integración debe detectar la generación Voyager instalada, en vez de fijar una única extensión.

### Qualcomm

Qualcomm ofrece un alcance de despliegue enorme, pero «NPU Qualcomm» esconde varias interfaces. Según el dispositivo y la categoría del producto, los desarrolladores se encuentran con Hexagon HTP mediante QAIRT/QNN, el flujo SNPE anterior, los servicios de compilación Qualcomm AI Hub, LiteRT o el proveedor de ejecución QNN de ONNX Runtime.

El repositorio oficial [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) ofrece evidencia sólida a nivel de modelo. Publica paquetes optimizados para YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 para detección/segmentación/pose, YOLOX, YOLO-World, YOLOR, RF-DETR y muchos modelos de clasificación, profundidad y pose. AI Hub también expone profiling específico por dispositivo, importante porque un modelo compilado para una generación HTP no es automáticamente portable a otra.

El principal coste de integración es el tamaño de la matriz: Android frente a Linux embebido, componentes de pedido QCS frente a Snapdragon de consumo, arquitectura HTP, versión QNN/QAIRT y esquema de cuantización, todo importa. Las páginas actuales de Qualcomm discrepan sobre el estado de [Dragonwing IQ-9075](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075): la página del producto lo marca como Active y su [EVK](https://www.qualcomm.com/developer/hardware/qualcomm-iq-9075-evaluation-kit-evk) se puede evaluar, mientras que el [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) sigue marcando IQ-9075 como Sampling y enumera longevidad hasta 2038. El prefijo de pedido/SKU es QCS9075; Qualcomm anuncia configuraciones de 50 y 100 TOPS densos INT8, además de compatibilidad con Ubuntu/Yocto. Hay que confirmar el estado de producción para el SKU exacto de pedido, y una integración LibreYOLO debería registrar ese SKU en vez de crear una carpeta genérica llamada «Qualcomm».

### DEEPX

Los aceleradores DX-M1/DX-M1M de DEEPX usan el DXNN SDK. DX-COM compila y cuantiza el modelo, DX-RT lo ejecuta y el artefacto desplegado usa el formato `.dxnn`. La empresa publica [DX-AllSuite](https://github.com/DEEPX-AI/dx-all-suite), ejemplos de aplicación en [DX-APP](https://github.com/DEEPX-AI/dx_app) y un amplio [Model Zoo en línea](https://developer.deepx.ai/modelzoo/).

El catálogo público informaba 354 entradas con DX-COM 2.4.0 y DX-RT 3.4.0 al revisarlo el 15 de agosto de 2026. La cobertura documentada incluye YOLOv3 a YOLO11 y YOLO26, YOLOX, SSD, EfficientDet, NanoDet, DAMO-YOLO, segmentación YOLO, pose y cajas orientadas. DEEPX es un objetivo de integración especialmente plausible porque los límites entre compilador/runtime y el artefacto de despliegue están claramente definidos, y la cobertura pública de visión es amplia.

## Los SoC industriales son varios ecosistemas, no uno

Los proveedores industriales suelen ofrecer ciclos de producto más largos y mejor integración con cámaras, seguridad y tiempo real que los SoC de placas para aficionados. Sus stacks de IA pueden ser más complejos porque un proveedor puede ofrecer simultáneamente varias arquitecturas de NPU no relacionadas.

### Texas Instruments

Los procesadores Edge AI actuales de TI incluyen AM62A, AM67A, AM68A, AM69A y dispositivos TDA4 relacionados. La ruta de software combina Processor SDK Linux, compilador/runtime TIDL, [Edge AI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) y [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo). TIDL puede integrarse con ONNX Runtime, TensorFlow Lite y otros runtimes de aplicación, y delegar subgrafos compatibles.

TI publica mucho más que una afirmación de importación de framework: el zoo contiene modelos convertidos y metadatos de rendimiento para detección de objetos, segmentación, pose, clasificación, profundidad y otras tareas. TI también advierte que los artefactos del model zoo son puntos de partida para desarrollo, no recursos listos automáticamente para producción. Es una salvedad valiosa que suele faltar en las comparativas de proveedores.

TI también incluye [TDA54-Q1](https://www.ti.com/product/TDA54-Q1) como Preview, con hasta cuatro NPU C7 y hasta 400 TOPS para ese componente; la familia TDA5 más amplia se anuncia hasta 1,200 TOPS. Son afirmaciones del proveedor para una nueva generación, no permiso para atribuir la validación de modelos TIDL AM6xA/TDA4 al TDA54-Q1 antes de que TI publique una matriz específica para ese destino.

### NXP

NXP requiere al menos cuatro descripciones de backend:

| Destino NXP | Acelerador | Ruta de compilador/runtime |
|---|---|---|
| i.MX 8M Plus | NPU VeriSilicon Vivante de 2.3-TOPS | eIQ con TIM-VX/VX delegate |
| i.MX 93 | Arm Ethos-U65 | TFLite cuantizado más Arm Vela |
| i.MX 95 | NPU NXP eIQ Neutron | Neutron Converter y eIQ runtime |
| Ara-1 / [Ara240](https://www.nxp.com/products/ARA240) | NPU discreta derivada de Kinara; Ara240 anuncia hasta 40 eTOPS | Ara SDK, integrado en la historia general de eIQ |

El [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) contiene recursos o recetas para YOLOv4-tiny, YOLOv8, NanoDet, CenterNet, FastestDet, SSD Lite, YOLACT y otros modelos; la entrada YOLOv5 se añadió solo como documentación. Una entrada validada para un backend no es evidencia para los otros tres. La propia [guía de exportación YOLO para plataformas NXP i.MX](https://community.nxp.com/t5/i-MX-Processors-Knowledge-Base/Exporting-YOLO-Models-for-NXP-i-MX-Platforms/ta-p/2381361) ilustra este problema de conversión específica del destino. NXP afirma que el monolítico [eIQ Toolkit dejó de recibir actualizaciones después de la versión 1.17 en Q3 de 2025](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/eIQ-FAQ/ta-p/1099741); los flujos actuales usan paquetes independientes como eIQ Neutron SDK.

NXP [completó la adquisición de Kinara en octubre de 2025](https://media.nxp.com/news-releases/news-release-details/nxp-completes-acquisitions-aviva-links-and-kinara-advance/). La [ficha Ara240](https://www.nxp.com/docs/en/fact-sheet/ARA240DNPUFS.pdf) llega a 40 eTOPS anunciados por el proveedor e informa 313 imágenes por segundo para YOLOv8n. NXP incluye Ara SDK y eIQ Toolkit en la página del producto, pero eso no establece el intercambio de artefactos con la ruta Neutron separada de [i.MX 95](https://eiq.nxp.com/learning-hub/convQuant/neutron.html). El [módulo M.2 de 16 GB](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT) está activo, mientras que la opción USB está en preproducción. NXP define la «e» de eTOPS como «equivalent» (equivalente), no «effective» (efectivo); no es la métrica TOPS INT8 densa de otro proveedor y no debe compararse directamente.

### STMicroelectronics

Los [dispositivos STM32N6x7](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html), incluidos STM32N657 y STM32N647, incorporan un acelerador Neural-ART de 600-GOPS en un producto de clase MCU; la línea de uso general N6x5 no incluye ese acelerador. [STM32Cube AI Studio](https://www.st.com/en/development-tools/stedgeai-cubeai.html) y ST Edge AI Core analizan y optimizan los modelos importados, mientras que [STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) ofrece flujos de despliegue, optimización, benchmarking y ejemplos.

La [descripción general actual de detección de objetos](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/docs/README_OVERVIEW.md) documenta Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 y ST-YOLOX, junto con servicios de clasificación, pose y segmentación. No es la misma clase de rendimiento que una tarjeta PCIe de 200-TOPS. Es interesante porque la captura de cámara, la inferencia y el control pueden convivir con límites muy estrictos de potencia y memoria.

### Renesas

Los procesadores RZ/V de Renesas integran aceleradores DRP-AI. El software ha evolucionado desde DRP-AI Translator y DRP-AI TVM hacia [RUHMI](https://www.renesas.com/en/software-tool/ruhmi-framework), impulsado por la tecnología MERA de EdgeCortix. El repositorio abierto [RZ/V DRP-AI TVM](https://github.com/renesas-rz/rzv_drp-ai_tvm) y la [lista de validación RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) mencionan variantes n/s/m de YOLOv5, YOLOv8, YOLO11 y YOLO26, YOLOX, redes de pose y segmentación, además de modelos de clasificación.

Renesas es un objetivo viable para robótica e industria, pero hay que registrar la placa exacta, la generación DRP-AI, la versión del traductor y el postprocesamiento en CPU para que los resultados sean reproducibles.

## Sensores inteligentes y procesadores orientados a cámaras

### Sony IMX500

Sony IMX500 no es un procesador de aplicaciones normal. Apila la captura de imagen y el procesamiento de IA para que la inferencia pueda realizarse en el sensor, reduciendo la cantidad de datos de imagen que debe salir de la cámara. Raspberry Pi AI Camera hace que esta arquitectura sea excepcionalmente accesible.

El repositorio oficial de modelos [Raspberry Pi IMX500](https://github.com/raspberrypi/imx500-models) publica modelos empaquetados YOLOv8n, YOLO11n, EfficientDet Lite0, NanoDet+ y SSD MobileNetV2 FPN Lite. La [documentación AI Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) explica la conversión, el empaquetado y el postprocesamiento en el sensor. La unidad de despliegue es un paquete RPK, no un archivo ONNX genérico, y la memoria del sensor y las restricciones de operadores son límites de diseño fundamentales. La [página del producto](https://www.raspberrypi.com/products/ai-camera/) de Raspberry Pi indica que esta cámara seguirá en producción al menos hasta enero de 2028.

### Ambarella

Los procesadores CVflow de Ambarella están muy establecidos en cámaras, drones y visión automotriz. Las familias de productos actuales incluyen CV72/CV75 para cámaras, CV5/CV52 para sistemas de visión de mayor rendimiento y dispositivos automotrices CV3-AD. La plataforma de desarrollo Cooper y el flujo de compilación/runtime CVflow constituyen el stack de software identificado públicamente.

El [model garden público para desarrolladores](https://www.ambarella.com/developer/model-garden/) menciona YOLOX, RTMDet, DeepLabV3+, TopFormer, OWL-ViT y modelos multimodales. Sin embargo, la documentación detallada del compilador y las descargas siguen orientadas a socios. Por tanto, una integración Ambarella debería comenzar con una relación con el proveedor o el OEM, no con la suposición de que hay disponible un paquete pip público.

### Synaptics Astra

Synaptics tiene tres destinos pertinentes. Los sistemas Astra SL1600/SL1680 usan el [toolkit SyNAP](https://developer.synaptics.com/docs/synap/introduction), que convierte un modelo de origen y una descripción YAML en un paquete `model.synap`. Los dispositivos SL2611/13/15/17/19 más recientes usan la [plataforma Torq](https://developer.synaptics.com/docs/torq/introduction), un flujo basado en IREE/MLIR que genera `.vmfb`. La [serie SR100](https://developer.synaptics.com/docs/sr/introduction-sr) es una familia MCU independiente basada en Cortex-M55 y Ethos-U55 con su propio SDK. Estos artefactos y API no son intercambiables.

El [tutorial oficial de benchmark YOLO](https://developer.synaptics.com/docs/sl/tutorials/vision/benchmark-yolo) es excepcionalmente concreto: exporta YOLOv8 a TFLite, realiza calibración asimétrica UINT8 en SyNAP, compila para SL1680 y luego ejecuta `synap_cli` y la aplicación de detección de objetos. Demuestra YOLOv8n/v8s en SL1680 y describe compatibilidad SyNAP hasta YOLO11. Una [guía independiente de detección de objetos SL261x](https://developer.synaptics.com/docs/sl/sl2600/getting-started/object-detection) ejecuta YOLOv8 desde un `.vmfb` Torq. Son ejemplos oficiales específicos de destinos, no una afirmación de que los tres grupos sean compatibles con cualquier grafo YOLO.

### MediaTek Genio

MediaTek merece una entrada completa porque su [IoT AI Hub](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) actual expone la matriz de hardware/software, paquetes de modelos y tablas de benchmarks que los análisis de mercado anteriores no podían auditar. Genio 360/360P/420/520/720 usan NeuroPilot 8 con MDLA 5.3; Genio 510/700 usan NP6 con MDLA 3.0; Genio 1200 usa NP6 con MDLA 2.0. MediaTek indica que la generación NeuroPilot/MDLA, el conjunto de operadores, el compilador y el runtime quedan vinculados por versión durante la vida de cada SoC.

La ruta de IA analítica se centra en TFLite:

```text
PyTorch model
    -> NeuroPilot Converter and representative calibration
    -> quantized .tflite
    -> version-matched ncc-tflite compiler
    -> chip-generation-specific .dla
    -> Neuron Runtime on MDLA
```

Los productos Genio más recientes también ofrecen delegación LiteRT en línea y una ruta ONNX Runtime CPU/NPU en sistemas operativos compatibles. Son modos de ejecución distintos de `.dla` fuera de línea. La [guía de arquitectura de software](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/software_architecture.html) y la [matriz de recursos](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) dejan clara la distinción.

La [tabla oficial de modelos analíticos](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical.html) publica entradas de benchmark YOLOv5s y YOLOv8s y guías de conversión para varias generaciones MDLA. Indica expresamente que no distribuye artefactos YOLO preconvertidos por las restricciones AGPL-3.0. Sus mediciones offline Quant8, 640 x 640, incluyen 5.35 ms y 8.04 ms respectivamente en Genio 720. La [página del modelo YOLOv8s](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical/YOLOv8s.html) publica tensores de entrada/salida, tiempos específicos del backend y advertencias sobre operaciones personalizadas MediaTek. Son benchmarks del proveedor, no resultados integrales de cámara.

El acceso es el factor limitante. La documentación Yocto pública es detallada, pero el paquete actual todo en uno de conversor/compilador NP8 y gran parte del material Android figuran como recursos bajo NDA o para clientes directos. Una cuenta normal de desarrollador no otorga el mismo acceso que una cuenta MediaTek Online de cliente. Por eso LibreYOLO debería tratar Genio como técnicamente demostrado, pero dependiente de una asociación para una integración reproducible del compilador para modelos propios.

## Ecosistemas Edge AI centrados en China

Algunas de las plataformas de visión más capaces y accesibles a bajo coste están poco representadas en las listas de mercado en inglés.

### D-Robotics y plataformas BPU derivadas de Horizon

D-Robotics mantiene el ecosistema de placas de desarrollo RDK alrededor de aceleradores BPU usados en RDK X3, X5, Ultra y productos nuevos de la serie S100. La rama actual `rdk_x5` de [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo) documenta rutas YOLOv5/v5u, YOLOv8/9/10/11/12/13/26, YOLOE, YOLO-World, detección, segmentación, pose, clasificación, OCR y CLIP para RDK X5. X3 usa una rama independiente, mientras que la entrega actual de la serie S está en la rama [`rdk_s`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s) del zoo principal; el antiguo repositorio [`rdk_model_zoo_s`](https://github.com/D-Robotics/rdk_model_zoo_s) contiene demostraciones históricas. Por tanto, la lista moderna de X5 no demuestra que todos esos modelos se validaran en X3, Ultra o S100.

OpenExplorer/Algorithm Toolchain importa ONNX o Caffe para PTQ y admite flujos QAT orientados a PyTorch. El despliegue depende de la plataforma: RDK X5 actual usa `.bin` mediante `hbm_runtime` sobre `libdnn`, mientras que el flujo principal actual `rdk_s` usa `.hbm` mediante una API `hbm_runtime` homónima sobre `libhbucp`; X3 conserva su rama e interfaces de inferencia anteriores. El material previo de `rdk_model_zoo_s` también describe rutas históricas `.bin` y `.hbm`, así que esos artefactos deben emparejarse con su rama y cadena de herramientas correspondientes. Los operadores no compatibles pueden ejecutarse en CPU como alternativa. Los ejemplos públicos son útiles, pero la combinación de versión de RDK OS, firmware de placa, cadena de herramientas y binario del modelo forma parte del contrato de compatibilidad.

### AXERA

Las familias AX650/AX630/AX620 de AXERA aparecen en cámaras y placas de IA compactas como la línea MaixCAM de Sipeed. Pulsar2 importa ONNX, realiza calibración y análisis de precisión, y genera `.axmodel`; AXEngine ejecuta el artefacto.

Los [ejemplos oficiales AXERA](https://github.com/AXERA-TECH/ax-samples) cubren YOLOv5/6/7/8/9/10/11/13/26, YOLOX y YOLO-World, con compatibilidad de tarea y chip que varía según la plataforma. En destinos compatibles, los ejemplos YOLO26 actuales incluyen detección, pose, segmentación y OBB. Los [ejemplos de conversión Pulsar2](https://pulsar2-docs.readthedocs.io/en/latest/appendix/model_convert_examples.html) muestran el trabajo real: nombres exactos de tensores, cortes de salida, transformaciones de layout, archivos de calibración y ajustes del hardware de destino.

### SOPHGO

TPU-MLIR de SOPHGO es uno de los stacks de compilador más atractivos para una integración independiente porque el compilador es de código abierto. Importa ONNX, PyTorch, TFLite y Caffe, reduce y cuantiza grafos, y genera `.bmodel` para destinos BM o `.cvimodel` para hardware CV18xx.

El proyecto [TPU-MLIR](https://github.com/sophgo/tpu-mlir) admite flujos FP32, BF16, FP16 e INT8 según el destino. La [colección de demostraciones SOPHON](https://github.com/sophgo/sophon-demo/blob/release/README_EN.md) menciona YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, variantes OBB/segmentación, SSD, CenterNet, RetinaFace, SAM/SAM2 y OCR. Como siempre, una demostración en BM1684X no valida el mismo artefacto en BM1688 o CV18xx.

### Huawei Ascend

La línea de inferencia edge de Huawei incluye silicio Ascend 310/310P/310B en productos Atlas 200I y 300I. CANN proporciona el compilador ATC, runtime AscendCL y kernels de operadores específicos del chip; ATC convierte ONNX y otras representaciones fuente en un modelo offline `.om`.

La [documentación actual Atlas 200I DK A2](https://www.hiascend.com/en/hardware/developer-kit-a2/specification) y los ejemplos CANN proporcionan una ruta YOLOv5 `.om` para Ascend 310B. El [Ascend ModelZoo](https://github.com/Ascend/modelzoo) más amplio y antiguo contiene YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN, pose y modelos de segmentación, pero no debe presentarse como si cada entrada se hubiera revalidado en 310B. CANN, kernels, firmware y SoC deben coincidir. Un `.om` para Ascend310P no es un artefacto Ascend genérico.

### Cambricon

Los aceleradores MLU de Cambricon usan Neuware y el motor de inferencia MagicMind. El repositorio público [MagicMind Cloud](https://github.com/Cambricon/magicmind_cloud) está fijado a MagicMind 1.7 y una matriz MLU370-X4/S4; es evidencia histórica de compatibilidad, no una matriz del SDK actual ni de validación MLU270. El repositorio documenta rutas PyTorch, ONNX, Caffe y Paddle; en 1.7 se eliminó la compatibilidad con TensorFlow aunque se conservan ejemplos históricos de TensorFlow. El [portal actual para desarrolladores serie 3](https://developer.cambricon.com/index/document/index/classid/3.html) de Cambricon enumera lanzamientos posteriores del SDK general, así que no deben presentarse como una misma versión el repositorio de ejemplos público y el stack comercial actual.

El repositorio oficial de modelos [MagicMind Cloud](https://github.com/Cambricon/magicmind_cloud) publica una matriz MLU370-X4/S4 excepcionalmente directa. Enumera ejemplos YOLOv3, Tiny-YOLOv3, YOLOv4, YOLOv5, YOLOv7, YOLOv8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN, DeepLab y UNet, e indica si hay ejemplos C++ o Python. La evidencia es sólida; la principal barrera es el acceso al SDK y los contenedores mediante los canales de Cambricon.

### Canaan/Kendryte

Los KPU K230/K230D actuales y K210/K510 heredados de Kendryte son importantes en el extremo de placas económicas y cámaras inteligentes. El compilador [nncase](https://github.com/kendryte/nncase) importa ONNX o TFLite, usa datos de calibración para conversión de punto fijo, simula el resultado en un host y genera `.kmodel`.

La [guía oficial de desarrollo nncase K230](https://github.com/kendryte/k230_docs/blob/main/en/01_software/board/ai/K230_nncase_Development_Guide.md) describe la compilación, simulación y ejecución de YOLOv5s. El [catálogo de demostraciones de IA](https://github.com/kendryte/k230_docs/blob/main/en/02_applications/ai_demos/K230_AI_Demo_Introduction.md) separado incluye artefactos YOLOv8n de detección, segmentación y pose, mientras que el [registro de cambios actual de CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) añade ejemplos optimizados YOLOv8/11/26 de clasificación, detección, segmentación, OBB y pose. El [resultado YOLOv5s](https://github.com/kendryte/k230_docs/blob/main/en/00_hardware/K230_datasheet.md) de la ficha técnica es el benchmark publicado real; las versiones del compilador y del complemento binario KPU deben coincidir con el SDK de la placa.

### Allwinner

V853 de Allwinner combina hardware multimedia orientado a cámaras con una NPU Vivante de 1-TOPS. La [guía oficial de NPU en inglés](https://docs.aw-ol.com/v853/en/npu/dev_npu/) describe la importación Acuity/Pegasus, cuantización, verificación y despliegue mediante `viplite`. Las páginas de modelos de Allwinner y la [guía YOLOv5](https://docs.aw-ol.com/v853/npu/npu_yolov5/) mencionan YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet y redes de rostros/personas.

Es compatibilidad real con visión artificial, pero sigue ligada a la cadena Tina Linux/Vivante anterior y a descargas de SDK que dependen de una cuenta. Debe figurar en el mapa de mercado con esa limitación explícita.

## Plataformas coreanas, taiwanesas y chinas poco conocidas

Las listas en inglés suelen pasar de Rockchip a NVIDIA y omitir una segunda categoría de silicio real y documentado. Algunos de estos ecosistemas tienen matrices YOLO actuales más amplias que startups occidentales más conocidas.

### Mobilint

El proveedor coreano de aceleradores Mobilint vende REGULUS de bajo consumo y ARIES MLA100 de mayor rendimiento. Su qb SDK acepta entradas PyTorch, TensorFlow, TFLite, ONNX y orientadas a Keras, cuantiza a INT8 y genera un artefacto compilado `.mxq`. La [matriz pública de visión](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) menciona detección YOLOv3/v5/v7/v8/v9/v10/11/12/26, además de variantes YOLO de segmentación, pose y OBB. La [página ARIES](https://www.mobilint.com/aries/mla100) incluso publica resultados específicos de YOLO11s y YOLO26m. Es una evidencia sólida de integración, aunque el acceso al SDK/hardware sigue siendo comercial.

### Sunplus

SP7350/C3V de Sunplus usa un stack NPU derivado de Vivante, pero lo empaqueta de forma distinta a Allwinner o al Amlogic heredado. Su flujo público combina conversión Acuity, un entorno Docker NPU, runtime SNNF y salida `.nb`. La [guía oficial personalizada de YOLOv8](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) describe la conversión y el despliegue, en vez de limitarse a afirmar compatibilidad con el framework. Compartir IP relacionada no hace que su `.nb` sea portable a otro SoC basado en Vivante.

### ESWIN Computing

La familia RISC-V de ESWIN incluye EIC7700/EIC7700X y EIC7702/EIC7702X de doble chip, con EIC7700 en placas disponibles como Milk-V Megrez. ENNP contiene EsQuant, el compilador EsAAC actual, generación de datos de referencia, simulación y runtime ESSDK, con salida compilada `.model`; el material ENNP anterior usaba el nombre `ennc-compile`. La [página actual de EsAAC](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) documenta entrada ONNX, así que otros frameworks deben exportarse o convertirse a esa IR compatible. El [resumen ENNP alojado en Milk-V](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/introduction) y el [tutorial YOLOv3](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3) establecen una ruta pública integral, pero no una matriz amplia y actual de modelos/precisión.

### Nuvoton M55M1

M55M1 de Nuvoton es un diseño de clase MCU Cortex-M55 más Ethos-U55-256, no un procesador de aplicaciones Linux. NuEdgeWise/NuML y Arm Vela preparan modelos TFLite Micro cuantizados. El repositorio público [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) menciona YOLOv8-nano, YOLOX-nano, YOLO Fastest v1.1, SSD-MobileNet FPNLite y varios clasificadores. Estos ejemplos pequeños son evidencia creíble para la clase de memoria/potencia del dispositivo, no para variantes YOLO estándar de 640 píxeles.

### Rebellions y FuriosaAI

La línea ATOM de Rebellions usa el SDK RBLN y artefactos compilados `.rbln`. Su [lanzamiento oficial de compatibilidad](https://docs.rbln.ai/v0.8.2/supports/release_note.html) menciona YOLOv3 tiny/full/SPP, YOLOv5 n a x, YOLOv6 n a l, variantes YOLOv7 y YOLOv8 n a x. La [matriz actual de compatibilidad de tarjetas](https://docs.rbln.ai/latest/supports/version_matrix.html) marca ATOM CA02 y ATOM+ CA12 como EoL, mientras que ATOM+ CA22 y ATOM-Max CA25 están activos. ATOM-Lite CA21 tiene página de producto, pero no aparece en esa matriz del SDK, así que el estado actual de su cadena de herramientas no está claro. La documentación es pública, pero el wheel del compilador requiere credenciales de Rebellions Portal.

FuriosaAI debe dividirse por generación. Warboy es el acelerador de visión anterior, con artefactos INT8 `.enf` y un [zoo oficial de modelos](https://developer.furiosa.ai/furiosa-models/latest/) que incluye SSD, YOLOv5M/L y YOLOv7-w6-pose. RNGD usa un [stack `.fxb`](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html) distinto, orientado principalmente a cargas LLM/VLM. La [hoja de ruta actual](https://developer.furiosa.ai/latest/en/overview/roadmap.html) de Furiosa registra que la compatibilidad de visión YOLOv8m se completó en 2024 Q4, pero las páginas públicas actuales de modelos compatibles y rendimiento se centran en LLM/VLM y no exponen una matriz detallada de precisión/rendimiento YOLOv8m. Eso evidencia una generación lanzada, no compatibilidad con Warboy.

### Realtek, Telechips, T-Head y SigmaStar

Estas cuatro plataformas son reales, pero sus superficies públicas para integrar modelos propios son más limitadas, antiguas o restringidas:

| Plataforma | Evidencia pública reproducible | Limitación que debe conservarse |
|---|---|---|
| Realtek AmebaPro2 | El [SDK Arduino/FreeRTOS](https://github.com/Ameba-AIoT/ameba-arduino-pro2) empaqueta modelos YOLOv3/4/7-tiny, SCRFD y MobileFaceNet `.nb` | Para convertir modelos propios hay que registrar interés/contactar con Realtek |
| Telechips TOPST TCC7500 | El [proyecto oficial de 2026](https://docs.topst.ai/blog/31) convirtió y desplegó YOLOv8s | La conversión UFLD v1/v2 falló por capas no compatibles; el artículo solo propone una solución de división/postprocesamiento. El acceso completo al compilador/operadores suele requerir autorización por correo |
| T-Head TH1520 | La [guía de aplicaciones](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) de Sipeed ejecuta YOLOv5n/s con HHB y CSI-NN2/SHL | El stack público existe, pero su mantenimiento y claridad de versiones van por detrás de los líderes actuales |
| SigmaStar SSU9383CM | La documentación actual establece las API del runtime MI_IPU; el material antiguo SGS_IPU documenta `.sim` a `sgsimg.img` y un postprocesador anterior menciona SSD y YOLOv1/2/3 | No hay evidencia pública de que los artefactos del compilador anterior o las rutas de modelos nombradas se apliquen a SSU9383CM |

### La visión inteligente HiSilicon no es Huawei Ascend

Los SoC de cámara Hi3516CV610/DV500, Hi3519DV500 y Hi3403V100 de HiSilicon exponen un flujo ATC a `.om` mediante runtimes NNN/SVP-NNN. El [zoo oficial de modelos HiSpark](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) menciona YOLOv3/4/5/6/7/8/9/10/11, segmentación/pose YOLO11, OBB/World/segmentación YOLOv8, OCR, profundidad y TinySAM en una matriz específica de destinos, especialmente para Hi3403 y Hi3591P. No demuestra que cada entrada se ejecute en todos los SoC de visión inteligente indicados arriba. Algunas entradas están en la hoja de ruta, así que deben separarse los ejemplos publicados del soporte previsto; el repositorio indica que los modelos ModelZoo suministrados son solo para uso no comercial. Compartir los términos ATC/`.om` no demuestra compatibilidad de artefactos ni runtime con la línea de productos Ascend basada en CANN.

## Empresas de aceleradores emergentes y especializados

### MemryX

Los módulos MX3 de MemryX usan una arquitectura de flujo de datos y pueden combinar varios chips. El [Neural Compiler](https://developer.memryx.com/tools/neural_compiler.html) acepta modelos ONNX, TensorFlow Lite, Keras y TensorFlow, y genera un paquete DFP que consume el runtime del acelerador. Sus [API de runtime](https://developer.memryx.com/api/accelerator/accelerator.html) admiten canalizaciones mult modelo y de streaming.

La documentación y los ejemplos de MemryX incluyen preprocesamiento y postprocesamiento YOLO optimizados para detección, segmentación y pose. Las [notas de versión del SDK](https://developer.memryx.com/release_notes.html) mencionan expresamente compatibilidad con YOLOv10, YOLO11 y YOLO26. El material público es útil desde el punto de vista técnico, aunque no presenta una tabla sencilla de modelos/precisión/dispositivo comparable a la de Axelera.

### Kneron

Los productos KL520/KL530/KL630/KL720/KL730 de Kneron abarcan aceleradores USB y embebidos pequeños. Kneron PLUS cubre el runtime de aplicación, mientras que Model Toolchain 0.33.1 actual documenta conversión, cuantización, evaluación, simulación y compilación `.nef` para esos destinos. KL830 aparece en algunas [API de runtime PLUS](https://doc.kneron.com/docs/plus_c/introduction/run_examples/), pero no en la [lista actual de destinos del compilador](https://doc.kneron.com/docs/toolchain/manual_1_overview/); no debe extenderse a KL830 una afirmación genérica de compilación `.nef` sin confirmación del proveedor.

El [ejemplo YOLO oficial](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) demuestra el proceso con Tiny-YOLOv3, y otros materiales Kneron cubren entrenamiento y despliegue de YOLOv5. La evidencia es creíble, pero más limitada que las matrices YOLO modernas amplias de Hailo, Axelera, Rockchip o DEEPX.

### SiMa.ai

MLSoC de SiMa.ai y la [plataforma Modalix de 50-TOPS en producción](https://sima.ai/press-release/sima-ai-next-gen-platform-for-physical-ai-in-production/) usan el entorno de software Palette. ModelSDK, el compilador MLA y ModelExecutor cubren importación, cuantización, compilación, profiling y ejecución. Según la carga y el producto, las herramientas admiten flujos orientados a INT8, INT16 y BF16.

Las notas de lanzamiento de la empresa mencionan canalizaciones validadas YOLOv7, YOLOv8 para detección/pose/segmentación, YOLOX y segmentación YOLOX, DETR, Mask R-CNN y EfficientDet. No debe ocultarse una limitación actual: las [notas Palette SDK 2.1](https://docs.sima.ai/v2.1.2/pages/release_notes/2.1.html) indican que QAT no funciona debido a una regresión de cuantización Python/PT2E. La solución documentada es crear el ONNX anotado en SDK 2.0 y compilarlo con 2.1. El acceso y la compra siguen orientados a empresas.

También está documentado el límite de despliegue: el [flujo de compilación ModelSDK](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) genera un `.tar.gz` compilado y puede producir un ELF ejecutable, mientras que [MPK Tool](https://docs.sima.ai/v2.1.1/pages/palette/mpk_tools.html) empaqueta un `.mpk` desplegable. Estas salidas siguen ligadas al destino MLSoC/Modalix y a la versión Palette.

### EdgeCortix

SAKURA-II es un acelerador anunciado de 60-TOPS para visión e IA generativa, compilado mediante el framework de software MERA. EdgeCortix también proporciona tecnología MERA al flujo RUHMI más reciente de Renesas.

El [material de producto SAKURA-II](https://www.edgecortix.com/en/edgecortix-sakura-ii-accelerator-brief) establece el hardware y el compilador, y el [hardware M.2/PCIe](https://www.edgecortix.com/en/hardware) se ofrece mediante solicitud de prueba o pedido. No se encontró una matriz pública actual de compatibilidad YOLO lo bastante precisa. Esa ausencia es información de compra útil: hay que solicitar validación del modelo antes de comprometerse con el hardware.

### BrainChip

Akida de BrainChip es un procesador neuromórfico de dominio de eventos, no una NPU convencional de tensores densos. Su [entorno MetaTF](https://brainchip.com/metatf-dev-tools/) consta de paquetes Python instalables para creación de modelos, cuantización de pocos bits, conversión, simulación y ejecución en hardware. BrainChip vende ahora un [coprocesador AKD1500 disponible en formato M.2](https://brainchip.com/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/), junto con hardware AKD1000 heredado e IP Akida 2.

La [ficha actual de modelos Akida 2](https://brainchip.com/wp-content/uploads/2025/04/Akida-2-Model-Card-V1.1-Mar.25.pdf) menciona un detector YOLOv2 AkidaNet0.5, CenterNet, AkidaUNet y redes de reconocimiento facial. Otro material de compatibilidad incluye FOMO y cargas basadas en eventos. Akida admite pesos y activaciones de muy pocos bits, pero para desplegarlo correctamente puede ser necesaria una conversión consciente de la arquitectura, en vez de tratarlo como otro destino ONNX INT8 genérico. Un resultado del zoo de modelos AKD1000 o Akida 2 no debe atribuirse a AKD1500 hasta que un informe explícito de ajuste/compatibilidad lo valide.

### Blaize

Blaize vende productos Pathfinder y Xplorer basados en P1600, con Picasso SDK, NetDeploy y AI Studio como ruta de software. Las [páginas de producto](https://www.blaize.com/products/) tienen claramente como objetivo la visión y Edge AI, pero no hay una matriz de compatibilidad modelo por modelo actual y pública.

Por tanto, esta guía clasifica Blaize como comercial y restringido, en vez de llenar el vacío con afirmaciones comunitarias. Debería exigirse como requisito un informe de compilación y precisión del proveedor para el modelo previsto.

## TinyML y visión en dispositivos endpoint

### Arm Ethos-U y Alif

Arm licencia IP de NPU Ethos-U55, U65 y U85 a empresas de chips. El compilador [Vela](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) toma grafos LiteRT/TFLite cuantizados y reescribe subgrafos compatibles en operaciones personalizadas Ethos-U. Los operadores no compatibles pueden permanecer en CPU, así que «la aplicación se ejecuta» y «toda la red se ejecuta en la NPU» son afirmaciones distintas.

Las implementaciones reales incluyen Ethos-U55 en Alif Ensemble E7, Ethos-U65 en NXP i.MX 93 y Ethos-U85 en sistemas más recientes. El [E7 AI/ML AppKit](https://alifsemi.com/support/kits/ensemble-e7appkit/) de Alif documenta dos aceleradores ML y demuestra la ruta TFLite a Vela de E7; [Ensemble E8](https://alifsemi.com/ensemble-e8-series/) combina un Ethos-U85 y dos NPU Ethos-U55. Alif también publica un [benchmark INT8 para detector facial YOLO-Fastest](https://alifsemi.com/faster-ai-mcu-inferencing-low-power-consumption/) a nivel de familia, a 192 por 192 en Cortex-M55 más Ethos-U55, pero no una matriz amplia de YOLO moderno por destino. Estos sistemas con memoria restringida encajan con redes pequeñas de clasificación y detección, no con cualquier detector de 640 píxeles solo porque pueda expresarse en TFLite.

### Infineon PSOC Edge, Himax WiseEye2 y Analog Devices MAX7800x

Las familias [PSOC Edge E83](https://documentation.infineon.com/psocedge/docs/xsk1761304020519) y E84 de Infineon combinan Cortex-M55 con Ethos-U55 y añaden un bloque NNLite separado en el lado M33 de bajo consumo. El [manual de arquitectura E84](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) documenta pesos de 8 bits, activaciones de 8 o 16 bits y un flujo Vela donde los operadores compatibles se convierten en un flujo de comandos de NPU y el trabajo no compatible permanece en CPU. [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter) acepta rutas TFLite, Keras y PyTorch, mientras que la documentación de arquitectura menciona MobileNetV1/V2 como kernels representativos, no como benchmarks de destino. El [E84 AI Kit](https://documentation.infineon.com/psocedge/docs/cci1762693051052) incluye cámara y usa ModusToolbox, pero el material público aún no ofrece una matriz de compatibilidad YOLO por modelo. Es una plataforma de visión programable real cuya evidencia por modelo está madurando, no un destino de detección general validado.

[HX6538 WiseEye2](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) de Himax también combina Cortex-M55 y Ethos-U55 para visión siempre activa. Su oferta pública de software es excepcionalmente concreta para esta clase de potencia: los ejemplos oficiales de [Grove Vision AI Module V2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) incluyen detección de objetos YOLOv8n, pose y clasificación de género, detección YOLO11n, face mesh y PeopleNet. La ruta unificada usa TFLite Micro y Vela, y luego CMSIS-NN y kernels de referencia como alternativa cuando es necesario. Son modelos y resoluciones deliberadamente pequeños, no evidencia de que quepa un grafo YOLO convencional de tamaño servidor.

Analog Devices sigue una ruta distinta con los [aceleradores CNN MAX78000/MAX78002](https://www.analog.com/en/products/max78002.html). La herramienta pública [`ai8x-synthesis`](https://github.com/analogdevicesinc/ai8x-synthesis) cuantiza una red entrenada y genera código C, pesos y configuración del acelerador específicos del destino, en lugar de un paquete runtime ONNX portable. La evidencia de primera parte incluye [identificación facial](https://www.analog.com/en/resources/app-notes/an-2616.html), detección de códigos QR TinyERSSD en el [zoo de modelos ADI](https://ez.analog.com/dsp/software-and-development-tools/edgebench/a/docs-faqs/DF912/available-models-in-adi-model-zoo) y clasificadores de imágenes. Los dispositivos son relevantes para visión en endpoints, pero no se encontró una matriz oficial actual de YOLO moderno.

### GreenWaves GAP9

GAP9 combina cómputo RISC-V con el motor neuronal NE16. NNTool importa y cuantiza redes, mientras que AutoTiler y GAP SDK generan código desplegable. El [menú de redes neuronales GAP9](https://github.com/GreenWaves-Technologies/nn_menu_gap9) público contiene aplicaciones MobileNet, EfficientNet, ResNet, MobileNet SSD, rostro y reconocimiento. GreenWaves también publica un [proyecto de detección de personas YOLOX](https://github.com/GreenWaves-Technologies/yolox_people_detection) con resultados de precisión y simulación.

Es evidencia TinyML inusualmente transparente, pero el SDK completo solo está disponible para clientes cualificados y los modelos son considerablemente más pequeños que las variantes YOLO convencionales de 640 por 640.

### Lattice sensAI

Lattice sensAI es una alternativa FPGA a una NPU fija. sensAI Studio y Neural Network Compiler mapean un grafo compatible a IP aceleradora configurable para dispositivos ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E y Avant-X. La salida está ligada a la FPGA seleccionada, el modo acelerador, el layout de memoria y el bitstream, no es un modelo runtime portable.

El [manual actual de Neural Network Compiler](https://www.latticesemi.com/-/media/LatticeSemi/Documents/UserManuals/MQ3/FPGA-UG-02052-8-0-Lattice-Neural-Network-Compiler-Software.ashx?document_id=52343) incluye una tabla topológica excepcionalmente directa. Enumera YOLOv1 para ECP5, YOLOv5 y YOLOv8 para modos optimizado/avanzado en CrossLink-NX y CertusPro-NX, y YOLO11 solo con IP en modo Advanced. También incluye SSD, MobileNetV2-SSD, ResNet, ENet, SqueezeDet y otras redes de visión, con las combinaciones no compatibles visibles por familia FPGA. El [Advanced CNN Accelerator](https://www.latticesemi.com/en/Products/DesignSoftwareAndIP/IntellectualProperty/IPCore/IPCores05/Advanced-CNN-Accelerator-IP) está dirigido a Avant-E/Avant-X/CertusPro-NX y es IP comercial. Esta es evidencia de la tabla del compilador, no una promesa de que un bitstream admita simultáneamente todas las topologías listadas.

### Microchip VectorBlox

El [VectorBlox SDK 3.1](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) público preprocesa y compila redes TFLite INT8 totalmente cuantizadas para IP CoreVectorBlox configurable en FPGA PolarFire SoC. `tflite_preprocess` y `vnnx_compile` generan recursos de despliegue `.vnnx`, `.hex` y `.ucomp`, y el repositorio incluye un simulador y un controlador C para el destino. Los grafos TensorFlow, ONNX y OpenVINO deben pasar antes por la ruta documentada de preparación TFLite/INT8.

La [matriz actual de tutoriales](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) incluye YOLOv5n, detección, clasificación, OBB, pose y segmentación YOLOv8n, YOLOv9t, variantes YOLO dispersas/comprimidas, MobileNet, EfficientNet-Lite0, ResNet18, MiDaS, FFNet y QuickSRNet. La [guía de postprocesamiento C](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/C_Postprocessing.md) menciona por separado YOLOv2/3/4/5 y decodificadores Ultralytics para detección, pose y OBB. SDK 3.1 actualmente es compatible con PolarFire SoC Video Kit y explícitamente no con el PolarFire Video Kit sin SoC, así que no deben confundirse las demostraciones antiguas de placas con el contrato de destino actual.

El SDK se puede descargar públicamente bajo una [licencia específica de Microchip](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/LICENSE.md) que restringe el software y sus derivados a productos Microchip. Microchip ofrece licencias Libero Silver y CoreVectorBlox gratuitas, pero sujetas a solicitud. El despliegue sigue dependiendo del sistema FPGA: la configuración del acelerador, el bitstream, el firmware, los datos de red generados por SDK y el layout de memoria deben coincidir. Una compilación VectorBlox correcta no convierte el modelo en un binario que se pueda copiar a cualquier diseño PolarFire.

### Syntiant

NDP200 de Syntiant está dirigido a inferencia siempre activa de visión, audio y sensores, por debajo del consumo de los SoC Linux normales. La [tabla de hardware actual](https://www.syntiant.com/hardware) indica que NDP200 está en producción masiva y NDP250 en fase de muestras. La [ficha NDP200](https://www.syntiant.com/ndp200/) documenta compatibilidad con redes CNN, RNN y totalmente conectadas por debajo de 1 mW; el [anuncio NDP250](https://www.syntiant.com/news/syntiant-unveils-ndp250-neural-decision-processor-with-next-gen-core-3-architecture) describe, en cambio, reconocimiento de imágenes siempre activo por debajo de 30 mW. «Menos de un milivatio» no debe generalizarse a ambos componentes.

El material público no establece compatibilidad amplia con YOLO, así que el producto debería evaluarse para clasificadores y detectores pequeños siempre activos mediante una evaluación de modelos respaldada por el proveedor, no suponiendo una exportación ONNX genérica.

### Google Coral: dos generaciones sin relación directa

Los productos heredados Edge TPU usan Edge TPU Compiler, `libedgetpu` y PyCoral con modelos TFLite totalmente INT8. Sus repositorios principales [Edge TPU](https://github.com/google-coral/edgetpu) y [PyCoral](https://github.com/google-coral/pycoral) están archivados. Es un riesgo real de mantenimiento, pero no un aviso formal de fin de vida del hardware.

Google también mantiene una [Coral NPU](https://github.com/google-coral/coralnpu) separada, activa y de código abierto: IP aceleradora RISC-V para integrarla en SoC de bajo consumo. El proyecto público ofrece actualmente RTL, simulación de hardware, una cadena de herramientas y ejemplos ELF. No es una sucesora USB/M.2 directa de Edge TPU y no publica una matriz actual de despliegue YOLO.

## Trampas de GPU adyacentes, NPU cliente y cómputo adaptativo

NVIDIA, AMD, Intel y Apple suelen aparecer en búsquedas de NPU, pero no exponen una clase intercambiable de acelerador.

**NVIDIA Jetson:** Jetson Orin combina una GPU CUDA con núcleos DLA dedicados. TensorRT puede compilar un motor serializado para DLA, pero las capas no compatibles se ejecutan en GPU solo si se habilita explícitamente la alternativa de GPU. La [tabla actual YOLO DeepStream](https://github.com/NVIDIA-AI-IOT/deepstream_tools/blob/main/yolo_deepstream/README.md) cubre YOLOv4/v7/v8/v9/11, pero la mayoría de las entradas son destinos GPU; su entrada explícita ONNX DLA es YOLOv8s INT8. Las [restricciones DLA](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html) de NVIDIA explican los límites de operadores. Thor marca otra diferencia: la [guía de migración DriveOS](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/NVIDIA_DriveOS_7.0.3_Migration_Guide.pdf) de NVIDIA indica que se eliminaron los núcleos DLA de Thor en favor de flexibilidad para planificar en GPU. Por tanto, «se ejecuta en Jetson» no identifica el acelerador.

**AMD Vitis AI:** La antigua generación Kria/DPU compilaba grafos `.xmodel` ejecutados mediante VART. La versión actual [Vitis AI 6.2](https://vitisai.docs.amd.com/en/6.2/) es GA para dos rutas independientes Versal AI Edge: [Gen1](https://vitisai.docs.amd.com/projects/gen1/en/latest/index.html) en VEK280/VE2802 usa ONNX, AMD Quark y un flujo snapshot/subgrafo ligado al destino; [Gen2](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html) en VEK385 tiene su propio contrato de compilación y runtime. Gen1 publica ejemplos YOLOv5, YOLOv7, YOLOv8 y YOLOX. Ryzen AI es un cuarto stack de NPU cliente independiente. No traslades artefactos ni validaciones entre DPU heredada, Versal Gen1, Versal Gen2 y Ryzen AI solo porque todos usen el nombre Vitis o AMD.

**Intel OpenVINO:** OpenVINO puede dirigirse a CPU, GPU y NPU Core Ultra con una API. La [documentación del plugin Intel NPU](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) identifica generaciones de NPU compatibles, mientras que la [matriz de modelos verificados](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html) expone columnas específicas del dispositivo. Un notebook YOLO demuestra una receta de aplicación, no validación NPU salvo que lo indique la matriz. Los operadores, las formas, la precisión y el controlador NPU instalado determinan si el grafo completo se ejecuta allí. OpenVINO IR (`.xml` más `.bin`) es una IR fuente reutilizable; una caché de modelo compilado depende del dispositivo y del software.

**Apple Core ML:** `coremltools` convierte modelos a `.mlpackage`, que Xcode compila como `.mlmodelc`. Core ML puede planificar trabajo entre CPU, GPU y Apple Neural Engine, pero abstrae deliberadamente el particionado exacto. Eso convierte Apple en una plataforma de despliegue excelente y en una mala candidata para afirmar que «todo el grafo YOLO se ejecutó en la NPU» sin que lo demuestre el profiling.

## Las empresas de IP NPU que están detrás de las empresas de chips

Algunas empresas no venden placas ni aceleradores empaquetados. Licencian diseños NPU a fabricantes de SoC. Importan porque una misma arquitectura subyacente puede aparecer bajo varias marcas de chips, aunque el SDK y el artefacto finales sigan personalizados por el licenciatario.

| Proveedor de IP | Familia NPU y software | Por qué importa |
|---|---|---|
| [Arm](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) | Ethos-U55/U65/U85 y Vela | Aparece en productos NXP, Alif, Infineon, Himax y otros; flujo orientado a TFLite/TOSA cuantizado |
| [Arm China](https://www.armchina.com/mountain?infoId=161&name=) | AIPU Zhouyi y Compass SDK | El [model zoo](https://github.com/Arm-China/Model_zoo) público menciona YOLOv1-tiny/v2/v3/v4/v5, YOLOX y YOLOv8-seg; son modelos de referencia, no evidencia para todos los chips licenciados |
| [VeriSilicon](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico) | Familia Vivante VIP y SDK Acuity | Hay familias NPU Vivante relacionadas en varios SoC, incluidas las rutas A311D e i.MX 8M Plus con fuentes distintas; cada fabricante de chip sigue ofreciendo su propia integración |
| [Imagination Technologies](https://www.imaginationtech.com/products/open-access/) | PowerVR Series3NX AX3146/AX3386/AX3596; IMG Series4 histórico; Neural Compute SDK/IMG DNN | IP NNA licenciable, no placa de venta minorista; Open Access ofrece configuraciones Series3NX de 1/5/10-TOPS, mientras que un programa oficial NC-SDK menciona PP-YOLOE, EfficientNet y HRNet, pero no todas las configuraciones IP |
| [CEVA](https://www.ceva-ip.com/product/ceva-neupro-studio/) | NeuPro-Nano/NeuPro-M y NeuPro Studio | IP NPU licenciable con importación, cuantización, compresión, compilación de grafo, simulación y generación C/C++ |
| [MIPS](https://mips.com/processor-solutions/arc-npx-family/) | ARC NPX6 y [MetaWare MX](https://mips.com/processor-solutions/arc-metaware-mx/), antes Synopsys ARC | IP NPU escalable y NN SDK adquiridos por GlobalFoundries e incorporados a la cartera de MIPS en [junio de 2026](https://mips.com/press-releases/gfmipsarcclose/); no es un destino de despliegue de venta minorista |
| [Cadence](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/neuroweave-sdk.html) | NPU Neo, DSP Tensilica y SDK NeuroWeave | Stack de compilador/intérprete usado por licenciatarios de SoC, con compatibilidad de red y cuantización ligada a configuraciones licenciadas |
| [Quadric](https://quadric.ai/npu-ip) | IP GPNPU Chimera y SDK | IP programable de estilo NPU/DSP licenciada a chips de otras empresas; el destino final es el silicio del licenciatario |
| [Expedera](https://www.expedera.com/products-overview/) | IP NPU Origin y stack de software | Arquitectura NPU edge licenciable, no una placa LibreYOLO que se pueda comprar directamente |

Imagination ilustra por qué las etiquetas de acceso y ciclo de vida deben acompañar a los nombres de arquitectura. Su programa [Open Access](https://www.imaginationtech.com/products/open-access/) ofrece tres configuraciones Series3NX probadas en silicio sin cuota inicial de licencia IP, pero solo después de evaluar la empresa y con soporte/mantenimiento de pago y royalties de producción. El [Neural Compute SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) proporciona herramientas de compilación, optimización, cuantización y runtime; una [colaboración oficial con Baidu PaddlePaddle](https://www.imaginationtech.com/news/imagination-and-baidu-paddlepaddle-create-open-source-machine-learning-library-for-model-zoo/) menciona detección PP-YOLOE, clasificación EfficientNet y segmentación HRNet. Esos ejemplos no validan cada configuración Series3NX o Series4, y las páginas individuales actuales de [productos Series4](https://www.imaginationtech.com/product/img-4nx-mc1/) indican que no están disponibles, así que Series4 debe considerarse histórica salvo que ventas confirme lo contrario.

Esta capa explica parecidos aparentes entre familias. No hace portables los artefactos. Dos SoC con IP Vivante o Ethos relacionada pueden tener mapas de memoria, controladores, versiones de compilador, conjuntos de operadores y runtimes del proveedor distintos.

## Lo que se despliega realmente: tabla de dependencia del artefacto

El último archivo de la canalización del compilador es donde termina la aparente portabilidad de ONNX. Los nombres y las extensiones cambian con el tiempo, pero la regla práctica se mantiene: conserva el modelo original y los datos de calibración porque, por lo general, el artefacto nativo no se puede trasladar a otra generación de NPU ni reconstruir de forma fiable con otra versión del SDK.

| Stack del proveedor | Entrada habitual del compilador | Qué llega al dispositivo | A qué queda ligado |
|---|---|---|---|
| [Amlogic AMLNN](https://github.com/Amlogic-NN/amlnn-toolkit) | ONNX, TFLite, TorchScript o PT2 | `.adla`; A311D heredado usa `.nb` | ID de destino, generación ADLA, compilación del compilador AMLNN, NNSDK2/runtime, controlador ADLA y BSP |
| [Rockchip RKNN](https://github.com/airockchip/rknn-toolkit2) | ONNX, TFLite o exportación de framework | `.rknn` | Versión de RKNN toolkit/runtime y familia NPU Rockchip |
| [Hailo](https://hailo.ai/developer-zone/documentation/) | ONNX o TensorFlow mediante un HAR intermedio | `.hef` | Arquitectura Hailo y generación de compilador/runtime |
| [Axelera Voyager](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) | ONNX o definición compatible del zoo | `.axm` en Pipeline Builder alfa, `.axe` como paquete de canalización; `.axmodel` clásico más manifiesto | Generación de API Voyager y configuración del destino Metis |
| [DEEPX DXNN](https://github.com/DEEPX-AI/dx-all-suite) | ONNX y rutas de frameworks compatibles | `.dxnn` | Versión DX-COM/DX-RT y destino DX-M |
| [Qualcomm QAIRT/QNN o SNPE](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | ONNX, TFLite o modelo de framework | Biblioteca de modelo/contexto binario QNN, o `.dlc` de SNPE | Arquitectura HTP, SoC, backend y versión del runtime |
| [MediaTek NeuroPilot](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) | TFLite convertido/cuantizado | `.dla` para Neuron Runtime offline; `.tflite` para modo delegado | Generación NP/MDLA, imagen del sistema operativo, compilador y runtime |
| [TI TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools) | ONNX o TFLite | Directorio de artefactos importados más archivos de modelo de aplicación | Herramientas/runtime TIDL y generación del procesador |
| [NXP eIQ](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | Normalmente TFLite u ONNX | Salida Vela, TIM-VX, Neutron o Ara específica del backend | El acelerador i.MX/Ara seleccionado y el BSP, no «NXP» de forma genérica |
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | TFLite, ONNX o modelo de framework compatible | Biblioteca/código de red y recursos de firmware generados | Destino MCU/NPU, configuración de memoria y versión de herramienta |
| [Infineon PSOC Edge](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) | Ruta TFLite/Keras/PyTorch cuantizada a enteros | TFLite optimizado por Vela con flujo de comandos Ethos-U más firmware de aplicación | Variante E83/E84, configuración Ethos-U, Vela, ModusToolbox y BSP; NNLite es un destino separado |
| [Renesas DRP-AI](https://github.com/renesas-rz/rzv_drp-ai_tvm) | ONNX/TFLite mediante TVM o Translator | Directorio de modelo de runtime con varios archivos de datos binarios | Dispositivo RZ/V y generación del traductor/runtime |
| [Sony IMX500](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | Red cuantizada/empaquetada del flujo Edge-MDT | Paquete `.rpk` | Firmware del sensor, presupuesto de memoria y versión del empaquetador |
| [Synaptics](https://developer.synaptics.com/) | TFLite/ONNX e importaciones compatibles | `.synap` en SyNAP; `.vmfb` en Torq | Generación de software/hardware SL16xx frente a SL261x |
| [Ambarella CVflow](https://www.ambarella.com/developer/) | Entrada de cadena de herramientas para socios | Paquete de despliegue solo para socios; contrato exacto del artefacto no documentado públicamente | Confirma la generación CVflow, el SDK/BSP y la canalización de cámara mediante Cooper Developer Zone |
| [D-Robotics BPU](https://github.com/D-Robotics/rdk_model_zoo) | ONNX y grafo compatible con la cadena de herramientas | X5 `.bin`; `rdk_s` actual `.hbm`; X3 usa un flujo de plataforma anterior | Generación BPU, rama del repositorio y versión correspondiente de OpenExplorer/runtime |
| [AXERA Pulsar2](https://github.com/AXERA-TECH/ax-samples) | ONNX | `.axmodel` | Destino de chip AX y versión de Pulsar2 y AXEngine |
| [SOPHGO TPU-MLIR](https://github.com/sophgo/tpu-mlir) | ONNX, TFLite, TorchScript y otros | `.bmodel` en BM, `.cvimodel` en CV18xx | Destino de chip BM/CV y generación del runtime SOPHON |
| [Huawei Ascend CANN](https://www.hiascend.com/en/software/cann) | ONNX o grafo de framework compatible | `.om` | Chip Ascend, CANN/ATC y firmware/controlador del dispositivo |
| [Cambricon MagicMind](https://github.com/Cambricon/magicmind_cloud) | Grafo de framework o modelo de intercambio | Motor/modelo MagicMind serializado | Arquitectura MLU y versión Neuware/MagicMind |
| [Canaan nncase](https://github.com/kendryte/nncase) | ONNX o TFLite | `.kmodel` | Generación KPU y complemento de destino nncase correspondiente |
| [Mobilint qb](https://www.mobilint.com/sdk-qb) | ONNX y modelos de frameworks compatibles | `.mxq` | Destino REGULUS/ARIES y compilador/runtime qb |
| [Rebellions RBLN](https://docs.rbln.ai/latest/index.html) | PyTorch 2 o grafo compatible con TensorFlow | `.rbln` | Generación ATOM y compilador/runtime RBLN |
| [FuriosaAI](https://developer.furiosa.ai/docs/latest/en/) | ONNX/TFLite y ruta compatible con framework | Warboy `.enf`; RNGD `.fxb` | Generaciones completamente distintas de acelerador y SDK |
| [Sunplus SNNF](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2004353133) | Red compatible con Acuity | `.nb` | Controlador NPU SP7350, runtime y BSP |
| [ESWIN ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) | ONNX en el flujo EsAAC actual documentado; otros frameworks requieren exportación/conversión | `.model` | Destino exacto de la familia EIC7700 y versión ENNP/ESSDK |
| [Nuvoton / Ethos-U](https://github.com/OpenNuvoton/NuEdgeWise) | TFLite cuantizado | TFLite optimizado por Vela con operaciones personalizadas Ethos-U | Plan de memoria M55M1, Vela y firmware |
| [Himax WiseEye2](https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2) | TFLite cuantizado completamente a enteros | `.tflite` optimizado por Vela en la memoria del modelo más firmware de placa como `output.img` | Configuración HX6538/WE2, Vela, TFLite Micro, controlador Ethos-U y kernels alternativos |
| [Analog Devices AI8X](https://github.com/analogdevicesinc/ai8x-synthesis) | Checkpoint PyTorch, YAML de red y entrada de muestra | Código fuente/encabezados C, pesos y firmware compilado específicos del dispositivo | Límites de memoria/capas MAX78000 frente a MAX78002, herramienta de síntesis y versión MSDK |
| [Realtek AmebaPro2](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Servicio/entrada de conversión del proveedor | `.nb` | Firmware RTL8735B y versión de API VoE/NeuralNetwork |
| [Telechips Enlight](https://docs.topst.ai/product/p/ai) | Ruta Darknet, TensorFlow, ONNX o PyTorch | `.enlight` intermedio más paquete compilado de despliegue | NPU TCC7500 y versiones TC-NN/Enlight |
| [T-Head HHB](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | ONNX y grafo de framework compatible | `hhb.bm`, parámetros/código generados y ejecutable | Stack CSI-NN2/SHL de TH1520 |
| [SigmaStar MI_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | La documentación actual SSU9383CM expone runtime MI_IPU; el material anterior usa entradas de familias TensorFlow/Caffe | `.sim` anterior convertido a `sgsimg.img`; artefacto público actual sin verificar | Generación exacta de IPU/SDK SigmaStar; compatibilidad del compilador heredado con SSU9383CM no verificada |
| [HiSilicon smart vision](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) | ONNX y grafo compatible con ATC | `.om` | SoC de visión inteligente y stack NNN/SVP-NNN exactos; no portabilidad Ascend genérica |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | ONNX, TFLite, Keras o TensorFlow | Paquete de flujo de datos `.dfp` | Configuración de hardware MX y versión del runtime/compilador |
| [Kneron](https://doc.kneron.com/docs/) | ONNX más configuración de la cadena de herramientas | `.nef`; KL730 NEFv2 puede contener `.kne` | Destino de compilador documentado, generación del chip, firmware y runtime PLUS; Toolchain 0.33.1 no establece la compilación KL830 |
| [SiMa.ai Palette](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) | Modelo de framework u ONNX compatible | `.tar.gz` compilado ModelSDK, ELF ejecutable documentado o paquete `.mpk` de MPK Tool | Destino MLSoC/Modalix y versión Palette |
| [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | ONNX o definición de red | Motor serializado, normalmente `.engine` o `.plan` | Arquitectura GPU/DLA, TensorRT, CUDA y a menudo el dispositivo |
| [AMD Vitis AI](https://vitisai.docs.amd.com/en/6.2/) | Grafo ONNX/framework cuantizado | `.xmodel` heredado; snapshots/subgrafos NPU Gen1; directorio de caché compilada o paquete `.rai` de producción Gen2 | Generación/configuración IP exacta de NPU, imagen de plataforma y matriz Vitis/Vivado/PetaLinux |
| [Microchip VectorBlox](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) | TFLite INT8 totalmente cuantizado | `.vnnx`, `.hex` y `.ucomp` más firmware/diseño FPGA correspondiente | Configuración CoreVectorBlox, PolarFire SoC Video Kit, SDK 3.1, bitstream y layout de memoria |
| [Intel OpenVINO](https://docs.openvino.ai/) | Modelo de framework u ONNX | IR `.xml` más `.bin`, o caché compilada específica del dispositivo | IR reutilizable; la caché compilada es específica del dispositivo, controlador y OpenVINO |
| [Google Edge TPU heredado](https://coral.ai/docs/edgetpu/models-intro/) | TFLite totalmente INT8 | `_edgetpu.tflite` compilado para Edge TPU | Compilador/runtime Edge TPU y operadores cuantizados compatibles; sin relación con la nueva IP Coral NPU |
| [Google Coral NPU IP](https://github.com/google-coral/coralnpu) | Ejemplos C/C++ del proyecto IP público actual | Ejemplos ELF para simulación RTL/hardware e integración futura en SoC; sin artefacto público de compilador YOLO | Implementación exacta de SoC licenciada/integrada y cadena de herramientas de código abierto en evolución |
| [Lattice sensAI](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | Descripción de red compatible | Bitstream FPGA, configuración del acelerador y pesos | Familia FPGA, modo IP sensAI e implementación de memoria |
| [Imagination NC-SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) | Caffe, TensorFlow u ONNX | El nombre de archivo del artefacto compilado para clientes no se especifica públicamente | Configuración NNA licenciada, integración SoC, NC-SDK/DDK y runtime del licenciatario |

Esta tabla no es solo una chuleta de extensiones. La versión del compilador, el identificador de destino, el controlador, el firmware y el BSP forman parte de la identidad reproducible del modelo. Un futuro exportador LibreYOLO debería escribirlos en un manifiesto legible por máquina junto a cada artefacto.
## Acceso a las herramientas y señales de ciclo de vida

La disponibilidad del hardware y el acceso al compilador son independientes. Puede ser fácil comprar una placa mientras que su compilador actual exige un acuerdo de cliente; un SDK público puede estar dirigido a silicio difícil de conseguir. Las siguientes son señales concretas de primera parte, no una clasificación universal de longevidad.

| Plataforma | Señal documentada | Qué significa en la práctica |
|---|---|---|
| Amlogic ADLA | Repositorios Apache-2.0 públicos y wheels descargables; los controladores/BSP de placa compatibles aún pueden requerir Amlogic o un proveedor de placas | Es fácil evaluar el compilador, pero deben confirmarse la redistribución de binarios y la matriz completa de compatibilidad |
| MediaTek Genio | IoT AI Hub y guías Yocto públicas; las herramientas NP8 todo en uno y los documentos Android se marcan para clientes directos/NDA | La evidencia de modelos se puede auditar, mientras que automatizar modelos propios puede requerir una asociación |
| Hailo | Model zoo y runtime públicos; Dataflow Compiler distribuido mediante Developer Zone; `hailo-camera-apps` público archivado el 3 de mayo de 2026 | El descubrimiento amplio de modelos es abierto, pero la compilación reproducible requiere la cuenta/versión correcta; el estado de archivo no demuestra EoL de Hailo-15 y debe confirmarse su paquete de aplicaciones actual para procesador de visión |
| Axelera AI | Documentación pública y [Voyager SDK público](https://github.com/axelera-ai-hub/voyager-sdk); el soporte a clientes requiere cuenta | SDK de acelerador autoservicio poco habitual, aunque el soporte de producto y la compra siguen siendo comerciales |
| Qualcomm Dragonwing | El [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) indica IQ-9075 Sampling con longevidad hasta 2038 y QCS6490 hasta julio de 2036, mientras que la página IQ-9075 indica Active | Buena señal para planificación industrial, pero el conflicto de estado entre fuentes de primera parte requiere confirmación por SKU y no garantiza que un SDK IA conserve la estabilidad ABI durante todo el periodo |
| Rebellions | La [matriz actual de compatibilidad](https://docs.rbln.ai/latest/supports/version_matrix.html) marca CA02 y CA12 EoL, y CA22/CA25 Active; CA21 no aparece | La compra y compatibilidad del SDK dependen de cada tarjeta incluso dentro de la familia ATOM |
| NVIDIA Jetson | La [tabla oficial de ciclo de vida de módulos](https://developer.nvidia.com/embedded/lifecycle) enumera módulos Orin hasta enero de 2032 y AGX Orin Industrial hasta julio de 2033; los kits de desarrollo no tienen compromiso de ciclo de vida | Diseña sistemas de producción en torno a módulos, no a la disponibilidad de kits de desarrollo |
| Sony IMX500 en Raspberry Pi | La [ficha de producto AI Camera](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf) indica producción al menos hasta enero de 2028 | Mínimo concreto para ese módulo de cámara, no para todos los productos IMX500 |
| Amlogic A311Y3 | La [página de lanzamiento de 2026](https://www.amlogic.com/News/index248.html) anuncia una garantía mínima de longevidad del producto de diez años | Señal prometedora de una nueva plataforma, que aún requiere detalles contractuales y del SKU para un programa de producto |
| Google Coral | Los repositorios Edge TPU y PyCoral heredados están archivados/en solo lectura; el repositorio [Coral NPU IP](https://github.com/google-coral/coralnpu) separado está activo | Riesgo de mantenimiento del software heredado, no por sí mismo un aviso formal de EoL del hardware; no dice nada sobre el ciclo de vida del proyecto IP de código abierto sin relación directa |

## Por qué TOPS no es una guía de compra

Los TOPS máximos pueden servir dentro de un proveedor y una arquitectura, pero las comparaciones entre proveedores suelen ser inválidas salvo que coincidan todos estos factores:

- Precisión numérica, incluidas INT8, INT4, FP16 o un modo mixto
- Si una operación multiply-accumulate cuenta como una o dos operaciones
- Aritmética densa frente a dispersa estructurada
- Resolución de entrada, tamaño del batch y grafo del modelo
- Precisión después de la cuantización
- Latencia solo de NPU frente a toda la canalización de la aplicación
- Transferencias de memoria y ejecución alternativa en host
- Estado térmico y funcionamiento sostenido, no de ráfaga

Un detector es una canalización: captura de imagen, redimensionamiento/letterbox, normalización, transferencia al dispositivo, inferencia neuronal, decodificación, NMS, escalado de coordenadas y lógica de aplicación. Los proveedores suelen publicar solo la parte central de inferencia neuronal. [MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/) resulta valioso porque define escenarios, objetivos de calidad y reglas de medición a nivel de sistema, en vez de comparar por separado cifras de marketing.

Para despliegues YOLO, el registro mínimo creíble de benchmark es:

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

Sin esos campos, dos cifras FPS normalmente miden cosas distintas.

## Cómo elegir una NPU para visión artificial

Elige en este orden, no por la cifra más grande impresa en una página de producto.

1. **Demuestra la compatibilidad del grafo.** Compila el modelo exportado exacto, no un checkpoint del zoo con nombre parecido.
2. **Mide la precisión.** Valida el artefacto compilado en el dataset real de la tarea después de calibrar y cuantizar.
3. **Mide de principio a fin.** Incluye conversión de entrada, transferencias, decodificación y NMS.
4. **Comprueba el acceso al software.** Averigua si el compilador es público, requiere registro o solo está disponible mediante un acuerdo comercial.
5. **Fija la matriz de versiones.** Registra compilador, runtime, controlador, firmware e identificadores del destino.
6. **Comprueba la realidad del sistema operativo.** La compatibilidad con Android, Debian, Yocto, Buildroot, RTOS y Windows no es intercambiable.
7. **Comprueba suministro y ciclo de vida.** Un chip técnicamente excelente es una mala elección para producción si los módulos, controladores o soporte a largo plazo son inciertos.
8. **Después compara coste, potencia y rendimiento.** Esas mediciones solo tienen sentido cuando el mismo modelo funciona correctamente en ambos destinos.

### Puntos de partida prácticos según el caso de uso

| Caso de uso | Plataformas que conviene evaluar primero | Motivo |
|---|---|---|
| Acelerador específico para Raspberry Pi | Hailo-8L/8 y Sony IMX500 | Productos oficiales de Raspberry Pi, imágenes de software y flujos concretos para desarrolladores |
| Complemento general Linux PCIe, M.2 o USB | Productos DEEPX, MemryX, Hailo y Axelera | Formatos de acelerador obtenibles, sujetos a compatibilidad con host/controladores |
| SBC Linux o cámara económica | Rockchip RK3588/RK3576, Amlogic ADLA si se consigue una placa/BSP compatible, AXERA, Canaan K230 o Sunplus SP7350 | Canalizaciones multimedia integradas y evidencia pública de modelos/compilador; verifica VIM4 V13A+ para la NPU A311D2 |
| Android móvil y de alto volumen | Qualcomm QNN/AI Hub, LiteRT, Apple Core ML | Amplia base instalada y runtimes móviles mantenidos |
| Linux industrial y robótica | TI TDA4/AM6xA, NXP i.MX, Renesas RZ/V, Hailo | Productos embebidos de larga duración y ecosistemas de cámara/E/S |
| Endpoint o MCU muy pequeño | STM32N6x7, Infineon PSOC Edge, Himax WiseEye2, Analog Devices MAX7800x, Nuvoton M55M1, Alif/Arm Ethos-U, GAP9 y Syntiant | Límites estrictos de potencia y memoria con herramientas específicas y restricciones de tamaño de modelo |
| Visión en FPGA configurable | Microchip VectorBlox, Lattice sensAI y destinos AMD Vitis AI | Canalizaciones de hardware flexibles, pero la configuración del acelerador, bitstream y compilador de modelos forman un único sistema versionado |
| Visión PCIe de alto rendimiento | Axelera Metis, Hailo, DEEPX, Mobilint, MemryX, Rebellions o SiMa.ai | Productos con aceleradores dedicados y orientación multistream |
| Ecosistema de producto centrado en China | Rockchip, AXERA, D-Robotics, SOPHGO, Huawei Ascend, HiSilicon y Cambricon | Placas regionales, cadenas de herramientas y ejemplos de modelos sólidos |

Esta tabla es una lista de candidatos de investigación, no una clasificación de rendimiento. La compra, la disponibilidad regional y el modelo concreto pueden invertir el orden.

## Qué significa esto para LibreYOLO

El diseño escalable no consiste en decenas de exportadores distintos. Consiste en un contrato de intercambio estricto y adaptadores pequeños de compilador/runtime de cada proveedor:

```text
LibreYOLO model
    -> deterministic static ONNX
    -> VendorCompiler.compile(model, target, calibration, precision)
    -> native artifact + manifest
    -> VendorRuntime.load() / infer()
    -> shared task-specific decode and Results objects
```

Cada artefacto nativo debería tener un manifiesto complementario con:

- Proveedor, destino de chip y destino de placa
- Versiones de compilador, runtime, controlador y firmware
- Checksum del checkpoint fuente y de ONNX
- Nombres, formas, layout, espacio de color, normalización y dtype de entrada
- Precisión de cuantización, escalas si se exponen y hash de los datos de calibración
- Nombres y formas de tensores de salida y su significado semántico
- Tarea de detección, nombres de clases y si la decodificación/NMS se ejecuta en chip o en host
- Resultados registrados de paridad numérica y precisión de tarea

LibreYOLO ya ofrece [exportación ONNX](/docs/export/onnx) portable, [herramientas de cuantización](/docs/export/quantization), una exportación [RKNN](/docs/export/rknn) directa pero deliberadamente acotada y un [flujo Hailo](/docs/export/hailo) honesto que utiliza el compilador externo. Según los criterios de esta guía, Amlogic ADLA es un candidato sólido para la siguiente integración: el toolkit es público, la cobertura de modelos se solapa mucho con LibreYOLO y la ruta A311D anterior puede mantenerse explícitamente separada.

La prioridad después de Amlogic debería basarse en el hardware de los usuarios y en la posibilidad de validar la precisión, no solo en la disponibilidad del compilador. DEEPX, Sony IMX500, D-Robotics, AXERA, SOPHGO y Qualcomm son técnicamente atractivos; TI, NXP, ST y Renesas adquieren especial valor cuando socios industriales pueden proporcionar placas y acceso a CI a largo plazo.

## En observación: silicio real sin contrato público de integración

Excluir por completo a una empresa puede hacer engañoso un mapa de mercado. Incluirla como «compatible» puede ser peor. Estos proveedores venden, muestrean o han anunciado silicio pertinente, pero la evidencia pública aún no establece una ruta reproducible actual de compilador, artefacto y modelo concreto adecuada para un backend LibreYOLO.

| Empresa | Qué es real | Por qué sigue en observación |
|---|---|---|
| [Samsung](https://semiconductor.samsung.com/processor/automotive-processor/exynos-auto-v920/) | Exynos Auto V920 incluye una NPU de doble núcleo anunciada hasta 23.1 TOPS | Samsung indica que su [Neural SDK ya no se ofrece a desarrolladores externos](https://developer.samsung.com/neural/overview.html); Samsung ONE/Circle no demuestra acceso a la NPU Exynos |
| [Novatek](https://www.novatek.com.tw/en-global/Milestone/aboutus_milestones) | Los hitos corporativos actuales mencionan SoC Edge AI y Edge Vision/Imaging AI; un hito de 2020 mencionó MobileNet, SSD y YOLOv3 | No hay matriz pública actual de número de componente/compilador/artefacto/operadores/modelos; el YOLOv3 histórico no prueba el SDK actual |
| [Nextchip](https://www.nextchip.com/en/adas/adas.php?idx=5) | Procesadores automotrices APACHE5/NVS2900 y [APACHE6/NVS3000](https://www.nextchip.com/en/adas/adas.php?idx=7) actuales con NPU aiWare de aiMotive; la página APACHE6 indica actualmente tanto 12 como 8 TOPS | Existen aiWare Studio y un runtime, pero no se encontró públicamente artefacto, guía de cuantización, lista de operadores ni matriz YOLO; las cifras contradictorias del proveedor no deben resolverse en silencio |
| [Black Sesame Technologies](https://bst.ai/en.html) | Silicio automotriz/Edge AI Huashan A1000/A2000 y Wudang serie C | Hay información pública del producto, pero no compilador autoservicio, contrato de artefacto ni model zoo reproducible |
| [Ingenic](https://en.ingenic.com.cn/products-detail/id-19.html) | SoC de cámara T41/T40 con afirmaciones sobre NPU de pocos bits; Magik AI describe PTQ/QAT y compilación de grafos | No se encontró un compilador actual descargable, contrato de artefacto ni lista exacta de YOLO del proveedor |
| [Fullhan](https://fullhan.com/en/index.php?a=type&c=article&tid=9) | Varios SoC IPC anuncian NPU de 0.5 a 2 TOPS, mientras que la [familia NVR MC6880](https://fullhan.com/en/index.php?a=type&c=article&tid=48) anuncia 4 TOPS | No hay documentación pública suficiente de compilador/runtime NN/framework/modelo para reproducirlo independientemente |
| [Goke Microelectronics](http://www.gokemicro.com/News/info.aspx?itemid=437) | Una página de primera parte para cámaras inteligentes GK7606V1/GK7206V1/GK7203V1 anuncia rendimiento NPU integrado, pero al publicarse solo se servía por HTTP heredado | No hay convertidor, contrato runtime ni matriz de modelos nombrados públicos; la ruta está orientada al canal OEM |
| [Chengheng Micro](https://en.chenghengmicro.com/) | CH37 anuncia 64 INT8 TOPS y modos FP16/FP32/FP64; la [cronología de 2026](https://chenghengmicro.com/about.html) de la empresa indica que inició producción de bajo volumen | No se encontró SDK, contrato de compilador ni validación de modelos nombrados públicos; considérala una plataforma de comercialización inicial sin autoservicio, no evidencia de despliegue reproducible |
| [Bouffalo Lab](https://github.com/bouffalolab/bouffalo_sdk) | BL808 incluye la NPU BLAI-100 y existen repositorios oficiales actuales de SDK | El SDK mantenido no expone una ruta oficial actual de conversión/ejemplos BLAI; los flujos que sobreviven son evidencia heredada o comunitaria |

Esta lista de observación se basa deliberadamente en la evidencia. Un proveedor puede pasar a las tablas de despliegue publicando una cadena de herramientas o una ruta de evaluación actuales, un contrato de artefacto específico del destino y al menos un modelo concreto con una receta integral.

## Lo que deliberadamente no se ha afirmado

Este artículo no afirma que todos los modelos mencionados funcionen desde cualquier repositorio upstream. Los modelos de los vendor zoos suelen estar modificados, divididos en diferentes nodos de salida o emparejados con postprocesamiento personalizado.

Tampoco convierte estas afirmaciones en declaraciones de compatibilidad:

- Un compilador importa ONNX.
- Un chip anuncia un controlador Android NNAPI.
- Un distribuidor dice que la placa es «compatible con YOLO».
- Un repositorio comunitario ejecuta una variante de un único modelo.
- Un modelo se compila sin errores.
- Un proveedor informa FPS solo de NPU sin un resultado de precisión.

Los aceleradores de los teléfonos Google Tensor y numerosos ASIC personalizados solo para OEM también existen, pero Google no ofrece Tensor como destino de compilador autoservicio general comparable a las plataformas anteriores. La existencia de una empresa, un diagrama de bloques NPU o la aceleración Android no bastan para inventar una afirmación de integración LibreYOLO.

Asimismo, quedan fuera de alcance los aceleradores de nube como Google TPU, AWS Inferentia/Trainium, Microsoft Maia y las tarjetas IA solo para centros de datos. Esta guía trata de hardware que un desarrollador de visión artificial puede desplegar razonablemente en el edge.

Las licencias de modelos son otra capa. Que un proveedor de chips publique una receta de conversión YOLO no concede derechos para usar o redistribuir los pesos upstream, el código de entrenamiento ni el derivado compilado. Hay que comprobar por separado la compatibilidad del hardware, la licencia del SDK y la licencia del modelo para el producto previsto.

## Metodología de investigación y correcciones

Para cada empresa, la investigación buscó cuatro tipos de fuentes primarias:

1. Una página de producto o ficha técnica actual que identificara el silicio.
2. Documentación del compilador y runtime que explicara la ruta real de despliegue.
3. Un model zoo, una tabla de compatibilidad o un tutorial integral que nombrara modelos de visión.
4. Una señal de ciclo de vida o acceso que mostrara si los desarrolladores podían obtener realmente las herramientas.

Las organizaciones GitHub de primera parte cuentan como fuentes del proveedor. La documentación de fabricantes de placas se etiqueta como evidencia del ecosistema cuando el fabricante del chip no publica el mismo material. Las afirmaciones de rendimiento de terceros se excluyeron de las tablas comparativas.

Después se volvió a auditar el borrador en pasadas separadas por grupos de proveedores y entre tablas. Esas revisiones volvieron a validar el alcance de destinos, nombres de artefactos, precisión, evidencia del modelo frente al postprocesador, estado de anunciado frente a disponible, etiquetas de ciclo de vida, denominadores de benchmarks y todos los recuentos de entidades. Cuando dos páginas de primera parte entran en conflicto, esta guía lo hace explícito en vez de escoger en silencio el valor más conveniente.

Este mercado cambia rápidamente. Si trabajas para una de estas empresas y un chip, SDK, lista de modelos o estado de acceso es incorrecto, envía a LibreYOLO la URL pública exacta de documentación y su versión. Las correcciones respaldadas por evidencia primaria deberían reemplazar este texto; no deberían hacerlo las afirmaciones de marketing sin documentación.

## Preguntas frecuentes

### ¿Cuántas empresas de NPU para Edge AI existen?

No hay un total universal de empresas porque se solapan los proveedores de productos, los licenciantes de IP, los sensores inteligentes, las GPU y los ASIC automotrices privados. Esta guía no exhaustiva sigue a 69 empresas y ecosistemas de plataforma con chips Edge AI pertinentes, plataformas aceleradoras, IP de NPU licenciable o silicio relevante en observación, a fecha de agosto de 2026.

### ¿Qué es una NPU?

Una unidad de procesamiento neuronal es hardware especializado para operaciones de redes neuronales, como convoluciones y multiplicación de matrices. Los proveedores usan términos como NPU, AIPU, BPU, KPU, DLA, HTP, TPU y MLA, y los modelos compilados generalmente no son portables entre empresas.

### ¿Qué empresas de NPU ofrecen compatibilidad oficial con modelos YOLO?

Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 mediante el repositorio oficial de Raspberry Pi AI Camera, STMicroelectronics, Renesas, Lattice, Himax, Microchip y otras empresas tienen entradas de primera parte en model zoos, tutoriales o resultados validados para al menos una generación de YOLO. La generación, tarea, chip de destino y versión del SDK concretos siguen siendo importantes.

### ¿Se puede ejecutar cualquier modelo ONNX en cualquier NPU?

No. ONNX es un formato de intercambio, no una garantía de compatibilidad con el hardware. El compilador de NPU debe admitir todos los operadores, las formas de tensores y los tipos de datos del grafo. Son habituales las formas estáticas, los cortes de grafo, las operaciones personalizadas y la ejecución alternativa en CPU.

### ¿Cuál es la mejor NPU para YOLO?

No hay un ganador universal. Hailo, Rockchip, Axelera AI, DEEPX y Amlogic tienen una amplia compatibilidad documentada con YOLO, mientras que Qualcomm ofrece un alcance excepcional de dispositivos. Elige según la precisión tras la cuantización, la latencia de toda la canalización, la potencia sostenida, el precio, la disponibilidad, el acceso al SDK y la compatibilidad con el sistema operativo.

### ¿Por qué no se pueden comparar directamente las cifras TOPS de las NPU?

Los proveedores pueden contabilizar precisiones, operaciones dispersas, operaciones multiply-accumulate y supuestos de utilización máxima distintos. TOPS no refleja el tráfico de memoria, los operadores no compatibles, el preprocesamiento, el postprocesamiento ni la carga del host. Compara el mismo modelo, precisión, resolución, precisión y canalización completa.

### ¿Qué es la calibración de una NPU?

La calibración ejecuta imágenes representativas, normalmente sin etiquetar, del despliegue a través del modelo para que el compilador estime los rangos de activación de INT8 u otro formato de baja precisión. Las imágenes aleatorias o poco representativas pueden generar un artefacto compilado y, aun así, perjudicar la precisión de la tarea.

### ¿LibreYOLO es compatible con NPU edge?

LibreYOLO exporta a ONNX y varios formatos de runtime, tiene una ruta directa al compilador RKNN para algunos modelos Rockchip y documenta el flujo externo de compilación para Hailo. Cualquier otro artefacto nativo de proveedor sigue necesitando su SDK y debe describirse como candidato a integración hasta que se compile, valide y ejecute en hardware.
