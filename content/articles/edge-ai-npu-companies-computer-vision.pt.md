---
title: "69 empresas de Edge AI com NPU: chips, SDKs e YOLO (2026)"
description: "Guia com fontes sobre 69 empresas de edge AI e ecossistemas de NPU: chips, SDKs, artefatos, quantização e suporte documentado a visão computacional e YOLO."
date: 2026-08-15
author: Xuban
tags: [edge-ai, npu, computer-vision, yolo, hardware, ai-accelerators, amlogic, hailo, rockchip]
faq:
  - q: "Quantas empresas de NPU para edge AI existem?"
    a: "Não existe um total universal de empresas porque fornecedores de produtos, licenciadores de IP, sensores inteligentes, GPUs e ASICs automotivos privados se sobrepõem. Este guia não exaustivo acompanha 69 empresas e ecossistemas de plataforma com chips de edge AI, plataformas aceleradoras, IP de NPU licenciável ou silício relevante em uma lista de observação, em agosto de 2026."
  - q: "O que é uma NPU?"
    a: "Uma unidade de processamento neural é um hardware especializado em operações de redes neurais, como convoluções e multiplicação de matrizes. Os termos dos fornecedores incluem NPU, AIPU, BPU, KPU, DLA, HTP, TPU e MLA, e os modelos compilados geralmente não são portáveis entre empresas."
  - q: "Quais empresas de NPU oferecem suporte oficial a modelos YOLO?"
    a: "Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 por meio do repositório oficial da AI Camera da Raspberry Pi, STMicroelectronics, Renesas, Lattice, Himax, Microchip e várias outras têm entradas em model zoos próprios, tutoriais ou resultados validados para pelo menos uma geração do YOLO. A geração, a tarefa, o chip alvo e a versão do SDK ainda importam."
  - q: "Qualquer modelo ONNX pode rodar em qualquer NPU?"
    a: "Não. ONNX é um formato de intercâmbio, não uma garantia de compatibilidade com o hardware. O compilador da NPU precisa oferecer suporte a cada operador, formato de tensor e tipo de dado do grafo. Formatos estáticos, cortes no grafo, operações personalizadas e fallback para CPU são comuns."
  - q: "Qual é a melhor NPU para YOLO?"
    a: "Não há uma vencedora universal. Hailo, Rockchip, Axelera AI, DEEPX e Amlogic têm ampla cobertura documentada de YOLO, enquanto a Qualcomm tem um alcance excepcional de dispositivos. Escolha considerando a acurácia após a quantização, a latência do pipeline completo, a potência sustentada, o preço, a disponibilidade, o acesso ao SDK e o suporte ao sistema operacional."
  - q: "Por que não dá para comparar diretamente os valores de TOPS das NPUs?"
    a: "Os fornecedores podem contar precisões, operações esparsas, multiplicações-acumulações e hipóteses de utilização de pico diferentes. TOPS não inclui tráfego de memória, operadores sem suporte, pré-processamento, pós-processamento nem a sobrecarga do host. Compare o mesmo modelo, precisão, resolução, acurácia e pipeline completo."
  - q: "O que é calibração de NPU?"
    a: "A calibração executa imagens representativas, geralmente sem rótulos, do ambiente de deploy pelo modelo para que o compilador estime os intervalos de ativação para INT8 ou outro formato de baixa precisão. Imagens aleatórias ou pouco representativas ainda podem gerar um artefato compilado, mas prejudicar a acurácia da tarefa."
  - q: "O LibreYOLO oferece suporte a NPUs de edge?"
    a: "O LibreYOLO exporta ONNX e vários formatos de runtime, tem um caminho direto de compilação RKNN para modelos Rockchip selecionados e documenta o fluxo externo de compilação Hailo. Todo outro artefato nativo do fornecedor ainda exige o SDK correspondente e deve ser descrito como candidato a integração até ser compilado, validado e executado no hardware."
---

**Não existe um único mercado de NPU. Este guia não exaustivo acompanha 69 empresas e ecossistemas de plataforma: 51 fornecedores com chips ou plataformas para deploy, nove fornecedores de IP de NPU e nove empresas claramente identificadas em uma lista de observação.** Eles abrangem várias classes incompatíveis de aceleradores e quase o mesmo número de pilhas de compilação. A maioria promete o mesmo caminho: exportar um modelo, quantizá-lo, compilá-lo, copiar um artefato proprietário para uma placa e chamar um runtime do fornecedor. Os detalhes determinam se esse caminho leva uma tarde ou um trimestre de engenharia.

Este guia mapeia as empresas que oferecem hardware de visão computacional confiável em 2026. Ele registra os chips relevantes, nomes de SDKs e compiladores, artefatos de deploy, nível de acesso público e modelos documentados pelo próprio fornecedor. Também dá atenção especial à Amlogic, cuja nova pilha ADLA difere bastante do ecossistema NPU mais antigo do A311D.

> **Escopo da pesquisa:** as fontes foram verificadas em 15 de agosto de 2026. As afirmações sobre modelos específicos vêm de páginas de produtos, documentação, model zoos, repositórios ou tutoriais dos próprios fornecedores, salvo indicação explícita em contrário. Os valores de TOPS anunciados são números dos fornecedores, não resultados de benchmark comparáveis de forma independente. Este é um guia para desenvolvedores, não uma recomendação de investimento nem um ranking pago.

**Navegação rápida:** [tabelas de empresas](#npu-companies-and-platforms-at-a-glance) | [análise aprofundada da Amlogic](#amlogic-two-npu-generations-not-one) | [perfis dos fornecedores](#the-strongest-public-deployment-ecosystems) | [tabela de artefatos compilados](#what-you-actually-deploy-the-artifact-lock-in-table) | [acesso e ciclo de vida](#tool-access-and-lifecycle-signals) | [roteiro do LibreYOLO](#what-this-means-for-libreyolo)

## A descoberta mais importante

O hardware é fragmentado, mas o padrão de deploy é surpreendentemente consistente:

```text
PyTorch checkpoint
    -> ONNX or TFLite interchange graph
    -> representative calibration data
    -> vendor quantizer and graph compiler
    -> chip-specific binary or model package
    -> vendor runtime on the target
    -> application preprocessing, decode, NMS and rendering
```

[O ONNX permite explicitamente runtimes, geradores de código e implementações de hardware](https://onnx.ai/onnx/repo-docs/IR.html), mas um arquivo ONNX é apenas o ponto de transferência. Isso não significa que todo operador ONNX possa ser mapeado para qualquer acelerador. Formatos estáticos de entrada, restrições de operadores suportados, regras de quantização e fallback no host determinam o que realmente roda na NPU.

Por isso, a pilha de software faz parte do chip. Uma NPU nominalmente rápida com um compilador restrito ou frágil pode ser um alvo de deploy pior do que um dispositivo menor com toolchain público, model zoo, simulador e runtime estável.

## O que significa "com suporte" neste guia

A literatura dos fornecedores usa a palavra *suporte* de forma bastante vaga. Este artigo separa quatro níveis de evidência:

| Nível de evidência | O que comprova | O que não comprova |
|---|---|---|
| **Modelo validado** | O fornecedor publica uma entrada de modelo específica para o chip, um resultado ou uma tabela de compatibilidade | Que seu checkpoint modificado mantém a mesma acurácia |
| **Exemplo oficial** | O fornecedor oferece um tutorial ou demo de ponta a ponta para um modelo identificado | Que outros tamanhos, tarefas ou gerações compilem |
| **Capacidade do compilador** | O SDK importa um framework ou expõe operadores suportados | Que um grafo YOLO específico funcione de ponta a ponta |
| **Marketing ou acesso restrito** | O produto se destina a visão e existe um SDK privado | Compatibilidade de modelo que possa ser reproduzida publicamente |

Essa distinção importa. "Importa ONNX" não equivale a "oferece suporte à segmentação do YOLO11". Um modelo pode ser analisado, mas cair para CPU, compilar com valores numéricos de saída incorretos, exceder a memória do acelerador ou perder acurácia durante a quantização.

## Empresas e plataformas de NPU em resumo

As tabelas misturam intencionalmente SoCs, aceleradores discretos, sensores inteligentes e alvos próximos de GPU/FPGA. Eles concorrem na mesma decisão de deploy, mesmo quando os fornecedores usam nomes como NPU, AIPU, BPU, KPU, DLA, HTP, TPU ou MLA.

### SoCs embarcados, sensores inteligentes e processadores industriais

| Empresa | Chips ou plataformas relevantes | SDK, compilador e artefato | Evidência de visão computacional documentada publicamente | Acesso |
|---|---|---|---|---|
| [Amlogic](https://github.com/Amlogic-NN/amlnn-toolkit) | A311D2, S928X, S905X5/S905D5, A311Y3, C308L/C302X/C302X2, T968D4, C305X2 e A123X | AMLNN Toolkit e `libnnsdk.so`, `.adla` compilado; pilha Acuity `.nb` legada separada para A311D | O [model playground](https://github.com/Amlogic-NN/amlnn-model-playground) documenta YOLOv5/6/7/8/10/11, YOLOX, YOLOE, YOLO-World, PP-YOLOE, segmentação, pose e OBB em A311D2, S905X5, A311Y3, C305X2 e A123X; os outros chips listados aqui são alvos do compilador, não entradas nessa matriz de suporte | GitHub e wheels públicos; ainda é necessário um BSP/driver compatível |
| [Rockchip](https://github.com/airockchip/rknn-toolkit2) | RK3562/3566/3568, RK3576, RK3588, RV1126B | RKNN-Toolkit2, RKNN Runtime, `.rknn` | YOLOv5/6/7/8/10/11, YOLOX, YOLO-World, PP-YOLOE, segmentação, pose e OBB no [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) | GitHub público; wheels binários |
| [Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | Plataformas Snapdragon e Dragonwing, QCS6490, QCS8550 e Dragonwing IQ-9075 | QAIRT/QNN, SNPE e AI Hub; binários de contexto QNN ou DLC | A [coleção de modelos do AI Hub](https://github.com/qualcomm/ai-hub-models) inclui YOLOv3/5/6/7/8/9/10/11/26, YOLOX, YOLO-World, YOLOR, RF-DETR e modelos de detecção/segmentação/pose | Documentação pública; requisitos de SDK/conta variam |
| [Texas Instruments](https://github.com/TexasInstruments/edgeai-tidl-tools) | AM62A, AM67A, AM68A, AM69A e TDA4x; TDA54-Q1 em preview | Processor SDK Edge AI e TIDL | O [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo) publica modelos otimizados de detecção, segmentação, pose e classificação para os caminhos atuais AM6xA/TDA4; isso ainda não é uma matriz de alvos para o TDA54-Q1 em preview | GitHub público e Processor SDK |
| [NXP](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | i.MX 8M Plus, i.MX 93, i.MX 95 e Kinara Ara-1/Ara240, adquiridos pela empresa | eIQ com TIM-VX, Vela, Neutron Converter e Ara SDK | O [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) inclui assets ou receitas para YOLOv4-tiny/v8, NanoDet, CenterNet, FastestDet, SSD Lite e YOLACT; a entrada de YOLOv5 contém apenas documentação | Componentes públicos e outros com acesso restrito por conta |
| [STMicroelectronics](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) | Variantes STM32N6x7, incluindo STM32N657/647, com acelerador Neural-ART | STM32Cube AI Studio e ST Edge AI Core | O repositório de serviços atual lista Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 e ST-YOLOX, além de pose e segmentação | Ferramentas e repositórios públicos |
| [Infineon](https://documentation.infineon.com/psocedge/) | PSOC Edge E83/E84 com Cortex-M55 e Ethos-U55; acelerador NNLite separado no lado M33 | ModusToolbox, [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter), TFLite Micro e Arm Vela | O material de arquitetura cita MobileNetV1/V2 como kernels representativos e o kit E84 AI inclui uma câmera, mas não foi encontrada uma matriz de validação YOLO própria para PSOC Edge | Documentação pública, componentes do SDK e kits atuais de avaliação E84 |
| [Renesas](https://github.com/renesas-rz/rzv_drp-ai_tvm) | RZ/V2L, V2M, V2MA, V2H e V2N | DRP-AI Translator, DRP-AI TVM e RUHMI | A [lista de modelos do RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) valida tamanhos YOLOv5/v8/v11/26, YOLOX e variantes de pose e segmentação | GitHub público e SDK da placa |
| [Sony](https://www.aitrios.sony-semicon.com/edge-ai-devices/imx500) | Sensor de visão inteligente IMX500 e Raspberry Pi AI Camera | Edge-MDT, Model Compression Toolkit e empacotador IMX500; pacote `.rpk` | O [zoo IMX500 da Raspberry Pi](https://github.com/raspberrypi/imx500-models) inclui YOLOv8n, YOLO11n, EfficientDet Lite, NanoDet+ e SSD | Fluxo público da Raspberry Pi; aplicam-se os termos das ferramentas Sony |
| [Ambarella](https://www.ambarella.com/developer/model-garden/) | Famílias CV72/CV75, CV5/CV52 e CV3-AD | Cooper Developer Platform, compilador CVflow e DAGs de runtime | O model garden público cita YOLOX-S, RTMDet-nano, DeepLabV3+, TopFormer, OWL-ViT e LLaVA OneVision | Principalmente restrito a parceiros |
| [Synaptics](https://developer.synaptics.com/docs/sl/overview) | Astra SL1600/SL1680; SL2611/13/15/17/19; família MCU SR100 separada | SyNAP `.synap` em SL16xx; Torq/IREE `.vmfb` em SL261x; SDK SR separado, voltado a Ethos-U55 | Benchmark SyNAP oficial de YOLOv8n/v8s e exemplo Torq YOLOv8 para SL261x; as evidências de SR devem ser avaliadas separadamente | Portal de desenvolvedores; alguns downloads restritos |
| [MediaTek](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) | Genio 360/360P/420/520/720 com NP8 e MDLA 5.3; Genio 510/700/1200 com NP6/MDLA mais antigos | NeuroPilot Converter, `ncc-tflite`, Neuron Runtime e `.dla`; caminho ONNX Runtime em alguns sistemas mais recentes | O IoT AI Hub oficial publica páginas de modelos e benchmarks para YOLOv5 e YOLOv8, além de modelos de classificação e face | Documentação Yocto pública; os pacotes NP8 principais e materiais Android exigem acesso direto de cliente/NDA |
| [Allwinner](https://docs.aw-ol.com/v853/en/npu/dev_npu/) | SoC de visão V853 com NPU Vivante de 1-TOPS | Conversão Acuity/Pegasus e runtime `viplite` | O material oficial do V853 cita YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet e redes de face/pessoa | Documentação pública; o download do pacote SDK pode exigir conta |
| [Canaan/Kendryte](https://github.com/kendryte/nncase) | K230/K230D atuais; KPU K210/K510 legados | Compilador e runtime `nncase`; `.kmodel` | O guia nncase do K230 aborda compilação/simulação/runtime de YOLOv5s; o catálogo oficial de demos e o [changelog atual do CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) documentam tarefas YOLOv8/11/26 | Compilador/PyPI públicos; o plug-in do chip é binário |
| [Alif Semiconductor](https://alifsemi.com/support/kits/ensemble-e7appkit/) | Ensemble E7 com dois aceleradores de ML, incluindo Ethos-U55; [E8](https://alifsemi.com/ensemble-e8-series/) com um NPU Ethos-U85 e dois Ethos-U55 | Deploy baseado em Arm Vela e TFLite Micro | A Alif publica um benchmark de detector de faces YOLO-Fastest INT8 no nível da família, mas não uma matriz ampla de YOLO moderno por alvo | Documentação pública e recursos de kits de avaliação |
| [Himax](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) | MCU de IA de endpoint HX6538 WiseEye2 com Cortex-M55 e Ethos-U55 | TFLite Micro e Arm Vela, com CMSIS-NN e fallback para kernels de referência | Os [exemplos oficiais do WiseEye2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) incluem detecção, pose e classificação com YOLOv8n, detecção com YOLO11n, face mesh e PeopleNet | GitHub e documentação públicos, além de placa parceira disponível para compra |
| [Analog Devices](https://www.analog.com/en/products/max78002.html) | MCUs de IA MAX78000 e MAX78002 com aceleradores CNN de baixo consumo | [`ai8x-training`](https://github.com/analogdevicesinc/ai8x-training), `ai8x-synthesis`/`izer` e MSDK; C e pesos gerados | O material oficial cobre identificação/detecção de faces, RetinaNet, Visual Wake Words, classificadores e reconhecimento de ações; não foi encontrado deploy YOLO de primeira parte | Ferramentas GitHub, documentação e placas de avaliação públicas |
| [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo) | Placas BPU RDK X3/X5/Ultra e da série S100 mais recente | OpenExplorer/Algorithm Toolchain e `hbm_runtime`; `.bin` no X5, `.hbm` no `rdk_s` atual | O branch `rdk_x5` atual documenta YOLOv5/v5u/v8/v9/v10/11/12/13/26, YOLOE, YOLO-World, segmentação, pose, classificação, OCR e CLIP para RDK X5; X3 e série S usam branches separados | GitHub público; alguns pacotes de toolchain são específicos da plataforma |
| [AXERA](https://github.com/AXERA-TECH/ax-samples) | AX650, AX637, AX630C, AX620Q e AX615 | Compilador Pulsar2 e AXEngine; `.axmodel` | Os exemplos oficiais atuais cobrem YOLOv5/6/7/8/9/10/11/13/26, YOLOX e YOLO-World; tarefas e alvos variam por chip | Exemplos e documentação públicos; compilador distribuído separadamente |
| [SOPHGO](https://github.com/sophgo/tpu-mlir) | BM1684/1684X/1688/1690 e CV186X/CV18xx | TPU-MLIR e runtime SOPHON; `.bmodel` ou `.cvimodel` | As demos oficiais cobrem YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, OBB, segmentação, face, SAM e OCR | Compilador de código aberto e runtime do fornecedor |
| [Huawei Ascend](https://www.hiascend.com/en/software/cann) | Ascend 310/310P/310B e produtos de edge Atlas 200I/300I | CANN, compilador ATC e AscendCL; `.om` | O [Ascend ModelZoo](https://github.com/Ascend/modelzoo) oficial inclui modelos YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN e segmentação | Documentação pública; pacotes CANN e kernels devem corresponder ao alvo |
| [Cambricon](https://github.com/Cambricon/magicmind_cloud) | MLU370-X4/S4 na matriz pública citada; MLU270 é hardware legado não coberto por ela | Neuware e MagicMind | A matriz MagicMind 1.7 do repositório lista modelos YOLOv3/4/5/7/8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN e segmentação | Exemplos históricos de modelos públicos; acesso ao SDK/container atual restrito |
| [Sunplus](https://sunplus.atlassian.net/wiki/spaces/C3/pages/1971126471/SP7350%2BSpecification) | SP7350/C3V, aproximadamente 4.1 a 4.6 TOPS anunciados | Docker Vivante Acuity NPU, runtime SNNF e `.nb` | A documentação oficial fornece YOLOv5 e um [deploy personalizado de YOLOv8 de ponta a ponta](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) | Fontes/documentação públicas e ecossistema de placas |
| [ESWIN Computing](https://www.eswincomputing.com/en/news/info/96.html) | SoCs RISC-V EIC7700/EIC7700X e [EIC7702/EIC7702X](https://www.eswincomputing.com/en/news/info/104.html) de dois dies | ENNP: EsQuant, compilador EsAAC atual, simulador e runtime ESSDK; `.model`; documentação antiga usava `ennc-compile` | A documentação ENNP hospedada pela Milk-V inclui um [tutorial de YOLOv3 de ponta a ponta](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3), além de exemplos MobileNetV2 e ResNet | Documentação mista da primeira parte e hospedada por fabricante de placas; downloads regionais |
| [Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m55-mcus/m55m1-series/index.html) | MCU M55M1 com Ethos-U55-256 | TFLite Micro, Arm Vela e NuEdgeWise/NuML | Os exemplos oficiais do [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) citam YOLOv8-nano, YOLOX-nano, YOLO Fastest, SSD-MobileNet e modelos de classificação | Toolchain de MCU em grande parte público |
| [Realtek](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Plataforma de câmera AmebaPro2 RTL8735B / AMB82-mini | SDK Arduino/FreeRTOS, VoE e NeuralNetwork API; `.nb` | Os modelos empacotados incluem YOLOv3-tiny, YOLOv4-tiny, YOLOv7-tiny, SCRFD e MobileFaceNet | Runtime público; acesso ao [conversor personalizado](https://ameba-doc-arduino-sdk.readthedocs-hosted.com/en/latest/FAQ/offline_ai_model_conversion_steps.html) exige contato |
| [Telechips](https://docs.topst.ai/product/p/ai) | Placa TCC7500 / TOPST AI, anunciada em 8 TOPS | TC-NN-Toolkit / Enlight SDK; intermediário `.enlight` e pacote compilado para deploy | Um [projeto oficial TOPST](https://docs.topst.ai/blog/31) fez deploy de YOLOv8s; as conversões UFLD v1/v2 falharam em camadas sem suporte e só foi proposta uma solução alternativa com divisão/pós-processamento. [YOLOv4](https://community.topst.ai/t/segmentation-fault/358) e [outro fluxo YOLOv8](https://community.topst.ai/t/segmentation-fault-error/415) aparecem em discussões da comunidade | Documentação da placa pública; downloads do compilador/operadores frequentemente exigem permissão |
| [T-Head](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | TH1520, anunciado em 4-TOPS INT8, na LicheePi 4A | Compilador HHB e CSI-NN2/SHL; `hhb.bm` mais código/parâmetros gerados | Exemplos oficiais do fornecedor da placa executam YOLOv5n/s e MobileNetV2 na NPU | Exemplos públicos; incerteza sobre envelhecimento da toolchain/manutenção |
| [SigmaStar](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | SoC de câmera inteligente atual SSU9383CM e famílias IPU antigas | Runtime MI_IPU atual; uma toolchain SGS_IPU [antiga](https://wx.comake.online/doc/doc/SigmaStarDocs-SSC9381G_9351_Pudding-ULS00V040-20210913/customer/development/dla/tools.html) gerava `.sim` para `sgsimg.img` | Um pós-processador de um SDK oficial [antigo](https://wx.comake.online/doc/doc/Sigmastar_SDK_v1.2.2/module/User_Guide/Common/SigmaStar_Post_Processing_Module.html) cita SSD e YOLOv1/2/3; a compatibilidade pública desses modelos e artefatos com SSU9383CM não foi verificada | Silício atual, mas evidências públicas citadas sobre compilador/modelos abrangem gerações diferentes |
| [HiSilicon](https://www.hisilicon.com/cn/products/smart-vision/machine-vision/hi3516cv610) | SoCs de visão inteligente Hi3516CV610/DV500, Hi3519DV500 e Hi3403V100 | ATC para `.om` com runtime de placa NNN/SVP-NNN | A matriz específica por alvo da HiSpark, especialmente para Hi3403 e Hi3591P, cita entradas de YOLOv3 a YOLO11, pose, segmentação, OBB, OCR e profundidade; isso não valida todos os SoCs desta linha | Ativa e distinta de Ascend; os modelos do repositório são identificados como somente para uso não comercial |

### Aceleradores e módulos edge dedicados

| Empresa | Silício relevante | SDK e artefato | Evidência pública de modelos | Acesso |
|---|---|---|---|---|
| [Hailo](https://github.com/hailo-ai/hailo_model_zoo) | Famílias Hailo-8, Hailo-8L, Hailo-10H e Hailo-15 | Dataflow Compiler e HailoRT; `.hef` | YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 e YOLO26, além de YOLOX, DAMO-YOLO, SSD, EfficientDet, segmentação, pose e OBB, com tabelas de modelos específicas por geração | Model Zoo público; compilador no Developer Zone |
| [Axelera AI](https://docs.axelera.ai/sdk/reference/models/model-zoo/) | Produtos Metis disponíveis; produtos Europa e Titania anunciados para o portfólio | Voyager SDK público; Pipeline Builder alfa `.axm`/`.axe`, bundles `.axmodel` clássicos | YOLOv3/5/7/8/9/10/11/26, YOLOX, YOLO-NAS, OBB, pose, segmentação e muitos modelos não YOLO validados | SDK público no GitHub; suporte ao cliente exige conta |
| [DEEPX](https://developer.deepx.ai/modelzoo/) | DX-M1 e DX-M1M | DXNN SDK, DX-COM e DX-RT; `.dxnn` | O Model Zoo listava 354 entradas com DX-COM 2.4.0/DX-RT 3.4.0 quando verificado em 15 de agosto de 2026, incluindo YOLOv3 a YOLO11 e YOLO26, YOLOX, SSD, EfficientDet, NanoDet, segmentação, pose e OBB | Recursos e suite para desenvolvedores públicos |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | Acelerador MX3 e módulos com vários chips | SDK MemryX, Neural Compiler e runtime; `.dfp` | Exemplos oficiais e notas de versão cobrem detecção, segmentação e pose, incluindo YOLOv10, YOLO11 e YOLO26 | Documentação e SDK públicos |
| [Kneron](https://doc.kneron.com/docs/) | KL520, KL530, KL630, KL720 e KL730 têm documentação atual do compilador; KL830 aparece em algumas APIs PLUS, mas não na lista atual de alvos do compilador | Kneron PLUS e Model Toolchain; `.nef` nos alvos documentados do compilador | O [fluxo YOLO oficial](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) é centrado em Tiny-YOLOv3; outros materiais cobrem treinamento e deploy de YOLOv5 | Documentação pública; pacotes/ferramentas variam por chip |
| [SiMa.ai](https://docs.sima.ai/pages/palette/modelsdk.html) | MLSoC e plataforma Modalix de produção, 50-TOPS | Palette, ModelSDK, MLA Compiler e ModelExecutor | Versões públicas validam YOLOv7/v8, YOLOX, pose/segmentação, DETR, Mask R-CNN e EfficientDet | Documentação pública; SDK do produto comercial |
| [EdgeCortix](https://www.edgecortix.com/en/hardware) | SAKURA-II, acelerador anunciado em 60-TOPS em produtos M.2 e PCIe | Compilador e framework MERA | O fornecedor descreve suporte de visão até IA generativa, mas não publica uma matriz de compatibilidade YOLO atual e suficientemente precisa | Teste/consulta comercial; validação conduzida por parceiros |
| [BrainChip](https://brainchip.com/metatf-dev-tools/) | Coprocessador/módulo M.2 AKD1500 disponível; plataformas AKD1000 legadas; IP Akida 2 | Pacotes MetaTF: `akida-models`, `quantizeml`, `cnn2snn` e runtime `akida` | O cartão de modelo Akida 2 cita detector YOLOv2 AkidaNet0.5, CenterNet, AkidaUNet e reconhecimento facial; esses resultados não validam automaticamente o AKD1500 | Pacotes Python principais públicos; mapeamento de hardware depende do alvo |
| [Blaize](https://www.blaize.com/products/) | Produtos P1600, Pathfinder e Xplorer | Picasso SDK, NetDeploy e AI Studio | Visão é um mercado-alvo, mas não foi encontrada uma matriz pública auditável de compatibilidade por modelo | Comercial/restrito |
| [Google Coral](https://github.com/google-coral/edgetpu) | Produtos Edge TPU USB, PCIe, M.2 e Dev Board legados; IP [Coral NPU](https://github.com/google-coral/coralnpu) de código aberto e ativo, separado | Edge TPU Compiler/`libedgetpu`/PyCoral legados; o novo Coral NPU RISC-V expõe IP, RTL/simulação e exemplos ELF | Os exemplos legados enfatizam SSD MobileNet, classificação, DeepLab e MoveNet; o novo IP não tem matriz pública de deploy YOLO e não substitui diretamente os produtos Edge TPU | Repositórios principais legados arquivados; IP Coral NPU novo em desenvolvimento ativo |
| [Lattice Semiconductor](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | FPGAs ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E e Avant-X | sensAI Studio, Neural Network Compiler e IP de acelerador configurável | A tabela atual do compilador lista YOLOv1, YOLOv5, YOLOv8 e YOLO11 em modos específicos por alvo, além de SSD, MobileNetV2-SSD, ResNet e ENet | Compilador/manuais públicos; termos de IP variam, e o Advanced CNN Accelerator é comercial |
| [Mobilint](https://www.mobilint.com/sdk-qb) | Aceleradores REGULUS de 10-TOPS e ARIES MLA100 de 80-TOPS | qb SDK; compilador INT8 e `.mxq` | O [model zoo](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) público lista detecção YOLOv3/5/7/8/9/10/11/12/26 e variantes de segmentação, pose e OBB | Documentação/modelos públicos; SDK e hardware comerciais |
| [Rebellions](https://rebellions.ai/developers/) | ATOM+ CA22 e ATOM-Max CA25 ativos; ATOM CA02/ATOM+ CA12 em fim de vida; situação da ferramenta ATOM-Lite CA21 incerta | Compilador/runtime/profiler RBLN SDK; `.rbln` | Versões oficiais de suporte citam YOLOv3, famílias YOLOv5/6/7/8 e tutoriais atuais de YOLOv8 | Documentação pública; wheel do compilador exige credenciais do portal |
| [FuriosaAI](https://furiosa.ai/warboy) | Warboy Gen1 voltado a visão; geração RNGD separada | Furiosa SDK; `.enf` INT8 no Warboy, `.fxb` separado no RNGD | O zoo Warboy documenta SSD, YOLOv5M/L e YOLOv7-w6-pose; o roadmap RNGD marca suporte a YOLOv8m concluído no quarto trimestre de 2024, mas não expõe uma matriz pública atual detalhada de visão | Acesso IAM/conta; mantenha as duas gerações separadas |

### Plataformas adjacentes que desenvolvedores comparam com NPUs

| Empresa | Hardware | Pilha de deploy | Por que entra na comparação |
|---|---|---|---|
| [NVIDIA](https://github.com/NVIDIA-AI-IOT/deepstream_tools/tree/main/yolo_deepstream) | GPU Jetson Orin mais DLA; GPU Jetson Thor sem DLA | JetPack, TensorRT e DeepStream; `.engine` | Deploy de visão extremamente maduro; as ferramentas oficiais DeepStream documentam YOLOv4/v7/v8/v9/11, incluindo uma configuração YOLO11 OBB. Muitos resultados usam a GPU, e camadas sem suporte no DLA do Orin podem cair para a GPU. |
| [AMD](https://vitisai.docs.amd.com/en/6.2/) | Alvos Kria/DPU legados; Vitis AI 6.2 GA para Versal AI Edge Gen1 VEK280/VE2802 e Gen2 VEK385; NPUs client Ryzen AI separados | `.xmodel` legado; snapshot NPU vinculado ao alvo em Gen1; diretório de cache compilado ou pacote de produção `.rai` em Gen2 | O Vitis AI publica material de visão, mas DPU legado, Versal Gen1, Versal Gen2 e Ryzen AI têm contratos de compilação e deploy separados. |
| [Microchip](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox) | FPGA PolarFire SoC com IP acelerador CoreVectorBlox configurável | VectorBlox SDK 3.1 público e Libero; assets de deploy `.vnnx`, `.hex` e `.ucomp` | Os [tutoriais](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) atuais cobrem YOLOv5n, detecção/classificação/OBB/pose/segmentação YOLOv8, YOLOv9t e outros modelos de visão; o SDK 3.1 está limitado atualmente ao PolarFire SoC Video Kit. |
| [Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) | NPU Core Ultra, CPU e GPU integrada/discreta | OpenVINO IR ou cache de modelo compilado | O OpenVINO oferece um caminho público entre XPU; use as colunas NPU da [matriz de modelos verificados](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html), sem presumir que todo exemplo YOLO do OpenVINO foi validado em NPU. |
| [Apple](https://developer.apple.com/machine-learning/core-ml/) | Apple Neural Engine em chips A-series a partir do A11 e em chips M-series | Core ML e `coremltools`; `.mlpackage`/`.mlmodelc` | NPU de consumo muito acessível pelo Core ML, mas a Apple abstrai a divisão exata entre CPU/GPU/Neural Engine em vez de expor um compilador de hardware específico para YOLO. |
| [GreenWaves Technologies](https://github.com/GreenWaves-Technologies/nn_menu_gap9) | GAP9 com engine neural NE16 | GAP SDK, NNTool e código gerado pelo AutoTiler | Os repositórios oficiais incluem MobileNet SSD, detecção facial, classificação e um detector dedicado de pessoas [YOLOX](https://github.com/GreenWaves-Technologies/yolox_people_detection). O SDK completo está limitado a clientes qualificados. |
| [Syntiant](https://www.syntiant.com/hardware) | Neural Decision Processors NDP200 em produção em massa e NDP250 em amostragem | SDK Syntiant e pacotes de deploy | Processadores de visão/sensores sempre ativos e de consumo ultrabaixo: o NDP200 é documentado abaixo de 1 mW, enquanto o reconhecimento de imagem no NDP250 é documentado abaixo de 30 mW. Não foi encontrada uma matriz pública ampla de YOLO. |

## Amlogic: duas gerações de NPU, não uma

A Amlogic merece uma explicação mais detalhada porque informações na web frequentemente misturam produtos incompatíveis. A empresa tem um caminho mais antigo baseado em VeriSilicon e um caminho proprietário ADLA mais recente. Eles não usam o mesmo compilador, artefato ou runtime.

| Geração | Chips representativos | NPU e toolchain | Artefato compilado | Situação prática |
|---|---|---|---|---|
| Legada | A311D e S905D3, comuns nas Khadas VIM3/VIM3L | VeriSilicon Vivante VIPNano-QI, toolkit Acuity, KSNN e `aml_npu_sdk` mais antigo | `.nb` dentro de um diretório de saída `nbg_unify` | Placas e demos existentes; integração legada separada |
| ADLA2 atual | C308L/C302X, S928X, A311D2, T968D4, S905X5/S905D5 e C302X2 | Amlogic ADLA2, AMLNN Toolkit e NNSDK2 | `.adla` específico para o alvo; W8A8 ou W8A16 | Pilha atual focada em inteiros |
| ADLA3 atual | A311Y3, C305X2 e A123X | Amlogic ADLA3, AMLNN Toolkit e NNSDK2 | `.adla` específico para o alvo; W4A8, W8A8, W4A16, W8A16 ou W16A16 | Adiciona recursos nativos INT4, FP16 e BF16 |

A [ficha técnica do A311D](https://dl.khadas.com/products/vim3/datasheet/a311d-datasheet.pdf) identifica a NPU original de 5-TOPS INT8. A antiga [página de aplicações Khadas VIM3](https://docs.khadas.com/products/sbc/vim3/npu/npu-app) e a [visão geral da NPU](https://docs.khadas.com/products/sbc/vim3/npu/start) documentam DenseNet CTC, MTCNN, RetinaFace, YOLOFace, YOLOv2, YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv7-tiny e YOLOv8n; a visão geral também documenta YOLOv8n-pose e marca FaceNet como obsoleto. Esses exemplos são reais, mas evidenciam o antigo ecossistema Acuity `.nb`, não os chips ADLA atuais. A311D não é A311D2, e C305X não é C305X2.

Há também uma segunda armadilha de hardware. O [guia de revisão da Khadas VIM4](https://docs.khadas.com/products/sbc/vim4/configurations/identify-version) diz que a placa original V12/A311D2 revisão B não tem NPU; a NPU anunciada de 3.2-TOPS aparece na V13A e nas placas posteriores que usam a peça A311D2-N0D revisão C. O nome de um produto, por si só, não basta para escolher um dispositivo de teste.

### A pilha AMLNN e ADLA atual

O [AMLNN Toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/7d3cc9a36179b756ce79c953c14d43acab1f2af5) atual abrange conversão de modelos, quantização, compilação, inferência e profiling. Seu [guia do usuário de maio de 2026](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/02_Amlogic_NPU_User_Guide_V0.1.pdf), fixado em uma versão específica, documenta importadores para ONNX, TFLite de ponto flutuante ou quantizado, TorchScript `.pt` e PyTorch 2 ExportedProgram/PT2. Os caminhos TensorFlow, Paddle e Keras aparecem como planejados no guia detalhado, embora a visão geral do repositório use termos mais amplos. O compilador gera `.adla`. O deploy Python usa AMLNN ou `amlnn_edge_toolkit_lite`; o C/C++ nativo se vincula a `libnnsdk.so` por meio de `nnsdk2.h`. O desenvolvimento no host pode usar ADB com `nnserver`; o runtime também tem como alvo Android, Buildroot, Yocto e alguns ambientes Debian/Armbian.

Os identificadores de plataforma publicados pelo toolkit atualmente incluem:

| ID do alvo | Plataforma |
|---:|---|
| `001` | C308L / C302X |
| `002` | S928X |
| `003` | A311D2 |
| `004` | T968D4 |
| `005` | S905X5 / S905D5 |
| `006` | C302X2 |
| `007` | A311Y3 |
| `008` | C305X2 / A123X nos documentos detalhados e na matriz de modelos |

Esse campo de alvo não é decorativo. A Amlogic exige correspondência entre `.adla`, `nnsdk2.h`, `libnnsdk.so`, BSP e geração do compilador/runtime. O [Quick Start](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/01_Amlogic_NPU_Quick_Start_guide_V0.1.pdf) fixado em uma versão específica exige driver ADLA 2.0.2 ou superior, NNSDK 3.0.0 ou superior e NNSDK2 1.0.0 ou superior; o [guia de deploy Android](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/06_Amlogic_NPU_Android_Deployment_Guide_V0.1.pdf) concretiza a relação entre biblioteca, header, driver e BSP. Trate `.adla` como um artefato de build vinculado à ABI, não como um modelo portável.

A quantização também depende da geração da NPU. Os alvos ADLA2 001 a 006 documentam compilação W8A8 e W8A16. Os alvos ADLA3 007 e 008 adicionam as combinações W4A8, W4A16, W8A8, W8A16 e W16A16, com suporte nativo a INT4, FP16 e BF16 no [guia de operadores](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/05_Amlogic_NPU_Support_Operator_List_V0.1.pdf). A Amlogic recomenda de 200 a 500 amostras representativas para calibração. A opção de dados aleatórios é explicitamente voltada a testes de desempenho, não a uma quantização que preserve a acurácia.

### Cobertura de modelos documentada pela Amlogic

O [model playground Amlogic fixado em uma versão](https://github.com/Amlogic-NN/amlnn-model-playground/blob/03ca63ef20c4f1560722de26ab910826884f334a/README.md) é uma evidência muito mais forte do que uma afirmação genérica de suporte a ONNX. Sua matriz publicada de suporte/exemplos cita A311D2, S905X5, A311Y3, C305X2 e A123X. A aceitação pelo compilador dos outros IDs de alvo não comprova que toda essa lista de modelos tenha sido validada neles.

| Tarefa | Modelos documentados |
|---|---|
| Classificação | MobileNetV2 e ResNet50-v2; DINO no ADLA3 |
| Detecção de objetos | PP-YOLOE, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv10, YOLO11, YOLOE, YOLO-World, YOLOX e detecção de QR |
| Face e gestos | RetinaFace e Gesture Recognition |
| Segmentação | DeepLabV3, PP-LiteSeg, YOLOv5-seg e YOLOv8-seg |
| Detecção orientada | YOLOv8 OBB |
| Pose | BlazePose detector/landmark e YOLOv8 pose |
| OCR e fala | LPRNet, variantes PaddleOCR, Whisper Tiny e SenseVoice em alguns chips mais novos |
| Transformers mais novos e trabalho multimodal | DETR, MobileSAM, CLIP/MobileCLIP e alguns exemplos LLM/VLM de baixa precisão em plataformas mais novas |

O mesmo repositório publica os seguintes números de runtime de modelo W8A8. São medições do fornecedor da rede neural na NPU, não de pipelines completos de câmera.

| Modelo e entrada | S905X5 | A311D2 | A311Y3 |
|---|---:|---:|---:|
| YOLOv8n, 640 x 640 | 101.72 FPS | 95.14 FPS | 191.06 FPS |
| YOLOv8s, 640 x 640 | 42.33 FPS | 42.77 FPS | 83.08 FPS |
| YOLOv8m, 640 x 640 | 19.67 FPS | 19.82 FPS | 35.30 FPS |
| YOLOv8l, 640 x 640 | 10.53 FPS | 10.12 FPS | 18.37 FPS |
| YOLO11n, 640 x 640 | 41.14 FPS | 41.48 FPS | 62.24 FPS |

As ressalvas são especialmente importantes. O guia de operadores da Amlogic posiciona NMS em software tanto no ADLA2 quanto no ADLA3. O README define esses resultados como medições nativas de runtime de modelo na NPU e exclui pré-processamento e pós-processamento; não informa se todas as transferências de memória estão incluídas. As linhas de acurácia YOLO usam subconjuntos COCO de 300 imagens, em vez do conjunto de validação completo, enquanto as linhas de classificação usam ImageNet `val1000`. As tabelas também mostram uma latência YOLOv8n no A311Y3 diferente da tabela de FPS, sob condições que o repositório não reconcilia. Clock, modo de potência, refrigeração, estado térmico estável, versão exata do SDK/BSP e duração do teste não são informados. Use os números para entender uma pilha de fornecedor, não para classificá-la frente a outra.

### Por que a Amlogic é estrategicamente interessante

A Amlogic combina quatro características incomuns: uma grande presença em SoCs embarcados, uma pilha de compilação que se tornou pública recentemente, uma matriz atual de modelos que se sobrepõe bastante às tarefas do LibreYOLO e pouca integração de primeira classe em bibliotecas de treinamento populares. Os repositórios atuais também são novos: apareceram publicamente no fim de 2025 e em 2026, com manuais substanciais datados de maio a julho de 2026. Isso cria uma oportunidade real para chegar primeiro e também um risco correspondente de instabilidade da API.

Uma integração limpa deve usar a exportação ONNX determinística do LibreYOLO como entrada do compilador, exigir imagens de calibração representativas para builds de baixa precisão, produzir `.adla` com um manifesto de metadados e carregar o modelo por um adaptador NNSDK2. O A311D legado deve usar um alvo `.nb` claramente diferente, pois apresentar `.nb` e `.adla` como um único backend causaria falhas silenciosas de compatibilidade. Os repositórios da Amlogic têm licenças Apache-2.0 no nível superior, mas os direitos de redistribuição dos wheels, bibliotecas compartilhadas, `nnserver` e binários Android AAR incluídos devem ser confirmados diretamente antes de o LibreYOLO espelhá-los.

## Os ecossistemas públicos de deploy mais fortes

Atualmente, as empresas a seguir oferecem a combinação mais clara de hardware disponível, material técnico público e evidências de modelos identificados. Isso não significa que sejam sempre mais rápidas. Significa que é mais fácil avaliar suas afirmações de compatibilidade antes de comprar hardware.

### Hailo

A Hailo é um exemplo de referência de ecossistema dedicado a aceleradores de visão. Os modelos são analisados em um Hailo Archive, otimizados e quantizados com imagens representativas e então compilados em um arquivo Hailo Executable Format (`.hef`). As aplicações carregam esse HEF pelo HailoRT.

O [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) contém receitas, modelos pré-treinados, configurações de pós-processamento e dados de desempenho para detecção, segmentação, classificação, pose e outras tarefas. A cobertura YOLO inclui YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 e YOLO26, além de YOLOX, DAMO-YOLO, SSD e EfficientDet. A [tabela de detecção de objetos Hailo-8](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.19.0/docs/public_models/HAILO8/HAILO8_object_detection.rst), fixada em uma versão, é uma fonte de compatibilidade melhor do que uma página genérica de produto, enquanto a [aplicação independente](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/README.md) documenta pipelines YOLO implantáveis.

Há uma fronteira importante entre versões. Hailo-8 e Hailo-8L usam a linha antiga Model Zoo 2.x, Dataflow Compiler 3.x e HailoRT 4.x, enquanto Hailo-10 e Hailo-15 usam a geração atual 5.x. Receitas, comportamento do parser e HEFs não são intercambiáveis só porque os dispositivos têm o nome Hailo.

O Hailo-15 também tem uma camada de aplicação distinta. Ele usa o Vision Processor Software Package e Hailo Media Library, em vez do caminho de host no estilo TAPPAS do Hailo-8/10H. O repositório de referência público [`hailo-camera-apps`](https://github.com/hailo-ai/hailo-camera-apps) foi arquivado em 3 de maio de 2026, enquanto [`hailo-apps-core`](https://github.com/hailo-ai/hailo-apps-core) continua sendo um framework de aplicação público separado. O arquivamento de um repositório não comprova o fim de vida do produto, mas projetos Hailo-15 devem confirmar o pacote de aplicação atual no Developer Zone.

O [guia de deploy Hailo](/docs/export/hailo) do LibreYOLO termina intencionalmente na transferência de um ONNX estático, pois o compilador proprietário não pode ser incluído como dependência Python. Esse é o limite honesto entre produzir um grafo adequado e afirmar que existe um HEF testado.

### Rockchip

A Rockchip tem uma das maiores presenças em Linux embarcado entre makers e no setor comercial, especialmente por meio de placas RK3588, RK3576 e RK356x. O [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) público converte e quantiza modelos em um host x86 Linux, e o RKNN Runtime ou Toolkit-Lite executa o arquivo `.rknn` resultante na placa.

O [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) é excepcionalmente explícito sobre chips, famílias de modelos e precisão. Sua tabela atual lista caminhos FP16 e INT8 para YOLOv5/6/7/8/10/11, YOLOX, PP-YOLOE, YOLO-World e variantes de segmentação YOLOv5/v8; YOLOv8 OBB e YOLOv8 pose aparecem somente como INT8. Também cobre OCR, face e outras redes de segmentação.

O LibreYOLO já tem um [exportador RKNN](/docs/export/rknn) direto e conservador. Ele valida quatro variantes exatas de detecção no RK3588, pode comparar o simulador do compilador com o ONNX Runtime e rejeita famílias não validadas, em vez de equiparar uma compilação bem-sucedida a previsões corretas. Essa afirmação de suporte restrita é um bom modelo para todo futuro backend de NPU.

### Axelera AI

A Axelera AI, cujo nome é frequentemente escrito errado como "Accelera", desenvolve o AIPU Metis e o vende em produtos M.2, PCIe e com vários chips. Metis é a família que pode ser encomendada publicamente; Europa e Titania são produtos anunciados para o portfólio, então sua disponibilidade não deve ser reduzida a um único status. O [Voyager SDK está disponível publicamente no GitHub](https://github.com/axelera-ai-hub/voyager-sdk) e gerencia seleção de modelos, compilação, quantização, execução e pipelines de aplicação; o suporte ao cliente ainda exige conta.

O [Voyager Model Zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo/) público está entre os mais informativos do mercado. Ele lista variante do modelo, tamanho de entrada, precisão, acurácia e desempenho medido para YOLOv3, YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 n/s/m/l/x, YOLOX, YOLO-NAS, OBB, pose e segmentação, além de muitos modelos de classificação e predição densa. Publicar a perda de quantização e a acurácia ao lado da velocidade é mais útil do que publicar apenas um valor de TOPS de pico.

Os nomes dos artefatos mudaram com a geração da API. O fluxo de compilação alfa do [Pipeline Builder](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) gera um modelo `.axm` e pode empacotar um pipeline portável como `.axe`. A [documentação do pipeline clássico](https://docs.axelera.ai/sdk/reference/pipeline/model-formats/) descreve `.axmodel` com metadados e manifesto do modelo. Uma integração precisa detectar a geração instalada do Voyager, em vez de fixar uma única extensão.

### Qualcomm

A Qualcomm oferece enorme alcance de deploy, mas "NPU Qualcomm" esconde várias interfaces. Desenvolvedores encontram o Hexagon HTP por meio de QAIRT/QNN, o fluxo SNPE mais antigo, serviços de compilação Qualcomm AI Hub, LiteRT ou o provedor de execução QNN do ONNX Runtime, dependendo do dispositivo e da categoria do produto.

O repositório oficial [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) é uma forte evidência no nível do modelo. Ele publica pacotes otimizados de detecção/segmentação/pose YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26, além de YOLOX, YOLO-World, YOLOR, RF-DETR e muitos modelos de classificação, profundidade e pose. O AI Hub também oferece profiling específico por dispositivo, importante porque um modelo compilado para uma geração HTP não é automaticamente portável para outra.

O principal custo de integração é o tamanho da matriz: Android versus Linux embarcado, SKUs QCS sob encomenda versus Snapdragon de consumo, arquitetura HTP, versão QNN/QAIRT e esquema de quantização importam. As páginas atuais da própria Qualcomm também divergem sobre o status do [Dragonwing IQ-9075](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075): a página do produto marca Active e seu [EVK](https://www.qualcomm.com/developer/hardware/qualcomm-iq-9075-evaluation-kit-evk) pode ser avaliado, enquanto o [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) ainda marca IQ-9075 como Sampling e lista longevidade até 2038. O prefixo do SKU para pedido é QCS9075; a Qualcomm anuncia configurações de 50 e 100 dense-INT8-TOPS e suporte Ubuntu/Yocto. Confirme o status de produção para o SKU exato e registre esse SKU em uma integração LibreYOLO, em vez de gerar uma pasta genérica chamada "Qualcomm".

### DEEPX

Os aceleradores DX-M1/DX-M1M da DEEPX usam o DXNN SDK. O DX-COM compila e quantiza o modelo, o DX-RT o executa e o artefato de deploy usa o formato `.dxnn`. A empresa publica o [DX-AllSuite](https://github.com/DEEPX-AI/dx-all-suite), exemplos de aplicação no [DX-APP](https://github.com/DEEPX-AI/dx_app) e um grande [Model Zoo online](https://developer.deepx.ai/modelzoo/).

O catálogo público informava 354 entradas com DX-COM 2.4.0 e DX-RT 3.4.0 quando verificado em 15 de agosto de 2026. A cobertura documentada inclui YOLOv3 a YOLO11 e YOLO26, YOLOX, SSD, EfficientDet, NanoDet, DAMO-YOLO, segmentação YOLO, pose e caixas orientadas. A DEEPX é um alvo de integração especialmente plausível porque os limites entre compilador/runtime e o artefato de deploy têm nomes claros, e a cobertura pública de visão é ampla.

## SoCs industriais são vários ecossistemas, não um só

Fornecedores industriais costumam ter ciclos de vida de produto mais longos e melhor integração de câmera, segurança e tempo real do que fabricantes de placas maker. Suas pilhas de IA podem ser mais complicadas porque um fornecedor pode disponibilizar simultaneamente várias arquiteturas NPU sem relação entre si.

### Texas Instruments

Os processadores atuais da TI para edge AI incluem AM62A, AM67A, AM68A, AM69A e dispositivos TDA4 relacionados. O caminho de software combina Processor SDK Linux, compilador/runtime TIDL, [Edge AI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) e [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo). O TIDL pode se integrar ao ONNX Runtime, TensorFlow Lite e outros runtimes de aplicação, enquanto delega subgrafos compatíveis.

A TI publica bem mais do que uma afirmação de importação de framework: o zoo contém modelos convertidos e metadados de desempenho para detecção de objetos, segmentação, pose, classificação, profundidade e outras tarefas. A TI também alerta que os artefatos do model zoo servem como pontos de partida para desenvolvimento, não como assets automaticamente prontos para produção. Essa ressalva valiosa costuma faltar nas comparações entre fornecedores.

A TI também lista o [TDA54-Q1](https://www.ti.com/product/TDA54-Q1) como Preview, com até quatro NPUs C7 e até 400 TOPS nesse componente; a família TDA5 mais ampla é anunciada com até 1,200 TOPS. Esses são números de produtos de uma nova geração divulgados pelo fornecedor, não autorização para transferir a validação de modelos TIDL AM6xA/TDA4 para o TDA54-Q1 antes que a TI publique uma matriz específica para esse alvo.

### NXP

A NXP exige pelo menos quatro descrições de backend:

| Alvo NXP | Acelerador | Caminho de compilador/runtime |
|---|---|---|
| i.MX 8M Plus | NPU VeriSilicon Vivante de 2.3-TOPS | eIQ com TIM-VX/VX delegate |
| i.MX 93 | Arm Ethos-U65 | TFLite quantizado mais Arm Vela |
| i.MX 95 | NPU NXP eIQ Neutron | Neutron Converter e runtime eIQ |
| Ara-1 / [Ara240](https://www.nxp.com/products/ARA240) | NPU discreta derivada da Kinara; Ara240 anuncia até 40 eTOPS | Ara SDK, integrado à oferta mais ampla eIQ |

O [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) contém assets ou receitas para YOLOv4-tiny, YOLOv8, NanoDet, CenterNet, FastestDet, SSD Lite, YOLACT e outros modelos; a entrada YOLOv5 foi adicionada somente como documentação. Uma entrada validada para um backend não comprova os outros três. As [orientações de exportação YOLO para plataformas i.MX da própria NXP](https://community.nxp.com/t5/i-MX-Processors-Knowledge-Base/Exporting-YOLO-Models-for-NXP-i-MX-Platforms/ta-p/2381361) ilustram esse problema de conversão específica por alvo. A NXP informa que o [eIQ Toolkit monolítico parou de receber atualizações após a versão 1.17 no terceiro trimestre de 2025](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/eIQ-FAQ/ta-p/1099741); os fluxos atuais usam pacotes independentes, como eIQ Neutron SDK.

A NXP [concluiu a aquisição da Kinara em outubro de 2025](https://media.nxp.com/news-releases/news-release-details/nxp-completes-acquisitions-aviva-links-and-kinara-advance/). A [ficha técnica Ara240](https://www.nxp.com/docs/en/fact-sheet/ARA240DNPUFS.pdf) anuncia até 40 eTOPS do fornecedor e relata 313 imagens por segundo para YOLOv8n. A NXP lista Ara SDK e eIQ Toolkit na página do produto, mas isso não estabelece intercâmbio de artefatos com o caminho separado [i.MX 95 Neutron](https://eiq.nxp.com/learning-hub/convQuant/neutron.html). O [módulo M.2 de 16 GB](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT) está ativo, enquanto a opção USB está em pré-produção. A NXP define o "e" em eTOPS como "equivalent" (equivalente), não "effective" (efetivo); essa métrica não corresponde à de TOPS dense-INT8 de outro fornecedor e não deve ser comparada diretamente.

### STMicroelectronics

Os [dispositivos STM32N6x7](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html), incluindo STM32N657 e STM32N647, levam um acelerador Neural-ART de 600-GOPS a um produto da classe MCU; a linha de uso geral N6x5 não inclui esse acelerador. O [STM32Cube AI Studio](https://www.st.com/en/development-tools/stedgeai-cubeai.html) e o ST Edge AI Core analisam e otimizam modelos importados, enquanto o [STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) oferece fluxos de deploy, otimização, benchmark e exemplos.

A [visão geral atual de detecção de objetos](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/docs/README_OVERVIEW.md) documenta Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 e ST-YOLOX, além de serviços de classificação, pose e segmentação. Essa não é a mesma classe de desempenho de uma placa PCIe de 200-TOPS. O interesse está em colocar entrada de câmera, inferência e controle em limites profundamente embarcados de potência e memória.

### Renesas

Os processadores RZ/V da Renesas integram aceleradores DRP-AI. O software evoluiu de DRP-AI Translator e DRP-AI TVM para [RUHMI](https://www.renesas.com/en/software-tool/ruhmi-framework), com tecnologia MERA da EdgeCortix. O repositório aberto [RZ/V DRP-AI TVM](https://github.com/renesas-rz/rzv_drp-ai_tvm) e a [lista de validação RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) citam variantes YOLOv5, YOLOv8, YOLO11 e YOLO26 n/s/m, YOLOX, pose e redes de segmentação, além de modelos de classificação.

A Renesas é um alvo confiável para robótica e indústria, mas a placa exata, geração DRP-AI, versão do translator e pós-processamento na CPU precisam ser registrados para garantir reprodutibilidade.

## Sensores inteligentes e processadores voltados a câmeras

### Sony IMX500

O IMX500 da Sony não é um processador de aplicação comum. Ele empilha processamento de imagem e IA para que a inferência ocorra no sensor, reduzindo a quantidade de dados de imagem que precisa sair da câmera. A Raspberry Pi AI Camera torna essa arquitetura especialmente acessível.

O repositório oficial de modelos [Raspberry Pi IMX500](https://github.com/raspberrypi/imx500-models) publica modelos empacotados YOLOv8n, YOLO11n, EfficientDet Lite0, NanoDet+ e SSD MobileNetV2 FPN Lite. A [documentação AI Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) explica conversão, empacotamento e pós-processamento no sensor. A unidade de deploy é um pacote RPK, não um arquivo ONNX genérico, e a memória do sensor e as restrições de operadores são limites centrais do projeto. A [página de produto](https://www.raspberrypi.com/products/ai-camera/) da Raspberry Pi diz que esta câmera continuará em produção até pelo menos janeiro de 2028.

### Ambarella

Os processadores CVflow da Ambarella estão consolidados em câmeras, drones e visão automotiva. As famílias atuais incluem CV72/CV75 para câmeras, CV5/CV52 para sistemas de visão de maior desempenho e dispositivos automotivos CV3-AD. A plataforma de desenvolvedor Cooper e o fluxo de compilação/runtime CVflow formam a pilha de software identificada publicamente.

O [model garden público para desenvolvedores](https://www.ambarella.com/developer/model-garden/) cita YOLOX, RTMDet, DeepLabV3+, TopFormer, OWL-ViT e modelos multimodais. Porém, a documentação detalhada do compilador e os downloads continuam voltados a parceiros. Uma integração Ambarella deve começar por uma relação com fornecedor ou OEM, não pela suposição de que existe um pacote pip público.

### Synaptics Astra

A Synaptics tem três alvos relevantes. Os sistemas Astra SL1600/SL1680 usam o [toolkit SyNAP](https://developer.synaptics.com/docs/synap/introduction), que converte um modelo de origem e uma descrição YAML em um pacote `model.synap`. Os dispositivos SL2611/13/15/17/19 mais novos usam a [plataforma Torq](https://developer.synaptics.com/docs/torq/introduction), um fluxo baseado em IREE/MLIR que produz `.vmfb`. A [série SR100](https://developer.synaptics.com/docs/sr/introduction-sr) é uma família MCU separada, baseada em Cortex-M55 e Ethos-U55, com SDK próprio. Esses artefatos e APIs não são intercambiáveis.

O [tutorial oficial de benchmark YOLO](https://developer.synaptics.com/docs/sl/tutorials/vision/benchmark-yolo) é especialmente concreto: exporta YOLOv8 para TFLite, faz calibração UINT8 assimétrica no SyNAP, compila para SL1680 e então executa `synap_cli` e a aplicação de detecção de objetos. Ele comprova YOLOv8n/v8s no SL1680 e descreve suporte SyNAP até YOLO11. Um [guia separado de detecção de objetos SL261x](https://developer.synaptics.com/docs/sl/sl2600/getting-started/object-detection) executa YOLOv8 com Torq `.vmfb`. São exemplos oficiais específicos para cada alvo, não uma afirmação de que todos os grafos YOLO têm suporte nas três famílias.

### MediaTek Genio

A MediaTek merece uma entrada completa porque seu [IoT AI Hub](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) atual agora expõe a matriz de hardware/software, pacotes de modelos e tabelas de benchmark que pesquisas antigas de mercado não conseguiam auditar. Genio 360/360P/420/520/720 usa NeuroPilot 8 com MDLA 5.3; Genio 510/700 usa NP6 com MDLA 3.0; Genio 1200 usa NP6 com MDLA 2.0. A MediaTek afirma que a geração NeuroPilot e MDLA, conjunto de operadores, compilador e runtime ficam vinculados por versão durante a vida útil de cada SoC.

O caminho de IA analítica é centrado em TFLite:

```text
PyTorch model
    -> NeuroPilot Converter and representative calibration
    -> quantized .tflite
    -> version-matched ncc-tflite compiler
    -> chip-generation-specific .dla
    -> Neuron Runtime on MDLA
```

Os produtos Genio mais novos também oferecem delegação online LiteRT e um caminho ONNX Runtime CPU/NPU em sistemas operacionais compatíveis. Esses modos de execução são diferentes do `.dla` offline. O [guia de arquitetura de software](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/software_architecture.html) e a [matriz de recursos](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) deixam a distinção explícita.

A [tabela oficial de modelos analíticos](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical.html) publica entradas de benchmark e guias de conversão para YOLOv5s e YOLOv8s em várias gerações MDLA. Ela declara explicitamente que não distribui artefatos YOLO pré-convertidos por causa das restrições AGPL-3.0. Suas medições offline Quant8, 640 x 640, incluem 5.35 ms e 8.04 ms, respectivamente, no Genio 720. A [página do modelo YOLOv8s](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical/YOLOv8s.html) publica tensores de entrada/saída, tempos específicos por backend e alertas sobre operações personalizadas da MediaTek. Esses são benchmarks do fornecedor, não resultados de câmera de ponta a ponta.

O acesso é o fator limitante. A documentação pública Yocto é detalhada, mas o bundle atual NP8 all-in-one do conversor/compilador e boa parte do material Android aparecem como recursos NDA ou para clientes diretos. Uma conta normal de desenvolvedor não concede o mesmo acesso que uma conta MediaTek Online de cliente. Portanto, o LibreYOLO deve tratar Genio como tecnicamente comprovado, mas dependente de parceria para uma integração reproduzível de compilador com modelos próprios.

## Ecossistemas de edge AI com foco na China

Várias das plataformas de visão mais capazes e acessíveis de baixo custo são pouco representadas em listas de mercado em inglês.

### D-Robotics e plataformas BPU derivadas da Horizon

A D-Robotics mantém o ecossistema de placas de desenvolvedor RDK em torno de aceleradores BPU usados em produtos RDK X3, X5, Ultra e da série S100 mais recente. O branch `rdk_x5` atual do [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo) documenta caminhos de YOLOv5/v5u, YOLOv8/9/10/11/12/13/26, YOLOE, YOLO-World, detecção, segmentação, pose, classificação, OCR e CLIP para RDK X5. O X3 usa um branch separado, enquanto a entrega atual da série S está no branch [`rdk_s`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s) do zoo principal; o repositório antigo [`rdk_model_zoo_s`](https://github.com/D-Robotics/rdk_model_zoo_s) contém demos históricas. Portanto, a lista moderna do X5 não comprova que todos os modelos foram validados no X3, Ultra ou S100.

OpenExplorer/Algorithm Toolchain importa ONNX ou Caffe para PTQ e oferece fluxos QAT voltados a PyTorch. O deploy varia por plataforma: o RDK X5 atual usa `.bin` via `hbm_runtime` sobre `libdnn`, enquanto o fluxo principal atual `rdk_s` usa `.hbm` via uma API `hbm_runtime` de mesmo nome sobre `libhbucp`; o X3 mantém seu branch e interfaces de inferência antigos. Materiais antigos de `rdk_model_zoo_s` também descrevem caminhos históricos `.bin` e `.hbm`, então esses artefatos precisam continuar associados ao branch e toolchain correspondentes. Operadores sem suporte podem cair para CPU. Os exemplos públicos são bons, mas o pareamento de versões entre RDK OS, firmware da placa, toolchain e binário do modelo continua fazendo parte do contrato de compatibilidade.

### AXERA

As famílias AX650/AX630/AX620 da AXERA aparecem em câmeras de IA compactas e placas como a linha MaixCAM da Sipeed. O Pulsar2 importa ONNX, faz calibração e análise de precisão e gera `.axmodel`; o AXEngine executa o artefato.

Os [exemplos oficiais AXERA](https://github.com/AXERA-TECH/ax-samples) cobrem YOLOv5/6/7/8/9/10/11/13/26, YOLOX e YOLO-World, com suporte a tarefas e chips variando por plataforma. Nos alvos compatíveis, os exemplos atuais de YOLO26 incluem detecção, pose, segmentação e OBB. Os [exemplos de conversão Pulsar2](https://pulsar2-docs.readthedocs.io/en/latest/appendix/model_convert_examples.html) mostram o trabalho real: nomes exatos de tensores, cortes de saída, transformações de layout, arquivos de calibração e configurações do hardware alvo.

### SOPHGO

O TPU-MLIR da SOPHGO é uma das pilhas de compilador mais atraentes para integração independente porque o próprio compilador é de código aberto. Ele importa ONNX, PyTorch, TFLite e Caffe, reduz e quantiza grafos e produz `.bmodel` para alvos BM ou `.cvimodel` para hardware CV18xx.

O [projeto TPU-MLIR](https://github.com/sophgo/tpu-mlir) oferece fluxos FP32, BF16, FP16 e INT8 conforme o alvo. A [coleção de demos SOPHON](https://github.com/sophgo/sophon-demo/blob/release/README_EN.md) cita YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, variantes OBB/segmentação, SSD, CenterNet, RetinaFace, SAM/SAM2 e OCR. Como sempre, uma demo em BM1684X não valida o mesmo artefato em BM1688 ou CV18xx.

### Huawei Ascend

A linha de inferência edge da Huawei inclui silício Ascend 310/310P/310B nos produtos Atlas 200I e 300I. O CANN fornece compilador ATC, runtime AscendCL e kernels de operadores específicos por chip; o ATC transforma ONNX e outras representações de origem em um modelo offline `.om`.

A documentação atual do [Atlas 200I DK A2](https://www.hiascend.com/en/hardware/developer-kit-a2/specification) e os exemplos CANN oferecem um caminho YOLOv5 `.om` para Ascend 310B. O [Ascend ModelZoo](https://github.com/Ascend/modelzoo) mais amplo e antigo contém YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN, pose e modelos de segmentação, mas não deve ser apresentado como se cada entrada tivesse sido revalidada no 310B. CANN, kernels, firmware e SoC precisam corresponder. Um `.om` para Ascend310P não é um artefato Ascend genérico.

### Cambricon

Os aceleradores MLU da Cambricon usam Neuware e o engine de inferência MagicMind. O repositório público [MagicMind Cloud](https://github.com/Cambricon/magicmind_cloud) está fixado em MagicMind 1.7 e em uma matriz MLU370-X4/S4; é evidência histórica de compatibilidade, não uma matriz do SDK atual ou validação MLU270. Esse repositório documenta caminhos PyTorch, ONNX, Caffe e Paddle; o suporte ao framework TensorFlow foi removido na versão 1.7, embora exemplos TensorFlow históricos continuem lá. O [portal atual para desenvolvedores da série 3](https://developer.cambricon.com/index/document/index/classid/3.html) lista versões gerais posteriores de SDK, portanto o repositório de exemplos público e a pilha comercial atual não devem ser apresentados como uma única versão.

O repositório oficial de modelos na nuvem [MagicMind](https://github.com/Cambricon/magicmind_cloud) publica uma matriz MLU370-X4/S4 especialmente direta. Ela identifica exemplos YOLOv3, Tiny-YOLOv3, YOLOv4, YOLOv5, YOLOv7, YOLOv8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN, DeepLab e UNet, incluindo se existem exemplos C++ ou Python. A evidência é forte; a principal barreira é o acesso ao SDK e container pelos canais da Cambricon.

### Canaan/Kendryte

Os chips KPU atuais K230/K230D e legados K210/K510 da Kendryte são importantes na faixa de placas e câmeras inteligentes de baixo custo. O compilador [nncase](https://github.com/kendryte/nncase) importa ONNX ou TFLite, usa dados de calibração para conversão em ponto fixo, simula o resultado no host e gera `.kmodel`.

O [guia de desenvolvimento nncase K230](https://github.com/kendryte/k230_docs/blob/main/en/01_software/board/ai/K230_nncase_Development_Guide.md) oficial explica a compilação, simulação e execução de YOLOv5s. O [catálogo separado de demos de IA](https://github.com/kendryte/k230_docs/blob/main/en/02_applications/ai_demos/K230_AI_Demo_Introduction.md) inclui artefatos de detecção, segmentação e pose YOLOv8n, enquanto o [changelog atual do CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) adiciona exemplos otimizados YOLOv8/11/26 de classificação, detecção, segmentação, OBB e pose. O [resultado YOLOv5s](https://github.com/kendryte/k230_docs/blob/main/en/00_hardware/K230_datasheet.md) da ficha técnica é o benchmark publicado real; as versões do compilador e do plug-in KPU binário precisam corresponder ao SDK da placa.

### Allwinner

O V853 da Allwinner combina hardware multimídia voltado a câmeras com uma NPU Vivante de 1-TOPS. O [guia NPU oficial em inglês](https://docs.aw-ol.com/v853/en/npu/dev_npu/) descreve importação Acuity/Pegasus, quantização, verificação e deploy por `viplite`. As páginas de modelos da Allwinner e o [guia YOLOv5](https://docs.aw-ol.com/v853/npu/npu_yolov5/) citam YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet e redes de face/pessoa.

Isso é suporte real a visão computacional, mas continua vinculado à toolchain Vivante/Tina Linux mais antiga e a downloads do SDK que dependem de conta. Ele pertence ao panorama de mercado com essa limitação explícita.

## Plataformas coreanas, taiwanesas e chinesas pouco conhecidas

Listas em inglês frequentemente passam direto de Rockchip para NVIDIA e deixam de fora uma segunda categoria de silício real e documentado. Alguns desses ecossistemas têm matrizes YOLO atuais mais amplas do que startups ocidentais mais conhecidas.

### Mobilint

A Mobilint, fornecedora coreana de aceleradores, vende REGULUS de baixo consumo e ARIES MLA100 de maior throughput. Seu qb SDK aceita entradas PyTorch, TensorFlow, TFLite, ONNX e orientadas a Keras, quantiza para INT8 e gera o artefato compilado `.mxq`. A [matriz de visão](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) pública cita detecção YOLOv3/v5/v7/v8/v9/v10/11/12/26, além de variantes de segmentação, pose e OBB YOLO. A [página ARIES](https://www.mobilint.com/aries/mla100) também publica resultados específicos de YOLO11s e YOLO26m. É uma forte evidência de integração, embora o acesso ao SDK/hardware continue comercial.

### Sunplus

O SP7350/C3V da Sunplus usa uma pilha NPU derivada de Vivante, mas empacota-a de forma diferente da Allwinner ou da Amlogic legada. O fluxo público combina conversão Acuity, ambiente Docker NPU, runtime SNNF e saída `.nb`. O [guia oficial personalizado de YOLOv8](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) percorre conversão e deploy, em vez de apenas afirmar suporte ao framework. IP relacionado não torna o `.nb` portável para outro SoC baseado em Vivante.

### ESWIN Computing

A família RISC-V da ESWIN inclui EIC7700/EIC7700X e EIC7702/EIC7702X de dois dies, sendo que o EIC7700 aparece em placas disponíveis como Milk-V Megrez. ENNP contém EsQuant, o compilador EsAAC atual, geração de dados de referência, simulação e runtime ESSDK, com saída `.model`; materiais antigos de ENNP usavam o nome `ennc-compile`. A [página atual do EsAAC](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) documenta entrada ONNX, então outros frameworks precisam ser exportados ou convertidos para essa IR suportada. A visão geral do [ENNP hospedada pela Milk-V](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/introduction) e o [tutorial YOLOv3](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3) estabelecem um caminho público completo, mas não uma matriz ampla e atual de modelos/acurácia.

### Nuvoton M55M1

O M55M1 da Nuvoton é um projeto da classe MCU Cortex-M55 e Ethos-U55-256, não um processador de aplicação Linux. NuEdgeWise/NuML e Arm Vela preparam modelos TFLite Micro quantizados. O repositório público [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) cita YOLOv8-nano, YOLOX-nano, YOLO Fastest v1.1, SSD-MobileNet FPNLite e vários classificadores. Esses exemplos de redes pequenas são evidências confiáveis para a classe de memória/potência do dispositivo, não para variantes YOLO padrão de 640 pixels.

### Rebellions e FuriosaAI

A linha ATOM da Rebellions usa o RBLN SDK e artefatos compilados `.rbln`. Sua [versão oficial de suporte](https://docs.rbln.ai/v0.8.2/supports/release_note.html) cita YOLOv3 tiny/full/SPP, YOLOv5 n a x, YOLOv6 n a l, variantes YOLOv7 e YOLOv8 n a x. A [matriz atual de suporte de placas](https://docs.rbln.ai/latest/supports/version_matrix.html) marca ATOM CA02 e ATOM+ CA12 como fim de vida, enquanto ATOM+ CA22 e ATOM-Max CA25 estão ativos. ATOM-Lite CA21 tem página de produto, mas não aparece nessa matriz do SDK, então sua situação atual na toolchain não está clara. A documentação é pública, mas o wheel do compilador exige credenciais do Rebellions Portal.

A FuriosaAI precisa ser separada por geração. Warboy é o acelerador mais antigo para visão, com artefatos INT8 `.enf` e um [model zoo oficial](https://developer.furiosa.ai/furiosa-models/latest/) que contém SSD, YOLOv5M/L e YOLOv7-w6-pose. RNGD usa uma [pilha `.fxb`](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html) diferente, voltada principalmente a cargas de trabalho LLM/VLM. O [roadmap atual](https://developer.furiosa.ai/latest/en/overview/roadmap.html) da Furiosa registra suporte a visão YOLOv8m concluído no quarto trimestre de 2024, mas suas superfícies públicas atuais de modelos suportados e desempenho continuam centradas em LLM/VLM e não expõem uma matriz detalhada de acurácia/desempenho para YOLOv8m. Isso é evidência de uma geração lançada, não uma afirmação de compatibilidade com Warboy.

### Realtek, Telechips, T-Head e SigmaStar

Essas quatro plataformas são reais, mas oferecem superfícies mais estreitas, antigas ou restritas para usar modelos próprios:

| Plataforma | Evidência pública reproduzível | Limitação a preservar |
|---|---|---|
| Realtek AmebaPro2 | O [SDK Arduino/FreeRTOS](https://github.com/Ameba-AIoT/ameba-arduino-pro2) empacota modelos YOLOv3/4/7-tiny, SCRFD e MobileFaceNet `.nb` | A conversão de modelos personalizados exige registrar interesse/contatar a Realtek |
| Telechips TOPST TCC7500 | O [projeto oficial TOPST](https://docs.topst.ai/blog/31) implantou YOLOv8s | A conversão UFLD v1/v2 falhou em camadas sem suporte; o artigo apenas propõe um caminho alternativo com divisão/pós-processamento. YOLOv4 e outro fluxo YOLOv8 aparecem em discussões da comunidade de suporte. Recursos completos do compilador/operadores costumam exigir autorização por e-mail |
| T-Head TH1520 | O [guia de aplicação](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) da Sipeed executa YOLOv5n/s com HHB e CSI-NN2/SHL | A pilha pública existe, mas a manutenção e a clareza de versões ficam atrás dos líderes atuais |
| SigmaStar SSU9383CM | A documentação atual estabelece APIs de runtime MI_IPU; materiais SGS_IPU antigos documentam `.sim` para `sgsimg.img` e um pós-processador antigo cita SSD e YOLOv1/2/3 | Não há evidência pública de que os artefatos do compilador mais antigo ou os caminhos de modelos citados se apliquem ao SSU9383CM |

### Visão inteligente HiSilicon não é Huawei Ascend

Os SoCs de câmera Hi3516CV610/DV500, Hi3519DV500 e Hi3403V100 da HiSilicon expõem um fluxo ATC para `.om` por runtimes NNN/SVP-NNN. O [model zoo oficial HiSpark](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) cita YOLOv3/4/5/6/7/8/9/10/11, segmentação/pose YOLO11, OBB/World/segmentação YOLOv8, OCR, profundidade e TinySAM em uma matriz específica por alvo, especialmente para Hi3403 e Hi3591P. Isso não comprova que cada entrada rode em todos os SoCs de visão inteligente listados acima. Algumas entradas são itens do roadmap, então exemplos lançados e suporte planejado precisam continuar separados, e o repositório informa que seus modelos ModelZoo fornecidos são somente para uso não comercial. O vocabulário comum ATC/`.om` não comprova compatibilidade de artefato ou runtime com a linha Ascend baseada em CANN.

## Empresas emergentes e especializadas em aceleradores

### MemryX

Os módulos MX3 da MemryX usam uma arquitetura dataflow e podem combinar vários chips. O [Neural Compiler](https://developer.memryx.com/tools/neural_compiler.html) aceita modelos ONNX, TensorFlow Lite, Keras e TensorFlow e gera um pacote DFP consumido pelo runtime do acelerador. Suas [APIs de runtime](https://developer.memryx.com/api/accelerator/accelerator.html) oferecem suporte a pipelines com vários modelos e streaming.

A documentação e os exemplos MemryX incluem pré-processamento e pós-processamento YOLO otimizados para detecção, segmentação e pose. As [notas de versão do SDK](https://developer.memryx.com/release_notes.html) citam explicitamente suporte a YOLOv10, YOLO11 e YOLO26. O material público é tecnicamente útil, embora não tenha uma única tabela simples de modelo/acurácia/dispositivo comparável à da Axelera.

### Kneron

Os produtos KL520/KL530/KL630/KL720/KL730 da Kneron abrangem pequenos aceleradores USB e embarcados. Kneron PLUS cobre o runtime de aplicação, enquanto a versão atual 0.33.1 do Model Toolchain documenta conversão, quantização, avaliação, simulação e compilação `.nef` para esses alvos. KL830 aparece em algumas [APIs de runtime PLUS](https://doc.kneron.com/docs/plus_c/introduction/run_examples/), mas não na [lista atual de alvos do compilador](https://doc.kneron.com/docs/toolchain/manual_1_overview/); não transfira uma afirmação genérica de compilação `.nef` ao KL830 sem confirmação do fornecedor.

O [exemplo YOLO oficial](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) demonstra o processo com Tiny-YOLOv3, e outros materiais Kneron cobrem treinamento e deploy de YOLOv5. A evidência é confiável, mas mais restrita do que as matrizes modernas amplas de YOLO da Hailo, Axelera, Rockchip ou DEEPX.

### SiMa.ai

O MLSoC da SiMa.ai e a [plataforma Modalix de produção de 50-TOPS](https://sima.ai/press-release/sima-ai-next-gen-platform-for-physical-ai-in-production/) usam o ambiente de software Palette. ModelSDK, compilador MLA e ModelExecutor cobrem importação, quantização, compilação, profiling e execução. As ferramentas oferecem fluxos voltados a INT8, INT16 e BF16, conforme a carga de trabalho e o produto.

As notas de versão da empresa citam pipelines validados YOLOv7, detecção/pose/segmentação YOLOv8, YOLOX e segmentação YOLOX, DETR, Mask R-CNN e EfficientDet. Uma limitação atual não deve ser omitida: as [notas da Palette SDK 2.1](https://docs.sima.ai/v2.1.2/pages/release_notes/2.1.html) dizem que QAT não funciona por causa de uma regressão de quantização Python/PT2E. A solução alternativa documentada é criar o ONNX anotado no SDK 2.0 e compilá-lo com 2.1. Acesso e aquisição continuam voltados a empresas.

O limite de deploy também está documentado: o [fluxo de compilação ModelSDK](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) gera um `.tar.gz` compilado e pode produzir um ELF executável, enquanto o [MPK Tool](https://docs.sima.ai/v2.1.1/pages/palette/mpk_tools.html) empacota um `.mpk` implantável. Essas saídas continuam vinculadas ao alvo MLSoC/Modalix e à versão Palette.

### EdgeCortix

SAKURA-II é um acelerador anunciado em 60-TOPS para visão e IA generativa, compilado pelo framework de software MERA. A EdgeCortix também fornece a tecnologia MERA para o fluxo RUHMI mais recente da Renesas.

O [material do produto SAKURA-II](https://www.edgecortix.com/en/edgecortix-sakura-ii-accelerator-brief) estabelece o hardware e o compilador, e o [hardware M.2/PCIe](https://www.edgecortix.com/en/hardware) é oferecido por consulta de teste ou pedido. Não foi encontrada uma matriz pública atual de compatibilidade YOLO suficientemente precisa. Essa ausência é informação útil para a compra: deve-se solicitar validação do modelo antes de se comprometer com o hardware.

### BrainChip

O Akida da BrainChip é um processador neuromórfico de domínio de eventos, não uma NPU convencional de tensores densos. O ambiente [MetaTF](https://brainchip.com/metatf-dev-tools/) consiste em pacotes Python instaláveis para criação de modelos, quantização de baixa precisão, conversão, simulação e execução no hardware. A BrainChip agora vende um [coprocessador AKD1500 em formato M.2 disponível para envio](https://brainchip.com/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/), além do hardware AKD1000 legado e IP Akida 2.

O [cartão de modelo Akida 2](https://brainchip.com/wp-content/uploads/2025/04/Akida-2-Model-Card-V1.1-Mar.25.pdf) atual cita um detector YOLOv2 AkidaNet0.5, CenterNet, AkidaUNet e redes de reconhecimento facial. Outros materiais de suporte incluem FOMO e cargas de trabalho baseadas em eventos. O Akida oferece suporte a pesos e ativações de precisão excepcionalmente baixa, mas um deploy bem-sucedido pode exigir conversão ciente da arquitetura, em vez de tratá-lo como mais um alvo genérico ONNX INT8. Um resultado de model zoo para AKD1000 ou Akida 2 não deve ser atribuído ao AKD1500 até que um relatório explícito de mapeamento/ajuste o valide.

### Blaize

A Blaize vende produtos Pathfinder e Xplorer baseados em P1600, com Picasso SDK, NetDeploy e AI Studio oferecendo o caminho de software. As [páginas de produtos](https://www.blaize.com/products/) visam claramente visão e edge AI, mas não há uma matriz pública atual de compatibilidade modelo a modelo.

Por isso, este guia classifica a Blaize como comercial e restrita, em vez de preencher a lacuna com afirmações da comunidade. Um relatório de compilação e acurácia fornecido pelo fornecedor para o modelo pretendido deve ser um pré-requisito.

## TinyML e visão em dispositivos de endpoint

### Arm Ethos-U e Alif

A Arm licencia IP NPU Ethos-U55, U65 e U85 para empresas de chips. O [compilador Vela](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) recebe grafos LiteRT/TFLite quantizados e reescreve subgrafos suportados em operações personalizadas Ethos-U. Operações sem suporte podem permanecer na CPU; portanto, "a aplicação executa" e "a rede inteira executa na NPU" são afirmações diferentes.

Implementações reais incluem Ethos-U55 no Alif Ensemble E7, Ethos-U65 no NXP i.MX 93 e Ethos-U85 em sistemas mais novos. O [E7 AI/ML AppKit](https://alifsemi.com/support/kits/ensemble-e7appkit/) da Alif documenta dois aceleradores de ML e demonstra o fluxo E7 de TFLite para Vela; o [Ensemble E8](https://alifsemi.com/ensemble-e8-series/) combina um Ethos-U85 com duas NPUs Ethos-U55. A Alif também publica um [benchmark de detector de faces YOLO-Fastest INT8](https://alifsemi.com/faster-ai-mcu-inferencing-low-power-consumption/) no nível da família, em 192 por 192 no Cortex-M55 mais Ethos-U55, mas não uma matriz ampla de YOLO moderno por alvo. Esses sistemas com memória limitada servem para redes pequenas de classificação e detecção, não para detectores arbitrários de 640 pixels só porque podem ser expressos em TFLite.

### Infineon PSOC Edge, Himax WiseEye2 e Analog Devices MAX7800x

As famílias [PSOC Edge E83](https://documentation.infineon.com/psocedge/docs/xsk1761304020519) e E84 da Infineon combinam Cortex-M55 e Ethos-U55 e acrescentam um bloco NNLite separado no lado M33 de baixo consumo. O [manual de arquitetura E84](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) documenta pesos de 8 bits, ativações de 8 ou 16 bits e um fluxo Vela no qual operadores suportados viram um fluxo de comandos da NPU, enquanto operações sem suporte continuam na CPU. O [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter) aceita TFLite e caminhos Keras e PyTorch, enquanto o material de arquitetura cita MobileNetV1/V2 como kernels representativos, não benchmarks para o alvo. O [E84 AI Kit](https://documentation.infineon.com/psocedge/docs/cci1762693051052) inclui uma câmera e usa ModusToolbox, mas o material público ainda não fornece uma matriz de compatibilidade YOLO identificada por modelo. É uma plataforma programável real para visão com evidências de modelo em desenvolvimento, não um alvo validado para detectores de uso geral.

O [HX6538 WiseEye2](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) da Himax também combina Cortex-M55 e Ethos-U55 para visão sempre ativa. Sua oferta pública de software é especialmente concreta para essa classe de potência: os [exemplos oficiais Grove Vision AI Module V2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) incluem detecção de objetos, pose e classificação de gênero YOLOv8n, detecção YOLO11n, face mesh e PeopleNet. O caminho unificado usa TFLite Micro e Vela, com fallback para CMSIS-NN e kernels de referência quando necessário. São modelos e resoluções deliberadamente pequenos, não evidência de que um grafo YOLO convencional de tamanho de servidor caiba.

A Analog Devices segue outro caminho com os [aceleradores CNN MAX78000/MAX78002](https://www.analog.com/en/products/max78002.html). A ferramenta pública [`ai8x-synthesis`](https://github.com/analogdevicesinc/ai8x-synthesis) quantiza uma rede treinada e gera código C específico do alvo, pesos e configuração do acelerador, em vez de um pacote portável de runtime ONNX. A evidência própria inclui [identificação facial](https://www.analog.com/en/resources/app-notes/an-2616.html), detecção QR TinierSSD no [model zoo da ADI](https://ez.analog.com/dsp/software-and-development-tools/edgebench/a/docs-faqs/DF912/available-models-in-adi-model-zoo) e classificadores de imagem. Os dispositivos são relevantes para visão de endpoint, mas não foi encontrada uma matriz oficial atual de YOLO moderno.

### GreenWaves GAP9

O GAP9 combina processamento RISC-V com o engine neural NE16. O NNTool importa e quantiza redes, enquanto AutoTiler e GAP SDK geram código implantável. O [menu de redes neurais GAP9](https://github.com/GreenWaves-Technologies/nn_menu_gap9) público contém MobileNet, EfficientNet, ResNet, MobileNet SSD, face e aplicações de reconhecimento. A GreenWaves também publica um [projeto de detecção de pessoas YOLOX](https://github.com/GreenWaves-Technologies/yolox_people_detection) com resultados de acurácia e simulação.

Essa é uma evidência TinyML excepcionalmente transparente, mas o SDK completo só está disponível para clientes qualificados e os modelos são bem menores que as variantes YOLO populares de 640 por 640.

### Lattice sensAI

O Lattice sensAI é uma alternativa baseada em FPGA a uma NPU fixa. O sensAI Studio e o Neural Network Compiler mapeiam um grafo suportado para IP acelerador configurável nos dispositivos ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E e Avant-X. A saída depende do FPGA selecionado, modo do acelerador, layout de memória e bitstream, em vez de ser um modelo de runtime portável.

O [manual atual do Neural Network Compiler](https://www.latticesemi.com/-/media/LatticeSemi/Documents/UserManuals/MQ3/FPGA-UG-02052-8-0-Lattice-Neural-Network-Compiler-Software.ashx?document_id=52343) inclui uma tabela de topologias especialmente direta. Ela lista YOLOv1 para ECP5, YOLOv5 e YOLOv8 nos modos otimizado/avançado de CrossLink-NX e CertusPro-NX e YOLO11 somente com IP Advanced. Também lista SSD, MobileNetV2-SSD, ResNet, ENet, SqueezeDet e outras redes de visão, com células sem suporte visíveis por família FPGA. O [Advanced CNN Accelerator](https://www.latticesemi.com/en/Products/DesignSoftwareAndIP/IntellectualProperty/IPCore/IPCores05/Advanced-CNN-Accelerator-IP) tem como alvo Avant-E/Avant-X/CertusPro-NX e é IP comercial. Essa é evidência da tabela do compilador, não uma promessa de que um bitstream suporte simultaneamente todas as topologias listadas.

### Microchip VectorBlox

O [VectorBlox SDK 3.1](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) público da Microchip pré-processa e compila redes TFLite INT8 totalmente quantizadas para IP CoreVectorBlox configurável em FPGAs PolarFire SoC. `tflite_preprocess` e `vnnx_compile` produzem assets de deploy `.vnnx`, `.hex` e `.ucomp`, e o repositório inclui um simulador e driver C para o alvo. Grafos TensorFlow, ONNX e OpenVINO de origem ainda precisam passar pelo caminho documentado de preparação TFLite/INT8.

A [matriz atual de tutoriais](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) inclui YOLOv5n, detecção/classificação/OBB/pose/segmentação YOLOv8n, YOLOv9t, variantes YOLO esparsas/comprimidas, MobileNet, EfficientNet-Lite0, ResNet18, MiDaS, FFNet e QuickSRNet. O [guia de pós-processamento C](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/C_Postprocessing.md) cita separadamente YOLOv2/3/4/5 e decodificadores de detecção, pose e OBB Ultralytics. O SDK 3.1 atualmente oferece suporte ao PolarFire SoC Video Kit e explicitamente não ao PolarFire Video Kit sem SoC; portanto, demos de placas antigas não devem ser confundidas com o contrato atual do alvo.

O SDK pode ser baixado publicamente sob uma [licença específica da Microchip](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/LICENSE.md) que restringe o software e derivados a produtos Microchip. A Microchip oferece licenças Libero Silver e CoreVectorBlox gratuitas, mas mediante solicitação. O deploy continua sendo um contrato de sistema FPGA: configuração do acelerador, bitstream, firmware, dados de rede gerados pelo SDK e layout de memória precisam corresponder. Uma compilação VectorBlox bem-sucedida não transforma o modelo em um binário que possa ser copiado para um PolarFire qualquer.

### Syntiant

O NDP200 da Syntiant tem como alvo inferência sempre ativa de visão, áudio e sensores abaixo do envelope de potência de SoCs Linux normais. A [tabela atual de hardware](https://www.syntiant.com/hardware) lista o NDP200 em produção em massa e o NDP250 em amostragem. O [resumo de produto NDP200](https://www.syntiant.com/ndp200/) documenta suporte a redes CNN, RNN e totalmente conectadas abaixo de 1 mW; o [anúncio NDP250](https://www.syntiant.com/news/syntiant-unveils-ndp250-neural-decision-processor-with-next-gen-core-3-architecture) descreve, em vez disso, reconhecimento de imagem sempre ativo abaixo de 30 mW. "Sub-miliwatt" não deve ser generalizado entre as duas peças.

O material público não comprova ampla compatibilidade YOLO, portanto o produto deve ser avaliado para classificadores e detectores pequenos sempre ativos por meio de uma análise de modelos apoiada pelo fornecedor, não por uma suposição genérica de exportação ONNX.

### Google Coral: duas gerações sem relação entre si

Os produtos Edge TPU legados usam Edge TPU Compiler, `libedgetpu` e PyCoral com modelos TFLite totalmente INT8. Seus principais repositórios [Edge TPU](https://github.com/google-coral/edgetpu) e [PyCoral](https://github.com/google-coral/pycoral) estão arquivados. Isso representa um risco real de manutenção, mas não é um aviso formal de fim de vida do hardware.

O Google também mantém um [Coral NPU](https://github.com/google-coral/coralnpu) de código aberto separado, ativo: IP acelerador RISC-V para integração em SoCs de baixo consumo. Atualmente, o projeto público expõe RTL, simulação de hardware, toolchain e exemplos ELF. Ele não é um substituto direto para Edge TPU USB/M.2 e não publica uma matriz atual de deploy YOLO.

## Armadilhas com GPU adjacente, NPU client e computação adaptativa

NVIDIA, AMD, Intel e Apple aparecem frequentemente em buscas por NPU, mas não expõem uma classe única e intercambiável de aceleradores.

**NVIDIA Jetson:** Jetson Orin combina GPU CUDA com núcleos DLA dedicados. O TensorRT pode construir um engine serializado para DLA, mas camadas sem suporte só rodam na GPU se o fallback para GPU estiver explicitamente habilitado. A [tabela YOLO DeepStream](https://github.com/NVIDIA-AI-IOT/deepstream_tools/blob/main/yolo_deepstream/README.md) atual cobre YOLOv4/v7/v8/v9/11, mas a maioria das entradas tem como alvo a GPU; a entrada ONNX explícita para DLA é YOLOv8s INT8. As [restrições DLA](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html) da NVIDIA explicam os limites de operadores. Thor é outra distinção: o [guia de migração DriveOS](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/NVIDIA_DriveOS_7.0.3_Migration_Guide.pdf) da NVIDIA diz que os núcleos DLA foram removidos do Thor em favor da flexibilidade de escalonamento na GPU. Portanto, "roda no Jetson" não identifica o acelerador.

**AMD Vitis AI:** a geração antiga Kria/DPU compilava grafos `.xmodel` executados pelo VART. O [Vitis AI 6.2](https://vitisai.docs.amd.com/en/6.2/) atual é GA para dois caminhos distintos Versal AI Edge: [Gen1](https://vitisai.docs.amd.com/projects/gen1/en/latest/index.html) em VEK280/VE2802 usa ONNX, AMD Quark e um fluxo de snapshot/subgrafo vinculado ao alvo; [Gen2](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html) em VEK385 tem contrato próprio de compilação e runtime. Gen1 publica exemplos YOLOv5, YOLOv7, YOLOv8 e YOLOX. Ryzen AI é uma quarta pilha NPU client separada. Não transfira artefatos ou validação entre DPU legado, Versal Gen1, Versal Gen2 e Ryzen AI só porque todos usam o nome Vitis ou AMD.

**Intel OpenVINO:** o OpenVINO pode ter como alvo CPU, GPU e NPU Core Ultra pela mesma API. A [documentação do plug-in NPU Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) identifica gerações NPU compatíveis, enquanto a [matriz de modelos verificados](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html) mostra colunas específicas do dispositivo. Um notebook YOLO comprova uma receita de aplicação, não validação em NPU, a menos que a matriz diga isso. Operações, formatos, precisão e driver NPU instalado ainda determinam se o grafo completo roda ali. OpenVINO IR (`.xml` mais `.bin`) é uma IR reutilizável de origem; um cache de modelo compilado é específico do dispositivo e do software.

**Apple Core ML:** `coremltools` converte modelos para `.mlpackage`, que o Xcode compila em `.mlmodelc`. Core ML pode distribuir o trabalho entre CPU, GPU e Apple Neural Engine, mas abstrai intencionalmente a partição exata. Isso faz da Apple uma excelente plataforma de deploy e uma opção ruim para afirmações como "o grafo YOLO inteiro rodou na NPU", a menos que o profiling comprove isso.

## As empresas de IP de NPU por trás das empresas de chips

Algumas empresas não vendem uma placa nem um acelerador empacotado. Elas licenciam designs de NPU para fornecedores de SoC. Importam porque a mesma arquitetura subjacente pode aparecer em várias marcas de chips, mas o SDK e o artefato finais ainda podem ser personalizados pelo licenciado.

| Fornecedor de IP | Família NPU e software | Por que importa |
|---|---|---|
| [Arm](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) | Ethos-U55/U65/U85 e Vela | Aparece em produtos NXP, Alif, Infineon, Himax e outros embarcados; fluxo orientado a TFLite/TOSA quantizado |
| [Arm China](https://www.armchina.com/mountain?infoId=161&name=) | AIPU Zhouyi e Compass SDK | O [model zoo](https://github.com/Arm-China/Model_zoo) público cita YOLOv1-tiny/v2/v3/v4/v5, YOLOX e YOLOv8-seg; entradas são modelos de referência, não prova para cada chip licenciado |
| [VeriSilicon](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico) | Família Vivante VIP e Acuity SDK | Famílias NPU Vivante relacionadas aparecem em vários SoCs, inclusive nos caminhos A311D e i.MX 8M Plus adquiridos separadamente; cada fornecedor de chip ainda oferece sua própria integração |
| [Imagination Technologies](https://www.imaginationtech.com/products/open-access/) | PowerVR Series3NX AX3146/AX3386/AX3596; IMG Series4 histórico; Neural Compute SDK/IMG DNN | IP NNA licenciável, não placa de varejo; Open Access expõe configurações Series3NX de 1/5/10-TOPS, enquanto um programa NC-SDK oficial cita PP-YOLOE, EfficientNet e HRNet, mas não cada configuração de IP |
| [CEVA](https://www.ceva-ip.com/product/ceva-neupro-studio/) | NeuPro-Nano/NeuPro-M e NeuPro Studio | IP NPU licenciável com importação, quantização, compressão, compilação de grafo, simulação e geração C/C++ |
| [MIPS](https://mips.com/processor-solutions/arc-npx-family/) | ARC NPX6 e [MetaWare MX](https://mips.com/processor-solutions/arc-metaware-mx/), anteriormente Synopsys ARC | IP NPU escalável e NN SDK adquiridos pela GlobalFoundries e colocados no portfólio MIPS em [junho de 2026](https://mips.com/press-releases/gfmipsarcclose/); não são um alvo de deploy de varejo |
| [Cadence](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/neuroweave-sdk.html) | Neo NPU, DSPs Tensilica e NeuroWeave SDK | Pilha de compilador/interpretador usada por licenciados de SoC, com suporte a redes e quantização vinculado às configurações licenciadas |
| [Quadric](https://quadric.ai/npu-ip) | IP GPNPU Chimera e SDK | IP programável no estilo NPU/DSP licenciado a chips de outras empresas; o alvo final é o silício do licenciado |
| [Expedera](https://www.expedera.com/products-overview/) | IP NPU Origin e pilha de software | Arquitetura NPU edge licenciável, não uma placa LibreYOLO que possa ser comprada diretamente |

A Imagination mostra por que rótulos de acesso e ciclo de vida devem aparecer ao lado dos nomes de arquitetura. Seu [programa Open Access](https://www.imaginationtech.com/products/open-access/) oferece três configurações Series3NX comprovadas em silício sem taxa inicial de licença de IP, mas somente após avaliação da empresa, com suporte/manutenção pagos e royalties de produção. O [Neural Compute SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) fornece ferramentas de compilação, otimização, quantização e runtime; uma [colaboração oficial Baidu PaddlePaddle](https://www.imaginationtech.com/news/imagination-and-baidu-paddlepaddle-create-open-source-machine-learning-library-for-model-zoo/) cita detecção PP-YOLOE, classificação EfficientNet e segmentação HRNet. Esses exemplos não validam todas as configurações Series3NX ou Series4, e as páginas atuais de produtos [Series4](https://www.imaginationtech.com/product/img-4nx-mc1/) marcam os produtos como indisponíveis; portanto, Series4 deve ser considerada histórica, a menos que a equipe de vendas confirme o contrário.

Essa camada explica semelhanças aparentes entre famílias. Ela não cria portabilidade de artefatos. Dois SoCs que contenham IP Vivante ou Ethos relacionado podem ter mapas de memória, drivers, versões de compilador, conjuntos de operadores e runtimes do fornecedor diferentes.

## O que você realmente implanta: a tabela de dependência dos artefatos

O último arquivo no pipeline do compilador é onde a aparente portabilidade ONNX termina. Nomes e extensões mudam com o tempo, mas a regra prática permanece: mantenha o modelo original e os dados de calibração, porque o artefato nativo geralmente não pode ser movido para outra geração de NPU nem recompilado com segurança usando outra versão do SDK.

| Pilha do fornecedor | Entrada comum do compilador | O que chega ao dispositivo | A que fica vinculado |
|---|---|---|---|
| [Amlogic AMLNN](https://github.com/Amlogic-NN/amlnn-toolkit) | ONNX, TFLite, TorchScript ou PT2 | `.adla`; A311D legado usa `.nb` | ID do alvo, geração ADLA, build do compilador AMLNN, NNSDK2/runtime, driver ADLA e BSP |
| [Rockchip RKNN](https://github.com/airockchip/rknn-toolkit2) | ONNX, TFLite ou exportação de framework | `.rknn` | Versão do toolkit/runtime RKNN e família de NPU Rockchip |
| [Hailo](https://hailo.ai/developer-zone/documentation/) | ONNX ou TensorFlow por meio de um HAR intermediário | `.hef` | Arquitetura Hailo e geração de compilador/runtime |
| [Axelera Voyager](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) | ONNX ou definição de zoo compatível | `.axm` no Pipeline Builder alfa, pacote de pipeline `.axe`; `.axmodel` clássico mais manifesto | Geração da API Voyager e configuração alvo Metis |
| [DEEPX DXNN](https://github.com/DEEPX-AI/dx-all-suite) | ONNX e caminhos de framework compatíveis | `.dxnn` | Versões DX-COM/DX-RT e alvo DX-M |
| [Qualcomm QAIRT/QNN ou SNPE](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | ONNX, TFLite ou modelo de framework | Biblioteca de modelo/binário de contexto QNN ou SNPE `.dlc` | Arquitetura HTP, SoC, backend e versão do runtime |
| [MediaTek NeuroPilot](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) | TFLite convertido/quantizado | `.dla` para Neuron Runtime offline; `.tflite` para modo delegado | Geração NP/MDLA, imagem do sistema operacional, compilador e runtime |
| [TI TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools) | ONNX ou TFLite | Diretório de artefatos importados mais arquivos do modelo de aplicação | Ferramentas/runtime TIDL e geração do processador |
| [NXP eIQ](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | Geralmente TFLite ou ONNX | Saída Vela, TIM-VX, Neutron ou Ara específica do backend | Acelerador i.MX/Ara selecionado e BSP, não "NXP" em geral |
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | TFLite, ONNX ou modelo de framework compatível | Biblioteca/código de rede e assets de firmware gerados | MCU/NPU alvo, configuração de memória e versão da ferramenta |
| [Infineon PSOC Edge](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) | Caminho TFLite/Keras/PyTorch quantizado em inteiros | TFLite otimizado pelo Vela com fluxo de comandos Ethos-U e firmware da aplicação | Variante E83/E84, configuração Ethos-U, Vela, ModusToolbox e BSP; NNLite é um alvo separado |
| [Renesas DRP-AI](https://github.com/renesas-rz/rzv_drp-ai_tvm) | ONNX/TFLite via TVM ou Translator | Diretório de modelo de runtime com vários arquivos de dados binários | Dispositivo RZ/V e geração do translator/runtime |
| [Sony IMX500](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | Rede quantizada/empacotada pelo fluxo Edge-MDT | Pacote `.rpk` | Firmware do sensor, orçamento de memória e versão do packer |
| [Synaptics](https://developer.synaptics.com/) | TFLite/ONNX e importações compatíveis | `.synap` no SyNAP; `.vmfb` no Torq | Geração de software/hardware SL16xx versus SL261x |
| [Ambarella CVflow](https://www.ambarella.com/developer/) | Entrada da toolchain para parceiros | Pacote de deploy restrito a parceiros; contrato exato do artefato não documentado publicamente | Confirme geração CVflow, SDK/BSP e pipeline de câmera pelo Cooper Developer Zone |
| [D-Robotics BPU](https://github.com/D-Robotics/rdk_model_zoo) | ONNX e grafo compatível com a toolchain | `.bin` no X5; `.hbm` no `rdk_s` atual; X3 usa fluxo de plataforma antigo | Geração BPU, branch do repositório e versão correspondente do OpenExplorer/runtime |
| [AXERA Pulsar2](https://github.com/AXERA-TECH/ax-samples) | ONNX | `.axmodel` | Alvo de chip AX e versão Pulsar2/AXEngine |
| [SOPHGO TPU-MLIR](https://github.com/sophgo/tpu-mlir) | ONNX, TFLite, TorchScript e outros | `.bmodel` em BM, `.cvimodel` em CV18xx | Alvo BM/CV e geração do runtime SOPHON |
| [Huawei Ascend CANN](https://www.hiascend.com/en/software/cann) | ONNX ou grafo de framework compatível | `.om` | Chip Ascend, CANN/ATC e firmware/driver do dispositivo |
| [Cambricon MagicMind](https://github.com/Cambricon/magicmind_cloud) | Grafo de framework ou modelo de intercâmbio | Modelo/engine MagicMind serializado | Arquitetura MLU e versão Neuware/MagicMind |
| [Canaan nncase](https://github.com/kendryte/nncase) | ONNX ou TFLite | `.kmodel` | Geração KPU e plug-in alvo nncase correspondente |
| [Mobilint qb](https://www.mobilint.com/sdk-qb) | ONNX e modelos de framework compatíveis | `.mxq` | Alvo REGULUS/ARIES e compilador/runtime qb |
| [Rebellions RBLN](https://docs.rbln.ai/latest/index.html) | PyTorch 2 ou grafo TensorFlow compatível | `.rbln` | Geração ATOM e compilador/runtime RBLN |
| [FuriosaAI](https://developer.furiosa.ai/docs/latest/en/) | ONNX/TFLite e caminho de framework compatível | Warboy `.enf`; RNGD `.fxb` | Gerações de acelerador e SDK completamente diferentes |
| [Sunplus SNNF](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2004353133) | Rede compatível com Acuity | `.nb` | Driver NPU SP7350, runtime e BSP |
| [ESWIN ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) | ONNX no fluxo EsAAC atual documentado; outros frameworks precisam ser exportados/convertidos | `.model` | Alvo exato da família EIC7700 e versão ENNP/ESSDK |
| [Nuvoton / Ethos-U](https://github.com/OpenNuvoton/NuEdgeWise) | TFLite quantizado | TFLite otimizado pelo Vela com operações personalizadas Ethos-U | Plano de memória M55M1, Vela e firmware |
| [Himax WiseEye2](https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2) | TFLite totalmente quantizado em inteiros | `.tflite` otimizado pelo Vela na memória de modelo mais firmware da placa, como `output.img` | Configuração HX6538/WE2, Vela, TFLite Micro, driver Ethos-U e kernels fallback |
| [Analog Devices AI8X](https://github.com/analogdevicesinc/ai8x-synthesis) | Checkpoint PyTorch, YAML da rede e entrada de amostra | Código-fonte/headers C específicos do dispositivo, pesos e firmware compilado gerados | Limites de memória/camadas MAX78000 versus MAX78002, ferramenta de síntese e versão MSDK |
| [Realtek AmebaPro2](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Serviço/conversor do fornecedor e entrada | `.nb` | Firmware RTL8735B e versão da API VoE/NeuralNetwork |
| [Telechips Enlight](https://docs.topst.ai/product/p/ai) | Caminho Darknet, TensorFlow, ONNX ou PyTorch | Intermediário `.enlight` e pacote compilado para deploy | NPU TCC7500 e versões TC-NN/Enlight |
| [T-Head HHB](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | ONNX e grafo de framework compatível | `hhb.bm`, parâmetros/código gerados e executável | Pilha CSI-NN2/SHL TH1520 |
| [SigmaStar MI_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | Documentação atual SSU9383CM expõe runtime MI_IPU; materiais antigos usam entradas da família TensorFlow/Caffe | `.sim` antigo convertido para `sgsimg.img`; artefato público atual não verificado | IPU SigmaStar exata e geração do SDK; compatibilidade do compilador legado com SSU9383CM não verificada |
| [HiSilicon smart vision](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) | ONNX e grafo compatível com ATC | `.om` | SoC exato de visão inteligente e pilha NNN/SVP-NNN; não é portabilidade Ascend genérica |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | ONNX, TFLite, Keras ou TensorFlow | Pacote dataflow `.dfp` | Configuração de hardware MX e versão de runtime/compilador |
| [Kneron](https://doc.kneron.com/docs/) | ONNX mais configuração da toolchain | `.nef`; KL730 NEFv2 pode conter `.kne` | Alvo documentado do compilador, geração do chip, firmware e runtime PLUS; compilação KL830 não está estabelecida pelo Toolchain 0.33.1 |
| [SiMa.ai Palette](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) | Modelo de framework compatível ou ONNX | `.tar.gz` compilado pelo ModelSDK, ELF executável documentado ou pacote `.mpk` do MPK Tool | Alvo MLSoC/Modalix e versão Palette |
| [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | ONNX ou definição de rede | Engine serializado, geralmente `.engine` ou `.plan` | Arquitetura GPU/DLA, TensorRT, CUDA e frequentemente o dispositivo |
| [AMD Vitis AI](https://vitisai.docs.amd.com/en/6.2/) | ONNX/framework quantizado | `.xmodel` legado; snapshot/subgrafos NPU Gen1; diretório de cache compilado ou pacote de produção `.rai` Gen2 | Geração/IP NPU exata, imagem da plataforma e matriz Vitis/Vivado/PetaLinux |
| [Microchip VectorBlox](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) | TFLite INT8 totalmente quantizado | `.vnnx`, `.hex` e `.ucomp` mais firmware/design FPGA correspondente | Configuração CoreVectorBlox, PolarFire SoC Video Kit, SDK 3.1, bitstream e layout de memória |
| [Intel OpenVINO](https://docs.openvino.ai/) | Modelo de framework ou ONNX | IR `.xml` mais `.bin`, ou cache compilado específico do dispositivo | IR é reutilizável; cache compilado é específico do dispositivo, driver e OpenVINO |
| [Google Edge TPU legado](https://coral.ai/docs/edgetpu/models-intro/) | TFLite totalmente INT8 | `_edgetpu.tflite` compilado para Edge TPU | Compilador/runtime Edge TPU e operadores quantizados compatíveis; sem relação com o novo IP Coral NPU |
| [Google Coral NPU IP](https://github.com/google-coral/coralnpu) | Exemplos C/C++ para o projeto de IP público atual | Exemplos ELF para simulação RTL/hardware e eventual integração em SoC; não há artefato de compilador YOLO público | Implementação exata de SoC licenciada/integrada e toolchain aberto em evolução |
| [Lattice sensAI](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | Descrição de rede compatível | Bitstream FPGA, configuração do acelerador e pesos | Família FPGA, modo IP sensAI e implementação de memória |
| [Imagination NC-SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) | Caffe, TensorFlow ou ONNX | Nome de arquivo do artefato compilado para o cliente não especificado publicamente | Configuração NNA licenciada, integração SoC, NC-SDK/DDK e runtime do licenciado |

Esta tabela não é apenas uma lista de extensões. A versão do compilador, identificador do alvo, driver, firmware e BSP fazem parte da identidade reprodutível do modelo. Um exportador LibreYOLO futuro deve gravá-los em um manifesto legível por máquina junto de cada artefato.

## Sinais de acesso às ferramentas e ciclo de vida

A disponibilidade de hardware e o acesso ao compilador são independentes. Uma placa pode ser fácil de comprar enquanto o compilador atual exige um acordo com o cliente; um SDK público pode ter como alvo silício difícil de obter. Os itens a seguir são sinais concretos de primeira parte, não um ranking universal de longevidade.

| Plataforma | Sinal documentado | O que significa na prática |
|---|---|---|
| Amlogic ADLA | Repositórios Apache-2.0 públicos e wheels para baixar; drivers/BSP compatíveis ainda podem exigir a Amlogic ou o fornecedor da placa | Avaliação fácil do compilador, mas a redistribuição de binários e a matriz completa de compatibilidade devem ser confirmadas |
| MediaTek Genio | IoT AI Hub e guias Yocto públicos; ferramentas NP8 all-in-one e documentos Android marcados como acesso direto de cliente/NDA | Evidência de modelos auditável, mas a automação para modelos próprios pode exigir parceria |
| Hailo | Model zoo e runtime públicos; Dataflow Compiler distribuído pelo Developer Zone; `hailo-camera-apps` público arquivado em 3 de maio de 2026 | Descoberta ampla de modelos aberta, mas compilação reproduzível exige conta/versão correta; o arquivamento não comprova fim de vida Hailo-15 e seu pacote atual de aplicação do processador de visão deve ser confirmado |
| Axelera AI | Documentação pública e [Voyager SDK público](https://github.com/axelera-ai-hub/voyager-sdk); suporte ao cliente exige conta | SDK de acelerador raramente disponível para autoatendimento, enquanto suporte ao produto e aquisição continuam comerciais |
| Qualcomm Dragonwing | O [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) lista IQ-9075 como Sampling com longevidade até 2038 e QCS6490 até julho de 2036, enquanto a página IQ-9075 diz Active | Forte sinal para planejamento industrial, mas o conflito entre status de fontes primárias exige confirmação por SKU e não garante estabilidade ABI de um SDK de IA durante todo o período |
| Rebellions | A [matriz de suporte](https://docs.rbln.ai/latest/supports/version_matrix.html) atual marca CA02 e CA12 como fim de vida e CA22/CA25 como ativos; CA21 não aparece | Aquisição e compatibilidade do SDK dependem da placa, inclusive dentro da família ATOM |
| NVIDIA Jetson | A [tabela oficial de ciclo de vida de módulos](https://developer.nvidia.com/embedded/lifecycle) lista módulos Orin até janeiro de 2032 e AGX Orin Industrial até julho de 2033; os kits de desenvolvedor não têm compromisso de ciclo de vida | Projete sistemas de produção em torno dos módulos, não da disponibilidade dos kits de desenvolvimento |
| Sony IMX500 na Raspberry Pi | O [resumo de produto AI Camera](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf) diz que a produção vai pelo menos até janeiro de 2028 | Mínimo concreto para esse módulo de câmera, não para todos os produtos IMX500 |
| Amlogic A311Y3 | A [página de lançamento de 2026](https://www.amlogic.com/News/index248.html) anuncia garantia de longevidade do produto por no mínimo dez anos | Sinal promissor para a nova plataforma, ainda sujeito a confirmação contratual e de SKU para um programa de produto |
| Google Coral | Os repositórios legados Edge TPU e PyCoral estão arquivados/somente leitura; o repositório separado [Coral NPU IP](https://github.com/google-coral/coralnpu) está ativo | Risco de manutenção do software legado, por si só não é aviso formal de fim de vida do hardware; não diz nada sobre o ciclo de vida do projeto de IP de código aberto sem relação |

## Por que TOPS não serve como guia de compra

TOPS de pico pode ser útil dentro de um fornecedor e arquitetura, mas comparações entre fornecedores geralmente são inválidas, a menos que todos os itens abaixo correspondam:

- Precisão numérica, incluindo INT8, INT4, FP16 ou modo misto
- Se uma multiplicação-acumulação conta como uma ou duas operações
- Aritmética densa ou esparsa estruturada
- Resolução de entrada, tamanho de batch e grafo do modelo
- Acurácia depois da quantização
- Latência somente da NPU ou do pipeline completo da aplicação
- Transferências de memória e fallback no host
- Estado térmico e operação sustentada, não em pico

Um detector faz parte de um pipeline: aquisição de imagem, resize/letterbox, normalização, transferência ao dispositivo, inferência neural, decode, NMS, ajuste de coordenadas e lógica da aplicação. Os fornecedores frequentemente publicam apenas a parte central da inferência neural. O [MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/) é valioso porque define cenários, metas de qualidade e regras de sistema medidas, em vez de comparar isoladamente aritmética de marketing.

Para deploys YOLO, o registro mínimo de benchmark confiável é:

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

Sem esses campos, dois números de FPS geralmente medem coisas diferentes.

## Como escolher uma NPU para visão computacional

Escolha nesta ordem, não pelo maior número impresso na página do produto.

1. **Comprove a compatibilidade do grafo.** Compile o modelo exportado exato, não um checkpoint de zoo com nome parecido.
2. **Meça a acurácia.** Valide o artefato compilado no dataset real da tarefa após calibração e quantização.
3. **Meça de ponta a ponta.** Inclua conversão de entrada, transferências, decode e NMS.
4. **Verifique o acesso ao software.** Determine se o compilador é público, exige cadastro ou só está disponível após acordo comercial.
5. **Congele a matriz de versões.** Registre compilador, runtime, driver, firmware e identificadores do alvo.
6. **Verifique a realidade do sistema operacional.** Suporte Android, Debian, Yocto, Buildroot, RTOS e Windows não é intercambiável.
7. **Confira fornecimento e ciclo de vida.** Um chip tecnicamente excelente é uma escolha ruim para produção se houver incerteza sobre módulos, drivers ou suporte de longo prazo.
8. **Só então compare custo, potência e desempenho.** Essas medições só fazem sentido depois que o mesmo modelo estiver correto nos dois alvos.

### Pontos de partida práticos por caso de uso

| Caso de uso | Plataformas que vale avaliar primeiro | Motivo |
|---|---|---|
| Acelerador específico para Raspberry Pi | Hailo-8L/8 e Sony IMX500 | Produtos oficiais Raspberry Pi, imagens de software e fluxos concretos para desenvolvedores |
| Acessório geral Linux PCIe, M.2 ou USB | Produtos DEEPX, MemryX, Hailo e Axelera | Formatos de acelerador disponíveis, sujeitos à compatibilidade de host/driver |
| SBC Linux ou câmera de baixo custo | Rockchip RK3588/RK3576, Amlogic ADLA se houver placa/BSP compatível, AXERA, Canaan K230 ou Sunplus SP7350 | Pipelines de mídia integrados e evidência pública de modelo/compilador; verifique VIM4 V13A ou posterior para NPU A311D2 |
| Android móvel e de alto volume | Qualcomm QNN/AI Hub, LiteRT, Apple Core ML | Grande base instalada e runtimes móveis mantidos |
| Linux industrial e robótica | TI TDA4/AM6xA, NXP i.MX, Renesas RZ/V, Hailo | Produtos embarcados de vida longa e ecossistemas de câmera/E/S |
| Endpoint muito pequeno ou MCU | STM32N6x7, Infineon PSOC Edge, Himax WiseEye2, Analog Devices MAX7800x, Nuvoton M55M1, Alif/Arm Ethos-U, GAP9 e Syntiant | Limites rigorosos de potência e memória, ferramentas específicas e restrições de tamanho de modelo |
| Visão configurável em FPGA | Microchip VectorBlox, Lattice sensAI e alvos AMD Vitis AI | Pipelines de hardware flexíveis, mas configuração do acelerador, bitstream e compilador de modelo formam um único sistema versionado |
| Visão PCIe de alto throughput | Axelera Metis, Hailo, DEEPX, Mobilint, MemryX, Rebellions ou SiMa.ai | Produtos de acelerador dedicado e foco em múltiplos streams |
| Ecossistema de produto centrado na China | Rockchip, AXERA, D-Robotics, SOPHGO, Huawei Ascend, HiSilicon e Cambricon | Placas regionais, toolchains e exemplos de modelos fortes |

Esta tabela é uma lista de pesquisa, não um ranking de desempenho. Aquisição, disponibilidade regional e o modelo exato podem inverter a ordem.

## O que isso significa para o LibreYOLO

O design escalável não consiste em dezenas de exportadores sem relação. É um único contrato estrito de intercâmbio mais pequenos adaptadores de compilador e runtime do fornecedor:

```text
LibreYOLO model
    -> deterministic static ONNX
    -> VendorCompiler.compile(model, target, calibration, precision)
    -> native artifact + manifest
    -> VendorRuntime.load() / infer()
    -> shared task-specific decode and Results objects
```

Todo artefato nativo deve ter um manifesto complementar com:

- Fornecedor, alvo do chip e alvo da placa
- Versões de compilador, runtime, driver e firmware
- Checksum do checkpoint de origem e do ONNX
- Nomes e formatos de entrada, layout, espaço de cores, normalização e dtype
- Precisão de quantização, escalas quando expostas e hash dos dados de calibração
- Nomes e formatos dos tensores de saída e significado semântico
- Tarefa de detecção, nomes das classes e indicação de decode/NMS no chip ou no host
- Resultados registrados de paridade numérica e acurácia da tarefa

O LibreYOLO já oferece [exportação ONNX portável](/docs/export/onnx), [ferramentas de quantização](/docs/export/quantization), uma [exportação RKNN direta, mas deliberadamente restrita](/docs/export/rknn) e um [fluxo Hailo honesto com compilador externo](/docs/export/hailo). Segundo os critérios usados neste guia, Amlogic ADLA é um forte próximo candidato a integração: o toolkit é público, a cobertura de modelos se sobrepõe muito ao LibreYOLO e o caminho A311D antigo pode ser mantido separado explicitamente.

A prioridade depois da Amlogic deve se basear no hardware dos usuários e na capacidade de executar validação de acurácia, não apenas na disponibilidade do compilador. DEEPX, Sony IMX500, D-Robotics, AXERA, SOPHGO e Qualcomm são tecnicamente atraentes; TI, NXP, ST e Renesas tornam-se especialmente valiosas quando parceiros industriais podem fornecer placas e acesso prolongado a CI.

## Lista de observação: silício real sem contrato público de integração

Excluir completamente uma empresa pode tornar enganoso um panorama de mercado. Incluí-la como "compatível" pode ser pior. Esses fornecedores vendem, estão testando ou anunciaram silício relevante, mas as evidências públicas ainda não estabelecem um caminho reproduzível e atual de compilador, artefato e modelo identificado, adequado a um backend LibreYOLO.

| Empresa | O que existe de fato | Por que permanece na lista de observação |
|---|---|---|
| [Samsung](https://semiconductor.samsung.com/processor/automotive-processor/exynos-auto-v920/) | Exynos Auto V920 inclui NPU de dois núcleos anunciada em até 23.1 TOPS | A Samsung diz que seu [Neural SDK não é mais fornecido a desenvolvedores terceiros](https://developer.samsung.com/neural/overview.html); Samsung ONE/Circle não comprova acesso à NPU Exynos |
| [Novatek](https://www.novatek.com.tw/en-global/Milestone/aboutus_milestones) | Marcos corporativos atuais citam Edge AI e SoCs Edge Vision/Imaging AI; um marco de 2020 mencionava MobileNet, SSD e YOLOv3 | Não há matriz pública atual de número de peça/compilador/artefato/operadores/modelos; YOLOv3 histórico não comprova SDK atual |
| [Nextchip](https://www.nextchip.com/en/adas/adas.php?idx=5) | Processadores automotivos APACHE5/NVS2900 e [APACHE6/NVS3000](https://www.nextchip.com/en/adas/adas.php?idx=7) atuais com NPU aiMotive aiWare; a própria página APACHE6 informa atualmente tanto 12 quanto 8 TOPS | aiWare Studio e um runtime existem, mas não foi encontrado artefato público, guia de quantização, lista de operadores ou matriz YOLO; os valores conflitantes do fornecedor não devem ser resolvidos em silêncio |
| [Black Sesame Technologies](https://bst.ai/en.html) | Silício de IA automotiva/edge Huashan A1000/A2000 e Wudang série C | Há informações públicas sobre os produtos, mas não compilador de autoatendimento, contrato de artefato ou model zoo reproduzível |
| [Ingenic](https://en.ingenic.com.cn/products-detail/id-19.html) | SoCs de câmera T41/T40 com alegações de NPU de baixa precisão; Magik AI descreve PTQ/QAT e compilação de grafo | Não foi encontrado compilador atual para baixar, contrato de artefato ou lista YOLO exata do fornecedor |
| [Fullhan](https://fullhan.com/en/index.php?a=type&c=article&tid=9) | Vários SoCs IPC anunciam NPUs de 0.5 a 2 TOPS, enquanto a família NVR [MC6880](https://fullhan.com/en/index.php?a=type&c=article&tid=48) anuncia 4 TOPS | Não há documentação pública de compilador/runtime/framework/modelo NN suficiente para reprodução independente |
| [Goke Microelectronics](http://www.gokemicro.com/News/info.aspx?itemid=437) | Uma página própria para peças de câmera inteligente GK7606V1/GK7206V1/GK7203V1 anuncia desempenho NPU integrado, mas no momento da publicação é servida apenas por HTTP legado | Não há conversor público, contrato de runtime ou matriz de modelos identificados; o canal é voltado a OEMs |
| [Chengheng Micro](https://en.chenghengmicro.com/) | CH37 anuncia 64 INT8 TOPS e modos FP16/FP32/FP64; a [linha do tempo da empresa em 2026](https://chenghengmicro.com/about.html) diz que entrou em produção de pequenos lotes | Não foi encontrado SDK público, contrato de compilador ou validação de modelos identificados; trate como plataforma comercial inicial, sem autoatendimento, não como evidência de deploy reproduzível |
| [Bouffalo Lab](https://github.com/bouffalolab/bouffalo_sdk) | BL808 inclui NPU BLAI-100 e existem repositórios oficiais atuais do SDK | O SDK mantido não expõe um caminho atual oficial de conversão/exemplo BLAI; os fluxos restantes são evidências legadas ou da comunidade |

Esta lista de observação se baseia intencionalmente em evidências. Um fornecedor pode passar às tabelas de deploy ao publicar uma toolchain atual ou caminho de avaliação, um contrato de artefato específico para o alvo e pelo menos um modelo identificado com uma receita de ponta a ponta.

## O que deliberadamente não foi afirmado

Este artigo não diz que todo modelo identificado funciona a partir de qualquer repositório upstream. Modelos de vendor zoo são frequentemente modificados, divididos em outros nós de saída ou associados a pós-processamento personalizado.

Também não transforma estas afirmações em alegações de suporte:

- Um compilador importa ONNX.
- Um chip anuncia um driver Android NNAPI.
- Um distribuidor diz que a placa é "compatível com YOLO".
- Um repositório comunitário executa uma bifurcação de um modelo.
- Um modelo compila sem erro.
- Um fornecedor relata FPS somente na NPU sem resultado de acurácia.

Aceleradores de telefone Google Tensor e diversos ASICs personalizados exclusivos de OEM também existem, mas o Google não expõe Tensor como alvo geral de compilador para autoatendimento, comparável às plataformas acima. A existência de uma empresa, um diagrama de blocos NPU ou aceleração Android não basta para inventar uma alegação de integração com LibreYOLO.

Da mesma forma, aceleradores de nuvem como Google TPU, AWS Inferentia/Trainium, Microsoft Maia e placas de IA exclusivas de data center estão fora do escopo. Este guia trata de hardware que um desenvolvedor de visão computacional poderia plausivelmente implantar na borda.

O licenciamento do modelo é uma camada separada. A publicação de uma receita de conversão YOLO por um fornecedor de chips não concede direitos para usar ou redistribuir os pesos upstream, o código de treinamento ou um derivado compilado. É preciso verificar o suporte de hardware, as licenças do SDK e as licenças do modelo para o produto pretendido.

## Metodologia da pesquisa e correções

Para cada empresa, a pesquisa procurou quatro tipos de fonte primária:

1. Uma página de produto atual ou ficha técnica que identificasse o silício.
2. Documentação de compilador e runtime explicando o caminho real de deploy.
3. Um model zoo, tabela de compatibilidade ou tutorial completo que citasse modelos de visão.
4. Um sinal de ciclo de vida ou acesso indicando se desenvolvedores conseguem obter as ferramentas.

Organizações GitHub próprias contam como fontes dos fornecedores. A documentação de fabricantes de placas é identificada como evidência do ecossistema quando a própria empresa de chips não publica o mesmo material. Alegações de desempenho de terceiros foram excluídas das tabelas comparativas.

A versão final foi então auditada novamente em rodadas separadas por grupo de fornecedores e em verificações cruzadas entre tabelas. Essas verificações revalidaram escopo dos alvos, nomes dos artefatos, precisão, evidências de modelos versus pós-processadores, status anunciado versus disponível, rótulos de ciclo de vida, denominadores de benchmark e a contagem de cada entidade. Quando duas páginas de primeira parte entram em conflito, este guia expõe o conflito em vez de escolher silenciosamente o valor mais conveniente.

Este mercado muda rapidamente. Se você trabalha em uma dessas empresas e um chip, SDK, lista de modelos ou status de acesso estiver incorreto, envie ao LibreYOLO a URL pública exata da documentação e sua versão. Correções apoiadas por evidências primárias devem substituir este texto; afirmações de marketing sem documentação não devem.

## FAQ

### Quantas empresas de NPU para edge AI existem?

Não existe um total universal de empresas porque fornecedores de produtos, licenciadores de IP, sensores inteligentes, GPUs e ASICs automotivos privados se sobrepõem. Este guia não exaustivo acompanha 69 empresas e ecossistemas de plataforma com chips de edge AI, plataformas aceleradoras, IP de NPU licenciável ou silício relevante em uma lista de observação, em agosto de 2026.

### O que é uma NPU?

Uma unidade de processamento neural é um hardware especializado em operações de redes neurais, como convoluções e multiplicação de matrizes. Os termos dos fornecedores incluem NPU, AIPU, BPU, KPU, DLA, HTP, TPU e MLA, e os modelos compilados geralmente não são portáveis entre empresas.

### Quais empresas de NPU oferecem suporte oficial a modelos YOLO?

Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 por meio do repositório oficial da AI Camera da Raspberry Pi, STMicroelectronics, Renesas, Lattice, Himax, Microchip e várias outras têm entradas em model zoos próprios, tutoriais ou resultados validados para pelo menos uma geração do YOLO. A geração, a tarefa, o chip alvo e a versão do SDK ainda importam.

### Qualquer modelo ONNX pode rodar em qualquer NPU?

Não. ONNX é um formato de intercâmbio, não uma garantia de compatibilidade com o hardware. O compilador da NPU precisa oferecer suporte a cada operador, formato de tensor e tipo de dado do grafo. Formatos estáticos, cortes no grafo, operações personalizadas e fallback para CPU são comuns.

### Qual é a melhor NPU para YOLO?

Não há uma vencedora universal. Hailo, Rockchip, Axelera AI, DEEPX e Amlogic têm ampla cobertura documentada de YOLO, enquanto a Qualcomm tem um alcance excepcional de dispositivos. Escolha considerando a acurácia após a quantização, a latência do pipeline completo, a potência sustentada, o preço, a disponibilidade, o acesso ao SDK e o suporte ao sistema operacional.

### Por que não dá para comparar diretamente os valores de TOPS das NPUs?

Os fornecedores podem contar precisões, operações esparsas, multiplicações-acumulações e hipóteses de utilização de pico diferentes. TOPS não inclui tráfego de memória, operadores sem suporte, pré-processamento, pós-processamento nem a sobrecarga do host. Compare o mesmo modelo, precisão, resolução, acurácia e pipeline completo.

### O que é calibração de NPU?

A calibração executa imagens representativas, geralmente sem rótulos, do ambiente de deploy pelo modelo para que o compilador estime os intervalos de ativação para INT8 ou outro formato de baixa precisão. Imagens aleatórias ou pouco representativas ainda podem gerar um artefato compilado, mas prejudicar a acurácia da tarefa.

### O LibreYOLO oferece suporte a NPUs de edge?

O LibreYOLO exporta ONNX e vários formatos de runtime, tem um caminho direto de compilação RKNN para modelos Rockchip selecionados e documenta o fluxo externo de compilação Hailo. Todo outro artefato nativo do fornecedor ainda exige o SDK correspondente e deve ser descrito como candidato a integração até ser compilado, validado e executado no hardware.
