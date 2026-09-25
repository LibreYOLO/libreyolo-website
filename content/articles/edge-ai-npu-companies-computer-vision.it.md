---
title: "69 aziende Edge AI con NPU: chip, SDK e YOLO (2026)"
description: "Guida documentata a 69 aziende Edge AI e agli ecosistemi NPU: chip, SDK, artefatti, quantizzazione e supporto documentato per computer vision e YOLO."
date: 2026-08-15
author: Xuban
tags: [edge-ai, npu, computer-vision, yolo, hardware, ai-accelerators, amlogic, hailo, rockchip]
faq:
  - q: "Quante aziende producono NPU per l'edge AI?"
    a: "Non esiste un totale universale di aziende, perché si sovrappongono fornitori di prodotti, licenziatari di IP, sensori intelligenti, GPU e ASIC automobilistici proprietari. Questa guida non esaustiva segue 69 aziende ed ecosistemi di piattaforma con chip edge AI pertinenti, piattaforme di accelerazione, IP NPU concessibile in licenza o silicio da tenere d'occhio con basi credibili, aggiornati ad agosto 2026."
  - q: "Che cos'è una NPU?"
    a: "Una neural processing unit è hardware specializzato per operazioni di rete neurale come convoluzioni e moltiplicazioni di matrici. I nomi usati dai fornitori includono NPU, AIPU, BPU, KPU, DLA, HTP, TPU e MLA; in genere i modelli compilati non sono portabili tra aziende."
  - q: "Quali aziende con NPU supportano ufficialmente i modelli YOLO?"
    a: "Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 tramite il repository ufficiale AI Camera di Raspberry Pi, STMicroelectronics, Renesas, Lattice, Himax, Microchip e diverse altre aziende pubblicano voci nei model zoo, tutorial o risultati convalidati per almeno una generazione di YOLO. La generazione esatta, il task, il chip di destinazione e la versione dell'SDK restano determinanti."
  - q: "Qualsiasi modello ONNX può essere eseguito su qualsiasi NPU?"
    a: "No. ONNX è un formato di interscambio, non una garanzia di compatibilità hardware. Il compilatore NPU deve supportare ogni operatore, forma dei tensori e tipo di dato del grafo. Forme statiche, tagli del grafo, operazioni personalizzate e fallback sulla CPU sono comuni."
  - q: "Qual è la NPU migliore per YOLO?"
    a: "Non esiste una vincitrice universale. Hailo, Rockchip, Axelera AI, DEEPX e Amlogic offrono un supporto YOLO documentato consistente, mentre Qualcomm raggiunge un numero eccezionale di dispositivi. Scegli in base all'accuratezza dopo la quantizzazione, alla latenza dell'intera pipeline, ai consumi sostenuti, al prezzo, alla disponibilità, all'accesso all'SDK e al supporto del sistema operativo."
  - q: "Perché non si possono confrontare direttamente i valori TOPS delle NPU?"
    a: "I fornitori possono conteggiare precisioni diverse, operazioni sparse, moltiplicazioni-accumulazioni e ipotesi diverse sull'utilizzo di picco. I TOPS non includono traffico di memoria, operatori non supportati, preelaborazione, postelaborazione e overhead dell'host. Confronta lo stesso modello, la stessa precisione, risoluzione, accuratezza e pipeline completa."
  - q: "Che cos'è la calibrazione di una NPU?"
    a: "La calibrazione esegue sul modello immagini rappresentative, di solito senza etichette, dell'ambiente di deployment, così che il compilatore possa stimare gli intervalli delle attivazioni per INT8 o un altro formato a bassa precisione. Anche immagini casuali o non rappresentative possono produrre un artefatto compilato, ma compromettere l'accuratezza del task."
  - q: "LibreYOLO supporta le NPU edge?"
    a: "LibreYOLO esporta ONNX e diversi formati per runtime, dispone di un percorso diretto verso il compilatore RKNN per alcuni modelli Rockchip e documenta il flusso esterno di compilazione Hailo. Per ogni altro artefatto nativo del fornitore serve ancora il relativo SDK; va considerato un candidato all'integrazione finché non viene compilato, convalidato ed eseguito su hardware."
---

**Non esiste un unico mercato delle NPU. Questa guida non esaustiva segue 69 aziende ed ecosistemi di piattaforma: 51 fornitori con chip o piattaforme distribuibili, nove fornitori di IP NPU e nove aziende da tenere d'occhio, indicate chiaramente.** Coprono classi di acceleratori incompatibili e quasi altrettanti stack di compilazione. Quasi tutti promettono lo stesso percorso: esportare un modello, quantizzarlo, compilarlo, copiare su una scheda un artefatto proprietario e avviarlo con il runtime del fornitore. I dettagli determinano se bastano un pomeriggio o un trimestre di ingegneria.

Questa guida mappa le aziende che nel 2026 forniscono hardware credibile per la computer vision. Riporta i chip pertinenti, i nomi di SDK e compilatori, gli artefatti di deployment, il livello di accesso pubblico e i modelli documentati dal fornitore. Dedica inoltre particolare attenzione ad Amlogic, il cui nuovo stack ADLA è sostanzialmente diverso dal precedente ecosistema NPU A311D.

> **Ambito della ricerca:** le fonti sono state verificate il 15 agosto 2026. Le affermazioni sui modelli specifici provengono da pagine prodotto, documentazione, model zoo, repository o tutorial dei fornitori, salvo dove indicato diversamente. I TOPS dichiarati sono valori dei fornitori, non benchmark confrontabili in modo indipendente. Questa è una guida per sviluppatori, non un consiglio d'investimento né una classifica a pagamento.

**Navigazione rapida:** [tabelle delle aziende](#npu-companies-and-platforms-at-a-glance) | [approfondimento su Amlogic](#amlogic-two-npu-generations-not-one) | [profili dei fornitori](#the-strongest-public-deployment-ecosystems) | [tabella degli artefatti compilati](#what-you-actually-deploy-the-artifact-lock-in-table) | [accesso e ciclo di vita](#tool-access-and-lifecycle-signals) | [roadmap LibreYOLO](#what-this-means-for-libreyolo)

## Il risultato più importante

L'hardware è frammentato, ma il modello di deployment è sorprendentemente uniforme:

```text
PyTorch checkpoint
    -> ONNX or TFLite interchange graph
    -> representative calibration data
    -> vendor quantizer and graph compiler
    -> chip-specific binary or model package
    -> vendor runtime on the target
    -> application preprocessing, decode, NMS and rendering
```

[ONNX consente esplicitamente runtime, generatori di codice e implementazioni hardware](https://onnx.ai/onnx/repo-docs/IR.html), ma un file ONNX è solo il punto di passaggio. Non significa che ogni operatore ONNX possa essere tradotto per ogni acceleratore. Forme statiche degli input, vincoli sugli operatori supportati, regole di quantizzazione e fallback sull'host determinano cosa viene davvero eseguito sulla NPU.

Lo stack software fa quindi parte del chip. Una NPU teoricamente veloce, ma con un compilatore ad accesso limitato o fragile, può essere un target di deployment peggiore di un dispositivo più piccolo con toolchain pubblica, model zoo, simulatore e runtime stabile.

## Cosa significa "supportato" in questa guida

La documentazione dei fornitori usa il termine *supporto* con molta disinvoltura. Questo articolo distingue quattro livelli di evidenza:

| Livello di evidenza | Cosa dimostra | Cosa non dimostra |
|---|---|---|
| **Modello convalidato** | Il fornitore pubblica una voce, un risultato o una tabella di compatibilità specifici per il chip | Che il tuo checkpoint modificato mantenga la stessa accuratezza |
| **Esempio ufficiale** | Il fornitore mette a disposizione un tutorial o una demo end-to-end per un modello indicato | Che vengano compilate altre dimensioni, task o generazioni |
| **Funzionalità del compilatore** | L'SDK importa un framework o espone gli operatori supportati | Che un particolare grafo YOLO funzioni dall'inizio alla fine |
| **Marketing o accesso limitato** | Il prodotto è pensato per la computer vision ed esiste un SDK privato | Una compatibilità dei modelli riproducibile pubblicamente |

La distinzione conta. "Importa ONNX" non equivale a "supporta la segmentazione YOLO11". Un modello può essere analizzato, ma poi passare alla CPU, compilarsi con valori numerici di output errati, superare la memoria dell'acceleratore o perdere accuratezza durante la quantizzazione.

## Aziende e piattaforme NPU a colpo d'occhio

Le tabelle includono intenzionalmente SoC, acceleratori discreti, sensori intelligenti e target adiacenti basati su GPU o FPGA. Competono per la stessa decisione di deployment anche quando i fornitori usano nomi come NPU, AIPU, BPU, KPU, DLA, HTP, TPU o MLA.

### SoC embedded, sensori intelligenti e processori industriali

| Azienda | Chip o piattaforme pertinenti | SDK, compilatore e artefatto | Evidenze pubbliche documentate di computer vision | Accesso |
|---|---|---|---|---|
| [Amlogic](https://github.com/Amlogic-NN/amlnn-toolkit) | A311D2, S928X, S905X5/S905D5, A311Y3, C308L/C302X/C302X2, T968D4, C305X2 e A123X | AMLNN Toolkit e `libnnsdk.so`, `.adla` compilato; stack Acuity legacy separato con `.nb` per A311D | Il [model playground](https://github.com/Amlogic-NN/amlnn-model-playground) documenta YOLOv5/6/7/8/10/11, YOLOX, YOLOE, YOLO-World, PP-YOLOE, segmentazione, posa e OBB su A311D2, S905X5, A311Y3, C305X2 e A123X; gli altri chip elencati sono target del compilatore, non voci di quella matrice di supporto | GitHub e pacchetti wheel pubblici; serve comunque un BSP/driver compatibile |
| [Rockchip](https://github.com/airockchip/rknn-toolkit2) | RK3562/3566/3568, RK3576, RK3588, RV1126B | RKNN-Toolkit2, RKNN Runtime, `.rknn` | YOLOv5/6/7/8/10/11, YOLOX, YOLO-World, PP-YOLOE, segmentazione, posa e OBB nel [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) | GitHub pubblico; pacchetti wheel binari |
| [Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | Piattaforme Snapdragon e Dragonwing, QCS6490, QCS8550 e Dragonwing IQ-9075 | QAIRT/QNN, SNPE e AI Hub; binari di contesto QNN o DLC | La [raccolta di modelli AI Hub](https://github.com/qualcomm/ai-hub-models) include YOLOv3/5/6/7/8/9/10/11/26, YOLOX, YOLO-World, YOLOR, RF-DETR e modelli di rilevamento/segmentazione/posa | Documentazione pubblica; requisiti di account e SDK variabili |
| [Texas Instruments](https://github.com/TexasInstruments/edgeai-tidl-tools) | AM62A, AM67A, AM68A, AM69A e TDA4x; TDA54-Q1 in anteprima | Processor SDK Edge AI e TIDL | L'[Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo) pubblica modelli ottimizzati di rilevamento, segmentazione, posa e classificazione per gli attuali percorsi AM6xA/TDA4; non è ancora una matrice di target per TDA54-Q1 in anteprima | GitHub pubblico e Processor SDK |
| [NXP](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | i.MX 8M Plus, i.MX 93, i.MX 95 e Kinara Ara-1/Ara240 acquisiti | eIQ con TIM-VX, Vela, Neutron Converter e Ara SDK | L'[eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) include asset o ricette per YOLOv4-tiny/v8, NanoDet, CenterNet, FastestDet, SSD Lite e YOLACT; la voce YOLOv5 è solo documentazione | Componenti in parte pubblici e in parte soggetti ad account |
| [STMicroelectronics](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) | Varianti STM32N6x7, tra cui STM32N657/647, con acceleratore Neural-ART | STM32Cube AI Studio e ST Edge AI Core | L'attuale repository dei servizi elenca Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 e ST-YOLOX, oltre a posa e segmentazione | Strumenti e repository pubblici |
| [Infineon](https://documentation.infineon.com/psocedge/) | PSOC Edge E83/E84 con Cortex-M55 più Ethos-U55; acceleratore NNLite separato sul lato M33 | ModusToolbox, [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter), TFLite Micro e Arm Vela | La documentazione dell'architettura indica MobileNetV1/V2 come kernel rappresentativi e il kit AI E84 include una videocamera, ma non è stata trovata una matrice di convalida YOLO per PSOC Edge pubblicata direttamente da Infineon | Documentazione pubblica, componenti SDK e attuali kit di valutazione E84 |
| [Renesas](https://github.com/renesas-rz/rzv_drp-ai_tvm) | RZ/V2L, V2M, V2MA, V2H e V2N | DRP-AI Translator, DRP-AI TVM e RUHMI | L'[elenco dei modelli RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) convalida dimensioni YOLOv5/v8/v11/26, YOLOX e varianti di posa e segmentazione | GitHub pubblico e SDK per schede |
| [Sony](https://www.aitrios.sony-semicon.com/edge-ai-devices/imx500) | Sensore intelligente IMX500 e Raspberry Pi AI Camera | Edge-MDT, Model Compression Toolkit e packer IMX500; pacchetto `.rpk` | Il [zoo Raspberry Pi IMX500](https://github.com/raspberrypi/imx500-models) include YOLOv8n, YOLO11n, EfficientDet Lite, NanoDet+ e SSD | Percorso Raspberry Pi pubblico; si applicano le condizioni degli strumenti Sony |
| [Ambarella](https://www.ambarella.com/developer/model-garden/) | Famiglie CV72/CV75, CV5/CV52 e CV3-AD | Cooper Developer Platform, compilatore CVflow e DAG per runtime | Il model garden pubblico cita YOLOX-S, RTMDet-nano, DeepLabV3+, TopFormer, OWL-ViT e LLaVA OneVision | Principalmente riservato ai partner |
| [Synaptics](https://developer.synaptics.com/docs/sl/overview) | Astra SL1600/SL1680; SL2611/13/15/17/19; famiglia MCU SR100 separata | SyNAP `.synap` su SL16xx; Torq/IREE `.vmfb` su SL261x; SDK SR separato, orientato a Ethos-U55 | Benchmark SyNAP ufficiale YOLOv8n/v8s ed esempio Torq YOLOv8 su SL261x; le evidenze SR vanno valutate separatamente | Portale sviluppatori; alcuni download richiedono accesso |
| [MediaTek](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) | Genio 360/360P/420/520/720 con NP8 e MDLA 5.3; Genio 510/700/1200 con NP6/MDLA precedente | NeuroPilot Converter, `ncc-tflite`, Neuron Runtime e `.dla`; percorso ONNX Runtime su alcuni sistemi più recenti | L'IoT AI Hub ufficiale pubblica pagine modello e benchmark per YOLOv5 e YOLOv8, oltre a modelli di classificazione e per volti | Documentazione Yocto pubblica; i pacchetti NP8 principali e i materiali Android richiedono accesso cliente diretto/NDA |
| [Allwinner](https://docs.aw-ol.com/v853/en/npu/dev_npu/) | SoC per visione V853 con NPU Vivante da 1 TOPS | Conversione Acuity/Pegasus e runtime `viplite` | La documentazione ufficiale V853 cita YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet e reti per volti/persone | Documenti pubblici; il download del pacchetto SDK può richiedere un account |
| [Canaan/Kendryte](https://github.com/kendryte/nncase) | K230/K230D attuali; KPU legacy K210/K510 | Compilatore e runtime `nncase`; `.kmodel` | La guida nncase K230 tratta compilazione/simulazione/runtime di YOLOv5s; il catalogo demo ufficiale e l'attuale [changelog CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) documentano task YOLOv8/11/26 | Compilatore/PyPI pubblici; plug-in per il chip binario |
| [Alif Semiconductor](https://alifsemi.com/support/kits/ensemble-e7appkit/) | Ensemble E7 con due acceleratori ML, incluso Ethos-U55; [E8](https://alifsemi.com/ensemble-e8-series/) con un Ethos-U85 e due NPU Ethos-U55 | Deployment basato su Arm Vela e TFLite Micro | Alif pubblica un benchmark INT8 per rilevamento di volti YOLO-Fastest a livello di famiglia, ma non una matrice moderna YOLO estesa per ogni target | Documentazione pubblica e risorse per kit di valutazione |
| [Himax](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) | MCU endpoint AI HX6538 WiseEye2 con Cortex-M55 ed Ethos-U55 | TFLite Micro e Arm Vela, con CMSIS-NN e fallback ai kernel di riferimento | Gli esempi ufficiali [WiseEye2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) includono rilevamento, posa e classificazione YOLOv8n, rilevamento YOLO11n, face mesh e PeopleNet | GitHub e documentazione pubblici, scheda partner acquistabile |
| [Analog Devices](https://www.analog.com/en/products/max78002.html) | MCU AI MAX78000 e MAX78002 con acceleratori CNN a basso consumo | [`ai8x-training`](https://github.com/analogdevicesinc/ai8x-training), `ai8x-synthesis`/`izer` e MSDK; C e pesi generati | I materiali ufficiali coprono identificazione/rilevamento di volti, RetinaNet, Visual Wake Words, classificatori e riconoscimento di azioni; non è stato trovato un deployment YOLO pubblicato direttamente da Analog Devices | Strumenti GitHub pubblici, documentazione e schede di valutazione |
| [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo) | Schede BPU RDK X3/X5/Ultra e nuove serie S100 | OpenExplorer/Algorithm Toolchain e `hbm_runtime`; `.bin` per X5, `.hbm` per l'attuale `rdk_s` | Il ramo attuale `rdk_x5` documenta YOLOv5/v5u/v8/v9/v10/11/12/13/26, YOLOE, YOLO-World, segmentazione, posa, classificazione, OCR e CLIP su RDK X5; X3 e serie S usano rami separati | GitHub pubblico; alcuni pacchetti toolchain sono specifici della piattaforma |
| [AXERA](https://github.com/AXERA-TECH/ax-samples) | AX650, AX637, AX630C, AX620Q e AX615 | Compilatore Pulsar2 e AXEngine; `.axmodel` | Gli esempi ufficiali correnti coprono YOLOv5/6/7/8/9/10/11/13/26, YOLOX e YOLO-World; task e target variano in base al chip | Esempi e documentazione pubblici; compilatore distribuito separatamente |
| [SOPHGO](https://github.com/sophgo/tpu-mlir) | BM1684/1684X/1688/1690 e CV186X/CV18xx | TPU-MLIR e runtime SOPHON; `.bmodel` o `.cvimodel` | Le demo ufficiali coprono YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, OBB, segmentazione, volti, SAM e OCR | Compilatore open source e runtime del fornitore |
| [Huawei Ascend](https://www.hiascend.com/en/software/cann) | Ascend 310/310P/310B e prodotti edge Atlas 200I/300I | CANN, compilatore ATC e AscendCL; `.om` | L'[Ascend ModelZoo](https://github.com/Ascend/modelzoo) ufficiale include YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN e modelli di segmentazione | Documentazione pubblica; pacchetti CANN e kernel devono corrispondere al target |
| [Cambricon](https://github.com/Cambricon/magicmind_cloud) | MLU370-X4/S4 nella matrice pubblica citata; MLU270 è hardware legacy non coperto dalla matrice | Neuware e MagicMind | La matrice MagicMind 1.7 del repository elenca YOLOv3/4/5/7/8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN e modelli di segmentazione | Esempi storici pubblici; accesso all'SDK/container attuale limitato |
| [Sunplus](https://sunplus.atlassian.net/wiki/spaces/C3/pages/1971126471/SP7350%2BSpecification) | SP7350/C3V, circa 4.1-4.6 TOPS dichiarati | Docker NPU Vivante Acuity, runtime SNNF e `.nb` | La documentazione ufficiale include YOLOv5 e un deployment YOLOv8 personalizzato end-to-end ([YOLOv8](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350)) | Fonti e documenti pubblici, ecosistema di schede |
| [ESWIN Computing](https://www.eswincomputing.com/en/news/info/96.html) | SoC RISC-V EIC7700/EIC7700X e dual-die [EIC7702/EIC7702X](https://www.eswincomputing.com/en/news/info/104.html) | ENNP: EsQuant, compilatore EsAAC attuale, simulatore e runtime ESSDK; `.model`; la documentazione precedente usava `ennc-compile` | La documentazione ENNP ospitata da Milk-V include un tutorial end-to-end [YOLOv3](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3), oltre a esempi MobileNetV2 e ResNet | Documentazione mista del produttore e del produttore della scheda; download regionali |
| [Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m55-mcus/m55m1-series/index.html) | MCU M55M1 con Ethos-U55-256 | TFLite Micro, Arm Vela e NuEdgeWise/NuML | Gli esempi ufficiali [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) citano YOLOv8-nano, YOLOX-nano, YOLO Fastest, SSD-MobileNet e modelli di classificazione | Toolchain MCU perlopiù pubblica |
| [Realtek](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Piattaforma camera AmebaPro2 RTL8735B / AMB82-mini | SDK Arduino/FreeRTOS, VoE e NeuralNetwork API; `.nb` | I modelli inclusi sono YOLOv3-tiny, YOLOv4-tiny, YOLOv7-tiny, SCRFD e MobileFaceNet | Runtime pubblico; per accedere al [convertitore personalizzato](https://ameba-doc-arduino-sdk.readthedocs-hosted.com/en/latest/FAQ/offline_ai_model_conversion_steps.html) occorre contattare l'azienda |
| [Telechips](https://docs.topst.ai/product/p/ai) | Scheda TCC7500 / TOPST AI, 8 TOPS dichiarati | TC-NN-Toolkit / Enlight SDK; intermedio `.enlight` e pacchetto di deployment compilato | Un [progetto TOPST ufficiale](https://docs.topst.ai/blog/31) ha distribuito YOLOv8s; le conversioni UFLD v1/v2 sono fallite su layer non supportati ed è stato proposto solo un workaround con suddivisione e postelaborazione. [YOLOv4](https://community.topst.ai/t/segmentation-fault/358) e un [altro flusso YOLOv8](https://community.topst.ai/t/segmentation-fault-error/415) compaiono in discussioni della community | Documentazione della scheda pubblica; download di compilatore/operatori spesso soggetti a permesso |
| [T-Head](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | TH1520, 4 TOPS INT8 dichiarati, su LicheePi 4A | Compilatore HHB e CSI-NN2/SHL; `hhb.bm` più codice/parametri generati | Gli esempi ufficiali del produttore della scheda eseguono YOLOv5n/s e MobileNetV2 sulla NPU | Esempi pubblici; toolchain datata e manutenzione incerta |
| [SigmaStar](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | SoC camera intelligente attuale SSU9383CM e precedenti famiglie IPU | Runtime MI_IPU attuale; un [vecchio toolchain SGS_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSC9381G_9351_Pudding-ULS00V040-20210913/customer/development/dla/tools.html) produceva `.sim` poi convertito in `sgsimg.img` | Un [postprocessore di un vecchio SDK ufficiale](https://wx.comake.online/doc/doc/Sigmastar_SDK_v1.2.2/module/User_Guide/Common/SigmaStar_Post_Processing_Module.html) cita SSD e YOLOv1/2/3; la compatibilità pubblica di modelli e artefatti con SSU9383CM non è verificata | Silicio attuale, ma le evidenze pubbliche su compilatore/modelli riguardano generazioni diverse |
| [HiSilicon](https://www.hisilicon.com/cn/products/smart-vision/machine-vision/hi3516cv610) | SoC smart vision Hi3516CV610/DV500, Hi3519DV500 e Hi3403V100 | ATC verso `.om` con runtime NNN/SVP-NNN sulle schede | La matrice HiSpark specifica per target, soprattutto per Hi3403 e Hi3591P, indica YOLOv3 fino a YOLO11, posa, segmentazione, OBB, OCR e stima della profondità; non convalida tutti i SoC della riga | Attiva e distinta da Ascend; i modelli del repository sono indicati solo per uso non commerciale |

### Acceleratori edge e moduli dedicati

| Azienda | Silicio pertinente | SDK e artefatto | Evidenze pubbliche sui modelli | Accesso |
|---|---|---|---|---|
| [Hailo](https://github.com/hailo-ai/hailo_model_zoo) | Famiglie Hailo-8, Hailo-8L, Hailo-10H e Hailo-15 | Dataflow Compiler e HailoRT; `.hef` | YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 e YOLO26, oltre a YOLOX, DAMO-YOLO, SSD, EfficientDet, segmentazione, posa e OBB, con tabelle specifiche per generazione | Model Zoo pubblico; compilatore tramite Developer Zone |
| [Axelera AI](https://docs.axelera.ai/sdk/reference/models/model-zoo/) | Prodotti Metis in distribuzione; prodotti annunciati della gamma Europa e Titania | Voyager SDK pubblico; Pipeline Builder alpha `.axm`/`.axe`, bundle `.axmodel` classici | YOLOv3/5/7/8/9/10/11/26, YOLOX, YOLO-NAS, OBB, posa, segmentazione e molti modelli non YOLO convalidati | SDK pubblico su GitHub; supporto clienti soggetto ad account |
| [DEEPX](https://developer.deepx.ai/modelzoo/) | DX-M1 e DX-M1M | DXNN SDK, DX-COM e DX-RT; `.dxnn` | Il Model Zoo riportava 354 voci con DX-COM 2.4.0/DX-RT 3.4.0 alla verifica del 15 agosto 2026, inclusi YOLOv3 fino a YOLO11 e YOLO26, YOLOX, SSD, EfficientDet, NanoDet, segmentazione, posa e OBB | Risorse e suite per sviluppatori pubbliche |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | Acceleratore MX3 e moduli multi-chip | MemryX SDK, Neural Compiler e runtime; `.dfp` | Esempi ufficiali e note di rilascio coprono rilevamento, segmentazione e posa, inclusi YOLOv10, YOLO11 e YOLO26 | Documentazione e SDK pubblici |
| [Kneron](https://doc.kneron.com/docs/) | KL520, KL530, KL630, KL720 e KL730 hanno documentazione del compilatore attuale; KL830 appare in alcune API PLUS ma non nell'elenco attuale dei target di compilazione | Kneron PLUS e Model Toolchain; `.nef` sui target documentati | Il [flusso YOLO ufficiale](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) si basa su Tiny-YOLOv3; altri materiali trattano YOLOv5 | Documentazione pubblica; pacchetti e download variano per chip |
| [SiMa.ai](https://docs.sima.ai/pages/palette/modelsdk.html) | MLSoC e piattaforma Modalix da 50 TOPS in produzione | Palette, ModelSDK, MLA Compiler e ModelExecutor | Le release pubbliche convalidano YOLOv7/v8, YOLOX, posa/segmentazione, DETR, Mask R-CNN ed EfficientDet | Documentazione pubblica; SDK commerciale |
| [EdgeCortix](https://www.edgecortix.com/en/hardware) | SAKURA-II, acceleratore dichiarato da 60 TOPS, in prodotti M.2 e PCIe | Compilatore MERA e framework | Il fornitore dichiara supporto dalla computer vision all'AI generativa, ma non pubblica una matrice YOLO attuale abbastanza precisa | Richiesta commerciale di prova/ordine; convalida con partner |
| [BrainChip](https://brainchip.com/metatf-dev-tools/) | Coprocessore/modulo M.2 AKD1500 in distribuzione; piattaforme AKD1000 legacy; IP Akida 2 | Pacchetti MetaTF: `akida-models`, `quantizeml`, `cnn2snn` e runtime `akida` | La scheda modello Akida 2 cita un rilevatore YOLOv2 AkidaNet0.5, CenterNet, AkidaUNet e riconoscimento facciale; i risultati non convalidano automaticamente AKD1500 | Pacchetti Python principali pubblici; mappatura hardware specifica per target |
| [Blaize](https://www.blaize.com/products/) | Prodotti P1600, Pathfinder e Xplorer | Picasso SDK, NetDeploy e AI Studio | La computer vision è un mercato di riferimento, ma non è stata trovata una matrice pubblica verificabile dei modelli | Commerciale/accesso limitato |
| [Google Coral](https://github.com/google-coral/edgetpu) | Prodotti Edge TPU legacy USB, PCIe, M.2 e Dev Board; IP [Coral NPU](https://github.com/google-coral/coralnpu) open source separato e attivo | Edge TPU Compiler/`libedgetpu`/PyCoral legacy; la nuova Coral NPU RISC-V espone IP, RTL/simulazione ed esempi ELF | Gli esempi legacy si concentrano su SSD MobileNet, classificazione, DeepLab e MoveNet; la nuova IP non ha una matrice YOLO pubblica e non è un successore diretto dei prodotti Edge TPU | Repository principali legacy archiviati; IP Coral NPU nuova in sviluppo |
| [Lattice Semiconductor](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | FPGA ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E e Avant-X | sensAI Studio, Neural Network Compiler e IP di acceleratore configurabile | La tabella attuale del compilatore elenca YOLOv1, YOLOv5, YOLOv8 e YOLO11 in modalità specifiche per target, oltre a SSD, MobileNetV2-SSD, ResNet ed ENet | Compilatore/manuali pubblici; condizioni IP variabili e Advanced CNN Accelerator commerciale |
| [Mobilint](https://www.mobilint.com/sdk-qb) | Acceleratori REGULUS da 10 TOPS e ARIES MLA100 da 80 TOPS | SDK qb; compilatore INT8 e `.mxq` | Il [model zoo pubblico](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) elenca rilevamento YOLOv3/5/7/8/9/10/11/12/26 e varianti di segmentazione, posa e OBB | Documentazione/modelli pubblici; SDK e hardware commerciali |
| [Rebellions](https://rebellions.ai/developers/) | ATOM+ CA22 e ATOM-Max CA25 attivi; ATOM CA02/ATOM+ CA12 a fine vita; stato toolchain ATOM-Lite CA21 incerto | Compilatore/runtime/profiler RBLN SDK; `.rbln` | Le release di supporto ufficiali citano YOLOv3, famiglie YOLOv5/6/7/8 e tutorial YOLOv8 attuali | Documentazione pubblica; il pacchetto compilatore richiede credenziali del portale |
| [FuriosaAI](https://furiosa.ai/warboy) | Warboy Gen1 orientata alla visione; generazione RNGD separata | Furiosa SDK; `.enf` INT8 su Warboy, `.fxb` separato su RNGD | Lo zoo Warboy documenta SSD, YOLOv5M/L e YOLOv7-w6-pose; la roadmap RNGD indica completato il supporto YOLOv8m nel quarto trimestre 2024, ma senza una matrice pubblica corrente e dettagliata per la visione | Accesso IAM/account; le due generazioni vanno tenute distinte |

### Piattaforme adiacenti che gli sviluppatori confrontano con le NPU

| Azienda | Hardware | Stack di deployment | Perché è nel confronto |
|---|---|---|---|
| [NVIDIA](https://github.com/NVIDIA-AI-IOT/deepstream_tools/tree/main/yolo_deepstream) | GPU Jetson Orin più DLA; Jetson Thor con GPU senza DLA | JetPack, TensorRT e DeepStream; `.engine` | Deployment di visione molto maturo; gli strumenti DeepStream ufficiali documentano YOLOv4/v7/v8/v9/11, inclusa una configurazione YOLO11 OBB. Molti risultati usano la GPU e i layer Orin non supportati dal DLA possono passare alla GPU. |
| [AMD](https://vitisai.docs.amd.com/en/6.2/) | Target DPU Kria/legacy; Vitis AI 6.2 GA per Versal AI Edge Gen1 VEK280/VE2802 e Gen2 VEK385; NPU client Ryzen AI separata | `.xmodel` legacy; snapshot NPU vincolato al target Gen1; directory cache compilata o pacchetto `.rai` di produzione Gen2 | Vitis AI pubblica materiali per la visione, ma DPU legacy, Versal Gen1, Versal Gen2 e Ryzen AI hanno contratti di compilazione e deployment distinti. |
| [Microchip](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox) | FPGA PolarFire SoC con IP acceleratore CoreVectorBlox configurabile | VectorBlox SDK 3.1 pubblico e Libero; asset `.vnnx`, `.hex` e `.ucomp` | Gli attuali [tutorial](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) coprono YOLOv5n, rilevamento/classificazione/OBB/posa/segmentazione YOLOv8, YOLOv9t e altri modelli; SDK 3.1 è al momento circoscritto a PolarFire SoC Video Kit. |
| [Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) | NPU Core Ultra, CPU e GPU integrata/discreta | OpenVINO IR o cache del modello compilato | OpenVINO offre un percorso pubblico cross-XPU; consulta le colonne NPU della [matrice dei modelli verificati](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html), senza presumere che ogni esempio YOLO OpenVINO sia convalidato su NPU. |
| [Apple](https://developer.apple.com/machine-learning/core-ml/) | Apple Neural Engine nei chip serie A da A11 in poi e serie M | Core ML e `coremltools`; `.mlpackage`/`.mlmodelc` | NPU consumer molto accessibile tramite Core ML, ma la ripartizione esatta tra CPU/GPU/Neural Engine è astratta. |
| [GreenWaves Technologies](https://github.com/GreenWaves-Technologies/nn_menu_gap9) | GAP9 con motore neurale NE16 | GAP SDK, NNTool e codice generato da AutoTiler | I repository ufficiali includono MobileNet SSD, rilevamento di volti, classificazione e un [rilevatore di persone YOLOX](https://github.com/GreenWaves-Technologies/yolox_people_detection) dedicato. L'SDK completo è limitato ai clienti qualificati. |
| [Syntiant](https://www.syntiant.com/hardware) | NDP200 in produzione di massa e processori Neural Decision NDP250 in campionatura | SDK Syntiant e pacchetti di deployment | Processori per visione/sensori a bassissimo consumo sempre attivi: NDP200 è documentato sotto 1 mW, mentre il riconoscimento immagini NDP250 è documentato sotto 30 mW. Non è stata trovata una matrice YOLO pubblica estesa. |

## Amlogic: due generazioni NPU, non una

Amlogic merita un approfondimento perché sul web spesso si mescolano prodotti incompatibili. L'azienda ha un percorso precedente basato su VeriSilicon e il nuovo percorso proprietario ADLA. Usano compilatori, artefatti e runtime diversi.

| Generazione | Chip rappresentativi | NPU e toolchain | Artefatto compilato | Stato pratico |
|---|---|---|---|---|
| Legacy | A311D e S905D3, spesso sulle Khadas VIM3/VIM3L | VeriSilicon Vivante VIPNano-QI, toolkit Acuity, KSNN e vecchio `aml_npu_sdk` | `.nb` dentro una directory di output `nbg_unify` | Schede ed esempi esistenti; integrazione legacy separata |
| ADLA2 attuale | C308L/C302X, S928X, A311D2, T968D4, S905X5/S905D5 e C302X2 | Amlogic ADLA2, AMLNN Toolkit e NNSDK2 | `.adla` specifico per target; W8A8 o W8A16 | Stack attuale focalizzato sugli interi |
| ADLA3 attuale | A311Y3, C305X2 e A123X | Amlogic ADLA3, AMLNN Toolkit e NNSDK2 | `.adla` specifico per target; W4A8, W8A8, W4A16, W8A16 o W16A16 | Aggiunge funzionalità native INT4, FP16 e BF16 |

La [scheda tecnica A311D](https://dl.khadas.com/products/vim3/datasheet/a311d-datasheet.pdf) identifica la NPU originale INT8 da 5 TOPS. La vecchia [pagina applicativa Khadas VIM3](https://docs.khadas.com/products/sbc/vim3/npu/npu-app) e la [panoramica NPU](https://docs.khadas.com/products/sbc/vim3/npu/start) documentano DenseNet CTC, MTCNN, RetinaFace, YOLOFace, YOLOv2, YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv7-tiny e YOLOv8n; la panoramica documenta inoltre YOLOv8n-pose e indica FaceNet come deprecato. Sono esempi reali, ma riguardano il vecchio ecosistema Acuity `.nb`, non gli attuali chip ADLA. A311D non è A311D2 e C305X non è C305X2.

C'è anche una seconda insidia hardware. La [guida alle revisioni Khadas VIM4](https://docs.khadas.com/products/sbc/vim4/configurations/identify-version) spiega che la scheda originale V12/A311D2 revisione B non ha una NPU; la NPU dichiarata da 3.2 TOPS appare sulle schede V13A e successive, con componente A311D2-N0D revisione C. Il solo nome del prodotto non basta per scegliere un dispositivo di test.

### Lo stack AMLNN e ADLA attuale

L'attuale [AMLNN Toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/7d3cc9a36179b756ce79c953c14d43acab1f2af5) copre conversione dei modelli, quantizzazione, compilazione, inferenza e profiling. La [guida utente di maggio 2026](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/02_Amlogic_NPU_User_Guide_V0.1.pdf), fissata a una revisione specifica, documenta importatori per ONNX, TFLite in virgola mobile o quantizzato, TorchScript `.pt` e PyTorch 2 ExportedProgram/PT2. Nella guida dettagliata i percorsi TensorFlow, Paddle e Keras sono indicati come pianificati, anche se la panoramica del repository usa formulazioni più ampie. Il compilatore produce `.adla`. Il deployment Python usa AMLNN o `amlnn_edge_toolkit_lite`; il C/C++ nativo collega `libnnsdk.so` tramite `nnsdk2.h`. Lo sviluppo host può usare ADB con `nnserver`; il runtime supporta anche Android, Buildroot, Yocto e alcuni ambienti Debian/Armbian.

Gli identificativi di piattaforma pubblicati dal toolkit sono:

| ID target | Piattaforma |
|---:|---|
| `001` | C308L / C302X |
| `002` | S928X |
| `003` | A311D2 |
| `004` | T968D4 |
| `005` | S905X5 / S905D5 |
| `006` | C302X2 |
| `007` | A311Y3 |
| `008` | C305X2 / A123X nella documentazione dettagliata e nella matrice modelli |

Il campo target non è cosmetico. Amlogic richiede che `.adla`, `nnsdk2.h`, `libnnsdk.so`, BSP e generazione di compilatore/runtime corrispondano. La [guida rapida](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/01_Amlogic_NPU_Quick_Start_guide_V0.1.pdf) fissata a una revisione richiede ADLA driver 2.0.2 o superiore, NNSDK 3.0.0 o superiore e NNSDK2 1.0.0 o superiore; la [guida al deployment Android](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/06_Amlogic_NPU_Android_Deployment_Guide_V0.1.pdf) chiarisce i rapporti tra libreria, header, driver e BSP. Considera `.adla` un artefatto di build vincolato all'ABI, non un modello portabile.

Anche la quantizzazione dipende dalla generazione NPU. I target ADLA2 da 001 a 006 documentano la compilazione W8A8 e W8A16. I target ADLA3 007 e 008 aggiungono le combinazioni W4A8, W4A16, W8A8, W8A16 e W16A16, con supporto nativo INT4, FP16 e BF16 nella [guida agli operatori](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/05_Amlogic_NPU_Support_Operator_List_V0.1.pdf). Amlogic consiglia da 200 a 500 campioni di calibrazione rappresentativi. L'opzione con dati casuali è esplicitamente destinata ai test di prestazioni, non a una quantizzazione che preservi l'accuratezza.

### Copertura dei modelli Amlogic documentata

Il [model playground Amlogic fissato a una revisione](https://github.com/Amlogic-NN/amlnn-model-playground/blob/03ca63ef20c4f1560722de26ab910826884f334a/README.md) è un'evidenza molto più solida di una generica affermazione sul supporto ONNX. La sua matrice di supporto/esempi pubblicata cita A311D2, S905X5, A311Y3, C305X2 e A123X. L'accettazione da parte del compilatore degli altri ID target non dimostra che l'intero elenco sia stato convalidato su di essi.

| Task | Modelli documentati |
|---|---|
| Classificazione | MobileNetV2 e ResNet50-v2; DINO su ADLA3 |
| Rilevamento di oggetti | PP-YOLOE, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv10, YOLO11, YOLOE, YOLO-World, YOLOX e rilevamento QR |
| Volti e gesti | RetinaFace e Gesture Recognition |
| Segmentazione | DeepLabV3, PP-LiteSeg, YOLOv5-seg e YOLOv8-seg |
| Rilevamento orientato | YOLOv8 OBB |
| Posa | Rilevatore/landmark BlazePose e YOLOv8 pose |
| OCR e voce | LPRNet, varianti PaddleOCR, Whisper Tiny e SenseVoice su alcuni chip più recenti |
| Transformer e multimodalità recenti | DETR, MobileSAM, CLIP/MobileCLIP e alcuni esempi LLM/VLM a bassa precisione su piattaforme recenti |

Lo stesso repository pubblica i seguenti valori W8A8 di runtime dei modelli. Sono misurazioni del fornitore sulla rete neurale eseguita sulla NPU, non pipeline complete da videocamera.

| Modello e input | S905X5 | A311D2 | A311Y3 |
|---|---:|---:|---:|
| YOLOv8n, 640 x 640 | 101.72 FPS | 95.14 FPS | 191.06 FPS |
| YOLOv8s, 640 x 640 | 42.33 FPS | 42.77 FPS | 83.08 FPS |
| YOLOv8m, 640 x 640 | 19.67 FPS | 19.82 FPS | 35.30 FPS |
| YOLOv8l, 640 x 640 | 10.53 FPS | 10.12 FPS | 18.37 FPS |
| YOLO11n, 640 x 640 | 41.14 FPS | 41.48 FPS | 62.24 FPS |

Le riserve sono particolarmente importanti. La guida agli operatori di Amlogic colloca NMS nel software sia su ADLA2 sia su ADLA3. Il README definisce questi valori come risultati nativi di runtime del modello NPU, escludendo preelaborazione e postelaborazione; non specifica se includano tutti i trasferimenti di memoria. Le righe sull'accuratezza YOLO usano sottoinsiemi COCO di 300 immagini, non l'intero set di validazione; le righe di classificazione usano ImageNet `val1000`. Le tabelle riportano inoltre una latenza YOLOv8n A311Y3 diversa da quella della tabella FPS, in condizioni che il repository non chiarisce. Non sono indicati clock, modalità di alimentazione, raffreddamento, stato termico a regime, versione esatta di SDK/BSP e durata del test. Usa i valori per capire uno stack del fornitore, non per classificarlo rispetto ad altri.

### Perché Amlogic è strategicamente interessante

Amlogic combina quattro caratteristiche insolite: un'ampia presenza nei SoC embedded, uno stack di compilazione appena reso pubblico, una matrice di modelli attuale che si sovrappone molto ai task di LibreYOLO e una presenza limitata di integrazioni di prima classe nelle principali librerie di training. Anche i repository attuali sono recenti: sono apparsi pubblicamente tra la fine del 2025 e l'inizio del 2026, con manuali sostanziali datati da maggio a luglio 2026. Questo crea un'opportunità concreta per muoversi per primi e un corrispondente rischio di stabilità delle API.

Un'integrazione pulita dovrebbe usare l'esportazione ONNX deterministica di LibreYOLO come input del compilatore, richiedere immagini di calibrazione rappresentative per le build a bassa precisione, produrre `.adla` con un manifest di metadati e caricarlo tramite un adapter NNSDK2. A311D legacy dovrebbe usare un target `.nb` chiaramente distinto, perché presentare `.nb` e `.adla` come un unico backend causerebbe incompatibilità silenziose. I repository Amlogic riportano licenze Apache-2.0 di primo livello, ma prima che LibreYOLO ne ridistribuisca mirror occorre confermare direttamente i diritti sui pacchetti wheel inclusi, sulle librerie condivise, su `nnserver` e sui binari Android AAR.

## Gli ecosistemi pubblici di deployment più solidi

Le aziende seguenti offrono attualmente la combinazione più chiara di hardware reperibile, materiali tecnici pubblici ed evidenze su modelli specifici. Questo non significa che siano sempre più veloci. Significa che è più facile valutare le loro dichiarazioni di compatibilità prima di acquistare l'hardware.

### Hailo

Hailo è un esempio di riferimento per un ecosistema di acceleratori per la visione dedicati. I modelli vengono analizzati in un Hailo Archive, ottimizzati e quantizzati usando immagini rappresentative, poi compilati in un file Hailo Executable Format (`.hef`). Le applicazioni caricano il file HEF tramite HailoRT.

Il [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) contiene ricette, modelli preaddestrati, configurazioni di postelaborazione e dati prestazionali per rilevamento, segmentazione, classificazione, posa e altri task. La copertura YOLO include YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 e YOLO26, oltre a YOLOX, DAMO-YOLO, SSD ed EfficientDet. La [tabella di object detection Hailo-8](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.19.0/docs/public_models/HAILO8/HAILO8_object_detection.rst), fissata a una versione, è una fonte di compatibilità migliore di una pagina prodotto generica; l'[applicazione standalone](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/README.md) documenta pipeline YOLO distribuibili.

C'è un'importante distinzione tra versioni. Hailo-8 e Hailo-8L usano la linea precedente Model Zoo 2.x, Dataflow Compiler 3.x e HailoRT 4.x, mentre Hailo-10 e Hailo-15 usano la generazione attuale 5.x. Ricette, comportamento degli analizzatori e file HEF non sono intercambiabili solo perché tutti i dispositivi si chiamano Hailo.

Hailo-15 ha anche un livello applicativo distinto. Usa Vision Processor Software Package e Hailo Media Library, non il percorso host in stile TAPPAS di Hailo-8/10H. Il repository di riferimento pubblico [`hailo-camera-apps`](https://github.com/hailo-ai/hailo-camera-apps) è stato archiviato il 3 maggio 2026, mentre [`hailo-apps-core`](https://github.com/hailo-ai/hailo-apps-core) resta un framework applicativo pubblico separato. L'archiviazione di un repository non prova la fine del ciclo di vita del prodotto, ma per i progetti Hailo-15 va verificato nel Developer Zone quale pacchetto applicativo sia attuale.

La [guida al deployment Hailo](/docs/export/hailo) di LibreYOLO si ferma intenzionalmente al passaggio statico ONNX, perché il compilatore proprietario non può essere incluso come dipendenza Python. Questo è il limite corretto tra la produzione di un grafo adatto e l'affermazione di avere un HEF testato.

### Rockchip

Rockchip ha una delle maggiori presenze nell'embedded Linux per hobbisti e aziende, soprattutto tramite schede RK3588, RK3576 e RK356x. Il [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) pubblico converte e quantizza modelli su un host Linux x86; RKNN Runtime o Toolkit-Lite eseguono il file `.rknn` risultante sulla scheda.

Il [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) è particolarmente chiaro su chip, famiglie di modelli e precisione. La tabella attuale elenca percorsi FP16 e INT8 per YOLOv5/6/7/8/10/11, YOLOX, PP-YOLOE, YOLO-World e varianti di segmentazione YOLOv5/v8; YOLOv8 OBB e YOLOv8 pose sono elencati solo per INT8. Copre anche OCR, volti e altre reti di segmentazione.

LibreYOLO dispone già di un [esportatore RKNN](/docs/export/rknn) diretto e prudente. Convalida quattro varianti esatte di rilevamento su RK3588, può confrontare il simulatore del compilatore con ONNX Runtime e rifiuta famiglie non convalidate, invece di equiparare una build riuscita a predizioni corrette. Questa dichiarazione di supporto circoscritta è un buon modello per ogni futuro backend NPU.

### Axelera AI

Axelera AI, spesso scritto erroneamente "Accelera", produce l'AIPU Metis e lo vende in prodotti M.2, PCIe e multi-chip. La famiglia Metis è acquistabile pubblicamente; Europa e Titania sono prodotti annunciati per la gamma, quindi la disponibilità non va appiattita in un solo stato. Il [Voyager SDK è pubblico su GitHub](https://github.com/axelera-ai-hub/voyager-sdk) e gestisce selezione, compilazione, quantizzazione, esecuzione e pipeline applicative; il supporto clienti richiede ancora un account.

Il [Voyager Model Zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo/) pubblico è tra i più informativi del settore. Elenca variante, dimensione d'input, precisione, accuratezza e prestazioni misurate per YOLOv3, YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 n/s/m/l/x, YOLOX, YOLO-NAS, OBB, posa e segmentazione, oltre a molti modelli di classificazione e predizione densa. Pubblicare la perdita di quantizzazione e l'accuratezza insieme alla velocità è più utile dei soli TOPS di picco.

I nomi degli artefatti sono cambiati con la generazione API. Il flusso di compilazione alpha [Pipeline Builder](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) produce un modello `.axm` e può impacchettare una pipeline portabile come `.axe`. La [documentazione della pipeline classica](https://docs.axelera.ai/sdk/reference/pipeline/model-formats/) descrive `.axmodel` più metadati del modello e un manifest. L'integrazione deve rilevare la generazione Voyager installata, invece di codificare un'unica estensione.

### Qualcomm

Qualcomm raggiunge un enorme numero di dispositivi, ma l'espressione "NPU Qualcomm" nasconde interfacce diverse. Gli sviluppatori incontrano Hexagon HTP tramite QAIRT/QNN, il flusso SNPE precedente, i servizi di compilazione Qualcomm AI Hub, LiteRT oppure il provider QNN di ONNX Runtime, a seconda del dispositivo e della categoria del prodotto.

Il repository ufficiale [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) offre solide evidenze a livello di modello. Pubblica pacchetti ottimizzati per rilevamento/segmentazione/posa YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11 e YOLO26, oltre a YOLOX, YOLO-World, YOLOR, RF-DETR e molti modelli di classificazione, profondità e posa. AI Hub consente anche profiling specifico per dispositivo, importante perché un modello compilato per una generazione HTP non è automaticamente portabile a un'altra.

Il costo principale d'integrazione deriva dall'ampiezza della matrice: Android o Linux embedded, componenti ordinabili QCS o Snapdragon consumer, architettura HTP, versione QNN/QAIRT e schema di quantizzazione sono tutti rilevanti. Le pagine attuali di Qualcomm sono inoltre in disaccordo sullo stato di [Dragonwing IQ-9075](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075): la pagina prodotto lo indica come Active e la sua [EVK](https://www.qualcomm.com/developer/hardware/qualcomm-iq-9075-evaluation-kit-evk) è valutabile, mentre il [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) lo indica ancora come Sampling e riporta una longevità fino al 2038. Il prefisso di ordinazione/SKU è QCS9075; Qualcomm dichiara configurazioni da 50 e 100 TOPS INT8 dense più supporto Ubuntu/Yocto. Lo stato di produzione va confermato per lo SKU ordinato esatto e un'integrazione LibreYOLO dovrebbe registrare quello SKU, non creare una cartella generica "Qualcomm".

### DEEPX

Gli acceleratori DX-M1/DX-M1M di DEEPX usano l'SDK DXNN. DX-COM compila e quantizza il modello, DX-RT lo esegue e l'artefatto distribuito usa il formato `.dxnn`. L'azienda pubblica [DX-AllSuite](https://github.com/DEEPX-AI/dx-all-suite), esempi applicativi in [DX-APP](https://github.com/DEEPX-AI/dx_app) e un ampio [Model Zoo online](https://developer.deepx.ai/modelzoo/).

Alla verifica del 15 agosto 2026 il catalogo pubblico riportava 354 voci con DX-COM 2.4.0 e DX-RT 3.4.0. La copertura documentata comprende YOLOv3 fino a YOLO11 e YOLO26, YOLOX, SSD, EfficientDet, NanoDet, DAMO-YOLO, segmentazione YOLO, posa e box orientati. DEEPX è un target d'integrazione particolarmente plausibile perché i confini tra compilatore e runtime e l'artefatto di deployment hanno nomi chiari, con un'ampia copertura pubblica della visione.

## I SoC industriali sono ecosistemi diversi

I fornitori industriali spesso offrono cicli di vita più lunghi e una migliore integrazione con videocamere, safety e sistemi real-time rispetto ai SoC per schede maker. I loro stack AI possono essere più complessi perché lo stesso fornitore può offrire contemporaneamente diverse architetture NPU senza relazione tra loro.

### Texas Instruments

Gli attuali processori edge AI di TI includono AM62A, AM67A, AM68A, AM69A e dispositivi TDA4 correlati. Il percorso software combina Processor SDK Linux, compilatore/runtime TIDL, [Edge AI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) ed [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo). TIDL può integrarsi con ONNX Runtime, TensorFlow Lite e altri runtime applicativi, delegando i sottografi supportati.

TI pubblica ben più di un'affermazione sull'importazione di framework: lo zoo contiene modelli convertiti e metadati prestazionali per rilevamento di oggetti, segmentazione, posa, classificazione, profondità e altri task. TI avverte inoltre che gli artefatti del model zoo sono punti di partenza per lo sviluppo e non asset automaticamente pronti per la produzione. È una precisazione utile, spesso assente nei confronti tra fornitori.

TI indica anche [TDA54-Q1](https://www.ti.com/product/TDA54-Q1) come Preview, con fino a quattro NPU C7 e fino a 400 TOPS per quel componente; la più ampia famiglia TDA5 è dichiarata fino a 1.200 TOPS. Sono dichiarazioni del fornitore per una nuova generazione, non autorizzano a trasferire la convalida TIDL dei modelli AM6xA/TDA4 a TDA54-Q1 prima della pubblicazione di una matrice specifica per target.

### NXP

NXP richiede almeno quattro descrizioni di backend:

| Target NXP | Acceleratore | Percorso compilatore/runtime |
|---|---|---|
| i.MX 8M Plus | NPU VeriSilicon Vivante da 2.3 TOPS | eIQ con TIM-VX/VX delegate |
| i.MX 93 | Arm Ethos-U65 | TFLite quantizzato e Arm Vela |
| i.MX 95 | NPU NXP eIQ Neutron | Neutron Converter ed eIQ runtime |
| Ara-1 / [Ara240](https://www.nxp.com/products/ARA240) | NPU discreta derivata da Kinara; Ara240 dichiara fino a 40 eTOPS | Ara SDK, integrato nella più ampia offerta eIQ |

L'[eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) contiene asset o ricette per YOLOv4-tiny, YOLOv8, NanoDet, CenterNet, FastestDet, SSD Lite, YOLACT e altri modelli; la voce YOLOv5 è stata aggiunta solo come documentazione. Una voce convalidata per un backend non dimostra la compatibilità degli altri tre. Le indicazioni NXP per [esportare YOLO sulle piattaforme i.MX](https://community.nxp.com/t5/i-MX-Processors-Knowledge-Base/Exporting-YOLO-Models-for-NXP-i-MX-Platforms/ta-p/2381361) illustrano questo problema di conversione specifico per target. NXP afferma che il monolitico [eIQ Toolkit non riceve aggiornamenti dopo la versione 1.17 del terzo trimestre 2025](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/eIQ-FAQ/ta-p/1099741); i flussi attuali usano pacchetti autonomi come eIQ Neutron SDK.

NXP [ha completato l'acquisizione di Kinara nell'ottobre 2025](https://media.nxp.com/news-releases/news-release-details/nxp-completes-acquisitions-aviva-links-and-kinara-advance/). La [scheda tecnica Ara240](https://www.nxp.com/docs/en/fact-sheet/ARA240DNPUFS.pdf) dichiara fino a 40 eTOPS e riporta 313 immagini al secondo per YOLOv8n. NXP elenca Ara SDK ed eIQ Toolkit nella pagina prodotto, ma ciò non dimostra l'intercambiabilità degli artefatti con il separato [percorso Neutron i.MX 95](https://eiq.nxp.com/learning-hub/convQuant/neutron.html). Il [modulo M.2 da 16 GB](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT) è attivo, mentre l'opzione USB è in preproduzione. NXP definisce la "e" in eTOPS come "equivalent", non "effective": non è la stessa metrica TOPS INT8 densa di altri fornitori e non va confrontata direttamente.

### STMicroelectronics

I [dispositivi STM32N6x7](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html), inclusi STM32N657 e STM32N647, integrano un acceleratore Neural-ART da 600 GOPS in un prodotto di classe MCU; la linea general-purpose N6x5 non include quell'acceleratore. [STM32Cube AI Studio](https://www.st.com/en/development-tools/stedgeai-cubeai.html) e ST Edge AI Core analizzano e ottimizzano i modelli importati, mentre [STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) fornisce flussi di deployment, ottimizzazione, benchmark ed esempi.

L'attuale [panoramica object detection](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/docs/README_OVERVIEW.md) documenta Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 e ST-YOLOX, oltre a servizi di classificazione, posa e segmentazione. Non è la stessa classe prestazionale di una scheda PCIe da 200 TOPS. È interessante perché acquisizione dalla videocamera, inferenza e controllo possono rientrare in limiti embedded molto stretti di potenza e memoria.

### Renesas

I processori Renesas RZ/V integrano acceleratori DRP-AI. Il software è passato da DRP-AI Translator e DRP-AI TVM verso [RUHMI](https://www.renesas.com/en/software-tool/ruhmi-framework), basato sulla tecnologia MERA di EdgeCortix. Il repository open [RZ/V DRP-AI TVM](https://github.com/renesas-rz/rzv_drp-ai_tvm) e l'[elenco di convalida RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) indicano varianti YOLOv5, YOLOv8, YOLO11 e YOLO26 n/s/m, YOLOX, reti per posa e segmentazione, oltre a modelli di classificazione.

Renesas è un target credibile per robotica e industria, ma per garantire la riproducibilità occorre registrare la scheda esatta, la generazione DRP-AI, la versione del translator e la postelaborazione lato CPU.

## Sensori intelligenti e processori progettati per le videocamere

### Sony IMX500

Sony IMX500 non è un normale processore applicativo. Integra rilevamento delle immagini ed elaborazione AI in modo che l'inferenza possa avvenire nel sensore, riducendo la quantità di dati immagine che deve uscire dalla videocamera. Raspberry Pi AI Camera rende questa architettura particolarmente accessibile.

Il repository ufficiale [Raspberry Pi IMX500](https://github.com/raspberrypi/imx500-models) pubblica modelli YOLOv8n, YOLO11n, EfficientDet Lite0, NanoDet+ e SSD MobileNetV2 FPN Lite già impacchettati. La [documentazione AI Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) spiega conversione, packaging e postelaborazione sul sensore. L'unità di deployment è un pacchetto RPK, non un file ONNX generico; memoria del sensore e vincoli sugli operatori sono limiti progettuali fondamentali. La [pagina prodotto](https://www.raspberrypi.com/products/ai-camera/) di Raspberry Pi dichiara che la videocamera resterà in produzione almeno fino a gennaio 2028.

### Ambarella

I processori CVflow di Ambarella sono consolidati in videocamere, droni e visione automotive. Le famiglie prodotto attuali includono CV72/CV75 per videocamere, CV5/CV52 per sistemi di visione con prestazioni superiori e dispositivi automotive CV3-AD. La piattaforma per sviluppatori Cooper e il flusso di compilazione/runtime CVflow costituiscono lo stack software dichiarato pubblicamente.

Il [model garden pubblico per sviluppatori](https://www.ambarella.com/developer/model-garden/) cita YOLOX, RTMDet, DeepLabV3+, TopFormer, OWL-ViT e modelli multimodali. Tuttavia, la documentazione dettagliata del compilatore e i download restano orientati ai partner. Un'integrazione Ambarella dovrebbe quindi partire da un rapporto con il fornitore o l'OEM, non dall'ipotesi che sia disponibile un pacchetto pip pubblico.

### Synaptics Astra

Synaptics ha tre target pertinenti. I sistemi Astra SL1600/SL1680 usano il [toolkit SyNAP](https://developer.synaptics.com/docs/synap/introduction), che converte un modello sorgente insieme a una descrizione YAML in un pacchetto `model.synap`. I dispositivi SL2611/13/15/17/19 più recenti usano la [piattaforma Torq](https://developer.synaptics.com/docs/torq/introduction), un flusso basato su IREE/MLIR che produce `.vmfb`. La [serie SR100](https://developer.synaptics.com/docs/sr/introduction-sr) è una famiglia MCU distinta, basata su Cortex-M55 ed Ethos-U55, con un proprio SDK. Questi artefatti e API non sono intercambiabili.

Il [tutorial ufficiale di benchmark YOLO](https://developer.synaptics.com/docs/sl/tutorials/vision/benchmark-yolo) è molto concreto: esporta YOLOv8 in TFLite, esegue la calibrazione asimmetrica UINT8 in SyNAP, compila per SL1680 e poi avvia `synap_cli` e l'applicazione object detection. Dimostra YOLOv8n/v8s su SL1680 e descrive il supporto SyNAP fino a YOLO11. Una [guida separata al rilevamento oggetti SL261x](https://developer.synaptics.com/docs/sl/sl2600/getting-started/object-detection) esegue YOLOv8 da un `.vmfb` Torq. Sono esempi ufficiali specifici per target, non la prova che ogni grafo YOLO sia supportato su tutte e tre le famiglie.

### MediaTek Genio

MediaTek merita una sezione dedicata perché il suo attuale [IoT AI Hub](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) espone ora la matrice hardware/software, pacchetti modello e tabelle di benchmark che le vecchie analisi del settore non riuscivano a verificare. Genio 360/360P/420/520/720 usa NeuroPilot 8 con MDLA 5.3; Genio 510/700 usa NP6 con MDLA 3.0; Genio 1200 usa NP6 con MDLA 2.0. MediaTek dichiara che generazione NeuroPilot/MDLA, operatori, compilatore e runtime sono vincolati alla versione per tutto il ciclo di vita del SoC.

Il percorso AI analitico è incentrato su TFLite:

```text
PyTorch model
    -> NeuroPilot Converter and representative calibration
    -> quantized .tflite
    -> version-matched ncc-tflite compiler
    -> chip-generation-specific .dla
    -> Neuron Runtime on MDLA
```

I prodotti Genio più recenti offrono anche delega LiteRT online e un percorso ONNX Runtime CPU/NPU sui sistemi operativi supportati. Sono modalità di esecuzione diverse dal deployment offline `.dla`. La [guida all'architettura software](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/software_architecture.html) e la [matrice delle risorse](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) rendono esplicita la distinzione.

La [tabella ufficiale dei modelli analitici](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical.html) pubblica voci benchmark YOLOv5s e YOLOv8s e guide di conversione per più generazioni MDLA. Non distribuisce esplicitamente artefatti YOLO preconvertiti a causa delle restrizioni AGPL-3.0. Le misurazioni offline Quant8 a 640 x 640 includono rispettivamente 5.35 ms e 8.04 ms su Genio 720. La [pagina del modello YOLOv8s](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical/YOLOv8s.html) pubblica tensori input/output, tempi specifici per backend e avvisi sulle operazioni personalizzate MediaTek. Sono benchmark del fornitore, non risultati di una pipeline completa da videocamera.

Il limite è l'accesso. La documentazione Yocto è pubblica e dettagliata, ma l'attuale bundle convertitore/compilatore completo NP8 e molti materiali Android sono indicati come risorse NDA o per clienti diretti. Un normale account sviluppatore non dà lo stesso accesso di un account cliente MediaTek Online. LibreYOLO dovrebbe quindi considerare Genio tecnicamente dimostrato, ma dipendente da una partnership per un'integrazione riproducibile del compilatore con modelli proprietari.

## Ecosistemi edge AI incentrati sulla Cina

Alcune delle piattaforme di visione più capaci e accessibili a basso costo sono poco rappresentate nelle liste di mercato in lingua inglese.

### D-Robotics e piattaforme BPU derivate da Horizon

D-Robotics gestisce l'ecosistema di schede per sviluppatori RDK basato su acceleratori BPU usati nei prodotti RDK X3, X5, Ultra e nelle nuove serie S100. Il ramo attuale `rdk_x5` del [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo) documenta percorsi YOLOv5/v5u, YOLOv8/9/10/11/12/13/26, YOLOE, YOLO-World, rilevamento, segmentazione, posa, classificazione, OCR e CLIP per RDK X5. X3 usa un ramo separato; la distribuzione attuale della serie S si trova nel ramo [`rdk_s`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s) del zoo principale; il vecchio repository [`rdk_model_zoo_s`](https://github.com/D-Robotics/rdk_model_zoo_s) contiene demo storiche. L'elenco moderno X5 non prova quindi che ogni modello sia stato convalidato su X3, Ultra o S100.

OpenExplorer/Algorithm Toolchain importa ONNX o Caffe per PTQ e supporta flussi QAT orientati a PyTorch. Il deployment dipende dalla piattaforma: l'attuale RDK X5 usa `.bin` tramite `hbm_runtime` sopra `libdnn`, mentre il flusso principale attuale `rdk_s` usa `.hbm` tramite API `hbm_runtime` omonima sopra `libhbucp`; X3 conserva il ramo e le interfacce d'inferenza precedenti. Anche i materiali del vecchio `rdk_model_zoo_s` descrivono percorsi storici `.bin` e `.hbm`, quindi quegli artefatti devono restare abbinati al rispettivo ramo e toolchain. Gli operatori non supportati possono passare alla CPU. Gli esempi pubblici sono validi, ma l'abbinamento di versione tra RDK OS, firmware della scheda, toolchain e binario del modello fa parte del contratto di compatibilità.

### AXERA

Le famiglie AX650/AX630/AX620 di AXERA compaiono in videocamere AI compatte e schede come la linea MaixCAM di Sipeed. Pulsar2 importa ONNX, esegue calibrazione e analisi della precisione, quindi produce `.axmodel`; AXEngine esegue l'artefatto.

Gli [esempi ufficiali AXERA](https://github.com/AXERA-TECH/ax-samples) coprono YOLOv5/6/7/8/9/10/11/13/26, YOLOX e YOLO-World, con supporto dei task e dei chip variabile secondo la piattaforma. Sui target supportati, gli esempi YOLO26 attuali includono rilevamento, posa, segmentazione e OBB. Gli [esempi di conversione Pulsar2](https://pulsar2-docs.readthedocs.io/en/latest/appendix/model_convert_examples.html) mostrano il lavoro effettivo: nomi esatti dei tensori, tagli dell'output, trasformazioni di layout, archivi di calibrazione e impostazioni hardware target.

### SOPHGO

TPU-MLIR di SOPHGO è uno stack di compilazione interessante per integrazioni indipendenti perché il compilatore stesso è open source. Importa ONNX, PyTorch, TFLite e Caffe, abbassa e quantizza i grafi, quindi produce `.bmodel` per target BM o `.cvimodel` per hardware CV18xx.

Il progetto [TPU-MLIR](https://github.com/sophgo/tpu-mlir) supporta flussi FP32, BF16, FP16 e INT8 in base al target. La raccolta di [demo SOPHON](https://github.com/sophgo/sophon-demo/blob/release/README_EN.md) cita YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, varianti OBB/segmentazione, SSD, CenterNet, RetinaFace, SAM/SAM2 e OCR. Come sempre, una demo su BM1684X non convalida lo stesso artefatto su BM1688 o CV18xx.

### Huawei Ascend

La linea edge di inferenza Huawei include silicio Ascend 310/310P/310B nei prodotti Atlas 200I e 300I. CANN fornisce il compilatore ATC, il runtime AscendCL e kernel specifici per chip; ATC trasforma ONNX e altre rappresentazioni sorgenti in un modello offline `.om`.

L'attuale [documentazione Atlas 200I DK A2](https://www.hiascend.com/en/hardware/developer-kit-a2/specification) e gli esempi CANN forniscono un percorso YOLOv5 `.om` per Ascend 310B. Il più ampio e precedente [Ascend ModelZoo](https://github.com/Ascend/modelzoo) contiene YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN, posa e modelli di segmentazione, ma non va descritto come se ogni voce fosse stata riconvalidata su 310B. CANN, kernel, firmware e SoC devono corrispondere. Un `.om` per Ascend310P non è un artefatto Ascend generico.

### Cambricon

Gli acceleratori MLU Cambricon usano Neuware e il motore d'inferenza MagicMind. Il repository pubblico [MagicMind Cloud](https://github.com/Cambricon/magicmind_cloud) è fissato a MagicMind 1.7 e a una matrice MLU370-X4/S4; è evidenza storica di compatibilità, non una matrice per l'SDK attuale o per MLU270. Il repository documenta percorsi PyTorch, ONNX, Caffe e Paddle; TensorFlow è stato rimosso dal supporto del framework in 1.7 anche se restano esempi storici TensorFlow. L'attuale [portale sviluppatori 3-series](https://developer.cambricon.com/index/document/index/classid/3.html) di Cambricon elenca release SDK complessive più recenti, quindi repository di esempi pubblici e stack commerciale attuale non vanno presentati come un'unica versione.

L'ufficiale [repository cloud di modelli MagicMind](https://github.com/Cambricon/magicmind_cloud) pubblica una matrice MLU370-X4/S4 particolarmente diretta. Indica esempi YOLOv3, Tiny-YOLOv3, YOLOv4, YOLOv5, YOLOv7, YOLOv8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN, DeepLab e UNet, specificando anche se esistono esempi C++ o Python. L'evidenza è solida; la barriera principale è ottenere SDK e container tramite i canali Cambricon.

### Canaan/Kendryte

I chip KPU attuali K230/K230D e quelli legacy K210/K510 di Kendryte sono importanti per le schede economiche e le videocamere intelligenti. Il compilatore [`nncase`](https://github.com/kendryte/nncase) importa ONNX o TFLite, usa dati di calibrazione per la conversione a virgola fissa, simula il risultato su host e genera `.kmodel`.

La [guida ufficiale allo sviluppo nncase K230](https://github.com/kendryte/k230_docs/blob/main/en/01_software/board/ai/K230_nncase_Development_Guide.md) mostra come compilare, simulare ed eseguire YOLOv5s. Il [catalogo separato delle demo AI](https://github.com/kendryte/k230_docs/blob/main/en/02_applications/ai_demos/K230_AI_Demo_Introduction.md) include artefatti YOLOv8n di rilevamento, segmentazione e posa, mentre l'attuale [changelog CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) aggiunge esempi ottimizzati YOLOv8/11/26 di classificazione, rilevamento, segmentazione, OBB e posa. Il risultato YOLOv5s nella [scheda tecnica](https://github.com/kendryte/k230_docs/blob/main/en/00_hardware/K230_datasheet.md) è il benchmark effettivamente pubblicato; versioni del compilatore e plug-in KPU binario devono corrispondere all'SDK della scheda.

### Allwinner

Allwinner V853 combina hardware multimediale orientato alle videocamere con una NPU Vivante da 1 TOPS. La [guida NPU ufficiale in inglese](https://docs.aw-ol.com/v853/en/npu/dev_npu/) descrive importazione Acuity/Pegasus, quantizzazione, verifica e deployment tramite `viplite`. Le pagine modello Allwinner e la [guida YOLOv5](https://docs.aw-ol.com/v853/npu/npu_yolov5/) citano YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet e reti per volti/persone.

È supporto reale alla computer vision, ma resta vincolato alla vecchia toolchain Tina Linux/Vivante e a download SDK che dipendono dall'account. Va incluso nella mappa con questo limite esplicito.

## Piattaforme coreane, taiwanesi e cinesi spesso trascurate

Le liste in inglese spesso passano da Rockchip a NVIDIA, ignorando una seconda fascia di silicio reale e documentato. Alcuni di questi ecosistemi hanno matrici YOLO attuali più ampie di quelle di startup occidentali più note.

### Mobilint

Il fornitore coreano Mobilint vende REGULUS a basso consumo e l'acceleratore ARIES MLA100 con throughput superiore. Il suo SDK qb accetta input orientati a PyTorch, TensorFlow, TFLite, ONNX e Keras, quantizza in INT8 e genera un artefatto `.mxq` compilato. La [matrice pubblica per la visione](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) elenca rilevamento YOLOv3/v5/v7/v8/v9/v10/11/12/26 e varianti YOLO per segmentazione, posa e OBB. La [pagina ARIES](https://www.mobilint.com/aries/mla100) pubblica persino risultati specifici YOLO11s e YOLO26m. È una forte evidenza d'integrazione, anche se l'accesso a SDK e hardware resta commerciale.

### Sunplus

Sunplus SP7350/C3V usa uno stack NPU derivato da Vivante, ma con un packaging diverso da Allwinner o dal vecchio Amlogic. Il flusso pubblico combina conversione Acuity, ambiente Docker NPU, runtime SNNF e output `.nb`. La [guida ufficiale per YOLOv8 personalizzato](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) descrive conversione e deployment, non si limita ad affermare il supporto del framework. IP correlati non rendono `.nb` portabile verso un altro SoC basato su Vivante.

### ESWIN Computing

La famiglia RISC-V ESWIN include EIC7700/EIC7700X e i dual-die EIC7702/EIC7702X; EIC7700 è presente in schede in vendita come Milk-V Megrez. ENNP include EsQuant, l'attuale compilatore EsAAC, generazione di dati di riferimento, simulazione e runtime ESSDK, con output `.model`; materiali ENNP più vecchi usavano il nome `ennc-compile`. L'attuale [pagina EsAAC](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) documenta input ONNX, quindi gli altri framework devono essere esportati o convertiti in quel formato IR supportato. La [panoramica ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/introduction) e il [tutorial YOLOv3](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3) ospitati da Milk-V dimostrano un percorso pubblico end-to-end, ma non un'ampia matrice attuale di modelli/accuratezza.

### Nuvoton M55M1

Nuvoton M55M1 è una soluzione di classe MCU Cortex-M55 più Ethos-U55-256, non un processore applicativo Linux. NuEdgeWise/NuML e Arm Vela preparano modelli TFLite Micro quantizzati. Il repository pubblico [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) cita YOLOv8-nano, YOLOX-nano, YOLO Fastest v1.1, SSD-MobileNet FPNLite e diversi classificatori. Sono esempi credibili per i vincoli di memoria e potenza del dispositivo, non prove per le varianti YOLO standard a 640 pixel.

### Rebellions e FuriosaAI

La linea ATOM di Rebellions usa l'RBLN SDK e artefatti compilati `.rbln`. La [release ufficiale di supporto](https://docs.rbln.ai/v0.8.2/supports/release_note.html) cita YOLOv3 tiny/full/SPP, YOLOv5 da n a x, YOLOv6 da n a l, varianti YOLOv7 e YOLOv8 da n a x. L'attuale [matrice di supporto delle schede](https://docs.rbln.ai/latest/supports/version_matrix.html) indica ATOM CA02 e ATOM+ CA12 come EoL, mentre ATOM+ CA22 e ATOM-Max CA25 sono Active. ATOM-Lite CA21 ha una pagina prodotto ma non compare nella matrice SDK, quindi lo stato attuale della toolchain è incerto. La documentazione è pubblica, ma il pacchetto del compilatore richiede credenziali Rebellions Portal.

FuriosaAI va suddivisa per generazione. Warboy è il vecchio acceleratore per computer vision, con artefatti INT8 `.enf` e un [model zoo ufficiale](https://developer.furiosa.ai/furiosa-models/latest/) che contiene SSD, YOLOv5M/L e YOLOv7-w6-pose. RNGD usa uno stack [`.fxb`](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html) diverso, rivolto soprattutto a carichi LLM/VLM. La [roadmap attuale](https://developer.furiosa.ai/latest/en/overview/roadmap.html) di Furiosa registra il supporto YOLOv8m per la visione come completato nel quarto trimestre 2024, ma le attuali risorse pubbliche su modelli supportati e prestazioni sono incentrate su LLM/VLM e non espongono una matrice dettagliata di accuratezza/prestazioni YOLOv8m. È una prova relativa a una generazione rilasciata, non una dichiarazione di compatibilità Warboy.

### Realtek, Telechips, T-Head e SigmaStar

Queste quattro piattaforme sono reali, ma offrono percorsi BYOM pubblici più limitati, datati o soggetti ad accesso:

| Piattaforma | Evidenza pubblica riproducibile | Limite da mantenere esplicito |
|---|---|---|
| Realtek AmebaPro2 | L'[SDK Arduino/FreeRTOS](https://github.com/Ameba-AIoT/ameba-arduino-pro2) include modelli YOLOv3/4/7-tiny, SCRFD e MobileFaceNet `.nb` | Per convertire modelli personalizzati occorre registrare il proprio interesse/contattare Realtek |
| Telechips TOPST TCC7500 | Il [progetto ufficiale 2026](https://docs.topst.ai/blog/31) ha convertito e distribuito YOLOv8s | La conversione UFLD v1/v2 non è riuscita con layer non supportati; l'articolo propone solo un workaround con suddivisione/postelaborazione. Per risorse complete del compilatore/operatori spesso occorre chiedere il permesso via email |
| T-Head TH1520 | La [guida applicativa Sipeed](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) esegue YOLOv5n/s con HHB e CSI-NN2/SHL | Lo stack è pubblico, ma manutenzione e chiarezza delle versioni sono indietro rispetto agli attuali leader |
| SigmaStar SSU9383CM | La documentazione attuale definisce le API runtime MI_IPU; il vecchio materiale SGS_IPU documenta `.sim` verso `sgsimg.img` e un vecchio postprocessore cita SSD e YOLOv1/2/3 | Non esistono prove pubbliche che i vecchi artefatti del compilatore o i percorsi per i modelli citati valgano per SSU9383CM |

### La smart vision HiSilicon è distinta da Huawei Ascend

I SoC per videocamere Hi3516CV610/DV500, Hi3519DV500 e Hi3403V100 di HiSilicon espongono un flusso ATC verso `.om` tramite runtime NNN/SVP-NNN. L'[HiSpark model zoo](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) ufficiale cita YOLOv3/4/5/6/7/8/9/10/11, segmentazione/posa YOLO11, OBB/World/segmentazione YOLOv8, OCR, profondità e TinySAM in una matrice specifica per target, soprattutto per Hi3403 e Hi3591P. Non dimostra che ogni voce funzioni su tutti i SoC smart vision elencati sopra. Alcune voci sono elementi di roadmap, quindi esempi rilasciati e supporto pianificato devono restare distinti; il repository dichiara inoltre che i modelli ModelZoo forniti sono solo per uso non commerciale. La terminologia ATC/`.om` condivisa non dimostra la compatibilità di artefatti o runtime con la linea Ascend basata su CANN.

## Aziende di acceleratori emergenti e specializzati

### MemryX

I moduli MemryX MX3 usano un'architettura dataflow e possono combinare più chip. Il [Neural Compiler](https://developer.memryx.com/tools/neural_compiler.html) accetta modelli ONNX, TensorFlow Lite, Keras e TensorFlow e genera un pacchetto DFP utilizzato dal runtime dell'acceleratore. Le [API runtime](https://developer.memryx.com/api/accelerator/accelerator.html) supportano pipeline multi-modello e streaming.

La documentazione e gli esempi MemryX includono preelaborazione e postelaborazione YOLO ottimizzate per rilevamento, segmentazione e posa. Le [note di rilascio SDK](https://developer.memryx.com/release_notes.html) citano esplicitamente il supporto YOLOv10, YOLO11 e YOLO26. I materiali pubblici sono tecnicamente utili, ma manca una tabella unica di modelli/accuratezza/dispositivi come quella di Axelera.

### Kneron

I prodotti KL520/KL530/KL630/KL720/KL730 di Kneron coprono piccoli acceleratori USB ed embedded. Kneron PLUS gestisce il runtime applicativo, mentre l'attuale Model Toolchain 0.33.1 documenta conversione, quantizzazione, valutazione, simulazione e compilazione `.nef` per questi target. KL830 appare in alcune [API runtime PLUS](https://doc.kneron.com/docs/plus_c/introduction/run_examples/), ma non nell'attuale [elenco dei target del compilatore](https://doc.kneron.com/docs/toolchain/manual_1_overview/); non attribuire una capacità generica di compilazione `.nef` a KL830 senza conferma del fornitore.

L'[esempio YOLO ufficiale](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) mostra il procedimento con Tiny-YOLOv3; altri materiali Kneron trattano training e deployment YOLOv5. Le prove sono credibili, ma più limitate delle ampie matrici YOLO moderne di Hailo, Axelera, Rockchip o DEEPX.

### SiMa.ai

La piattaforma MLSoC di SiMa.ai e il [sistema Modalix da 50 TOPS in produzione](https://sima.ai/press-release/sima-ai-next-gen-platform-for-physical-ai-in-production/) usano l'ambiente software Palette. ModelSDK, il compilatore MLA e ModelExecutor coprono importazione, quantizzazione, compilazione, profiling ed esecuzione. A seconda del carico e del prodotto, gli strumenti supportano flussi orientati a INT8, INT16 e BF16.

Le note di rilascio dell'azienda citano pipeline convalidate YOLOv7, YOLOv8 per rilevamento/posa/segmentazione, YOLOX e segmentazione YOLOX, DETR, Mask R-CNN ed EfficientDet. Non va nascosto un limite attuale: le [note Palette SDK 2.1](https://docs.sima.ai/v2.1.2/pages/release_notes/2.1.html) dichiarano che QAT non funziona per una regressione di quantizzazione Python/PT2E. Il workaround documentato è creare l'ONNX annotato in SDK 2.0 e compilarlo con 2.1. Accesso e acquisto restano orientati alle aziende.

Anche il confine del deployment è documentato: il [flusso di compilazione ModelSDK](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) produce un `.tar.gz` compilato e può generare un ELF eseguibile, mentre [MPK Tool](https://docs.sima.ai/v2.1.1/pages/palette/mpk_tools.html) impacchetta un `.mpk` distribuibile. Questi output restano vincolati al target MLSoC/Modalix e alla release Palette.

### EdgeCortix

SAKURA-II è un acceleratore dichiarato da 60 TOPS per visione e AI generativa, compilato con il framework software MERA. EdgeCortix fornisce inoltre la tecnologia MERA al più recente flusso RUHMI di Renesas.

I materiali prodotto [SAKURA-II](https://www.edgecortix.com/en/edgecortix-sakura-ii-accelerator-brief) definiscono hardware e compilatore; l'hardware [M.2/PCIe](https://www.edgecortix.com/en/hardware) è proposto tramite richiesta di prova o ordine. Non è stata trovata una matrice YOLO pubblica, attuale e abbastanza precisa. L'assenza è un'informazione utile per l'acquisto: prima di scegliere l'hardware occorre chiedere la convalida del modello.

### BrainChip

Akida di BrainChip è un processore neuromorfico a eventi, non una NPU convenzionale a tensori densi. L'ambiente [MetaTF](https://brainchip.com/metatf-dev-tools/) comprende pacchetti Python installabili per creare modelli, quantizzarli a bassa precisione, convertirli, simularli ed eseguirli sull'hardware. BrainChip vende ora il coprocessore [AKD1500 in formato M.2, disponibile](https://brainchip.com/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/), oltre all'hardware AKD1000 legacy e all'IP Akida 2.

L'attuale [scheda modello Akida 2](https://brainchip.com/wp-content/uploads/2025/04/Akida-2-Model-Card-V1.1-Mar.25.pdf) cita un rilevatore YOLOv2 AkidaNet0.5, CenterNet, AkidaUNet e reti di riconoscimento facciale. Altri materiali trattano FOMO e carichi basati su eventi. Akida supporta pesi e attivazioni a precisione molto bassa, ma il deployment riuscito può richiedere una conversione consapevole dell'architettura, non un target ONNX INT8 generico. Un risultato del model zoo AKD1000 o Akida 2 non va attribuito ad AKD1500 finché un rapporto esplicito di mappatura/adattamento non lo convalida.

### Blaize

Blaize vende prodotti Pathfinder e Xplorer basati su P1600; il percorso software usa Picasso SDK, NetDeploy e AI Studio. Le [pagine prodotto](https://www.blaize.com/products/) indicano chiaramente visione e edge AI come mercati di riferimento, ma non è disponibile una matrice pubblica attuale di compatibilità per singolo modello.

Questa guida classifica quindi Blaize come commerciale e ad accesso limitato, senza colmare la lacuna con affermazioni della community. Prima di procedere dovrebbe essere richiesto un rapporto di compilazione e accuratezza del fornitore per il modello previsto.

## TinyML e visione endpoint

### Arm Ethos-U e Alif

Arm concede in licenza l'IP NPU Ethos-U55, U65 e U85 alle aziende che producono chip. Il compilatore [Vela](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) prende grafi LiteRT/TFLite quantizzati e riscrive i sottografi supportati come operazioni personalizzate Ethos-U. Le operazioni non supportate possono restare sulla CPU, quindi "l'applicazione funziona" e "l'intera rete gira sulla NPU" sono affermazioni diverse.

Implementazioni reali includono Ethos-U55 in Alif Ensemble E7, Ethos-U65 in NXP i.MX 93 ed Ethos-U85 nei sistemi più recenti. L'[E7 AI/ML AppKit](https://alifsemi.com/support/kits/ensemble-e7appkit/) di Alif documenta due acceleratori ML e mostra il percorso TFLite-to-Vela su E7; [Ensemble E8](https://alifsemi.com/ensemble-e8-series/) combina un Ethos-U85 con due NPU Ethos-U55. Alif pubblica anche un benchmark INT8 di famiglia per il [rilevamento di volti YOLO-Fastest](https://alifsemi.com/faster-ai-mcu-inferencing-low-power-consumption/) a 192 per 192 su Cortex-M55 più Ethos-U55, ma non un'ampia matrice moderna YOLO specifica per target. Questi sistemi con poca memoria sono adatti a piccole reti di classificazione e rilevamento, non a rilevatori arbitrari a 640 pixel solo perché esprimibili in TFLite.

### Infineon PSOC Edge, Himax WiseEye2 e Analog Devices MAX7800x

Le famiglie [PSOC Edge E83](https://documentation.infineon.com/psocedge/docs/xsk1761304020519) ed E84 di Infineon abbinano Cortex-M55 a Ethos-U55 e aggiungono un blocco NNLite separato sul lato M33 a basso consumo. Il [manuale dell'architettura E84](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) documenta pesi a 8 bit, attivazioni a 8 o 16 bit e un flusso Vela in cui gli operatori supportati diventano un flusso di comandi NPU e il lavoro non supportato resta sulla CPU. Il [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter) accetta percorsi TFLite, Keras e PyTorch, mentre i materiali architetturali citano MobileNetV1/V2 come kernel rappresentativi, non come benchmark dei target. L'[E84 AI Kit](https://documentation.infineon.com/psocedge/docs/cci1762693051052) include una videocamera e usa ModusToolbox, ma i materiali pubblici non offrono ancora una matrice di compatibilità YOLO per modelli specifici. È una vera piattaforma di visione programmabile con evidenze sui modelli in crescita, non un target generico di rilevamento già convalidato.

Anche [HX6538 WiseEye2](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) di Himax combina Cortex-M55 ed Ethos-U55 per la visione sempre attiva. La documentazione software pubblica è particolarmente concreta per questa classe di consumi: gli [esempi ufficiali Grove Vision AI Module V2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) includono rilevamento oggetti YOLOv8n, posa e classificazione del genere, rilevamento YOLO11n, face mesh e PeopleNet. Il percorso unificato usa TFLite Micro e Vela, poi ripiega su CMSIS-NN e sui kernel di riferimento quando serve. Si tratta di modelli e risoluzioni volutamente ridotti, non di prove che un grafo YOLO convenzionale per server possa essere eseguito.

Analog Devices segue un percorso diverso con gli [acceleratori CNN MAX78000/MAX78002](https://www.analog.com/en/products/max78002.html). Lo strumento pubblico [`ai8x-synthesis`](https://github.com/analogdevicesinc/ai8x-synthesis) quantizza la rete addestrata e genera codice C, pesi e configurazione dell'acceleratore specifici per il target, invece di un pacchetto runtime ONNX portabile. Le evidenze dirette includono [identificazione facciale](https://www.analog.com/en/resources/app-notes/an-2616.html), rilevamento QR TinierSSD nello [zoo modelli ADI](https://ez.analog.com/dsp/software-and-development-tools/edgebench/a/docs-faqs/DF912/available-models-in-adi-model-zoo) e classificatori d'immagini. I dispositivi sono pertinenti alla visione endpoint, ma non è stata trovata una matrice ufficiale attuale per YOLO moderno.

### GreenWaves GAP9

GAP9 combina calcolo RISC-V con il motore neurale NE16. NNTool importa e quantizza le reti, mentre AutoTiler e GAP SDK generano codice distribuibile. Il [menu reti neurali GAP9](https://github.com/GreenWaves-Technologies/nn_menu_gap9) pubblico contiene applicazioni MobileNet, EfficientNet, ResNet, MobileNet SSD, rilevamento volti e riconoscimento. GreenWaves pubblica anche un [progetto YOLOX di rilevamento persone](https://github.com/GreenWaves-Technologies/yolox_people_detection) con risultati di accuratezza e simulazione.

Sono evidenze TinyML insolitamente trasparenti, ma l'SDK completo è disponibile solo ai clienti qualificati e i modelli sono molto più piccoli delle varianti YOLO convenzionali a 640 x 640.

### Lattice sensAI

Lattice sensAI è un'alternativa FPGA a una NPU fissa. sensAI Studio e Neural Network Compiler mappano un grafo supportato su IP acceleratore configurabile per dispositivi ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E e Avant-X. L'output è vincolato all'FPGA scelto, alla modalità dell'acceleratore, al layout di memoria e al bitstream, non è un modello runtime portabile.

L'attuale [manuale Neural Network Compiler](https://www.latticesemi.com/-/media/LatticeSemi/Documents/UserManuals/MQ3/FPGA-UG-02052-8-0-Lattice-Neural-Network-Compiler-Software.ashx?document_id=52343) include una tabella topologica particolarmente diretta. Elenca YOLOv1 per ECP5, YOLOv5 e YOLOv8 per le modalità optimized/advanced su CrossLink-NX e CertusPro-NX, e YOLO11 solo con IP Advanced. Elenca anche SSD, MobileNetV2-SSD, ResNet, ENet, SqueezeDet e altre reti di visione, mostrando le celle non supportate per famiglia FPGA. L'[Advanced CNN Accelerator](https://www.latticesemi.com/en/Products/DesignSoftwareAndIP/IntellectualProperty/IPCore/IPCores05/Advanced-CNN-Accelerator-IP) è IP commerciale per Avant-E/Avant-X/CertusPro-NX. Sono evidenze da tabella del compilatore, non una promessa che un bitstream supporti simultaneamente ogni topologia.

### Microchip VectorBlox

Il [VectorBlox SDK 3.1](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) pubblico di Microchip preelabora e compila reti TFLite INT8 quantizzate completamente per IP CoreVectorBlox configurabile negli FPGA PolarFire SoC. `tflite_preprocess` e `vnnx_compile` producono asset `.vnnx`, `.hex` e `.ucomp`; il repository include un simulatore e un driver C lato target. I grafi TensorFlow, ONNX e OpenVINO upstream devono comunque passare dal percorso documentato di preparazione TFLite/INT8.

L'attuale [matrice dei tutorial](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) include YOLOv5n, YOLOv8n per rilevamento, classificazione, OBB, posa e segmentazione, YOLOv9t, varianti YOLO sparse/compresse, MobileNet, EfficientNet-Lite0, ResNet18, MiDaS, FFNet e QuickSRNet. La [guida C alla postelaborazione](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/C_Postprocessing.md) cita separatamente decoder YOLOv2/3/4/5 e Ultralytics per rilevamento, posa e OBB. SDK 3.1 supporta attualmente PolarFire SoC Video Kit ed esclude esplicitamente PolarFire Video Kit senza SoC, quindi le demo per schede più vecchie non vanno confuse con il contratto attuale.

L'SDK è scaricabile pubblicamente con una [licenza specifica di Microchip](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/LICENSE.md) che limita software e derivati ai prodotti Microchip. Microchip offre licenze Libero Silver e CoreVectorBlox gratuite, ma su richiesta. Il deployment resta un contratto di sistema FPGA: configurazione dell'acceleratore, bitstream, firmware, dati di rete generati dall'SDK e layout di memoria devono corrispondere. Una compilazione VectorBlox riuscita non trasforma il modello in un binario copiabile su un progetto PolarFire qualsiasi.

### Syntiant

NDP200 di Syntiant è destinato a inferenza sempre attiva per visione, audio e sensori, al di sotto dei consumi dei normali SoC Linux. La [tabella hardware](https://www.syntiant.com/hardware) attuale indica NDP200 in produzione di massa e NDP250 in campionatura. La [scheda prodotto NDP200](https://www.syntiant.com/ndp200/) documenta supporto per reti CNN, RNN e fully connected sotto 1 mW; l'[annuncio NDP250](https://www.syntiant.com/news/syntiant-unveils-ndp250-neural-decision-processor-with-next-gen-core-3-architecture) descrive invece riconoscimento immagini sempre attivo sotto 30 mW. Il dato "inferiore al milliwatt" non va generalizzato a entrambi i componenti.

I materiali pubblici non dimostrano ampia compatibilità YOLO, quindi il prodotto va valutato per piccoli classificatori e rilevatori sempre attivi tramite una valutazione dei modelli supportata dal fornitore, non presumendo un'esportazione ONNX generica.

### Google Coral: due generazioni non correlate

I prodotti Edge TPU legacy usano Edge TPU Compiler, `libedgetpu` e PyCoral con modelli TFLite interamente INT8. I repository principali [Edge TPU](https://github.com/google-coral/edgetpu) e [PyCoral](https://github.com/google-coral/pycoral) sono archiviati. È un rischio concreto per la manutenzione, ma non un annuncio formale di fine vita dell'hardware.

Google mantiene anche una [Coral NPU](https://github.com/google-coral/coralnpu) distinta, attiva e open source: IP acceleratore RISC-V da integrare in SoC a basso consumo. Il progetto pubblico offre al momento RTL, simulazione hardware, toolchain ed esempi ELF. Non è un successore USB/M.2 intercambiabile di Edge TPU e non pubblica una matrice attuale di deployment YOLO.

## Insidie tra GPU, NPU client e calcolo adattivo

NVIDIA, AMD, Intel e Apple compaiono spesso nelle ricerche sulle NPU, ma non offrono una sola categoria di acceleratore intercambiabile.

**NVIDIA Jetson:** Jetson Orin combina una GPU CUDA con core DLA dedicati. TensorRT può costruire un engine serializzato per DLA, ma i layer non supportati girano sulla GPU solo se il fallback GPU è abilitato esplicitamente. L'attuale [tabella DeepStream YOLO](https://github.com/NVIDIA-AI-IOT/deepstream_tools/blob/main/yolo_deepstream/README.md) copre YOLOv4/v7/v8/v9/11, ma la maggior parte delle voci ha come target la GPU; l'unica voce ONNX DLA esplicita è YOLOv8s INT8. Le [restrizioni DLA](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html) di NVIDIA spiegano i limiti degli operatori. Thor è un altro caso: la [guida alla migrazione DriveOS](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/NVIDIA_DriveOS_7.0.3_Migration_Guide.pdf) di NVIDIA dichiara che Thor ha rimosso i core DLA per favorire la flessibilità di scheduling GPU. "Funziona su Jetson" quindi non identifica l'acceleratore.

**AMD Vitis AI:** la vecchia generazione Kria/DPU compilava grafi `.xmodel` eseguiti tramite VART. L'attuale [Vitis AI 6.2](https://vitisai.docs.amd.com/en/6.2/) è GA per due percorsi distinti Versal AI Edge: [Gen1](https://vitisai.docs.amd.com/projects/gen1/en/latest/index.html) su VEK280/VE2802 usa ONNX, AMD Quark e un flusso snapshot/sottografo legato al target; [Gen2](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html) su VEK385 ha un proprio contratto di compilazione e runtime. Gen1 pubblica esempi YOLOv5, YOLOv7, YOLOv8 e YOLOX. Ryzen AI è un quarto stack NPU client separato. Non trasferire artefatti o convalide tra DPU legacy, Versal Gen1, Versal Gen2 e Ryzen AI solo perché condividono il nome Vitis o AMD.

**Intel OpenVINO:** OpenVINO può usare CPU, GPU e NPU Core Ultra tramite un'unica API. La [documentazione del plug-in NPU Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) identifica le generazioni NPU supportate, mentre la [matrice dei modelli verificati](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html) espone colonne specifiche per dispositivo. Un notebook YOLO dimostra una ricetta applicativa, non la convalida NPU, a meno che non lo indichi la matrice. Operazioni, forme, precisione supportate e driver NPU installato determinano ancora se il grafo completo viene eseguito lì. OpenVINO IR (`.xml` più `.bin`) è un IR sorgente riutilizzabile; la cache di un modello compilato dipende dal dispositivo e dal software.

**Apple Core ML:** `coremltools` converte i modelli in `.mlpackage`, che Xcode compila in `.mlmodelc`. Core ML può pianificare il lavoro su CPU, GPU e Apple Neural Engine, ma astrae intenzionalmente la ripartizione precisa. Apple è quindi un'eccellente piattaforma di deployment e una cattiva base per affermare che "l'intero grafo YOLO è stato eseguito sulla NPU" senza prove di profiling.

## Le aziende di IP NPU dietro ai produttori di chip

Alcune aziende non vendono schede o acceleratori confezionati. Concedono in licenza progetti NPU ai produttori di SoC. Sono rilevanti perché la stessa architettura di base può apparire sotto diversi marchi di chip, ma l'SDK e l'artefatto finali possono comunque essere personalizzati dal licenziatario.

| Fornitore IP | Famiglia NPU e software | Perché è rilevante |
|---|---|---|
| [Arm](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) | Ethos-U55/U65/U85 e Vela | Presente in prodotti embedded NXP, Alif, Infineon, Himax e altri; flusso orientato a TFLite/TOSA quantizzato |
| [Arm China](https://www.armchina.com/mountain?infoId=161&name=) | AIPU Zhouyi e Compass SDK | Il [model zoo](https://github.com/Arm-China/Model_zoo) pubblico cita YOLOv1-tiny/v2/v3/v4/v5, YOLOX e YOLOv8-seg; sono modelli di riferimento, non prove per ogni chip licenziatario |
| [VeriSilicon](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico) | Famiglia Vivante VIP e SDK Acuity | Famiglie NPU Vivante correlate compaiono in più SoC, inclusi i percorsi A311D e i.MX 8M Plus acquisiti separatamente; ogni produttore di chip fornisce la propria integrazione |
| [Imagination Technologies](https://www.imaginationtech.com/products/open-access/) | PowerVR Series3NX AX3146/AX3386/AX3596; IMG Series4 storico; Neural Compute SDK/IMG DNN | IP NNA concessibile in licenza, non una scheda retail; Open Access propone configurazioni Series3NX da 1/5/10 TOPS, mentre un programma NC-SDK ufficiale cita PP-YOLOE, EfficientNet e HRNet senza coprire ogni configurazione IP |
| [CEVA](https://www.ceva-ip.com/product/ceva-neupro-studio/) | NeuPro-Nano/NeuPro-M e NeuPro Studio | IP NPU con licenza, con importazione, quantizzazione, compressione, compilazione del grafo, simulazione e generazione C/C++ |
| [MIPS](https://mips.com/processor-solutions/arc-npx-family/) | ARC NPX6 e [MetaWare MX](https://mips.com/processor-solutions/arc-metaware-mx/), precedentemente Synopsys ARC | IP NPU scalabile e NN SDK acquisiti da GlobalFoundries e inseriti nel portafoglio MIPS nel [giugno 2026](https://mips.com/press-releases/gfmipsarcclose/); non sono un target di deployment retail |
| [Cadence](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/neuroweave-sdk.html) | Neo NPU, DSP Tensilica e NeuroWeave SDK | Stack compilatore/interprete usato dai licenziatari SoC, con reti e quantizzazione legate alle configurazioni concesse in licenza |
| [Quadric](https://quadric.ai/npu-ip) | IP GPNPU Chimera e SDK | IP programmabile in stile NPU/DSP integrato nei chip altrui; il target finale è il silicio del licenziatario |
| [Expedera](https://www.expedera.com/products-overview/) | IP NPU Origin e stack software | Architettura edge NPU concessa in licenza, non una scheda LibreYOLO acquistabile direttamente |

Imagination mostra perché le indicazioni su accesso e ciclo di vita devono affiancare i nomi delle architetture. Il programma [Open Access](https://www.imaginationtech.com/products/open-access/) offre tre configurazioni Series3NX verificate sul silicio senza tariffa iniziale per la licenza IP, ma solo dopo la valutazione dell'azienda, con supporto/manutenzione a pagamento e royalty sulla produzione. Il [Neural Compute SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) fornisce strumenti per compilazione, ottimizzazione, quantizzazione e runtime; una collaborazione ufficiale con [Baidu PaddlePaddle](https://www.imaginationtech.com/news/imagination-and-baidu-paddlepaddle-create-open-source-machine-learning-library-for-model-zoo/) cita rilevamento PP-YOLOE, classificazione EfficientNet e segmentazione HRNet. Questi esempi non convalidano ogni configurazione Series3NX o Series4; inoltre le attuali pagine prodotto [Series4](https://www.imaginationtech.com/product/img-4nx-mc1/) indicano i prodotti come non disponibili, quindi Series4 va considerata storica finché il reparto vendite non conferma il contrario.

Questo livello spiega somiglianze apparenti tra famiglie. Non rende portabili gli artefatti. Due SoC con IP Vivante o Ethos correlato possono avere mappe di memoria, driver, versioni del compilatore, set di operatori e runtime differenti.

## Cosa distribuisci davvero: tabella del vincolo sugli artefatti

È l'ultimo file della pipeline del compilatore a segnare la fine dell'apparente portabilità ONNX. Nomi ed estensioni cambiano, ma la regola pratica è stabile: conserva il modello originale e i dati di calibrazione perché l'artefatto nativo di solito non si può spostare su un'altra generazione NPU né ricompilare in modo affidabile con un'altra release dell'SDK.

| Stack del fornitore | Input comune del compilatore | Cosa arriva al dispositivo | A cosa è vincolato |
|---|---|---|---|
| [Amlogic AMLNN](https://github.com/Amlogic-NN/amlnn-toolkit) | ONNX, TFLite, TorchScript o PT2 | `.adla`; il vecchio A311D usa `.nb` | ID target, generazione ADLA, build compilatore AMLNN, NNSDK2/runtime, driver ADLA e BSP |
| [Rockchip RKNN](https://github.com/airockchip/rknn-toolkit2) | ONNX, TFLite o esportazione framework | `.rknn` | Versione toolkit/runtime RKNN e famiglia NPU Rockchip |
| [Hailo](https://hailo.ai/developer-zone/documentation/) | ONNX o TensorFlow tramite HAR intermedio | `.hef` | Architettura Hailo e generazione compilatore/runtime |
| [Axelera Voyager](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) | ONNX o definizione supportata dello zoo | `.axm` in Pipeline Builder alpha, pacchetto pipeline `.axe`; `.axmodel` classico più manifest | Generazione API Voyager e configurazione target Metis |
| [DEEPX DXNN](https://github.com/DEEPX-AI/dx-all-suite) | ONNX e percorsi framework supportati | `.dxnn` | Versione DX-COM/DX-RT e target DX-M |
| [Qualcomm QAIRT/QNN o SNPE](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | ONNX, TFLite o modello framework | Libreria modello/contesto QNN binaria o SNPE `.dlc` | Architettura HTP, SoC, backend e versione runtime |
| [MediaTek NeuroPilot](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) | TFLite convertito/quantizzato | `.dla` per Neuron Runtime offline; `.tflite` per modalità delegata | Generazione NP/MDLA, immagine OS, compilatore e runtime |
| [TI TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools) | ONNX o TFLite | Directory di artefatti importati e file modello applicativi | TIDL tools/runtime e generazione del processore |
| [NXP eIQ](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | Solitamente TFLite o ONNX | Output Vela, TIM-VX, Neutron o Ara specifico per backend | Acceleratore i.MX/Ara scelto e BSP, non "NXP" genericamente |
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | TFLite, ONNX o modello da framework supportato | Libreria/codice di rete generato e asset firmware | Target MCU/NPU, configurazione memoria e versione dello strumento |
| [Infineon PSOC Edge](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) | Percorso TFLite/Keras/PyTorch quantizzato a interi | TFLite ottimizzato da Vela con flusso comandi Ethos-U e firmware applicativo | Variante E83/E84, configurazione Ethos-U, Vela, ModusToolbox e BSP; NNLite è un target separato |
| [Renesas DRP-AI](https://github.com/renesas-rz/rzv_drp-ai_tvm) | ONNX/TFLite tramite TVM o Translator | Directory modello runtime con più file binari | Dispositivo RZ/V e generazione translator/runtime |
| [Sony IMX500](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | Rete quantizzata/impacchettata dal flusso Edge-MDT | Pacchetto `.rpk` | Firmware del sensore, budget memoria e versione packer |
| [Synaptics](https://developer.synaptics.com/) | TFLite/ONNX e importazioni supportate | `.synap` su SyNAP; `.vmfb` su Torq | Generazione software/hardware SL16xx o SL261x |
| [Ambarella CVflow](https://www.ambarella.com/developer/) | Input dalla toolchain per partner | Pacchetto deployment riservato ai partner; contratto esatto dell'artefatto non documentato pubblicamente | Verifica generazione CVflow, SDK/BSP e pipeline camera tramite Cooper Developer Zone |
| [D-Robotics BPU](https://github.com/D-Robotics/rdk_model_zoo) | ONNX e grafo supportato dalla toolchain | `.bin` X5; `.hbm` attuale `rdk_s`; X3 usa un flusso precedente | Generazione BPU, ramo repository e release OpenExplorer/runtime corrispondente |
| [AXERA Pulsar2](https://github.com/AXERA-TECH/ax-samples) | ONNX | `.axmodel` | Target chip AX, versione Pulsar2 e AXEngine |
| [SOPHGO TPU-MLIR](https://github.com/sophgo/tpu-mlir) | ONNX, TFLite, TorchScript e altri | `.bmodel` su BM, `.cvimodel` su CV18xx | Target chip BM/CV e generazione runtime SOPHON |
| [Huawei Ascend CANN](https://www.hiascend.com/en/software/cann) | ONNX o grafo da framework supportato | `.om` | Chip Ascend, CANN/ATC e firmware/driver del dispositivo |
| [Cambricon MagicMind](https://github.com/Cambricon/magicmind_cloud) | Grafo framework o modello di interscambio | Engine/modello MagicMind serializzato | Architettura MLU e versione Neuware/MagicMind |
| [Canaan nncase](https://github.com/kendryte/nncase) | ONNX o TFLite | `.kmodel` | Generazione KPU e plug-in target nncase corrispondente |
| [Mobilint qb](https://www.mobilint.com/sdk-qb) | ONNX e modelli framework supportati | `.mxq` | Target REGULUS/ARIES e compilatore/runtime qb |
| [Rebellions RBLN](https://docs.rbln.ai/latest/index.html) | PyTorch 2 o grafo TensorFlow supportato | `.rbln` | Generazione ATOM e compilatore/runtime RBLN |
| [FuriosaAI](https://developer.furiosa.ai/docs/latest/en/) | ONNX/TFLite e percorso framework supportato | Warboy `.enf`; RNGD `.fxb` | Generazioni di acceleratore e SDK completamente diverse |
| [Sunplus SNNF](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2004353133) | Rete supportata da Acuity | `.nb` | Driver NPU SP7350, runtime e BSP |
| [ESWIN ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) | ONNX nel flusso EsAAC attuale documentato; gli altri framework richiedono esportazione/conversione | `.model` | Target esatto della famiglia EIC7700 e release ENNP/ESSDK |
| [Nuvoton / Ethos-U](https://github.com/OpenNuvoton/NuEdgeWise) | TFLite quantizzato | TFLite ottimizzato da Vela con operazioni personalizzate Ethos-U | Piano memoria M55M1, Vela e firmware |
| [Himax WiseEye2](https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2) | TFLite quantizzato interamente a interi | `.tflite` ottimizzato da Vela nella memoria modello più firmware scheda, per esempio `output.img` | Configurazione HX6538/WE2, Vela, TFLite Micro, driver Ethos-U e kernel fallback |
| [Analog Devices AI8X](https://github.com/analogdevicesinc/ai8x-synthesis) | Checkpoint PyTorch, YAML della rete e input campione | Sorgenti/header C specifici del dispositivo, pesi e firmware compilato generati | Limiti memoria/layer MAX78000 rispetto a MAX78002, strumento di sintesi e release MSDK |
| [Realtek AmebaPro2](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Servizio di conversione/input del fornitore | `.nb` | Firmware RTL8735B e release API VoE/NeuralNetwork |
| [Telechips Enlight](https://docs.topst.ai/product/p/ai) | Percorso Darknet, TensorFlow, ONNX o PyTorch | Intermedio `.enlight` più pacchetto deployment compilato | NPU TCC7500 e versioni TC-NN/Enlight |
| [T-Head HHB](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | ONNX e grafo framework supportato | `hhb.bm`, parametri/codice generati ed eseguibile | Stack CSI-NN2/SHL TH1520 |
| [SigmaStar MI_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | I documenti SSU9383CM attuali espongono il runtime MI_IPU; il materiale precedente usa input TensorFlow/Caffe | Il vecchio `.sim` convertito in `sgsimg.img`; artefatto pubblico attuale non verificato | IPU SigmaStar e generazione SDK esatti; compatibilità del vecchio compilatore con SSU9383CM non verificata |
| [HiSilicon smart vision](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) | ONNX e grafo supportato da ATC | `.om` | SoC smart vision e stack NNN/SVP-NNN esatti; non portabilità generica Ascend |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | ONNX, TFLite, Keras o TensorFlow | Pacchetto dataflow `.dfp` | Configurazione hardware MX e release runtime/compilatore |
| [Kneron](https://doc.kneron.com/docs/) | ONNX più configurazione toolchain | `.nef`; KL730 NEFv2 può contenere `.kne` | Target documentato del compilatore, generazione chip, firmware e runtime PLUS; compilazione KL830 non dimostrata da Toolchain 0.33.1 |
| [SiMa.ai Palette](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) | Framework o modello ONNX supportato | `.tar.gz` compilato ModelSDK, ELF eseguibile documentato o pacchetto MPK Tool `.mpk` | Target MLSoC/Modalix e release Palette |
| [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | ONNX o definizione della rete | Engine serializzato, di solito `.engine` o `.plan` | Architettura GPU/DLA, TensorRT, CUDA e spesso dispositivo |
| [AMD Vitis AI](https://vitisai.docs.amd.com/en/6.2/) | ONNX/framework quantizzato | `.xmodel` legacy; snapshot/sottografi NPU Gen1; directory cache compilata o pacchetto `.rai` di produzione Gen2 | Generazione NPU/IP esatta, immagine piattaforma e matrice Vitis/Vivado/PetaLinux |
| [Microchip VectorBlox](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) | TFLite INT8 quantizzato interamente | `.vnnx`, `.hex` e `.ucomp` più firmware/progetto FPGA corrispondenti | Configurazione CoreVectorBlox, PolarFire SoC Video Kit, SDK 3.1, bitstream e layout memoria |
| [Intel OpenVINO](https://docs.openvino.ai/) | Modello framework o ONNX | IR `.xml` più `.bin` oppure cache compilata specifica per dispositivo | IR riutilizzabile; cache compilata specifica per dispositivo, driver e OpenVINO |
| [Google Edge TPU legacy](https://coral.ai/docs/edgetpu/models-intro/) | TFLite interamente INT8 | `_edgetpu.tflite` compilato per Edge TPU | Edge TPU compiler/runtime e operatori quantizzati supportati; non correlato alla nuova IP Coral NPU |
| [Google Coral NPU IP](https://github.com/google-coral/coralnpu) | Esempi C/C++ per il progetto IP pubblico attuale | Esempi ELF per simulazione RTL/hardware e futura integrazione SoC; nessun artefatto YOLO compilatore pubblico | Implementazione SoC esatta, integrata/licenziata, e toolchain open source in evoluzione |
| [Lattice sensAI](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | Descrizione rete supportata | Bitstream FPGA, configurazione acceleratore e pesi | Famiglia FPGA, modalità IP sensAI e implementazione memoria |
| [Imagination NC-SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) | Caffe, TensorFlow o ONNX | Nome file dell'artefatto compilato per il cliente non specificato pubblicamente | Configurazione NNA licenziata, integrazione SoC, NC-SDK/DDK e runtime licenziatario |

Questa tabella non è solo un promemoria delle estensioni. Versione del compilatore, identificativo target, driver, firmware e BSP fanno parte dell'identità riproducibile del modello. Un futuro esportatore LibreYOLO dovrebbe registrarli in un manifest leggibile dalle macchine accanto a ogni artefatto.

## Accesso agli strumenti e segnali sul ciclo di vita

Disponibilità dell'hardware e accesso al compilatore sono due cose indipendenti. Una scheda può essere facile da acquistare mentre il compilatore attuale richiede un accordo cliente; un SDK pubblico può supportare silicio difficile da reperire. Quelli seguenti sono segnali concreti di prima parte, non una classifica universale della longevità.

| Piattaforma | Segnale documentato | Implicazioni pratiche |
|---|---|---|
| Amlogic ADLA | Repository Apache-2.0 pubblici e pacchetti wheel scaricabili; i driver/BSP compatibili possono comunque richiedere Amlogic o il produttore della scheda | Facile valutare il compilatore, ma occorre confermare ridistribuzione dei binari e matrice completa di compatibilità |
| MediaTek Genio | IoT AI Hub e guide Yocto pubblici; strumenti all-in-one NP8 e documenti Android indicati per clienti diretti/NDA | Le evidenze sui modelli sono verificabili; l'automazione con modelli propri può richiedere una partnership |
| Hailo | Model zoo e runtime pubblici; Dataflow Compiler distribuito tramite Developer Zone; `hailo-camera-apps` pubblico archiviato il 3 maggio 2026 | La ricerca dei modelli è aperta, ma una compilazione riproducibile richiede account/versione corretti; l'archiviazione non dimostra EOL di Hailo-15 e va confermato il pacchetto applicativo attuale per il processore visivo |
| Axelera AI | Documentazione pubblica e [Voyager SDK pubblico](https://github.com/axelera-ai-hub/voyager-sdk); il supporto clienti richiede un account | Un raro SDK acceleratore utilizzabile autonomamente; supporto prodotti e acquisti restano commerciali |
| Qualcomm Dragonwing | Il [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) indica IQ-9075 Sampling con longevità fino al 2038 e QCS6490 fino a luglio 2036, mentre la pagina IQ-9075 dice Active | Segnale utile per la pianificazione industriale, ma il conflitto tra fonti del produttore richiede conferma per SKU e non garantisce stabilità ABI dell'SDK AI per tutto il periodo |
| Rebellions | L'attuale [matrice di supporto](https://docs.rbln.ai/latest/supports/version_matrix.html) indica CA02 e CA12 EoL e CA22/CA25 Active; CA21 è assente | Acquisto e compatibilità SDK dipendono dalla scheda, anche all'interno della famiglia ATOM |
| NVIDIA Jetson | La [tabella ufficiale del ciclo di vita dei moduli](https://developer.nvidia.com/embedded/lifecycle) elenca moduli Orin fino a gennaio 2032 e AGX Orin Industrial fino a luglio 2033; i kit sviluppatore non hanno un impegno di ciclo di vita | Progetta i sistemi di produzione sui moduli, non sulla disponibilità dei kit di sviluppo |
| Sony IMX500 su Raspberry Pi | La [scheda prodotto AI Camera](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf) dichiara produzione almeno fino a gennaio 2028 | Un minimo concreto per quel modulo videocamera, non per ogni prodotto IMX500 |
| Amlogic A311Y3 | La [pagina di lancio 2026](https://www.amlogic.com/News/index248.html) dichiara una garanzia minima di longevità del prodotto di dieci anni | Segnale promettente per la nuova piattaforma, da verificare comunque per contratto e SKU nel programma prodotto |
| Google Coral | I repository Edge TPU e PyCoral legacy sono archiviati/in sola lettura; il repository separato [Coral NPU IP](https://github.com/google-coral/coralnpu) è attivo | Rischio di manutenzione software legacy, di per sé non un avviso formale EOL hardware; non dice nulla sul ciclo di vita del progetto open source IP non correlato |

## Perché i TOPS non bastano per scegliere

I TOPS di picco possono essere utili all'interno di un singolo fornitore e architettura, ma i confronti tra fornitori di solito non sono validi a meno che corrispondano tutti questi elementi:

- Precisione numerica, inclusi INT8, INT4, FP16 o modalità miste
- Se una moltiplicazione-accumulazione conta come una o due operazioni
- Calcolo denso o sparso strutturato
- Risoluzione d'ingresso, dimensione del batch e grafo del modello
- Accuratezza dopo la quantizzazione
- Latenza solo NPU o pipeline applicativa completa
- Trasferimenti di memoria e fallback sull'host
- Stato termico e funzionamento sostenuto, non a raffica

Un rilevatore fa parte di una pipeline: acquisizione immagine, resize/letterbox, normalizzazione, trasferimento al dispositivo, inferenza neurale, decodifica, NMS, conversione delle coordinate e logica applicativa. I fornitori spesso pubblicano solo la fase centrale di inferenza neurale. [MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/) è utile perché definisce scenari, requisiti di qualità e regole per misurare il sistema completo invece di confrontare numeri di marketing isolati.

Per i deployment YOLO, una scheda benchmark minima credibile registra:

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

Senza questi campi, due valori FPS misurano spesso cose diverse.

## Come scegliere una NPU per computer vision

Scegli in questo ordine, non in base al numero più grande stampato sulla pagina prodotto.

1. **Verifica la compatibilità del grafo.** Compila il modello esportato esatto, non un checkpoint dello zoo dal nome simile.
2. **Misura l'accuratezza.** Convalida l'artefatto compilato sul dataset reale del task, dopo calibrazione e quantizzazione.
3. **Misura l'intero flusso.** Includi conversione dell'input, trasferimenti, decodifica e NMS.
4. **Verifica l'accesso software.** Scopri se il compilatore è pubblico, disponibile previa registrazione o solo dopo un accordo commerciale.
5. **Fissa la matrice delle versioni.** Registra compilatore, runtime, driver, firmware e identificativi dei target.
6. **Verifica i sistemi operativi effettivamente supportati.** Android, Debian, Yocto, Buildroot, RTOS e Windows non sono intercambiabili.
7. **Controlla fornitura e ciclo di vita.** Un chip eccellente dal punto di vista tecnico è una scelta di produzione debole se moduli, driver o supporto a lungo termine sono incerti.
8. **Poi confronta costo, consumi e prestazioni.** Queste misure hanno senso solo dopo che lo stesso modello funziona correttamente su entrambi i target.

### Punti di partenza pratici per caso d'uso

| Caso d'uso | Piattaforme da valutare per prime | Motivo |
|---|---|---|
| Acceleratore specifico per Raspberry Pi | Hailo-8L/8 e Sony IMX500 | Prodotti ufficiali Raspberry Pi, immagini software e flussi concreti per sviluppatori |
| Espansione generica Linux PCIe, M.2 o USB | Prodotti DEEPX, MemryX, Hailo e Axelera | Formati acceleratore reperibili, subordinati alla compatibilità host/driver |
| SBC Linux o videocamera a basso costo | Rockchip RK3588/RK3576, Amlogic ADLA se è reperibile una scheda/BSP supportata, AXERA, Canaan K230 o Sunplus SP7350 | Pipeline multimediali integrate ed evidenze pubbliche su modelli/compilatori; per la NPU A311D2 verificare VIM4 V13A o successive |
| Android mobile e ad alto volume | Qualcomm QNN/AI Hub, LiteRT, Apple Core ML | Ampia base installata e runtime mobile mantenuti |
| Linux industriale e robotica | TI TDA4/AM6xA, NXP i.MX, Renesas RZ/V, Hailo | Prodotti embedded longevi ed ecosistemi videocamera/I-O |
| Endpoint minuscolo o MCU | STM32N6x7, Infineon PSOC Edge, Himax WiseEye2, Analog Devices MAX7800x, Nuvoton M55M1, Alif/Arm Ethos-U, GAP9 e Syntiant | Limiti stringenti di potenza e memoria, strumenti specifici e vincoli sulla dimensione dei modelli |
| Visione su FPGA configurabile | Microchip VectorBlox, Lattice sensAI e target AMD Vitis AI | Pipeline hardware flessibili, ma configurazione acceleratore, bitstream e compilatore del modello costituiscono un unico sistema versionato |
| Visione PCIe ad alto throughput | Axelera Metis, Hailo, DEEPX, Mobilint, MemryX, Rebellions o SiMa.ai | Acceleratori dedicati e orientamento ai flussi multi-stream |
| Ecosistema prodotto incentrato sulla Cina | Rockchip, AXERA, D-Robotics, SOPHGO, Huawei Ascend, HiSilicon e Cambricon | Schede regionali, toolchain ed esempi di modelli solidi |

Questa tabella è una selezione per la ricerca, non una classifica delle prestazioni. Acquisti, disponibilità regionale e modello esatto possono cambiare l'ordine.

## Cosa significa per LibreYOLO

Un'architettura scalabile non consiste in decine di esportatori scollegati. Serve un unico contratto rigoroso di interscambio più piccoli adapter per compilatori e runtime dei fornitori:

```text
LibreYOLO model
    -> deterministic static ONNX
    -> VendorCompiler.compile(model, target, calibration, precision)
    -> native artifact + manifest
    -> VendorRuntime.load() / infer()
    -> shared task-specific decode and Results objects
```

Ogni artefatto nativo dovrebbe avere un manifest accessorio con:

- Fornitore, target chip e target scheda
- Versioni di compilatore, runtime, driver e firmware
- Checksum del checkpoint sorgente e di ONNX
- Nomi input, forme, layout, spazio colore, normalizzazione e dtype
- Precisione di quantizzazione, scale se esposte e hash dei dati di calibrazione
- Nomi e forme dei tensori di output e significato semantico
- Task di rilevamento, nomi classi e indicazione se decode/NMS avviene sul chip o sull'host
- Parità numerica e accuratezza sul task registrate

LibreYOLO fornisce già [esportazione ONNX portabile](/docs/export/onnx), [strumenti di quantizzazione](/docs/export/quantization), un'[esportazione RKNN](/docs/export/rknn) diretta ma volutamente limitata e un [flusso Hailo](/docs/export/hailo) onesto con compilatore esterno. Secondo i criteri usati in questa guida, Amlogic ADLA è un forte candidato alla prossima integrazione: il toolkit è pubblico, la copertura dei modelli coincide ampiamente con LibreYOLO e il vecchio percorso A311D può restare chiaramente separato.

La priorità dopo Amlogic dovrebbe dipendere dall'hardware degli utenti e dalla possibilità di convalidare l'accuratezza, non solo dalla disponibilità del compilatore. DEEPX, Sony IMX500, D-Robotics, AXERA, SOPHGO e Qualcomm sono interessanti sul piano tecnico; TI, NXP, ST e Renesas diventano particolarmente preziose quando i partner industriali possono fornire schede e accesso CI a lungo termine.

## Da tenere d'occhio: silicio reale senza contratto d'integrazione pubblico

Escludere del tutto un'azienda può rendere fuorviante una mappa di mercato. Includerla come "supportata" può essere peggio. Questi fornitori vendono, campionano o hanno annunciato silicio pertinente, ma le evidenze pubbliche non dimostrano ancora un percorso attuale e riproducibile con compilatore, artefatto e modello specifico, adatto a un backend LibreYOLO.

| Azienda | Cosa è reale | Perché resta da monitorare |
|---|---|---|
| [Samsung](https://semiconductor.samsung.com/processor/automotive-processor/exynos-auto-v920/) | Exynos Auto V920 include una NPU dual-core dichiarata fino a 23.1 TOPS | Samsung dichiara che il suo [Neural SDK non è più fornito a sviluppatori terzi](https://developer.samsung.com/neural/overview.html); Samsung ONE/Circle non dimostra accesso alla NPU Exynos |
| [Novatek](https://www.novatek.com.tw/en-global/Milestone/aboutus_milestones) | Le tappe aziendali attuali citano SoC Edge AI ed Edge Vision/Imaging AI; una tappa del 2020 menzionava MobileNet, SSD e YOLOv3 | Nessuna matrice pubblica attuale di codici prodotto/compilatore/artefatti/operatori/modelli; il vecchio YOLOv3 non dimostra l'SDK attuale |
| [Nextchip](https://www.nextchip.com/en/adas/adas.php?idx=5) | Processori automotive APACHE5/NVS2900 e attuali [APACHE6/NVS3000](https://www.nextchip.com/en/adas/adas.php?idx=7) con NPU aiMotive aiWare; la pagina APACHE6 dichiara sia 12 sia 8 TOPS | Esistono aiWare Studio e un runtime, ma non sono stati trovati artefatto pubblico, guida alla quantizzazione, elenco operatori o matrice YOLO; i valori discordanti del fornitore non vanno risolti in silenzio |
| [Black Sesame Technologies](https://bst.ai/en.html) | Silicio AI automotive/edge Huashan A1000/A2000 e serie Wudang C | Esistono informazioni pubbliche sul prodotto, ma non compilatore self-service, contratto artefatto o model zoo riproducibile |
| [Ingenic](https://en.ingenic.com.cn/products-detail/id-19.html) | SoC camera T41/T40 con dichiarazioni NPU a bassa precisione; Magik AI descrive PTQ/QAT e compilazione del grafo | Non è stato trovato un compilatore attuale scaricabile, contratto artefatto o elenco esatto di modelli YOLO del fornitore |
| [Fullhan](https://fullhan.com/en/index.php?a=type&c=article&tid=9) | Diversi SoC IPC dichiarano NPU da 0.5 a 2 TOPS, mentre la [famiglia NVR MC6880](https://fullhan.com/en/index.php?a=type&c=article&tid=48) dichiara 4 TOPS | Documentazione pubblica di compilatore/runtime/framework/modello insufficiente per una riproduzione indipendente |
| [Goke Microelectronics](http://www.gokemicro.com/News/info.aspx?itemid=437) | Una pagina ufficiale per componenti smart camera GK7606V1/GK7206V1/GK7203V1 dichiara prestazioni NPU integrate, ma al momento della pubblicazione è servita solo tramite il vecchio protocollo HTTP | Nessun convertitore pubblico, contratto runtime o matrice di modelli nominati; il canale è orientato agli OEM |
| [Chengheng Micro](https://en.chenghengmicro.com/) | CH37 dichiara 64 TOPS INT8 e modalità FP16/FP32/FP64; la [cronologia 2026](https://chenghengmicro.com/about.html) afferma che l'azienda è entrata nella produzione in piccoli lotti | Non sono stati trovati SDK pubblico, contratto del compilatore o convalida di modelli nominati; è una piattaforma commerciale iniziale, non self-service, non un'evidenza di deployment riproducibile |
| [Bouffalo Lab](https://github.com/bouffalolab/bouffalo_sdk) | BL808 include NPU BLAI-100 e sono disponibili repository SDK ufficiali attuali | L'SDK mantenuto non espone un percorso ufficiale attuale per conversione/esempi BLAI; i flussi rimasti sono legacy o prove della community |

Questa lista da monitorare si basa intenzionalmente sulle evidenze. Un fornitore può entrare nelle tabelle delle piattaforme distribuibili pubblicando una toolchain o un percorso di valutazione attuale, un contratto dell'artefatto specifico per target e almeno un modello nominato con una ricetta end-to-end.

## Cosa non è stato affermato

Questo articolo non sostiene che ogni modello citato funzioni a partire da ogni repository upstream. I modelli dei vendor zoo sono spesso modificati, suddivisi in corrispondenza di nodi di output diversi o abbinati a postelaborazione personalizzata.

Non trasforma neppure queste affermazioni in dichiarazioni di supporto:

- Un compilatore importa ONNX.
- Un chip dichiara un driver Android NNAPI.
- Un distributore afferma che la scheda è "compatibile con YOLO".
- Un repository della community esegue un fork di un modello.
- Un modello si compila senza errori.
- Un fornitore dichiara FPS solo NPU senza un risultato di accuratezza.

Gli acceleratori Tensor per telefoni Google e numerosi ASIC personalizzati riservati agli OEM sono reali, ma Google non espone Tensor come target di compilazione self-service generico paragonabile alle piattaforme sopra. L'esistenza di un'azienda, uno schema a blocchi NPU o l'accelerazione Android non bastano per inventare una dichiarazione d'integrazione LibreYOLO.

Allo stesso modo, gli acceleratori cloud come Google TPU, AWS Inferentia/Trainium, Microsoft Maia e le schede AI solo per data center sono fuori ambito. Questa guida riguarda hardware che uno sviluppatore di computer vision può realisticamente distribuire all'edge.

La licenza del modello è un livello separato. La ricetta di conversione YOLO pubblicata da un produttore di chip non concede i diritti per usare o ridistribuire pesi upstream, codice di training o derivati compilati. Per il prodotto previsto occorre verificare separatamente supporto hardware, licenza SDK e licenza del modello.

## Metodologia di ricerca e correzioni

Per ogni azienda la ricerca ha esaminato quattro fonti primarie:

1. Una pagina prodotto o scheda tecnica attuale che identifichi il silicio.
2. Documentazione di compilatore e runtime che spieghi il vero percorso di deployment.
3. Model zoo, tabella di compatibilità o tutorial end-to-end che citi modelli di visione.
4. Un segnale relativo a ciclo di vita o accesso che mostri se gli sviluppatori possono ottenere gli strumenti.

Le organizzazioni GitHub di prima parte contano come fonti del fornitore. La documentazione del produttore della scheda è indicata come evidenza dell'ecosistema quando l'azienda che produce il chip non pubblica gli stessi materiali. Le dichiarazioni prestazionali di terze parti sono state escluse dalle tabelle comparative.

La bozza completa è stata poi riesaminata in passate separate per gruppi di fornitori e tabelle incrociate. I controlli hanno verificato nuovamente ambito dei target, nomi degli artefatti, precisione, evidenze su modelli e postprocessori, stato annunciato o in distribuzione, etichette del ciclo di vita, denominatori dei benchmark e conteggi di tutte le entità. Quando due pagine di prima parte sono in conflitto, la guida mostra il conflitto invece di scegliere in silenzio il valore più comodo.

Questo mercato cambia rapidamente. Se lavori per una di queste aziende e un chip, SDK, elenco di modelli o stato d'accesso non è corretto, invia a LibreYOLO l'URL esatto della documentazione pubblica e la versione. Le correzioni supportate da evidenze primarie dovrebbero sostituire questo testo; le affermazioni di marketing non documentate no.

## FAQ

### Quante aziende producono NPU per l'edge AI?

Non esiste un totale universale di aziende, perché si sovrappongono fornitori di prodotti, licenziatari di IP, sensori intelligenti, GPU e ASIC automobilistici proprietari. Questa guida non esaustiva segue 69 aziende ed ecosistemi di piattaforma con chip edge AI pertinenti, piattaforme di accelerazione, IP NPU concessibile in licenza o silicio da tenere d'occhio con basi credibili, aggiornati ad agosto 2026.

### Che cos'è una NPU?

Una neural processing unit è hardware specializzato per operazioni di rete neurale come convoluzioni e moltiplicazioni di matrici. I nomi usati dai fornitori includono NPU, AIPU, BPU, KPU, DLA, HTP, TPU e MLA; in genere i modelli compilati non sono portabili tra aziende.

### Quali aziende con NPU supportano ufficialmente i modelli YOLO?

Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 tramite il repository ufficiale AI Camera di Raspberry Pi, STMicroelectronics, Renesas, Lattice, Himax, Microchip e diverse altre aziende pubblicano voci nei model zoo, tutorial o risultati convalidati per almeno una generazione di YOLO. La generazione esatta, il task, il chip di destinazione e la versione dell'SDK restano determinanti.

### Qualsiasi modello ONNX può essere eseguito su qualsiasi NPU?

No. ONNX è un formato di interscambio, non una garanzia di compatibilità hardware. Il compilatore NPU deve supportare ogni operatore, forma dei tensori e tipo di dato del grafo. Forme statiche, tagli del grafo, operazioni personalizzate e fallback sulla CPU sono comuni.

### Qual è la NPU migliore per YOLO?

Non esiste una vincitrice universale. Hailo, Rockchip, Axelera AI, DEEPX e Amlogic offrono un supporto YOLO documentato consistente, mentre Qualcomm raggiunge un numero eccezionale di dispositivi. Scegli in base all'accuratezza dopo la quantizzazione, alla latenza dell'intera pipeline, ai consumi sostenuti, al prezzo, alla disponibilità, all'accesso all'SDK e al supporto del sistema operativo.

### Perché non si possono confrontare direttamente i valori TOPS delle NPU?

I fornitori possono conteggiare precisioni diverse, operazioni sparse, moltiplicazioni-accumulazioni e ipotesi diverse sull'utilizzo di picco. I TOPS non includono traffico di memoria, operatori non supportati, preelaborazione, postelaborazione e overhead dell'host. Confronta lo stesso modello, la stessa precisione, risoluzione, accuratezza e pipeline completa.

### Che cos'è la calibrazione di una NPU?

La calibrazione esegue sul modello immagini rappresentative, di solito senza etichette, dell'ambiente di deployment, così che il compilatore possa stimare gli intervalli delle attivazioni per INT8 o un altro formato a bassa precisione. Anche immagini casuali o non rappresentative possono produrre un artefatto compilato, ma compromettere l'accuratezza del task.

### LibreYOLO supporta le NPU edge?

LibreYOLO esporta ONNX e diversi formati per runtime, dispone di un percorso diretto verso il compilatore RKNN per alcuni modelli Rockchip e documenta il flusso esterno di compilazione Hailo. Per ogni altro artefatto nativo del fornitore serve ancora il relativo SDK; va considerato un candidato all'integrazione finché non viene compilato, convalidato ed eseguito su hardware.
