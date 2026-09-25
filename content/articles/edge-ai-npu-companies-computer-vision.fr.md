---
title: "69 entreprises Edge AI et NPU : puces, SDK et YOLO (2026)"
description: "Guide sourcé de 69 entreprises Edge AI et écosystèmes NPU : puces, SDK, artefacts, quantification et prise en charge documentée de la vision par ordinateur et de YOLO."
date: 2026-08-15
author: Xuban
tags: [edge-ai, npu, computer-vision, yolo, hardware, ai-accelerators, amlogic, hailo, rockchip]
faq:
  - q: "Combien existe-t-il d'entreprises Edge AI avec NPU ?"
    a: "Il n'existe pas de nombre universel, car les fabricants de produits, les fournisseurs de licences IP, les capteurs intelligents, les GPU et les ASIC automobiles privés se recoupent. Ce guide non exhaustif suit 69 entreprises et écosystèmes de plateformes proposant des puces Edge AI, des plateformes d'accélération, des IP NPU sous licence ou des puces pertinentes à surveiller en août 2026."
  - q: "Qu'est-ce qu'un NPU ?"
    a: "Une unité de traitement neuronal est un matériel spécialisé dans les opérations des réseaux de neurones, comme les convolutions et les multiplications matricielles. Les fabricants emploient notamment les termes NPU, AIPU, BPU, KPU, DLA, HTP, TPU et MLA. Leurs modèles compilés ne sont généralement pas portables d'un fabricant à l'autre."
  - q: "Quelles entreprises NPU prennent officiellement en charge les modèles YOLO ?"
    a: "Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 via le dépôt officiel AI Camera de Raspberry Pi, STMicroelectronics, Renesas, Lattice, Himax, Microchip et plusieurs autres publient des entrées de model zoo, des tutoriels ou des résultats validés par leurs équipes pour au moins une génération de YOLO. La génération, la tâche, la puce ciblée et la version du SDK restent déterminantes."
  - q: "Un modèle ONNX quelconque peut-il s'exécuter sur n'importe quel NPU ?"
    a: "Non. ONNX est un format d'échange, pas une garantie de compatibilité matérielle. Le compilateur NPU doit prendre en charge chaque opérateur, forme de tenseur et type de données du graphe. Les formes statiques, les découpages de graphe, les opérations personnalisées et le repli sur le CPU sont fréquents."
  - q: "Quel est le meilleur NPU pour YOLO ?"
    a: "Il n'existe pas de vainqueur universel. Hailo, Rockchip, Axelera AI, DEEPX et Amlogic disposent d'une couverture YOLO bien documentée, tandis que Qualcomm bénéficie d'une présence exceptionnelle dans les appareils. Choisissez selon l'exactitude après quantification, la latence de bout en bout, la puissance soutenue, le prix, la disponibilité, l'accès au SDK et la prise en charge du système d'exploitation."
  - q: "Pourquoi ne peut-on pas comparer directement les valeurs TOPS des NPU ?"
    a: "Les fabricants peuvent compter différemment les précisions, les opérations creuses, les multiplications-accumulations et les hypothèses d'utilisation de pointe. Les TOPS ne tiennent pas compte du trafic mémoire, des opérateurs non pris en charge, du prétraitement, du post-traitement ni de la surcharge de l'hôte. Comparez le même modèle, la même précision, la même résolution, la même exactitude et le pipeline complet."
  - q: "Qu'est-ce que la calibration d'un NPU ?"
    a: "La calibration fait passer des images représentatives du déploiement, généralement non annotées, dans le modèle afin que le compilateur estime les plages d'activation pour INT8 ou un autre format de faible précision. Des images aléatoires ou non représentatives peuvent produire un artefact compilé tout en dégradant l'exactitude sur la tâche."
  - q: "LibreYOLO prend-il en charge les NPU edge ?"
    a: "LibreYOLO exporte vers ONNX et plusieurs formats de runtime, propose un chemin direct de compilation RKNN pour certains modèles Rockchip et documente le flux de compilation Hailo externe. Tout autre artefact natif d'un fabricant exige encore son SDK et doit être présenté comme un candidat d'intégration tant qu'il n'a pas été compilé, validé et exécuté sur le matériel."
---

**Il n'existe pas de marché NPU unique. Ce guide non exhaustif suit 69 entreprises et écosystèmes de plateformes : 51 fabricants de puces ou plateformes déployables, neuf fournisseurs d'IP NPU et neuf entreprises clairement indiquées comme à surveiller.** Ils couvrent plusieurs catégories d'accélérateurs incompatibles et presque autant de chaînes de compilation. La plupart promettent le même parcours : exporter un modèle, le quantifier, le compiler, copier un artefact propriétaire sur une carte et appeler un runtime du fabricant. Les détails déterminent si cela prend un après-midi ou un trimestre d'ingénierie.

Ce guide recense les entreprises qui fournissent du matériel crédible pour la vision par ordinateur en 2026. Il indique les puces pertinentes, les noms des SDK et compilateurs, les artefacts de déploiement, le niveau d'accès public et les modèles réellement documentés par le fabricant. Il s'attarde aussi sur Amlogic, dont la nouvelle pile ADLA diffère sensiblement de l'ancien écosystème NPU A311D.

> **Périmètre de recherche :** les sources ont été vérifiées le 15 août 2026. Les mentions de modèles précis proviennent de pages produit, de documentation, de model zoos, de dépôts ou de tutoriels du fabricant, sauf indication contraire. Les TOPS annoncés sont des chiffres fournis par les fabricants et ne constituent pas des résultats de benchmark comparables de manière indépendante. Il s'agit d'un guide pour les développeurs, pas d'un conseil en investissement ni d'un classement payant.

**Navigation rapide :** [tableaux des entreprises](#npu-companies-and-platforms-at-a-glance) | [analyse détaillée d'Amlogic](#amlogic-two-npu-generations-not-one) | [profils des fabricants](#the-strongest-public-deployment-ecosystems) | [tableau des artefacts compilés](#what-you-actually-deploy-the-artifact-lock-in-table) | [accès et cycle de vie](#tool-access-and-lifecycle-signals) | [feuille de route LibreYOLO](#what-this-means-for-libreyolo)

## Le constat essentiel

Le matériel est fragmenté, mais le schéma de déploiement est remarquablement cohérent :

```text
PyTorch checkpoint
    -> ONNX or TFLite interchange graph
    -> representative calibration data
    -> vendor quantizer and graph compiler
    -> chip-specific binary or model package
    -> vendor runtime on the target
    -> application preprocessing, decode, NMS and rendering
```

[ONNX autorise explicitement les runtimes, les générateurs de code et les implémentations matérielles](https://onnx.ai/onnx/repo-docs/IR.html), mais un fichier ONNX ne constitue qu'un point de transfert. Cela ne signifie pas que chaque opérateur ONNX peut être abaissé vers chaque accélérateur. Les formes d'entrée statiques, les contraintes sur les opérateurs pris en charge, les règles de quantification et le repli côté hôte déterminent ce qui s'exécute réellement sur le NPU.

La pile logicielle fait donc partie de la puce. Un NPU rapide sur le papier, mais doté d'un compilateur verrouillé ou fragile, peut être une moins bonne cible qu'un appareil plus modeste avec une chaîne d'outils publique, un model zoo, un simulateur et un runtime stable.

## Ce que signifie « pris en charge » dans ce guide

La documentation des fabricants emploie le mot *support* avec beaucoup de latitude. Cet article distingue quatre niveaux de preuve :

| Niveau de preuve | Ce que cela prouve | Ce que cela ne prouve pas |
|---|---|---|
| **Modèle validé** | Le fabricant publie une entrée, un résultat ou une table de compatibilité pour un modèle et une puce précis | Votre checkpoint modifié conserve la même exactitude |
| **Exemple officiel** | Le fabricant fournit un tutoriel ou une démo de bout en bout pour un modèle donné | Les autres tailles, tâches ou générations compilent |
| **Capacité du compilateur** | Le SDK importe un framework ou expose les opérateurs pris en charge | Un graphe YOLO particulier fonctionne de bout en bout |
| **Marketing ou accès restreint** | Le produit vise la vision et un SDK privé existe | La compatibilité du modèle peut être reproduite publiquement |

Cette distinction compte. « Importe ONNX » n'équivaut pas à « prend en charge la segmentation YOLO11 ». Un modèle peut être analysé puis basculer sur le CPU, compiler avec des valeurs de sortie incorrectes, dépasser la mémoire de l'accélérateur ou perdre en exactitude pendant la quantification.

## Entreprises et plateformes NPU en un coup d'œil

Les tableaux mêlent volontairement SoC, accélérateurs discrets, capteurs intelligents et cibles GPU/FPGA proches. Ils interviennent dans le même choix de déploiement, même si les fabricants parlent de NPU, AIPU, BPU, KPU, DLA, HTP, TPU ou MLA.

### SoC embarqués, capteurs intelligents et processeurs industriels

| Entreprise | Puces ou plateformes pertinentes | SDK, compilateur et artefact | Preuves publiques de prise en charge de la vision par ordinateur | Accès |
|---|---|---|---|---|
| [Amlogic](https://github.com/Amlogic-NN/amlnn-toolkit) | A311D2, S928X, S905X5/S905D5, A311Y3, C308L/C302X/C302X2, T968D4, C305X2 et A123X | AMLNN Toolkit et `libnnsdk.so`, `.adla` compilé ; pile Acuity `.nb` héritée distincte pour A311D | Le [model playground](https://github.com/Amlogic-NN/amlnn-model-playground) documente YOLOv5/6/7/8/10/11, YOLOX, YOLOE, YOLO-World, PP-YOLOE, la segmentation, la pose et OBB sur A311D2, S905X5, A311Y3, C305X2 et A123X. Les autres puces listées sont des cibles du compilateur, mais ne figurent pas dans cette matrice de prise en charge. | GitHub et wheels publics ; BSP/pilote compatible requis |
| [Rockchip](https://github.com/airockchip/rknn-toolkit2) | RK3562/3566/3568, RK3576, RK3588, RV1126B | RKNN-Toolkit2, RKNN Runtime, `.rknn` | YOLOv5/6/7/8/10/11, YOLOX, YOLO-World, PP-YOLOE, segmentation, pose et OBB dans le [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) | GitHub public ; wheels binaires |
| [Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | Plateformes Snapdragon et Dragonwing, QCS6490, QCS8550 et Dragonwing IQ-9075 | QAIRT/QNN, SNPE et AI Hub ; binaires de contexte QNN ou DLC | La [collection de modèles AI Hub](https://github.com/qualcomm/ai-hub-models) inclut YOLOv3/5/6/7/8/9/10/11/26, YOLOX, YOLO-World, YOLOR, RF-DETR et des modèles de détection, segmentation et pose | Documentation publique ; exigences de SDK et de compte variables |
| [Texas Instruments](https://github.com/TexasInstruments/edgeai-tidl-tools) | AM62A, AM67A, AM68A, AM69A et TDA4x ; TDA54-Q1 en préversion | Processor SDK Edge AI et TIDL | Le [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo) publie des modèles optimisés de détection, segmentation, pose et classification pour les parcours AM6xA/TDA4 actuels. Il ne s'agit pas encore d'une matrice de cibles pour le TDA54-Q1 en préversion. | GitHub public et Processor SDK |
| [NXP](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | i.MX 8M Plus, i.MX 93, i.MX 95 et Kinara Ara-1/Ara240 acquis | eIQ avec TIM-VX, Vela, Neutron Converter et Ara SDK | Le [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) comprend des artefacts ou recettes pour YOLOv4-tiny/v8, NanoDet, CenterNet, FastestDet, SSD Lite et YOLACT ; l'entrée YOLOv5 se limite à la documentation. | Composants publics et soumis à un compte |
| [STMicroelectronics](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) | Variantes STM32N6x7, dont STM32N657/647, avec accélérateur Neural-ART | STM32Cube AI Studio et ST Edge AI Core | Le dépôt de services actuel répertorie Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 et ST-YOLOX, ainsi que la pose et la segmentation | Outils et dépôts publics |
| [Infineon](https://documentation.infineon.com/psocedge/) | PSOC Edge E83/E84 avec Cortex-M55 et Ethos-U55 ; accélérateur NNLite distinct côté M33 | ModusToolbox, [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter), TFLite Micro et Arm Vela | La documentation d'architecture cite MobileNetV1/V2 comme noyaux représentatifs et le kit AI E84 comprend une caméra, mais aucune matrice de validation YOLO propre au PSOC Edge n'a été trouvée. | Documentation et composants SDK publics, kits d'évaluation E84 actuels |
| [Renesas](https://github.com/renesas-rz/rzv_drp-ai_tvm) | RZ/V2L, V2M, V2MA, V2H et V2N | DRP-AI Translator, DRP-AI TVM et RUHMI | La [liste des modèles RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) valide des tailles de YOLOv5/v8/v11/26, ainsi que des variantes YOLOX, pose et segmentation | GitHub public et SDK de la carte |
| [Sony](https://www.aitrios.sony-semicon.com/edge-ai-devices/imx500) | Capteur de vision intelligent IMX500 et Raspberry Pi AI Camera | Edge-MDT, Model Compression Toolkit et packer IMX500 ; paquet `.rpk` | Le zoo [Raspberry Pi IMX500](https://github.com/raspberrypi/imx500-models) comprend YOLOv8n, YOLO11n, EfficientDet Lite, NanoDet+ et SSD | Parcours Raspberry Pi public ; conditions des outils Sony applicables |
| [Ambarella](https://www.ambarella.com/developer/model-garden/) | Familles CV72/CV75, CV5/CV52 et CV3-AD | Cooper Developer Platform, compilateur CVflow et DAG de runtime | Le model garden public cite YOLOX-S, RTMDet-nano, DeepLabV3+, TopFormer, OWL-ViT et LLaVA OneVision | Principalement réservé aux partenaires |
| [Synaptics](https://developer.synaptics.com/docs/sl/overview) | Astra SL1600/SL1680 ; SL2611/13/15/17/19 ; famille MCU SR100 distincte | SyNAP `.synap` sur SL16xx ; Torq/IREE `.vmfb` sur SL261x ; SDK SR distinct orienté Ethos-U55 | Benchmark officiel SyNAP YOLOv8n/v8s et exemple Torq YOLOv8 sur SL261x ; les preuves pour SR doivent être évaluées séparément | Portail développeurs ; certains téléchargements sont restreints |
| [MediaTek](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) | Genio 360/360P/420/520/720 avec NP8 et MDLA 5.3 ; Genio 510/700/1200 avec NP6/MDLA plus anciens | NeuroPilot Converter, `ncc-tflite`, Neuron Runtime et `.dla` ; parcours ONNX Runtime sur certains systèmes récents | L'IoT AI Hub officiel publie des pages de modèles et benchmarks YOLOv5 et YOLOv8, ainsi que des modèles de classification et de visage | Documentation Yocto publique ; lots NP8 essentiels et documentation Android soumis à un accès client direct/NDA |
| [Allwinner](https://docs.aw-ol.com/v853/en/npu/dev_npu/) | SoC vision V853 avec NPU Vivante de 1-TOPS | Conversion Acuity/Pegasus et runtime `viplite` | La documentation officielle V853 cite YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet et des réseaux visage/personne | Documentation publique ; téléchargement du SDK parfois soumis à un compte |
| [Canaan/Kendryte](https://github.com/kendryte/nncase) | K230/K230D actuels ; KPU K210/K510 historiques | Compilateur et runtime `nncase` ; `.kmodel` | Le guide nncase K230 traite de la compilation, simulation et exécution de YOLOv5s ; le catalogue officiel de démos et le [journal des modifications CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) actuel documentent des tâches YOLOv8/11/26 | Compilateur/PyPI publics ; plug-in de puce binaire |
| [Alif Semiconductor](https://alifsemi.com/support/kits/ensemble-e7appkit/) | Ensemble E7 avec deux accélérateurs ML, dont Ethos-U55 ; [E8](https://alifsemi.com/ensemble-e8-series/) avec un Ethos-U85 et deux NPU Ethos-U55 | Déploiement fondé sur Arm Vela et TFLite Micro | Alif publie un benchmark INT8 à l'échelle de la famille pour le détecteur de visage YOLO-Fastest, mais pas de vaste matrice moderne YOLO par cible | Documentation publique et ressources pour kits d'évaluation |
| [Himax](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) | MCU endpoint IA HX6538 WiseEye2 avec Cortex-M55 et Ethos-U55 | TFLite Micro et Arm Vela, avec CMSIS-NN et repli sur les noyaux de référence | Les exemples officiels [WiseEye2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) comprennent la détection, la pose et la classification YOLOv8n, la détection YOLO11n, le face mesh et PeopleNet | GitHub et documentation publics, carte partenaire disponible à l'achat |
| [Analog Devices](https://www.analog.com/en/products/max78002.html) | MCU IA MAX78000 et MAX78002 avec accélérateurs CNN basse consommation | [`ai8x-training`](https://github.com/analogdevicesinc/ai8x-training), `ai8x-synthesis`/`izer` et MSDK ; code C et poids générés | La documentation officielle couvre l'identification/détection de visages, RetinaNet, Visual Wake Words, des classificateurs et la reconnaissance d'actions ; aucun déploiement YOLO officiel n'a été trouvé | Outils GitHub, documentation et cartes d'évaluation publics |
| [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo) | Cartes BPU RDK X3/X5/Ultra et nouvelles séries S100 | OpenExplorer/Algorithm Toolchain et `hbm_runtime` ; `.bin` pour X5, `.hbm` pour le `rdk_s` actuel | La branche `rdk_x5` actuelle documente YOLOv5/v5u/v8/v9/v10/11/12/13/26, YOLOE, YOLO-World, segmentation, pose, classification, OCR et CLIP pour RDK X5 ; X3 et série S ont des branches distinctes | GitHub public ; certains paquets de chaîne d'outils dépendent de la plateforme |
| [AXERA](https://github.com/AXERA-TECH/ax-samples) | AX650, AX637, AX630C, AX620Q et AX615 | Compilateur Pulsar2 et AXEngine ; `.axmodel` | Les exemples officiels actuels couvrent YOLOv5/6/7/8/9/10/11/13/26, YOLOX et YOLO-World ; tâches et cibles varient selon la puce | Exemples et documentation publics ; compilateur distribué séparément |
| [SOPHGO](https://github.com/sophgo/tpu-mlir) | BM1684/1684X/1688/1690 et CV186X/CV18xx | TPU-MLIR et runtime SOPHON ; `.bmodel` ou `.cvimodel` | Les démos officielles couvrent YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, OBB, segmentation, visage, SAM et OCR | Compilateur open source et runtime du fabricant |
| [Huawei Ascend](https://www.hiascend.com/en/software/cann) | Ascend 310/310P/310B et produits edge Atlas 200I/300I | CANN, compilateur ATC et AscendCL ; `.om` | Le [ModelZoo Ascend](https://github.com/Ascend/modelzoo) officiel comprend YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN et des modèles de segmentation | Documentation publique ; les paquets CANN et noyaux doivent correspondre à la cible |
| [Cambricon](https://github.com/Cambricon/magicmind_cloud) | MLU370-X4/S4 dans la matrice publique citée ; MLU270 matériel historique non couvert par cette matrice | Neuware et MagicMind | La matrice MagicMind 1.7 du dépôt répertorie YOLOv3/4/5/7/8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN et des modèles de segmentation | Anciens exemples publics ; accès au SDK/conteneur actuel restreint |
| [Sunplus](https://sunplus.atlassian.net/wiki/spaces/C3/pages/1971126471/SP7350%2BSpecification) | SP7350/C3V, environ 4,1 à 4,6 TOPS annoncés | Docker NPU Vivante Acuity, runtime SNNF et `.nb` | La documentation officielle fournit YOLOv5 et un [déploiement YOLOv8 personnalisé](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) complet | Sources et documentation publiques, écosystème de cartes |
| [ESWIN Computing](https://www.eswincomputing.com/en/news/info/96.html) | SoC RISC-V EIC7700/EIC7700X et [EIC7702/EIC7702X](https://www.eswincomputing.com/en/news/info/104.html) à deux puces | ENNP : EsQuant, compilateur EsAAC actuel, simulateur et runtime ESSDK ; `.model` ; les anciennes docs utilisaient `ennc-compile` | La documentation ENNP hébergée par Milk-V inclut un [tutoriel YOLOv3](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3) de bout en bout, ainsi que des exemples MobileNetV2 et ResNet | Documentation mixte, du fabricant et hébergée par le fabricant de cartes ; téléchargements régionaux |
| [Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m55-mcus/m55m1-series/index.html) | MCU M55M1 avec Ethos-U55-256 | TFLite Micro, Arm Vela et NuEdgeWise/NuML | Les exemples officiels [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) citent YOLOv8-nano, YOLOX-nano, YOLO Fastest, SSD-MobileNet et des modèles de classification | Chaîne d'outils MCU essentiellement publique |
| [Realtek](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Plateforme caméra AmebaPro2 RTL8735B / AMB82-mini | SDK Arduino/FreeRTOS, VoE et API NeuralNetwork ; `.nb` | Les modèles fournis comprennent YOLOv3-tiny, YOLOv4-tiny, YOLOv7-tiny, SCRFD et MobileFaceNet | Runtime public ; l'accès au [convertisseur personnalisé](https://ameba-doc-arduino-sdk.readthedocs-hosted.com/en/latest/FAQ/offline_ai_model_conversion_steps.html) nécessite une prise de contact |
| [Telechips](https://docs.topst.ai/product/p/ai) | Carte TCC7500 / TOPST AI, 8 TOPS annoncés | TC-NN-Toolkit / Enlight SDK ; `.enlight` intermédiaire et bundle de déploiement compilé | Un [projet TOPST officiel](https://docs.topst.ai/blog/31) a déployé YOLOv8s ; ses conversions UFLD v1/v2 ont échoué sur des couches non prises en charge et seule une solution de contournement avec découpage/post-traitement a été proposée. [YOLOv4](https://community.topst.ai/t/segmentation-fault/358) et [un autre parcours YOLOv8](https://community.topst.ai/t/segmentation-fault-error/415) sont évoqués dans des fils de support communautaires. | Documentation de la carte publique ; téléchargements du compilateur/opérateurs souvent soumis à autorisation |
| [T-Head](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | TH1520, 4-TOPS INT8 annoncés, sur LicheePi 4A | Compilateur HHB et CSI-NN2/SHL ; `hhb.bm` plus code/paramètres générés | Les exemples officiels du fabricant de cartes exécutent YOLOv5n/s et MobileNetV2 sur le NPU | Exemples publics ; incertitudes liées à l'ancienneté de la chaîne et à sa maintenance |
| [SigmaStar](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | SoC caméra intelligent SSU9383CM actuel et anciennes familles IPU | Runtime MI_IPU actuel ; l'[ancienne chaîne SGS_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSC9381G_9351_Pudding-ULS00V040-20210913/customer/development/dla/tools.html) produisait `.sim` puis `sgsimg.img` | Un ancien postprocesseur du SDK [officiel](https://wx.comake.online/doc/doc/Sigmastar_SDK_v1.2.2/module/User_Guide/Common/SigmaStar_Post_Processing_Module.html) cite SSD et YOLOv1/2/3 ; leur compatibilité publique avec le SSU9383CM n'est pas vérifiée | Silicium actuel, mais les preuves publiques du compilateur et des modèles portent sur des générations différentes |
| [HiSilicon](https://www.hisilicon.com/cn/products/smart-vision/machine-vision/hi3516cv610) | SoC de vision intelligente Hi3516CV610/DV500, Hi3519DV500 et Hi3403V100 | ATC vers `.om` avec runtime de carte NNN/SVP-NNN | La matrice HiSpark propre aux cibles, notamment Hi3403 et Hi3591P, cite YOLOv3 à YOLO11, pose, segmentation, OBB, OCR et profondeur ; elle ne valide pas ces modèles sur tous les SoC de cette ligne | Actif et distinct d'Ascend ; les modèles du dépôt sont indiqués comme réservés à un usage non commercial |

### Accélérateurs edge dédiés et modules

| Entreprise | Silicium pertinent | SDK et artefact | Modèles documentés publiquement | Accès |
|---|---|---|---|---|
| [Hailo](https://github.com/hailo-ai/hailo_model_zoo) | Familles Hailo-8, Hailo-8L, Hailo-10H et Hailo-15 | Dataflow Compiler et HailoRT ; `.hef` | YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 et YOLO26, ainsi que YOLOX, DAMO-YOLO, SSD, EfficientDet, segmentation, pose et OBB, avec des tables de modèles propres à chaque génération | Model Zoo public ; compilateur via Developer Zone |
| [Axelera AI](https://docs.axelera.ai/sdk/reference/models/model-zoo/) | Produits Metis commercialisés ; produits Europa et Titania annoncés | SDK Voyager public ; Pipeline Builder alpha `.axm`/`.axe`, bundles `.axmodel` classiques | YOLOv3/5/7/8/9/10/11/26, YOLOX, YOLO-NAS, OBB, pose, segmentation et de nombreux modèles sans YOLO validés | SDK public sur GitHub ; support client soumis à un compte |
| [DEEPX](https://developer.deepx.ai/modelzoo/) | DX-M1 et DX-M1M | SDK DXNN, DX-COM et DX-RT ; `.dxnn` | Le Model Zoo recensait 354 entrées avec DX-COM 2.4.0/DX-RT 3.4.0 au 15 août 2026, notamment YOLOv3 à YOLO11 et YOLO26, YOLOX, SSD, EfficientDet, NanoDet, segmentation, pose et OBB | Ressources développeur et suite publiques |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | Accélérateur MX3 et modules multi-puces | SDK MemryX, Neural Compiler et runtime ; `.dfp` | Les exemples officiels et notes de version couvrent la détection, la segmentation et la pose, dont YOLOv10, YOLO11 et YOLO26 | Documentation et SDK publics |
| [Kneron](https://doc.kneron.com/docs/) | Documentation de compilation actuelle pour KL520, KL530, KL630, KL720 et KL730 ; KL830 présent dans certaines API PLUS, mais absent de la liste actuelle des cibles du compilateur | Kneron PLUS et Model Toolchain ; `.nef` pour les cibles documentées | Le [parcours YOLO officiel](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) porte sur Tiny-YOLOv3 ; d'autres documents traitent de l'entraînement et du déploiement YOLOv5 | Documentation publique ; paquets et téléchargements variables selon la puce |
| [SiMa.ai](https://docs.sima.ai/pages/palette/modelsdk.html) | MLSoC et plateforme Modalix 50-TOPS en production | Palette, ModelSDK, MLA Compiler et ModelExecutor | Les versions publiques valident YOLOv7/v8, YOLOX, pose/segmentation, DETR, Mask R-CNN et EfficientDet | Documentation publique ; SDK produit commercial |
| [EdgeCortix](https://www.edgecortix.com/en/hardware) | Accélérateur SAKURA-II annoncé à 60-TOPS, en cartes M.2 et PCIe | Compilateur et framework MERA | Le fabricant annonce la prise en charge de la vision à l'IA générative, mais ne publie pas de matrice YOLO actuelle assez précise | Essai/demande de commande commerciale ; validation avec un partenaire |
| [BrainChip](https://brainchip.com/metatf-dev-tools/) | Coprocesseur/module M.2 AKD1500 commercialisé ; plateformes AKD1000 historiques ; IP Akida 2 | Paquets MetaTF : `akida-models`, `quantizeml`, `cnn2snn` et runtime `akida` | La fiche modèle Akida 2 cite un détecteur YOLOv2 AkidaNet0.5, CenterNet, AkidaUNet et la reconnaissance faciale ; ces résultats ne valident pas automatiquement AKD1500 | Principaux paquets Python publics ; mappage matériel propre à la cible |
| [Blaize](https://www.blaize.com/products/) | Produits P1600, Pathfinder et Xplorer | Picasso SDK, NetDeploy et AI Studio | La vision est un marché cible, mais aucune matrice publique vérifiable de compatibilité par modèle n'a été trouvée | Commercial/restreint |
| [Google Coral](https://github.com/google-coral/edgetpu) | Anciens produits Edge TPU USB, PCIe, M.2 et Dev Board ; IP [Coral NPU](https://github.com/google-coral/coralnpu) open source et active, distincte | Ancien Edge TPU Compiler/`libedgetpu`/PyCoral ; la nouvelle Coral NPU RISC-V expose l'IP, la RTL/simulation et des exemples ELF | Les anciens exemples portent surtout sur SSD MobileNet, la classification, DeepLab et MoveNet ; la nouvelle IP ne possède pas de matrice publique de déploiement YOLO et ne remplace pas directement Edge TPU | Principaux dépôts de l'ancienne génération archivés ; développement actif de la nouvelle IP Coral NPU |
| [Lattice Semiconductor](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | FPGA ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E et Avant-X | sensAI Studio, Neural Network Compiler et IP d'accélérateur configurable | La table actuelle du compilateur cite YOLOv1, YOLOv5, YOLOv8 et YOLO11 dans des modes propres à chaque cible, ainsi que SSD, MobileNetV2-SSD, ResNet et ENet | Compilateur et manuels publics ; conditions IP variables, Advanced CNN Accelerator commercial |
| [Mobilint](https://www.mobilint.com/sdk-qb) | Accélérateurs REGULUS 10-TOPS et ARIES MLA100 80-TOPS | SDK qb ; compilateur INT8 et `.mxq` | Le [model zoo](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) public répertorie YOLOv3/5/7/8/9/10/11/12/26 en détection, et des variantes segmentation, pose et OBB | Documentation et modèles publics ; SDK et matériel commerciaux |
| [Rebellions](https://rebellions.ai/developers/) | ATOM+ CA22 et ATOM-Max CA25 actifs ; ATOM CA02/ATOM+ CA12 en fin de vie ; statut des outils ATOM-Lite CA21 incertain | Compilateur/runtime/profiler RBLN SDK ; `.rbln` | Les versions officielles nomment les familles YOLOv3, YOLOv5/6/7/8 et des tutoriels YOLOv8 actuels | Documentation publique ; wheel du compilateur nécessitant les identifiants du portail |
| [FuriosaAI](https://furiosa.ai/warboy) | Warboy Gen1 pour la vision ; génération RNGD distincte | SDK Furiosa ; `.enf` INT8 sur Warboy, `.fxb` distinct sur RNGD | Le zoo Warboy documente SSD, YOLOv5M/L et YOLOv7-w6-pose ; la feuille de route RNGD marque la prise en charge de YOLOv8m comme terminée au T4 2024, mais ne publie pas de matrice vision actuelle détaillée | Accès IAM/compte ; distinguer les deux générations |

### Plateformes proches que les développeurs comparent aux NPU

| Entreprise | Matériel | Pile de déploiement | Pourquoi l'inclure dans la comparaison |
|---|---|---|---|
| [NVIDIA](https://github.com/NVIDIA-AI-IOT/deepstream_tools/tree/main/yolo_deepstream) | GPU et DLA Jetson Orin ; GPU Jetson Thor sans DLA | JetPack, TensorRT et DeepStream ; `.engine` | Déploiement vision très mature ; les outils DeepStream officiels documentent YOLOv4/v7/v8/v9/11, notamment une configuration OBB YOLO11. Beaucoup de résultats utilisent le GPU et les couches Orin DLA non prises en charge peuvent basculer sur le GPU. |
| [AMD](https://vitisai.docs.amd.com/en/6.2/) | Cibles Kria/DPU historiques ; Vitis AI 6.2 GA pour Versal AI Edge Gen1 VEK280/VE2802 et Gen2 VEK385 ; NPU client Ryzen AI séparés | Ancien `.xmodel` ; snapshot NPU Gen1 lié à la cible ; cache compilé Gen2 ou paquet de production `.rai` | Vitis AI publie des ressources vision, mais DPU historique, Versal Gen1, Versal Gen2 et Ryzen AI correspondent à des contrats de compilation et de déploiement distincts. |
| [Microchip](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox) | FPGA PolarFire SoC avec IP d'accélérateur CoreVectorBlox configurable | VectorBlox SDK 3.1 et Libero publics ; artefacts `.vnnx`, `.hex` et `.ucomp` | Les [tutoriels actuels](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) couvrent YOLOv5n, détection/classification/OBB/pose/segmentation YOLOv8, YOLOv9t et d'autres modèles vision ; SDK 3.1 cible actuellement le PolarFire SoC Video Kit. |
| [Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) | NPU Core Ultra, CPU et GPU intégrés/discrets | OpenVINO IR ou cache de modèle compilé | OpenVINO fournit un parcours public multi-XPU ; consultez les colonnes NPU de la [matrice des modèles vérifiés](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html), sans supposer que chaque exemple YOLO OpenVINO a été validé sur NPU. |
| [Apple](https://developer.apple.com/machine-learning/core-ml/) | Apple Neural Engine dans les puces A-series à partir de A11 et dans les puces M-series | Core ML et `coremltools` ; `.mlpackage`/`.mlmodelc` | NPU grand public très accessible via Core ML, mais Apple masque la répartition exacte entre CPU/GPU/Neural Engine au lieu d'exposer un compilateur matériel propre à YOLO. |
| [GreenWaves Technologies](https://github.com/GreenWaves-Technologies/nn_menu_gap9) | GAP9 avec moteur neuronal NE16 | GAP SDK, NNTool et code généré par AutoTiler | Les dépôts officiels incluent MobileNet SSD, détection faciale, classification et un [détecteur de personnes YOLOX](https://github.com/GreenWaves-Technologies/yolox_people_detection) dédié. Le SDK complet est réservé aux clients qualifiés. |
| [Syntiant](https://www.syntiant.com/hardware) | Processeurs Neural Decision NDP200 en production de masse et NDP250 en échantillonnage | SDK Syntiant et paquets de déploiement | Processeurs vision/capteurs toujours actifs à très basse consommation : le NDP200 est documenté sous 1 mW et la reconnaissance d'image du NDP250 sous 30 mW. Aucune large matrice YOLO publique n'a été trouvée. |

## Amlogic : deux générations de NPU, pas une seule

Amlogic mérite une explication plus détaillée, car les informations en ligne mélangent souvent des produits incompatibles. L'entreprise propose une ancienne voie fondée sur VeriSilicon et une nouvelle voie propriétaire ADLA. Elles n'utilisent ni le même compilateur, ni le même artefact, ni le même runtime.

| Génération | Puces représentatives | NPU et chaîne d'outils | Artefact compilé | État pratique |
|---|---|---|---|---|
| Historique | A311D et S905D3, fréquents sur Khadas VIM3/VIM3L | VeriSilicon Vivante VIPNano-QI, toolkit Acuity, KSNN et ancien `aml_npu_sdk` | `.nb` dans un répertoire de sortie `nbg_unify` | Cartes et démos existantes ; intégration historique séparée |
| ADLA2 actuel | C308L/C302X, S928X, A311D2, T968D4, S905X5/S905D5 et C302X2 | Amlogic ADLA2, AMLNN Toolkit et NNSDK2 | `.adla` propre à la cible ; W8A8 ou W8A16 | Pile actuelle orientée entier |
| ADLA3 actuel | A311Y3, C305X2 et A123X | Amlogic ADLA3, AMLNN Toolkit et NNSDK2 | `.adla` propre à la cible ; W4A8, W8A8, W4A16, W8A16 ou W16A16 | Ajoute la prise en charge native INT4, FP16 et BF16 |

La [fiche technique A311D](https://dl.khadas.com/products/vim3/datasheet/a311d-datasheet.pdf) identifie le NPU INT8 original de 5-TOPS. L'ancienne [page d'application Khadas VIM3](https://docs.khadas.com/products/sbc/vim3/npu/npu-app) et la [présentation du NPU](https://docs.khadas.com/products/sbc/vim3/npu/start) documentent DenseNet CTC, MTCNN, RetinaFace, YOLOFace, YOLOv2, YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv7-tiny et YOLOv8n ; la présentation ajoute YOLOv8n-pose et indique que FaceNet est obsolète. Ces exemples sont réels, mais prouvent le fonctionnement de l'ancien écosystème Acuity `.nb`, pas celui des puces ADLA actuelles. A311D n'est pas A311D2, et C305X n'est pas C305X2.

Un second piège concerne le matériel. Le [guide des révisions Khadas VIM4](https://docs.khadas.com/products/sbc/vim4/configurations/identify-version) indique que la carte d'origine V12/A311D2 révision B ne possède pas de NPU ; le NPU annoncé à 3,2-TOPS apparaît sur les cartes V13A et ultérieures, équipées de la puce A311D2-N0D révision C. Le nom du produit ne suffit pas pour choisir un appareil de test.

### La pile AMLNN et ADLA actuelle

L'[AMLNN Toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/7d3cc9a36179b756ce79c953c14d43acab1f2af5) actuel couvre la conversion, la quantification, la compilation, l'inférence et le profiling des modèles. Son [guide utilisateur de mai 2026](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/02_Amlogic_NPU_User_Guide_V0.1.pdf), épinglé sur une révision précise, documente les importateurs ONNX, TFLite en virgule flottante ou quantifié, TorchScript `.pt` et PyTorch 2 ExportedProgram/PT2. Le guide détaillé indique que les parcours TensorFlow, Paddle et Keras sont prévus, malgré la formulation plus large de la présentation du dépôt. Le compilateur produit des `.adla`. Le déploiement Python utilise AMLNN ou `amlnn_edge_toolkit_lite` ; en C/C++ natif, il faut lier `libnnsdk.so` via `nnsdk2.h`. Le développement côté hôte peut utiliser ADB avec `nnserver` ; le runtime cible également Android, Buildroot, Yocto et certains environnements Debian/Armbian.

Les identifiants de plateforme publiés par le toolkit comprennent actuellement :

| ID de cible | Plateforme |
|---:|---|
| `001` | C308L / C302X |
| `002` | S928X |
| `003` | A311D2 |
| `004` | T968D4 |
| `005` | S905X5 / S905D5 |
| `006` | C302X2 |
| `007` | A311Y3 |
| `008` | C305X2 / A123X dans les documents détaillés et la matrice des modèles |

Ce champ de cible n'est pas décoratif. Amlogic exige que `.adla`, `nnsdk2.h`, `libnnsdk.so`, le BSP et la génération du compilateur/runtime correspondent. Le [guide de démarrage rapide](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/01_Amlogic_NPU_Quick_Start_guide_V0.1.pdf) épinglé requiert un pilote ADLA 2.0.2 ou plus récent, NNSDK 3.0.0 ou plus récent et NNSDK2 1.0.0 ou plus récent ; le [guide de déploiement Android](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/06_Amlogic_NPU_Android_Deployment_Guide_V0.1.pdf) détaille les dépendances entre bibliothèque, en-tête, pilote et BSP. Considérez `.adla` comme un artefact de build lié à l'ABI, pas comme un modèle portable.

La quantification dépend elle aussi de la génération du NPU. Les cibles ADLA2 001 à 006 documentent la compilation W8A8 et W8A16. Les cibles ADLA3 007 et 008 ajoutent les combinaisons W4A8, W4A16, W8A8, W8A16 et W16A16, avec une prise en charge native INT4, FP16 et BF16 dans le [guide des opérateurs](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/05_Amlogic_NPU_Support_Operator_List_V0.1.pdf). Amlogic recommande 200 à 500 échantillons de calibration représentatifs. L'option de données aléatoires sert explicitement aux tests de performance, pas à une quantification préservant l'exactitude.

### Couverture des modèles documentée par Amlogic

Le [model playground Amlogic épinglé](https://github.com/Amlogic-NN/amlnn-model-playground/blob/03ca63ef20c4f1560722de26ab910826884f334a/README.md) constitue une preuve bien plus solide qu'une affirmation générique de prise en charge ONNX. Sa matrice publique d'exemples et de modèles cite A311D2, S905X5, A311Y3, C305X2 et A123X. Le fait que le compilateur accepte d'autres ID de cible ne prouve pas que toute cette liste de modèles y a été validée.

| Tâche | Modèles documentés |
|---|---|
| Classification | MobileNetV2 et ResNet50-v2 ; DINO sur ADLA3 |
| Détection d'objets | PP-YOLOE, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv10, YOLO11, YOLOE, YOLO-World, YOLOX et détection QR |
| Visage et gestes | RetinaFace et Gesture Recognition |
| Segmentation | DeepLabV3, PP-LiteSeg, YOLOv5-seg et YOLOv8-seg |
| Détection orientée | YOLOv8 OBB |
| Pose | BlazePose detector/landmark et YOLOv8 pose |
| OCR et parole | LPRNet, variantes PaddleOCR, Whisper Tiny et SenseVoice sur certaines puces récentes |
| Transformers récents et multimodal | DETR, MobileSAM, CLIP/MobileCLIP et certains exemples LLM/VLM basse précision sur les plateformes récentes |

Le même dépôt publie les mesures suivantes d'exécution de modèles W8A8. Ce sont des mesures du réseau neuronal sur le NPU communiquées par le fabricant, et non celles d'un pipeline caméra complet.

| Modèle et entrée | S905X5 | A311D2 | A311Y3 |
|---|---:|---:|---:|
| YOLOv8n, 640 x 640 | 101.72 FPS | 95.14 FPS | 191.06 FPS |
| YOLOv8s, 640 x 640 | 42.33 FPS | 42.77 FPS | 83.08 FPS |
| YOLOv8m, 640 x 640 | 19.67 FPS | 19.82 FPS | 35.30 FPS |
| YOLOv8l, 640 x 640 | 10.53 FPS | 10.12 FPS | 18.37 FPS |
| YOLO11n, 640 x 640 | 41.14 FPS | 41.48 FPS | 62.24 FPS |

Les réserves sont particulièrement importantes. Le guide des opérateurs Amlogic place NMS dans le logiciel pour ADLA2 comme ADLA3. Le README définit ces mesures comme des résultats d'exécution du modèle natif sur NPU, sans prétraitement ni post-traitement ; il ne précise pas si tous les transferts mémoire sont inclus. Les mesures d'exactitude YOLO utilisent des sous-ensembles COCO de 300 images et non l'ensemble complet de validation ; les mesures de classification utilisent ImageNet `val1000`. Les tableaux rapportent aussi une latence YOLOv8n A311Y3 différente des FPS, dans des conditions que le dépôt ne rapproche pas. Les fréquences, le mode d'alimentation, le refroidissement, l'état thermique stabilisé, les versions exactes du SDK/BSP et la durée du test ne sont pas indiqués. Utilisez ces chiffres pour comprendre une pile du fabricant, pas pour la classer face à une autre.

### Pourquoi Amlogic est intéressant sur le plan stratégique

Amlogic réunit quatre propriétés inhabituelles : une large présence dans les SoC embarqués, une nouvelle pile de compilation publique, une matrice actuelle de modèles qui recoupe largement les tâches de LibreYOLO et peu d'intégrations natives dans les principales bibliothèques d'entraînement. Les dépôts actuels sont aussi récents : ils sont apparus publiquement fin 2025 et début 2026, et les manuels détaillés datent de mai à juillet 2026. Il existe donc une vraie possibilité d'être parmi les premiers, avec le risque correspondant de stabilité de l'API.

Une intégration propre devrait utiliser l'export ONNX déterministe de LibreYOLO en entrée du compilateur, exiger des images de calibration représentatives pour les builds basse précision, produire `.adla` avec un manifeste de métadonnées et charger le résultat via un adaptateur NNSDK2. A311D historique devrait avoir une cible `.nb` clairement distincte, car présenter `.nb` et `.adla` comme un seul backend entraînerait des erreurs de compatibilité silencieuses. Les dépôts Amlogic ont des licences Apache-2.0 à leur niveau principal, mais les droits de redistribution des wheels, bibliothèques partagées, `nnserver` et binaires AAR Android fournis doivent être confirmés directement avant que LibreYOLO les republie.

## Les écosystèmes publics de déploiement les plus solides

Les entreprises suivantes offrent actuellement la combinaison la plus claire de matériel accessible, de documentation technique publique et de preuves concernant des modèles nommés. Cela ne les rend pas systématiquement plus rapides. Leurs affirmations de compatibilité sont simplement plus faciles à évaluer avant l'achat du matériel.

### Hailo

Hailo est l'exemple de référence d'un écosystème dédié à l'accélération de la vision. Les modèles sont analysés et convertis en Hailo Archive, optimisés et quantifiés avec des images représentatives, puis compilés dans un fichier Hailo Executable Format (`.hef`). Les applications chargent ce HEF via HailoRT.

Le [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) contient des recettes, des modèles pré-entraînés, des configurations de post-traitement et des mesures de performance pour la détection, la segmentation, la classification, la pose et d'autres tâches. Sa couverture YOLO comprend YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 et YOLO26, ainsi que YOLOX, DAMO-YOLO, SSD et EfficientDet. La [table de détection d'objets Hailo-8](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.19.0/docs/public_models/HAILO8/HAILO8_object_detection.rst), épinglée sur une version précise, est une meilleure source de compatibilité qu'une page produit générique. L'[application autonome](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/README.md) documente aussi des pipelines YOLO déployables.

Une frontière de version est à prendre en compte. Hailo-8 et Hailo-8L utilisent l'ancienne génération Model Zoo 2.x, Dataflow Compiler 3.x et HailoRT 4.x, tandis que Hailo-10 et Hailo-15 utilisent la génération actuelle 5.x. Les recettes, le comportement des parseurs et les HEF ne sont pas interchangeables simplement parce que les appareils portent le nom Hailo.

Hailo-15 possède également une couche applicative distincte. Elle utilise Vision Processor Software Package et Hailo Media Library, et non le parcours hôte de type TAPPAS pour Hailo-8/10H. Le dépôt de référence public [`hailo-camera-apps`](https://github.com/hailo-ai/hailo-camera-apps) a été archivé le 3 mai 2026, tandis que [`hailo-apps-core`](https://github.com/hailo-ai/hailo-apps-core) reste un framework applicatif public distinct. L'archivage d'un dépôt ne prouve pas la fin de vie du produit, mais les projets Hailo-15 devraient confirmer le paquet applicatif actuel dans le Developer Zone.

Le [guide de déploiement Hailo](/docs/export/hailo) de LibreYOLO s'arrête volontairement à un transfert ONNX statique, car le compilateur propriétaire ne peut pas être fourni comme dépendance Python. C'est la limite honnête entre produire un graphe adapté et prétendre avoir un HEF testé.

### Rockchip

Rockchip possède l'un des plus grands écosystèmes Linux embarqués, amateur et commercial, notamment grâce aux cartes RK3588, RK3576 et RK356x. Le [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) public convertit et quantifie les modèles sur un hôte Linux x86 ; RKNN Runtime ou Toolkit-Lite exécute ensuite le fichier `.rknn` obtenu sur la carte.

Le [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) précise particulièrement bien les puces, familles de modèles et précisions. Sa table actuelle répertorie les parcours FP16 et INT8 pour YOLOv5/6/7/8/10/11, YOLOX, PP-YOLOE, YOLO-World et les variantes de segmentation YOLOv5/v8 ; YOLOv8 OBB et YOLOv8 pose sont répertoriés uniquement en INT8. Elle couvre également l'OCR, le visage et d'autres réseaux de segmentation.

LibreYOLO dispose déjà d'un [exporteur RKNN](/docs/export/rknn) direct et prudent. Il valide quatre variantes exactes de détection sur RK3588, peut comparer le simulateur du compilateur à ONNX Runtime et rejette les familles non validées au lieu d'assimiler une compilation réussie à des prédictions correctes. Cette revendication de prise en charge ciblée est un bon modèle pour chaque futur backend NPU.

### Axelera AI

Axelera AI, souvent orthographié à tort « Accelera », développe l'AIPU Metis, vendu en produits M.2, PCIe et multipuces. Metis est la famille disponible à la commande publique ; Europa et Titania sont des produits annoncés dans le portefeuille, leur disponibilité ne doit donc pas être ramenée à un statut unique. Le [SDK Voyager est public sur GitHub](https://github.com/axelera-ai-hub/voyager-sdk) et gère la sélection, la compilation, la quantification, l'exécution des modèles et les pipelines applicatifs ; le support client nécessite toujours un compte.

Le [Voyager Model Zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo/) public compte parmi les plus informatifs du marché. Il répertorie les variantes de modèle, les tailles d'entrée, la précision, l'exactitude et les performances mesurées pour YOLOv3, YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 n/s/m/l/x, YOLOX, YOLO-NAS, OBB, pose et segmentation, ainsi que de nombreux modèles de classification et de prédiction dense. Publier les pertes de quantification et l'exactitude à côté de la vitesse est plus utile que de publier seulement une valeur TOPS de pointe.

Le nom des artefacts a changé avec la génération de l'API. Le flux de compilation alpha [Pipeline Builder](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) produit un modèle `.axm` et peut empaqueter un pipeline portable en `.axe`. La [documentation du pipeline classique](https://docs.axelera.ai/sdk/reference/pipeline/model-formats/) décrit `.axmodel` avec les métadonnées et un manifeste. Une intégration doit détecter la génération Voyager installée au lieu de coder en dur une extension unique.

### Qualcomm

Qualcomm offre une immense portée de déploiement, mais l'expression « NPU Qualcomm » masque plusieurs interfaces. Selon l'appareil et la catégorie de produit, les développeurs utilisent Hexagon HTP via QAIRT/QNN, l'ancien flux SNPE, les services de compilation Qualcomm AI Hub, LiteRT ou le fournisseur d'exécution QNN d'ONNX Runtime.

Le dépôt officiel [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) constitue une preuve solide au niveau des modèles. Il publie des paquets optimisés pour YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, la détection/segmentation/pose YOLO26, YOLOX, YOLO-World, YOLOR, RF-DETR et de nombreux modèles de classification, profondeur et pose. AI Hub propose aussi du profiling par appareil, ce qui importe, car un modèle compilé pour une génération HTP n'est pas automatiquement portable vers une autre.

Le coût principal d'intégration vient de la taille de la matrice : Android ou Linux embarqué, références QCS ou Snapdragon grand public, architecture HTP, version QNN/QAIRT et schéma de quantification ont tous leur importance. Les pages actuelles de Qualcomm se contredisent au sujet du statut [Dragonwing IQ-9075](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075) : la page produit le marque Active et son [EVK](https://www.qualcomm.com/developer/hardware/qualcomm-iq-9075-evaluation-kit-evk) peut être évalué, tandis que le [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) marque toujours IQ-9075 Sampling et indique une longévité jusqu'en 2038. Le préfixe de commande/SKU est QCS9075 ; Qualcomm annonce des configurations de 50 et 100 TOPS INT8 denses, ainsi que la prise en charge d'Ubuntu/Yocto. Il faut confirmer le statut de production pour le SKU de commande précis. Une intégration LibreYOLO devrait enregistrer ce SKU au lieu de créer un dossier générique « Qualcomm ».

### DEEPX

Les accélérateurs DX-M1/DX-M1M de DEEPX utilisent le SDK DXNN. DX-COM compile et quantifie le modèle, DX-RT l'exécute et l'artefact déployé utilise le format `.dxnn`. L'entreprise publie [DX-AllSuite](https://github.com/DEEPX-AI/dx-all-suite), des exemples applicatifs dans [DX-APP](https://github.com/DEEPX-AI/dx_app) et un vaste [Model Zoo en ligne](https://developer.deepx.ai/modelzoo/).

Le catalogue public comptait 354 entrées avec DX-COM 2.4.0 et DX-RT 3.4.0 lors de sa vérification le 15 août 2026. Sa couverture documentée comprend YOLOv3 à YOLO11 et YOLO26, YOLOX, SSD, EfficientDet, NanoDet, DAMO-YOLO, YOLO segmentation, pose et boîtes orientées. DEEPX est une cible d'intégration particulièrement plausible, car les limites entre compilateur, runtime et artefact de déploiement sont clairement définies et la couverture vision publique est large.

## Les SoC industriels forment plusieurs écosystèmes

Les fabricants industriels proposent souvent des cycles de vie plus longs ainsi qu'une meilleure intégration caméra, sûreté et temps réel que les SoC des cartes de makers. Leurs piles IA peuvent être plus complexes, car un même fabricant commercialise parfois plusieurs architectures NPU sans lien entre elles.

### Texas Instruments

Les processeurs Edge AI actuels de TI comprennent AM62A, AM67A, AM68A, AM69A et les appareils TDA4 associés. Le parcours logiciel combine Processor SDK Linux, le compilateur/runtime TIDL, [Edge AI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) et le [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo). TIDL peut s'intégrer à ONNX Runtime, TensorFlow Lite et à d'autres runtimes applicatifs, en déchargeant les sous-graphes pris en charge.

TI publie bien davantage qu'une simple affirmation d'import de framework : le zoo contient des modèles convertis et des métadonnées de performance pour la détection d'objets, la segmentation, la pose, la classification, la profondeur et d'autres tâches. TI précise aussi que les artefacts du model zoo sont des points de départ pour le développement, pas automatiquement des ressources prêtes pour la production. Cette réserve manque souvent dans les comparatifs de fabricants.

TI présente également le [TDA54-Q1](https://www.ti.com/product/TDA54-Q1) comme étant en préversion, avec jusqu'à quatre NPU C7 et jusqu'à 400 TOPS pour cette puce ; la famille TDA5 est annoncée jusqu'à 1 200 TOPS. Ce sont les chiffres du fabricant pour une nouvelle génération, et ils ne permettent pas de transférer la validation de modèles TIDL AM6xA/TDA4 au TDA54-Q1 avant la publication d'une matrice propre à cette cible par TI.

### NXP

NXP nécessite au moins quatre descriptions de backend :

| Cible NXP | Accélérateur | Parcours compilateur/runtime |
|---|---|---|
| i.MX 8M Plus | NPU VeriSilicon Vivante de 2.3-TOPS | eIQ avec TIM-VX/VX delegate |
| i.MX 93 | Arm Ethos-U65 | TFLite quantifié avec Arm Vela |
| i.MX 95 | NPU eIQ Neutron de NXP | Neutron Converter et runtime eIQ |
| Ara-1 / [Ara240](https://www.nxp.com/products/ARA240) | NPU discret dérivé de Kinara ; Ara240 annoncé jusqu'à 40 eTOPS | Ara SDK, intégré au parcours eIQ général |

Le [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) contient des artefacts ou recettes pour YOLOv4-tiny, YOLOv8, NanoDet, CenterNet, FastestDet, SSD Lite, YOLACT et d'autres modèles ; son entrée YOLOv5 a été ajoutée uniquement à titre documentaire. Une entrée validée pour un backend ne prouve pas sa prise en charge par les quatre. Les [conseils d'export YOLO pour les plateformes i.MX NXP](https://community.nxp.com/t5/i-MX-Processors-Knowledge-Base/Exporting-YOLO-Models-for-NXP-i-MX-Platforms/ta-p/2381361) de NXP illustrent ce problème de conversion propre à chaque cible. NXP indique que l'eIQ Toolkit monolithique [n'a plus été mis à jour après la version 1.17 au T3 2025](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/eIQ-FAQ/ta-p/1099741) ; les parcours actuels utilisent des paquets autonomes comme eIQ Neutron SDK.

NXP a [finalisé l'acquisition de Kinara en octobre 2025](https://media.nxp.com/news-releases/news-release-details/nxp-completes-acquisitions-aviva-links-and-kinara-advance/). La [fiche technique Ara240](https://www.nxp.com/docs/en/fact-sheet/ARA240DNPUFS.pdf) annonce jusqu'à 40 eTOPS et indique 313 images par seconde pour YOLOv8n. NXP répertorie Ara SDK et eIQ Toolkit sur la page produit, mais cela ne prouve pas l'interchangeabilité des artefacts avec le parcours Neutron distinct de [l'i.MX 95](https://eiq.nxp.com/learning-hub/convQuant/neutron.html). Le [module M.2 de 16 GB](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT) est actif tandis que l'option USB est en préproduction. NXP définit le « e » de eTOPS comme « equivalent », et non « effective » ; ce n'est pas la mesure TOPS INT8 denses d'un autre fabricant et il ne faut pas les comparer directement.

### STMicroelectronics

Les [appareils STM32N6x7](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html), dont STM32N657 et STM32N647, intègrent un accélérateur Neural-ART de 600-GOPS dans un produit de classe MCU ; la gamme polyvalente N6x5 n'inclut pas cet accélérateur. [STM32Cube AI Studio](https://www.st.com/en/development-tools/stedgeai-cubeai.html) et ST Edge AI Core analysent et optimisent les modèles importés, tandis que [STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) fournit les parcours de déploiement, d'optimisation, de benchmark et d'exemple.

La [présentation actuelle de la détection d'objets](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/docs/README_OVERVIEW.md) documente Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 et ST-YOLOX, ainsi que des services de classification, pose et segmentation. Ce produit ne joue pas dans la même catégorie de performance qu'une carte PCIe de 200-TOPS. Son intérêt tient à la possibilité d'intégrer acquisition caméra, inférence et contrôle dans une enveloppe de puissance et de mémoire très contrainte.

### Renesas

Les processeurs Renesas RZ/V intègrent des accélérateurs DRP-AI. Le logiciel a évolué de DRP-AI Translator et DRP-AI TVM vers [RUHMI](https://www.renesas.com/en/software-tool/ruhmi-framework), fondé sur la technologie MERA d'EdgeCortix. Le [dépôt RZ/V DRP-AI TVM](https://github.com/renesas-rz/rzv_drp-ai_tvm) ouvert et la [liste de validation RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) citent YOLOv5, YOLOv8, YOLO11 et YOLO26 n/s/m, YOLOX, des réseaux de pose et segmentation, ainsi que des modèles de classification.

Renesas constitue une cible crédible en robotique et dans l'industrie, mais il faut consigner la carte précise, la génération DRP-AI, la version du traducteur et le post-traitement côté CPU pour assurer la reproductibilité.

## Capteurs intelligents et processeurs conçus pour les caméras

### Sony IMX500

Le Sony IMX500 n'est pas un processeur applicatif classique. Il combine détection d'image et traitement IA afin de réaliser l'inférence dans le capteur et de réduire la quantité de données image qui quitte la caméra. Raspberry Pi AI Camera rend cette architecture particulièrement accessible.

Le dépôt officiel des [modèles Raspberry Pi IMX500](https://github.com/raspberrypi/imx500-models) publie des paquets YOLOv8n, YOLO11n, EfficientDet Lite0, NanoDet+ et SSD MobileNetV2 FPN Lite. La [documentation AI Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) explique la conversion, l'empaquetage et le post-traitement dans le capteur. L'unité de déploiement est un paquet RPK et non un fichier ONNX générique ; la mémoire du capteur et les contraintes d'opérateurs sont des limites de conception centrales. La [page produit](https://www.raspberrypi.com/products/ai-camera/) de Raspberry Pi indique que cette caméra restera en production au moins jusqu'en janvier 2028.

### Ambarella

Les processeurs CVflow d'Ambarella sont bien implantés dans les caméras, les drones et la vision automobile. Les familles actuelles comprennent CV72/CV75 pour les caméras, CV5/CV52 pour les systèmes de vision plus puissants et les appareils automobiles CV3-AD. La plateforme développeur Cooper et le flux de compilation/runtime CVflow constituent la pile logicielle nommée publiquement.

Le [model garden développeur public](https://www.ambarella.com/developer/model-garden/) cite YOLOX, RTMDet, DeepLabV3+, TopFormer, OWL-ViT et des modèles multimodaux. Toutefois, la documentation détaillée du compilateur et les téléchargements restent destinés aux partenaires. Une intégration Ambarella devrait donc commencer par une relation avec le fabricant ou un OEM, et non par l'hypothèse qu'un paquet pip public est disponible.

### Synaptics Astra

Synaptics propose trois cibles pertinentes. Les systèmes Astra SL1600/SL1680 utilisent le [toolkit SyNAP](https://developer.synaptics.com/docs/synap/introduction), qui convertit un modèle source et une description YAML en paquet `model.synap`. Les appareils SL2611/13/15/17/19 plus récents utilisent la [plateforme Torq](https://developer.synaptics.com/docs/torq/introduction), un flux fondé sur IREE/MLIR qui produit `.vmfb`. La [série SR100](https://developer.synaptics.com/docs/sr/introduction-sr) est une famille MCU distincte, bâtie autour de Cortex-M55 et Ethos-U55 avec son propre SDK. Ces artefacts et API ne sont pas interchangeables.

Le [tutoriel benchmark YOLO officiel](https://developer.synaptics.com/docs/sl/tutorials/vision/benchmark-yolo) est particulièrement concret : il exporte YOLOv8 vers TFLite, effectue une calibration UINT8 asymétrique dans SyNAP, compile pour SL1680, puis lance `synap_cli` et l'application de détection d'objets. Il prouve YOLOv8n/v8s sur SL1680 et décrit la prise en charge de SyNAP jusqu'à YOLO11. Un [guide distinct de détection d'objets SL261x](https://developer.synaptics.com/docs/sl/sl2600/getting-started/object-detection) exécute YOLOv8 à partir d'un `.vmfb` Torq. Ce sont des exemples officiels propres à leurs cibles, pas une affirmation que tous les graphes YOLO sont pris en charge sur les trois familles.

### MediaTek Genio

MediaTek mérite une rubrique complète, car son [IoT AI Hub](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) actuel expose désormais la matrice matériel/logiciel, les paquets de modèles et les tableaux de benchmark qui étaient difficiles à vérifier dans les anciennes études de marché. Genio 360/360P/420/520/720 utilise NeuroPilot 8 avec MDLA 5.3 ; Genio 510/700 utilise NP6 avec MDLA 3.0 ; Genio 1200 utilise NP6 avec MDLA 2.0. MediaTek indique que les générations NeuroPilot et MDLA, les opérateurs, le compilateur et le runtime sont liés par version pendant toute la durée de vie de chaque SoC.

Le parcours d'IA analytique est centré sur TFLite :

```text
PyTorch model
    -> NeuroPilot Converter and representative calibration
    -> quantized .tflite
    -> version-matched ncc-tflite compiler
    -> chip-generation-specific .dla
    -> Neuron Runtime on MDLA
```

Les produits Genio récents proposent également la délégation LiteRT en ligne et un parcours ONNX Runtime CPU/NPU sur les systèmes d'exploitation pris en charge. Ces modes d'exécution diffèrent du `.dla` hors ligne. Le [guide d'architecture logicielle](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/software_architecture.html) et la [matrice de ressources](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) rendent cette distinction explicite.

La [table officielle des modèles analytiques](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical.html) publie des entrées YOLOv5s et YOLOv8s avec benchmarks et guides de conversion couvrant plusieurs générations MDLA. Elle ne distribue explicitement pas d'artefacts YOLO préconvertis en raison des restrictions AGPL-3.0. Ses mesures hors ligne Quant8 en 640 x 640 comprennent respectivement 5.35 ms et 8.04 ms sur Genio 720. La [page du modèle YOLOv8s](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical/YOLOv8s.html) publie les tenseurs d'entrée/de sortie, les temps propres à chaque backend et des avertissements sur les opérations personnalisées MediaTek. Ce sont des benchmarks du fabricant, pas des résultats caméra de bout en bout.

L'accès constitue le facteur limitant. La documentation Yocto est publique et détaillée, mais le lot convertisseur/compilateur intégré NP8 actuel et une grande partie de la documentation Android sont indiqués comme ressources sous NDA ou réservées aux clients directs. Un compte développeur ordinaire ne donne pas le même accès qu'un compte client MediaTek Online. LibreYOLO devrait donc considérer Genio comme techniquement validé, mais dépendant d'un partenariat pour une intégration de compilation reproductible de modèles personnalisés.

## Écosystèmes Edge AI centrés sur la Chine

Plusieurs des plateformes vision à bas coût les plus performantes et accessibles apparaissent peu dans les études de marché en anglais.

### D-Robotics et plateformes BPU dérivées de Horizon

D-Robotics maintient l'écosystème de cartes de développement RDK autour des accélérateurs BPU utilisés dans RDK X3, X5, Ultra et les produits récents de série S100. La branche `rdk_x5` actuelle du [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo) documente YOLOv5/v5u, YOLOv8/9/10/11/12/13/26, YOLOE, YOLO-World, la détection, la segmentation, la pose, la classification, l'OCR et les parcours CLIP pour RDK X5. X3 utilise une branche distincte, tandis que le déploiement actuel de la série S se trouve dans la branche [`rdk_s`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s) du zoo principal ; l'ancien dépôt [`rdk_model_zoo_s`](https://github.com/D-Robotics/rdk_model_zoo_s) contient des démos historiques. La liste moderne X5 ne prouve donc pas que chaque modèle a été validé sur X3, Ultra ou S100.

OpenExplorer/Algorithm Toolchain importe ONNX ou Caffe pour le PTQ et prend en charge des flux QAT orientés PyTorch. Le déploiement dépend de la plateforme : le RDK X5 actuel utilise `.bin` via `hbm_runtime` sur `libdnn`, tandis que le flux principal `rdk_s` actuel utilise `.hbm` via une API `hbm_runtime` homonyme sur `libhbucp` ; X3 conserve son ancienne branche et ses interfaces d'inférence. L'ancien contenu de `rdk_model_zoo_s` décrit également des parcours historiques `.bin` et `.hbm` : ces artefacts doivent rester associés à leur branche et chaîne d'outils correspondantes. Les opérateurs non pris en charge peuvent s'exécuter sur CPU. Les exemples publics sont utiles, mais la correspondance entre version de RDK OS, firmware de carte, chaîne d'outils et binaire de modèle fait partie du contrat de compatibilité.

### AXERA

Les familles AX650/AX630/AX620 d'AXERA sont présentes dans des caméras et cartes IA compactes, notamment la gamme MaixCAM de Sipeed. Pulsar2 importe ONNX, effectue la calibration et l'analyse de précision, puis produit `.axmodel` ; AXEngine exécute l'artefact.

Les [exemples AXERA officiels](https://github.com/AXERA-TECH/ax-samples) couvrent YOLOv5/6/7/8/9/10/11/13/26, YOLOX et YOLO-World ; la prise en charge des tâches et puces varie selon la plateforme. Sur les cibles prises en charge, les exemples YOLO26 actuels couvrent détection, pose, segmentation et OBB. Les [exemples de conversion Pulsar2](https://pulsar2-docs.readthedocs.io/en/latest/appendix/model_convert_examples.html) montrent le travail réel : noms de tenseurs exacts, coupures des sorties, transformations de format, archives de calibration et paramètres du matériel cible.

### SOPHGO

TPU-MLIR de SOPHGO est l'une des piles de compilation les plus intéressantes pour une intégration indépendante, car le compilateur lui-même est open source. Il importe ONNX, PyTorch, TFLite et Caffe, abaisse et quantifie les graphes, puis produit `.bmodel` pour les cibles BM ou `.cvimodel` pour le matériel CV18xx.

Le [projet TPU-MLIR](https://github.com/sophgo/tpu-mlir) prend en charge les parcours FP32, BF16, FP16 et INT8 selon la cible. La [collection de démos SOPHON](https://github.com/sophgo/sophon-demo/blob/release/README_EN.md) cite YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, des variantes OBB/segmentation, SSD, CenterNet, RetinaFace, SAM/SAM2 et OCR. Comme toujours, une démo sur BM1684X ne valide pas le même artefact sur BM1688 ou CV18xx.

### Huawei Ascend

La gamme d'inférence edge de Huawei comprend le silicium Ascend 310/310P/310B dans les produits Atlas 200I et 300I. CANN fournit le compilateur ATC, le runtime AscendCL et des noyaux d'opérateurs propres aux puces ; ATC convertit ONNX et d'autres représentations sources en modèle `.om` hors ligne.

La [documentation Atlas 200I DK A2](https://www.hiascend.com/en/hardware/developer-kit-a2/specification) actuelle et les exemples CANN proposent un parcours YOLOv5 `.om` pour Ascend 310B. Le [ModelZoo Ascend](https://github.com/Ascend/modelzoo) plus large et plus ancien contient YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN, pose et segmentation, mais ne doit pas être présenté comme si chaque entrée avait été validée de nouveau sur 310B. CANN, les noyaux, le firmware et le SoC doivent correspondre. Un `.om` pour Ascend310P n'est pas un artefact Ascend générique.

### Cambricon

Les accélérateurs MLU de Cambricon utilisent Neuware et le moteur d'inférence MagicMind. Le dépôt public [MagicMind Cloud](https://github.com/Cambricon/magicmind_cloud) est épinglé sur MagicMind 1.7 et une matrice MLU370-X4/S4 ; il apporte des preuves historiques de compatibilité et ne valide ni le SDK actuel ni MLU270. Le dépôt documente les parcours PyTorch, ONNX, Caffe et Paddle ; la prise en charge du framework TensorFlow a été supprimée en 1.7 même si des exemples TensorFlow historiques subsistent. Le [portail développeur série 3](https://developer.cambricon.com/index/document/index/classid/3.html) actuel de Cambricon liste des versions ultérieures du SDK global : le dépôt d'exemples publics et la pile commerciale actuelle ne doivent pas être présentés comme une seule version.

Le dépôt officiel de modèles [MagicMind cloud](https://github.com/Cambricon/magicmind_cloud) publie une matrice MLU370-X4/S4 particulièrement directe. Il répertorie YOLOv3, Tiny-YOLOv3, YOLOv4, YOLOv5, YOLOv7, YOLOv8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN, DeepLab et UNet, en précisant notamment si des exemples C++ ou Python existent. Les preuves sont solides ; le principal obstacle reste l'accès au SDK et au conteneur via les canaux Cambricon.

### Canaan/Kendryte

Les puces KPU actuelles K230/K230D et historiques K210/K510 de Kendryte sont importantes pour les cartes à bas coût et les caméras intelligentes. Le compilateur [nncase](https://github.com/kendryte/nncase) importe ONNX ou TFLite, utilise des données de calibration pour la conversion en virgule fixe, simule le résultat sur hôte et produit `.kmodel`.

Le [guide officiel de développement nncase K230](https://github.com/kendryte/k230_docs/blob/main/en/01_software/board/ai/K230_nncase_Development_Guide.md) explique comment compiler, simuler et exécuter YOLOv5s. Le [catalogue de démos IA](https://github.com/kendryte/k230_docs/blob/main/en/02_applications/ai_demos/K230_AI_Demo_Introduction.md) distinct comprend des artefacts de détection, segmentation et pose YOLOv8n, tandis que le [journal des modifications CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) actuel ajoute des exemples optimisés YOLOv8/11/26 de classification, détection, segmentation, OBB et pose. Le résultat YOLOv5s de la fiche technique est le [benchmark publié](https://github.com/kendryte/k230_docs/blob/main/en/00_hardware/K230_datasheet.md) ; les versions du compilateur et du plug-in binaire KPU doivent correspondre au SDK de la carte.

### Allwinner

Allwinner V853 associe du matériel média orienté caméra à un NPU Vivante de 1-TOPS. Le [guide NPU officiel en anglais](https://docs.aw-ol.com/v853/en/npu/dev_npu/) décrit l'import Acuity/Pegasus, la quantification, la vérification et le déploiement via `viplite`. Les pages de modèles d'Allwinner et son [guide YOLOv5](https://docs.aw-ol.com/v853/npu/npu_yolov5/) citent YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet et des réseaux visage/personne.

Il s'agit d'une prise en charge réelle de la vision par ordinateur, mais qui reste liée à l'ancienne chaîne Tina Linux/Vivante et aux téléchargements du SDK soumis à un compte. La plateforme a sa place dans la cartographie du marché, avec cette limite clairement indiquée.

## Plateformes coréennes, taïwanaises et chinoises moins connues

Les listes en anglais passent souvent directement de Rockchip à NVIDIA et omettent un deuxième groupe de silicium documenté et réellement disponible. Certains de ces écosystèmes présentent une matrice YOLO actuelle plus vaste que celle de jeunes entreprises occidentales plus connues.

### Mobilint

Le fabricant coréen Mobilint commercialise REGULUS basse consommation et ARIES MLA100 à débit supérieur. Son SDK qb accepte des entrées PyTorch, TensorFlow, TFLite, ONNX et Keras, quantifie en INT8 et produit l'artefact compilé `.mxq`. La [matrice vision publique](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) cite YOLOv3/v5/v7/v8/v9/v10/11/12/26 en détection, plus des variantes YOLO de segmentation, pose et OBB. La [page ARIES](https://www.mobilint.com/aries/mla100) publie même des résultats spécifiques à YOLO11s et YOLO26m. Les preuves d'intégration sont solides, même si l'accès au SDK et au matériel reste commercial.

### Sunplus

SP7350/C3V de Sunplus utilise une pile NPU dérivée de Vivante, empaquetée différemment de celle d'Allwinner ou de l'ancien Amlogic. Son parcours public combine conversion Acuity, environnement Docker NPU, runtime SNNF et sortie `.nb`. Le [guide YOLOv8 personnalisé](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) officiel décrit conversion et déploiement au lieu de simplement revendiquer la prise en charge d'un framework. Une IP apparentée ne rend pas `.nb` portable vers un autre SoC basé sur Vivante.

### ESWIN Computing

La famille RISC-V d'ESWIN comprend EIC7700/EIC7700X et EIC7702/EIC7702X à deux puces ; EIC7700 équipe notamment la carte Milk-V Megrez commercialisée. ENNP réunit EsQuant, le compilateur EsAAC actuel, la génération de données de référence, la simulation et le runtime ESSDK, avec une sortie compilée `.model` ; d'anciens documents ENNP utilisaient le nom `ennc-compile`. La [page EsAAC](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) actuelle documente l'entrée ONNX ; les autres frameworks doivent donc être exportés ou convertis vers cette représentation prise en charge. La [présentation ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/introduction) et le [tutoriel YOLOv3](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3) hébergés par Milk-V établissent un parcours public de bout en bout, mais pas une vaste matrice actuelle de modèles/exactitude.

### Nuvoton M55M1

Le M55M1 de Nuvoton est une conception Cortex-M55 de classe MCU avec Ethos-U55-256, et non un processeur applicatif Linux. NuEdgeWise/NuML et Arm Vela préparent les modèles TFLite Micro quantifiés. Le dépôt public [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) cite YOLOv8-nano, YOLOX-nano, YOLO Fastest v1.1, SSD-MobileNet FPNLite et plusieurs classificateurs. Ces petits exemples de réseaux constituent des preuves crédibles pour cette catégorie de mémoire/puissance, pas pour les variantes YOLO classiques à 640 pixels.

### Rebellions et FuriosaAI

La famille ATOM de Rebellions utilise le SDK RBLN et les artefacts compilés `.rbln`. Sa [version de prise en charge officielle](https://docs.rbln.ai/v0.8.2/supports/release_note.html) cite YOLOv3 tiny/full/SPP, YOLOv5 n à x, YOLOv6 n à l, des variantes YOLOv7 et YOLOv8 n à x. La [matrice actuelle de prise en charge des cartes](https://docs.rbln.ai/latest/supports/version_matrix.html) indique ATOM CA02 et ATOM+ CA12 en fin de vie, tandis que ATOM+ CA22 et ATOM-Max CA25 sont actifs. ATOM-Lite CA21 possède une page produit, mais est absent de cette matrice du SDK ; le statut de sa chaîne d'outils actuelle reste donc incertain. La documentation est publique, mais le wheel compilateur nécessite les identifiants du portail Rebellions.

FuriosaAI doit être réparti par génération. Warboy est l'ancien accélérateur de vision, avec artefacts INT8 `.enf` et [model zoo officiel](https://developer.furiosa.ai/furiosa-models/latest/) contenant SSD, YOLOv5M/L et YOLOv7-w6-pose. RNGD utilise une [pile `.fxb`](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html) différente, principalement destinée aux charges LLM/VLM. La [feuille de route](https://developer.furiosa.ai/latest/en/overview/roadmap.html) actuelle de Furiosa indique que la prise en charge vision de YOLOv8m a été achevée au T4 2024, mais les pages publiques actuelles sur les modèles pris en charge et les performances sont centrées sur LLM/VLM et ne présentent pas de matrice détaillée d'exactitude/performance pour YOLOv8m. Il s'agit d'une preuve concernant une génération commercialisée, pas d'une affirmation de compatibilité avec Warboy.

### Realtek, Telechips, T-Head et SigmaStar

Ces quatre plateformes existent, mais leur parcours de modèles personnalisés est plus restreint, plus ancien ou réservé :

| Plateforme | Preuve publique reproductible | Limite à conserver |
|---|---|---|
| Realtek AmebaPro2 | Le [SDK Arduino/FreeRTOS](https://github.com/Ameba-AIoT/ameba-arduino-pro2) fournit des modèles `.nb` YOLOv3/4/7-tiny, SCRFD et MobileFaceNet | Pour convertir un modèle personnalisé, il faut se manifester ou contacter Realtek |
| Telechips TOPST TCC7500 | Le [projet officiel 2026](https://docs.topst.ai/blog/31) a converti et déployé YOLOv8s | La conversion UFLD v1/v2 a échoué sur des couches non prises en charge ; l'article propose seulement une solution de contournement avec découpage/post-traitement. Les ressources complètes du compilateur/des opérateurs nécessitent souvent une autorisation par e-mail |
| T-Head TH1520 | Le [guide d'application Sipeed](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) exécute YOLOv5n/s avec HHB et CSI-NN2/SHL | La pile publique existe, mais sa maintenance et la clarté des versions sont à la traîne des principaux acteurs actuels |
| SigmaStar SSU9383CM | La documentation actuelle établit les API du runtime MI_IPU ; les documents plus anciens de SGS_IPU décrivent `.sim` vers `sgsimg.img` et un postprocesseur historique cite SSD et YOLOv1/2/3 | Aucune preuve publique ne confirme que les artefacts du compilateur ancien ou les parcours de modèles cités s'appliquent au SSU9383CM |

### HiSilicon smart vision n'est pas Huawei Ascend

Les SoC caméra HiSilicon Hi3516CV610/DV500, Hi3519DV500 et Hi3403V100 exposent un flux ATC vers `.om` via les runtimes NNN/SVP-NNN. Le [model zoo HiSpark](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) officiel cite YOLOv3/4/5/6/7/8/9/10/11, segmentation/pose YOLO11, OBB/World/segmentation YOLOv8, OCR, profondeur et TinySAM dans une matrice propre aux cibles, notamment Hi3403 et Hi3591P. Cela ne prouve pas que chaque entrée fonctionne sur tous les SoC de vision intelligente cités plus haut. Certaines entrées sont des éléments de feuille de route : il faut distinguer les exemples disponibles du support prévu. Le dépôt précise que les modèles fournis par son ModelZoo sont réservés à un usage non commercial. Le vocabulaire ATC/`.om` commun ne prouve pas la compatibilité d'artefact ou de runtime avec la gamme Ascend fondée sur CANN.

## Entreprises émergentes et accélérateurs spécialisés

### MemryX

Les modules MX3 de MemryX utilisent une architecture dataflow et peuvent associer plusieurs puces. Le [Neural Compiler](https://developer.memryx.com/tools/neural_compiler.html) accepte des modèles ONNX, TensorFlow Lite, Keras et TensorFlow, puis produit un paquet DFP utilisé par le runtime de l'accélérateur. Ses [API runtime](https://developer.memryx.com/api/accelerator/accelerator.html) prennent en charge des pipelines multi-modèles et de streaming.

La documentation et les exemples MemryX incluent le prétraitement et le post-traitement YOLO optimisés pour la détection, la segmentation et la pose. Les [notes de version du SDK](https://developer.memryx.com/release_notes.html) citent explicitement la prise en charge de YOLOv10, YOLO11 et YOLO26. La documentation publique est techniquement utile, bien qu'elle ne comporte pas de tableau unique modèles/exactitude/appareils comparable à celui d'Axelera.

### Kneron

Les produits KL520/KL530/KL630/KL720/KL730 de Kneron couvrent de petits accélérateurs USB et embarqués. Kneron PLUS gère le runtime applicatif, tandis que la version actuelle 0.33.1 de Model Toolchain documente la conversion, la quantification, l'évaluation, la simulation et la compilation `.nef` pour ces cibles. KL830 apparaît dans certaines [API runtime PLUS](https://doc.kneron.com/docs/plus_c/introduction/run_examples/), mais pas dans la [liste actuelle des cibles du compilateur](https://doc.kneron.com/docs/toolchain/manual_1_overview/) ; sans confirmation du fabricant, il ne faut pas généraliser la compilation `.nef` à KL830.

L'[exemple YOLO officiel](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) montre le processus avec Tiny-YOLOv3 et d'autres ressources Kneron couvrent l'entraînement et le déploiement YOLOv5. Les preuves sont crédibles, mais la couverture est plus étroite que les vastes matrices modernes de Hailo, Axelera, Rockchip ou DEEPX.

### SiMa.ai

MLSoC et la [plateforme Modalix 50-TOPS en production](https://sima.ai/press-release/sima-ai-next-gen-platform-for-physical-ai-in-production/) de SiMa.ai utilisent l'environnement logiciel Palette. ModelSDK, le compilateur MLA et ModelExecutor couvrent import, quantification, compilation, profiling et exécution. Selon la charge et le produit, les outils prennent en charge des parcours INT8, INT16 et BF16.

Les notes de version de l'entreprise citent les pipelines YOLOv7, détection/pose/segmentation YOLOv8, YOLOX et segmentation YOLOX, DETR, Mask R-CNN et EfficientDet validés. Il ne faut pas masquer une limite actuelle : les [notes de Palette SDK 2.1](https://docs.sima.ai/v2.1.2/pages/release_notes/2.1.html) signalent que le QAT ne fonctionne pas à cause d'une régression de quantification Python/PT2E. La solution de contournement documentée consiste à créer le ONNX annoté dans le SDK 2.0 et à le compiler avec la version 2.1. L'accès et l'achat du produit sont destinés aux entreprises.

La limite de déploiement est également documentée : le [flux de compilation ModelSDK](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) produit un `.tar.gz` compilé et peut générer un ELF exécutable, tandis que [MPK Tool](https://docs.sima.ai/v2.1.1/pages/palette/mpk_tools.html) empaquette un `.mpk` déployable. Ces sorties sont liées à la cible MLSoC/Modalix et à la version Palette.

### EdgeCortix

SAKURA-II est un accélérateur annoncé à 60-TOPS pour la vision et l'IA générative, compilé via le framework logiciel MERA. EdgeCortix fournit également la technologie MERA au nouveau flux RUHMI de Renesas.

La [documentation produit SAKURA-II](https://www.edgecortix.com/en/edgecortix-sakura-ii-accelerator-brief) établit l'existence du matériel et du compilateur, et les [cartes M.2/PCIe](https://www.edgecortix.com/en/hardware) sont proposées à l'essai ou sur demande de commande. Aucune matrice YOLO publique suffisamment précise et récente n'a été trouvée. Ce manque constitue une information utile pour l'achat : demander une validation du modèle avant de s'engager sur le matériel.

### BrainChip

Akida de BrainChip est un processeur neuromorphique à événements, plutôt qu'un NPU classique à tenseurs denses. Son [environnement MetaTF](https://brainchip.com/metatf-dev-tools/) comprend des paquets Python installables pour créer des modèles, les quantifier en basse précision, les convertir, les simuler et les exécuter sur le matériel. BrainChip commercialise désormais le [coprocesseur AKD1500 disponible, au format M.2](https://brainchip.com/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/), avec l'ancien matériel AKD1000 et l'IP Akida 2.

La [fiche modèle Akida 2](https://brainchip.com/wp-content/uploads/2025/04/Akida-2-Model-Card-V1.1-Mar.25.pdf) actuelle cite un détecteur YOLOv2 AkidaNet0.5, CenterNet, AkidaUNet et des réseaux de reconnaissance faciale. D'autres documents couvrent FOMO et les charges à événements. Akida prend en charge des poids et activations exceptionnellement peu précis, mais un déploiement réussi peut nécessiter une conversion adaptée à l'architecture, au lieu de le traiter comme une cible ONNX INT8 générique. Un résultat du model zoo AKD1000 ou Akida 2 ne doit pas être attribué à AKD1500 sans rapport explicite de mappage/compatibilité.

### Blaize

Blaize commercialise des produits Pathfinder et Xplorer à base de P1600, avec Picasso SDK, NetDeploy et AI Studio comme parcours logiciel. Les [pages produit](https://www.blaize.com/products/) ciblent clairement la vision et l'Edge AI, mais il n'existe pas de matrice publique actuelle de compatibilité par modèle.

Ce guide classe donc Blaize comme commercial et à accès restreint plutôt que de combler ce manque avec des affirmations communautaires. Il faudrait exiger un rapport de compilation et d'exactitude fourni par le fabricant pour le modèle visé.

## TinyML et vision endpoint

### Arm Ethos-U et Alif

Arm concède sous licence les IP NPU Ethos-U55, U65 et U85 aux fabricants de puces. Le compilateur [Vela](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) prend des graphes LiteRT/TFLite quantifiés et transforme les sous-graphes pris en charge en opérations personnalisées Ethos-U. Les opérations non prises en charge peuvent rester sur le CPU ; « l'application s'exécute » et « tout le réseau s'exécute sur le NPU » sont donc deux affirmations différentes.

Parmi les implémentations figurent Ethos-U55 dans Alif Ensemble E7, Ethos-U65 dans NXP i.MX 93 et Ethos-U85 dans des systèmes récents. Le [kit E7 AI/ML AppKit](https://alifsemi.com/support/kits/ensemble-e7appkit/) d'Alif documente deux accélérateurs ML et montre le parcours TFLite vers Vela du E7 ; l'[Ensemble E8](https://alifsemi.com/ensemble-e8-series/) associe un Ethos-U85 à deux NPU Ethos-U55. Alif publie aussi un [benchmark de détection de visages YOLO-Fastest INT8](https://alifsemi.com/faster-ai-mcu-inferencing-low-power-consumption/) pour sa famille, en 192 par 192 sur Cortex-M55 plus Ethos-U55, mais pas de vaste matrice moderne YOLO par cible. Ces systèmes contraints en mémoire conviennent aux petits réseaux de classification et de détection, pas à des détecteurs arbitraires de 640 pixels simplement parce qu'ils sont exprimables en TFLite.

### Infineon PSOC Edge, Himax WiseEye2 et Analog Devices MAX7800x

Les familles [PSOC Edge E83](https://documentation.infineon.com/psocedge/docs/xsk1761304020519) et E84 d'Infineon associent Cortex-M55 à Ethos-U55 et ajoutent un bloc NNLite séparé côté basse consommation M33. Le [manuel d'architecture E84](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) documente des poids 8 bits, des activations 8 ou 16 bits et un flux Vela où les opérateurs pris en charge deviennent un flux de commandes NPU tandis que les opérations non prises en charge restent sur CPU. Le [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter) accepte les parcours TFLite, Keras et PyTorch ; la documentation d'architecture cite MobileNetV1/V2 comme noyaux représentatifs, et non comme benchmarks de la cible. Le [kit AI E84](https://documentation.infineon.com/psocedge/docs/cci1762693051052) comprend une caméra et utilise ModusToolbox, mais les ressources publiques ne fournissent pas encore de matrice YOLO nommée. Il s'agit d'une véritable plateforme vision programmable dont les preuves au niveau des modèles se développent, pas d'une cible de détection généraliste validée.

Le [HX6538 WiseEye2](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) de Himax associe lui aussi Cortex-M55 et Ethos-U55 pour la vision toujours active. Ses ressources logicielles publiques sont particulièrement concrètes pour cette catégorie de puissance : les exemples officiels du [Grove Vision AI Module V2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) comprennent la détection d'objets YOLOv8n, la pose, la classification de genre, la détection YOLO11n, le face mesh et PeopleNet. Le parcours unifié utilise TFLite Micro et Vela, puis se replie sur CMSIS-NN et les noyaux de référence si nécessaire. Il s'agit volontairement de petits modèles et résolutions, pas d'une preuve qu'un graphe YOLO classique de taille serveur tient dans le système.

Analog Devices emprunte une autre voie avec les [accélérateurs CNN MAX78000/MAX78002](https://www.analog.com/en/products/max78002.html). L'outil public [`ai8x-synthesis`](https://github.com/analogdevicesinc/ai8x-synthesis) quantifie un réseau entraîné et génère du code C, des poids et une configuration d'accélérateur propres à la cible, au lieu d'un paquet runtime ONNX portable. Les preuves du fabricant comprennent [l'identification des visages](https://www.analog.com/en/resources/app-notes/an-2616.html), la détection QR TinierSSD dans le [model zoo ADI](https://ez.analog.com/dsp/software-and-development-tools/edgebench/a/docs-faqs/DF912/available-models-in-adi-model-zoo) et des classificateurs d'image. Ces appareils conviennent à la vision endpoint, mais aucune matrice YOLO moderne officielle n'a été trouvée.

### GreenWaves GAP9

GAP9 associe des calculs RISC-V au moteur neuronal NE16. NNTool importe et quantifie les réseaux, tandis qu'AutoTiler et le GAP SDK génèrent le code déployable. Le [menu de réseaux neuronaux GAP9](https://github.com/GreenWaves-Technologies/nn_menu_gap9) public comprend MobileNet, EfficientNet, ResNet, MobileNet SSD et des applications faciales/de reconnaissance. GreenWaves publie aussi un [projet de détection de personnes YOLOX](https://github.com/GreenWaves-Technologies/yolox_people_detection) avec des résultats d'exactitude et de simulation.

Il s'agit de preuves TinyML exceptionnellement transparentes, mais le SDK complet est réservé aux clients qualifiés et les modèles sont bien plus petits que les variantes YOLO courantes en 640 par 640.

### Lattice sensAI

Lattice sensAI est une alternative fondée sur FPGA à un NPU fixe. sensAI Studio et Neural Network Compiler mappent un graphe pris en charge sur une IP d'accélérateur configurable pour les appareils ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E et Avant-X. La sortie dépend du FPGA choisi, du mode d'accélérateur, de la disposition mémoire et du bitstream, et n'est donc pas un modèle runtime portable.

Le [manuel actuel de Neural Network Compiler](https://www.latticesemi.com/-/media/LatticeSemi/Documents/UserManuals/MQ3/FPGA-UG-02052-8-0-Lattice-Neural-Network-Compiler-Software.ashx?document_id=52343) contient une table de topologies particulièrement directe. Elle répertorie YOLOv1 pour ECP5, YOLOv5 et YOLOv8 pour les modes optimisés/avancés de CrossLink-NX et CertusPro-NX, et YOLO11 uniquement avec l'IP Advanced mode. Elle cite aussi SSD, MobileNetV2-SSD, ResNet, ENet, SqueezeDet et d'autres réseaux vision, en rendant visibles les cellules non prises en charge selon la famille de FPGA. L'[Advanced CNN Accelerator](https://www.latticesemi.com/en/Products/DesignSoftwareAndIP/IntellectualProperty/IPCore/IPCores05/Advanced-CNN-Accelerator-IP) cible Avant-E/Avant-X/CertusPro-NX et constitue une IP commerciale. Cette table du compilateur ne garantit pas qu'un bitstream prenne en charge simultanément toutes les topologies listées.

### Microchip VectorBlox

Le [VectorBlox SDK 3.1](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) public de Microchip prétraite et compile des réseaux TFLite INT8 entièrement quantifiés pour l'IP CoreVectorBlox configurable des FPGA PolarFire SoC. `tflite_preprocess` et `vnnx_compile` produisent les artefacts de déploiement `.vnnx`, `.hex` et `.ucomp` ; le dépôt comprend un simulateur et un pilote C côté cible. Les graphes TensorFlow, ONNX et OpenVINO en amont doivent toujours passer par le parcours documenté de préparation TFLite/INT8.

La [matrice actuelle des tutoriels](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) inclut YOLOv5n, détection, classification, OBB, pose et segmentation YOLOv8n, YOLOv9t, des variantes YOLO creuses/compressées, MobileNet, EfficientNet-Lite0, ResNet18, MiDaS, FFNet et QuickSRNet. Le [guide du post-traitement C](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/C_Postprocessing.md) cite séparément YOLOv2/3/4/5 et les décodeurs de détection, pose et OBB Ultralytics. SDK 3.1 prend actuellement en charge PolarFire SoC Video Kit, et explicitement pas PolarFire Video Kit sans SoC : les anciennes démos de cartes ne doivent pas être confondues avec le contrat de cible actuel.

Le SDK est téléchargeable publiquement sous une [licence propre à Microchip](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/LICENSE.md) qui limite le logiciel et ses dérivés aux produits Microchip. Microchip propose des licences Libero Silver et CoreVectorBlox gratuites sur demande. Le déploiement reste un contrat système FPGA : configuration de l'accélérateur, bitstream, firmware, données réseau générées par le SDK et disposition mémoire doivent concorder. Une compilation VectorBlox réussie ne transforme pas le modèle en binaire copiable vers n'importe quelle conception PolarFire.

### Syntiant

Le NDP200 de Syntiant cible l'inférence toujours active de vision, audio et capteurs dans une enveloppe de puissance inférieure à celle des SoC Linux classiques. La [table matériel actuelle](https://www.syntiant.com/hardware) indique que le NDP200 est en production de masse et le NDP250 en échantillonnage. La [fiche produit NDP200](https://www.syntiant.com/ndp200/) documente la prise en charge de réseaux CNN, RNN et entièrement connectés sous 1 mW ; l'[annonce NDP250](https://www.syntiant.com/news/syntiant-unveils-ndp250-neural-decision-processor-with-next-gen-core-3-architecture) décrit plutôt la reconnaissance d'images toujours active sous 30 mW. Il ne faut pas généraliser la consommation « sous le milliwatt » aux deux puces.

Les ressources publiques ne prouvent pas de compatibilité étendue avec YOLO. Il vaut donc mieux évaluer ce produit pour de petits classificateurs et détecteurs toujours actifs, au moyen d'une étude de modèle prise en charge par le fabricant, plutôt que de supposer qu'un export ONNX générique fonctionnera.

### Google Coral : deux générations sans lien direct

Les produits Edge TPU historiques utilisent Edge TPU Compiler, `libedgetpu` et PyCoral avec des modèles TFLite entièrement INT8. Leurs principaux dépôts [Edge TPU](https://github.com/google-coral/edgetpu) et [PyCoral](https://github.com/google-coral/pycoral) sont archivés. Cela représente un véritable risque de maintenance, mais pas un avis officiel de fin de vie du matériel.

Google maintient également un projet [Coral NPU](https://github.com/google-coral/coralnpu) open source, distinct et actif : une IP d'accélérateur RISC-V destinée à être intégrée dans des SoC basse consommation. Le projet public fournit actuellement de la RTL, de la simulation matérielle, une chaîne d'outils et des exemples ELF. Il ne s'agit pas d'un successeur Edge TPU USB/M.2 prêt à l'emploi et il ne publie pas de matrice actuelle de déploiement YOLO.

## Pièges des GPU, NPU client et calcul adaptatif

NVIDIA, AMD, Intel et Apple apparaissent souvent dans les recherches sur les NPU, mais ne proposent pas une seule catégorie d'accélérateur interchangeable.

**NVIDIA Jetson :** Jetson Orin associe un GPU CUDA à des cœurs DLA dédiés. TensorRT peut construire un moteur sérialisé pour DLA, mais les couches non prises en charge s'exécutent sur le GPU uniquement si le repli GPU est explicitement activé. La [table YOLO DeepStream](https://github.com/NVIDIA-AI-IOT/deepstream_tools/blob/main/yolo_deepstream/README.md) actuelle couvre YOLOv4/v7/v8/v9/11, mais la plupart des entrées ciblent le GPU ; son entrée ONNX DLA explicite est YOLOv8s INT8. Les [restrictions DLA](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html) de NVIDIA détaillent les limites d'opérateurs. Thor est un autre cas : le [guide de migration DriveOS](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/NVIDIA_DriveOS_7.0.3_Migration_Guide.pdf) de NVIDIA indique que les cœurs DLA ont été retirés de Thor au profit de la flexibilité d'ordonnancement GPU. « Fonctionne sur Jetson » ne permet donc pas d'identifier l'accélérateur utilisé.

**AMD Vitis AI :** l'ancienne génération Kria/DPU compilait des graphes `.xmodel` exécutés via VART. La version actuelle [Vitis AI 6.2](https://vitisai.docs.amd.com/en/6.2/) est GA pour deux parcours Versal AI Edge distincts : [Gen1](https://vitisai.docs.amd.com/projects/gen1/en/latest/index.html), sur VEK280/VE2802, utilise ONNX, AMD Quark et un flux de snapshots/sous-graphes lié à la cible ; [Gen2](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html), sur VEK385, possède son propre contrat de compilation et runtime. Gen1 publie des exemples YOLOv5, YOLOv7, YOLOv8 et YOLOX. Ryzen AI constitue une quatrième pile NPU client distincte. Ne réutilisez pas les artefacts ou validations entre ancien DPU, Versal Gen1, Versal Gen2 et Ryzen AI au seul motif qu'ils portent le nom Vitis ou AMD.

**Intel OpenVINO :** OpenVINO peut cibler CPU, GPU et NPU Core Ultra via une API unique. La [documentation du plug-in NPU Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) identifie les générations NPU prises en charge, tandis que la [matrice des modèles vérifiés](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html) présente des colonnes propres aux appareils. Un notebook YOLO démontre une recette applicative, pas une validation NPU, sauf mention dans la matrice. Les opérations et formes prises en charge, la précision et le pilote NPU installé déterminent encore si le graphe complet peut s'y exécuter. L'IR OpenVINO (`.xml` plus `.bin`) est une représentation source réutilisable ; le cache du modèle compilé dépend de l'appareil et du logiciel.

**Apple Core ML :** `coremltools` convertit les modèles en `.mlpackage`, que Xcode compile en `.mlmodelc`. Core ML peut répartir le travail entre CPU, GPU et Apple Neural Engine, mais masque volontairement la répartition précise. Apple est ainsi une excellente plateforme de déploiement, mais il est difficile d'affirmer que « tout le graphe YOLO s'est exécuté sur le NPU » sans preuves issues du profiling.

## Les entreprises d'IP NPU derrière les fabricants de puces

Certaines entreprises ne vendent ni carte ni accélérateur conditionné. Elles concèdent des conceptions NPU sous licence aux fabricants de SoC. Elles comptent, car une même architecture sous-jacente peut apparaître sous plusieurs marques de puces, alors que le SDK final et l'artefact peuvent rester personnalisés par le titulaire de la licence.

| Fournisseur d'IP | Famille NPU et logiciel | Pourquoi c'est important |
|---|---|---|
| [Arm](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) | Ethos-U55/U65/U85 et Vela | Présent dans les produits embarqués de NXP, Alif, Infineon, Himax et d'autres ; flux TFLite/TOSA quantifié |
| [Arm China](https://www.armchina.com/mountain?infoId=161&name=) | AIPU Zhouyi et Compass SDK | Le [model zoo](https://github.com/Arm-China/Model_zoo) public cite YOLOv1-tiny/v2/v3/v4/v5, YOLOX et YOLOv8-seg ; ce sont des modèles de référence, pas des preuves pour toutes les puces des titulaires de licence |
| [VeriSilicon](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico) | Famille Vivante VIP et SDK Acuity | Des familles NPU Vivante apparentées figurent dans plusieurs SoC, notamment les parcours A311D et i.MX 8M Plus acquis séparément ; chaque fabricant de puce fournit toujours sa propre intégration |
| [Imagination Technologies](https://www.imaginationtech.com/products/open-access/) | PowerVR Series3NX AX3146/AX3386/AX3596 ; ancien IMG Series4 ; Neural Compute SDK/IMG DNN | IP NNA sous licence et non carte vendue au détail ; Open Access propose des configurations Series3NX de 1/5/10-TOPS éprouvées sur silicium, tandis qu'un programme officiel NC-SDK cite PP-YOLOE, EfficientNet et HRNet, mais pas toutes les configurations IP |
| [CEVA](https://www.ceva-ip.com/product/ceva-neupro-studio/) | NeuPro-Nano/NeuPro-M et NeuPro Studio | IP NPU sous licence avec import, quantification, compression, compilation de graphe, simulation et génération C/C++ |
| [MIPS](https://mips.com/processor-solutions/arc-npx-family/) | ARC NPX6 et [MetaWare MX](https://mips.com/processor-solutions/arc-metaware-mx/), anciennement Synopsys ARC | IP NPU évolutive et SDK NN acquis par GlobalFoundries puis intégré au portefeuille MIPS en [juin 2026](https://mips.com/press-releases/gfmipsarcclose/) ; ce n'est pas une cible de déploiement vendue au détail |
| [Cadence](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/neuroweave-sdk.html) | NPU Neo, DSP Tensilica et NeuroWeave SDK | Pile compilateur/interpréteur utilisée par les titulaires de licence SoC, avec la prise en charge des réseaux et de la quantification liée aux configurations concédées |
| [Quadric](https://quadric.ai/npu-ip) | IP et SDK GPNPU Chimera | IP programmable de type NPU/DSP concédée à d'autres entreprises ; la cible finale est le silicium du licencié |
| [Expedera](https://www.expedera.com/products-overview/) | IP NPU Origin et pile logicielle | Architecture NPU edge sous licence, pas une carte LibreYOLO achetable directement |

Imagination montre pourquoi les mentions d'accès et de cycle de vie doivent accompagner les noms d'architecture. Son [programme Open Access](https://www.imaginationtech.com/products/open-access/) propose trois configurations Series3NX éprouvées sur silicium sans frais de licence IP initiaux, après évaluation de l'entreprise, mais avec support/maintenance payants et redevances de production. Le [Neural Compute SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) fournit des outils de compilation, d'optimisation, de quantification et de runtime ; une [collaboration officielle avec Baidu PaddlePaddle](https://www.imaginationtech.com/news/imagination-and-baidu-paddlepaddle-create-open-source-machine-learning-library-for-model-zoo/) cite détection PP-YOLOE, classification EfficientNet et segmentation HRNet. Ces exemples ne valident pas toutes les configurations Series3NX ou Series4. De plus, les pages individuelles actuelles de [produits Series4](https://www.imaginationtech.com/product/img-4nx-mc1/) indiquent leur indisponibilité : considérez Series4 comme historique, sauf confirmation commerciale contraire.

Cette couche explique les ressemblances apparentes entre familles. Elle ne rend pas les artefacts portables. Deux SoC intégrant des IP Vivante ou Ethos proches peuvent avoir des cartes mémoire, pilotes, versions de compilateur, ensembles d'opérateurs et runtimes différents.

## Ce que vous déployez vraiment : le tableau de dépendance aux artefacts

C'est le dernier fichier de la chaîne de compilation qui met fin à la portabilité ONNX apparente. Les noms et extensions changent au fil du temps, mais la règle pratique reste la même : conservez le modèle d'origine et les données de calibration, car l'artefact natif ne peut généralement pas être déplacé vers une autre génération de NPU ni recompilé de façon fiable avec une autre version du SDK.

| Pile du fabricant | Entrée habituelle du compilateur | Fichier déployé sur l'appareil | Éléments auxquels il est lié |
|---|---|---|---|
| [Amlogic AMLNN](https://github.com/Amlogic-NN/amlnn-toolkit) | ONNX, TFLite, TorchScript ou PT2 | `.adla` ; A311D historique utilise `.nb` | ID de cible, génération ADLA, build du compilateur AMLNN, NNSDK2/runtime, pilote ADLA et BSP |
| [Rockchip RKNN](https://github.com/airockchip/rknn-toolkit2) | ONNX, TFLite ou export de framework | `.rknn` | Version de RKNN Toolkit/runtime et famille de NPU Rockchip |
| [Hailo](https://hailo.ai/developer-zone/documentation/) | ONNX ou TensorFlow via un HAR intermédiaire | `.hef` | Architecture Hailo et génération du compilateur/runtime |
| [Axelera Voyager](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) | ONNX ou définition du zoo prise en charge | `.axm` dans Pipeline Builder alpha, paquet pipeline `.axe` ; `.axmodel` classique avec manifeste | Génération de l'API Voyager et configuration cible Metis |
| [DEEPX DXNN](https://github.com/DEEPX-AI/dx-all-suite) | ONNX et parcours de framework pris en charge | `.dxnn` | Version DX-COM/DX-RT et cible DX-M |
| [Qualcomm QAIRT/QNN ou SNPE](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | ONNX, TFLite ou modèle de framework | Bibliothèque/modèle QNN ou binaire de contexte, ou SNPE `.dlc` | Architecture HTP, SoC, backend et version runtime |
| [MediaTek NeuroPilot](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) | TFLite converti/quantifié | `.dla` pour Neuron Runtime hors ligne ; `.tflite` pour le mode délégué | Génération NP/MDLA, image du système d'exploitation, compilateur et runtime |
| [TI TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools) | ONNX ou TFLite | Répertoire d'artefacts importés et fichiers de modèle applicatif | TIDL Tools/runtime et génération du processeur |
| [NXP eIQ](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | Généralement TFLite ou ONNX | Sortie Vela, TIM-VX, Neutron ou Ara propre au backend | Accélérateur i.MX/Ara choisi et BSP, pas « NXP » en général |
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | TFLite, ONNX ou modèle de framework pris en charge | Bibliothèque/code réseau généré et ressources firmware | Cible MCU/NPU, configuration mémoire et version de l'outil |
| [Infineon PSOC Edge](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) | Parcours TFLite/Keras/PyTorch quantifié en entier | TFLite optimisé par Vela avec flux de commandes Ethos-U et firmware applicatif | Variante E83/E84, configuration Ethos-U, Vela, ModusToolbox et BSP ; NNLite est une cible séparée |
| [Renesas DRP-AI](https://github.com/renesas-rz/rzv_drp-ai_tvm) | ONNX/TFLite via TVM ou Translator | Répertoire de modèles runtime contenant plusieurs fichiers binaires | Appareil RZ/V et génération du traducteur/runtime |
| [Sony IMX500](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | Réseau quantifié/packagé via Edge-MDT | Paquet `.rpk` | Firmware du capteur, budget mémoire et version du packer |
| [Synaptics](https://developer.synaptics.com/) | TFLite/ONNX et imports pris en charge | `.synap` avec SyNAP ; `.vmfb` avec Torq | Génération logicielle/matérielle SL16xx ou SL261x |
| [Ambarella CVflow](https://www.ambarella.com/developer/) | Entrée de la chaîne d'outils partenaire | Paquet de déploiement partenaire ; contrat d'artefact exact non documenté publiquement | Confirmer la génération CVflow, le SDK/BSP et le pipeline caméra via Cooper Developer Zone |
| [D-Robotics BPU](https://github.com/D-Robotics/rdk_model_zoo) | ONNX et graphe accepté par la chaîne d'outils | `.bin` pour X5 ; `.hbm` pour le `rdk_s` actuel ; flux plateforme plus ancien pour X3 | Génération BPU, branche du dépôt et version OpenExplorer/runtime correspondante |
| [AXERA Pulsar2](https://github.com/AXERA-TECH/ax-samples) | ONNX | `.axmodel` | Cible AX, version Pulsar2 et AXEngine |
| [SOPHGO TPU-MLIR](https://github.com/sophgo/tpu-mlir) | ONNX, TFLite, TorchScript et autres | `.bmodel` sur BM, `.cvimodel` sur CV18xx | Cible de puce BM/CV et génération runtime SOPHON |
| [Huawei Ascend CANN](https://www.hiascend.com/en/software/cann) | ONNX ou graphe de framework pris en charge | `.om` | Puce Ascend, CANN/ATC et firmware/pilote de l'appareil |
| [Cambricon MagicMind](https://github.com/Cambricon/magicmind_cloud) | Graphe de framework ou modèle d'échange | Moteur/modèle MagicMind sérialisé | Architecture MLU et version Neuware/MagicMind |
| [Canaan nncase](https://github.com/kendryte/nncase) | ONNX ou TFLite | `.kmodel` | Génération KPU et plug-in cible nncase correspondant |
| [Mobilint qb](https://www.mobilint.com/sdk-qb) | ONNX et modèles de framework pris en charge | `.mxq` | Cible REGULUS/ARIES et compilateur/runtime qb |
| [Rebellions RBLN](https://docs.rbln.ai/latest/index.html) | PyTorch 2 ou graphe TensorFlow pris en charge | `.rbln` | Génération ATOM et compilateur/runtime RBLN |
| [FuriosaAI](https://developer.furiosa.ai/docs/latest/en/) | ONNX/TFLite et parcours de framework pris en charge | Warboy `.enf` ; RNGD `.fxb` | Générations d'accélérateur et de SDK entièrement distinctes |
| [Sunplus SNNF](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2004353133) | Réseau accepté par Acuity | `.nb` | Pilote NPU SP7350, runtime et BSP |
| [ESWIN ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) | ONNX dans le flux EsAAC actuel documenté ; les autres frameworks nécessitent export/conversion | `.model` | Cible précise de la famille EIC7700 et version ENNP/ESSDK |
| [Nuvoton / Ethos-U](https://github.com/OpenNuvoton/NuEdgeWise) | TFLite quantifié | TFLite optimisé par Vela avec opérations personnalisées Ethos-U | Plan mémoire M55M1, Vela et firmware |
| [Himax WiseEye2](https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2) | TFLite entièrement quantifié en entier | `.tflite` optimisé Vela dans la mémoire modèle, plus firmware de carte tel que `output.img` | Configuration HX6538/WE2, Vela, TFLite Micro, pilote Ethos-U et noyaux de repli |
| [Analog Devices AI8X](https://github.com/analogdevicesinc/ai8x-synthesis) | Checkpoint PyTorch, YAML réseau et entrée d'exemple | Code C/en-têtes propres à l'appareil, poids et firmware compilé générés | Limites mémoire/couches MAX78000 et MAX78002, outil de synthèse et version MSDK |
| [Realtek AmebaPro2](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Entrée/service de conversion du fabricant | `.nb` | Firmware RTL8735B et version API VoE/NeuralNetwork |
| [Telechips Enlight](https://docs.topst.ai/product/p/ai) | Parcours Darknet, TensorFlow, ONNX ou PyTorch | `.enlight` intermédiaire et bundle de déploiement compilé | NPU TCC7500 et versions TC-NN/Enlight |
| [T-Head HHB](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | ONNX et graphe de framework pris en charge | `hhb.bm`, paramètres/code générés et exécutable | Pile CSI-NN2/SHL TH1520 |
| [SigmaStar MI_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | Les documents SSU9383CM actuels exposent le runtime MI_IPU ; les anciens contenus utilisent des entrées TensorFlow/Caffe | Ancien `.sim` converti en `sgsimg.img` ; artefact public actuel non vérifié | Génération IPU et SDK SigmaStar précis ; compatibilité du compilateur ancien avec SSU9383CM non vérifiée |
| [HiSilicon smart vision](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) | ONNX et graphe accepté par ATC | `.om` | SoC de vision intelligente et pile NNN/SVP-NNN exacts ; pas une portabilité Ascend générique |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | ONNX, TFLite, Keras ou TensorFlow | Paquet dataflow `.dfp` | Configuration matérielle MX et version runtime/compilateur |
| [Kneron](https://doc.kneron.com/docs/) | ONNX et configuration de la chaîne d'outils | `.nef` ; NEFv2 KL730 pouvant contenir `.kne` | Cible compilateur documentée, génération de puce, firmware et runtime PLUS ; la compilation KL830 n'est pas établie par Toolchain 0.33.1 |
| [SiMa.ai Palette](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) | Framework ou modèle ONNX pris en charge | `.tar.gz` compilé par ModelSDK, ELF exécutable documenté ou paquet `.mpk` de MPK Tool | Cible MLSoC/Modalix et version Palette |
| [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | ONNX ou définition de réseau | Moteur sérialisé, souvent `.engine` ou `.plan` | Architecture GPU/DLA, TensorRT, CUDA et souvent appareil |
| [AMD Vitis AI](https://vitisai.docs.amd.com/en/6.2/) | Graphe ONNX/framework quantifié | Ancien `.xmodel` ; snapshots/sous-graphes NPU Gen1 ; répertoire de cache compilé ou paquet de production `.rai` Gen2 | Génération NPU/IP exacte, image plateforme et matrice Vitis/Vivado/PetaLinux |
| [Microchip VectorBlox](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) | TFLite INT8 entièrement quantifié | `.vnnx`, `.hex` et `.ucomp` avec firmware/conception FPGA correspondants | Configuration CoreVectorBlox, PolarFire SoC Video Kit, SDK 3.1, bitstream et disposition mémoire |
| [Intel OpenVINO](https://docs.openvino.ai/) | Modèle de framework ou ONNX | IR `.xml` plus `.bin`, ou cache compilé propre à l'appareil | IR réutilisable ; cache compilé spécifique à l'appareil, au pilote et à OpenVINO |
| [Google Edge TPU historique](https://coral.ai/docs/edgetpu/models-intro/) | TFLite entièrement INT8 | `_edgetpu.tflite` compilé pour Edge TPU | Compilateur/runtime Edge TPU et opérateurs quantifiés pris en charge ; sans lien avec la nouvelle IP Coral NPU |
| [Google Coral NPU IP](https://github.com/google-coral/coralnpu) | Exemples C/C++ du projet IP public actuel | Exemples ELF pour simulation RTL/matérielle et future intégration SoC ; aucun artefact compilateur YOLO public | Implémentation SoC exacte, intégrée/sous licence, et chaîne open source en évolution |
| [Lattice sensAI](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | Description du réseau pris en charge | Bitstream FPGA, configuration de l'accélérateur et poids | Famille FPGA, mode IP sensAI et implémentation mémoire |
| [Imagination NC-SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) | Caffe, TensorFlow ou ONNX | Nom de l'artefact compilé destiné au client non précisé publiquement | Configuration NNA sous licence, intégration SoC, NC-SDK/DDK et runtime du licencié |

Ce tableau n'est pas un simple aide-mémoire des extensions. La version du compilateur, l'identifiant de cible, le pilote, le firmware et le BSP font partie de l'identité reproductible du modèle. Un futur exporteur LibreYOLO devrait les inscrire dans un manifeste lisible par machine à côté de chaque artefact.

## Accès aux outils et signaux de cycle de vie

La disponibilité du matériel et l'accès au compilateur sont deux choses distinctes. Une carte peut être facile à acheter alors que son compilateur actuel nécessite un accord client ; un SDK public peut viser un silicium difficile à trouver. Les éléments ci-dessous sont des signaux concrets issus des fabricants, pas un classement universel de longévité.

| Plateforme | Signal documenté | Conséquence pratique |
|---|---|---|
| Amlogic ADLA | Dépôts Apache-2.0 publics et wheels téléchargeables ; les pilotes/BSP des cartes compatibles peuvent encore nécessiter Amlogic ou le fabricant de la carte | Évaluation facile du compilateur, mais confirmer les droits de redistribution des binaires et la matrice complète de compatibilité |
| MediaTek Genio | IoT AI Hub et guides Yocto publics ; outils NP8 tout-en-un et documentation Android indiqués comme réservés aux clients directs/sous NDA | Les preuves pour les modèles sont vérifiables ; l'automatisation avec ses propres modèles peut nécessiter un partenariat |
| Hailo | Model zoo et runtime publics ; Dataflow Compiler distribué via Developer Zone ; le dépôt public `hailo-camera-apps` a été archivé le 3 mai 2026 | La découverte des modèles est largement ouverte, mais une compilation reproductible nécessite le bon compte/la bonne version ; l'archivage ne prouve pas la fin de vie Hailo-15 et il faut confirmer son paquet applicatif de processeur vision actuel |
| Axelera AI | Documentation publique et [SDK Voyager public](https://github.com/axelera-ai-hub/voyager-sdk) ; support client sur compte | Rare SDK d'accélérateur en libre-service, tandis que le support produit et l'achat restent commerciaux |
| Qualcomm Dragonwing | Le [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) indique IQ-9075 Sampling avec une longévité jusqu'en 2038 et QCS6490 jusqu'en juillet 2036, alors que la page IQ-9075 indique Active | Signal solide pour planifier des produits industriels, mais le conflit entre sources du fabricant nécessite une confirmation par SKU et ne garantit pas la stabilité ABI d'un SDK IA durant toute la période |
| Rebellions | La [matrice actuelle](https://docs.rbln.ai/latest/supports/version_matrix.html) indique CA02 et CA12 en fin de vie, CA22/CA25 actifs ; CA21 est absent | L'achat et la compatibilité du SDK dépendent de la carte, même au sein de la famille ATOM |
| NVIDIA Jetson | La [table officielle de cycle de vie des modules](https://developer.nvidia.com/embedded/lifecycle) indique les modules Orin jusqu'en janvier 2032 et AGX Orin Industrial jusqu'en juillet 2033 ; les kits développeur n'ont aucun engagement de cycle de vie | Concevoir les systèmes de production autour des modules, pas de la disponibilité des kits de développement |
| Sony IMX500 sur Raspberry Pi | La [fiche produit AI Camera](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf) indique une production jusqu'en janvier 2028 au minimum | Minimum concret pour ce module caméra, pas pour tous les produits IMX500 |
| Amlogic A311Y3 | La [page de lancement 2026](https://www.amlogic.com/News/index248.html) annonce une garantie de longévité produit d'au moins dix ans | Signal prometteur pour cette nouvelle plateforme, qui nécessite encore les détails contractuels et de SKU pour un programme produit |
| Google Coral | Les dépôts Edge TPU et PyCoral historiques sont archivés/en lecture seule ; le dépôt distinct [Coral NPU IP](https://github.com/google-coral/coralnpu) est actif | Risque de maintenance des anciens logiciels, mais pas en soi un avis officiel de fin de vie du matériel ; cela ne renseigne pas sur le cycle de vie du projet IP open source sans rapport direct |

## Pourquoi les TOPS ne suffisent pas à choisir

Les TOPS de pointe peuvent être utiles au sein d'un même fabricant et d'une même architecture, mais les comparaisons entre fabricants sont généralement invalides à moins que les éléments suivants concordent :

- Précision numérique, INT8, INT4, FP16 ou mode mixte compris
- Convention de comptage d'une multiplication-accumulation, comme une ou deux opérations
- Calcul dense ou creux structuré
- Résolution d'entrée, taille de batch et graphe du modèle
- Exactitude après quantification
- Latence NPU uniquement ou pipeline applicatif complet
- Transferts mémoire et repli sur hôte
- État thermique et fonctionnement soutenu, et non en rafale

Un détecteur fait partie d'un pipeline : acquisition de l'image, redimensionnement/letterbox, normalisation, transfert vers l'appareil, inférence neuronale, décodage, NMS, mise à l'échelle des coordonnées et logique applicative. Les fabricants ne publient souvent que la partie centrale, l'inférence neuronale. [MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/) est utile car il définit des scénarios, des exigences de qualité et des règles de mesure système au lieu de comparer isolément des chiffres marketing.

Pour les déploiements YOLO, un compte rendu de benchmark crédible devrait au minimum contenir :

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

Sans ces champs, deux chiffres FPS mesurent généralement des choses différentes.

## Comment choisir un NPU pour la vision par ordinateur

Procédez dans cet ordre, plutôt que de vous fier au plus grand chiffre imprimé sur une page produit.

1. **Prouvez la compatibilité du graphe.** Compilez le modèle exporté exact, pas un checkpoint de zoo portant un nom similaire.
2. **Mesurez l'exactitude.** Validez l'artefact compilé sur le vrai dataset de la tâche après calibration et quantification.
3. **Mesurez le pipeline de bout en bout.** Incluez conversion d'entrée, transferts, décodage et NMS.
4. **Vérifiez l'accès au logiciel.** Déterminez si le compilateur est public, soumis à inscription ou disponible seulement après un accord commercial.
5. **Figez la matrice des versions.** Consignez le compilateur, le runtime, le pilote, le firmware et les identifiants de cible.
6. **Vérifiez la réalité des systèmes d'exploitation.** Android, Debian, Yocto, Buildroot, RTOS et Windows ne sont pas interchangeables.
7. **Vérifiez l'approvisionnement et le cycle de vie.** Une excellente puce est un mauvais choix de production si les modules, pilotes ou le support à long terme sont incertains.
8. **Comparez ensuite coût, puissance et performances.** Ces mesures ne deviennent pertinentes qu'une fois que le même modèle fonctionne correctement sur les deux cibles.

### Premières pistes pratiques selon le cas d'usage

| Cas d'usage | Plateformes à évaluer en premier | Raison |
|---|---|---|
| Accélérateur spécifique Raspberry Pi | Hailo-8L/8 et Sony IMX500 | Produits Raspberry Pi officiels, images logicielles et parcours développeur concrets |
| Extension Linux générale PCIe, M.2 ou USB | DEEPX, MemryX, Hailo et produits Axelera | Formats d'accélérateurs disponibles, sous réserve de compatibilité hôte/pilote |
| SBC Linux ou caméra à bas coût | Rockchip RK3588/RK3576, Amlogic ADLA si une carte/BSP pris en charge est disponible, AXERA, Canaan K230 ou Sunplus SP7350 | Pipelines média intégrés et preuves publiques des modèles/compilateurs ; vérifier VIM4 V13A+ pour le NPU A311D2 |
| Mobile et Android à gros volumes | Qualcomm QNN/AI Hub, LiteRT, Apple Core ML | Base installée importante et runtimes mobiles maintenus |
| Linux industriel et robotique | TI TDA4/AM6xA, NXP i.MX, Renesas RZ/V, Hailo | Produits embarqués longue durée et écosystèmes caméra/E/S |
| Endpoint ou MCU très compact | STM32N6x7, Infineon PSOC Edge, Himax WiseEye2, Analog Devices MAX7800x, Nuvoton M55M1, Alif/Arm Ethos-U, GAP9 et Syntiant | Enveloppes de puissance et mémoire réduites, outils dédiés et limites de taille de modèle |
| Vision FPGA configurable | Microchip VectorBlox, Lattice sensAI et cibles AMD Vitis AI | Pipelines matériels flexibles, mais la configuration de l'accélérateur, le bitstream et le compilateur de modèle forment un système versionné unique |
| Vision PCIe à haut débit | Axelera Metis, Hailo, DEEPX, Mobilint, MemryX, Rebellions ou SiMa.ai | Produits d'accélération dédiés et positionnement multi-flux |
| Écosystème de produits centré sur la Chine | Rockchip, AXERA, D-Robotics, SOPHGO, Huawei Ascend, HiSilicon et Cambricon | Cartes, chaînes d'outils et exemples de modèles régionaux solides |

Ce tableau est une liste de pistes issue de la recherche, pas un classement de performance. L'achat, la disponibilité régionale et le modèle exact peuvent inverser l'ordre.

## Conséquences pour LibreYOLO

Une architecture évolutive ne consiste pas à créer des dizaines d'exporteurs sans lien. Il faut un contrat d'échange strict, accompagné de petits adaptateurs de compilation et de runtime pour chaque fabricant :

```text
LibreYOLO model
    -> deterministic static ONNX
    -> VendorCompiler.compile(model, target, calibration, precision)
    -> native artifact + manifest
    -> VendorRuntime.load() / infer()
    -> shared task-specific decode and Results objects
```

Chaque artefact natif devrait avoir un manifeste associé contenant :

- Fabricant, cible puce et cible carte
- Versions du compilateur, runtime, pilote et firmware
- Sommes de contrôle du checkpoint source et ONNX
- Noms d'entrée, formes, disposition, espace colorimétrique, normalisation et type de données
- Précision de quantification, échelles si disponibles et empreinte des données de calibration
- Noms et formes des tenseurs de sortie et leur signification
- Tâche de détection, noms de classes et indication si le décodage/NMS est sur la puce ou côté hôte
- Parité numérique et résultats d'exactitude de la tâche enregistrés

LibreYOLO fournit déjà un [export ONNX portable](/docs/export/onnx), des [outils de quantification](/docs/export/quantization), un [export RKNN](/docs/export/rknn) direct mais volontairement limité, et un [parcours Hailo](/docs/export/hailo) honnêtement fondé sur un compilateur externe. Selon les critères de ce guide, Amlogic ADLA est un excellent prochain candidat d'intégration : le toolkit est public, la couverture des modèles recoupe fortement celle de LibreYOLO et l'ancien parcours A311D peut rester distinct.

Après Amlogic, il faut établir les priorités selon le matériel des utilisateurs et la possibilité de valider l'exactitude, et pas seulement selon la disponibilité du compilateur. DEEPX, Sony IMX500, D-Robotics, AXERA, SOPHGO et Qualcomm sont des options techniquement intéressantes ; TI, NXP, ST et Renesas deviennent particulièrement utiles si des partenaires industriels peuvent fournir cartes et accès continu à la CI.

## À surveiller : silicium réel sans contrat d'intégration public

Exclure complètement une entreprise peut fausser la cartographie du marché. La déclarer « prise en charge » peut être encore pire. Ces fabricants vendent, échantillonnent ou ont annoncé des puces pertinentes, mais les preuves publiques n'établissent pas encore un parcours reproductible actuel avec compilateur, artefact et modèle nommé adapté à un backend LibreYOLO.

| Entreprise | Éléments réels | Pourquoi la garder dans la liste de surveillance |
|---|---|---|
| [Samsung](https://semiconductor.samsung.com/processor/automotive-processor/exynos-auto-v920/) | Exynos Auto V920 comprend un NPU bicœur annoncé jusqu'à 23.1 TOPS | Samsung indique que son [Neural SDK n'est plus fourni aux développeurs tiers](https://developer.samsung.com/neural/overview.html) ; Samsung ONE/Circle ne prouve pas l'accès au NPU Exynos |
| [Novatek](https://www.novatek.com.tw/en-global/Milestone/aboutus_milestones) | Les étapes actuelles de l'entreprise citent les SoC Edge AI et Edge Vision/Imaging AI ; une étape de 2020 mentionnait MobileNet, SSD et YOLOv3 | Aucune matrice publique actuelle de références/compilateur/artefact/opérateurs/modèles ; YOLOv3 historique ne prouve pas la prise en charge par le SDK actuel |
| [Nextchip](https://www.nextchip.com/en/adas/adas.php?idx=5) | Processeurs automobiles APACHE5/NVS2900 et [APACHE6/NVS3000](https://www.nextchip.com/en/adas/adas.php?idx=7) actuels avec NPU aiMotive aiWare ; la page APACHE6 indique actuellement à la fois 12 et 8 TOPS | aiWare Studio et un runtime existent, mais aucun artefact public, guide de quantification, liste d'opérateurs ou matrice YOLO n'a été trouvé ; les chiffres contradictoires du fabricant ne doivent pas être résolus arbitrairement |
| [Black Sesame Technologies](https://bst.ai/en.html) | Silicium automobile/Edge AI Huashan A1000/A2000 et série Wudang C | Les informations produit sont publiques, mais pas de compilateur en libre-service, contrat d'artefact ou model zoo reproductible |
| [Ingenic](https://en.ingenic.com.cn/products-detail/id-19.html) | SoC caméra T41/T40 avec annonces de NPU basse précision ; Magik AI décrit PTQ/QAT et la compilation de graphe | Aucun compilateur actuel téléchargeable, contrat d'artefact ou liste YOLO exacte du fabricant n'a été trouvé |
| [Fullhan](https://fullhan.com/en/index.php?a=type&c=article&tid=9) | Plusieurs SoC IPC annoncent des NPU de 0.5 à 2 TOPS ; la [famille NVR MC6880](https://fullhan.com/en/index.php?a=type&c=article&tid=48) annonce 4 TOPS | Aucune documentation publique du compilateur/runtime/framework/modèles NN suffisante pour une reproduction indépendante |
| [Goke Microelectronics](http://www.gokemicro.com/News/info.aspx?itemid=437) | Une page du fabricant pour les composants caméra intelligente GK7606V1/GK7206V1/GK7203V1 annonce les performances du NPU intégré, mais n'est servie qu'en HTTP historique au moment de la publication | Pas de convertisseur public, contrat runtime ou matrice de modèles nommés ; le canal est destiné aux OEM |
| [Chengheng Micro](https://en.chenghengmicro.com/) | CH37 annonce 64 TOPS INT8 et les modes FP16/FP32/FP64 ; la [chronologie 2026](https://chenghengmicro.com/about.html) indique le démarrage d'une production en petite série | Aucun SDK public, contrat compilateur ou modèle nommé validé trouvé ; considérer cette plateforme comme en phase de début de commercialisation et non libre-service, plutôt que comme preuve de déploiement reproductible |
| [Bouffalo Lab](https://github.com/bouffalolab/bouffalo_sdk) | BL808 intègre le NPU BLAI-100 et des dépôts SDK officiels actuels existent | Le SDK maintenu n'expose pas de parcours officiel actuel de conversion/exemple BLAI ; les flux existants relèvent de l'historique ou de la communauté |

Cette liste de surveillance repose volontairement sur les preuves. Un fabricant peut rejoindre les tableaux de déploiement en publiant une chaîne d'outils ou un parcours d'évaluation à jour, un contrat d'artefact propre à la cible et au moins un modèle nommé avec une recette de bout en bout.

## Ce que cet article n'affirme volontairement pas

Cet article ne prétend pas que chaque modèle cité fonctionne à partir de n'importe quel dépôt amont. Les modèles des vendor zoos sont souvent modifiés, découpés à différents nœuds de sortie ou associés à un post-traitement personnalisé.

Il ne transforme pas non plus les affirmations suivantes en déclarations de prise en charge :

- Un compilateur importe ONNX.
- Une puce annonce un pilote Android NNAPI.
- Un distributeur affirme que la carte est « compatible YOLO ».
- Un dépôt communautaire exécute une variante d'un modèle.
- Un modèle compile sans erreur.
- Un fabricant annonce des FPS NPU uniquement sans résultat d'exactitude.

Les accélérateurs de téléphone Google Tensor et de nombreux ASIC personnalisés réservés aux OEM existent aussi, mais Google ne propose pas Tensor comme cible de compilateur générale en libre-service comparable aux plateformes ci-dessus. L'existence d'une entreprise, un schéma bloc de NPU ou l'accélération Android ne suffit pas pour inventer une affirmation d'intégration LibreYOLO.

De même, les accélérateurs cloud tels que Google TPU, AWS Inferentia/Trainium, Microsoft Maia et les cartes IA exclusivement destinées aux centres de données sont hors périmètre. Ce guide porte sur le matériel qu'un développeur de vision par ordinateur peut raisonnablement déployer à la périphérie.

La licence du modèle est un sujet distinct. La publication par un fabricant de puce d'une recette de conversion YOLO n'accorde aucun droit d'utiliser ou redistribuer les poids amont, le code d'entraînement ou un dérivé compilé. Il faut vérifier séparément la prise en charge matérielle, la licence du SDK et la licence du modèle pour le produit visé.

## Méthodologie de recherche et corrections

Pour chaque entreprise, la recherche s'est appuyée sur quatre types de sources principales :

1. Une page produit ou une fiche technique actuelle identifiant le silicium.
2. La documentation du compilateur et du runtime expliquant le vrai parcours de déploiement.
3. Un model zoo, une table de compatibilité ou un tutoriel de bout en bout citant des modèles de vision.
4. Un signal de cycle de vie ou d'accès indiquant si les développeurs peuvent réellement obtenir les outils.

Les organisations GitHub officielles comptent comme sources du fabricant. La documentation d'un fabricant de cartes est signalée comme preuve au niveau de l'écosystème lorsque l'entreprise qui fabrique la puce ne publie pas les mêmes ressources. Les affirmations de performance tierces ont été exclues des tableaux comparatifs.

Le brouillon complet a ensuite été réaudité lors de passages distincts par groupes de fabricants et à travers les tableaux. Ces vérifications ont réévalué les périmètres de cible, les noms d'artefacts, la précision, les preuves concernant modèles et postprocesseurs, les différences entre annonces et produits commercialisés, les libellés de cycle de vie, les dénominateurs de benchmarks et tous les comptes d'entités. Lorsque deux pages du fabricant se contredisent, ce guide expose le conflit au lieu de choisir discrètement la valeur la plus commode.

Ce marché évolue rapidement. Si vous travaillez pour l'une de ces entreprises et qu'une puce, un SDK, une liste de modèles ou un statut d'accès est inexact, envoyez à LibreYOLO l'URL exacte de la documentation publique et sa version. Les corrections étayées par des sources primaires remplaceront ce texte ; les affirmations marketing non documentées ne le feront pas.

## FAQ

### Combien existe-t-il d'entreprises Edge AI avec NPU ?

Il n'existe pas de nombre universel, car les fabricants de produits, les fournisseurs de licences IP, les capteurs intelligents, les GPU et les ASIC automobiles privés se recoupent. Ce guide non exhaustif suit 69 entreprises et écosystèmes de plateformes proposant des puces Edge AI, des plateformes d'accélération, des IP NPU sous licence ou des puces pertinentes à surveiller en août 2026.

### Qu'est-ce qu'un NPU ?

Une unité de traitement neuronal est un matériel spécialisé dans les opérations des réseaux de neurones, comme les convolutions et les multiplications matricielles. Les fabricants emploient notamment les termes NPU, AIPU, BPU, KPU, DLA, HTP, TPU et MLA. Leurs modèles compilés ne sont généralement pas portables d'un fabricant à l'autre.

### Quelles entreprises NPU prennent officiellement en charge les modèles YOLO ?

Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 via le dépôt officiel AI Camera de Raspberry Pi, STMicroelectronics, Renesas, Lattice, Himax, Microchip et plusieurs autres publient des entrées de model zoo, des tutoriels ou des résultats validés par leurs équipes pour au moins une génération de YOLO. La génération, la tâche, la puce ciblée et la version du SDK restent déterminantes.

### Un modèle ONNX quelconque peut-il s'exécuter sur n'importe quel NPU ?

Non. ONNX est un format d'échange, pas une garantie de compatibilité matérielle. Le compilateur NPU doit prendre en charge chaque opérateur, forme de tenseur et type de données du graphe. Les formes statiques, les découpages de graphe, les opérations personnalisées et le repli sur le CPU sont fréquents.

### Quel est le meilleur NPU pour YOLO ?

Il n'existe pas de vainqueur universel. Hailo, Rockchip, Axelera AI, DEEPX et Amlogic disposent d'une couverture YOLO bien documentée, tandis que Qualcomm bénéficie d'une présence exceptionnelle dans les appareils. Choisissez selon l'exactitude après quantification, la latence de bout en bout, la puissance soutenue, le prix, la disponibilité, l'accès au SDK et la prise en charge du système d'exploitation.

### Pourquoi ne peut-on pas comparer directement les valeurs TOPS des NPU ?

Les fabricants peuvent compter différemment les précisions, les opérations creuses, les multiplications-accumulations et les hypothèses d'utilisation de pointe. Les TOPS ne tiennent pas compte du trafic mémoire, des opérateurs non pris en charge, du prétraitement, du post-traitement ni de la surcharge de l'hôte. Comparez le même modèle, la même précision, la même résolution, la même exactitude et le pipeline complet.

### Qu'est-ce que la calibration d'un NPU ?

La calibration fait passer des images représentatives du déploiement, généralement non annotées, dans le modèle afin que le compilateur estime les plages d'activation pour INT8 ou un autre format de faible précision. Des images aléatoires ou non représentatives peuvent produire un artefact compilé tout en dégradant l'exactitude sur la tâche.

### LibreYOLO prend-il en charge les NPU edge ?

LibreYOLO exporte vers ONNX et plusieurs formats de runtime, propose un chemin direct de compilation RKNN pour certains modèles Rockchip et documente le flux de compilation Hailo externe. Tout autre artefact natif d'un fabricant exige encore son SDK et doit être présenté comme un candidat d'intégration tant qu'il n'a pas été compilé, validé et exécuté sur le matériel.