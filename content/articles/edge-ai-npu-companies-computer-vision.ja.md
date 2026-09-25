---
title: "Edge AI NPU企業69社を比較：チップ、SDK、YOLO対応（2026年）"
description: "Edge AI企業とNPUエコシステム69社を一次資料に基づいて解説します。チップ、SDK、成果物、量子化、コンピュータビジョンおよびYOLOの公開済み対応状況をまとめました。"
date: 2026-08-15
author: Xuban
tags: [edge-ai, npu, computer-vision, yolo, hardware, ai-accelerators, amlogic, hailo, rockchip]
faq:
  - q: "Edge AIのNPU企業は何社ありますか？"
    a: "製品ベンダー、IPライセンサー、スマートセンサー、GPU、非公開の車載ASICが重なり合うため、普遍的な企業数はありません。このガイドはすべてを網羅するものではありませんが、2026年8月時点で、関連するEdge AIチップ、アクセラレータープラットフォーム、ライセンス提供されるNPU IP、または有望なウォッチリスト対象シリコンを持つ企業とプラットフォームエコシステムを69件追跡しています。"
  - q: "NPUとは何ですか？"
    a: "NPU（ニューラル処理ユニット）は、畳み込みや行列乗算など、ニューラルネットワークの演算に特化したハードウェアです。ベンダーによってNPU、AIPU、BPU、KPU、DLA、HTP、TPU、MLAなどの名称が使われ、コンパイル済みモデルは一般に企業間で移植できません。"
  - q: "YOLOモデルを公式にサポートするNPU企業はどこですか？"
    a: "Amlogic、Hailo、Axelera AI、Rockchip、Qualcomm、MediaTek、DEEPX、D-Robotics、AXERA、SOPHGO、Huawei、Cambricon、Raspberry Piの公式AI Cameraリポジトリ経由のSony IMX500、STMicroelectronics、Renesas、Lattice、Himax、Microchipなどは、少なくとも1世代のYOLOについて、ファーストパーティのモデルズー項目、チュートリアル、または検証済み結果を公開しています。対応する世代、タスク、チップの対象、SDKバージョンを個別に確認する必要があります。"
  - q: "どのONNXモデルでも、どのNPUでも実行できますか？"
    a: "いいえ。ONNXは交換形式であり、ハードウェア互換性を保証するものではありません。NPUコンパイラーは、グラフ内のすべての演算子、テンソル形状、データ型をサポートする必要があります。静的形状、グラフ分割、カスタム演算、CPUフォールバックはよくあります。"
  - q: "YOLOに最適なNPUはどれですか？"
    a: "万能な勝者はいません。Hailo、Rockchip、Axelera AI、DEEPX、Amlogicは、公開資料で確認できるYOLO対応が充実しており、Qualcommは特に幅広いデバイスに展開されています。量子化後の精度、パイプライン全体のレイテンシ、持続消費電力、価格、入手性、SDKへのアクセス、OS対応で選んでください。"
  - q: "NPUのTOPS値を直接比較できないのはなぜですか？"
    a: "ベンダーによって、数値精度、スパース演算、積和演算、ピーク稼働率の前提が異なります。TOPSにはメモリ転送、未対応演算子、前処理、後処理、ホスト側のオーバーヘッドが含まれません。同じモデル、数値精度、解像度、タスク精度、パイプライン全体で比較してください。"
  - q: "NPUのキャリブレーションとは何ですか？"
    a: "キャリブレーションでは、代表的なデプロイ画像（通常はラベルなし）をモデルに通し、コンパイラーがINT8などの低精度形式向けにアクティベーション範囲を推定します。ランダムな画像や代表性のない画像でもコンパイル成果物が生成されることはありますが、タスク精度を損なう可能性があります。"
  - q: "LibreYOLOはエッジNPUをサポートしていますか？"
    a: "LibreYOLOはONNXや複数のランタイム形式にエクスポートでき、一部のRockchipモデル向けに直接RKNNコンパイラーパスを備え、外部Hailoコンパイル手順も説明しています。それ以外のベンダー固有成果物にはベンダーSDKが必要です。ハードウェア上でコンパイル、検証、実行するまでは、統合候補として説明してください。"
---

**単一のNPU市場はありません。このガイドはすべてを網羅するものではありませんが、企業とプラットフォームエコシステムを合わせて69件追跡しています。内訳は、デプロイ可能なチップまたはプラットフォームを持つベンダー51社、NPU IPサプライヤー9社、ウォッチリスト企業9社です。**互換性のない複数の種類のアクセラレーターと、それに近い数のコンパイラースタックにまたがっています。多くのベンダーが約束する手順は共通しています。モデルをエクスポートし、量子化し、コンパイルし、独自形式の成果物をボードにコピーして、ベンダーのランタイムを呼び出します。その手順が午後だけで終わるか、エンジニアリングに四半期かかるかは、細部で決まります。

このガイドでは、2026年に信頼できるコンピュータビジョン向けハードウェアを提供する企業を整理しています。対象チップ、SDKとコンパイラーの名称、デプロイ成果物、公開アクセスの範囲、ベンダーが実際に資料で取り上げているモデルを記録しています。また、新しいADLAスタックが旧A311D NPUエコシステムと実質的に異なるAmlogicを詳しく取り上げます。

> **調査範囲：**資料は2026年8月15日に確認しました。特に断りがない限り、モデル名を挙げた記述は、製品ページ、ドキュメント、モデルズー、リポジトリ、ベンダーチュートリアルなどの一次資料に基づきます。公称TOPSはベンダーの数値であり、相互に比較できる独立ベンチマーク結果ではありません。これは開発者向けガイドであり、投資助言や有料ランキングではありません。

**クイックナビゲーション：**[企業一覧](#npu-companies-and-platforms-at-a-glance) | [Amlogicの詳説](#amlogic-two-npu-generations-not-one) | [ベンダープロファイル](#the-strongest-public-deployment-ecosystems) | [コンパイル済み成果物の表](#what-you-actually-deploy-the-artifact-lock-in-table) | [アクセスとライフサイクル](#tool-access-and-lifecycle-signals) | [LibreYOLOのロードマップ](#what-this-means-for-libreyolo)

## 最も重要な発見

ハードウェアは細分化されていますが、デプロイの手順は驚くほど共通しています。

```text
PyTorch checkpoint
    -> ONNX or TFLite interchange graph
    -> representative calibration data
    -> vendor quantizer and graph compiler
    -> chip-specific binary or model package
    -> vendor runtime on the target
    -> application preprocessing, decode, NMS and rendering
```

[ONNXはランタイム、コードジェネレーター、ハードウェア実装を明示的に許容しています](https://onnx.ai/onnx/repo-docs/IR.html)。ただし、ONNXファイルは受け渡し形式にすぎません。すべてのONNX演算子をあらゆるアクセラレーターに変換できるという意味ではありません。静的入力形状、サポート対象の演算子、量子化規則、ホスト側フォールバックによって、実際にNPUで実行される内容が決まります。

そのため、ソフトウェアスタックもチップの一部です。公称性能が高くても、コンパイラーが制限付き、または不安定なNPUは、公開ツールチェーン、モデルズー、シミュレーター、安定したランタイムを備えた小型デバイスより、デプロイ先として劣ることがあります。

## このガイドでの「サポート」の意味

ベンダー資料では「サポート」という言葉が非常に曖昧に使われています。この記事では、証拠を4段階に分けます。

| 証拠レベル | 証明できること | 証明できないこと |
|---|---|---|
| **検証済みモデル** | ベンダーがチップ固有のモデル項目、結果、互換性表を公開していること | 変更したチェックポイントでも同じ精度を保てること |
| **公式サンプル** | ベンダーが特定モデルのエンドツーエンドのチュートリアルやデモを提供していること | 別のサイズ、タスク、世代もコンパイルできること |
| **コンパイラー機能** | SDKがフレームワークをインポートする、または対応演算子を公開していること | 特定のYOLOグラフがエンドツーエンドで動作すること |
| **マーケティングまたはアクセス制限付き** | 製品がビジョン用途を想定し、非公開SDKがあること | 公開環境で再現可能なモデル互換性 |

この区別は重要です。「ONNXをインポートできる」ことは「YOLO11セグメンテーションに対応する」ことと同じではありません。モデルを解析できても、CPUにフォールバックしたり、出力数値を誤ってコンパイルしたり、アクセラレーターのメモリを超過したり、量子化で精度を落としたりすることがあります。

## NPU企業とプラットフォームの概要

以下の表では、SoC、ディスクリートアクセラレーター、スマートセンサー、近接領域のGPU/FPGAを意図的に並べています。NPU、AIPU、BPU、KPU、DLA、HTP、TPU、MLAなど名称は違っても、同じデプロイ先の選択肢になります。

### 組み込みSoC、スマートセンサー、産業用プロセッサー

| 企業 | 関連チップまたはプラットフォーム | SDK、コンパイラー、成果物 | 公開資料で確認できるコンピュータビジョン対応 | アクセス |
|---|---|---|---|---|
| [Amlogic](https://github.com/Amlogic-NN/amlnn-toolkit) | A311D2、S928X、S905X5/S905D5、A311Y3、C308L/C302X/C302X2、T968D4、C305X2、A123X | AMLNN Toolkitと`libnnsdk.so`、コンパイル済み`.adla`。A311D向けには別の旧Acuity`.nb`スタック | [モデルプレイグラウンド](https://github.com/Amlogic-NN/amlnn-model-playground)では、A311D2、S905X5、A311Y3、C305X2、A123X上のYOLOv5/6/7/8/10/11、YOLOX、YOLOE、YOLO-World、PP-YOLOE、セグメンテーション、姿勢推定、OBBを公開。ここに挙げた他のチップはコンパイラーの対象であり、この対応表への掲載ではありません | 公開GitHubとwheel。対応BSP/ドライバーも必要 |
| [Rockchip](https://github.com/airockchip/rknn-toolkit2) | RK3562/3566/3568、RK3576、RK3588、RV1126B | RKNN-Toolkit2、RKNN Runtime、`.rknn` | [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)にYOLOv5/6/7/8/10/11、YOLOX、YOLO-World、PP-YOLOE、セグメンテーション、姿勢推定、OBB | 公開GitHub、バイナリーwheel |
| [Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | SnapdragonおよびDragonwingプラットフォーム、QCS6490、QCS8550、Dragonwing IQ-9075 | QAIRT/QNN、SNPE、AI Hub。QNNコンテキストバイナリーまたはDLC | [AI Hubのモデルコレクション](https://github.com/qualcomm/ai-hub-models)にはYOLOv3/5/6/7/8/9/10/11/26、YOLOX、YOLO-World、YOLOR、RF-DETR、検出・セグメンテーション・姿勢推定モデルが含まれます | ドキュメントは公開。SDKとアカウントの要件は異なります |
| [Texas Instruments](https://github.com/TexasInstruments/edgeai-tidl-tools) | AM62A、AM67A、AM68A、AM69A、TDA4x。プレビュー版TDA54-Q1 | Processor SDK Edge AI、TIDL | [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo)は、現行AM6xA/TDA4向けに最適化した検出、セグメンテーション、姿勢推定、分類モデルを公開。これはプレビュー版TDA54-Q1向けの対象表ではありません | 公開GitHubとProcessor SDK |
| [NXP](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | i.MX 8M Plus、i.MX 93、i.MX 95、買収したKinara Ara-1/Ara240 | TIM-VX、Vela、Neutron Converter、Ara SDKを含むeIQ | [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo)にはYOLOv4-tiny/v8、NanoDet、CenterNet、FastestDet、SSD Lite、YOLACTの成果物またはレシピが含まれます。YOLOv5の項目はドキュメントのみです | 公開コンポーネントとアカウント制限付きコンポーネントが混在 |
| [STMicroelectronics](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) | Neural-ARTアクセラレーター搭載のSTM32N6x7各種。STM32N657/647など | STM32Cube AI Studio、ST Edge AI Core | 現行サービスリポジトリにはTiny YOLOv2、YOLOv5u、YOLOv8、YOLO11、YOLO26、ST-YOLOX、姿勢推定、セグメンテーションが掲載されています | ツールとリポジトリは公開 |
| [Infineon](https://documentation.infineon.com/psocedge/) | Cortex-M55とEthos-U55を備えるPSOC Edge E83/E84。M33側には別のNNLiteアクセラレーター | ModusToolbox、[DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter)、TFLite Micro、Arm Vela | アーキテクチャ資料では代表的なカーネルとしてMobileNetV1/V2が挙がり、E84 AI Kitにはカメラが含まれます。ただし、PSOC EdgeのファーストパーティYOLO検証表は確認できませんでした | 公開ドキュメント、SDKコンポーネント、販売中のE84評価キット |
| [Renesas](https://github.com/renesas-rz/rzv_drp-ai_tvm) | RZ/V2L、V2M、V2MA、V2H、V2N | DRP-AI Translator、DRP-AI TVM、RUHMI | [RZ/V2Hモデル一覧](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md)でYOLOv5/v8/v11/26各サイズ、YOLOX、姿勢推定、セグメンテーションの派生モデルを検証 | 公開GitHubとボードSDK |
| [Sony](https://www.aitrios.sony-semicon.com/edge-ai-devices/imx500) | IMX500インテリジェントビジョンセンサー、Raspberry Pi AI Camera | Edge-MDT、Model Compression Toolkit、IMX500 packer。`.rpk`パッケージ | [Raspberry Pi IMX500 zoo](https://github.com/raspberrypi/imx500-models)にはYOLOv8n、YOLO11n、EfficientDet Lite、NanoDet+、SSDが含まれます | Raspberry Piの手順は公開。Sonyのツール利用条件が適用されます |
| [Ambarella](https://www.ambarella.com/developer/model-garden/) | CV72/CV75、CV5/CV52、CV3-ADファミリー | Cooper Developer Platform、CVflowコンパイラー、ランタイムDAG | 公開モデルガーデンにはYOLOX-S、RTMDet-nano、DeepLabV3+、TopFormer、OWL-ViT、LLaVA OneVisionが掲載されています | 主にパートナー向け |
| [Synaptics](https://developer.synaptics.com/docs/sl/overview) | Astra SL1600/SL1680、SL2611/13/15/17/19。別系統のSR100 MCUファミリー | SL16xxではSyNAPの`.synap`、SL261xではTorq/IREEの`.vmfb`。SR向けにはEthos-U55を中心とする別SDK | 公式YOLOv8n/v8s SyNAPベンチマークとSL261x TorqのYOLOv8サンプル。SRの証拠は別途評価が必要です | 開発者ポータル。一部ダウンロードは制限付き |
| [MediaTek](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) | NP8とMDLA 5.3を搭載するGenio 360/360P/420/520/720。旧NP6/MDLAを搭載するGenio 510/700/1200 | NeuroPilot Converter、`ncc-tflite`、Neuron Runtime、`.dla`。一部の新システムではONNX Runtime経路 | 公式IoT AI HubではYOLOv5とYOLOv8のモデルページおよびベンチマーク、分類・顔モデルを公開 | Yoctoドキュメントは公開。主要なNP8バンドルとAndroid資料は直接取引先またはNDAが必要 |
| [Allwinner](https://docs.aw-ol.com/v853/en/npu/dev_npu/) | 1-TOPS Vivante NPUを搭載したV853ビジョンSoC | Acuity/Pegasus変換、`viplite`ランタイム | 公式V853資料にはYOLOv2/3/4/4-tiny/5/5s、RetinaNet、MobileNet、ResNet、顔・人物ネットワークが記載されています | 公開ドキュメント。SDKパッケージのダウンロードにアカウントが必要な場合があります |
| [Canaan/Kendryte](https://github.com/kendryte/nncase) | 現行K230/K230D、旧K210/K510 KPU | `nncase`コンパイラーとランタイム。`.kmodel` | K230のnncaseガイドではYOLOv5sのコンパイル、シミュレーション、ランタイムを説明。公式デモカタログと現行[CanMV変更履歴](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md)にはYOLOv8/11/26の各タスクが記載されています | 公開コンパイラー/PyPI。チッププラグインはバイナリー |
| [Alif Semiconductor](https://alifsemi.com/support/kits/ensemble-e7appkit/) | Ethos-U55を含む2基のMLアクセラレーターを備えたEnsemble E7。[E8](https://alifsemi.com/ensemble-e8-series/)にはEthos-U85が1基、Ethos-U55 NPUが2基 | Arm VelaとTFLite Microに基づくデプロイ | AlifはファミリーレベルのINT8 YOLO-Fastest顔検出器ベンチマークを公開していますが、現代的なYOLOの広範な対象別対応表はありません | 公開ドキュメントと評価キットの資料 |
| [Himax](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) | Cortex-M55とEthos-U55を備えたHX6538 WiseEye2エンドポイントAI MCU | TFLite MicroとArm Vela。CMSIS-NNおよびリファレンスカーネルにフォールバック | 公式[WiseEye2サンプル](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2)にはYOLOv8nの検出・姿勢推定・分類、YOLO11n検出、フェイスメッシュ、PeopleNetが含まれます | 公開GitHub、ドキュメント、購入可能なパートナーボード |
| [Analog Devices](https://www.analog.com/en/products/max78002.html) | 低消費電力CNNアクセラレーター搭載AI MCUのMAX78000とMAX78002 | [`ai8x-training`](https://github.com/analogdevicesinc/ai8x-training)、`ai8x-synthesis`/`izer`、MSDK。生成されるCコードと重み | 公式資料では顔の識別・検出、RetinaNet、Visual Wake Words、分類器、動作認識を扱います。ファーストパーティのYOLOデプロイは確認できませんでした | 公開GitHubツール、ドキュメント、評価ボード |
| [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo) | RDK X3/X5/Ultra、新しいS100シリーズのBPUボード | OpenExplorer/Algorithm Toolchain、`hbm_runtime`。X5は`.bin`、現行`rdk_s`は`.hbm` | 現行`rdk_x5`ブランチはRDK X5向けにYOLOv5/v5u/v8/v9/v10/11/12/13/26、YOLOE、YOLO-World、セグメンテーション、姿勢推定、分類、OCR、CLIPを記載。X3とSシリーズは別ブランチです | 公開GitHub。一部ツールチェーンパッケージはプラットフォーム専用 |
| [AXERA](https://github.com/AXERA-TECH/ax-samples) | AX650、AX637、AX630C、AX620Q、AX615 | Pulsar2コンパイラー、AXEngine。`.axmodel` | 現行公式サンプルではYOLOv5/6/7/8/9/10/11/13/26、YOLOX、YOLO-Worldを扱います。タスクと対象チップは異なります | サンプルとドキュメントは公開。コンパイラーは別途配布 |
| [SOPHGO](https://github.com/sophgo/tpu-mlir) | BM1684/1684X/1688/1690、CV186X/CV18xx | TPU-MLIR、SOPHONランタイム。`.bmodel`または`.cvimodel` | 公式デモではYOLOv3/4/5/7/8/9/10/11/12/26、YOLOX、PP-YOLOE、YOLO-World、OBB、セグメンテーション、顔認識、SAM、OCRを扱います | オープンソースコンパイラーとベンダーランタイム |
| [Huawei Ascend](https://www.hiascend.com/en/software/cann) | Ascend 310/310P/310B、Atlas 200I/300Iエッジ製品 | CANN、ATCコンパイラー、AscendCL。`.om` | 公式[Ascend ModelZoo](https://github.com/Ascend/modelzoo)にはYOLOv2/3/4/5、YOLOX、YOLOR、SSD、RetinaNet、Mask R-CNN、セグメンテーションモデルが含まれます | 公開ドキュメント。CANNパッケージとカーネルを対象チップに合わせる必要があります |
| [Cambricon](https://github.com/Cambricon/magicmind_cloud) | 引用されている公開表のMLU370-X4/S4。MLU270は旧製品で、その表の対象外 | Neuware、MagicMind | リポジトリのMagicMind 1.7対応表にはYOLOv3/4/5/7/8、PP-YOLOE、SSD、RetinaFace、RetinaNet、Mask R-CNN、セグメンテーションモデルが掲載されています | 過去のモデル例は公開。現行SDK/コンテナーはアクセス制限付き |
| [Sunplus](https://sunplus.atlassian.net/wiki/spaces/C3/pages/1971126471/SP7350%2BSpecification) | 公称約4.1〜4.6 TOPSのSP7350/C3V | Vivante Acuity NPU Docker、SNNFランタイム、`.nb` | 公式ドキュメントではYOLOv5と、カスタムのエンドツーエンド[YOLOv8デプロイ](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350)を説明しています | 公開ソース/ドキュメントとボードエコシステム |
| [ESWIN Computing](https://www.eswincomputing.com/en/news/info/96.html) | EIC7700/EIC7700X、デュアルダイの[EIC7702/EIC7702X](https://www.eswincomputing.com/en/news/info/104.html) RISC-V SoC | ENNP：EsQuant、現行EsAACコンパイラー、シミュレーター、ESSDKランタイム。`.model`。旧ドキュメントでは`ennc-compile`を使用 | Milk-V提供のENNPドキュメントにエンドツーエンドの[YOLOv3チュートリアル](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3)、MobileNetV2、ResNetの例があります | ファーストパーティとボード提供ドキュメントが混在。地域別ダウンロード |
| [Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m55-mcus/m55m1-series/index.html) | Ethos-U55-256搭載のM55M1 MCU | TFLite Micro、Arm Vela、NuEdgeWise/NuML | 公式[NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise)の例にはYOLOv8-nano、YOLOX-nano、YOLO Fastest、SSD-MobileNet、分類モデルがあります | 主に公開されたMCUツールチェーン |
| [Realtek](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | AmebaPro2 RTL8735B / AMB82-miniカメラプラットフォーム | Arduino/FreeRTOS SDK、VoE、NeuralNetwork API。`.nb` | パッケージ済みモデルにはYOLOv3-tiny、YOLOv4-tiny、YOLOv7-tiny、SCRFD、MobileFaceNetが含まれます | ランタイムは公開。[カスタム変換ツールへのアクセス](https://ameba-doc-arduino-sdk.readthedocs-hosted.com/en/latest/FAQ/offline_ai_model_conversion_steps.html)には問い合わせが必要 |
| [Telechips](https://docs.topst.ai/product/p/ai) | 公称8 TOPSのTCC7500 / TOPST AIボード | TC-NN-Toolkit / Enlight SDK。中間形式`.enlight`とコンパイル済みデプロイバンドル | [公式TOPSTプロジェクト](https://docs.topst.ai/blog/31)でYOLOv8sをデプロイ。UFLD v1/v2の変換は未対応の層で失敗し、分割と後処理による回避策だけが提案されました。[YOLOv4](https://community.topst.ai/t/segmentation-fault/358)と[別のYOLOv8手順](https://community.topst.ai/t/segmentation-fault-error/415)はコミュニティのサポートスレッドにあります | ボードのドキュメントは公開。コンパイラー/演算子のダウンロードは権限が必要な場合があります |
| [T-Head](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | LicheePi 4A搭載のTH1520、公称4-TOPS INT8 | HHBコンパイラー、CSI-NN2/SHL。`hhb.bm`と生成コード/パラメーター | 公式ボードベンダーの例ではYOLOv5n/s、MobileNetV2をNPUで実行 | 公開例。ツールチェーンの古さと保守状況に不確実性があります |
| [SigmaStar](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | 現行SSU9383CMスマートカメラSoCと旧IPUファミリー | 現行MI_IPUランタイム。旧[SGS_IPUツールチェーン](https://wx.comake.online/doc/doc/SigmaStarDocs-SSC9381G_9351_Pudding-ULS00V040-20210913/customer/development/dla/tools.html)では`.sim`から`sgsimg.img`を生成 | [旧公式SDKの後処理機能](https://wx.comake.online/doc/doc/Sigmastar_SDK_v1.2.2/module/User_Guide/Common/SigmaStar_Post_Processing_Module.html)にSSD、YOLOv1/2/3が記載されています。SSU9383CMでの互換性は未検証です | 現行シリコンですが、公開資料にあるコンパイラー/モデルの世代は分かれています |
| [HiSilicon](https://www.hisilicon.com/cn/products/smart-vision/machine-vision/hi3516cv610) | Hi3516CV610/DV500、Hi3519DV500、Hi3403V100スマートビジョンSoC | NNN/SVP-NNNボードランタイムを介したATCから`.om`への変換 | Hi3403、Hi3591Pなどを対象とするHiSparkの対応表にはYOLOv3からYOLO11、姿勢推定、セグメンテーション、OBB、OCR、深度の項目があります。この表は行内のすべてのSoCでの検証を示すものではありません | Ascendとは異なる現行製品。リポジトリのモデルは非商用利用のみと表示されています |

### ディスクリートのエッジアクセラレーターとモジュール

| 企業 | 関連シリコン | SDKと成果物 | 公開資料で確認できるモデル対応 | アクセス |
|---|---|---|---|---|
| [Hailo](https://github.com/hailo-ai/hailo_model_zoo) | Hailo-8、Hailo-8L、Hailo-10H、Hailo-15ファミリー | Dataflow Compiler、HailoRT。`.hef` | YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12、YOLO26、YOLOX、DAMO-YOLO、SSD、EfficientDet、セグメンテーション、姿勢推定、OBB。世代ごとのモデル表あり | Model Zooは公開。コンパイラーはDeveloper Zone経由 |
| [Axelera AI](https://docs.axelera.ai/sdk/reference/models/model-zoo/) | 出荷中のMetis製品。発表済みのEuropaおよびTitania製品群 | 公開Voyager SDK。アルファ版Pipeline Builderは`.axm`/`.axe`、従来版は`.axmodel`バンドル | YOLOv3/5/7/8/9/10/11/26の検証済み例、YOLOX、YOLO-NAS、OBB、姿勢推定、セグメンテーション、多数の非YOLOモデル | SDKはGitHubで公開。顧客サポートはアカウント制 |
| [DEEPX](https://developer.deepx.ai/modelzoo/) | DX-M1、DX-M1M | DXNN SDK、DX-COM、DX-RT。`.dxnn` | 2026年8月15日の確認時、DX-COM 2.4.0/DX-RT 3.4.0でModel Zooに354件が掲載されていました。YOLOv3からYOLO11、YOLO26、YOLOX、SSD、EfficientDet、NanoDet、セグメンテーション、姿勢推定、OBBを含みます | 開発者向けリソースとスイートは公開 |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | MX3アクセラレーターとマルチチップモジュール | MemryX SDK、Neural Compiler、ランタイム。`.dfp` | 公式例とリリースノートでは、YOLOv10、YOLO11、YOLO26を含む検出、セグメンテーション、姿勢推定を扱います | 公開ドキュメントとSDK |
| [Kneron](https://doc.kneron.com/docs/) | KL520、KL530、KL630、KL720、KL730は現行コンパイラードキュメントに掲載。KL830は一部PLUS APIにありますが、現行コンパイラー対象一覧にはありません | Kneron PLUS、Model Toolchain。ドキュメントにあるコンパイラー対象向けの`.nef` | 公式[YOLO手順](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/)はTiny-YOLOv3を中心に説明。ほかの資料ではYOLOv5も扱います | 公開ドキュメント。ツールパッケージ/ダウンロードはチップごとに異なります |
| [SiMa.ai](https://docs.sima.ai/pages/palette/modelsdk.html) | MLSoC、量産中の50-TOPS Modalixプラットフォーム | Palette、ModelSDK、MLA Compiler、ModelExecutor | 公開リリースではYOLOv7/v8、YOLOX、姿勢推定/セグメンテーション、DETR、Mask R-CNN、EfficientDetを検証 | ドキュメントは公開。製品SDKは商用 |
| [EdgeCortix](https://www.edgecortix.com/en/hardware) | M.2およびPCIe製品の公称60-TOPSアクセラレーターSAKURA-II | MERAコンパイラー、MERAソフトウェアフレームワーク | ビジョンから生成AIまでの対応を説明していますが、現行YOLOの互換性を十分に特定した公開表はありません | 商用トライアル/注文の問い合わせ。パートナー主導の検証 |
| [BrainChip](https://brainchip.com/metatf-dev-tools/) | 出荷中のAKD1500コプロセッサー/M.2モジュール、旧AKD1000プラットフォーム、Akida 2 IP | MetaTFパッケージ：`akida-models`、`quantizeml`、`cnn2snn`、`akida`ランタイム | Akida 2モデルカードにはAkidaNet0.5 YOLOv2検出器、CenterNet、AkidaUNet、顔認識が掲載されています。この結果は自動的にAKD1500の検証を意味しません | 中核Pythonパッケージは公開。ハードウェアへのマッピングは対象固有 |
| [Blaize](https://www.blaize.com/products/) | P1600、Pathfinder、Xplorer製品 | Picasso SDK、NetDeploy、AI Studio | ビジョンを対象市場としていますが、監査可能な公開のモデル別互換性表は確認できませんでした | 商用/アクセス制限付き |
| [Google Coral](https://github.com/google-coral/edgetpu) | 旧Edge TPU USB、PCIe、M.2、Dev Board製品。別系統で開発中のオープンソース[Coral NPU IP](https://github.com/google-coral/coralnpu) | 旧Edge TPU Compiler/`libedgetpu`/PyCoral。新しいRISC-V Coral NPUはIP、RTL/シミュレーション、ELFの例を公開 | 旧製品の例はSSD MobileNet、分類、DeepLab、MoveNetが中心。新しいIPには公開YOLOデプロイ表がなく、Edge TPU製品の後継としてそのまま使えるものではありません | 旧中核リポジトリはアーカイブ済み。新しいCoral NPU IPは開発継続中 |
| [Lattice Semiconductor](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | ECP5、iCE40 UltraPlus、CrossLink-NX、CertusPro-NX、Avant-E、Avant-X FPGA | sensAI Studio、Neural Network Compiler、構成可能なアクセラレーターIP | 現行コンパイラー表では、対象ごとのモードでYOLOv1、YOLOv5、YOLOv8、YOLO11を掲載。SSD、MobileNetV2-SSD、ResNet、ENetも含みます | コンパイラー/マニュアルは公開。IPの条件は異なり、Advanced CNN Acceleratorは商用 |
| [Mobilint](https://www.mobilint.com/sdk-qb) | 公称10-TOPSのREGULUS、80-TOPSのARIES MLA100アクセラレーター | qb SDK、INT8コンパイラー、`.mxq` | 公開[モデルズー](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md)にはYOLOv3/5/7/8/9/10/11/12/26の検出、セグメンテーション、姿勢推定、OBBが掲載されています | 公開ドキュメント/モデル。SDKとハードウェアは商用 |
| [Rebellions](https://rebellions.ai/developers/) | 現行のATOM+ CA22、ATOM-Max CA25。EoLのATOM CA02/ATOM+ CA12。ATOM-Lite CA21のツール状況は不明 | RBLN SDKのコンパイラー、ランタイム、プロファイラー。`.rbln` | 公式対応リリースにはYOLOv3、YOLOv5/6/7/8ファミリー、現行YOLOv8チュートリアルが掲載されています | ドキュメントは公開。コンパイラーwheelにはポータル認証が必要 |
| [FuriosaAI](https://furiosa.ai/warboy) | ビジョン向けWarboy Gen1。別世代のRNGD | Furiosa SDK。WarboyのINT8は`.enf`。RNGDは別途`.fxb` | Warboy zooではSSD、YOLOv5M/L、YOLOv7-w6-poseを説明。RNGDのロードマップではYOLOv8m対応が2024 Q4完了とされていますが、現行の公開ビジョン詳細表はありません | IAM/アカウントが必要。2つの世代を分けて扱ってください |

### NPUと比較される近接プラットフォーム

| 企業 | ハードウェア | デプロイスタック | 比較対象となる理由 |
|---|---|---|---|
| [NVIDIA](https://github.com/NVIDIA-AI-IOT/deepstream_tools/tree/main/yolo_deepstream) | DLA搭載Jetson Orin GPU。DLA非搭載のJetson Thor GPU | JetPack、TensorRT、DeepStream。`.engine` | 成熟度の高いビジョンデプロイ環境です。公式DeepStreamツールではYOLOv4/v7/v8/v9/11を説明し、YOLO11 OBB設定もあります。多くの結果はGPUを使い、Orin DLAの未対応の層はGPUにフォールバックできます |
| [AMD](https://vitisai.docs.amd.com/en/6.2/) | Kria/旧DPU対象。Versal AI Edge Gen1 VEK280/VE2802向けVitis AI 6.2 GA、Gen2 VEK385。別系統のRyzen AIクライアントNPU | 旧`.xmodel`。Gen1は対象固定のNPU snapshot。Gen2はコンパイル済みキャッシュディレクトリまたは本番用`.rai`パッケージ | Vitis AIはビジョン関連資料を公開していますが、旧DPU、Versal Gen1、Versal Gen2、Ryzen AIはコンパイルおよびデプロイの契約が別です |
| [Microchip](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox) | 構成可能なCoreVectorBloxアクセラレーターIP搭載のPolarFire SoC FPGA | 公開VectorBlox SDK 3.1、Libero。`.vnnx`、`.hex`、`.ucomp`デプロイ資産 | 現行[チュートリアル](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md)ではYOLOv5n、YOLOv8の検出/分類/OBB/姿勢推定/セグメンテーション、YOLOv9tなどを扱います。SDK 3.1の現行対象はPolarFire SoC Video Kitです |
| [Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) | Core Ultra NPU、CPU、内蔵/ディスクリートGPU | OpenVINO IRまたはコンパイル済みモデルキャッシュ | OpenVINOは公開のクロスXPU経路を提供します。すべてのYOLO例がNPU検証済みだと仮定せず、[検証済みモデル表](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html)のNPU列を確認してください |
| [Apple](https://developer.apple.com/machine-learning/core-ml/) | A11以降のAシリーズチップ、MシリーズチップのApple Neural Engine | Core ML、`coremltools`。`.mlpackage`/`.mlmodelc` | Core MLを通じて利用しやすいコンシューマー向けNPUですが、CPU/GPU/Neural Engineの厳密な処理分担は抽象化されています。YOLO専用のハードウェアコンパイラーではありません |
| [GreenWaves Technologies](https://github.com/GreenWaves-Technologies/nn_menu_gap9) | NE16ニューラルエンジン搭載GAP9 | GAP SDK、NNTool、AutoTiler生成コード | 公式リポジトリにはMobileNet SSD、顔検出、分類、専用の[YOLOX人物検出器](https://github.com/GreenWaves-Technologies/yolox_people_detection)があります。完全版SDKは認定顧客に限定されます |
| [Syntiant](https://www.syntiant.com/hardware) | 量産中のNDP200、サンプリング中のNDP250 Neural Decision Processor | Syntiant SDK、デプロイパッケージ | 超低消費電力の常時稼働ビジョン/センサープロセッサーです。NDP200は1 mW未満、NDP250画像認識は30 mW未満と資料に記載されています。広範な公開YOLO対応表は確認できませんでした |

## Amlogic：1つではなく2世代のNPU

ウェブ上の情報では互換性のない製品がまとめられていることが多いため、Amlogicには詳しい説明が必要です。同社には、旧VeriSiliconベースの経路と、新しい独自ADLA経路があります。コンパイラー、成果物、ランタイムは共通ではありません。

| 世代 | 代表的なチップ | NPUとツールチェーン | コンパイル済み成果物 | 実用上の状況 |
|---|---|---|---|---|
| 旧世代 | A311D、S905D3。Khadas VIM3/VIM3Lでよく見られます | VeriSilicon Vivante VIPNano-QI、Acuity toolkit、KSNN、旧`aml_npu_sdk` | `nbg_unify`出力ディレクトリ内の`.nb` | 既存ボードとデモ。旧方式の統合が別途必要 |
| 現行ADLA2 | C308L/C302X、S928X、A311D2、T968D4、S905X5/S905D5、C302X2 | Amlogic ADLA2、AMLNN Toolkit、NNSDK2 | 対象ごとの`.adla`。W8A8またはW8A16 | 整数演算中心の現行スタック |
| 現行ADLA3 | A311Y3、C305X2、A123X | Amlogic ADLA3、AMLNN Toolkit、NNSDK2 | 対象ごとの`.adla`。W4A8、W8A8、W4A16、W8A16、W16A16 | ネイティブINT4、FP16、BF16に対応 |

[A311Dのデータシート](https://dl.khadas.com/products/vim3/datasheet/a311d-datasheet.pdf)には、旧5-TOPS INT8 NPUが記載されています。旧[Khadas VIM3アプリケーションページ](https://docs.khadas.com/products/sbc/vim3/npu/npu-app)と[NPU概要](https://docs.khadas.com/products/sbc/vim3/npu/start)には、DenseNet CTC、MTCNN、RetinaFace、YOLOFace、YOLOv2、YOLOv3、YOLOv3-tiny、YOLOv4、YOLOv7-tiny、YOLOv8nが掲載され、概要にはYOLOv8n-poseも記載されています。FaceNetは非推奨です。これらの例は実在しますが、旧Acuity`.nb`エコシステムの証拠であり、現行ADLAチップの証拠ではありません。A311DとA311D2、C305XとC305X2は別製品です。

ハードウェアにはもう1つ注意点があります。[Khadas VIM4のリビジョンガイド](https://docs.khadas.com/products/sbc/vim4/configurations/identify-version)によると、初期のV12/A311D2リビジョンBボードにはNPUがありません。公称3.2-TOPS NPUが搭載されたのは、A311D2-N0DリビジョンC部品を使うV13A以降です。テスト用デバイスを選ぶ際、製品名だけでは不十分です。

### 現行AMLNNおよびADLAスタック

現行の[AMLNN Toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/7d3cc9a36179b756ce79c953c14d43acab1f2af5)は、モデル変換、量子化、コンパイル、推論、プロファイリングに対応します。固定した[2026年5月版ユーザーガイド](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/02_Amlogic_NPU_User_Guide_V0.1.pdf)には、ONNX、浮動小数点または量子化済みTFLite、TorchScript`.pt`、PyTorch 2 ExportedProgram/PT2のインポーターが記載されています。詳細ガイドではTensorFlow、Paddle、Keras経路が計画中とされています。リポジトリ概要では、より広く説明されています。コンパイラーの出力は`.adla`です。PythonデプロイではAMLNNまたは`amlnn_edge_toolkit_lite`を使います。ネイティブC/C++では、`nnsdk2.h`を介して`libnnsdk.so`をリンクします。ホスト側開発ではADB経由で`nnserver`を使えます。ランタイムはAndroid、Buildroot、Yocto、一部のDebian/Armbian環境も対象です。

ツールキットに公開されているプラットフォームIDは次のとおりです。

| 対象ID | プラットフォーム |
|---:|---|
| `001` | C308L / C302X |
| `002` | S928X |
| `003` | A311D2 |
| `004` | T968D4 |
| `005` | S905X5 / S905D5 |
| `006` | C302X2 |
| `007` | A311Y3 |
| `008` | 詳細資料とモデル対応表ではC305X2 / A123X |

この対象フィールドは形式的なものではありません。Amlogicでは、`.adla`、`nnsdk2.h`、`libnnsdk.so`、BSP、コンパイラー/ランタイム世代を一致させる必要があります。固定版[クイックスタート](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/01_Amlogic_NPU_Quick_Start_guide_V0.1.pdf)では、ADLAドライバー2.0.2以降、NNSDK 3.0.0以降、NNSDK2 1.0.0以降を求めています。[Androidデプロイガイド](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/06_Amlogic_NPU_Android_Deployment_Guide_V0.1.pdf)では、ライブラリ、ヘッダー、ドライバー、BSPの関係を具体的に説明しています。`.adla`は移植可能なモデルではなく、ABIに依存するビルド成果物として扱ってください。

量子化方法もNPU世代に依存します。ADLA2の対象001〜006ではW8A8、W8A16コンパイルが記載されています。ADLA3の対象007、008では、[演算子ガイド](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/05_Amlogic_NPU_Support_Operator_List_V0.1.pdf)にあるネイティブINT4、FP16、BF16対応に加えて、W4A8、W4A16、W8A8、W8A16、W16A16を利用できます。Amlogicは代表的なキャリブレーションサンプルを200〜500枚使うことを推奨しています。ランダムデータのオプションは、精度を保つ量子化ではなく性能テスト用だと明記されています。

### Amlogicの公開モデル対応

[固定版Amlogicモデルプレイグラウンド](https://github.com/Amlogic-NN/amlnn-model-playground/blob/03ca63ef20c4f1560722de26ab910826884f334a/README.md)は、一般的なONNX対応の主張より強い証拠です。公開された対応/サンプル表にはA311D2、S905X5、A311Y3、C305X2、A123Xが記載されています。他の対象IDがコンパイラーで受け付けられることは、このモデル一覧全体の検証を意味しません。

| タスク | 公開資料に記載されたモデル |
|---|---|
| 分類 | MobileNetV2、ResNet50-v2。ADLA3ではDINO |
| 物体検出 | PP-YOLOE、YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv10、YOLO11、YOLOE、YOLO-World、YOLOX、QR検出 |
| 顔・ジェスチャー | RetinaFace、Gesture Recognition |
| セグメンテーション | DeepLabV3、PP-LiteSeg、YOLOv5-seg、YOLOv8-seg |
| 回転物体検出 | YOLOv8 OBB |
| 姿勢推定 | BlazePose detector/landmark、YOLOv8 pose |
| OCRと音声 | LPRNet、PaddleOCR各種、Whisper Tiny、対象を限定した新しいチップ上のSenseVoice |
| 新しいTransformer・マルチモーダル処理 | DETR、MobileSAM、CLIP/MobileCLIP、新しいプラットフォーム上の一部低ビットLLM/VLM例 |

同じリポジトリでは、次のW8A8モデルのランタイム数値も公開しています。数値はNPU上でのニューラルネットワークのベンダー測定値であり、カメラパイプライン全体の性能ではありません。

| モデルと入力 | S905X5 | A311D2 | A311Y3 |
|---|---:|---:|---:|
| YOLOv8n、640 x 640 | 101.72 FPS | 95.14 FPS | 191.06 FPS |
| YOLOv8s、640 x 640 | 42.33 FPS | 42.77 FPS | 83.08 FPS |
| YOLOv8m、640 x 640 | 19.67 FPS | 19.82 FPS | 35.30 FPS |
| YOLOv8l、640 x 640 | 10.53 FPS | 10.12 FPS | 18.37 FPS |
| YOLO11n、640 x 640 | 41.14 FPS | 41.48 FPS | 62.24 FPS |

注意点は特に重要です。Amlogicの演算子ガイドでは、ADLA2とADLA3の両方でNMSをソフトウェアで実行するとしています。READMEでは、前処理と後処理を除外したNPUモデル本体の実行結果と定義しています。メモリ転送をすべて含むかは明らかにされていません。YOLOの精度行はCOCO検証セット全体ではなく、300枚のサブセットを使用しています。分類行ではImageNetの`val1000`を使用しています。また、表には条件の説明がないまま、A311Y3のYOLOv8nについてFPS表とは異なるレイテンシも記載されています。クロック、電力モード、冷却、定常温度、正確なSDK/BSPバージョン、テスト時間は開示されていません。数値は1つのベンダースタックを理解するために使い、他社とのランキングには使わないでください。

### Amlogicが戦略的に興味深い理由

Amlogicには、大きな組み込みSoCの展開規模、新たに公開されたコンパイラースタック、LibreYOLOのタスク領域と大きく重なる現行モデル表、主要な学習ライブラリでのファーストクラス統合の少なさという、珍しい特徴が4つあります。現行リポジトリは2025年末から2026年初めに公開されたばかりで、実質的なマニュアルも2026年5〜7月のものです。そのため、先行者利益を得る機会がある一方、APIの安定性にはリスクもあります。


堅牢な統合では、LibreYOLOの決定論的ONNXエクスポートをコンパイラー入力にし、低精度ビルドでは代表的なキャリブレーション画像を必須にします。`.adla`とメタデータマニフェストを生成し、NNSDK2アダプター経由で読み込みます。旧A311Dは明確に異なる`.nb`対象として扱ってください。`.nb`と`.adla`を1つのバックエンドとして見せると、互換性の問題を気付かずに引き起こします。AmlogicのリポジトリはトップレベルのライセンスがApache-2.0ですが、同梱wheel、共有ライブラリ、`nnserver`、Android AARバイナリーの再配布権は、LibreYOLOでミラーする前に直接確認する必要があります。

## 公開デプロイ環境が特に充実した企業

次に挙げる企業は、現在、入手可能なハードウェア、公開技術資料、モデル名を示す証拠の組み合わせが特に明確です。どこでも最速という意味ではありません。ハードウェアを購入する前に互換性の主張を評価しやすいという意味です。

### Hailo

Hailoは、専用ビジョンアクセラレーターのエコシステムを代表する例です。モデルをHailo Archiveに解析し、代表画像で最適化と量子化を行い、その後Hailo Executable Format（`.hef`）ファイルにコンパイルします。アプリケーションはHailoRT経由でこのHEFを読み込みます。

[Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)には、検出、セグメンテーション、分類、姿勢推定などのレシピ、学習済みモデル、後処理設定、性能データが含まれます。YOLOではYOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12、YOLO26に加え、YOLOX、DAMO-YOLO、SSD、EfficientDetを扱います。一般的な製品ページより、バージョン固定の[Hailo-8物体検出表](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.19.0/docs/public_models/HAILO8/HAILO8_object_detection.rst)のほうが互換性を確認しやすい資料です。[スタンドアロンアプリケーション](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/README.md)では、デプロイ可能なYOLOパイプラインを説明しています。

重要なバージョン境界があります。Hailo-8とHailo-8Lは旧Model Zoo 2.x、Dataflow Compiler 3.x、HailoRT 4.x系列を使います。一方、Hailo-10とHailo-15は現行5.x世代です。Hailoという名称が共通でも、レシピ、パーサーの挙動、HEFに互換性はありません。

Hailo-15には別のアプリケーション層もあります。Hailo-8/10HのTAPPAS形式のホスト経路ではなく、Vision Processor Software PackageとHailo Media Libraryを使います。公開リファレンスリポジトリ[`hailo-camera-apps`](https://github.com/hailo-ai/hailo-camera-apps)は2026年5月3日にアーカイブされましたが、[`hailo-apps-core`](https://github.com/hailo-ai/hailo-apps-core)は別の公開アプリケーションフレームワークとして継続しています。リポジトリのアーカイブは製品EOLの証明ではありません。Hailo-15のプロジェクトではDeveloper Zoneで現行のアプリケーションパッケージを確認してください。

LibreYOLOの[Hailoデプロイガイド](/docs/export/hailo)は、プロプライエタリなコンパイラーをPython依存関係として同梱できないため、静的ONNXを受け渡すところまでを扱います。適切なグラフを作ることと、HEFの動作検証済みを主張することの境界を正直に示しています。

### Rockchip

Rockchipは、特にRK3588、RK3576、RK356xのボードを通じて、趣味用途と商用組み込みLinuxで最大級の展開規模を持ちます。公開されている[RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)は、x86 Linuxホスト上でモデルを変換・量子化します。ボード上ではRKNN RuntimeまたはToolkit-Liteが生成した`.rknn`ファイルを実行します。

[RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)は、チップ、モデルファミリー、精度を明示している点で非常にわかりやすい資料です。現行表にはYOLOv5/6/7/8/10/11、YOLOX、PP-YOLOE、YOLO-World、YOLOv5/v8セグメンテーションのFP16とINT8経路が掲載されています。YOLOv8 OBBとYOLOv8 poseはINT8のみです。OCR、顔認識、ほかのセグメンテーションネットワークも扱います。

LibreYOLOにはすでに、慎重に範囲を絞った直接[RKNNエクスポーター](/docs/export/rknn)があります。RK3588上の検出バリアントを厳密に4種類検証し、コンパイラーシミュレーターとONNX Runtimeを比較できます。検証済みファミリー以外は、ビルド成功を正しい予測と同一視せずに拒否します。この限定的な対応表明は、今後のすべてのNPUバックエンドの手本になります。

### Axelera AI

Axelera AIは「Accelera」と誤記されることがありますが、Metis AIPUを開発し、M.2、PCIe、マルチチップ製品で販売しています。現在注文可能なのはMetisファミリーです。EuropaとTitaniaは発表済み製品群なので、すべてを同じ販売状況として扱うべきではありません。[Voyager SDKはGitHubで公開](https://github.com/axelera-ai-hub/voyager-sdk)されており、モデル選択、コンパイル、量子化、実行、アプリケーションパイプラインを扱います。顧客サポートは引き続きアカウント制です。

公開[Voyager Model Zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo/)は、市場でも特に参考になる資料です。YOLOv3、YOLOv5、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLO11、YOLO26 n/s/m/l/x、YOLOX、YOLO-NAS、OBB、姿勢推定、セグメンテーションについて、モデルのバリアント、入力サイズ、数値精度、精度指標、測定性能が掲載されています。分類や高密度予測モデルも多数あります。量子化による精度低下と精度を速度と並べて公開するほうが、公称ピークTOPSだけを示すより有用です。

API世代の変更に伴い、成果物の名称も変わりました。アルファ版[Pipeline Builderのコンパイル手順](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/)では`.axm`モデルを出力し、移植可能なパイプラインを`.axe`としてパッケージ化できます。[従来版のパイプラインドキュメント](https://docs.axelera.ai/sdk/reference/pipeline/model-formats/)では、`.axmodel`とモデルメタデータ、マニフェストを説明しています。統合側では拡張子を固定せず、インストールされたVoyagerの世代を検出する必要があります。

### Qualcomm

Qualcommのデプロイ対象は非常に幅広い一方、「Qualcomm NPU」という名称の裏には複数のインターフェースがあります。デバイスや製品クラスによって、QAIRT/QNN経由のHexagon HTP、旧SNPE経路、Qualcomm AI Hubコンパイルサービス、LiteRT、ONNX RuntimeのQNN Execution Providerを使います。

公式[Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models)リポジトリは、モデル単位の強い証拠です。YOLOv3、YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLO11、YOLO26の検出/セグメンテーション/姿勢推定、YOLOX、YOLO-World、YOLOR、RF-DETRに加え、分類、深度、姿勢推定モデルの最適化済みパッケージを公開しています。AI Hubではデバイス固有のプロファイリングも提供します。あるHTP世代向けにコンパイルしたモデルが、別世代に自動的に移植できるわけではないためです。

主な統合作業は対象の組み合わせの多さです。Androidと組み込みLinux、QCSの発注部品とコンシューマー向けSnapdragon、HTPアーキテクチャ、QNN/QAIRTバージョン、量子化方式をすべて考慮する必要があります。Qualcommの現行ページ間でも[Dragonwing IQ-9075](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075)の状況に食い違いがあります。製品ページではActive、[EVK](https://www.qualcomm.com/developer/hardware/qualcomm-iq-9075-evaluation-kit-evk)では評価可能とされていますが、[Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program)では引き続きSamplingと表示され、2038年までの長期供給が記載されています。発注/SKUの接頭辞はQCS9075です。Qualcommは高密度INT8で50および100 TOPSの構成、Ubuntu/Yocto対応を公称しています。正確な発注SKUの量産状況を確認し、LibreYOLO統合では「Qualcomm」という一般的なフォルダー名ではなく、そのSKUを記録してください。

### DEEPX

DEEPXのDX-M1/DX-M1MアクセラレーターはDXNN SDKを使います。DX-COMがモデルをコンパイル、量子化し、DX-RTが実行します。デプロイ成果物は`.dxnn`形式です。同社は[DX-AllSuite](https://github.com/DEEPX-AI/dx-all-suite)、[DX-APP](https://github.com/DEEPX-AI/dx_app)のアプリケーション例、大規模な[オンラインModel Zoo](https://developer.deepx.ai/modelzoo/)を公開しています。

公開カタログでは、2026年8月15日の確認時にDX-COM 2.4.0とDX-RT 3.4.0の組み合わせで354件が報告されていました。公開資料の対応範囲には、YOLOv3からYOLO11、YOLO26、YOLOX、SSD、EfficientDet、NanoDet、DAMO-YOLO、YOLOセグメンテーション、姿勢推定、回転ボックスが含まれます。コンパイラー/ランタイムの境界とデプロイ成果物が明確で、公開ビジョン対応も幅広いため、DEEPXは特に有力な統合候補です。

## 産業用SoCには複数のエコシステムがある

産業用ベンダーは、メーカー向けボードのSoCより製品ライフサイクルが長く、カメラ、安全性、リアルタイム処理との統合に優れることがよくあります。一方、1社が複数の無関係なNPUアーキテクチャを出荷するため、AIスタックは複雑になりがちです。

### Texas Instruments

TIの現行エッジAIプロセッサーにはAM62A、AM67A、AM68A、AM69A、関連するTDA4デバイスがあります。ソフトウェア経路はProcessor SDK Linux、TIDLコンパイラー/ランタイム、[Edge AI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools)、[Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo)を組み合わせます。TIDLはONNX Runtime、TensorFlow Lite、他のアプリケーションランタイムと統合し、対応するサブグラフをオフロードできます。

TIはフレームワークのインポート機能だけでなく、より多くの資料を公開しています。モデルズーには物体検出、セグメンテーション、姿勢推定、分類、深度などの変換済みモデルと性能メタデータがあります。TIは、モデルズーの成果物を開発の出発点とし、本番ですぐに使える資産とは限らないとも注意しています。ベンダー比較では欠けがちな重要な注意点です。

TIは[TDA54-Q1](https://www.ti.com/product/TDA54-Q1)もPreviewとして掲載しています。この部品は最大4基のC7 NPUを搭載し、最大400 TOPSです。より広いTDA5ファミリーでは最大1,200 TOPSと公称されています。これは新世代に関するベンダー製品の主張であり、TIが対象別対応表を公開する前に、AM6xA/TDA4のTIDLモデル検証をTDA54-Q1へ引き継げることを意味しません。

### NXP

NXPでは、少なくとも4つのバックエンドを説明する必要があります。

| NXPの対象 | アクセラレーター | コンパイラー/ランタイム経路 |
|---|---|---|
| i.MX 8M Plus | 2.3-TOPS VeriSilicon Vivante NPU | TIM-VX/VX delegateを使うeIQ |
| i.MX 93 | Arm Ethos-U65 | 量子化TFLiteとArm Vela |
| i.MX 95 | NXP eIQ Neutron NPU | Neutron ConverterとeIQランタイム |
| Ara-1 / [Ara240](https://www.nxp.com/products/ARA240) | Kinara由来のディスクリートNPU。Ara240は最大40 eTOPSを公称 | Ara SDK。eIQ全体との統合も進行 |

[eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo)にはYOLOv4-tiny、YOLOv8、NanoDet、CenterNet、FastestDet、SSD Lite、YOLACTなどの成果物やレシピがあります。YOLOv5の項目はドキュメントのみです。1つのバックエンド向けに検証された項目が、4つすべてに対応する証拠ではありません。NXP自身の[i.MXプラットフォーム向けYOLOエクスポートガイダンス](https://community.nxp.com/t5/i-MX-Processors-Knowledge-Base/Exporting-YOLO-Models-for-NXP-i-MX-Platforms/ta-p/2381361)も、対象ごとの変換が必要であることを示しています。NXPによると、モノリシックな[eIQ Toolkitはバージョン1.17以降、2025 Q3に更新を終了](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/eIQ-FAQ/ta-p/1099741)しました。現在はeIQ Neutron SDKなどの独立パッケージを使います。

NXPは[2025年10月にKinaraの買収を完了](https://media.nxp.com/news-releases/news-release-details/nxp-completes-acquisitions-aviva-links-and-kinara-advance/)しました。[Ara240の仕様書](https://www.nxp.com/docs/en/fact-sheet/ARA240DNPUFS.pdf)ではベンダー公称最大40 eTOPSと、YOLOv8nで毎秒313枚を報告しています。NXPは製品ページでAra SDKとeIQ Toolkitの両方を案内していますが、別系統の[i.MX 95 Neutron経路](https://eiq.nxp.com/learning-hub/convQuant/neutron.html)と成果物を相互利用できるとは限りません。[16 GB M.2モジュール](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT)は販売中ですが、USBオプションは量産前です。NXPはeTOPSの「e」を「equivalent（相当）」と定義しており、「effective（実効）」ではありません。他社の高密度INT8 TOPS指標と直接比較すべきではありません。

### STMicroelectronics

[STM32N6x7デバイス](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html)にはSTM32N657、STM32N647などが含まれ、MCUクラス製品に600-GOPS Neural-ARTアクセラレーターを搭載しています。汎用のN6x5シリーズにはこのアクセラレーターはありません。[STM32Cube AI Studio](https://www.st.com/en/development-tools/stedgeai-cubeai.html)とST Edge AI Coreがインポートモデルを解析・最適化し、[STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services)がデプロイ、最適化、ベンチマーク、サンプル手順を提供します。

現行の[物体検出の概要](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/docs/README_OVERVIEW.md)では、Tiny YOLOv2、YOLOv5u、YOLOv8、YOLO11、YOLO26、ST-YOLOXに加え、分類、姿勢推定、セグメンテーションのサービスを説明しています。これは200-TOPS PCIeカードと同等の性能クラスではありません。カメラ入力、推論、制御を、厳しい電力・メモリ制約の組み込み環境に収められる点が興味深いです。

### Renesas

Renesas RZ/VプロセッサーにはDRP-AIアクセラレーターが統合されています。ソフトウェアはDRP-AI Translator、DRP-AI TVMから、EdgeCortix MERA技術を使う[RUHMI](https://www.renesas.com/en/software-tool/ruhmi-framework)へと進化しています。公開されている[RZ/V DRP-AI TVMリポジトリ](https://github.com/renesas-rz/rzv_drp-ai_tvm)と[RZ/V2H検証一覧](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md)には、分類モデルに加え、YOLOv5、YOLOv8、YOLO11、YOLO26のn/s/mバリアント、YOLOX、姿勢推定、セグメンテーションが掲載されています。

Renesasはロボティクスや産業用途で有力な対象ですが、再現性のためにボード、DRP-AI世代、Translatorのバージョン、CPU側後処理を記録する必要があります。

## スマートセンサーとカメラ向けプロセッサー

### Sony IMX500

SonyのIMX500は通常のアプリケーションプロセッサーではありません。画像センシングとAI処理を積層しており、センサー内で推論することで、カメラから外へ送る画像データを減らせます。Raspberry Pi AI Cameraによって、この構成を特に利用しやすくなっています。

公式の[Raspberry Pi IMX500モデルリポジトリ](https://github.com/raspberrypi/imx500-models)では、YOLOv8n、YOLO11n、EfficientDet Lite0、NanoDet+、SSD MobileNetV2 FPN Liteのパッケージ済みモデルを公開しています。[AI Cameraドキュメント](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)では、変換、パッケージ化、センサー内後処理を説明しています。デプロイ単位は汎用ONNXファイルではなくRPKパッケージで、センサーのメモリと演算子の制約が設計上の重要な制限になります。Raspberry Piの[製品ページ](https://www.raspberrypi.com/products/ai-camera/)によると、このカメラは少なくとも2028年1月まで生産されます。

### Ambarella

AmbarellaのCVflowプロセッサーはカメラ、ドローン、車載ビジョンで広く使われています。現行ファミリーにはカメラ向けCV72/CV75、高性能ビジョンシステム向けCV5/CV52、車載向けCV3-ADデバイスがあります。Cooper開発者プラットフォームとCVflowコンパイル/ランタイム経路が、公開資料に記載されたソフトウェアスタックです。

[公開の開発者向けモデルガーデン](https://www.ambarella.com/developer/model-garden/)にはYOLOX、RTMDet、DeepLabV3+、TopFormer、OWL-ViT、マルチモーダルモデルが掲載されています。ただし、コンパイラーの詳しいドキュメントとダウンロードはパートナー向けです。Ambarellaを統合する場合、公開pipパッケージがあると想定せず、ベンダーやOEMとの連携から始めるべきです。

### Synaptics Astra

Synapticsには関連する対象が3種類あります。SL1600/SL1680 Astraシステムは、ソースモデルとYAML記述から`model.synap`パッケージを生成する[SyNAP toolkit](https://developer.synaptics.com/docs/synap/introduction)を使います。新しいSL2611/13/15/17/19デバイスは、`.vmfb`を生成するIREE/MLIRベースの[Torqプラットフォーム](https://developer.synaptics.com/docs/torq/introduction)を使います。[SR100シリーズ](https://developer.synaptics.com/docs/sr/introduction-sr)はCortex-M55とEthos-U55を中心とする別のMCUファミリーで、専用SDKを持ちます。これらの成果物とAPIは相互利用できません。

公式[YOLOベンチマークチュートリアル](https://developer.synaptics.com/docs/sl/tutorials/vision/benchmark-yolo)は、具体的な手順を示しています。YOLOv8をTFLiteにエクスポートし、SyNAPで非対称UINT8キャリブレーションを行い、SL1680向けにコンパイルした後、`synap_cli`と物体検出アプリケーションを実行します。SL1680上のYOLOv8n/v8sを実証し、SyNAPではYOLO11まで対応すると説明しています。別の[SL261x物体検出ガイド](https://developer.synaptics.com/docs/sl/sl2600/getting-started/object-detection)では、Torqの`.vmfb`からYOLOv8を実行します。いずれも対象を限定した公式例であり、3ファミリーすべてのYOLOグラフに対応するという主張ではありません。

### MediaTek Genio

MediaTekを詳しく取り上げる理由は、現行の[IoT AI Hub](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html)に、以前の市場調査では監査できなかったハードウェア/ソフトウェア対応表、モデルパッケージ、ベンチマーク表が公開されたためです。Genio 360/360P/420/520/720はMDLA 5.3搭載NeuroPilot 8を使います。Genio 510/700はMDLA 3.0搭載NP6、Genio 1200はMDLA 2.0搭載NP6を使います。MediaTekによると、NeuroPilotとMDLAの世代、演算子セット、コンパイラー、ランタイムは、各SoCの製品寿命を通じてバージョンに依存します。

分析系AIの経路はTFLite中心です。

```text
PyTorch model
    -> NeuroPilot Converter and representative calibration
    -> quantized .tflite
    -> version-matched ncc-tflite compiler
    -> chip-generation-specific .dla
    -> Neuron Runtime on MDLA
```

新しいGenio製品では、オンラインLiteRT delegationと、対応OS上のONNX Runtime CPU/NPU経路も利用できます。これらはオフライン`.dla`とは異なる実行モードです。[ソフトウェアアーキテクチャガイド](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/software_architecture.html)と[リソース一覧](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html)では、その違いを明示しています。

公式の[分析系モデル表](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical.html)には、複数MDLA世代向けのYOLOv5sとYOLOv8sのベンチマーク項目、変換ガイドが掲載されています。AGPL-3.0の制約により、変換済みYOLO成果物は配布しないと明記されています。Quant8、640 x 640のオフライン測定では、Genio 720上でそれぞれ5.35 ms、8.04 msです。[YOLOv8sモデルページ](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical/YOLOv8s.html)には入力/出力テンソル、バックエンド別時間、MediaTek独自演算への注意が記載されています。これらはベンダーベンチマークであり、カメラ処理全体の結果ではありません。

アクセスが制約になります。Yoctoの公開ドキュメントは詳細ですが、現行NP8のオールインワン変換/コンパイラーバンドルやAndroid資料の多くは、NDAまたは直接取引先向けです。通常の開発者アカウントでは、MediaTek Onlineの顧客アカウントと同じアクセス権を得られません。そのためLibreYOLOでは、Genioを技術的には実証済みだが、再現可能な独自モデル用コンパイラー統合にはパートナー連携が必要な対象として扱うべきです。

## 中国中心のエッジAIエコシステム

英語の市場一覧では、RockchipからNVIDIAへ話が飛び、その間にあるエコシステムが抜け落ちることがよくあります。

### D-RoboticsとHorizon由来のBPUプラットフォーム

D-Roboticsは、RDK X3、X5、Ultra、新しいS100シリーズの製品で使われるBPUアクセラレーターを中心に、RDK開発ボードのエコシステムを維持しています。[RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo)の現行`rdk_x5`ブランチには、RDK X5向けのYOLOv5/v5u、YOLOv8/9/10/11/12/13/26、YOLOE、YOLO-World、検出、セグメンテーション、姿勢推定、分類、OCR、CLIPの手順が記載されています。X3は別ブランチを使用し、現行Sシリーズ向け資産はメインzooの[`rdk_s`ブランチ](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s)にあります。旧[`rdk_model_zoo_s`リポジトリ](https://github.com/D-Robotics/rdk_model_zoo_s)には過去のデモがあります。したがって、現行X5一覧は、すべてのモデルがX3、Ultra、S100でも検証済みである証拠ではありません。

OpenExplorer/Algorithm ToolchainはONNXまたはCaffeをインポートしてPTQを行い、PyTorchを用いるQAT手順にも対応しています。デプロイはプラットフォームごとに異なります。現行RDK X5は`.bin`を使い、`libdnn`経由で`hbm_runtime`から実行します。現行の主要`rdk_s`経路は`.hbm`を使い、`libhbucp`上の同名`hbm_runtime` APIから実行します。X3は古いブランチと推論インターフェースを維持しています。旧`rdk_model_zoo_s`資料にも過去の`.bin`/`.hbm`経路が記載されているため、これらの成果物は対応するブランチとツールチェーンに組み合わせる必要があります。未対応演算子はCPUにフォールバックする場合があります。公開例は有用ですが、RDK OS、ボードファームウェア、ツールチェーン、モデルバイナリーのバージョン対応も互換性契約の一部です。

### AXERA

AXERAのAX650/AX630/AX620ファミリーは、SipeedのMaixCAMシリーズなどの小型AIカメラやボードで使われています。Pulsar2はONNXをインポートし、キャリブレーションと精度分析を行い、`.axmodel`を出力します。成果物はAXEngineが実行します。

[AXERA公式サンプル](https://github.com/AXERA-TECH/ax-samples)ではYOLOv5/6/7/8/9/10/11/13/26、YOLOX、YOLO-Worldを扱います。タスクと対応チップはプラットフォームごとに異なります。対応対象上の現行YOLO26例には、検出、姿勢推定、セグメンテーション、OBBがあります。[Pulsar2変換例](https://pulsar2-docs.readthedocs.io/en/latest/appendix/model_convert_examples.html)からは、実際の作業内容がわかります。正確なテンソル名、出力ノードの分割、レイアウト変換、キャリブレーションアーカイブ、対象ハードウェア設定などです。

### SOPHGO

SOPHGOのTPU-MLIRは、独立した統合にとって魅力のあるコンパイラースタックの1つです。コンパイラー自体がオープンソースだからです。ONNX、PyTorch、TFLite、Caffeをインポートし、グラフを変換・量子化して、BM対象では`.bmodel`、CV18xxハードウェアでは`.cvimodel`を出力します。

[TPU-MLIRプロジェクト](https://github.com/sophgo/tpu-mlir)は、対象に応じてFP32、BF16、FP16、INT8経路に対応しています。[SOPHONデモ集](https://github.com/sophgo/sophon-demo/blob/release/README_EN.md)には、YOLOv3/4/5/7/8/9/10/11/12/26、YOLOX、PP-YOLOE、YOLO-World、OBB/セグメンテーション各種、SSD、CenterNet、RetinaFace、SAM/SAM2、OCRが掲載されています。BM1684X上のデモは、BM1688やCV18xx上で同じ成果物が使えることを示すものではありません。

### Huawei Ascend

Huaweiのエッジ推論製品には、Atlas 200I、300I製品のAscend 310/310P/310Bシリコンが含まれます。CANNはATCコンパイラー、AscendCLランタイム、チップ固有の演算子カーネルを提供します。ATCはONNXなどのソース形式をオフライン`.om`モデルに変換します。

現行の[Atlas 200I DK A2ドキュメント](https://www.hiascend.com/en/hardware/developer-kit-a2/specification)とCANNサンプルには、Ascend 310B向けのYOLOv5`.om`経路があります。より広範で古い[Ascend ModelZoo](https://github.com/Ascend/modelzoo)には、YOLOv2/3/4/5、YOLOX、YOLOR、SSD、RetinaNet、Mask R-CNN、姿勢推定、セグメンテーションモデルがあります。ただし、その全項目が310Bで再検証済みであるかのように紹介すべきではありません。CANN、カーネル、ファームウェア、SoCを一致させる必要があります。Ascend310P向けの`.om`は、汎用Ascend成果物ではありません。

### Cambricon

CambriconのMLUアクセラレーターはNeuwareとMagicMind推論エンジンを使います。公開の[MagicMind Cloudリポジトリ](https://github.com/Cambricon/magicmind_cloud)はMagicMind 1.7とMLU370-X4/S4の対応表に固定されています。これは過去の互換性の証拠であり、現行SDKやMLU270の検証表ではありません。このリポジトリではPyTorch、ONNX、Caffe、Paddleの経路が記載されています。TensorFlowフレームワーク対応は1.7で削除されましたが、過去のTensorFlow例は残っています。Cambriconの現行[3-series開発者ポータル](https://developer.cambricon.com/index/document/index/classid/3.html)には、より新しい全体SDKリリースが掲載されています。公開サンプルリポジトリと現行商用スタックを同じバージョンとして扱うべきではありません。

公式の[MagicMindクラウドモデルリポジトリ](https://github.com/Cambricon/magicmind_cloud)は、MLU370-X4/S4について、特に直接的な対応表を公開しています。YOLOv3、Tiny-YOLOv3、YOLOv4、YOLOv5、YOLOv7、YOLOv8、PP-YOLOE、SSD、RetinaFace、RetinaNet、Mask R-CNN、DeepLab、UNetの例と、C++またはPython例の有無が示されています。証拠は強く、主な障壁はCambricon経由でのSDKとコンテナーへのアクセスです。

### Canaan/Kendryte

Kendryteの現行K230/K230Dと旧K210/K510 KPUチップは、低価格ボードやスマートカメラの市場で重要です。[nncaseコンパイラー](https://github.com/kendryte/nncase)はONNXまたはTFLiteをインポートし、固定小数点変換にはキャリブレーションデータを使います。ホスト上で結果をシミュレートし、`.kmodel`を出力します。

公式の[K230 nncase開発ガイド](https://github.com/kendryte/k230_docs/blob/main/en/01_software/board/ai/K230_nncase_Development_Guide.md)では、YOLOv5sのコンパイル、シミュレーション、実行手順を説明しています。別の[AIデモ一覧](https://github.com/kendryte/k230_docs/blob/main/en/02_applications/ai_demos/K230_AI_Demo_Introduction.md)にはYOLOv8nの検出、セグメンテーション、姿勢推定の成果物が含まれます。現行[CanMV変更履歴](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md)には、最適化されたYOLOv8/11/26の分類、検出、セグメンテーション、OBB、姿勢推定の例が加わっています。データシートの[YOLOv5s結果](https://github.com/kendryte/k230_docs/blob/main/en/00_hardware/K230_datasheet.md)は公開ベンチマークの実測値です。コンパイラーとバイナリーKPUプラグインのバージョンは、ボードSDKと一致させる必要があります。

### Allwinner

AllwinnerのV853は、カメラ向けメディアハードウェアと1-TOPS Vivante NPUを組み合わせています。[公式英語版NPUガイド](https://docs.aw-ol.com/v853/en/npu/dev_npu/)では、Acuity/Pegasusによるインポート、量子化、検証、`viplite`経由のデプロイを説明しています。Allwinner自身のモデルページと[YOLOv5ガイド](https://docs.aw-ol.com/v853/npu/npu_yolov5/)には、YOLOv2/3/4/4-tiny/5/5s、RetinaNet、MobileNet、ResNet、顔・人物ネットワークが記載されています。

これは実際のコンピュータビジョン対応ですが、古いTina Linux/Vivanteツールチェーンとアカウントが必要なSDKダウンロードに結び付いています。その制約を明示したうえで、市場一覧に含めています。

## 見落とされがちな韓国、台湾、中国のプラットフォーム

英語の一覧ではRockchipからNVIDIAへ飛ぶことがよくあり、実際に資料のあるシリコンの中位層が抜け落ちます。これらの中には、知名度の高い欧米スタートアップより幅広い現行YOLO対応表を持つエコシステムもあります。

### Mobilint

韓国のアクセラレーターベンダーMobilintは、低消費電力のREGULUSと高スループットのARIES MLA100を販売しています。qb SDKはPyTorch、TensorFlow、TFLite、ONNX、Keras向け入力を受け付け、INT8に量子化してコンパイル済み`.mxq`成果物を出力します。公開[ビジョン対応表](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md)には、YOLOv3/v5/v7/v8/v9/v10/11/12/26検出に加え、YOLOセグメンテーション、姿勢推定、OBB各種が掲載されています。[ARIES製品ページ](https://www.mobilint.com/aries/mla100)には、YOLO11s、YOLO26m固有の結果もあります。統合を裏付ける証拠は強いですが、SDK/ハードウェアは商用です。

### Sunplus

Sunplus SP7350/C3VはVivante派生のNPUスタックを使いますが、Allwinnerや旧Amlogicとは異なるパッケージになっています。公開手順では、Acuity変換、NPU Docker環境、SNNFランタイム、`.nb`出力を組み合わせます。公式[カスタムYOLOv8ガイド](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350)は、フレームワークへの対応を主張するだけでなく、変換からデプロイまでを説明しています。関連IPでも、`.nb`を別のVivante系SoCへ移植できるわけではありません。

### ESWIN Computing

ESWINのRISC-VファミリーにはEIC7700/EIC7700X、デュアルダイのEIC7702/EIC7702Xが含まれます。EIC7700はMilk-V Megrezなど出荷中のボードに搭載されています。ENNPはEsQuant、現行EsAACコンパイラー、正解データ生成、シミュレーション、ESSDKランタイムを備え、コンパイル後は`.model`を出力します。旧ENNP資料では`ennc-compile`という名称が使われています。現行[EsAACページ](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac)ではONNX入力を説明しているため、ほかのフレームワークには対応IRへのエクスポートまたは変換が必要です。Milk-V提供の[ENNP概要](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/introduction)と[YOLOv3チュートリアル](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3)で、公開されたエンドツーエンドの手順を確認できます。ただし、現在の幅広いモデル/精度一覧までは確認できません。

### Nuvoton M55M1

NuvotonのM55M1は、Linuxアプリケーションプロセッサーではなく、Cortex-M55とEthos-U55-256を組み合わせたMCUクラスの設計です。NuEdgeWise/NuMLとArm Velaを使い、量子化TFLite Microモデルを準備します。公開[NuEdgeWiseリポジトリ](https://github.com/OpenNuvoton/NuEdgeWise)にはYOLOv8-nano、YOLOX-nano、YOLO Fastest v1.1、SSD-MobileNet FPNLite、複数の分類器が記載されています。これらの小型ネットワーク例はデバイスのメモリ/電力クラスに関する信頼できる証拠ですが、一般的な640ピクセルYOLOバリアントに対応する証拠ではありません。

### RebellionsとFuriosaAI

RebellionsのATOMシリーズはRBLN SDKを使い、コンパイル済み`.rbln`成果物を生成します。公式[対応リリース](https://docs.rbln.ai/v0.8.2/supports/release_note.html)にはYOLOv3 tiny/full/SPP、YOLOv5 nからx、YOLOv6 nからl、YOLOv7各種、YOLOv8 nからxが記載されています。現行[カード対応表](https://docs.rbln.ai/latest/supports/version_matrix.html)ではATOM CA02とATOM+ CA12がEoL、ATOM+ CA22とATOM-Max CA25がActiveです。ATOM-Lite CA21には製品ページがありますが、このSDK対応表にはなく、現行ツールチェーンの状況が不明です。ドキュメントは公開されていますが、コンパイラーwheelにはRebellions Portalの認証が必要です。

FuriosaAIは世代ごとに分けて扱う必要があります。Warboyは旧世代のコンピュータビジョンアクセラレーターで、INT8の`.enf`成果物を使います。公式[モデルズー](https://developer.furiosa.ai/furiosa-models/latest/)にはSSD、YOLOv5M/L、YOLOv7-w6-poseがあります。RNGDは主にLLM/VLM向けの別[`.fxb`スタック](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html)を使います。Furiosaの現行[ロードマップ](https://developer.furiosa.ai/latest/en/overview/roadmap.html)では、ビジョン向けYOLOv8m対応が2024 Q4に完了したと記録されています。しかし、現在公開されている対応モデルや性能情報はLLM/VLM中心で、詳細なYOLOv8m精度/性能表はありません。これはリリース済み世代についての証拠であり、Warboyの対応を示すものではありません。

### Realtek、Telechips、T-Head、SigmaStar

これら4つのプラットフォームは実在しますが、独自モデルの公開手段は限られているか、旧式であるか、アクセス制限付きです。

| プラットフォーム | 再現可能な公開資料 | 明示すべき制限 |
|---|---|---|
| Realtek AmebaPro2 | [Arduino/FreeRTOS SDK](https://github.com/Ameba-AIoT/ameba-arduino-pro2)にYOLOv3/4/7-tiny、SCRFD、MobileFaceNetの`.nb`モデルが含まれます | カスタムモデルの変換には、Realtekへの登録または問い合わせが必要です |
| Telechips TOPST TCC7500 | 公式[2026年プロジェクト](https://docs.topst.ai/blog/31)でYOLOv8sを変換してデプロイ | UFLD v1/v2の変換は未対応の層で失敗しました。記事では分割/後処理の回避策だけを提案しています。完全なコンパイラー/演算子リソースを得るには、通常メールでの許可が必要です |
| T-Head TH1520 | Sipeedの[アプリケーションガイド](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html)ではHHBとCSI-NN2/SHLでYOLOv5n/sを実行 | 公開スタックはありますが、現行の主要製品と比べ保守やバージョンの明確さが遅れています |
| SigmaStar SSU9383CM | 現行ドキュメントではMI_IPUランタイムAPIを説明。旧SGS_IPU資料には`.sim`から`sgsimg.img`への変換が記載され、旧後処理機能はSSDとYOLOv1/2/3を挙げています | 旧コンパイラー成果物やモデル手順がSSU9383CMに適用できるという公開証拠はありません |

### HiSiliconのスマートビジョンはHuawei Ascendではない

HiSiliconのHi3516CV610/DV500、Hi3519DV500、Hi3403V100カメラSoCは、NNN/SVP-NNNランタイムを通じてATCから`.om`を生成する手順を公開しています。公式の[HiSpark model zoo](https://gitee.com/HiSpark/modelzoo/blob/master/README.md)には、特にHi3403とHi3591Pを対象とした表に、YOLOv3/4/5/6/7/8/9/10/11、YOLO11セグメンテーション/姿勢推定、YOLOv8 OBB/World/セグメンテーション、OCR、深度、TinySAMが掲載されています。上記のスマートビジョンSoCすべてで全項目が実行できることを示すものではありません。一部はロードマップ項目です。公開済み例と計画中の対応は分けて扱う必要があり、リポジトリには提供モデルを非商用利用に限定する記述があります。ATC/`.om`という用語が共通していても、CANNベースのAscend製品ラインと成果物やランタイムに互換性があるとは限りません。

## 新興企業と特殊用途アクセラレーター

### MemryX

MemryX MX3モジュールはデータフローアーキテクチャを使い、複数チップを組み合わせられます。[Neural Compiler](https://developer.memryx.com/tools/neural_compiler.html)はONNX、TensorFlow Lite、Keras、TensorFlowモデルを受け付け、アクセラレーターランタイムが使うDFPパッケージを出力します。[ランタイムAPI](https://developer.memryx.com/api/accelerator/accelerator.html)は複数モデルやストリーミングのパイプラインにも対応します。

MemryXのドキュメントと例では、検出、セグメンテーション、姿勢推定向けに最適化したYOLOの前処理/後処理を扱います。[SDKリリースノート](https://developer.memryx.com/release_notes.html)にはYOLOv10、YOLO11、YOLO26への対応が明記されています。公開資料は技術的に有用ですが、Axeleraに見られるような、モデル、精度、デバイスをまとめた単一の表はありません。

### Kneron

KneronのKL520/KL530/KL630/KL720/KL730製品は、小型USBから組み込みアクセラレーターまでをカバーします。Kneron PLUSはアプリケーションランタイムを担います。現行Model Toolchain 0.33.1のドキュメントでは、これらの対象向けに変換、量子化、評価、シミュレーション、`.nef`コンパイルを扱います。KL830は一部の[PLUSランタイムAPI](https://doc.kneron.com/docs/plus_c/introduction/run_examples/)にありますが、現行の[コンパイラー対象一覧](https://doc.kneron.com/docs/toolchain/manual_1_overview/)にはありません。ベンダー確認なしに、一般的な`.nef`コンパイル対応をKL830へ適用してはなりません。

公式の[YOLO例](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/)では、Tiny-YOLOv3を使った処理手順を説明しています。ほかのKneron資料ではYOLOv5の学習とデプロイも扱います。信頼できる証拠ですが、Hailo、Axelera、Rockchip、DEEPXの幅広い現代YOLO対応表より範囲は狭いです。

### SiMa.ai

SiMa.aiのMLSoCと[量産中の50-TOPS Modalixプラットフォーム](https://sima.ai/press-release/sima-ai-next-gen-platform-for-physical-ai-in-production/)はPaletteソフトウェア環境を使います。ModelSDK、MLAコンパイラー、ModelExecutorでインポート、量子化、コンパイル、プロファイリング、実行を扱います。ワークロードと製品に応じてINT8、INT16、BF16系の経路をサポートします。

同社のリリースノートには、検証済みのYOLOv7、YOLOv8検出/姿勢推定/セグメンテーション、YOLOX、YOLOXセグメンテーション、DETR、Mask R-CNN、EfficientDetのパイプラインが記載されています。現行の制限を隠すべきではありません。[Palette SDK 2.1の注記](https://docs.sima.ai/v2.1.2/pages/release_notes/2.1.html)によると、Python/PT2E量子化のリグレッションによりQATは機能しません。記載されている回避策は、SDK 2.0でアノテーション済みONNXを作成し、それを2.1でコンパイルする方法です。アクセスと調達は企業向けです。

デプロイ境界もドキュメント化されています。[ModelSDKコンパイル手順](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html)では、コンパイル済み`.tar.gz`を出力し、実行可能ELFも生成できます。[MPK Tool](https://docs.sima.ai/v2.1.1/pages/palette/mpk_tools.html)はデプロイ用`.mpk`をパッケージ化します。これらの出力はMLSoC/Modalixの対象とPaletteのリリースに結び付いています。

### EdgeCortix

SAKURA-IIはビジョンと生成AI向けの公称60-TOPSアクセラレーターで、MERAソフトウェアフレームワーク経由でコンパイルします。EdgeCortixはRenesasの新しいRUHMI経路にもMERA技術を提供しています。

[SAKURA-II製品資料](https://www.edgecortix.com/en/edgecortix-sakura-ii-accelerator-brief)ではハードウェアとコンパイラーを説明し、[M.2/PCIeハードウェア](https://www.edgecortix.com/en/hardware)はトライアルまたは注文の問い合わせが可能です。十分に具体的で現行の公開YOLO互換性表は確認できませんでした。この情報の欠落も購入判断に役立ちます。ハードウェアを選定する前に、対象モデルの検証を依頼してください。

### BrainChip

BrainChipのAkidaは、一般的な密テンソル型NPUではなく、ニューロモーフィックなイベント領域プロセッサーです。[MetaTF環境](https://brainchip.com/metatf-dev-tools/)には、モデル作成、低ビット量子化、変換、シミュレーション、ハードウェア実行用のインストール可能なPythonパッケージが含まれます。BrainChipは旧AKD1000ハードウェアやAkida 2 IPとともに、[出荷中のAKD1500コプロセッサーをM.2形式で販売](https://brainchip.com/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/)しています。

現行の[Akida 2モデルカード](https://brainchip.com/wp-content/uploads/2025/04/Akida-2-Model-Card-V1.1-Mar.25.pdf)には、AkidaNet0.5 YOLOv2検出器、CenterNet、AkidaUNet、顔認識ネットワークが掲載されています。ほかの対応資料にはFOMOやイベントベースのワークロードもあります。Akidaは重みとアクティベーションを非常に低ビットで処理できますが、汎用INT8 ONNX対象として扱うのではなく、アーキテクチャを考慮した変換が必要な場合があります。AKD1000やAkida 2のモデルズー結果をAKD1500の結果として引用するには、明示的なマッピング/適合レポートが必要です。

### Blaize

BlaizeはP1600ベースのPathfinder、Xplorer製品を販売し、Picasso SDK、NetDeploy、AI Studioをソフトウェア経路として提供しています。[製品ページ](https://www.blaize.com/products/)ではビジョンとエッジAIを明確に対象としていますが、現行の公開モデル別互換性表はありません。

このガイドでは、コミュニティの主張で不足を埋めず、Blaizeを商用かつアクセス制限付きに分類しています。対象モデルのベンダー提供コンパイル/精度レポートを、選定の前提にしてください。

## TinyMLとエンドポイントビジョン

### Arm Ethos-UとAlif

ArmはEthos-U55、U65、U85のNPU IPをチップ企業にライセンスしています。[Velaコンパイラー](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u)は量子化LiteRT/TFLiteグラフを受け取り、対応サブグラフをEthos-U独自の演算に変換します。未対応演算はCPUに残る場合があるため、「アプリケーションが動く」ことと「ネットワーク全体がNPUで動く」ことは別の主張です。

実際の製品には、Alif Ensemble E7のEthos-U55、NXP i.MX 93のEthos-U65、新しいシステムのEthos-U85があります。Alifの[E7 AI/ML AppKit](https://alifsemi.com/support/kits/ensemble-e7appkit/)では、2基のMLアクセラレーターと、E7のTFLiteからVelaへの経路を実演しています。[Ensemble E8](https://alifsemi.com/ensemble-e8-series/)にはEthos-U85が1基、Ethos-U55 NPUが2基搭載されています。AlifはCortex-M55とEthos-U55で動作する、192 x 192のファミリーレベル[INT8 YOLO-Fastest顔検出器ベンチマーク](https://alifsemi.com/faster-ai-mcu-inferencing-low-power-consumption/)も公開していますが、現代的なYOLOの幅広い対象別対応表はありません。これらのメモリ制約のあるシステムは、小型の分類・検出ネットワークに適しています。TFLiteで表現できるというだけで、任意の640ピクセル検出器に適しているわけではありません。

### Infineon PSOC Edge、Himax WiseEye2、Analog Devices MAX7800x

Infineonの[PSOC Edge E83](https://documentation.infineon.com/psocedge/docs/xsk1761304020519)とE84ファミリーは、Cortex-M55とEthos-U55を組み合わせ、低消費電力のM33側には別のNNLiteブロックを追加しています。[E84アーキテクチャマニュアル](https://documentation.infineon.com/psocedge/docs/bwb1750411526047)では、8-bit重み、8または16-bitアクティベーション、対応演算子をVelaでNPUコマンドストリームに変換し、未対応の処理をCPUに残す手順を説明しています。[DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter)はTFLite、Keras、PyTorch経路を受け付けます。アーキテクチャ資料ではMobileNetV1/V2を代表的なカーネルとして挙げていますが、対象別ベンチマークではありません。[E84 AI Kit](https://documentation.infineon.com/psocedge/docs/cci1762693051052)にはカメラが付属し、ModusToolboxを使用します。ただし、公開資料にYOLOモデルの対応表はまだありません。これはモデル単位の証拠が拡充中の、実際にプログラム可能なビジョンプラットフォームです。汎用検出器として検証済みの対象ではありません。

Himaxの[HX6538 WiseEye2](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/)も、常時稼働ビジョン向けにCortex-M55とEthos-U55を組み合わせています。この電力クラスとしては、公開ソフトウェア資料が特に具体的です。公式[Grove Vision AI Module V2の例](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2)には、YOLOv8n物体検出、姿勢推定、性別分類、YOLO11n検出、フェイスメッシュ、PeopleNetがあります。統合経路はTFLite MicroとVelaを使い、必要に応じてCMSIS-NNとリファレンスカーネルにフォールバックします。これらは意図的に小型のモデルと解像度であり、一般的なサーバー規模のYOLOグラフが収まる証拠ではありません。

Analog Devicesの[MAX78000/MAX78002 CNNアクセラレーター](https://www.analog.com/en/products/max78002.html)は別の方式を取ります。公開[`ai8x-synthesis`](https://github.com/analogdevicesinc/ai8x-synthesis)ツールは学習済みネットワークを量子化し、移植可能なONNXランタイムパッケージではなく、対象固有のCコード、重み、アクセラレーター設定を生成します。ファーストパーティの資料には、[顔識別](https://www.analog.com/en/resources/app-notes/an-2616.html)、[ADIモデルズー](https://ez.analog.com/dsp/software-and-development-tools/edgebench/a/docs-faqs/DF912/available-models-in-adi-model-zoo)にあるTinierSSD QR検出、画像分類器があります。エンドポイントビジョンに関係するデバイスですが、現代的YOLOに関する現行の公式対応表は確認できませんでした。

### GreenWaves GAP9

GAP9はRISC-V計算機能とNE16ニューラルエンジンを組み合わせています。NNToolがネットワークをインポート、量子化し、AutoTilerとGAP SDKがデプロイ可能なコードを生成します。公開[GAP9ニューラルネットワーク一覧](https://github.com/GreenWaves-Technologies/nn_menu_gap9)にはMobileNet、EfficientNet、ResNet、MobileNet SSD、顔認識、認識アプリケーションがあります。GreenWavesは精度とシミュレーション結果を含む[YOLOX人物検出プロジェクト](https://github.com/GreenWaves-Technologies/yolox_people_detection)も公開しています。

これはTinyML分野でも特に透明性の高い証拠です。ただし、完全版SDKは認定顧客に限定され、モデルは一般的な640 x 640 YOLOバリアントより大幅に小型です。

### Lattice sensAI

Lattice sensAIは、固定NPUに代わるFPGAベースの選択肢です。sensAI StudioとNeural Network Compilerは、対応グラフをECP5、iCE40 UltraPlus、CrossLink-NX、CertusPro-NX、Avant-E、Avant-Xデバイス向けの構成可能なアクセラレーターIPに割り当てます。出力は選択したFPGA、アクセラレーターモード、メモリ配置、ビットストリームに結び付いており、移植可能なランタイムモデルではありません。

現行の[Neural Network Compilerマニュアル](https://www.latticesemi.com/-/media/LatticeSemi/Documents/UserManuals/MQ3/FPGA-UG-02052-8-0-Lattice-Neural-Network-Compiler-Software.ashx?document_id=52343)には、特に直接的なトポロジー表があります。ECP5向けYOLOv1、最適化/AdvancedモードのCrossLink-NXおよびCertusPro-NX向けYOLOv5/YOLOv8、AdvancedモードIPのみのYOLO11が記載されています。さらにSSD、MobileNetV2-SSD、ResNet、ENet、SqueezeDetなどのビジョンネットワークも、FPGAファミリーごとに未対応欄を明示して掲載されています。[Advanced CNN Accelerator](https://www.latticesemi.com/en/Products/DesignSoftwareAndIP/IntellectualProperty/IPCore/IPCores05/Advanced-CNN-Accelerator-IP)はAvant-E/Avant-X/CertusPro-NXを対象とする商用IPです。これはコンパイラー表の証拠であり、1つのビットストリームで列挙したすべてのトポロジーを同時に使えるという保証ではありません。

### Microchip VectorBlox

Microchipの公開[VectorBlox SDK 3.1](https://github.com/Microchip-Vectorblox/VectorBlox-SDK)は、PolarFire SoC FPGA内の構成可能なCoreVectorBlox IP向けに、完全量子化INT8 TFLiteネットワークを前処理してコンパイルします。`tflite_preprocess`と`vnnx_compile`で、`.vnnx`、`.hex`、`.ucomp`のデプロイ資産を生成します。リポジトリにはシミュレーターと対象ボード上のCドライバーも含まれます。上流のTensorFlow、ONNX、OpenVINOグラフは、資料に記載されたTFLite/INT8準備手順を通す必要があります。

現行の[チュートリアル対応表](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md)には、YOLOv5n、YOLOv8nの検出、分類、OBB、姿勢推定、セグメンテーション、YOLOv9t、疎/圧縮YOLO派生モデル、MobileNet、EfficientNet-Lite0、ResNet18、MiDaS、FFNet、QuickSRNetが含まれます。[Cの後処理ガイド](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/C_Postprocessing.md)では、YOLOv2/3/4/5、Ultralyticsの検出、姿勢推定、OBBデコーダーも別に挙げています。SDK 3.1の現行対応対象はPolarFire SoC Video Kitであり、SoC非搭載のPolarFire Video Kitは明示的に対象外です。そのため旧ボードのデモを現行の対象契約と混同してはなりません。

SDKは[Microchip独自ライセンス](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/LICENSE.md)のもとで公開され、ソフトウェアと派生物の利用はMicrochip製品に限定されています。Microchipは無料ですが申請が必要なLibero SilverとCoreVectorBloxライセンスを提供します。デプロイには引き続きFPGAシステム全体の整合が必要です。アクセラレーター設定、ビットストリーム、ファームウェア、SDK生成ネットワークデータ、メモリ配置を一致させる必要があります。VectorBloxでコンパイルできても、任意のPolarFire設計へコピーできるバイナリーになるわけではありません。

### Syntiant

SyntiantのNDP200は、通常のLinux SoCより低い電力範囲で常時稼働するビジョン、音声、センサー推論を対象にしています。現行[ハードウェア表](https://www.syntiant.com/hardware)ではNDP200が量産中、NDP250がサンプリング中です。[NDP200製品概要](https://www.syntiant.com/ndp200/)では、CNN、RNN、全結合ネットワークを1 mW未満でサポートするとしています。一方、[NDP250発表](https://www.syntiant.com/news/syntiant-unveils-ndp250-neural-decision-processor-with-next-gen-core-3-architecture)では、常時稼働の画像認識を30 mW未満としています。「サブミリワット」を両方の部品に一般化すべきではありません。

公開資料から幅広いYOLO互換性は確認できません。そのため、汎用ONNXエクスポートを想定するのではなく、ベンダー支援のモデル評価を通じて、小規模な常時稼働分類器や検出器に適するか検討してください。

### Google Coral：無関係な2世代

旧Edge TPU製品は、完全INT8 TFLiteモデルとEdge TPU Compiler、`libedgetpu`、PyCoralを使います。中核の[Edge TPU](https://github.com/google-coral/edgetpu)と[PyCoral](https://github.com/google-coral/pycoral)リポジトリはアーカイブされています。これは実際の保守リスクですが、正式なハードウェアEOL通知ではありません。

Googleは別系統で、開発が継続中のオープンソース[Coral NPU](https://github.com/google-coral/coralnpu)を維持しています。これは低消費電力SoCに組み込むためのRISC-VアクセラレーターIPです。公開プロジェクトではRTL、ハードウェアシミュレーション、ツールチェーン、ELFの例を現在提供しています。旧Edge TPUのUSB/M.2製品の後継としてそのまま使えるものではなく、現行YOLOデプロイ表もありません。

## 近接するGPU、クライアントNPU、適応型コンピューティングの注意点

NVIDIA、AMD、Intel、AppleはNPU検索に登場しますが、相互に置き換え可能な単一種類のアクセラレーターを提供しているわけではありません。

**NVIDIA Jetson：**Jetson OrinはCUDA GPUと専用DLAコアを組み合わせています。TensorRTはDLA用シリアル化エンジンを生成できますが、未対応の層をGPUで実行するにはGPUフォールバックを明示的に有効にする必要があります。現行の[DeepStream YOLO表](https://github.com/NVIDIA-AI-IOT/deepstream_tools/blob/main/yolo_deepstream/README.md)ではYOLOv4/v7/v8/v9/11を扱いますが、多くの項目はGPU向けです。明示的なONNX DLA項目はYOLOv8s INT8です。NVIDIAの[DLA制約](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html)に演算子の制限が記載されています。Thorはさらに別です。NVIDIAの[DriveOS移行ガイド](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/NVIDIA_DriveOS_7.0.3_Migration_Guide.pdf)によると、ThorではGPUスケジューリングの柔軟性を優先してDLAコアを削除しました。つまり「Jetson上で動く」だけでは、使われたアクセラレーターは特定できません。

**AMD Vitis AI：**旧Kria/DPU世代は`.xmodel`グラフをコンパイルし、VARTで実行していました。現行[Vitis AI 6.2](https://vitisai.docs.amd.com/en/6.2/)は、Versal AI Edge向けの別々の2経路でGAとなっています。[Gen1](https://vitisai.docs.amd.com/projects/gen1/en/latest/index.html)はVEK280/VE2802上でONNX、AMD Quark、対象固定のsnapshot/サブグラフ経路を使います。[Gen2](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html)はVEK385上で別のコンパイル/ランタイム契約を持ちます。Gen1の例ではYOLOv5、YOLOv7、YOLOv8、YOLOXを扱います。Ryzen AIは4つ目の別系統クライアントNPUスタックです。すべてVitisまたはAMDの名を持つというだけで、旧DPU、Versal Gen1、Versal Gen2、Ryzen AI間で成果物や検証結果を移してはなりません。

**Intel OpenVINO：**OpenVINOは1つのAPIからCPU、GPU、Core Ultra NPUを対象にできます。[Intel NPUプラグインのドキュメント](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md)には対応NPU世代が記載され、[検証済みモデル表](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html)ではデバイスごとの列を確認できます。YOLOノートブックはアプリケーション手順を示す資料です。表でNPU検証を確認できない限り、NPUで検証済みとは言えません。対応演算子、形状、精度、インストール済みNPUドライバーによって、グラフ全体が実行できるか決まります。OpenVINO IR（`.xml`と`.bin`）は再利用可能なソースレベルIRですが、コンパイル済みモデルキャッシュはデバイスとソフトウェアに依存します。

**Apple Core ML：**`coremltools`はモデルを`.mlpackage`に変換し、Xcodeが`.mlmodelc`へコンパイルします。Core MLはCPU、GPU、Apple Neural Engine間で処理を割り当てられますが、実際の分割を意図的に抽象化しています。そのためAppleは優れたデプロイ先ですが、プロファイリングで裏付けない限り「YOLOのグラフ全体がNPUで動いた」と主張するのは適切ではありません。

## チップ企業を支えるNPU IP企業

ボードやパッケージ済みアクセラレーターを直接販売せず、SoCベンダーにNPU設計をライセンスする企業もあります。同じ基礎アーキテクチャが複数のチップブランドに搭載されることがあるため重要ですが、最終SDKと成果物はライセンシーによって独自化される場合があります。

| IPサプライヤー | NPUファミリーとソフトウェア | 関連する理由 |
|---|---|---|
| [Arm](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) | Ethos-U55/U65/U85、Vela | NXP、Alif、Infineon、Himaxなどの組み込み製品に搭載。量子化TFLite/TOSA中心の経路 |
| [Arm China](https://www.armchina.com/mountain?infoId=161&name=) | Zhouyi AIPU、Compass SDK | 公開[モデルズー](https://github.com/Arm-China/Model_zoo)にYOLOv1-tiny/v2/v3/v4/v5、YOLOX、YOLOv8-segが掲載されています。リファレンスモデルであり、すべてのライセンシーチップの証拠ではありません |
| [VeriSilicon](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico) | Vivante VIPファミリー、Acuity SDK | A311Dやi.MX 8M Plusなど、個別に調達する複数のSoCに関連Vivante NPUファミリーが搭載されています。各チップベンダーが個別に統合を提供します |
| [Imagination Technologies](https://www.imaginationtech.com/products/open-access/) | PowerVR Series3NX AX3146/AX3386/AX3596、旧IMG Series4、Neural Compute SDK/IMG DNN | 小売ボードではなく、ライセンス提供されるNNA IPです。Open Accessでは1/5/10-TOPSのSeries3NX構成を公開。公式NC-SDKプログラムではPP-YOLOE、EfficientNet、HRNetを挙げていますが、すべてのIP構成を網羅してはいません |
| [CEVA](https://www.ceva-ip.com/product/ceva-neupro-studio/) | NeuPro-Nano/NeuPro-M、NeuPro Studio | インポート、量子化、圧縮、グラフコンパイル、シミュレーション、C/C++コード生成を備えるライセンス提供NPU IP |
| [MIPS](https://mips.com/processor-solutions/arc-npx-family/) | ARC NPX6、[MetaWare MX](https://mips.com/processor-solutions/arc-metaware-mx/)。旧Synopsys ARC | スケーラブルNPU IPとNN SDK。2026年6月の[買収](https://mips.com/press-releases/gfmipsarcclose/)後、GlobalFoundriesからMIPSの製品群に移りました。小売デプロイ先ではありません |
| [Cadence](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/neuroweave-sdk.html) | Neo NPU、Tensilica DSP、NeuroWeave SDK | SoCライセンシーが使うコンパイラー/インタープリタースタック。ネットワークと量子化対応はライセンス構成に依存します |
| [Quadric](https://quadric.ai/npu-ip) | Chimera GPNPU IP、SDK | 他社チップにライセンスされるプログラム可能なNPU/DSP型IP。最終的な対象はライセンシーのシリコンです |
| [Expedera](https://www.expedera.com/products-overview/) | Origin NPU IP、ソフトウェアスタック | ライセンス提供のエッジNPUアーキテクチャ。LibreYOLO向けに直接購入できるボードではありません |

Imaginationを見ると、アーキテクチャ名の横にアクセスとライフサイクルを示す理由がわかります。[Open Accessプログラム](https://www.imaginationtech.com/products/open-access/)では、実シリコンで検証済みのSeries3NX構成を3種類、IPの初期ライセンス料なしで提供します。ただし、企業審査に加え、有償サポート/保守と製品ロイヤリティが必要です。[Neural Compute SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/)は、コンパイル、最適化、量子化、ランタイムの各ツールを提供します。公式の[Baidu PaddlePaddleとの連携](https://www.imaginationtech.com/news/imagination-and-baidu-paddlepaddle-create-open-source-machine-learning-library-for-model-zoo/)では、PP-YOLOE検出、EfficientNet分類、HRNetセグメンテーションを取り上げています。これらの例はすべてのSeries3NXまたはSeries4構成を検証するものではありません。また、現行の個別[Series4製品ページ](https://www.imaginationtech.com/product/img-4nx-mc1/)では入手不可と表示されているため、販売部門が確認するまではSeries4を過去の製品として扱ってください。

このIP層が、ファミリー間に一見似ている点が生まれる理由です。成果物の移植性が生まれるわけではありません。VivanteやEthos IPに関連性のあるSoC同士でも、メモリマップ、ドライバー、コンパイラーバージョン、演算子セット、ベンダーランタイムは異なる場合があります。

## 実際にデプロイするもの：成果物ロックインの表

コンパイラーパイプラインの最後にあるファイルを見れば、見かけ上のONNX移植性がどこで終わるかがわかります。名称や拡張子は変わりますが、実務上の原則は変わりません。ネイティブ成果物は通常、別のNPU世代へ移せず、別のSDKリリースで確実に再生成できるとも限りません。そのため、元モデルとキャリブレーションデータを保持してください。

| ベンダースタック | 一般的なコンパイラー入力 | デバイスに配置するもの | 依存する対象 |
|---|---|---|---|
| [Amlogic AMLNN](https://github.com/Amlogic-NN/amlnn-toolkit) | ONNX、TFLite、TorchScript、PT2 | `.adla`。旧A311Dは`.nb` | 対象ID、ADLA世代、AMLNNコンパイラービルド、NNSDK2/ランタイム、ADLAドライバー、BSP |
| [Rockchip RKNN](https://github.com/airockchip/rknn-toolkit2) | ONNX、TFLite、フレームワークのエクスポート | `.rknn` | RKNN Toolkit/Runtimeのバージョン、Rockchip NPUファミリー |
| [Hailo](https://hailo.ai/developer-zone/documentation/) | 中間HARを介したONNXまたはTensorFlow | `.hef` | Hailoアーキテクチャ、コンパイラー/ランタイム世代 |
| [Axelera Voyager](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) | ONNX、対応zoo定義 | アルファ版Pipeline Builderでは`.axm`、`.axe`のパイプラインパッケージ。従来版では`.axmodel`とマニフェスト | Voyager API世代、Metis対象設定 |
| [DEEPX DXNN](https://github.com/DEEPX-AI/dx-all-suite) | ONNX、対応フレームワークの経路 | `.dxnn` | DX-COM/DX-RTのバージョン、DX-M対象 |
| [Qualcomm QAIRT/QNNまたはSNPE](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | ONNX、TFLite、フレームワークモデル | QNNモデルライブラリ/コンテキストバイナリー、またはSNPEの`.dlc` | HTPアーキテクチャ、SoC、バックエンド、ランタイムバージョン |
| [MediaTek NeuroPilot](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) | 変換/量子化済みTFLite | オフラインNeuron Runtime用`.dla`。委譲モード用`.tflite` | NP/MDLA世代、OSイメージ、コンパイラー、ランタイム |
| [TI TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools) | ONNX、TFLite | インポート済み成果物ディレクトリとアプリケーションモデルファイル | TIDLツール/ランタイム、プロセッサー世代 |
| [NXP eIQ](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | 通常はTFLiteまたはONNX | バックエンド固有のVela、TIM-VX、Neutron、Ara出力 | 選択したi.MX/AraアクセラレーターとBSP。「NXP」全般ではありません |
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | TFLite、ONNX、対応フレームワークモデル | 生成されたネットワークライブラリ/コード、ファームウェア資産 | MCU/NPU対象、メモリ設定、ツールバージョン |
| [Infineon PSOC Edge](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) | 整数量子化TFLite/Keras/PyTorch経路 | Ethos-Uコマンドストリーム付きVela最適化TFLiteとアプリケーションファームウェア | E83/E84バリアント、Ethos-U設定、Vela、ModusToolbox、BSP。NNLiteは別対象 |
| [Renesas DRP-AI](https://github.com/renesas-rz/rzv_drp-ai_tvm) | TVMまたはTranslator経由のONNX/TFLite | 複数のバイナリーデータファイルを含むランタイムモデルディレクトリ | RZ/Vデバイス、Translator/ランタイム世代 |
| [Sony IMX500](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | Edge-MDT経路で量子化/パッケージ化したネットワーク | `.rpk`パッケージ | センサーファームウェア、メモリ予算、packerのバージョン |
| [Synaptics](https://developer.synaptics.com/) | TFLite/ONNXと対応インポート形式 | SyNAPでは`.synap`、Torqでは`.vmfb` | SL16xxとSL261xのソフトウェア/ハードウェア世代 |
| [Ambarella CVflow](https://www.ambarella.com/developer/) | パートナー用ツールチェーンの入力 | パートナー限定のデプロイパッケージ。成果物の正確な仕様は公開されていません | Cooper Developer ZoneでCVflow世代、SDK/BSP、カメラパイプラインを確認 |
| [D-Robotics BPU](https://github.com/D-Robotics/rdk_model_zoo) | ONNX、ツールチェーン対応グラフ | X5は`.bin`、現行`rdk_s`は`.hbm`。X3は旧プラットフォーム経路 | BPU世代、リポジトリブランチ、対応するOpenExplorer/ランタイムリリース |
| [AXERA Pulsar2](https://github.com/AXERA-TECH/ax-samples) | ONNX | `.axmodel` | AXチップ対象、Pulsar2、AXEngineのバージョン |
| [SOPHGO TPU-MLIR](https://github.com/sophgo/tpu-mlir) | ONNX、TFLite、TorchScriptなど | BMでは`.bmodel`、CV18xxでは`.cvimodel` | BM/CVチップ対象、SOPHONランタイム世代 |
| [Huawei Ascend CANN](https://www.hiascend.com/en/software/cann) | ONNX、対応フレームワークグラフ | `.om` | Ascendチップ、CANN/ATC、デバイスファームウェア/ドライバー |
| [Cambricon MagicMind](https://github.com/Cambricon/magicmind_cloud) | フレームワークグラフ、交換形式モデル | シリアル化MagicMindエンジン/モデル | MLUアーキテクチャ、Neuware/MagicMindのバージョン |
| [Canaan nncase](https://github.com/kendryte/nncase) | ONNX、TFLite | `.kmodel` | KPU世代、対応するnncase対象プラグイン |
| [Mobilint qb](https://www.mobilint.com/sdk-qb) | ONNX、対応フレームワークモデル | `.mxq` | REGULUS/ARIES対象、qbコンパイラー/ランタイム |
| [Rebellions RBLN](https://docs.rbln.ai/latest/index.html) | PyTorch 2、TensorFlow対応グラフ | `.rbln` | ATOM世代、RBLNコンパイラー/ランタイム |
| [FuriosaAI](https://developer.furiosa.ai/docs/latest/en/) | ONNX/TFLite、対応フレームワーク経路 | Warboyは`.enf`、RNGDは`.fxb` | アクセラレーターとSDKの世代がそれぞれ異なります |
| [Sunplus SNNF](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2004353133) | Acuity対応ネットワーク | `.nb` | SP7350のNPUドライバー、ランタイム、BSP |
| [ESWIN ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) | 現行EsAAC手順ではONNX。他のフレームワークにはエクスポート/変換が必要 | `.model` | 正確なEIC7700ファミリー対象、ENNP/ESSDKリリース |
| [Nuvoton / Ethos-U](https://github.com/OpenNuvoton/NuEdgeWise) | 量子化TFLite | Ethos-U独自演算を含むVela最適化TFLite | M55M1のメモリ計画、Vela、ファームウェア |
| [Himax WiseEye2](https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2) | 完全整数量子化TFLite | モデルフラッシュ上のVela最適化`.tflite`と`output.img`などのボードファームウェア | HX6538/WE2設定、Vela、TFLite Micro、Ethos-Uドライバー、フォールバックカーネル |
| [Analog Devices AI8X](https://github.com/analogdevicesinc/ai8x-synthesis) | PyTorchチェックポイント、ネットワークYAML、入力サンプル | デバイス固有のCソース/ヘッダー、重み、コンパイル済みファームウェア | MAX78000/MAX78002のメモリ/層の制限、合成ツール、MSDKリリース |
| [Realtek AmebaPro2](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | ベンダー変換サービス/入力 | `.nb` | RTL8735Bファームウェア、VoE/NeuralNetwork APIのリリース |
| [Telechips Enlight](https://docs.topst.ai/product/p/ai) | Darknet、TensorFlow、ONNX、PyTorchの経路 | `.enlight`中間形式、コンパイル済みデプロイバンドル | TCC7500 NPU、TC-NN/Enlightのバージョン |
| [T-Head HHB](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | ONNX、対応フレームワークグラフ | `hhb.bm`、生成パラメーター/コード、実行ファイル | TH1520 CSI-NN2/SHLスタック |
| [SigmaStar MI_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | 現行SSU9383CM資料ではMI_IPUランタイムを公開。旧資料ではTensorFlow/Caffe系入力を使用 | 旧`.sim`を`sgsimg.img`へ変換。現行公開成果物は未確認 | SigmaStar IPUとSDKの正確な世代。旧コンパイラーとSSU9383CMの互換性は未検証 |
| [HiSilicon smart vision](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) | ONNX、ATC対応グラフ | `.om` | 正確なスマートビジョンSoC、NNN/SVP-NNNスタック。汎用Ascendへの移植性はありません |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | ONNX、TFLite、Keras、TensorFlow | `.dfp`データフローパッケージ | MXハードウェア設定、ランタイム/コンパイラーリリース |
| [Kneron](https://doc.kneron.com/docs/) | ONNXとツールチェーン設定 | `.nef`。KL730 NEFv2には`.kne`を含められます | ドキュメントにあるコンパイラー対象、チップ世代、ファームウェア、PLUSランタイム。KL830のコンパイルはToolchain 0.33.1で確認できません |
| [SiMa.ai Palette](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) | 対応フレームワークまたはONNXモデル | ModelSDKコンパイル済み`.tar.gz`、記載された実行可能ELF、MPK Toolの`.mpk`パッケージ | MLSoC/Modalix対象、Paletteリリース |
| [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | ONNX、ネットワーク定義 | シリアル化エンジン。一般的な拡張子は`.engine`または`.plan` | GPU/DLAアーキテクチャ、TensorRT、CUDA、多くの場合デバイス |
| [AMD Vitis AI](https://vitisai.docs.amd.com/en/6.2/) | 量子化ONNX/フレームワークグラフ | 旧`.xmodel`。Gen1 NPU snapshot/サブグラフ。Gen2コンパイル済みキャッシュディレクトリまたは本番用`.rai`パッケージ | 正確なNPU世代/IP構成、プラットフォームイメージ、Vitis/Vivado/PetaLinux対応表 |
| [Microchip VectorBlox](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) | 完全量子化INT8 TFLite | `.vnnx`、`.hex`、`.ucomp`、対応ファームウェア/FPGA設計 | CoreVectorBlox構成、PolarFire SoC Video Kit、SDK 3.1、ビットストリーム、メモリ配置 |
| [Intel OpenVINO](https://docs.openvino.ai/) | フレームワークモデルまたはONNX | IRの`.xml`と`.bin`、またはデバイス固有のコンパイル済みキャッシュ | IRは再利用可能。コンパイル済みキャッシュはデバイス、ドライバー、OpenVINOに依存 |
| [旧Google Edge TPU](https://coral.ai/docs/edgetpu/models-intro/) | 完全INT8 TFLite | Edge TPUコンパイル済み`_edgetpu.tflite` | Edge TPU Compiler/Runtime、対応量子化演算子。新しいCoral NPU IPとは別系統 |
| [Google Coral NPU IP](https://github.com/google-coral/coralnpu) | 現行の公開IPプロジェクト用C/C++例 | RTL/ハードウェアシミュレーションおよび将来のSoC統合向けELF例。公開YOLOコンパイラー成果物はありません | 正確なライセンス/統合済みSoC実装、開発中のオープンソースツールチェーン |
| [Lattice sensAI](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | 対応ネットワーク記述 | FPGAビットストリーム、アクセラレーター設定、重み | FPGAファミリー、sensAI IPモード、メモリ実装 |
| [Imagination NC-SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) | Caffe、TensorFlow、ONNX | 顧客向けのコンパイル済み成果物ファイル名は非公開 | ライセンス対象NNA構成、SoC統合、NC-SDK/DDK、ライセンシーランタイム |

この表は拡張子の一覧だけではありません。コンパイラーバージョン、対象ID、ドライバー、ファームウェア、BSPは、モデルの再現可能な識別情報の一部です。今後LibreYOLOにエクスポーターを追加する場合は、成果物ごとに機械可読なマニフェストへこれらを記録してください。

## ツールへのアクセスとライフサイクルの兆候

ハードウェアの入手性とコンパイラーへのアクセスは別問題です。ボードを簡単に購入できても現行コンパイラーには顧客契約が必要な場合があり、SDKが公開されていても対象シリコンの入手が難しい場合があります。以下は一次資料で確認できる具体的な兆候であり、普遍的な製品寿命ランキングではありません。

| プラットフォーム | ドキュメントにある兆候 | 実務上の意味 |
|---|---|---|
| Amlogic ADLA | Apache-2.0の公開リポジトリとダウンロード可能なwheel。対応ボードのドライバー/BSPにはAmlogicまたはボードベンダーの支援が必要な場合があります | コンパイラーの評価は容易ですが、バイナリーの再配布と完全な互換性一覧は確認が必要です |
| MediaTek Genio | 公開IoT AI Hub、Yoctoガイド。NP8オールインワンツールとAndroid資料は直接取引先/NDA向け | モデルの証拠は監査できますが、独自モデルの自動化にはパートナー連携が必要な場合があります |
| Hailo | 公開モデルズーとランタイム。Dataflow CompilerはDeveloper Zoneから配布。公開`hailo-camera-apps`は2026年5月3日にアーカイブ | 幅広いモデルの調査は公開されていますが、再現可能なコンパイルには正しいアカウント/バージョンが必要です。アーカイブはHailo-15のEOLを示しません。現行のビジョンプロセッサー用アプリケーションパッケージは要確認です |
| Axelera AI | 公開ドキュメントと[公開Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk)。顧客サポートにはアカウントが必要 | セルフサービス可能な希少なアクセラレーターSDKですが、製品サポートと調達は商用です |
| Qualcomm Dragonwing | [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program)ではIQ-9075がSampling、2038年まで、QCS6490が2036年7月までと表示。一方、IQ-9075の製品ページではActiveと表示 | 産業用の計画には強い兆候ですが、一次資料間の状況差はSKU単位の確認が必要です。期間中に1つのAI SDKのABIが安定する保証はありません |
| Rebellions | 現行[対応表](https://docs.rbln.ai/latest/supports/version_matrix.html)ではCA02/CA12がEoL、CA22/CA25がActive。CA21は一覧にありません | ATOMファミリー内でも調達とSDK互換性はカードごとに異なります |
| NVIDIA Jetson | [公式モジュールライフサイクル表](https://developer.nvidia.com/embedded/lifecycle)ではOrinモジュールは2032年1月まで、AGX Orin Industrialは2033年7月まで。開発キットにはライフサイクルの確約がありません | 本番システムは開発キットの入手性ではなくモジュールを前提に設計してください |
| Raspberry PiのSony IMX500 | [AI Camera製品概要](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf)では少なくとも2028年1月まで生産と記載 | そのカメラモジュールの最低期間です。すべてのIMX500製品には当てはまりません |
| Amlogic A311Y3 | [2026年発売ページ](https://www.amlogic.com/News/index248.html)で最低10年間の製品寿命保証を公表 | 新しいプラットフォームの有望な兆候ですが、製品計画では契約とSKUの詳細確認が必要です |
| Google Coral | 旧Edge TPUとPyCoralのリポジトリはアーカイブ/読み取り専用。別系統の[Coral NPU IP](https://github.com/google-coral/coralnpu)リポジトリは更新中 | 旧ソフトウェアの保守リスクを示しますが、それだけで正式なハードウェアEOL通知とはなりません。別プロジェクトのオープンソースIPの寿命も示しません |

## TOPSが購入ガイドにならない理由

ピークTOPSは1社内、同じアーキテクチャ内なら参考になりますが、次のすべてが一致しない限り、メーカー間の比較は通常成り立ちません。

- INT8、INT4、FP16、混合モードなど、数値精度
- 1回の積和演算を1演算または2演算のどちらと数えるか
- 密演算か構造化スパース演算か
- 入力解像度、バッチサイズ、モデルグラフ
- 量子化後の精度
- NPU単体のレイテンシか、アプリケーションパイプライン全体か
- メモリ転送とホスト側フォールバック
- 温度状態、バースト時ではなく持続時の動作

検出器はパイプラインです。画像取得、リサイズ/レターボックス、正規化、デバイス転送、ニューラル推論、デコード、NMS、座標スケーリング、アプリケーション処理を含みます。ベンダーが公開するのは、多くの場合このうちニューラル推論の部分だけです。[MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/)は、単独のマーケティング演算量を比較するのではなく、シナリオ、品質目標、システムレベルの測定規則を定義しているため参考になります。

YOLOのデプロイでは、最低限、信頼できるベンチマーク記録に次を含めてください。

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

これらの情報がなければ、2つのFPS値は通常、異なる内容を測定しています。

## コンピュータビジョン用NPUの選び方

製品ページに最大の数字があるものではなく、次の順番で選んでください。

1. **グラフ互換性を証明します。**似た名前のモデルズーチェックポイントではなく、エクスポートした正確なモデルをコンパイルします。
2. **精度を測定します。**キャリブレーションと量子化の後、実際のタスクデータセットでコンパイル済み成果物を検証します。
3. **エンドツーエンドで測定します。**入力変換、転送、デコード、NMSを含めます。
4. **ソフトウェアへのアクセスを確認します。**コンパイラーが公開、登録制、商用契約のみのいずれかを調べます。
5. **バージョン対応を固定します。**コンパイラー、ランタイム、ドライバー、ファームウェア、対象IDを記録します。
6. **OSの実情を確認します。**Android、Debian、Yocto、Buildroot、RTOS、Windowsへの対応は相互に代替できません。
7. **供給とライフサイクルを確認します。**モジュール、ドライバー、長期サポートが不確かなチップは、技術的に優れていても本番採用に適しません。
8. **その後で価格、電力、性能を比較します。**両方の対象で同じモデルが正しく動いた後に、これらの数値が意味を持ちます。

### 用途別の実践的な出発点

| 用途 | 最初に評価したいプラットフォーム | 理由 |
|---|---|---|
| Raspberry Pi専用アクセラレーター | Hailo-8L/8、Sony IMX500 | Raspberry Pi公式製品、ソフトウェアイメージ、具体的な開発手順 |
| 一般的なLinux PCIe、M.2、USB増設 | DEEPX、MemryX、Hailo、Axelera製品 | 入手可能なアクセラレーターのフォームファクター。ホスト/ドライバー互換性は要確認 |
| 低価格Linux SBCまたはカメラ | Rockchip RK3588/RK3576、対応ボード/BSPを入手できるAmlogic ADLA、AXERA、Canaan K230、Sunplus SP7350 | 統合メディアパイプラインと公開モデル/コンパイラー資料。A311D2 NPUにはVIM4 V13A以降が必要な点を確認 |
| モバイルと大量出荷のAndroid | Qualcomm QNN/AI Hub、LiteRT、Apple Core ML | 大規模な導入基盤と保守されるモバイルランタイム |
| 産業用Linuxとロボティクス | TI TDA4/AM6xA、NXP i.MX、Renesas RZ/V、Hailo | 長寿命の組み込み製品、カメラ/IOエコシステム |
| 小型エンドポイントまたはMCU | STM32N6x7、Infineon PSOC Edge、Himax WiseEye2、Analog Devices MAX7800x、Nuvoton M55M1、Alif/Arm Ethos-U、GAP9、Syntiant | 厳しい電力/メモリ制約向けの専用ツール。モデルサイズにも制約があります |
| 構成可能なFPGAビジョン | Microchip VectorBlox、Lattice sensAI、AMD Vitis AI対象 | 柔軟なハードウェアパイプライン。ただしアクセラレーター構成、ビットストリーム、モデルコンパイラーは1つのバージョン管理システムです |
| 高スループットPCIeビジョン | Axelera Metis、Hailo、DEEPX、Mobilint、MemryX、Rebellions、SiMa.ai | 専用アクセラレーター製品とマルチストリームを想定した製品群 |
| 中国中心の製品エコシステム | Rockchip、AXERA、D-Robotics、SOPHGO、Huawei Ascend、HiSilicon、Cambricon | 地域のボード、ツールチェーン、モデル例が充実 |

この表は調査候補の一覧であり、性能ランキングではありません。調達、地域での入手性、正確なモデルによって順序が変わる場合があります。

## LibreYOLOにとっての意味

拡張しやすい設計は、互いに無関係なエクスポーターを何十個も用意することではありません。厳密な交換契約1つと、小規模なベンダー用コンパイラー/ランタイムアダプターで構成します。

```text
LibreYOLO model
    -> deterministic static ONNX
    -> VendorCompiler.compile(model, target, calibration, precision)
    -> native artifact + manifest
    -> VendorRuntime.load() / infer()
    -> shared task-specific decode and Results objects
```

すべてのネイティブ成果物には、次の項目を含むサイドカーマニフェストが必要です。

- ベンダー、チップ対象、ボード対象
- コンパイラー、ランタイム、ドライバー、ファームウェアのバージョン
- 元チェックポイントとONNXのチェックサム
- 入力名、形状、レイアウト、色空間、正規化、dtype
- 量子化精度、公開されている場合はスケール、キャリブレーションデータのハッシュ
- 出力テンソル名、形状、意味
- 検出タスク、クラス名、デコード/NMSがチップ上かホスト側か
- 記録した数値パリティとタスク精度の結果

LibreYOLOは、移植可能な[ONNXエクスポート](/docs/export/onnx)、[量子化ツール](/docs/export/quantization)、対象を意図的に絞った直接[RKNNエクスポート](/docs/export/rknn)、外部コンパイラーを使う[Hailo手順](/docs/export/hailo)をすでに提供しています。このガイドの基準では、Amlogic ADLAは次の有力な統合候補です。ツールキットは公開され、モデル対応はLibreYOLOと大きく重なり、旧A311D経路を明確に分けられます。

Amlogic以降の優先順位は、コンパイラーの入手性だけでなく、ユーザーが使うハードウェアと精度検証の実施可能性に基づいて決めるべきです。DEEPX、Sony IMX500、D-Robotics、AXERA、SOPHGO、Qualcommは技術的に魅力があります。TI、NXP、ST、Renesasは、産業分野のパートナーからボードと長期CIアクセスを得られる場合に特に有用です。

## ウォッチリスト：公開統合契約がない実在シリコン

企業を完全に除外すると市場一覧が誤解を招く場合があります。「対応済み」として含めると、さらに問題になることがあります。次のベンダーは関連シリコンを販売、サンプリング、または発表していますが、公開資料だけでは、LibreYOLOバックエンドに適した再現可能な現行コンパイラー、成果物、モデル名付きのデプロイ手順を確認できません。

| 企業 | 確認できること | ウォッチリストに留める理由 |
|---|---|---|
| [Samsung](https://semiconductor.samsung.com/processor/automotive-processor/exynos-auto-v920/) | Exynos Auto V920には最大23.1 TOPSを公称するデュアルコアNPUがあります | Samsungは[Neural SDKをサードパーティ開発者に提供しなくなった](https://developer.samsung.com/neural/overview.html)としています。Samsung ONE/Circleがあることは、Exynos NPUへのアクセスを証明しません |
| [Novatek](https://www.novatek.com.tw/en-global/Milestone/aboutus_milestones) | 現行の企業沿革にEdge AI、Edge Vision/Imaging AI SoCが挙げられています。2020年の沿革にはMobileNet、SSD、YOLOv3の記載がありました | 現行の型番/コンパイラー/成果物/演算子/モデル対応表は公開されていません。過去のYOLOv3は現行SDKの証拠ではありません |
| [Nextchip](https://www.nextchip.com/en/adas/adas.php?idx=5) | aiMotive aiWare NPU搭載のAPACHE5/NVS2900、現行[APACHE6/NVS3000](https://www.nextchip.com/en/adas/adas.php?idx=7)車載プロセッサー。APACHE6ページには現在12 TOPSと8 TOPSの両方が記載されています | aiWare Studioとランタイムはありますが、成果物、量子化ガイド、演算子一覧、YOLO対応表は確認できません。相反するベンダー数値を暗黙にどちらかへ決めてはなりません |
| [Black Sesame Technologies](https://bst.ai/en.html) | Huashan A1000/A2000、Wudang Cシリーズの車載/エッジAIシリコン | 公開製品情報はありますが、セルフサービスのコンパイラー、成果物仕様、再現可能なモデルズーはありません |
| [Ingenic](https://en.ingenic.com.cn/products-detail/id-19.html) | 低ビットNPUを公称するT41/T40カメラSoC。Magik AIはPTQ/QATとグラフコンパイルを説明 | 現行のダウンロード可能なコンパイラー、成果物仕様、ベンダーYOLO一覧は確認できませんでした |
| [Fullhan](https://fullhan.com/en/index.php?a=type&c=article&tid=9) | 複数のIPC SoCが0.5〜2 TOPSのNPUを公称し、[MC6880 NVRファミリー](https://fullhan.com/en/index.php?a=type&c=article&tid=48)は4 TOPSを公称 | 独立して再現するのに十分な公開NNコンパイラー/ランタイム/フレームワーク/モデル資料がありません |
| [Goke Microelectronics](http://www.gokemicro.com/News/info.aspx?itemid=437) | GK7606V1/GK7206V1/GK7203V1スマートカメラ部品のファーストパーティページでは統合NPU性能を公称。ただし公開時点で旧式のHTTPのみ | 公開コンバーター、ランタイム仕様、モデル名付き対応表はありません。OEM向けの販売経路です |
| [Chengheng Micro](https://en.chenghengmicro.com/) | CH37は64 INT8 TOPSとFP16/FP32/FP64モードを公称。会社の[2026年沿革](https://chenghengmicro.com/about.html)では小規模量産開始と記載 | 公開SDK、コンパイラー仕様、モデル名付き検証は見つかりませんでした。再現可能なデプロイ環境というより、商用化の初期段階でセルフサービス不可のプラットフォームとして扱います |
| [Bouffalo Lab](https://github.com/bouffalolab/bouffalo_sdk) | BL808にはBLAI-100 NPUがあり、現行の公式SDKリポジトリもあります | 維持されているSDKでは現行BLAI変換/例の経路が確認できません。残っている手順は旧式またはコミュニティ資料です |

このウォッチリストは証拠に基づいています。現行ツールチェーンまたは評価手順、対象固有の成果物仕様、エンドツーエンド手順を伴うモデル名付き例を公開すれば、ベンダーはデプロイ可能な一覧へ移せます。

## 意図的に主張していないこと

この記事は、列挙したすべてのモデルが上流リポジトリのまま動くとは主張していません。ベンダーのzooモデルは、変更を加えられていたり、異なる出力ノードで分割されていたり、カスタム後処理と組み合わされていたりすることがよくあります。

また、以下を対応済みの主張にはしていません。

- コンパイラーがONNXをインポートする
- チップにAndroid NNAPIドライバーがある
- 販売店がボードを「YOLO対応」と紹介している
- コミュニティリポジトリで特定モデルのフォークが動く
- エラーなしでモデルをコンパイルできる
- ベンダーが精度結果なしにNPU単体FPSを報告している

Google Tensorスマートフォンのアクセラレーターや、OEM専用カスタムASICも実在します。しかしGoogleは、Tensorを上記プラットフォームのような一般開発者向けセルフサービスコンパイラー対象として提供していません。会社の存在、NPUブロック図、Androidアクセラレーションだけでは、LibreYOLOとの統合を主張する根拠になりません。

同様に、Google TPU、AWS Inferentia/Trainium、Microsoft Maia、データセンター専用AIカードなどのクラウドアクセラレーターは対象外です。このガイドは、コンピュータビジョン開発者がエッジで現実的にデプロイできるハードウェアを扱います。

モデルライセンスは別の層です。チップベンダーがYOLOの変換レシピを公開していても、上流の重み、学習コード、コンパイル済み派生物を使用/再配布する権利が得られるわけではありません。対象製品について、ハードウェア対応、SDKライセンス、モデルライセンスをそれぞれ確認する必要があります。

## 調査方法と訂正

各企業について、次の4つの一次資料を調べました。

1. シリコンを特定する現行製品ページまたはデータシート
2. 実際のデプロイ経路を説明するコンパイラー/ランタイムドキュメント
3. ビジョンモデル名が記載されたモデルズー、互換性表、エンドツーエンドチュートリアル
4. 開発者が実際にツールを入手できるかを示すライフサイクルまたはアクセス情報

ファーストパーティのGitHub組織もベンダー資料に含めています。チップ企業自身が同じ内容を公開していない場合は、ボードベンダーの資料をエコシステム資料として示しています。サードパーティの性能主張は比較表から除外しました。

完成した草稿を、企業群別の再調査と表を横断した確認で再監査しました。対象範囲、成果物名、精度、モデルと後処理の証拠、発表済み/出荷中の区別、ライフサイクル表示、ベンチマークの分母、企業数を再確認しました。一次資料2つの内容が食い違う場合は、都合のよい方を黙って選ばず、食い違いを明記しています。

この市場は急速に変化します。対象企業の担当者で、チップ、SDK、モデル一覧、アクセス状況に誤りを見つけた場合は、正確な公開ドキュメントのURLとバージョンをLibreYOLOへお知らせください。一次資料で裏付けられる訂正で本文を更新します。ドキュメントにないマーケティング上の主張は追加しません。

## FAQ

### Edge AIのNPU企業は何社ありますか？

製品ベンダー、IPライセンサー、スマートセンサー、GPU、非公開の車載ASICが重なり合うため、普遍的な企業数はありません。このガイドはすべてを網羅するものではありませんが、2026年8月時点で、関連するEdge AIチップ、アクセラレータープラットフォーム、ライセンス提供されるNPU IP、または有望なウォッチリスト対象シリコンを持つ企業とプラットフォームエコシステムを69件追跡しています。

### NPUとは何ですか？

NPU（ニューラル処理ユニット）は、畳み込みや行列乗算など、ニューラルネットワークの演算に特化したハードウェアです。ベンダーによってNPU、AIPU、BPU、KPU、DLA、HTP、TPU、MLAなどの名称が使われ、コンパイル済みモデルは一般に企業間で移植できません。

### YOLOモデルを公式にサポートするNPU企業はどこですか？

Amlogic、Hailo、Axelera AI、Rockchip、Qualcomm、MediaTek、DEEPX、D-Robotics、AXERA、SOPHGO、Huawei、Cambricon、Raspberry Piの公式AI Cameraリポジトリ経由のSony IMX500、STMicroelectronics、Renesas、Lattice、Himax、Microchipなどは、少なくとも1世代のYOLOについて、ファーストパーティのモデルズー項目、チュートリアル、または検証済み結果を公開しています。対応する世代、タスク、チップの対象、SDKバージョンを個別に確認する必要があります。

### どのONNXモデルでも、どのNPUでも実行できますか？

いいえ。ONNXは交換形式であり、ハードウェア互換性を保証するものではありません。NPUコンパイラーは、グラフ内のすべての演算子、テンソル形状、データ型をサポートする必要があります。静的形状、グラフ分割、カスタム演算、CPUフォールバックはよくあります。

### YOLOに最適なNPUはどれですか？

万能な勝者はいません。Hailo、Rockchip、Axelera AI、DEEPX、Amlogicは、公開資料で確認できるYOLO対応が充実しており、Qualcommは特に幅広いデバイスに展開されています。量子化後の精度、パイプライン全体のレイテンシ、持続消費電力、価格、入手性、SDKへのアクセス、OS対応で選んでください。

### NPUのTOPS値を直接比較できないのはなぜですか？

ベンダーによって、数値精度、スパース演算、積和演算、ピーク稼働率の前提が異なります。TOPSにはメモリ転送、未対応演算子、前処理、後処理、ホスト側のオーバーヘッドが含まれません。同じモデル、数値精度、解像度、タスク精度、パイプライン全体で比較してください。

### NPUのキャリブレーションとは何ですか？

キャリブレーションでは、代表的なデプロイ画像（通常はラベルなし）をモデルに通し、コンパイラーがINT8などの低精度形式向けにアクティベーション範囲を推定します。ランダムな画像や代表性のない画像でもコンパイル成果物が生成されることはありますが、タスク精度を損なう可能性があります。

### LibreYOLOはエッジNPUをサポートしていますか？

LibreYOLOはONNXや複数のランタイム形式にエクスポートでき、一部のRockchipモデル向けに直接RKNNコンパイラーパスを備え、外部Hailoコンパイル手順も説明しています。それ以外のベンダー固有成果物にはベンダーSDKが必要です。ハードウェア上でコンパイル、検証、実行するまでは、統合候補として説明してください。
