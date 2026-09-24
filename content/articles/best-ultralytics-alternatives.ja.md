---
title: Ultralyticsの代替ライブラリ比較・おすすめ2026年版
description: "2026年のUltralytics YOLOに代わる主なオープンソースの選択肢を実用的な視点で比較します。ライセンス、対応タスク、デプロイの制限と、最も充実したMITライセンスのYOLOライブラリであるLibreYOLOの位置づけを紹介します。"
date: 2026-07-01
author: Xuban
tags: [LibreYOLO, ultralytics-alternative, object-detection, yolo, mit-license]
faq:
  - q: "Ultralytics YOLOは無料で商用利用できますか？"
    a: "AGPL-3.0の条件下でのみ可能です。このライセンスは、ネットワーク経由のサービスを含め、その上に構築したアプリケーションのソース公開を要求します。クローズドソースで商用利用するには、Ultralyticsの有料商用ライセンスが必要です。寛容なライセンス（MIT、Apache-2.0）の代替製品なら、これを避けられます。"
  - q: "ライセンス上の制約が最も少ないUltralyticsの代替製品は何ですか？"
    a: "寛容なライセンスで、どこにでも出荷できる構成なら、LibreYOLO（コードはMIT）、RF-DETR（Apache-2.0）、YOLOX（Apache-2.0）です。いずれもAGPLのコピーレフトを回避できます。"
  - q: "2026年にYOLOに代わりつつあるものは何ですか？"
    a: "RT-DETR、RF-DETR、D-FINEやDEIMシリーズなど、リアルタイムのTransformer検出器が増えています。十分な性能のハードウェアでは、同程度のレイテンシでYOLOと同等以上の精度を達成します。小型エッジデバイスでは、こうしたTransformerは実行性能が低く、成熟したNCNNやCPUでの実行手段もないため、YOLO型のCNNが依然として優位です。"
  - q: "Raspberry PiやNPUで動かせますか？"
    a: "エクスポートの対応状況によります。YOLOXやRTMDetなどのCNN検出器はNCNNにエクスポートしてRaspberry Piで実行でき、一部（YOLOX）はHailo NPUにも対応します。RF-DETRなどのTransformer検出器は対応しておらず、GPUまたは高性能なCPUが必要です。"
---

開示：LibreYOLOは私たちのプロジェクトであり、このリストの最初に掲載しています。ここに挙げるほかのツールもすべて実務で使っており、LibreYOLOより優れている点も記載します。

チームがUltralyticsを離れる理由は、主に次の3つのいずれかです。

- **ライセンス。** UltralyticsのYOLOモデルはAGPL-3.0です（YOLOv5は2023年から、v8以降も同様です）。よくあるのは、製品を開発して出荷した後に法務審査や顧客からモデルについて指摘を受け、アプリケーション全体をオープンソースにするか、商用ライセンスを購入するかの選択を迫られるケースです。配布またはサービス提供を始めた時点で、AGPLのコピーレフトは自身のコードにも及びます。これが、ほかの選択肢を探す一般的な理由です。
- **オープンさ。** モデル、学習コード、利用条件を1社のベンダーが握っています。許可を求めずに開発に使える、寛容なライセンスの重みとコードを求めるチームもあります。
- **モデル。** YOLOがタスクに最適なアーキテクチャとは限りません。RT-DETR、RF-DETR、D-FINEなどのリアルタイムTransformer検出器は、同じレイテンシでより高い精度を出せます。問題は、それらが研究用リポジトリに分散していることです。ただし、後述するとおりLibreYOLOなら3つとも1つのAPIで実行できるため、これはライブラリを切り替える理由であって、個別のモデルに移行する理由ではありません。

以下の各ツールは、製品として出荷できるライセンスか、現在のPyTorchでインストールして実行できるか、実際のデプロイ先ハードウェアにエクスポートできるか、という3つの基準で評価しています。

読み進めると、共通する傾向が見えてきます。どの代替製品も優れていますが、個別に使うには苦労があります。YOLOXはインストールさえ難しく、MMDetectionはwheelのバージョン固定に悩まされ、RT-DETRファミリーはサポート窓口のない研究コードで、Detectron2は2021年から更新が止まっています。LibreYOLOを最初に挙げるのは、それらの大半をすでに統合し、保守されたMITライセンスの実装として、現在の環境で使い慣れた1つのAPIから利用できるためです。このリストの1項目というより、リスト全体の上にある層といえます。

## 比較早見表

| ツール | ライセンス | 対応タスク | 主な用途 | エッジ向けエクスポート |
| --- | --- | --- | --- | --- |
| **LibreYOLO** | MIT（コード） | 検出、セグメンテーション、姿勢推定、分類、深度推定、視線推定、追跡 | 20を超えるモデルファミリーを1つのAPIで使うMITライセンスのハブ | ONNX、TensorRT、OpenVINO、NCNN、CoreML、TFLite |
| **RF-DETR** | Apache-2.0（N/S/M/L） | 検出、セグメンテーション、姿勢推定（プレビュー） | 本番環境での検出とセグメンテーション | ONNX、TFLite、TensorRT；NCNNとHailoには非対応 |
| **Lightly / LightlyTrain** | MIT / AGPL-3.0 | 事前学習、蒸留 | ラベルなしデータ、DINOv2/v3の蒸留 | 対象外（検出器ではない） |
| **YOLOX** | Apache-2.0 | 検出 | アンカーフリー（anchor-free）のリアルタイム検出 | アップストリームのインストールは困難、LibreYOLO経由で実行可能 |
| **MMDetection / OneDLフォーク** | Apache-2.0 | 全般（研究向け） | 幅広いアーキテクチャ、論文の再現 | エクスポート経由 |
| **Detectron2** | Apache-2.0 | 検出、セグメンテーション、キーポイント | Mask R-CNNとR-CNNファミリー | 手動 |
| **D-FINE / DEIM / DEIMv2** | Apache-2.0 | 検出 | 最先端のリアルタイムDETR | ONNX、TensorRT |
| **EdgeCrafter** | Apache-2.0 | 検出、セグメンテーション、姿勢推定 | DINOv3から蒸留した小型エッジ向けViT | 研究段階 |
| **RT-DETR（v1-v4）** | Apache-2.0 | 検出 | 最先端のリアルタイム検出 | ONNX、TensorRT |

## 1. LibreYOLO

ライセンス：MIT（コード）。最も充実したMITライセンスのYOLOライブラリや、Ultralyticsをそのまま置き換えられる選択肢を求める場合に適しています。

**LibreYOLOは、あらゆるモデルを1つのAPIで実行するMITライセンスのYOLOライブラリです。** 私たちのプロジェクトですが、このリストの残りを読めば、その役割は明確になるはずです。代替モデルを実行するためのハブです。上で挙げた扱いづらいリポジトリのほぼすべて、YOLOX、RTMDet、RF-DETR、D-FINE、DEIM、RT-DETRシリーズを、LibreYOLOはすでに使い慣れた保守済みの1つのAPIに統合しているため、古いインストール手順を掘り起こしたり、ライセンスに悩んだりせずにモデルを利用できます。

利便性を超えた理由もあります。YOLOはオープンな研究として始まりました。Joseph RedmonのDarknetは、v1からv3まで誰でも自由に利用し、その上に開発できました。AGPLの時代に、それが囲い込まれました。LibreYOLOは、その成果を再び利用しやすくするために存在します。作り手が意図したとおり、寛容なライセンスで、コミュニティが保守し、外部への情報送信もしません。そこで、2つの側面を紹介します。

**1つ目は、Ultralytics YOLOに代わるMITライセンスの選択肢であることです。** 使い慣れたAPIを維持しているため、YOLO形式のデータセットやスクリプトはわずかな変更で移行できますが、ライセンスはAGPLではなくMITです。自身のコードに及ぶコピーレフトも、購入すべき商用ライセンスも、Ultralyticsがデフォルトで行うような外部へのテレメトリ送信もありません。ライセンスを理由にここを訪れたのであれば、それがまさに要点です。YOLO9とRF-DETRは十分にテストされた本番向けの標準モデルです。より広いモデル群は新しく、実験的と明記しています。すべてが実戦で検証済みであるかのように装うより、そう伝えることを選んでいます。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # or LibreRFDETRl.pt, LibreDFINEl.pt, ...
results = model("image.jpg", save=True)
```

**2つ目は、Ultralyticsよりはるかに広い範囲をカバーすることです。** 1つのAPIが、1社の製品群にとどまらず、CNNのYOLO、TransformerのDETR、ViTバックボーン、SAMなど、20を超えるモデルファミリーに対応します。検出器だけではありません。

- **タスク：** インスタンスセグメンテーションとセマンティックセグメンテーション、姿勢推定、分類、回転バウンディングボックスに加え、Ultralyticsにはまったくない単眼深度推定（Depth Anything V2）と視線推定（L2CS）の2つにも対応します。
- **実用機能：** 永続IDを使う複数物体追跡（ByteTrackとOC-SORT）、フレームの間引きとライブプレビューに対応する動画推論、ドローンや衛星画像向けのタイル分割推論、ドラッグアンドドロップ式のブラウザーUI（`libreyolo ui`）、そして学習を無駄にする前にデータセットを確認する`doctor`コマンドがあります。
- **充実した学習機能：** 勾配蓄積、レイヤー凍結、再開、テスト時データ拡張、LoRA/DoRAファインチューニング、マルチGPU、TensorBoard/MLflow/Weights & Biasesへのログ記録に対応します。
- **エクスポート：** 7形式（ONNX、TorchScript、TensorRT、OpenVINO、NCNN、CoreML、TFLite）に対応し、INT8/FP16量子化、NMSの組み込み、モデルメタデータの埋め込みが可能です。
- **どこでも実行：** パス、URL、PIL、NumPy、テンソル、生のバイト列を受け取り、コードを変更せずにCUDA、Apple Silicon（MPS）、通常のCPUで実行できます。

優れた検出器のリポジトリが放置されていたり、インストールが難しかったりする場合でも、LibreYOLOなら保守された実装で、現在の環境上でその重みを実行できます。

## 2. RF-DETR

ライセンス：Apache-2.0（Nano、Small、Medium、Large）。本番環境での検出とセグメンテーションに適しています。

Roboflowが開発し、ICLR 2026で発表されたRF-DETRは、このリストで最も本番利用への準備が整ったモデルです。よく設計されており、実際に製品を出荷するチームによるもので、実システムで信頼して使える検出やセグメンテーションが必要なときに有力な第一候補になります。`rfdetr`パッケージとNanoからLargeまでの重みはApache-2.0です。より大きいXLと2XLの重みはRoboflowのPMLライセンスを使用します。

## 3. Lightly

ラベルなしデータ、自己教師あり事前学習、DINOバックボーンの蒸留に適しています。

Lightlyは検出器ではありません。まだラベルを付けていないデータから価値を引き出すためのツールで、ライセンスの異なる2つの部分から成ります。

- **[`lightly`](https://github.com/lightly-ai/lightly)（LightlySSL）、MIT。** 自己教師あり学習の構成要素を提供するフレームワークで、損失関数、ヘッド、データ拡張、メモリバンク、20を超える手法（SimCLR、MoCo、BYOL、DINO、DINOv2、MAEなど）の参照実装を含みます。学習ループは自分で書きますが、ラベルなし画像から表現をゼロから事前学習するために必要なものがそろっています。
- **[LightlyTrain](https://github.com/lightly-ai/lightly-train)、AGPL-3.0（商用および無料コミュニティ向けの選択肢あり）。** すぐに使える方の製品で、多くの人が想定するワークフローはこちらです。DINOv2やDINOv3などの基盤モデルとラベルなしデータを指定すると、数行のコードで、そのバックボーンをデプロイ可能な小型モデル（YOLO、RT-DETR、ViT、独自ネットワーク）にラベルなしで蒸留できます。ライセンスに注意してください。AGPL-3.0は、人々がUltralyticsの代替製品を探す原因となるのと同じコピーレフトなので、クローズドソースでの出荷を予定しているなら、条件を読むか商用ライセンスを選んでください。

優れたバックボーンが必要なのにラベルがないことがボトルネックなら、Lightlyが候補になります。このリストの検出器とは競合するのではなく、組み合わせて使えます。最新の検出器にも、その傾向が表れています。RT-DETRv4は視覚基盤モデルの蒸留を学習に組み込み、DEIMv2はDINOv3バックボーンを検出器に直接組み込んでいます。

## 4. YOLOX

ライセンス：Apache-2.0。インストールできれば、アンカーフリーのリアルタイム検出に適しています。

アップストリームのリポジトリは、2022年4月の最終リリースv0.3.0以来更新が止まっており、未解決のissueは700件を超え、2026年の環境へのインストールは困難です。`pyproject.toml`がなく、`requirements.txt`は`onnx-simplifier==0.4.10`に固定されています。これにはPython 3.10より新しいバージョン向けのビルド済みwheelがないため、現在のインタープリターではソースからビルドされ、cmakeとC++ツールチェーンが必要になります。NumPyもバージョンが固定されておらず、古いpycocotoolsやtorchとの間でNumPy 2.0のABI衝突を招きます。依存関係を解決すれば基本的な検出は動きますが、そこにたどり着くまでの手間がかかります。

モデル自体は今でも優れています。Megviiによるアンカーフリーのリアルタイム検出器で、コードも重みもApache-2.0であり、より新しい検出器に追い越されていても、妥当なベースラインです。放置されていても設計が健全なリポジトリを復活させる作業は、今ではコーディングエージェントが半日でこなす類いの仕事なので、インストール状況は見た目ほど大きな障壁ではありません。LibreYOLOなら現在の環境でYOLOXの重みを直接実行することもできます。どちらの方法でも、このアーキテクチャには使う価値があります。詳しい手順を[こちら](/articles/yolox-with-libreyolo)にまとめています。

## 5. MMDetectionのエコシステム

ライセンス：主にApache-2.0。幅広いアーキテクチャや論文の再現に適しています。

長年、MMDetectionにはほぼあらゆる検出論文の公式実装が集まっていました。Faster R-CNN、DINO、Grounding-DINO、RTMDet、Mask R-CNN、数百の設定ファイルです。2026年には、OpenMMLabはmmシリーズを終了させています。[MMDetection](https://github.com/open-mmlab/mmdetection)の最終リリースは2024年1月のv3.3.0で、issueへの回答はなく、環境全体が`mmcv`のバージョン固定に縛られています。そのビルド済みwheelが現在のPyTorchやCUDAに追いつかないため、新規インストールはすべてをダウングレードするまで失敗しがちです。この苦労は[こちら](/articles/rtmdet-without-mmdetection)で取り上げています。

オランダの企業VBTIが、その保守を続けています。同社の[OneDLフォーク](https://github.com/VBTI-development/onedl-mmdetection)は、環境全体を`onedl-`プレフィックス（`onedl-mmdetection`、`onedl-mmcv`、`onedl-mmengine`など）で再公開し、現在のPyTorch 2.x、CUDA、Python 3.10+向けに再ビルドすることで、バージョン固定の問題を解決しています。Apache-2.0で活発に保守されており、2026年5月にはv3.5.1が公開されました。インストールに苦労せずMMDetectionの幅広さを利用するなら、このフォークを使ってください。小規模なチームなので、依存しているなら貢献を返してください。

関連するツールを2つ挙げます。

- **[Detectron2](https://github.com/facebookresearch/detectron2)**（Meta、Apache-2.0）は保守モードで、2021年のv0.6以降タグ付きリリースがありませんが、信頼性があり、インスタンスセグメンテーションやパノプティックセグメンテーション向けのMask R-CNNとR-CNNファミリーを入手するうえで、今でも最も扱いやすい提供元です。
- **[TorchVision](https://github.com/pytorch/vision)**（BSD-3-Clause、活発に保守）はFaster R-CNN、RetinaNet、FCOS、SSD、Mask R-CNN、Keypoint R-CNNを提供しています。最新のYOLOやDETRはなく、学習ループも自分で書く必要がありますが、追加依存関係のないベースラインです。

商用利用について1点注意があります。OpenMMLabのYOLO再実装を集めた**MMYOLO**は、Apache-2.0ではなくGPL-3.0で、2023年8月から更新が止まっています。

## 6. RT-DETRファミリー

ライセンス：すべてApache-2.0。研究段階のコードで最先端のリアルタイム検出を利用する場合に適しています。

現在のリアルタイム検出の最先端は、YOLOではなく、互いに関連するTransformer検出器の一群です。その大半はRT-DETRから派生し、バックボーンやエンコーダーのコードの多くを共有し、ほぼすべてがApache-2.0なので、相互の切り替えが容易です。大半は検出専用ですが、1つの例外（EdgeCrafter）は同じ蒸留の考え方をセグメンテーションと姿勢推定にも拡張しています。共通する負担は、これらが研究用リポジトリだという点です。学習は設定ファイルで行い、使い勝手はUltralyticsと異なり、サポート窓口もありません。モデルは強力で、ONNXとTensorRTへのエクスポートは通常よく整備されています。

- **[D-FINE](https://github.com/Peterande/D-FINE)**（USTC、ICLR 2025 Spotlight）。ボックス回帰を、自己蒸留を伴う分布の精緻化として定式化し直しています。レイテンシに対する精度が高く（D-FINE-XはT4で約13 ms、約55.8 AP）、サイズはNからXまであり、Hugging Face Transformersに統合されているため、この中では独自リポジトリの外で最も簡単に読み込んでファインチューニングできます。
- **[DEIMとDEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)**（Intellindust AI Lab）。DEIM（CVPR 2025）はD-FINEやRT-DETRv2に追加できる学習レシピ（Dense O2O matching）で、学習時間をほぼ半分に短縮します。DEIMv2はDINOv3の特徴を追加し、モバイル向けにパラメータ数1M未満のAttoまで小型化しています。DEIMv2-Sは、10M未満で初めて50 APを超えたモデルです。活発に保守されており、2026年にも更新が続いています。
- **[EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter)**（Intellindust AI Lab、2026）。同じ研究所の最新の成果で、この中で検出以外にも対応する選択肢です。検出（ECDet）、インスタンスセグメンテーション（ECSeg）、姿勢推定（ECPose）をカバーし、それぞれS/M/L/Xのサイズがあり、すべてApache-2.0です。DINOv3で事前学習した大型ViTをタスク専用の教師モデルに適応させ、エッジ向けに設計された小型の生徒バックボーンに蒸留します。サイズに対する数値は高く、ECDet-Sは10M未満のパラメータでCOCOのみを使って51.7 AP、ECPose-Xは74.8 AP（YOLO26Pose-Xの71.6を上回る）を達成し、インスタンスセグメンテーションもRF-DETRに匹敵します。登場したばかりなので研究用リリースとして扱うべきですが、3つのタスクすべてに対応する小型モデルは珍しい存在です。
- **[RT-DETRとRT-DETRv2](https://github.com/lyuwenyu/RT-DETR)**（Baidu）。元祖の「DETRs beat YOLOs on real-time detection」という研究（CVPR 2024）で、このファミリーの起点です。NMS不要、エンドツーエンドで、エクスポートも容易です。v2は2024年のbag-of-freebiesによる更新で、同じレイテンシでそのまま置き換えられる改良版です。Paddleのオリジナル実装と正式なPyTorch移植版がこのリポジトリを共有し、HF Transformersにも含まれています。
- **[RT-DETRv3](https://github.com/clxia12/RT-DETRv3)**（Baidu、WACV 2025 Oral）と **[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)**（北京大学と清華大学、2025年後半）。v3はPaddlePaddle専用です。v4はこのシリーズで現時点の最高性能（Xで57 AP近く）を持ち、PyTorchベースで、視覚基盤モデルを軽量検出器に蒸留することで推論コストを増やさずに動作します。そのコードベースは、設定変更によってD-FINE、DEIM、RT-DETRv2も再現できます。
- **[LW-DETR](https://github.com/Atten4Vis/LW-DETR)**（Baidu）。YOLOを置き換えるTransformerとして提案された、標準的なViTを使うDETRです。サイズはtinyからxlargeまであり、ONNXとTensorRTへのエクスポートに対応します。オープンボキャブラリ検出のOVLW-DETRの基盤でもあります。最近の活動は少なめなので、安定した研究用リリースとして扱ってください。

LibreYOLOは、これらの多く（D-FINE、DEIM、DEIMv2、RT-DETRv2、RT-DETRv4、RTMDet、EdgeCrafter）をすでに1つのAPIに統合しており、YOLO9とRF-DETRが最もよくテストされたモデルです。参照実装や最新の研究用チェックポイントについては、上記のソースリポジトリが正式な提供元です。

## よくある質問

**Ultralytics YOLOは無料で商用利用できますか？**
AGPL-3.0の条件下でのみ可能です。このライセンスは、ネットワーク経由のサービスを含め、その上に構築したアプリケーションのソース公開を要求します。クローズドソースで商用利用するには、Ultralyticsの有料商用ライセンスが必要です。寛容なライセンス（MIT、Apache-2.0）の代替製品なら、これを避けられます。

**ライセンス上の制約が最も少ないUltralyticsの代替製品は何ですか？**
寛容なライセンスで、どこにでも出荷できる構成なら、LibreYOLO（コードはMIT）、RF-DETR（Apache-2.0）、YOLOX（Apache-2.0）です。いずれもAGPLのコピーレフトを回避できます。

**2026年にYOLOに代わりつつあるものは何ですか？**
RT-DETR、RF-DETR、D-FINEやDEIMシリーズなど、リアルタイムのTransformer検出器が増えています。十分な性能のハードウェアでは、同程度のレイテンシでYOLOと同等以上の精度を達成します。小型エッジデバイスでは、こうしたTransformerは実行性能が低く、成熟したNCNNやCPUでの実行手段もないため、YOLO型のCNNが依然として優位です。

**Raspberry PiやNPUで動かせますか？**
エクスポートの対応状況によります。YOLOXやRTMDetなどのCNN検出器はNCNNにエクスポートしてRaspberry Piで実行でき、一部（YOLOX）はHailo NPUにも対応します。RF-DETRなどのTransformer検出器は対応しておらず、GPUまたは高性能なCPUが必要です。

## 試してみる

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLOはMITライセンスで、Linux、Mac、Windowsに対応し、コードを変更せずにGPU、Apple Silicon、通常のCPUで動作します。1つのAPIでRF-DETR、D-FINE、DEIM、YOLOX、YOLO-NAS、RTMDet、RT-DETR、EdgeCrafterなどを扱え、検出、セグメンテーション、姿勢推定、分類、深度推定、視線推定、追跡をカバーします。

GitHubでスターを付ける：[github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | ドキュメント：[libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
