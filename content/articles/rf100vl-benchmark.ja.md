---
title: "RF100-VLベンチマーク：LibreYOLOのYOLOモデルは未知のデータセットにどこまで汎化するか"
description: RF100-VLで、100種類の実データセットを使ってファインチューニングしたYOLOv9、YOLOX、YOLO-NAS、EdgeCrafter、RF-DETRの汎化性能を測定しました。
date: 2026-09-05
author: Xuban
layout: paper
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
---

RF100-VLでは、検出器がCOCO以外のデータにどれだけ適応できるかを調べます。赤外線、X線、顕微鏡画像、スポーツ、無線スペクトル画像、小さな物体、ビデオゲームなど、大きく異なる100種類のデータセットで各モデルを個別にファインチューニングしました。LibreYOLOの17構成をベンチマークし、合計**1700回のファインチューニング**を実施しました。

<div>
<rf100vl-explorer></rf100vl-explorer>
</div>

## 結果

<div>
<rf100vl-results-chart></rf100vl-results-chart>
</div>

RF-DETR-Lが**61.76**でこの比較をリードしました。次いでMが61.13、Sが60.41でした。RF-DETRの4つの結果は、[Roboflowが公開している数値](https://rfdetr.roboflow.com/latest/learn/benchmarks/)に近く、LibreYOLOでのRF-DETR学習を検証するうえで有用です。

モデルサイズがすべての結果を予測するわけではありません。YOLO-NAS-SとMは58.00と57.99で実質的に同点です。EdgeCrafter-Mは57.93で、Sの55.99、Lの56.11を上回りました。これらの性能低下は今後調査する予定です。この一斉評価は、条件を統制したスケーリング実験ではありません。解像度、事前学習済み重み、精度、レシピが異なります。

バックボーンにLoRAを適用したRF-DETR-Sは57.19で、全層ファインチューニングの60.41と比べて低い結果でした。全層ファインチューニングは95データセットで優れています。記録された中央値では所要時間の差はわずか7分で、ジョブ時間の合計はほぼ変わりません。

YOLOv9の行は、重要な学習修正を加える前の結果です。異常なMの結果をきっかけに、PGIの欠落、異なるレターボックス規約、100ラベルの学習上限が判明しました。アーカイブされた数値は、実際に実行されたコードの結果を示しています。YOLOv9アーキテクチャの評価を決定づけるものではありません。

以下からモデルを選ぶと、100種類すべてのデータセットでのスコアを確認できます。

<div>
<rf100vl-results-detail></rf100vl-results-detail>
</div>

## 方法

各データセットについて、`train`で学習し、検証AP50:95が最良のチェックポイントを選択して、`test`で1回評価しました。概要スコアは、100件のテスト結果の単純平均です。

| 設定 | キャンペーンの規則 |
|---|---|
| 学習 | 100エポック、早期終了なし |
| 実効バッチサイズ | 16 |
| シード | 0、データセットごとに完了した実行は1回 |
| チェックポイント | EMA重みを使用した検証AP50:95が最良のもの |
| テストスコア | `maxDets=500`のpycocotools |
| チューニング | ファミリーごとに固定レシピ1つ、データセットごとの調整なし |

各ファミリーで独自の学習レシピを使用しました。

| ファミリー | 解像度 | 精度 | 拡張の概要 |
|---|---:|---:|---|
| YOLOv9 T/S/M | 640 | FP32 | Mosaic、HSV、反転。最後の15エポックは強いデータ拡張を無効化 |
| YOLOX Nano/Tiny | 416 | FP32 | Mosaic、mixup、HSV、アフィン変換、反転。最後の15エポックは調整期間 |
| YOLOX S/M | 640 | FP32 | 同じファミリーレシピを使い、サイズ別の学習率を設定 |
| YOLO-NAS S/M | 640 | FP32 | Mixup、HSV、反転。Mosaicは無効 |
| EdgeCrafter S/M/L | 640 | FP16 | 反転。LibreYOLOのファミリー既定値 |
| RF-DETR N/S/M/LおよびS LoRA | 384/512/576/704、LoRAは512 | BF16 | マルチスケール、クロップ・リサイズ、反転 |

これらはシード1つでの結果です。小さな差が性能向上を裏付けるとは限りません。RF-DETR MとLは、利用可能なメモリに収めるため勾配蓄積を使用し、密なデータセットの一部では物理バッチサイズを小さくしました。公開されているRF-DETRのレシピはCOCOチェックポイントから開始しますが、Roboflowの過去の表では非公開のObjects365チェックポイントが使われています。

学習時間はキャンペーンの記録であり、条件を統制した速度ベンチマークではありません。ジョブがGPUを共有したり、再試行や再開を行ったりする場合がありました。任意のT4単一アーティファクトのレイテンシープロトコルは実施していません。YOLO-NASでは、Deciの別ライセンスで提供される[非商用の事前学習済み重み](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md)を使用します。

## ワークロードによる改善点

小規模なテストでは、学習が開始することは確認できます。しかし、数週間にわたって多くのアーキテクチャとデータセットでファインチューニングを続ける状況は再現できません。この規模では、遅い処理経路、メモリ負荷、正確性の問題がはっきり見えるようになりました。

- **画像読み込みと検証。** 決定論的なリサイズをデータ拡張前にキャッシュし、検証ワーカーとバッファーを再利用し、対応する場合は検証のフォワード処理をグラフ化しました。[PR #677](https://github.com/LibreYOLO/libreyolo/pull/677)、[PR #682](https://github.com/LibreYOLO/libreyolo/pull/682)
- **学習時のCUDAグラフ。** キャプチャ対応をYOLOv9とRF-DETRから24ファミリーに拡大しました。また、DataLoaderのpin-memory競合と、学習遷移時のグラフ無効化も修正しました。YOLOv9-Tのテストでは、報告APを維持したまま所要時間が428.4秒から367.7秒に短縮しました。[PR #671](https://github.com/LibreYOLO/libreyolo/pull/671)、[PR #681](https://github.com/LibreYOLO/libreyolo/pull/681)、[PR #716](https://github.com/LibreYOLO/libreyolo/pull/716)
- **COCOスコアリング。** 密なデータセットでは、評価に学習時間以上かかることがありました。faster-coco-evalの統合により、100のテスト分割すべての保存済み予測を、131.4秒から8.4秒でスコアリングできるようになりました。1,400個のメトリクス値のうち1,381個はビット単位で一致し、最大差は2.22e-16で、概要APに変化はありませんでした。[PR #708](https://github.com/LibreYOLO/libreyolo/pull/708)
- **RF-DETRと共有学習経路。** L1マッチングの高速化、デバイス同期の削減、AdamWの融合、マッチャーメモリの上限設定により、記録されたテストでRF-DETR-Sのステップ時間を266msから234msに短縮しました。同じ調査で、ほかの5つのマッチャー、YOLOv9のロギング、EdgeCrafterのテンソル再利用も改善しました。[PR #761](https://github.com/LibreYOLO/libreyolo/pull/761)、[PR #762](https://github.com/LibreYOLO/libreyolo/pull/762)、[PR #765](https://github.com/LibreYOLO/libreyolo/pull/765)
- **アテンション。** 条件を満たすアテンション処理をPyTorch SDPAに振り分け、Apache-2.0のCUDA実装を統合し、混合精度での実行を修正しました。また、推論用にリポジトリ内のTriton変形可能アテンションカーネルも作成しました。記載されたテストでは、単独で約4倍高速化し、RF-DETR-Nのエンドツーエンド処理では約7%高速化しました。[PR #712](https://github.com/LibreYOLO/libreyolo/pull/712)、[PR #713](https://github.com/LibreYOLO/libreyolo/pull/713)、[PR #760](https://github.com/LibreYOLO/libreyolo/pull/760)、[PR #784](https://github.com/LibreYOLO/libreyolo/pull/784)、[PR #790](https://github.com/LibreYOLO/libreyolo/pull/790)
- **学習の正確性。** YOLOXのBatchNorm再構築を修正し、YOLOv9のPGI、レターボックスのメタデータ、ラベル容量、モメンタムのウォームアップを復元しました。このベンチマークから得たチェックポイントと失敗パターンが、両方の問題の発見につながりました。[PR #700](https://github.com/LibreYOLO/libreyolo/pull/700)、[PR #796](https://github.com/LibreYOLO/libreyolo/pull/796)

これらの数値は、それぞれ異なるワークロードで別々に行ったテストの結果です。1つの概要速度向上率を算出するために掛け合わせるべきではありません。キャンペーン開始前と比べて、LibreYOLOは実際の学習ワークロードでより高速かつ安定して動作するようになりました。

## ハーネスと成果物

公開されている[学習ハーネス](https://github.com/LibreYOLO/vision-analysis-benchmark/tree/rf100vl-harness)は、GPU間でデータセットをスケジュールし、中断したジョブを再開し、OOMからの復旧に対応し、レシピ、データセットのバージョン、ログ、チェックポイント、予測を記録します。

[run-rf100vl-benchmarkスキル](https://github.com/LibreYOLO/libreyolo/blob/release/skills/run-rf100vl-benchmark/SKILL.md)では、AIコーディングエージェント向けのプロトコルとVast.aiの運用手順を説明しています。[結果アーカイブ](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main)には、公開された実行結果がすべて含まれています。

## Roboflowへの謝辞

COCO以外のベンチマークを実施したいものの、その実行費用を負担できないと書いた後、Joseph NelsonがGPUの支援を申し出てくれました。Matvei Popovは参考となる設定を共有し、評価設定を説明し、結果が出るたびに確認してくれました。

その支援のおかげで、17構成にわたるLibreYOLOの学習を検証し、モデル選びに役立つ根拠を公開し、ハーネスをリリースし、弱点が明らかになるまでライブラリに負荷をかけることができました。

計算資源、時間、助言を提供してくれたJoseph、Matvei、そしてRoboflowチームに感謝します。
