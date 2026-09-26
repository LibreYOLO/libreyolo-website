---
title: 学習パフォーマンス
seo_title: '学習を高速化: CUDA Graph、AMP、プロファイラー'
description: >-
  学習を高速化します。stepをCUDA
  Graphへキャプチャし、AMPのdtypeを選び、組み込みプロファイラーで実際に時間を消費している箇所を特定します。
lead: >-
  学習stepの速度を変える手段は3つあります。混合精度、ネットワークの順伝播と逆伝播をCUDA
  Graphへキャプチャする方法、プロファイラーが実際のボトルネックとして示す箇所への対処です。
keywords:
  - CUDA Graph 学習
  - 学習 高速化
  - 混合精度 学習
  - bfloat16 学習
  - PyTorch profiler
  - DataLoader ボトルネック
  - カーネル起動 オーバーヘッド
  - GPU 使用率
last_verified: 1.5.0
snippets:
  profile:
    - label: プロファイル後も学習を継続
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 実際のstepの短い区間をプロファイルして判定を表示し、フックを
        # 除去して実行を継続します。
        model.train(data="my-dataset.yaml", epochs=100, profile=True)
    - label: 測定だけを行って停止
      language: bash
      code: |
        # no_aug_epochs=0を設定し、測定区間を満たすのに必要なエポックだけを実行します。
        libreyolo profile run coco128 --weights LibreYOLO9s.pt --size s
    - label: 結果を詳しく調査
      language: bash
      code: |
        libreyolo profile summary runs/profile/prof/profile.json
        libreyolo profile phases runs/profile/prof/profile.json
        libreyolo profile kernels runs/profile/prof/profile.json --top 10
  graph:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 cuda_graph=true
  amp:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", amp=True, amp_dtype="bfloat16")
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          amp_dtype=bfloat16
source_hash: ee5bb727065b6099
---

## 変更する前に測定する

以下の3つの手段は異なる問題を解決するため、誤ったものを適用しても何も変わりません。プロファイラーで問題を特定してください。

<code-tabs name="profile" />

`profile=True`は実際の学習stepの一定区間を測定します。デフォルトでは最初の5stepを破棄し、次の20stepを測定します。レポートを表示してアーティファクトを書き込んだ後、フックを除去して学習を継続します。無効時のコストはなく、分散学習では無視されます。

レポートの最後には4つの判定のいずれかが示されます。

| 判定 | 意味 | 対処方法 |
|---|---|---|
| `dataloader` | GPUが入力データを待っている | `workers`を増やす、`cache="ram"`または`"disk"`、軽いデータ拡張、大きなバッチ |
| `host / launch` | GPUへの供給が遅く、小さなカーネルが多数ある | 大きなバッチ、CUDA Graph、stepごとのホスト同期を減らす |
| `compute` | GPUが飽和している | AMPまたはbfloat16を使うか、そのまま受け入れる |
| `memory-pressure` | アロケーターが頻繁に動作し、VRAMが限界にある | バッチを下げる。この場合の使用率は信頼できない |

使用率は、同期していないstep時間に対するカーネル稼働時間です。測定区間は意図的に分割されています。前半は追加同期なしで実行し、実際のオーバーラップを反映した判定を行います。後半だけが各フェーズを同期で囲み、GPU時間を割り当てます。すべてのフェーズを同期するとデータローダーワーカーに余裕が生まれ、入力不足が隠れるため、構成比の数値は判定には使用しません。

実行ディレクトリには4つのファイルが保存されます。単独でブラウザーから開ける`timeline.html`、PerfettoまたはNsight用の`profile_trace.json`、`profile_summary.json`、自己完結型で持ち運びやすく`libreyolo profile`サブコマンドへ再入力できる`profile.json`です。

`profile run`について知っておくべき点が2つあります。1つ目は`no_aug_epochs=0`を設定することです。プロファイラーはエポック0を測定するため、デフォルトの`no_aug_epochs`を使った短い実行では、実際の学習に使うデータローダーではなく、軽いデータ拡張なしのデータローダーを測定してしまうためです。2つ目は、`--repeat N`で平均と標準偏差を報告することです。起動時間が支配的なstepはノイズが大きく、1回の実行では誤解を招くため重要です。試行ごとのディレクトリ`prof_1`、`prof_2`などと、集計した`profile_repeat.json`を書き込みます。

## 混合精度

`amp=True`はほとんどのファミリーでデフォルトとなり、CUDA autocast下で順伝播を実行します。`amp_dtype`は`float16`または`bfloat16`を選びます。

<code-tabs name="amp" />

Float16には動的loss scalingが必要で、有効なgradient scalerが作成されます。bfloat16は指数範囲が広いため不要で、scalerは無効になります。D-FINE、DEIM、YOLO-NAS、FOMOの4ファミリーは`amp=False`で提供され、DEIMの設定は継承によってRT-DETRv4にも引き継がれます。D-FINEは理由を明記しています。デコーダーが、float16で表現できる最大の有限値65504に活性値を制限するためです。

bfloat16非対応ハードウェアで要求した場合の動作を含む引数の意味は、[ハイパーパラメーター](/docs/train/hyperparameters)にあります。

## CUDA Graph

`cuda_graph=True`はネットワークの学習用順伝播と逆伝播をCUDA Graphへキャプチャし、stepごとのカーネル起動オーバーヘッドを除去します。

<code-tabs name="graph" />

このフラグは常に安全に渡せます。キャプチャできないファミリー、タスク、構成では1行をログに記録し、変更なしでeager学習を行います。

キャプチャされるのはネットワークだけです。lossは意図的にeagerのままです。物体検出のlossは真偽値マスクで選択し、Hungarian matchingを実行し、割り当て結果で分岐しますが、これらはグラフに記録できないためです。オプティマイザーのstep、勾配クリッピング、EMA更新、学習率スケジュールもeagerのままです。

このため、速度向上の上限はstepに占めるネットワークの割合で決まり、その割合は大きく異なります。RTX 5070 Ti、640 px、バッチ8での測定では、ネットワークが占める割合はYOLOv9-tで84%、YOLOv7-bで44%、YOLOX-tで31%、RTMDet-tで26%です。最後の2つはstepの大半をラベル割り当て処理に費やすため、ネットワークのキャプチャによる効果が最も小さくなります。

### 得られる効果

以下のすべての数値は、RTX 5070 Ti、Windows、AMP、共有の保存済み状態から各条件につき1プロセスを使用し、実データの1バッチを再生してデータローダーを除外し、ウォームアップ後24stepのうち最速値を採用したものです。物体検出は640 px、画像分類は224 pxで、バッチサイズは行ごとに示します。

| ファミリー | サイズ | バッチ | Eager | Graph | 高速化 |
|---|---|---:|---:|---:|---:|
| FOMO | s | 16 | 7.0 ms | 1.9 ms | 3.63x |
| MobileNetV4 | s | 16 | 14.5 ms | 5.3 ms | 2.74x |
| EfficientNetV2 | b0 | 16 | 29.0 ms | 11.9 ms | 2.44x |
| YOLOv9 | t | 8 | 93.6 ms | 47.0 ms | 1.99x |
| NAFNet | s | 8 | 132.5 ms | 105.5 ms | 1.26x |
| PicoDet | s | 8 | 145.0 ms | 118.7 ms | 1.22x |
| D-FINE | n | 4 | 185.3 ms | 159.2 ms | 1.16x |
| RF-DETR | n | 4 | 276.3 ms | 239.8 ms | 1.15x |
| YOLOX | t | 8 | 102.2 ms | 90.5 ms | 1.13x |
| RTMDet | t | 8 | 149.7 ms | 136.2 ms | 1.10x |
| YOLOv7 | b | 4 | 102.5 ms | 98.0 ms | 1.05x |

これらはGPUのstepだけを分離した値です。完全なファインチューニングではデータローダーと検証のコストも発生します。同じマシンで、406枚の物体検出データセット、20エポック、バッチ8、640 px、データローダーワーカー4のYOLOv9-tを測定すると、実時間はeagerで428.4秒、Graphで367.7秒でした。1.16倍の高速化で、どちらもmAP50-95は0.6394です。

これらの数値を変える要因は3つあります。小さなバッチでは起動時間が支配的で、大きなバッチでは計算時間が支配的です。そのため、RT-DETR-r18はバッチ2で1.19倍、バッチ8で1.04倍になります。起動オーバーヘッドはWindowsで最も大きく、Linuxでの向上率は表のおよそ3分の1から2分の1です。また、データローダーがボトルネックの実行では実時間がまったく変わりません。これが最初にプロファイラーを使う理由です。

`amp=False`でも同じ方法でキャプチャされますが、fp32カーネルは実行時間が長いため、stepの起動時間への依存が小さくなり、多くのファミリーで効果が低下します。同じハードウェアで、バッチ16のMobileNetV4-sはAMP時の2.74倍からfp32時の3.61倍へ変化します。一方、バッチ8のYOLOv9-tは1.99倍から1.69倍、バッチ4のRT-DETR-r18は1.12倍から0.99倍へ変化します。

### キャプチャの適用範囲

| タスク | ファミリー |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

それ以外は1行をログに記録し、eagerへフォールバックします。対象ファミリーのその他のタスク、一覧にないファミリー、分散実行、蒸留実行が該当します。実行時にキャプチャが失敗した場合も、処理を失敗させず、残りの実行をeagerへ切り替えます。

エンコーダーデコーダー型の検出器であるD-FINE、DEIM、DEIMv2、RT-DETR v1、v2、v4、ECでは、バックボーンとエンコーダーだけをキャプチャします。これらのデコーダーは正解データを読み取って対照的ノイズ除去クエリーを構築します。そのクエリー数はバッチ内で最大の正解データ数に従うため、トークン数がバッチごとに変わります。

### 形状

グラフはキャプチャ時とまったく同じ入力形状でだけ有効です。トレーナーはバッチ形状を数え、同じ形状が3回現れるとキャプチャします。マルチスケールのバッチやエポック末尾の不完全なバッチなど、それ以外の形状ではeagerで実行します。

これは毎回バッチサイズを変更するDETRファミリーで注意が必要です。`multi_scale=True`では、短い実行中に同じ形状がキャプチャに必要な回数だけ現れない場合があります。高速化が目的なら`multi_scale=False`を渡してください。

YOLOXは実行の途中でキャプチャ領域の計算内容を変更し、`no_aug_epochs`でMosaicが終了するとL1回帰分岐を有効にします。その時点でトレーナーはキャプチャを無効化し、新しい形状が安定してから再度キャプチャします。

### 数値とメモリ

ほとんどのファミリーは、AMP下でeagerとビット単位に同じlossの推移を再現します。FOMOとLingBot-Visionでは加算順序が異なるため、float32の最下位ビットが変わります。変形可能アテンション検出器のD-FINE、DEIM、DEIMv2、RT-DETR、RF-DETR、ECは、自身のeager実行同士でも再現しません。逆伝播がatomic演算で累積され、TF32畳み込みが起動ごとに縮約順序を選ぶためです。Graph実行はそのばらつきの範囲内に収まります。RTMDetでは139個の勾配のうち2つで相対的に約3e-4の差があります。ピラミッドレベル間でヘッドの畳み込みを共有し、2つの逆伝播経路が3つの寄与を異なる順序で加算するためです。SegFormerはキャプチャ領域内にStochastic Depthがあるため、再生したグラフは独自の乱数系列を使用し、eagerと同一ではなく統計的に同等です。マネージャーはキャプチャ時に1回、そのことをログへ記録します。

`amp=False`では、キャプチャの有無にかかわらず、このハードウェアでビット単位の同一性は得られません。同一シードのYOLOv9-t eager実行を2回行うと20stepで相対36%、YOLOX-tでは2.6%ずれます。cuDNNが一部のfp32畳み込み形状に対して非決定的な重み勾配アルゴリズムを選ぶためです。

キャプチャしたグラフは静的な入力、出力、ワークスペースのバッファーを固定するため、ピークVRAMはおよそ活性値1組分だけ増えます。上記のファミリーでは、ピーク割り当て量の変化は-5%から+19%でした。相対コストが最も大きいのは、もともとの活性値が小さい小規模な画像分類モデルです。224 px、バッチ16のResNet-18では、eagerの0.48 GBからGraphの0.57 GBへ増えました。上限を超える場合は、バッチを下げるかフラグを無効にしてください。

## 関連項目

- `batch`、`nbs`、`cache`、`workers`については[ハイパーパラメーター](/docs/train/hyperparameters)を参照してください。
- CUDA Graphとプロファイラーのどちらも利用できない条件については、[マルチGPU学習](/docs/train/multi-gpu)を参照してください。
- 推論と学習を統合した対応マトリックス、区間分割、数値再現の契約については、[CUDA Graph](/docs/reference/cuda-graphs)を参照してください。
