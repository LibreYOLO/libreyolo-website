---
title: libreyolo train
seo_title: libreyolo train コマンドリファレンス
description: コマンドラインからのモデル学習：59個すべての引数とそのデフォルト値、モデルファミリーのデフォルト値による上書きの仕組み、そしてファミリーが無視する引数。
lead: >-
  1つのデータセットで1つのモデルを学習し、チェックポイント・指標・ログを実行ディレクトリに書き出します。以下の各引数にはコマンド定義由来のデフォルト値があり、モデルファミリー自身の学習設定がそれを置き換えることがあります。
keywords:
  - libreyolo train cli
  - libreyolo 学習 コマンド
  - yolo コマンドライン 学習
  - libreyolo train 引数
  - libreyolo dry_run 確認
  - yolo 層 freeze 学習
last_verified: 1.5.0
meta:
  - label: コマンド
    value: libreyolo train
    mono: true
  - label: 必須
    value: data
    mono: true
  - label: 出力
    value: runs/train/exp配下のチェックポイント・指標・ログ
snippets:
  examples:
    - label: 基本
      language: bash
      code: >
        # coco8.yamlはパッケージ同梱で初回使用時に8枚の画像をダウンロード

        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10 imgsz=640
        batch=8
    - label: 先に解決後の設定を確認
      language: bash
      code: >
        # ファミリーのデフォルト値を含めて実行時に使われる設定を表示し

        # 学習もデータ読み込みもせずに終了

        libreyolo train model=LibreDFINEn.pt data=coco8.yaml epochs=10
        dry_run=true
    - label: レシピを明示した名前付きの実行
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml \
          epochs=50 batch=8 optimizer=adamw lr0=0.001 weight_decay=0.0001 \
          patience=20 save_period=5 project=runs/train name=yolo9s-coco8 exist_ok=true
source_hash: 3aad4298310d3081
---

## 書式

```bash
libreyolo train data=<dataset.yaml> [model=<name|path>] [key=value ...]
```

引数は`key=value`のペアで、POSIX形式も使えるため、`epochs=50`と`--epochs 50`は
同じ引数です。真偽値は`true`と`false`を受け付けます：フラグに否定形がある場合、
`amp=false`は`--no-amp`になります。

## 引数

### モデルとデータ

| 引数 | デフォルト値 | 意味 |
|---|---|---|
| `data` | | データセットYAMLへのパス（YOLO形式、例：`coco8.yaml`）。必須 |
| `model` | `yolox-s` | モデル名または重みへのパス |
| `task` | | タスクの明示的な上書き：`detect`、`segment`、`semantic`、`pose`、`classify`、`gaze`、`obb`、`point`、`depth` |
| `pretrained` | `true` | 学習済みの重みを使います。`false`にするとアーキテクチャを構築してゼロから学習します |
| `allow_download_scripts` | `false` | データセットYAMLのダウンロードブロックに埋め込まれたPythonを許可 |

### 学習ループ

| 引数 | デフォルト値 | 意味 |
|---|---|---|
| `epochs` | `300` | 学習エポック数 |
| `batch` | `16` | デバイスあたりのバッチサイズ |
| `imgsz` | `640` | 学習時の画像サイズ：`640`（正方形）または`480x640`（HxW） |
| `device` | `auto` | デバイス：`0`、`cpu`、`mps`、`auto` |
| `workers` | `4` | データローダーのワーカー数 |
| `cache` | `false` | データ読み込みを高速化するために画像をキャッシュ：`ram`、`disk`、`true`、`false` |
| `seed` | `0` | 乱数シード |
| `resume` | | 学習を再開：`true`、またはチェックポイントへのパス |
| `amp` | `true` | 自動混合精度 |
| `amp_dtype` | `float16` | CUDA AMPのdtype：`float16`または`bfloat16` |
| `cuda_graph` | `false` | 学習の順伝播と逆伝播をCUDAグラフにキャプチャします。シングルGPUかつ対応ファミリーのみで、それ以外はeager実行のままです |
| `lora` | `false` | LoRAファインチューニング。補足に挙げたTransformerファミリー向け |
| `freeze` | | 層を凍結：整数の個数、インデックスのリスト、またはモジュール名 |

### 蒸留

| 引数 | デフォルト値 | 意味 |
|---|---|---|
| `distill_model` | | 教師：検出器のチェックポイント、またはバックボーンの特徴量蒸留に使う`dinov2`のような基盤教師のID |
| `dis` | | 蒸留損失の重み。未設定の場合は損失タイプごとの公開されたデフォルト値 |
| `distill_loss_type` | `mgd` | 検出器教師向けの特徴量損失：`mgd`、`cwd`。基盤教師は常に`feat_mse`を使います |

### オプティマイザー

| 引数 | デフォルト値 | 意味 |
|---|---|---|
| `optimizer` | `sgd` | オプティマイザー：`sgd`、`adam`、`adamw` |
| `lr0` | `0.01` | 初期学習率 |
| `momentum` | `0.937` | SGDのモーメンタム、およびAdam系オプティマイザーの一次モーメント係数 |
| `weight_decay` | `0.0005` | L2正則化 |
| `nesterov` | `true` | Nesterovモーメンタム |

### スケジューラー

| 引数 | デフォルト値 | 意味 |
|---|---|---|
| `scheduler` | `yoloxwarmcos` | 学習率スケジュールの種類 |
| `warmup_epochs` | `5` | ウォームアップの長さ |
| `warmup_lr_start` | `0.0` | ウォームアップ開始時の学習率 |
| `min_lr_ratio` | `0.05` | 最小学習率の比率 |
| `lr_drop` | `100` | RF-DETRのstep方式で学習率を下げるエポック |

### データ拡張

| 引数 | デフォルト値 | 意味 |
|---|---|---|
| `mosaic` | `1.0` | Mosaicの確率 |
| `mixup` | `1.0` | Mixupの確率 |
| `hsv_prob` | `1.0` | HSVジッターの確率 |
| `flip_prob` | `0.5` | 左右反転の確率 |
| `degrees` | `10.0` | 回転の範囲（度単位、プラスマイナス両方向） |
| `translate` | `0.1` | 平行移動の比率 |
| `shear` | `2.0` | せん断の角度 |
| `mosaic_scale` | `(0.1,2.0)` | Mosaicのスケール範囲 |
| `mixup_scale` | `(0.5,1.5)` | Mixupのスケール範囲 |
| `no_aug_epochs` | `15` | 最後のNエポックでデータ拡張を無効化 |

### EMA

| 引数 | デフォルト値 | 意味 |
|---|---|---|
| `ema` | `true` | 指数移動平均 |
| `ema_decay` | `0.9998` | EMAの減衰係数 |

### 学習中の検証

| 引数 | デフォルト値 | 意味 |
|---|---|---|
| `val` | `true` | 学習中に検証 |
| `eval_interval` | `10` | Nエポックごとに検証 |
| `max_det` | `300` | 検証時のNMS後の画像1枚あたりの最大検出数 |
| `eval_max_det` | | COCO評価器の上限。未設定の場合はpycocotoolsのAP@100の慣例 |
| `faster_coco_eval` | `true` | インストール済みならCOCO指標にfaster-coco-evalのC++バックエンドを使用し、なければpycocotoolsにフォールバック |
| `save_plots` | `false` | 学習中に最終的な検証プロットを保存 |
| `patience` | `50` | 早期終了（early stopping）の待機エポック数。`0`で無効 |

### 出力

| 引数 | デフォルト値 | 意味 |
|---|---|---|
| `project` | `runs/train` | 出力ディレクトリのルート |
| `name` | `exp` | 実験名 |
| `exist_ok` | `false` | 既存の出力ディレクトリを再利用 |
| `save_period` | `10` | Nエポックごとにチェックポイントを保存 |
| `log_interval` | `10` | Nバッチごとに損失を記録 |

### エージェント向けフラグ

| 引数 | デフォルト値 | 意味 |
|---|---|---|
| `json` | `false` | 標準出力へのJSON出力 |
| `quiet` | `false` | 標準エラー出力を抑制 |
| `dry_run` | `false` | 実行せずに設定を解決して表示 |
| `help_json` | `false` | コマンドスキーマをJSONで出力して終了 |

## 使用例

<code-tabs name="examples" />

## 補足

### 上記のデフォルト値が常に使われるとは限りません

モデルファミリーはそれぞれ独自の学習設定を持っており、その設定がベースの設定と
異なる場合、明示的に設定しなかった引数についてはファミリー側の値がコマンドの
デフォルト値を置き換えます。自分で引数を設定した場合は常にそちらが優先されます。
`libreyolo cfg`はベースのデフォルト値とファミリーごとの上書きを表示するので、
あるファミリーが実際に何を使うかはこれで確認できます。

これが最も影響するのは`imgsz`です。コマンドのデフォルト値は`640`ですが、これは
すべてのチェックポイントのネイティブ入力ではありません：公開されているRF-DETRの
検出サイズは384、512、576、704で、YOLOXの`n`と`t`のチェックポイントは416です。
RF-DETRとDEIMv2は`imgsz`が明示的に設定されたときだけ値を渡す扱いになっており、
それ以外の場合はそれぞれのサイズがそのまま有効です。ほかのファミリーには指定
された値がそのまま渡され、そのサイズで学習します。厳しいのはFOMOです：各サイズは
ネイティブ入力（96、192、224）しか受け付けないため、FOMOの実行では`imgsz`を
合わせて設定する必要があり、そうでなければエラーで停止します。RF-DETRはさらに、
値がパッチサイズとウィンドウ数の積で割り切れることを要求し、割り切れない場合は
最も近い2つの有効なサイズを報告します。

### ファミリーが無視する引数

すべてのファミリーがすべての引数を読むわけではなく、それが表れるのはデータ拡張の
引数です。RF-DETR、D-FINE、DEIM、DEIMv2、RT-DETRv4、DINOv2はMosaicもMixupも
アフィン変換も行わないパススルーのパイプラインで学習するため、`mosaic`、`mixup`、
`hsv_prob`、`degrees`、`translate`、`shear`、`mosaic_scale`、`mixup_scale`は
そこでは何にも届きません。ECは同じパイプラインを使いますが、タスクが姿勢推定の
ときは`hsv_prob`、`degrees`、`translate`を読みます。分類系のファミリーとSegFormer、
NAFNetは、反転が設定可能な確率ではなく固定の確率で行われるため、その一式に加えて
`flip_prob`も無視します。YOLO-NASは代わりに常時有効なサンプルごとのアフィン変換で
データ拡張するため、`mosaic`だけを無視します。RF-DETRはそのリストに加えてさらに
3つ、`optimizer`、`momentum`、`nesterov`を無視します。

これらを設定してもエラーにはなりません。実行時にはファミリー名と無視する引数を
挙げた行が標準エラー出力に記録されてから学習が始まり、その行がインストールされて
いるバージョンにとっての確定的なリストになります。それが唯一の手がかりでもある
ため、`quiet=true`を指定したスクリプト実行では、標準エラー出力のほかの内容と
ともにこの警告も抑制されます。

`val=false`も関連するケースです。ほとんどのファミリーでは`eval_interval`を`0`に
設定しますが、RF-DETRはその方法で検証を無効にできず、要求を無視したことをログに
出力します。

### 知っておくとよいその他の挙動

`lora=true`を受け付けるのはRF-DETR、D-FINE、DEIM、DEIMv2、RT-DETRのv1、v2、v4、
EC、ConvNeXtです。ほかのファミリーはLoRAなしで学習するのではなく、
`config_unsupported`で終了します。

`pretrained=false`と`resume`の併用は、両者が正反対のことを求めているため、ゼロ
からの学習に対応するファミリーでは拒否されます。

`mosaic`と`mixup`は、設定フィールド`mosaic_prob`と`mixup_prob`のコマンドライン上
の書き方です。MixupがMosaicのサンプルにしか適用されないファミリーでは、`mosaic`が
0のまま`mixup`を0より大きくしても一度も発火せず、実行時にその旨が示されます。

`dry_run=true`はモデル参照を解決し、ファミリーのデフォルト値を適用して、学習に
使われる設定を表示します。データセットは読み込まないため、引数が期待どおりの値に
なっているかを安く確認する方法になります。

標準出力には最終的な結果オブジェクトが流れ、進捗と警告は標準エラー出力に出ます。
終了コードは、成功時が`0`、使い方や設定の誤りが`2`、データセットが見つからないか
読み込めない場合が`3`、モデルを読み込めない場合が`4`、そのほかの実行時エラーが
`1`です。

関連：実行に踏み切る前にデータセットを確認するなら
[`libreyolo doctor`](/docs/cli/doctor)、ブラウザーで実行を見守るなら
[`libreyolo monitor`](/docs/cli/monitor)、結果を測定するなら
[`libreyolo val`](/docs/cli/val)。
