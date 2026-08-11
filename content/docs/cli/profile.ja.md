---
title: libreyolo profile
seo_title: libreyolo profile コマンドリファレンス
description: 学習と推論の速度を測定し、その結果を読み解きます：profileの全サブコマンド、その引数とデフォルト値、そして読み方ごとに何を報告するか。
lead: >-
  学習ステップや推論呼び出しのどこに時間がかかっているかを測定し、自己完結したプロファイルを書き出し、そのプロファイルをいくつかの読み方で読み返すコマンドグループです。
keywords:
  - libreyolo profile コマンド
  - yolo 学習 プロファイリング
  - 推論 レイテンシ 計測
  - gpu カーネル プロファイリング
  - libreyolo profile 比較
last_verified: 1.5.0
meta:
  - label: コマンド
    value: libreyolo profile
    mono: true
  - label: 出力
    value: profile.json and profile_trace.json under runs/profile
    mono: true
snippets:
  examples:
    - label: 推論を測定
      language: bash
      code: |
        # source引数を省略すると同梱のサンプル画像を使用
        libreyolo profile infer --device cpu --warmup 5 --runs 20
    - label: 判定を読む
      language: bash
      code: |
        libreyolo profile summary runs/profile/infer/profile.json
    - label: 2回の測定を比較
      language: bash
      code: >
        libreyolo profile infer --device cpu --warmup 5 --runs 20 --project
        runs/profile/a

        libreyolo profile infer --device cpu --warmup 5 --runs 20 --batch 4
        --project runs/profile/b


        libreyolo profile compare runs/profile/a/infer/profile.json \
          runs/profile/b/infer/profile.json
source_hash: b967e869fd9ba418
---

## 書式

```bash
libreyolo profile <subcommand> [<positional>] [--flag value ...]
```

このグループは`key=value`形式の引数を受け付けません。サブコマンドは位置引数と
POSIXフラグを使うため、`weights=LibreYOLO9t.pt`ではなく
`--weights LibreYOLO9t.pt`と書きます。サブコマンドを付けずに`libreyolo profile`
を実行すると、その一覧が表示されます。

2つのサブコマンドが測定してプロファイルを書き出し、残りはそれを読みます。`run`
と`infer`はどちらも同じ自己完結した`profile.json`を出力するため、読み取り側の
サブコマンドはどちらのプロファイルに対しても動作します。

## profile run

短い学習をプロファイリングしながら実行し、プロファイルを書き出します。

```bash
libreyolo profile run <data> [--flag value ...]
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `data` | | 位置引数。データセットのYAMLまたは名前（例：`coco128`）。必須 |
| `--weights` | `LibreYOLO9t.pt` | モデルの重みファイルまたは名前 |
| `--size` | `t` | モデルのサイズバリアント |
| `--batch` | `16` | マイクロバッチ。`-1`はVRAMの約70%に自動で合わせる |
| `--imgsz` | `640` | 学習時の画像サイズ |
| `--workers` | `8` | データローダーのワーカー数 |
| `--amp` | `true` | モデルファミリーのAMPパスを使用。`--no-amp`で無効化 |
| `--steps` | `20` | プロファイリング対象、つまり測定対象のステップ数 |
| `--warmup` | `5` | 測定前のウォームアップステップ数 |
| `--repeat` | `1` | N回繰り返して平均と標準偏差を求める |
| `--device` | `0` | デバイス |
| `--project` | `runs/profile` | 出力ディレクトリのルート |
| `--json` | `false` | JSONを標準出力に出力 |

測定される区間は`--warmup`と`--steps`を足した反復回数です。これを満たせないほど
小さいデータセットではプロファイルが生成されず、コマンドは終了コード`3`で終了し、
3つの回避策を示します：より大きなデータセット、より少ないステップ数、より小さい
バッチです。

`--repeat`に1より大きい値を指定すると、集計版の
`runs/profile/profile_repeat.json`が書き出され、そのスカラー指標は試行全体で
平均される一方、カーネル一覧は最後の試行のものになります。これは`compare`で
有意性の判定を得るための前提条件でもあります：1回の実行だけでは判定できません。

## profile infer

推論のパスをプロファイリングし、プロファイルを書き出します。

```bash
libreyolo profile infer [<source>] [--flag value ...]
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `source` | | 位置引数。画像またはディレクトリ。省略すると同梱のサンプル画像 |
| `--weights` | `LibreYOLO9t.pt` | モデルの重みファイルまたは名前 |
| `--size` | `t` | モデルのサイズバリアント |
| `--batch` | `1` | 1回のフォワードパスあたりの画像数 |
| `--imgsz` | `640` | 入力画像のサイズ |
| `--half` | `false` | フォワードをautocastで実行、CUDAのみ。`--no-half`で無効化 |
| `--amp-dtype` | `float16` | CUDAのautocastのdtype：`float16`または`bfloat16` |
| `--warmup` | `20` | 測定前のウォームアップ反復回数 |
| `--runs` | `100` | 測定する反復回数 |
| `--repeat` | `1` | N回繰り返して平均と標準偏差を求める |
| `--conf` | `0.25` | 信頼度のしきい値。NMSの処理量に影響 |
| `--iou` | `0.45` | NMSのIoUしきい値 |
| `--max-det` | `300` | 画像あたりの最大検出数。NMSの処理量に影響 |
| `--device` | `0` | デバイス |
| `--trace` | `true` | カーネルとopを掘り下げるためのChromeトレースを出力。`--no-trace`で省略 |
| `--project` | `runs/profile` | 出力ディレクトリのルート |
| `--json` | `false` | JSONを標準出力に出力 |

p50、p90、p99のレイテンシ、1秒あたりの画像数で表したスループット、そして前処理・
フォワード・後処理へのステージ分割を報告します。3つのしきい値の引数がここにあるのは、
それらが後処理の数値を動かすからです。

## profile summary

```bash
libreyolo profile summary <trace> [--json]
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `trace` | | 位置引数。`profile.json`または`profile_trace.json`のパス。必須 |
| `--json` | `false` | JSONを標準出力に出力 |

大づかみな読み方です：ステップ時間、スループット、GPU使用率、Tensor Coreの比率、
VRAMのピーク、ホスト側のオーバーヘッド、ステップあたりのカーネル起動数、理由付きの
ボトルネック判定、カテゴリ別のカーネル構成、そしてステップあたりの上位カーネル。
推論のプロファイルでは、レイテンシのパーセンタイルとステージ分割も表示します。

そこで測定された使用率とスループットは信頼できないため、VRAMのスラッシングが
起きている状態で取得したプロファイルには印が付きます。

## profile get

```bash
libreyolo profile get <trace> [<field>] [--json]
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `trace` | | 位置引数。プロファイルのパス。必須 |
| `field` | | 位置引数。指標名。省略すると利用できる指標の一覧を表示 |
| `--json` | `false` | JSONを標準出力に出力 |

スクリプトのループで使えるように、1つの指標だけを出力します。未知のフィールドを
指定すると終了コード`2`で終了し、一覧表示の形を案内します。

## profile phases

```bash
libreyolo profile phases <trace> [--json]
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `trace` | | 位置引数。プロファイルのパス。必須 |
| `--json` | `false` | JSONを標準出力に出力 |

フェーズごとのGPUミリ秒、実時間ミリ秒、カーネル数、op数です：forward、backward、
dataload、to_device、optimizer。

## profile kernels

```bash
libreyolo profile kernels <trace> [--flag value ...]
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `trace` | | 位置引数。プロファイルのパス。必須 |
| `--top` | `20` | GPU時間の上位N件を表示 |
| `--category` | | カテゴリ名の部分一致で絞り込み：`gemm`、`layout`、`norm`、`elementwise` |
| `--grep` | | カーネル名の正規表現で絞り込み |
| `--tensorcore` | `false` | Tensor Coreのカーネルのみ |
| `--sort` | `time` | `time`、`count`、`name`のいずれか |
| `--phase` | | 1つのフェーズに限定：`forward`、`backward`、`dataload`、`to_device`、`optimizer` |
| `--json` | `false` | JSONを標準出力に出力 |

解析の最下層です：個々のGPUカーネルと、そのGPU時間に占める割合、ステップあたりの
ミリ秒、ステップあたりの呼び出し回数、カテゴリ。未知の`--phase`を指定すると
終了コード`2`で終了し、そのプロファイルに含まれるフェーズを一覧表示します。

## profile ops

```bash
libreyolo profile ops <trace> [--flag value ...]
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `trace` | | 位置引数。プロファイルのパス。必須 |
| `--top` | `20` | CPU時間の上位N件を表示 |
| `--phase` | | 1つのフェーズに限定 |
| `--json` | `false` | JSONを標準出力に出力 |

デバイス視点ではなくフレームワーク視点です：`aten`とautogradのopをCPU時間で並べた
もので、ホスト側の起動コストはここに現れます。

## profile compare

```bash
libreyolo profile compare <before> <after> [--json]
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `before` | | 位置引数。ベースラインのプロファイル。必須 |
| `after` | | 位置引数。新しいプロファイル。必須 |
| `--json` | `false` | JSONを標準出力に出力 |

スループット、画像あたりのミリ秒、GPU使用率、ホスト側のオーバーヘッド、ステップ
あたりのカーネル起動数、ボトルネック判定の差分を表示します。

有意性の判定には、両方のプロファイルを`--repeat`を2以上にして測定しておく必要が
あります。その条件を満たしていれば、差が合成標準誤差の2倍を超えたときに有意と
みなされ、出力には実際に行った比較が表示されます。満たしていない場合、その行には
1回の実行では判定を支えられないと表示されます。

## profile what-if

```bash
libreyolo profile what-if <trace> [--flag value ...]
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `trace` | | 位置引数。プロファイルのパス。必須 |
| `--remove-category` | | あるカーネルカテゴリを取り除いた場合を試算：`gemm`、`layout`、`norm`、`elementwise` |
| `--remove-launches` | | ステップあたりN回のカーネル起動を取り除いた場合を試算。たとえばop融合による削減 |
| `--json` | `false` | JSONを標準出力に出力 |

変更を書く前に、その変更で何が得られるかを見積もります。2つのオプションのうち
どちらか1つが必須で、どちらも指定しないと終了コード`2`で終了します。

試算はそのプロファイル自身の判定に従います。GPU使用率が80%未満なら、削減量は
起動回数の減少に実測の起動1回あたりのホストコストを掛けたものとしてモデル化し、
80%以上なら、GPUの処理量の減少としてモデル化します。起動1回あたりのコストは
近似であり、確かめる方法は2回目の測定しかないため、結果には注意書きのフィールドが
付きます。

## 使用例

<code-tabs name="examples" />

## 補足

プロファイラーは測定して報告します。何も変更しません：判定を読み、設定やコードを
編集し、再実行して比較する、というループのために作られています。

`--device`のデフォルトは`0`で、これはCUDAデバイス0のことです。`--device cpu`を
渡すとCPU上で測定し、GPUカーネルの詳細は付かないものの、読み取り側のサブコマンドが
そのまま受け付けられるプロファイルを生成します。

すべてのサブコマンドが`--json`に対応しており、読み取り側は標準出力にしか出力しない
ため、このグループはスクリプトから使えます。

終了コードはこのグループ独自のものです：存在しないファイルや解決できない引数は`2`、
`run`がプロファイルを生成しなかった場合は`3`、トレースを解析できない場合は`1`です。

関連：[`libreyolo train`](/docs/cli/train)。学習のプロファイルは、たいていこの
コマンドの引数を調整するために取得されます。
