---
title: libreyolo quantize
seo_title: libreyolo quantize コマンドリファレンス
description: コマンドラインでPyTorchのチェックポイントを量子化します。レシピ、キャリブレーション引数、デフォルト値、そして各レシピが受け付けるファミリー。
lead: >-
  モデルの浮動小数点モジュールを量子化済みのモジュールに置き換え、レシピが統計量を必要とする場合はラベルなし画像でキャリブレーションを行い、結果をPyTorchのチェックポイントとして保存します。
keywords:
  - libreyolo quantize cli
  - int8 量子化 コマンド
  - fp8 量子化
  - 学習後量子化 yolo
  - libreyolo quantize 引数
last_verified: 1.5.0
meta:
  - label: コマンド
    value: libreyolo quantize
    mono: true
  - label: 必須
    value: model
    mono: true
  - label: 出力
    value: 接尾辞の前に-<recipe>を付けたソースパス、たとえばLibreYOLO9s-int8.pt
    mono: true
snippets:
  examples:
    - label: 基本
      language: bash
      code: |
        # coco128でキャリブレーションしてLibreYOLO9s-int8.ptを出力
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8
    - label: キャストのみ、キャリブレーションなし
      language: bash
      code: |
        libreyolo quantize model=LibreYOLO9s.pt recipe=fp16 calib=none \
          out=weights/LibreYOLO9s-fp16.pt
    - label: キャリブレーションを広げてから精度を回復
      language: bash
      code: >
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8 \
          calib=coco128.yaml samples=256 batch=16 algorithm=minmax

        # 量子化したチェックポイントで量子化認識学習を行うと精度が回復

        libreyolo train model=LibreYOLO9s-int8.pt data=coco8.yaml epochs=10
        lr0=0.001
source_hash: 7ae663e9f117826e
---

## 概要

```bash
libreyolo quantize model=<name|path> [recipe=<recipe>] [key=value ...]
```

引数は`key=value`のペアで、POSIX形式も使えるため、`recipe=int8`と
`--recipe int8`は同じ引数です。

## 引数

| 引数 | デフォルト | 意味 |
|---|---|---|
| `model` | | モデルの重み`.pt`。必須 |
| `recipe` | `int8` | 量子化レシピ：`fp16`、`bf16`、`fp8`、`int8`、`w4a16`、`w4a8`、`nvfp4`、`mxfp4`、`int2` |
| `calib` | `coco128.yaml` | キャリブレーション画像：データYAML、または組み込みデータセットの名前。ラベルなしで、順伝播のみ。`none`でキャリブレーションをスキップ |
| `samples` | `128` | キャリブレーション画像の最大枚数 |
| `batch` | `8` | キャリブレーションのバッチサイズ |
| `algorithm` | `auto` | 活性化範囲の推定：`auto`（minmaxを選択）、`minmax`、`percentile` |
| `out` | | 出力チェックポイントのパス。デフォルトは接尾辞の前に`-<recipe>`を付けたソースパス |
| `device` | `auto` | デバイス |
| `allow_download_scripts` | `false` | データセットYAMLのダウンロードブロックに埋め込まれたPythonを許可 |
| `json` | `false` | stdoutへのJSON出力 |
| `quiet` | `false` | stderrを抑制 |
| `help_json` | `false` | コマンドのスキーマをJSONで出力して終了 |

## 例

<code-tabs name="examples" />

## 注記

### どのファミリーが受け付けるか

量子化は4つのファミリーを対象とします。`yolo9`、`rfdetr`、`birefnet`、
`feynobg`です。それ以外のファミリーは、このリストを添えて`quantize_failed`で
終了します。

### 各レシピが変更するもの

`fp16`と`bf16`はキャストです。dtypeを変えるだけでキャリブレーションは不要で、
これらには`calib=none`が正しい設定です。

`int8`と`fp8`は`Conv2d`と`Linear`のモジュールを量子化するため、畳み込み系の
ファミリーに向いています。

`w4a16`、`w4a8`、`nvfp4`、`mxfp4`、`int2`は`nn.Linear`だけを量子化するので、
transformer系のファミリーが対象です。`yolo9`でこれらのいずれかを指定すると、
そこでは8ビット未満の高速化がGEMMにしか効かず畳み込みはより高い精度のまま
残るため、量子化されていないモデルを黙って出力するのではなく、説明を添えて
拒否されます。

`int8`、`fp8`、`w4a8`、`int2`は活性化のキャリブレーション統計量を必要とします。
`int2`はさらに事後の回復のための学習も必要なので、トレーナーを持たない
`birefnet`と`feynobg`では拒否されます。

レシピにかかわらず、各ファミリーは一部のモジュールを浮動小数点のまま保持
します。最初の層、予測ヘッド、そしてYOLOv9では、固定の積分期待値演算子であり
量子化してはならないDFL畳み込みです。

### キャリブレーションデータは学習データではありません

`calib`は、活性化範囲を導出するために順伝播のみで使う小さなラベルなし画像
セットを指します。評価に使われることはなく、そのラベルが読まれることも
ありません。デフォルトの`coco128.yaml`は初回使用時にURLからダウンロードされる
ため追加の許可は不要ですが、Pythonのダウンロードスクリプトが埋め込まれたYAMLには
`allow_download_scripts=true`が必要です。

`algorithm=percentile`も利用できますが、transformer系のファミリーでは精度を
下げることがあり、そのため`auto`はminmaxを選択します。

### 精度を回復する

出力は通常のPyTorchチェックポイントなので、
[`libreyolo train`](/docs/cli/train)がそのまま受け付けます。量子化した
チェックポイントを学習することが量子化認識学習であり、
`distill_model=<teacher>`を加えると量子化認識蒸留になります。

### 出力と終了コード

結果には、保存先のパス、レシピ、実行モード、キャリブレーションが実行されたか
どうか、種類ごとに置き換えられたモジュールの数が出力されます。終了コードは
成功時が`0`、モデルを読み込めない場合が`4`、量子化または保存に失敗した場合が
`5`、その他のランタイムエラーが`1`です。

関連：[`libreyolo export`](/docs/cli/export)は、PyTorchを離れ、代わりにデプロイ
用の成果物を書き出します。
