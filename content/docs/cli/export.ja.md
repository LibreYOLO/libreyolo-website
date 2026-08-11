---
title: libreyolo export
seo_title: libreyolo export コマンドリファレンス
description: チェックポイントをデプロイ形式にエクスポートします：各引数とそのデフォルト値、成果物の書き出し先、そしてコマンドが拒否する組み合わせ。
lead: 1つのチェックポイントを1つのデプロイ形式に変換し、成果物をweights/以下に書き出します。下の引数のどれが適用されるかは形式が決めます。
keywords:
  - libreyolo export cli
  - libreyolo export コマンド
  - yolo onnx エクスポート
  - tensorrt エクスポート コマンド
  - libreyolo export 引数
last_verified: 1.5.0
meta:
  - label: コマンド
    value: libreyolo export
    mono: true
  - label: 必須
    value: model
    mono: true
  - label: 出力
    value: 'weights/<checkpoint-stem>[_fp16|_int8]<format-suffix>'
    mono: true
snippets:
  examples:
    - label: 基本
      language: bash
      code: |
        # weights/LibreYOLO9s.onnxを出力
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: グラフ内のNMS
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx \
          nms=true conf=0.25 iou=0.45 max_det=300
    - label: 成果物を実行する
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640

        # ファクトリーはファイル拡張子で振り分けるので、エクスポート結果もチェックポイントと同じように読み込まれる
        libreyolo predict model=weights/LibreYOLO9s.onnx \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: ef2ca20af3814109
---

## 概要

```bash
libreyolo export model=<name|path> [format=<format>] [key=value ...]
```

引数は`key=value`のペアで、POSIX形式も使えるため、`format=onnx`と`--format onnx`は同じ引数です。

## 引数

| 引数 | デフォルト | 意味 |
|---|---|---|
| `model` | | モデルの重み`.pt`。必須 |
| `format` | `onnx` | エクスポート形式：`onnx`、`torchscript`、`executorch`、`tensorrt`、`openvino`、`paddle`、`mnn`、`rknn`、`ncnn`、`tflite`、`coreml`、`coreai` |
| `name` | | RKNNのターゲットプラットフォームで、現在は`rk3588`のみ。他の形式と一緒に渡すと拒否されます |
| `imgsz` | | 入力画像サイズ：`640`または`480x640`（HxW）。`480,640`も受け付けます。未設定の場合はモデル自身のサイズ |
| `batch` | `1` | エクスポート時のバッチサイズ |
| `half` | `false` | FP16精度 |
| `int8` | `false` | INT8量子化 |
| `dynamic` | `false` | 動的な入力形状（ONNX） |
| `simplify` | `true` | ONNXグラフの簡略化 |
| `nms` | `false` | NMSをモデルに埋め込みます。ONNXとCoreMLのみ |
| `conf` | `0.25` | 埋め込みNMSの信頼度しきい値 |
| `iou` | `0.45` | 埋め込みNMSのIoUしきい値 |
| `max_det` | `300` | ONNXの埋め込みNMSでの最大検出数 |
| `opset` | | ONNXのopsetバージョン。未設定の場合は自動で選ばれます |
| `data` | | INT8用のキャリブレーションデータ |
| `fraction` | `1.0` | 使用するキャリブレーションデータの割合 |
| `device` | `auto` | トレースに使うデバイス |
| `allow_download_scripts` | `false` | データセットYAMLのdownloadブロックに埋め込まれたPythonを許可します |
| `json` | `false` | stdoutへのJSON出力 |
| `quiet` | `false` | stderrを抑制します |
| `verbose` | `false` | 詳細なエクスポートログ |
| `verify` | `false` | RKNN Toolkit2のPCシミュレーターを実行し、ONNX Runtimeと比較します。RKNNのみ |
| `help_json` | `false` | コマンドのスキーマをJSONで出力して終了します |

`engine`は`tensorrt`のエイリアス、`litert`は`tflite`のエイリアスです。どちらも何かが書き出される前に正式な名前へ解決されるため、JSON出力とログ行が報告するのは常に`tensorrt`または`tflite`です。

## 例

<code-tabs name="examples" />

## 説明

### ファイルの書き出し先

このコマンドは出力パスを受け取りません。成果物は`weights/`に書き出され、名前は元のチェックポイントのステム名にその形式の拡張子を付けたもので、FP16かINT8のどちらかの精度を指定した場合は`_fp16`または`_int8`が間に挿入されます。`LibreYOLO9s.pt`をFP16でONNXにエクスポートすると`weights/LibreYOLO9s_fp16.onnx`になります。JSON結果には、解決後の`output_path`、MB単位のファイルサイズ、そして`[batch, 3, height, width]`という形の入力形状が入ります。

### 拒否される組み合わせ

`nms=true`はONNXとCoreMLでは受け付けられ、それ以外のすべての形式では`nms_unsupported_format`で拒否されます。ONNXでは、埋め込まれたグラフがバッチ1に固定されるため`dynamic`を強制的にオフにし、そのことをstderrに出力します。CoreMLでは`conf`と`iou`は受け付けますが`max_det`は受け付けないので、デフォルト以外の`max_det`を`format=coreml nms=true`と一緒に渡すと`config_unsupported`で終了します。

`half=true`と`int8=true`を同時に指定してもエラーにはなりません。INT8が優先され、`half`は破棄され、警告がstderrに出ます。

`name`と`verify`は現時点ではRKNN用のオプションです。どちらかを別の形式と一緒に渡すと、無視されるのではなく`config_unsupported`で終了します。

### モデルファミリーがどの形式に対応するか

対応状況はグローバルではなく、ファミリーごと・タスクごとに決まります。`libreyolo formats family=<family> task=<task>`は、その組み合わせについて各形式のティアを、理由と付随する制約とあわせて表示します。引数は[`libreyolo formats`](/docs/cli/utilities)を参照してください。

オプションのインストールが必要な形式もあれば、ツールチェーンが必要な形式もあります。Pythonの依存関係が足りない場合は`export_dep_missing`で終了し、その形式が生成できない精度を指定した場合は`format_precision_unsupported`で終了します。

### エクスポートしたものを実行する

エクスポートした成果物は、チェックポイントと同じモデルファクトリーからファイル拡張子をキーにして読み込まれるため、`libreyolo predict model=weights/LibreYOLO9s.onnx`は追加の変換なしで動きます。例外は3つの推論オプションで、ランタイムバックエンドでは拒否されます：`tiling`、`overlap_ratio`、`output_file_format`。

独自のページを持つデプロイ先が2つあります：[NVIDIA DeepStream](/docs/export/deepstream)と[NVIDIA Jetson](/docs/export/jetson)。

### 出力と終了コード

結果はstdoutに、進捗はstderrに出ます。終了コードは、成功が`0`、使い方や設定の誤りが`2`、モデルを読み込めない場合が`4`、未知の形式・エクスポート依存関係の不足・非対応の精度・拒否された埋め込みNMSの要求が`5`、その他のランタイム障害が`1`です。

関連：[`libreyolo quantize`](/docs/cli/quantize)は、PyTorchの中にとどまり、デプロイ用の成果物ではなくチェックポイントを書き出します。
