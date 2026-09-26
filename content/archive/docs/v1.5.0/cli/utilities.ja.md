---
title: libreyoloユーティリティコマンド
seo_title: libreyolo CLIユーティリティコマンドリファレンス
description: >-
  version、checks、models、formats、cfg、info、metadata、enroll、compareというLibreYOLOの小さなコマンドを、それぞれの引数とデフォルト値とともに説明します。
lead: >-
  計算ではなく報告や確認を行う9つのコマンドです。環境の情報、モデルと形式の一覧、解決済みのデフォルト値、チェックポイントの詳細を表示し、顔ギャラリーの構築と照会も行います。
keywords:
  - libreyolo version
  - libreyolo checks 環境確認
  - libreyolo モデル一覧
  - libreyolo エクスポート形式 一覧
  - libreyolo デフォルト設定 確認
  - libreyolo info モデル情報
  - yolo チェックポイント メタデータ 確認
  - libreyolo 顔 登録 gallery
  - 顔 コサイン類似度 比較 cli
last_verified: 1.5.0
meta:
  - label: コマンド
    value: 'version, checks, models, formats, cfg, info, metadata, enroll, compare'
    mono: true
  - label: 出力
    value: stdout。テキスト形式、またはjson=trueを指定するとschema_versionを持つ1つのオブジェクト
snippets:
  examples:
    - label: 環境
      language: bash
      code: |
        libreyolo version
        libreyolo checks
    - label: 何が使えるか
      language: bash
      code: |
        libreyolo models
        libreyolo formats family=yolo9 task=detect
    - label: チェックポイントを確認
      language: bash
      code: |
        libreyolo info model=LibreYOLO9s.pt
        libreyolo metadata path=weights/LibreYOLO9s.pt
source_hash: 7b5b53c46df00c06
---

## 書式

```bash
libreyolo <command> [key=value ...]
```

引数は`key=value`のペアで、POSIX形式も使えるため、`model=x`と`--model x`は同じ引数です。ここに挙げるコマンドはすべて結果をstdoutに書き出し、`json=true`と`quiet=true`を受け付けます。

ルートコマンドには独自のフラグが1つあり、`libreyolo --version`はバージョン文字列を表示して終了します。これは下の`version`コマンドよりも小さな出力です。

## version

LibreYOLOのバージョンと、実行環境のPython、torch、CUDAのバージョンを表示します。

```bash
libreyolo version
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `json` | `false` | stdoutにJSONを出力 |
| `quiet` | `false` | stderrを抑制 |

## checks

環境をより詳しく表示します：Python、torch、CUDA、cuDNN、検出されたすべてのGPUの名前とメモリ、そしてエクスポート処理が使うオプションパッケージそれぞれのインストール済みバージョンです。

```bash
libreyolo checks
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `json` | `false` | stdoutにJSONを出力 |
| `quiet` | `false` | stderrを抑制 |

パッケージ一覧は`onnx`、`onnxruntime`、`tensorrt`、`openvino`、`paddlepaddle`、`x2paddle`、`mnn`、`ncnn`、`onnx2tf`、`ai-edge-litert`、`transformers`、`scipy`を対象とします。インストールされていないパッケージは省略されずに未インストールとして報告されるため、エクスポートの失敗をこのコマンド1つで依存関係の不足までたどれます。

## models

すべてのモデルファミリーを、そのタスク、サイズ、チェックポイントに解決されるCLI名、および各サイズの入力解像度とともに一覧表示します。

```bash
libreyolo models
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `json` | `false` | stdoutにJSONを出力 |
| `quiet` | `false` | stderrを抑制 |

オプションの依存関係がインストールされていないファミリーは、利用不可として、それを利用可能にする`pip install`の行とともに表示されます。CLI名は`model=`が短縮形として受け付けるものです：`yolox-s`は`LibreYOLOXs.pt`に解決され、物体検出以外のタスクにはタスクのサフィックスが付きます。

## formats

インストール済みの環境で生成できるエクスポート形式を、各形式のファイル拡張子、およびFP16とINT8に対応しているかどうかとともに一覧表示します。

```bash
libreyolo formats [family=<family>] [task=<task>]
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `family` | | 1つのモデルファミリーの対応レベルを表示。`model=`も同じオプションとして受け付け可能 |
| `task` | | モデルの正規タスク。未指定ならファミリーのデフォルトタスク |
| `json` | `false` | stdoutにJSONを出力 |
| `quiet` | `false` | stderrを抑制 |

`family`を指定しない場合、出力は形式の一覧だけです。指定すると、各形式にそのファミリーとタスクでの対応レベル、その理由、および付随する制約が加わります。未知のファミリー、またはそのファミリーが対応していないタスクは、使用方法のエラーになります。

形式のエイリアスは正規名の隣に表示されます：`tensorrt`には`engine`、`tflite`には`litert`。

## cfg

解決済みのデフォルト設定を表示します：学習のデフォルト、検証のデフォルト、推論のデフォルト、およびファミリーごとのオーバーライドです。

```bash
libreyolo cfg
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `json` | `false` | stdoutにJSONを出力 |
| `quiet` | `false` | stderrを抑制 |

値はコピーではなく設定のdataclassから読み取られるため、引数を渡さなかったときに学習が使う値については、これが根拠になります。`family_overrides`は、指定していない設定でファミリーが学習された理由に答えるセクションです。これらのオーバーライドがどのように適用されるかは[`libreyolo train`](/docs/cli/train)を参照してください。

## info

モデルをCPU上に読み込み、そのファミリー、サイズ、パラメータ数、クラス、および各形式のエクスポート対応レベルを報告します。

```bash
libreyolo info model=<name|path>
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `model` | | モデル名または重みへのパス。必須 |
| `detailed` | `false` | パラメータごとの詳細を含める |
| `json` | `false` | stdoutにJSONを出力 |
| `quiet` | `false` | stderrを抑制 |

## metadata

モデルを構築せずにチェックポイントのメタデータを読み取り、LibreYOLOのチェックポイントスキーマと照合して検証します。

```bash
libreyolo metadata path=<checkpoint.pt>
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `path` | | `.pt`チェックポイントへのパス。必須 |
| `json` | `false` | stdoutにJSONを出力 |
| `quiet` | `false` | stderrを抑制 |

大きなテンソルを含むエントリはそのまま表示されずに要約されるため、学習用のフルチェックポイントでも出力は読みやすいままです。存在しないチェックポイントは`checkpoint_not_found`で終了し、メタデータの検証に失敗したチェックポイントはエラーを表示して`1`で終了します。

## enroll

人物ごとにフォルダーを分けたツリーから顔ギャラリーを構築し、以降の推論が見つけた顔に名前を付けられるようにします。

```bash
libreyolo enroll model=<embedder> source=<people-dir> gallery=<gallery.npz>
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `model` | | 顔の埋め込みベクトルのモデル。パスまたは名前。必須 |
| `source` | | 人物ごとにフォルダーを分けたツリー、`source/<identity>/*.jpg`。必須 |
| `gallery` | | 出力するギャラリーファイル`.npz`。既存の場合はその場に追記。必須 |
| `face_detector` | | 顔検出器：YuNetの`.onnx`、またはLibreYOLOの検出器。未指定ならファミリーのデフォルトの検出器 |
| `device` | `auto` | デバイス：`0`、`cpu`、`mps`、`auto` |
| `json` | `false` | stdoutにJSONを出力 |
| `quiet` | `false` | stderrを抑制 |

```bash
# people/ には識別名ごとに1つのフォルダーを置き フォルダー名がそのまま識別名になる
libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=people.npz
```

サブフォルダー名が識別名になります。顔を検出できない参照画像はstderrに1行出したうえでスキップされ、残りは処理が続きます。識別名のサブフォルダーがないソース、または顔が1つも見つからなかったソースはエラーです。

生成されたファイルを`gallery=people.npz`として[`libreyolo predict`](/docs/cli/predict)に渡すと、検出結果に識別名と一致スコアが付きます。

## compare

2枚の顔画像のコサイン類似度と、それが同一人物のしきい値を超えるかどうかを報告します。

```bash
libreyolo compare model=<embedder> source=<a.jpg> source2=<b.jpg>
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `model` | | 顔の埋め込みベクトルのモデル。パスまたは名前。必須 |
| `source` | | 1枚目の画像。必須 |
| `source2` | | 比較対象となる2枚目の画像。必須 |
| `face_detector` | | 顔検出器：YuNetの`.onnx`、またはLibreYOLOの検出器 |
| `threshold` | `0.4` | 同一人物と判定するためのコサイン類似度のしきい値 |
| `device` | `auto` | デバイス：`0`、`cpu`、`mps`、`auto` |
| `json` | `false` | stdoutにJSONを出力 |
| `quiet` | `false` | stderrを抑制 |

```bash
libreyolo compare model=librefacerec-l.onnx source=a.jpg source2=b.jpg
```

`libreyolo verify`はこのコマンドの2つ目の名前として登録されており、同じ引数を取ります。

`compare`と`enroll`はどちらも、タスクが顔の埋め込みベクトルであるモデルを必要とします。それ以外は`config_unsupported`で終了します。ソースにはローカルの画像パスと、`http`または`https`のURLのどちらも指定できます。

## 例

<code-tabs name="examples" />

## 補足

結果はstdoutに出力され、進捗と警告はstderrに出力されます。`json=true`は`schema_version`を持つ1つのオブジェクトを表示し、これがスクリプトから読み取るための形式です。テキスト出力がデフォルトで、人が読むためのものです。

終了コードはCLIの他の部分と同じ対応です：成功なら`0`、使用方法または設定のエラーなら`2`、ソースが見つからないときは`3`、モデルまたはチェックポイントを読み込めないときは`4`、その他の実行時エラーは`1`です。

関連：データセット側の確認コマンドである[`libreyolo doctor`](/docs/cli/doctor)と、性能側の[`libreyolo profile`](/docs/cli/profile)。
