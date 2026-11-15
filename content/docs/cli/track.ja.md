---
title: libreyolo track
seo_title: libreyolo track コマンドリファレンス
description: コマンドラインから動画や画像シーケンスに対して物体追跡を行います：デフォルト値付きのすべての引数、4種類のトラッカー、1フレーム1行のJSON出力。
lead: >-
  動画ファイル、画像ディレクトリ、画像リストに対してmodel.track()を実行し、各フレームのトラックIDを1フレームにつき1行で出力します。1.8.0の新機能で、チェックポイントでもエクスポートした成果物でも同じように動作します。
keywords:
  - libreyolo track cli
  - libreyolo 追跡 コマンド
  - bytetrack cli
  - botsort コマンドライン
  - 複数物体追跡 cli
  - libreyolo track json
last_verified: "1.8.0"
meta:
  - label: コマンド
    value: libreyolo track
    mono: true
  - label: 必須
    value: source
    mono: true
  - label: 出力
    value: stdoutに1フレームにつき1行。save=trueのときはアノテーション済みの動画をruns/track/の下に出力
snippets:
  examples:
    - label: 基本
      language: bash
      code: |
        libreyolo track model=LibreYOLO9s.pt source=video.mp4
    - label: 別のトラッカー、動画を保存
      language: bash
      code: |
        libreyolo track model=LibreYOLO9s.pt source=video.mp4 \
          tracker=botsort classes="[0]" save=true
    - label: エクスポートしたモデル、フレームごとのJSON
      language: bash
      code: |
        libreyolo track model=weights/LibreYOLO9s.onnx source=frames/ fps=25 \
          json=true quiet=true
source_hash: 65b603ce6ee7a80f
---

## 書式

```bash
libreyolo track source=<video|directory|image|a.jpg,b.jpg> [model=<name|path>] [key=value ...]
```

引数は`key=value`の組で指定し、POSIX形式も使えるため、`tracker=botsort`と`--tracker botsort`は同じ引数です。

## 引数

| 引数 | デフォルト | 意味 |
|---|---|---|
| `source` | | 動画ファイル、画像ディレクトリ、画像パス、またはカンマ区切りの`a.jpg,b.jpg`形式の画像リスト。必須 |
| `model` | `yolox-s` | モデル名またはパス。エクスポートした成果物も使えます |
| `tracker` | `bytetrack` | `bytetrack`、`botsort`、`ocsort`、`deepocsort`のいずれか。`.yaml`の表記でも同じ組み込みトラッカーが選ばれます |
| `track_conf` | `0.25` | トラッカーの第1段階の対応付けにおける信頼度のしきい値 |
| `conf` | | 検出のしきい値。未設定の場合、検出器はトラッカー自身の低いしきい値で実行され、弱いボックスも復帰用に残ります |
| `iou` | `0.45` | NMSのIoUしきい値 |
| `imgsz` | | 入力画像サイズ：`640`（正方形）または`480x640`（HxW）。未設定ならモデル自身の入力サイズ |
| `classes` | | クラスIDで絞り込みます（例：`[0,2,5]`）。整数1つだけでも受け付けます |
| `max_det` | `300` | 1フレームあたりの最大検出数 |
| `save` | `false` | アノテーション済みの動画を保存 |
| `show` | `false` | 追跡したフレームを表示。`q`で停止 |
| `vid_stride` | `1` | Nフレームごとに処理 |
| `fps` | `30` | 画像シーケンスで想定するフレームレート。動画では無視されます |
| `output_path` | | 出力動画のパス。デフォルトは`runs/track/<name>.mp4`で、実行が別の実行を上書きしないよう番号が付きます |
| `color_format` | `auto` | 入力の色形式：`auto`、`rgb`、`bgr` |
| `device` | `auto` | デバイス：`0`、`cpu`、`mps`、`auto` |
| `json` | `false` | stdoutへのJSON出力。1フレームにつき1オブジェクト |
| `quiet` | `false` | stderrを抑制 |
| `verbose` | `false` | stderrへの詳細出力 |
| `help_json` | `false` | コマンドのスキーマをJSONで出力して終了 |

## 例

<code-tabs name="examples" />

## 注意事項

各実行は1つのソースを最初のフレームから追跡するため、`persist`オプションはありません：トラックIDがコマンド間で引き継がれることはありません。ライブソースは`config_unsupported`で拒否されます：Webカメラとストリームです。動画ファイル、画像ディレクトリ、画像リストを使ってください。

人が読む形式の出力は1フレームにつき1行で、たとえば`frame 12 video.mp4: 1920x1080 3 tracks (ids 1, 2, 5)`のようになり、stderrの最後にはフレーム数と一意なトラックIDの数が出ます。`json=true`では、各フレームが1つのJSONオブジェクトになり、`tracker`、`frame_index`、`detections`リストを持ちます。このリストの各ボックスは`track_id`、`class`、`class_id`、`confidence`、`bbox_xyxy`を持ちます。

関連付けるボックスを持たないタスクのモデルや、`track()`を持たないモデルタイプは`config_unsupported`で終了します。終了コードは[`libreyolo predict`](/docs/cli/predict)に従います：成功時は`0`、使い方または設定のエラーは`2`、ソースが見つからない場合は`3`、モデルを読み込めない場合は`4`、その他の実行時エラーは`1`です。

トラッカー、その設定、Pythonの`track()`呼び出しについては[物体追跡](/docs/tasks/object-tracking)で説明しています。
