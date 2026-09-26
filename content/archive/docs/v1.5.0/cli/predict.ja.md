---
title: libreyolo predict
seo_title: libreyolo predict コマンドリファレンス
description: コマンドラインから推論を実行します：すべての引数、CLI定義から読み取られるデフォルト値、そしてstdoutに出る内容を変えるフラグ。
lead: >-
  読み込んだモデルを1つのソースに対して実行し、推論結果を出力します。ソースは画像、ディレクトリ、動画、URL、ライブストリームのいずれかであり、モデルはチェックポイントでもエクスポートした成果物でもかまいません。
keywords:
  - libreyolo predict cli
  - libreyolo 推論 コマンド
  - yolo コマンドライン 推論
  - libreyolo predict 引数
  - libreyolo json 出力
last_verified: 1.5.0
meta:
  - label: コマンド
    value: libreyolo predict
    mono: true
  - label: 必須
    value: source
    mono: true
  - label: 出力
    value: 推論結果をstdoutに出力。save=trueのときはアノテーション済みファイルをruns/detect/predictの下に出力
snippets:
  examples:
    - label: 基本
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: アノテーション済み画像を保存
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=true \
          project=runs/detect name=parkour exist_ok=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: クラスを絞り込み、JSONをstdoutへ
      language: bash
      code: >
        # チェックポイント同梱のCOCOクラスリストではクラス0がperson

        libreyolo predict model=LibreYOLO9s.pt classes="[0]" conf=0.4 max_det=50
        \
          json=true quiet=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: 7e46c7ed7dd9e6c4
---

## 書式

```bash
libreyolo predict source=<path|url|index> [model=<name|path>] [key=value ...]
```

引数は`key=value`のペアです。同じコマンドはPOSIX形式も受け付けるため、
`conf=0.4`と`--conf 0.4`は互換で、`save=true`と書いた真偽値は`--save`に
なります。アンダースコアを含む名前はどちらの綴りでも受け付けます：
`max_det=50`と`--max-det 50`は同じオプションに届きます。

`libreyolo detect predict ...`も受け付けられ、動作は同一です。タスク名は解析
前に取り除かれます。

## 引数

| 引数 | デフォルト | 意味 |
|---|---|---|
| `source` | | 画像のパス、ディレクトリ、またはURL。必須 |
| `model` | `yolox-s` | モデル名またはパス |
| `conf` | `0.25` | 信頼度のしきい値 |
| `iou` | `0.45` | NMSのIoUしきい値 |
| `imgsz` | | 入力画像サイズ：`640`（正方形）または`480x640`（高さ×幅）。未指定のときはモデル自身の入力サイズ |
| `classes` | | クラスIDによる絞り込み。例：`[0,2,5]`。整数1つの指定も可 |
| `max_det` | `300` | 画像1枚あたりの最大検出数 |
| `half` | `false` | FP16推論（CUDAのみ、モデル側の対応が必要） |
| `save` | `false` | アノテーション済み画像の保存 |
| `batch` | `1` | ディレクトリをソースにしたときの1回の順伝播あたりの画像数。1より大きいと、対応するモデルでは実際のバッチ推論を実行 |
| `stream` | `false` | 結果を逐次生成。Webカメラとライブストリームでは自動的に有効 |
| `stream_buffer` | `false` | 最新の1フレームだけを保持するのではなく、ライブの全フレームをバッファリング |
| `vid_stride` | `1` | 動画またはライブのNフレームごとに1フレームを処理 |
| `show` | `false` | 動画とライブの結果を表示。`q`で停止 |
| `tiling` | `false` | 大きな画像に対するタイル分割推論 |
| `overlap_ratio` | `0.2` | タイルの重なり比率 |
| `output_path` | | 出力先の明示指定。指定しない場合、`save=true`のときは`project/name` |
| `color_format` | `auto` | 入力の色：`auto`、`rgb`、`bgr` |
| `output_file_format` | | 出力形式：`jpg`、`png`、`webp` |
| `device` | `auto` | デバイス：`0`、`cpu`、`mps`、`auto` |
| `face_detector` | | 顔検出モデル（パスまたはCLI名）。視線推定モデルでは必須 |
| `gallery` | | 顔を照合するための`libreyolo enroll`製の顔ギャラリー`.npz`。顔の埋め込みベクトルモデル専用 |
| `gallery_threshold` | `0.4` | ギャラリー内の人物と一致とみなすコサインしきい値 |
| `project` | `runs/detect` | 出力ディレクトリのルート |
| `name` | `predict` | 実験名 |
| `exist_ok` | `false` | 既存の出力ディレクトリの再利用 |
| `json` | `false` | stdoutへのJSON出力 |
| `quiet` | `false` | stderrの抑制 |
| `verbose` | `false` | stderrへの詳細出力 |
| `help_json` | `false` | コマンドのスキーマをJSONで出力して終了 |

## 使用例

<code-tabs name="examples" />

## 補足

エクスポートした成果物はチェックポイントと同じ方法で読み込まれるため、
`model=weights/LibreYOLO9s.onnx`と`model=weights/LibreYOLO9s.engine`はどちらも
`model`の有効な値です。これらのランタイムでは、3つのオプションが無視されるの
ではなく拒否されます：ランタイムのバックエンドが`tiling`、`overlap_ratio`、
`output_file_format`を扱えない場合、`config_unsupported`で終了します。

`half`は逆です。エクスポートしたランタイムはこれを受け取ってFP16で実行し、
ネイティブのPyTorch推論は無視したことをログに記録してFP32のまま続行します。

視線推定モデルは2段構成で自前の検出器を持たないため、`face_detector`の指定が
必須です。`gallery`はタスクが`embed`のモデルにだけ適用され、それ以外に渡すと
`config_unsupported`で終了します。

stdoutには結果だけが流れ、進捗、警告、エラーはstderrに送られます。`json=true`
は1回の実行につき1つのJSONオブジェクトを、ストリーミング時はフレームごとに
1つを出力し、いずれにも`schema_version`が含まれます。`quiet=true`はstderrを
黙らせます。両方を併用すると、機械的な読み手にきれいなstdoutストリームを渡せ
ます。

終了コードは、成功時が`0`、使い方や設定の誤りが`2`、ソースが見つからないときが
`3`、モデルを読み込めないときが`4`、その他の実行時エラーが`1`です。

`help_json=true`は何も実行せずにコマンドのパラメータ、型、デフォルト値、
フラグをJSONで出力し、インストール済みのバージョンからこの表を読み返す確実な
方法になります。

関連：データセット上で指標を実測するには[`libreyolo val`](/docs/cli/val)、上で
挙げたランタイム成果物を生成するには[`libreyolo export`](/docs/cli/export)。
