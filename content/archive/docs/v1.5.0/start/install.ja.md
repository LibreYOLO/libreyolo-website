---
title: インストール
seo_title: LibreYOLOをインストール
description: >-
  PyPIからLibreYOLOをインストールし、モデルファミリーやエクスポート先に必要な任意の追加パッケージを選び、PyTorchがGPUを認識していることを確認します。
lead: >-
  LibreYOLOはlibreyoloとしてPyPIで公開されています。基本パッケージは予測、学習、検証と、PyTorch以外を必要としないモデルファミリーに対応し、任意の追加パッケージでその他の機能を導入できます。
keywords:
  - LibreYOLO インストール
  - pip install LibreYOLO
  - LibreYOLO extras
  - LibreYOLO CUDA
  - LibreYOLO GPU
  - LibreYOLO 必要環境
last_verified: 1.5.0
meta:
  - label: パッケージ
    value: libreyolo
    mono: true
  - label: Python
    value: 3.10以降
  - label: コードライセンス
    value: MIT
  - label: コア依存関係
    value: PyTorch 2.4以降
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install libreyolo
    - label: 追加パッケージを含める
      language: bash
      code: |
        # 複数を一度にインストールするにはカンマで区切ります。
        pip install "libreyolo[rfdetr,onnx]"
    - label: すべて
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: ソースからインストール
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # Python、PyTorch、CUDA、cuDNN、認識されている全GPU、インストール済みの
        # 任意パッケージを表示します。
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: モデル一覧
      language: bash
      code: |
        # 登録済みの各ファミリーと、タスク、サイズ、入力解像度を表示します。
        # 追加パッケージがないファミリーには、有効化するpipコマンドも表示されます。
        libreyolo models
source_hash: 34fc6d3e24d03fb4
---

## インストール

<code-tabs name="install" />

Python 3.10以降が必要です。基本インストールにはPyTorch、torchvision、NumPy、Pillow、OpenCV、PyYAML、requests、mss、tqdm、pycocotools、typer、click、safetensors、SciPyが含まれます。このため、YOLOv9や追加依存関係のないその他のファミリーは、`pip install libreyolo`の直後から動作します。

cloneすると、安定版ブランチ`release`がcheckoutされます。このブランチのコードが本ドキュメントと一致します。未公開の作業を含む統合ブランチは`dev`です。

## 任意の追加パッケージ

追加パッケージは角括弧付きの名前で、モデルファミリーまたはエクスポート先に必要な依存関係を追加します。その他は変わらず、追加パッケージの有無にかかわらずAPIは同じです。

### モデルファミリー

| 追加パッケージ | 追加内容 |
|---|---|
| `rfdetr` | RF-DETRのバックボーンを提供する`transformers` |
| `eomt` | `transformers` |
| `midas` | MiDaSのViT-L/16およびEfficientNet-Lite3エンコーダーを提供する`timm` 1.0.x |
| `vlm` | `transformers`、`num2words`、`decord`、`lmdb`、`peft` |
| `sam` | `transformers`、`timm` |
| `openvocab` | `transformers`、`timm`、`regex`、`ftfy` |
| `sensenova` | `transformers`、`accelerate`と、macOS以外では`bitsandbytes` |
| `modus` | `transformers`、`accelerate` |
| `clip` | 同梱のCLIP text tokenizerに必要な`regex`と`ftfy` |
| `siglip2` | 多言語SigLIP 2 tokenizerに必要な`sentencepiece` |
| `gaze` | L2CSチェックポイントの自動ダウンロードを有効にする`gdown` |
| `rtdetr` | なし。RT-DETRに追加依存関係はありませんが、名前を安定して維持 |

### エクスポートとランタイム

| 追加パッケージ | 追加内容 |
|---|---|
| `onnx` | `onnx`、`onnxsim`、`onnxruntime` |
| `tensorrt` | macOS以外で`tensorrt-cu12` 10.16.1.11と`pycuda` |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`。macOSのみ |
| `tflite`（エイリアス`litert`） | `libreyolo[onnx]`に加え、`onnx2tf`、`ai-edge-litert`、`onnx-graphsurgeon`、`onnx-simplifier` |
| `mnn` | `libreyolo[onnx]`と`MNN` |
| `ncnn` | `pnnx`と`ncnn` |
| `paddle` | `libreyolo[onnx]`に加え、`paddlepaddle` 2.6.2と`x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | HTTPおよびHTTPS V2推論用の`tritonclient[http]` |

### 学習、評価、ロギング

| 追加パッケージ | 追加内容 |
|---|---|
| `lora` | `lora=True`のファインチューニング用に`libreyolo[rfdetr]`と`peft` |
| `plots` | `matplotlib` |
| `fast-eval` | C++ COCO評価バックエンドの`faster-coco-eval` |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`（エイリアス`dvc`） | `dvclive` |

`fast-eval`が必須依存関係ではなく明示的な追加になっているのは、ビルド済みwheelがないプラットフォームでも通常インストールを失敗させないためです。パッケージがない場合、COCO評価はpycocotoolsへフォールバックし、実行を継続します。

### ツール

| 追加パッケージ | 追加内容 |
|---|---|
| `stream` | YouTubeページURLの解決にだけ必要な`yt-dlp` |
| `tracking` | なし。すべてのtracking依存関係はコア依存関係に含まれる |
| `label` | `libreyolo label`でクリックによるマスク補助を有効にする`libreyolo[sam]` |
| `hub-kernels` | コンパイル済みHubカーネル用の任意ローダー`kernels`。インストールによりRF-DETRの予測が浮動小数点の許容範囲内で変わる可能性については[カーネル](/docs/reference/kernels)を参照 |
| `clip-convert` | 重み変換と同等性確認用の`libreyolo[clip]`と`open_clip_torch` |
| `siglip2-convert` | 同じ目的の`libreyolo[siglip2]`と`transformers` |

Webカメラ、RTSP、RTMP、TCP、UDP、HLS、ローカルのマルチストリームリストに追加パッケージは不要です。必要なのはYouTubeページURLだけです。

### 集約追加パッケージ

`libreyolo[all]`はモデル、エクスポート、tracking、ロギングの追加パッケージを1つのコマンドでインストールします。意図的に除外されるものもあります。`neptune`は、安定版`neptune-scale`がprotobuf 7未満を要求する一方、TFLite経路がprotobuf 7を要求するため除外されます。`executorch`は組み合わせるPyTorchバージョンを制限し、`coreai`は`coreai-torch`がPyTorchを2.11.xへ固定して環境全体をそのバージョンへ移行させるため除外されます。`fast-eval`、`hub-kernels`、`clip-convert`、`siglip2-convert`も含まれません。必要なものを名前でインストールしてください。

## プラットフォームの制約

3つの追加パッケージは依存関係マーカーでプラットフォームが限定されています。このためインストール自体はどこでも成功し、wheelが存在しない環境では導入されるものが少なくなります。

| 追加パッケージ | 制約 |
|---|---|
| `coreai` | macOSのみ。Core AIツールチェーンはほかの環境で変換も実行もできない |
| `tensorrt` | CUDAのないmacOSではスキップ |
| `tflite`、`litert` | `onnx2tf`と`ai-edge-litert`にPython 3.12以降が必要 |

`sensenova`はwheelが公開されていないmacOSで`bitsandbytes`をスキップし、その他は通常どおりインストールします。

ディスク容量が制約なら、その大半はPyTorchで、さらに大半はデフォルトwheelに同梱されるCUDAペイロードです。CPU専用wheelなら機能を失わずにそれを除去できます。PyTorchを一切含めないマシンでONNX物体検出を使う場合は、[軽量インストール](/docs/lightweight-install)を参照してください。

## GPUとCUDA

デバイスはモデル構築時に選択されます。デフォルトの`device="auto"`は、`torch.cuda.is_available()`がtrueならCUDA、次に`torch.backends.mps.is_available()`がtrueならMetal Performance Shaders、それ以外ではCPUを使用します。ライブラリ内のその他の箇所はハードウェアを検査しないため、PyTorchがGPUを認識できなければLibreYOLOも認識できません。

デバイスを固定するには、モデルまたは`predict`、`train`、`val`、`export`へ`device`を渡します。`"cpu"`、`"cuda"`、`"cuda:0"`、`"mps"`、`0`のような整数、`"0"`のような数字の文字列を受け付けます。最後の2つは`cuda:<n>`へ展開されます。

最初に`libreyolo checks`を実行してください。PyTorchバージョン、PyTorchのビルド対象CUDAおよびcuDNNバージョン、認識されている各GPUとメモリを表示します。NVIDIA GPUを搭載するマシンでCUDAなしと報告される場合、pipが解決したPyTorch wheelはCPUビルドです。先にPyTorchのインデックスからCUDAビルドをインストールし、次にLibreYOLOをインストールしてください。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

これはリポジトリ自身がLinuxとWindowsのuv管理環境で固定しているものと同じインデックスです。CUDA 12.8ランタイムの要件としてNVIDIA driver 555以降が必要です。PyTorchのダウンロードホストはDarwinビルドを公開していないため、macOSではPyPI wheelを維持します。

## インストールの確認

<code-tabs name="verify" />

追加パッケージが有効になったか確認する最も速い方法は`libreyolo models`です。依存関係がないファミリーには、有効化する正確なpipコマンドが表示されます。どちらのコマンドも`--json`を受け付け、同じデータを機械可読オブジェクトとしてstdoutへ表示します。
