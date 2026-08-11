---
title: インストール
seo_title: LibreYOLOをインストール
description: >-
  PyPIからLibreYOLOをインストールし、モデルファミリーやエクスポート先に必要な任意追加パッケージを選び、PyTorchがGPUを認識することを確認します。
lead: >-
  LibreYOLOはlibreyoloとしてPyPIで公開されています。基本パッケージは推論、学習、検証と、PyTorch以外を必要としないモデルファミリーに対応し、任意追加パッケージで残りの機能を追加できます。
keywords:
  - LibreYOLO インストール
  - pip install LibreYOLO
  - LibreYOLO 追加パッケージ
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
  - label: 中核依存関係
    value: PyTorch 2.4以降
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install libreyolo
    - label: 追加パッケージを指定
      language: bash
      code: |
        # 複数を1回でインストールするにはカンマで区切る
        pip install "libreyolo[rfdetr,onnx]"
    - label: すべて
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: ソースから
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # Python Torch CUDA cuDNN 認識されている全GPU
        # インストール済みの任意パッケージを表示する
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: モデル一覧
      language: bash
      code: |
        # 登録済みの全ファミリーについてタスク サイズ 入力解像度を表示する
        # 追加パッケージがないファミリーには有効化するpipコマンドも表示される
        libreyolo models
source_hash: 34fc6d3e24d03fb4
---

## インストール

<code-tabs name="install" />

Python 3.10以降が必要です。基本インストールではPyTorch、torchvision、NumPy、Pillow、OpenCV、PyYAML、requests、mss、tqdm、pycocotools、typer、click、safetensors、SciPyが導入されます。そのため、YOLOv9など追加依存関係のないファミリーは`pip install libreyolo`の直後から動作します。

クローンでは、このドキュメントと一致する安定版ブランチ`release`がチェックアウトされます。未リリースの作業を含む統合ブランチは`dev`です。

## 任意追加パッケージ

追加パッケージは角括弧で囲んだ名前で、モデルファミリーまたはエクスポート先が必要とする依存関係を追加します。それ以外は変わらず、追加パッケージの有無にかかわらずAPIは同じです。

### モデルファミリー

| 追加パッケージ | 追加内容 |
|---|---|
| `rfdetr` | RF-DETRバックボーンを提供する`transformers` |
| `eomt` | `transformers` |
| `midas` | MiDaSのViT-L/16とEfficientNet-Lite3エンコーダーを提供する`timm` 1.0.x |
| `vlm` | `transformers`、`num2words`、`decord`、`lmdb`、`peft` |
| `sam` | `transformers`、`timm` |
| `openvocab` | `transformers`、`timm`、`regex`、`ftfy` |
| `sensenova` | `transformers`、`accelerate`、macOS以外では`bitsandbytes` |
| `modus` | `transformers`、`accelerate` |
| `clip` | 同梱CLIPテキストトークナイザーに必要な`regex`と`ftfy` |
| `siglip2` | 多言語SigLIP 2トークナイザーに必要な`sentencepiece` |
| `gaze` | L2CSチェックポイントの自動ダウンロードを有効にする`gdown` |
| `rtdetr` | なし。RT-DETRに追加依存関係は不要ですが、名前の安定性のため維持 |

### エクスポートとランタイム

| 追加パッケージ | 追加内容 |
|---|---|
| `onnx` | `onnx`、`onnxsim`、`onnxruntime` |
| `tensorrt` | macOS以外で`tensorrt-cu12` 10.16.1.11と`pycuda` |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`、macOSのみ |
| `tflite`、別名`litert` | `libreyolo[onnx]`に加えて`onnx2tf`、`ai-edge-litert`、`onnx-graphsurgeon`、`onnx-simplifier` |
| `mnn` | `libreyolo[onnx]`に加えて`MNN` |
| `ncnn` | `pnnx`と`ncnn` |
| `paddle` | `libreyolo[onnx]`に加えて`paddlepaddle` 2.6.2と`x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | HTTPおよびHTTPS V2推論用の`tritonclient[http]` |

### 学習、評価、ロギング

| 追加パッケージ | 追加内容 |
|---|---|
| `lora` | `lora=True`のファインチューニング用に`libreyolo[rfdetr]`と`peft` |
| `plots` | `matplotlib` |
| `fast-eval` | C++のCOCO評価バックエンド`faster-coco-eval` |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`、別名`dvc` | `dvclive` |

`fast-eval`は必須依存関係ではなく任意です。ビルド済みwheelのないプラットフォームで通常のインストールが壊れないようにするためです。パッケージがない場合、COCO評価はpycocotoolsへフォールバックし、実行を続けます。

### ツール

| 追加パッケージ | 追加内容 |
|---|---|
| `stream` | YouTubeページURLの解決だけに必要な`yt-dlp` |
| `tracking` | なし。追跡の依存関係はすべて中核依存関係に含まれる |
| `label` | `libreyolo label`でクリックからマスクを生成する補助機能を有効にする`libreyolo[sam]` |
| `hub-kernels` | コンパイル済みHubカーネル用の任意ローダー`kernels`。インストールによりRF-DETRの推論結果が浮動小数点許容差の範囲で変わる可能性については[カーネル](/docs/reference/kernels)を参照 |
| `clip-convert` | 重み変換と一致度確認用の`libreyolo[clip]`と`open_clip_torch` |
| `siglip2-convert` | 同じ目的の`libreyolo[siglip2]`と`transformers` |

Webカメラ、RTSP、RTMP、TCP、UDP、HLS、ローカルのマルチストリームリストに追加パッケージは不要です。必要なのはYouTubeページURLだけです。

### 集約追加パッケージ

`libreyolo[all]`はモデル、エクスポート、追跡、ロギングの追加パッケージを1つのコマンドでインストールします。一部は意図的に含まれません。`neptune`は安定版`neptune-scale`がprotobuf 7未満を必要とする一方、TFLite経路はprotobuf 7を必要とするため除外されます。`executorch`は対応するPyTorchバージョンを制約し、`coreai`は`coreai-torch`がPyTorchを2.11.xへ固定して環境全体をそのバージョンへ移すため除外されます。`fast-eval`、`hub-kernels`、`clip-convert`、`siglip2-convert`も含まれません。必要なものは名前で個別にインストールしてください。

## プラットフォームの制約

3つの追加パッケージは依存関係マーカーでプラットフォームを限定します。そのため、wheelがない場所でもインストール自体は成功し、導入されるものが少なくなります。

| 追加パッケージ | 制約 |
|---|---|
| `coreai` | macOSのみ。Core AIツールチェーンは他では変換も実行もできない |
| `tensorrt` | CUDAのないmacOSでは省略 |
| `tflite`、`litert` | `onnx2tf`と`ai-edge-litert`にはPython 3.12以降が必要 |

`sensenova`はwheelが公開されていないmacOSで`bitsandbytes`を省略し、それ以外は通常どおりインストールします。

ディスク容量が制約になる場合、その大半はPyTorchで、PyTorchの大半はデフォルトwheelに同梱されるCUDAペイロードです。CPU専用wheelを使えば機能を失わず削減できます。torchを一切含めないマシンでONNX検出を実行する方法は[軽量インストール](/docs/lightweight-install)を参照してください。

## GPUとCUDA

デバイスはモデル構築時に選択されます。デフォルトの`device="auto"`は`torch.cuda.is_available()`がtrueならCUDA、次に`torch.backends.mps.is_available()`がtrueならMetal Performance Shaders、それ以外はCPUを使用します。ライブラリの他の場所でハードウェアを調べることはないため、PyTorchがGPUを認識できなければLibreYOLOも認識できません。

デバイスを固定するには、モデルまたは`predict`、`train`、`val`、`export`へ`device`を渡します。`"cpu"`、`"cuda"`、`"cuda:0"`、`"mps"`、`0`のような整数、`"0"`のような数字の文字列を受け付けます。最後の2つは`cuda:<n>`へ展開されます。

最初に`libreyolo checks`を実行してください。Torchのバージョン、Torchのビルドに使われたCUDAとcuDNNのバージョン、認識されているすべてのGPUとメモリを表示します。NVIDIAカードがあるマシンでCUDAなしと表示される場合、pipが解決したPyTorch wheelはCPUビルドです。先にPyTorchのインデックスからCUDAビルドをインストールし、その後にLibreYOLOをインストールしてください。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

これはリポジトリがLinuxとWindowsのuv管理環境で固定しているものと同じインデックスです。CUDA 12.8ランタイムの要件としてNVIDIAドライバー555以降が必要です。PyTorchのダウンロードホストはDarwin向けビルドを公開していないため、macOSではPyPIのwheelを維持します。

## インストールを確認

<code-tabs name="verify" />

`libreyolo models`は追加パッケージが有効になったか確認する最速の方法です。依存関係のないファミリーには、有効化する正確なpipコマンドが表示されます。どちらのコマンドも`--json`を受け付け、同じデータを機械可読なオブジェクトとして標準出力へ表示します。
