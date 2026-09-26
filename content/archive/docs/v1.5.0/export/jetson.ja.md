---
title: NVIDIA Jetson
seo_title: NVIDIA JetsonにLibreYOLOとPyTorchをインストールする
description: >-
  NVIDIA
  JetsonにLibreYOLOをインストールします：JetPackが省いている4つのCUDAライブラリ、PyTorchに必要な--no-depsの手順、そしてOrin
  Nanoでの実測値。
lead: >-
  NVIDIA JetsonのボードはLibreYOLOを標準のaarch64向けPyTorch
  wheelで動かせます。Jetson専用のtorchビルドは不要ですが、JetPackはtorchがリンクする4つのライブラリを省いているため、インストール時にこちらで補う必要があります。
keywords:
  - NVIDIA Jetson
  - Jetson Orin Nano
  - JetPack 7.2
  - jetson pytorch インストール
  - nvidia-cudnn-cu13
  - nvidia-nccl-cu13
  - nvidia-cusparselt-cu13
  - nvidia-nvshmem-cu13
  - torch.cuda.is_available
  - no kernel image is available for execution on the device
  - jetson tensorrt エクスポート
  - aarch64 wheel
last_verified: 1.4.0
meta:
  - label: ボード
    value: Jetson Orin Nano Super Developer Kit、8 GB、GPUのcompute capability 8.7
  - label: プラットフォーム
    value: JetPack 7.2（L4T R39.2）、Ubuntu 24.04、CUDA 13、Python 3.12.3、aarch64
  - label: 検証したスタック
    value: >-
      libreyolo 1.4.0、torch 2.13.0+cu130、torchvision 0.28.0+cu130、opencv
      5.0.0、numpy 2.5.1、2026-07-27時点
  - label: JetPackに含まれないもの
    value: >-
      nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-cusparselt-cu13,
      nvidia-nvshmem-cu13
    mono: true
  - label: ベンチマーク
    value: このボードでの検証済み実行223件、12ファミリーにまたがる58モデルを、PyTorch、ONNX Runtime、TensorRTで計測
    links:
      - label: visionanalysis.org/hardware/jetson_orin
        href: 'https://www.visionanalysis.org/hardware/jetson_orin'
  - label: 追跡先
    value: issue 648のJetson側
    links:
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
verification: >-
  インストール手順と想定される出力は、Jetson Orin Nano
  Superで2026-07-27に実施したインストール作業から採録しました。レイテンシと精度の行は、visionanalysis.orgの背後にある検証済み結果のスナップショットから、hardware
  jetson_orinで絞り込んだもので、libreyolo
  1.2.0.dev0を用いて2026年6月に計測しました。エクスポートとローダーの挙動は、libreyolo/export/exporter.py、libreyolo/export/tensorrt.py、libreyolo/models/__init__.pyを読んで確認しました。
snippets:
  prep:
    - label: システムパッケージと仮想環境
      language: bash
      code: |
        # JetPackはpipやvenvモジュールを同梱しない
        sudo apt update
        sudo apt install -y python3.12-venv python3-pip

        python3 -m venv ~/libreyolo
        source ~/libreyolo/bin/activate
        pip install -U pip wheel setuptools
  torch:
    - label: PyTorchをCUDA 13のwheelインデックスから
      language: bash
      code: |
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu130 \
          --extra-index-url https://pypi.org/simple
    - label: JetPackが同梱しない4つのライブラリ
      language: bash
      code: |
        pip install nvidia-cudnn-cu13 nvidia-nccl-cu13 \
                    nvidia-cusparselt-cu13 nvidia-nvshmem-cu13
    - label: pipがcuda-toolkit 13.0.3を要求する場合は--no-depsでインストール
      language: bash
      code: |
        # --no-depsではtorchのPython依存関係も手で指定する必要あり
        pip install --no-deps \
          torch torchvision \
          nvidia-cudnn-cu13 nvidia-nccl-cu13 \
          nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 \
          filelock typing_extensions sympy networkx jinja2 markupsafe mpmath \
          fsspec numpy pillow
  ldd:
    - label: 推測せずに次に足りないライブラリを特定
      language: bash
      code: >
        ldd
        "$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" \
          | grep "not found"

        # torchの全ライブラリでまだ足りないものを一度に列挙

        ldd "$VIRTUAL_ENV"/lib/python3.12/site-packages/torch/lib/*.so
        2>/dev/null \
          | grep "not found" | sort -u
  install:
    - label: LibreYOLOはtorchより先ではなく後にインストール
      language: bash
      code: |
        # torchは充足済みなのでpipはCUDAビルドをそのまま残す
        pip install libreyolo

        # ONNXのエクストラが必要なのはエクスポート時のみ
        # TensorRTのエクスポートもONNX経由なので下のエクスポート節の前に追加
        pip install "libreyolo[onnx]"
  verify:
    - label: バージョンとデバイス
      language: python
      code: |
        import cv2
        import numpy
        import torch

        import libreyolo

        print("torch", torch.__version__, "cuda", torch.cuda.is_available())
        print("gpu", torch.cuda.get_device_name(0))
        print("libreyolo", libreyolo.__version__)
        print("cv2", cv2.__version__, "numpy", numpy.__version__)
      expect: |
        torch 2.13.0+cu130 cuda True
        gpu Orin
        libreyolo 1.4.0
        cv2 5.0.0 numpy 2.5.1
    - label: 次に実際のカーネルを実行
      language: python
      code: |
        import torch

        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        # 初回利用時にチェックポイントをダウンロード
        model = LibreYOLO9("libreyolo9s.pt", size="s")

        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict --source
        https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        --model libreyolo9s.pt --save
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreYOLO9, SAMPLE_IMAGE


        # libreyolo9s.onnxを書き出してからそれをもとにlibreyolo9s.engineをビルド

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="tensorrt",
        half=True)


        # エンジンも同じエントリーポイントから読み込める

        result = LibreYOLO("libreyolo9s.engine").predict(SAMPLE_IMAGE)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model libreyolo9s.pt --format tensorrt --half
  power:
    - label: 電力モードとクロック
      language: bash
      code: |
        sudo nvpmodel -q      # このボードが公開するモードと現在有効なモード
        sudo nvpmodel -m 0    # ここで検証したボードでの最上位モード
        sudo jetson_clocks

        tegrastats            # 負荷をリアルタイム表示; Tegraではnvidia-smiの機能は限定的
source_hash: c07ff908503e89b5
---

## このページが記録していること

このページが記録しているのは、エンドツーエンドで検証した1つの構成であり、対応表
ではありません。ボードはメモリ8 GBのJetson Orin Nano Super Developer Kitで、
JetPack 7.2（L4T R39.2、Ubuntu 24.04、CUDA 13、Python 3.12.3）が動作しており、その
上で立ち上がったスタックは`libreyolo 1.4.0`と`torch 2.13.0+cu130`、OpenCV 5.0.0、
NumPy 2.5.1でした。`torch.cuda.is_available()`は`True`を返し、GPUは自身を`Orin`と
報告しました。

これ以外のJetPackリリース、これ以外のJetsonボード、これ以外のCUDAバージョンは検証
していません。以下の手順は、その組み合わせで動作したものです。

この作業は2026-07-27にLibreYOLO 1.4.0に対して実施したもので、1.5.0のハードウェア
では再実施していません：1.5.0のツリーで1.4.0の検証を残しているのはこのページだけ
であり、フロントマターが`last_verified: "1.4.0"`となっているのはそのためです。
1.5.0の変更点は、ここで説明するインストール経路、4つの不足ライブラリ、エクスポート
のフラグのいずれにも影響しないため、コマンドはそのまま通用する見込みですが、以下の
出力に含まれるバージョン番号は1.4.0が表示したものであり、1.5.0での計測ではありま
せん。

この構成には、多くのJetsonガイドが述べていることと食い違う点が2つあります。wheelは
CUDA 13向けに公開されている通常のaarch64ビルドなので、Jetson専用のtorchビルドは必要
ありません。そしてJetPackは、それらのwheelがリンクする4つのライブラリを同梱していない
ため、4つすべてを入れるまで`import torch`は1つずつライブラリ名を挙げて失敗します。

## インストール

JetPackのイメージにはpipも`venv`モジュールも入っていないため、まずはこの2つから
です。

<code-tabs name="prep" />

8 GBのボードは、大きめのチェックポイントには手狭です。読み込む前にNVMe上にスワップ
を追加しておくと、実行中にメモリ不足で強制終了されるのを避けられます。

次にPyTorchです。CUDA 13のインデックスがaarch64のwheelを提供し、追加インデックスが
PyPIからピュアPythonの依存関係を供給します。

<code-tabs name="torch" />

4つの`nvidia-*-cu13`のwheelは見落としやすい部分です。JetPackが提供するのはGPU
ドライバーであり、cuDNN、NCCL、cuSPARSELt、NVSHMEMは含まれておらず、これらがないと
torchはインポートを拒否します。4つを一度にインストールするほうが、例外を1つずつ
潰しながら見つけていくより速く済みます。

3つ目のスニペットは特定の失敗に対応するものです：CUDA 13ビルドのtorchの依存関係
メタデータは`cuda-toolkit==13.0.3`を要求しますが、これにはPyPI上にaarch64のwheelが
ないため、何もダウンロードされないうちに依存解決が失敗します。`--no-deps`は依存
解決を飛ばすので、すべての依存関係をコマンドラインで指定する必要があります。

LibreYOLOは最後に入れます。先にインストールすると、pipが自前でtorchを選んでしまい、
このプラットフォームではそれはCUDAビルドではありません。

<code-tabs name="install" />

残りの依存関係は、OpenCV、NumPy、SciPy、pycocotools、safetensorsを含め、すべて
ビルド済みのaarch64のwheelとして解決されます。ソースからのコンパイルは発生しません。

## CUDAが動作するか確認する

<code-tabs name="verify" />

2つ目のスニペットは1つ目と同じくらい重要です。GPUアーキテクチャが合っていないwheel
でも`torch.cuda.is_available() == True`とは報告され、そのうえで最初の実際の演算が
`CUDA error: no kernel image is available for execution on the device`で失敗します。
デバイス上での行列積が、これを捕まえる確認方法です。

## 推論を実行する

<code-tabs name="predict" />

`predict`は他のプラットフォームと同じ`Results`オブジェクトを返すため、各モデルの
ページはそのまま当てはまります。

## TensorRTへエクスポートする

このボードでは、すべてのランタイムで計測した55モデルすべてについて、TensorRTが
PyTorchとONNX Runtimeのどちらよりも高速でした。

<code-tabs name="export" />

`format="tensorrt"`はまずONNXのグラフを書き出し、そこからエンジンをビルドするため、
`onnx`のエクストラをインストールしておく必要があります。`LibreYOLO()`はファイルの
拡張子で処理を振り分けるので、`.engine`ファイルは`.pt`のチェックポイントと同じ
呼び出しで読み込めます。

Jetsonでは`tensorrt`のpipエクストラを使わないでください。これはCUDA 13の
プラットフォームに対して、CUDA 12ビルドである`tensorrt-cu12`を固定します。代わりに
JetPackがインストールするTensorRTを使ってください。仮想環境の外では動くのに中で
`import tensorrt`が失敗する場合は、システムのモジュールが見えるように
`--system-site-packages`を付けて環境を作り直してください。

シリアライズされたTensorRTのエンジンは、デバイス、GPUアーキテクチャ、そしてビルドに
使ったTensorRTのバージョンに紐づきます。ワークステーションでビルドしたエンジンは
Jetsonでは読み込めないため、ビルドの手順はボード上で実行します。

## このボードでの計測結果

1枚あたりのレイテンシ、バッチサイズ1、前処理と後処理を含むエンドツーエンドの値で、
COCO val2017（500枚のサブセット）に対し`conf=0.001`および`max_det=300`で計測しま
した。計測した58モデルのうち5つです：

| モデル | 入力（px） | PyTorch FP32（ms） | ONNX FP32（ms） | TensorRT FP32（ms） | TensorRT FP16（ms） | mAP 50-95 |
|---|---:|---:|---:|---:|---:|---:|
| DEIMv2-Atto | 320 | 64.9 | 22.8 | 12.3 | 11.2 | 27.49 |
| YOLOX-Tiny | 416 | 49.2 | 31.8 | 23.0 | 19.4 | 35.45 |
| YOLO9-t | 640 | 101.2 | 53.8 | 36.0 | 29.1 | 41.78 |
| RT-DETR-r18 | 640 | 98.3 | 103.7 | 45.3 | 25.7 | 49.72 |
| D-FINE-s | 640 | 96.8 | 96.1 | 44.7 | 33.1 | 53.45 |

mAPの列は、TensorRT FP16での実行そのもののスコアです。4つのランタイムすべてで計測
した55モデル全体で、PyTorch FP32のスコアとTensorRT FP16のスコアの差が最も大きかった
のはDEIMv2-Xの0.59ポイントでした。ランタイムによって違うのは速度であり、精度では
ありません。

TensorRT FP32は、その55モデルすべてでPyTorchとONNX Runtimeのどちらよりも高速でした。
TensorRT FP16も55モデルすべてでPyTorch FP32より高速で、その差は1.68〜6.22倍、中央値
は3.39倍でした。ばらつくのはONNX Runtimeです：55モデルのうち23モデルでPyTorchより
遅く、RT-DETR-r18の行もその1つです。

すべての数値の測定条件：`libreyolo 1.2.0.dev0`、`torch 2.12.0+cu130`、
Python 3.12.3、CUDA 13、ドライバー595.78、ONNX Runtime 1.24.0、2026年6月に計測。
Jetsonのレイテンシは有効な電力モードにも左右されますが、ベンチマークの記録にはその
情報が含まれていません。

<code-tabs name="power" />

他の53モデルと精度の全列を含む223件の実行はすべて、
[Vision AnalysisのJetson Orinページ](https://www.visionanalysis.org/hardware/jetson_orin)
で公開しています。

## トラブルシューティング

### import torchが共有ライブラリ名を挙げて失敗する

上記4つのライブラリのいずれかが足りていません。どれなのかを推測するのではなく、
バイナリから読み取ってください：

<code-tabs name="ldd" />

不足している項目は、それぞれ1つのwheelに対応します：

| 不足しているライブラリ | wheel |
|---|---|
| cuDNN | `nvidia-cudnn-cu13` |
| NCCL | `nvidia-nccl-cu13` |
| cuSPARSELt | `nvidia-cusparselt-cu13` |
| NVSHMEM | `nvidia-nvshmem-cu13` |

### このGPUに対応するビルドがないとtorchが警告する

動作する構成でも、最初のCUDA呼び出しで次のように表示されます：

```text
UserWarning: Found GPU0 Orin which is of compute capability (CC) 8.7.
The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:
- 8.0 which supports hardware CC >=8.0,<9.0 except {8.7}
- 9.0 which supports hardware CC >=9.0,<10.0
- 10.0 which supports hardware CC >=10.0,<11.0 except {10.1}
- 11.0 which supports hardware CC >=11.0,<12.0
- 12.0 which supports hardware CC >=12.0,<13.0
No published PyTorch CUDA builds for release 2.13.0+cu130 support this GPU.
```

このボードでは、この警告は見た目だけの問題です。wheelには`sm_80`のカーネルが含まれて
おり、Orinはそれを実行します。同じ警告は、上のベンチマークの各行を出した、同じ
インデックスの以前のwheelでも表示されていました。メッセージを信じるか疑うかではなく、
CUDAの確認で使った行列積で確かめてください。

### CUDA error: no kernel image is available for execution on the device

インストールしたwheelが別のGPUアーキテクチャ向けにビルドされています。これは
NVIDIAの`sbsa`インデックスのwheelで起きることで、これらはJetsonのシリコンではなく
サーバー向けARM GPUを対象としています。インストールの節にあるCUDA 13のインデックス
から入れ直してください。

### pipがcuda-toolkit 13.0.3を見つけられない

これにはaarch64のwheelがありません。インストールの節にある`--no-deps`の形を使い、
torchの依存関係を明示的に指定してください。

### libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file

aarch64版のtorchのwheelは、CPU演算のためにNVIDIA Performance Librariesをリンクします。
これらをインストールし、ライブラリパスに追加してください：

```bash
pip install nvpl-lapack nvpl-blas --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
```

このインデックスは、この2つのCPUライブラリについては問題ありません。上の
「no kernel image」の失敗を引き起こすのは、このインデックスのtorchビルドです。

### JetPack 7.2に適合しないwheelの入手元

| 入手元 | Orin Nano Superでの結果 |
|---|---|
| `pypi.jetson-ai-lab.io/sbsa/cu130`のtorch | サーバー向けARM GPU用にビルドされています。インポートは通り、CUDAは利用可能と報告されますが、「no kernel image is available for execution on the device」で失敗します。 |
| `pypi.jetson-ai-lab.io/jp6/*`のtorch | CUDA 12とPython 3.10向けのビルドです。このイメージのPython 3.12にはインストールできません。 |
| JetPack 6のPyTorchコンテナー | JetPack 7のホストではCUDAの初期化がエラー801で失敗します。 |
| torchをソースからビルド | 動作しますが、8 GBのボードでは数時間かかり、CUDA 13のwheelを入れてしまえば不要です。 |

## DeepStream

Pythonのループではなく本格的な動画パイプラインが必要な場合は、`deepstream=True`で
エクスポートし、グラフを`nvinfer`で実行してください。この経路には専用のページが
あり、生成される`nvinfer`の設定、バウンディングボックスのパーサーのビルド、既知の
落とし穴を扱っています：[DeepStream](/docs/export/deepstream)。

DeepStreamのパイプライン自体は、Jetsonではなくx86の独立GPUで検証しました。
エクスポートの取り決めはアーキテクチャに依存しませんが、aarch64でのパイプライン実行
は未実施です。

## 未検証の項目

- 7.2以外のJetPackリリース、およびR39.2以外のL4Tリリース。
- Orin Nano Super 8 GB以外のJetsonボード。
- ボード上での学習。推論とエクスポートは実施しましたが、学習の実行はしていません。
- INT8のエンジン。このボードにはFP32とFP16の行しかありません。
- 1を超えるバッチサイズ。上記の計測はすべてバッチ1です。
