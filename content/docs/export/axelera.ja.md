---
title: Axelera Metis
seo_title: LibreYOLOのモデルをAxelera Metisへエクスポート
description: >-
  LibreYOLOのモデルをAxelera Metis AIPU向けのINT8の.axmにコンパイルします：自分でインストールするSDK、エクスポートの引数、パリティチェック、コンパイルできるファミリーとサイズ。
lead: >-
  format="axelera"はLibreYOLOのモデルをAxelera社のMetis AIPU向けのINT8の.axmにコンパイルし、エクスポートしたディレクトリはLibreYOLO()で読み戻して、PyTorchモデルと同じpredict、val、trackの呼び出しで使えます。1.8.0の新機能で、プレビュー扱いです。
keywords:
  - libreyolo axelera
  - axelera metis
  - metis aipu
  - voyager sdk
  - axelera-devkit
  - axelera-rt
  - int8 npu エクスポート
  - axm モデル
last_verified: "1.8.0"
meta:
  - label: フォーマット
    value: 'export(format="axelera")'
    mono: true
  - label: 数値精度
    value: INT8のみです。half=True、quantize=16、quantize=32は例外を送出します。
  - label: コンパイル環境
    value: Linux x86_64（Ubuntu 22.04または24.04、あるいはWSL2）、Python 3.10〜3.13。コンパイルにMetisデバイスは不要です。
  - label: SDK
    value: Axelera社独自のパッケージインデックスから入手するaxelera-devkitとaxelera-rt 1.9.0。LibreYOLOの追加インストールではありません
  - label: 出力
    value: "weights/<stem>_axelera_model/"
    mono: true
  - label: ステータス
    value: プレビューです。23のファミリーを、SDK 1.9.0を使い1台のMetis M.2 Maxでコンパイル・計測しました。
verification: >-
  1.8.0リリース候補時点のdevブランチのdocs/axelera.md、libreyolo/export/axelera.py、libreyolo/export/support.py、libreyolo/cli/commands/export.pyを確認しました。精度の数値はライブラリ自身によるINT8とFP32の対応付き計測であり、このサイトでは再実行していません。
snippets:
  install:
    - label: 専用の環境
      language: bash
      code: |
        python -m venv ~/venvs/axelera && source ~/venvs/axelera/bin/activate
        pip install "libreyolo[onnx]"   # コンパイルにはonnxの追加インストールが必要

        # SDKは最後にインストール。そのwheelはLibreYOLOも使うパッケージのバージョンを固定する
        # コンパイル（エクスポート）用
        pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-devkit==1.9.0
        # 実行（.axmの推論）用
        pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-rt==1.9.0
    - label: LibreYOLOにインストールさせる
      language: bash
      code: |
        # これがない場合、SDKが見つからないと上記のpipコマンドを示して失敗
        export LIBREYOLO_AUTOINSTALL=1
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        path = model.export(format="axelera")   # weights/LibreYOLO9t_axelera_model

        # 自分のドメインでキャリブレーションし、パリティレポートを出力
        path = model.export(
            format="axelera",
            data="my_dataset.yaml",
            fraction=0.5,
            verify=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=axelera
        libreyolo export model=LibreYOLO9t.pt format=axelera verify=true
  run:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        npu = LibreYOLO("weights/LibreYOLO9t_axelera_model")   # またはその中の.axm
        results = npu(SAMPLE_IMAGE, conf=0.25)
        metrics = npu.val(data="coco128.yaml")
        for r in npu.track("video.mp4"):
            print(r.track_id)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=weights/LibreYOLO9t_axelera_model source=image.jpg
        libreyolo val model=weights/LibreYOLO9t_axelera_model data=coco128.yaml
        libreyolo track model=weights/LibreYOLO9t_axelera_model source=video.mp4
source_hash: 19ff2c735643dec7
---

## インストール

Axelera SDKはAxelera社のEULAに基づくプロプライエタリなソフトウェアで、Axelera社独自のパッケージインデックスから配布されています。PyPIにはなく、LibreYOLOの追加インストールでもなく、LibreYOLOがこれを同梱することもありません。`axelera-devkit`はコンパイルを行い、`axelera-rt`はコンパイル済みモデルを実行します。デプロイ先のマシンに必要なのは`axelera-rt`だけです。

<code-tabs name="install" />

専用の仮想環境を使ってください。SDK 1.9.0のwheelはLibreYOLOも使ういくつかのパッケージのバージョンを固定しており、pipはインストール時にそれらをダウングレードします：

| パッケージ | 固定されるバージョン |
|---|---|
| `torch` | `>=2.7,<2.15` |
| `typer` | `>=0.12,<0.13` |
| `click` | `==8.1.8` |
| `rich` | `<14` |
| `onnx` | `<1.19` |
| `opencv-python` | `<5` |
| `pillow` | `<13` |

LibreYOLOはこれらすべてのバージョンで動作します。SDKのあとに他のLibreYOLOの追加インストールを入れると再びアップグレードされることがあるため、SDKは最後にインストールしてください。パッケージが不足している場合、エクスポートや読み込みは正確な`pip install`コマンドを示して失敗します。`LIBREYOLO_AUTOINSTALL=1`を設定すると、LibreYOLOがそのコマンドを代わりに実行し、そのことをログに記録します。

実行には、Metisデバイス、Axelera社の`metis-dkms`カーネルドライバー（SDK 1.9では1.7.0以降）、`axelera-rt`が必要です。`axdevice`は各デバイスをファームウェアとともに一覧表示し、`libreyolo checks`はAxelera関連のパッケージと認識しているMetisデバイスを一覧表示します。

## エクスポート

<code-tabs name="export" />

| 引数 | Axeleraでの動作 |
|---|---|
| `imgsz` | コンパイル済みモデルに固定されます。32の倍数です。デフォルトはチェックポイントのサイズ |
| `data` | キャリブレーション用のデータセットYAML。検出と姿勢推定ではデフォルトは`coco128.yaml` |
| `fraction` | キャリブレーション用分割のうち使用する割合。100〜400枚の画像を推奨し、100枚未満では警告がログに出ます |
| `batch` | キャリブレーション用ローダーのバッチサイズのみ。コンパイル済みモデルは一度に1枚の画像を受け取ります |
| `device` | モデルのトレースに使うtorchのデバイスで、Metisデバイスではありません |
| `quantize`、`int8` | INT8は常に有効です。`quantize=16`、`quantize=32`、`half=True`は例外を送出します |
| `dynamic` | 常に静的です。`dynamic=True`は例外を送出します |
| `opset` | 中間ONNXグラフのopset。14〜17（デフォルトは17） |
| `nms` | 非対応：NMSはホストで実行されます |
| `output_path` | デフォルトは`weights/<checkpoint stem>_axelera_model` |
| `verify` | 量子化モデルをFP32と比較し、デバイスがあればデバイスとも比較した`<stem>.parity.json`を出力します |
| `compiler_config` | Pythonのみ：Axeleraの`CompilerConfig`フィールドのdict。未知のフィールドとLibreYOLOが管理するフィールドは例外を送出します |

キャリブレーションにはモデル自身の前処理が使われます。ファインチューニングしたモデルでは、そのモデル自身のドメインの画像でキャリブレーションしてください。コンパイルには数分から数十分かかり、数GBのRAMを使います。

```text
weights/LibreYOLO9t_axelera_model/
    LibreYOLO9t.axm               compiled model
    compiler_config_final.toml    compiler settings used for the build
    metadata.yaml                 LibreYOLO metadata
    LibreYOLO9t.parity.json       only with verify=True
```

ディレクトリ名はチェックポイントのファイル名から付けられます。YOLO-NASの学習済み重みは初回使用時に変換されるため、そのエクスポートは`weights/LibreYOLONASs-converted_axelera_model`に出力されます。同じパスに再エクスポートした場合、ディレクトリが置き換えられるのは新しいビルドが成功したあとだけで、ONNXファイルやコンパイラーのビルドファイルは残りません。

## 成果物を実行する

<code-tabs name="run" />

ネットワークはMetis AIPUで実行され、前処理、ボックスのデコード、NMSはホストで実行されるため、ボックスの座標とスコアが8ビットのテンソルを通ることはありません。結果はPyTorchモデルと同じ`Results`オブジェクトです。入力サイズはエクスポート時に固定されます。`device="auto"`、`"axelera"`、`"metis"`、`"aipu"`、`"cpu"`はいずれもAIPUで実行され、CUDAデバイスは警告とともに無視されます。ディレクトリは、Metisデバイスと`axelera-rt`がある任意のLinuxマシンにコピーして使えます。

## 対応ファミリー

以下のすべてのサイズはコンパイル、Metisデバイスでの実行、エクスポートができます。「1.0以内」は、対応付けたセットでINT8の結果がFP32から1.0ポイント以内であることを意味します。計測した低下幅がそれより大きいサイズもエクスポートでき、エクスポート時にその低下幅が一度だけ警告としてログに出ます。精度がエクスポートを妨げることはありません。各ファミリーとタスクはエクスポート対応表にそれぞれ`validated`または`available`のセルを持ち（`libreyolo formats --family yolonas`で表示されます）、記載のないものはコンパイル前に例外を送出します。

| ファミリー | タスク | サイズ | 計測した低下幅 |
|---|---|---|---|
| YOLO-NAS | detect | s, m, l | 1.0以内 |
| YOLO-NAS | pose | n, s, m, l | nとsはより大きい |
| YOLO-NAS-R | obb | s, m, l | 1.0以内。キャリブレーションデータが必要 |
| YOLO9 | detect | t, s, m, c | 1.0以内 |
| YOLOX | detect | n, t, s, m, l, x | nはより大きい |
| YOLOv2 | detect | t, b | 1.0以内 |
| YOLOv3 | detect | t, b, spp | 1.0以内 |
| YOLOv4 | detect | t, b | bはより大きい |
| YOLOv7 | detect | b | 1.0以内 |
| PP-YOLOE | detect | s, m, l, x | s、m、xはより大きい |
| PicoDet | detect | s, m, l | より大きい |
| RTMDet | detect | t, s, m, l | より大きい |
| RTMDet-Ins | segment | t, s, m, l | より大きい |
| RetinaNet | detect | r50 | より大きい |
| SSD | detect | 300 | 1.0以内 |
| EfficientDet | detect | d0, d1, d2 | より大きい |
| ResNet | classify | 18, 34, 50, 101 | 1.0以内 |
| MobileNetV4 | classify | s, m, l | sとlはより大きい |
| EfficientNetV2 | classify | b0 | より大きい |
| VGG | classify | 16, 19, 16bn, 19bn | 19bnはより大きい |
| AlexNet | classify | b | 1.0以内 |
| FCN | semantic | r50, r101 | 1.0以内 |
| DeepLabV3 | semantic | mv3 | より大きい |
| MiDaS | depth | s | より大きい |
| QuickSRNet | restore | m2 | より大きい |
| Real-ESRGAN | restore | x4t | より大きい |

計測した差分の一部を示します。COCO val2017の1000枚の画像でのmAP50-95で、デバイス上のINT8を同じ画像でのFP32と比較したものです：YOLO9-sは0.4502から0.4450（-0.52）、YOLO9-tは-0.45、YOLO-NAS-sは-0.30、YOLOX-sは-0.48、YOLOX-nは-1.81（300枚）、PicoDet-mは-2.86、RTMDet-sは-6.75です。ResNet-50はImageNetV2のtop-1で0.7ポイント低下します。すべての数値を載せたサイズ別の表はライブラリの`docs/axelera.md`にあります。

構造上の理由でブロックされているもの：EfficientNetV2のb1〜b3、EfficientDet-d3、RTMDet-x、RTMDet-Ins-x、DeepLabV3のr50とr101、MiDaS-l、RetinaNet-r50v2、FCOS、HRNet、CenterNet、Faster R-CNN、ConvNeXtとConvNeXt V2、YOLOv1、EC、TinyFormer、NAFNet、TEED、ZipDepth。RF-DETRとその他のTransformer系ファミリーは対象外です。YOLO9-E2E、YOLO9-P2、EfficientDet-d4、Real-ESRGANのx4とx2、PIDNet、DEKR、UNetはまだ計測されていません。

## ライセンス

LibreYOLOのコードはMITです。`axelera-devkit`と`axelera-rt`はAxelera社のプロプライエタリなソフトウェアで、自分でインストールします。コンパイル済みモデルは元になった重みのライセンスを引き継ぎます：YOLO-NASの学習済み重みは非商用かつ再配布不可であるため、LibreYOLOはYOLO-NASの`.axm`ファイルをホストしておらず、各ユーザーが自分でコンパイルします。`axelera`のセルがどのように決まるかは[エクスポート対応表](/docs/reference/export-matrix)を参照してください。
