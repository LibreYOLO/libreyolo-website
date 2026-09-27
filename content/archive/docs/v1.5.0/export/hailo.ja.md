---
title: Hailo
seo_title: LibreYOLOモデルをHailoアクセラレーターで実行
description: >-
  LibreYOLOモデルをHailo-8またはHailo-8Lにデプロイする方法：静的ONNXエクスポート、自分で実行するDataflow
  Compilerの工程、コンパイル可能なアーキテクチャを説明します。
lead: >-
  Hailoアクセラレーター向けのコンパイルには、HailoのDeveloper Zoneで配布される独自SDKのHailo Dataflow
  Compilerを使用します。LibreYOLOが担当する処理は通常の静的ONNXエクスポートです。その後、DFCで解析、量子化、HEFへのコンパイルを行います。
keywords:
  - libreyolo hailo
  - hailo-8
  - hailo-8l
  - raspberry pi ai kit
  - ai hat+
  - hailo dataflow compiler
  - hef コンパイル
  - hailortcli
last_verified: 1.5.0
meta:
  - label: LibreYOLOの工程
    value: 'export(format="onnx", imgsz=640, dynamic=False)'
    mono: true
  - label: フォーマットではないもの
    value: format="hef"はありません。DFCをpipの依存関係にはできません。
  - label: 追加パッケージ
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: コンパイルホスト
    value: WSL2 Ubuntu 22.04を含むLinux x86_64。ARMではコンパイルできません。
  - label: コンパイル可能
    value: 純粋なCNNによる固定形状グラフ。アテンション、動的形状、LayerNorm中心の設計は対象外です。
  - label: 状態
    value: DFCから実行可能なHEFまでの全工程を完了したLibreYOLOファミリーはまだありません。
verification: >-
  devブランチのskills/libreyolo-export-hailo/SKILL.md、libreyolo/export/onnx.py、libreyolo/cli/commands/export.pyを参照しました。DFCの制約はそのスキルに記録された内容です。LibreYOLOのHEFはまだコンパイルも測定もされていません。
snippets:
  install:
    - label: LibreYOLO側
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 自分でインストールするHailo側
      language: text
      code: >
        Prerequisites, none of them installable from PyPI:


        - A Linux x86_64 machine. WSL2 Ubuntu 22.04 works. The Raspberry Pi is a
          runtime target, never the compile host.
        - The Dataflow Compiler wheel (hailo_sdk_client) from the Hailo
        Developer
          Zone, which is free to register for.
        - For Hailo-8 and Hailo-8L, the Hailo Model Zoo v2.x line, for its
          recipes and NMS configurations.
        - A GPU on the compile host is strongly recommended: the quantization
          step takes hours without one.
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Hailoにはバッチ1と固定解像度が必要で動的軸は不要
        # Python APIのデフォルトはdynamic=Trueのため明示的に無効化
        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    - label: CLI
      language: bash
      code: |
        # CLIのデフォルトは静的形状
        libreyolo export --model LibreYOLOXs.pt --format onnx --imgsz 640
    - label: コンパイル前にグラフが静的か確認
      language: python
      code: |
        import onnx

        graph = onnx.load("weights/LibreYOLOXs.onnx").graph
        shape = graph.input[0].type.tensor_type.shape
        print([d.dim_value or d.dim_param for d in shape.dim])
  compile:
    - label: 解析、量子化、コンパイル
      language: python
      code: >
        from pathlib import Path


        import numpy as np

        from hailo_sdk_client import ClientRunner

        from PIL import Image


        ONNX = "weights/LibreYOLOXs.onnx"

        HW_ARCH = "hailo8"     # hailo8 | hailo8l | hailo10h

        IMGSZ = 640


        runner = ClientRunner(hw_arch=HW_ARCH)


        # YOLOXではend_node_namesなしで一度変換するとDFCログに

        # 推奨される終端ノードが表示されるためそれらを指定して再実行

        runner.translate_onnx_model(ONNX)


        # 正規化はLibreYOLOの前処理と一致させる必要あり YOLOXとYOLO9は

        # 平均や標準偏差を使わず0-255から0-1へのスケールのみ必要

        script = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0,
        255.0])\n"


        # オプション HailoにNMSを任せる 設定はクラス数と入力サイズの両方に

        # 固有なためCOCO-80設定は3クラスにファインチューニングしたモデルには不適切

        # この行がなければHEFは未処理のヘッドテンソルを出力しアプリがデコード

        # script += 'nms_postprocess("yolox_nms_config.json", meta_arch=yolox,
        engine=cpu)\n'


        runner.load_model_script(script)


        # キャリブレーション画像はデプロイデータを代表する必要あり

        # ランダム画像でもコンパイルできるが精度が暗黙に失われる

        calib_paths = sorted(Path("calib_images").glob("*.jpg"))[:128]

        calib = np.stack([
            np.asarray(
                Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ)),
                dtype=np.float32,
            )
            for p in calib_paths
        ])


        runner.optimize(calib)

        Path("libreyoloxs.hef").write_bytes(runner.compile())
    - label: YOLO9の終端ノード
      language: python
      code: |
        # LibreYOLOグラフは他のエクスポート向け設定で見られるmodel.Nではなく
        # /head/...接頭辞を使用 コピーした設定とは一致しないため解析に失敗したら
        # 自分のグラフ内の名前を確認
        END_NODES = [
            "/head/cv2.0/cv2.0.2/Conv", "/head/cv3.0/cv3.0.2/Conv",
            "/head/cv2.1/cv2.1.2/Conv", "/head/cv3.1/cv3.1.2/Conv",
            "/head/cv2.2/cv2.2.2/Conv", "/head/cv3.2/cv3.2.2/Conv",
        ]
        runner.translate_onnx_model(ONNX, end_node_names=END_NODES)
  device:
    - label: AI KitまたはAI HAT+を搭載したRaspberry Pi 5
      language: bash
      code: |
        sudo apt install dkms hailo-all
        hailortcli fw-control identify       # デバイスを確認してアーキテクチャ名を表示
        hailortcli run libreyoloxs.hef       # スモークテストとスループット
source_hash: 33b077f1c23d5535
---

## インストール

LibreYOLOに `format="hef"` はなく、今後も追加されません。Hailo Dataflow CompilerはDeveloper Zoneへの登録後に非公開の`wheel`として配布される独自SDKであるため、依存関係や追加パッケージにはできません。デプロイは2段階です。LibreYOLOで静的ONNXファイルを書き出し、そのファイルをDFCで処理します。

```text
Libre<Model>.pt  ->  ONNX  ->  HAR (parse)  ->  HAR (quantize INT8)  ->  HEF
                 [libreyolo]           [Hailo DFC, installed by you]
```

<code-tabs name="install" />

## エクスポート

<code-tabs name="export" />

`half=True` を渡さないでください。DFCはFP32 ONNXを取り込み、独自にINT8量子化を行います。`nms=True` も渡さないでください。NMSはHailoが`nms_postprocess`で処理するか、アプリケーションが処理します。NMSサブグラフは終端ノードより後では不要です。デフォルトのopsetを使用できます。DFCパーサーが拒否する場合は、`opset=11` で再エクスポートしてください。

DFCは指定した終端ノードでグラフを切り取ります。このノードは検出ヘッドの畳み込みであり、それより後のすべてを破棄します。そのため、LibreYOLOの通常のデコード済みONNXは入力として使用できます。デコード末尾はパーサーによって単に無視されます。

## コンパイル

<code-tabs name="compile" />

ターゲットに合わせて `hw_arch` を選択します。Hailo-8、26 TOPSのAI HAT+、M.2およびPCIeモジュールには `hailo8` を使用します。Hailo-8L、Raspberry Pi AI Kit、13 TOPSのAI HAT+には `hailo8l` を使用します。Hailo-10Hには `hailo10h` を使用しますが、対応する新しいDFCとModel Zooが必要です。不明な場合は、デバイス上の `hailortcli fw-control identify` で確認できます。

2つのファミリーはHailoRTのNMSメタアーキテクチャに対応するため、Hailoがコンパイル済みパイプライン内で抑制を処理できます。YOLOXでは `meta_arch=yolox` を使用します。YOLO9ではHailoの分離ヘッド型メタアーキテクチャを使用し、ヘッドの配置は同一です。Hailo Model Zooから対応する `nms_postprocess` 設定を取得し、クラス数と入力サイズに合わせて調整してください。その他の畳み込み検出器は、対応するメタアーキテクチャがないグラフとしてコンパイルされます。HEFは未処理のヘッドテンソルを出力し、アプリケーションがCPU上でデコードとNMSを実行します。

失敗した場合はコンパイルログを保管してください。どの修正も、失敗した正確な層名または演算子名に左右されます。

## 成果物の実行

<code-tabs name="device" />

アプリケーションでの推論には `hailo_platform` Python APIを使用します。`nms_postprocess` を組み込んだ場合、出力は `(batch, num_classes, max_dets, 5)` となり、モデル座標の `[y1, x1, y2, x2, score]` を保持します。元画像へのスケーリングはアプリケーション側で行います。実行時にLibreYOLOの `Results` パイプラインは関与しません。HEFは独立した成果物であり、前処理と後処理はアプリケーションが担当します。

## 制約

モデルがHailo-8またはHailo-8Lをターゲットにできるかどうかは、名前ではなくアーキテクチャの特性です。そのため、次の規則はこのページの作成後に追加されたファミリーにも適用されます。

次のいずれかを含むモデルはコンパイルできません。

- 自己、クロス、変形可能、ウィンドウ方式など、あらゆる種類のアテンション。すべてのDETR形式の検出器、すべてのオープンボキャブラリ検出器またはテキスト条件付き検出器、すべてのViTバックボーン、すべての言語または視覚言語タワーが対象外になります。Hailo独自のZooには、手作業で調整されたTransformer HEFがいくつか含まれます。これはベンダー固有の作業によるものであり、任意のアテンショングラフをコンパイルできる証拠にはなりません。
- 動的形状またはデータ依存の制御フロー。DFCは1つの固定入力形状と静的グラフをコンパイルします。そのため、可変のクエリ数、テキストプロンプト、動的top-k、`NonZero`、動的インデックスを持つ `Gather` または `TopK`、`grid_sample` はすべて対象外です。
- LayerNormまたはGELUが中心の設計。BatchNormは畳み込みに明確に統合できますが、LayerNormのサポートは不十分で、GELUはネイティブな活性化関数ではありません。そのため、名目上は畳み込み型であっても、ConvNeXt形式のスタックは適していません。
- ネイティブ解像度の画像間変換処理。復元モデルは入力解像度全体で実行され、実用的なHailo SRAMの容量を超えます。

畳み込みのみを使用し、ReLUまたはSiLUを伴うBatchNormを採用し、入力サイズが固定であれば、そのファミリーは候補になります。このライブラリでは、CNNの1段階検出器が該当し、YOLOXとYOLO9が主なターゲットです。その他には、アプリケーション側でデコードするPicoDet、YOLO-NAS、RTMDetなどの畳み込み検出器、ResNet、MobileNetV4-conv、EfficientNetV2などのCNN分類器が該当します。分類器のうち、HailoのModel ZooにレシピがあるResNetが最もよくサポートされています。また、FOMOによる点検出やResNetバックボーン上のL2CSによる視線推定など、小規模な畳み込みタスクヘッドも原理上はコンパイルできますが、Hailoのレシピはありません。

このページで何もサポート済みとして提示していない理由として、1つの状態に関する注意点があります。DFCから実行可能なHEFまでの全工程を完了したLibreYOLOファミリーはありません。上記の規則はアーキテクチャからコンパイル可能性を予測するものです。HEFをコンパイルして測定するまでは、パーサーの動作、量子化、精度は未検証です。そのため、各候補には固有の記録された証拠が必要です。具体的には、正確なチェックポイントからコンパイルしたHEF、記録されたDFC、Model Zoo、HailoRTのバージョン、文書化されたキャリブレーション、スループット値だけでなくFP32ベースラインと比較したデバイス上の精度が必要です。

モデルが要件を満たさない場合は、同等性が記録されているランタイムを代替として使用できます。[ONNX](/docs/export/onnx)、[TensorRT](/docs/export/tensorrt)、[OpenVINO](/docs/export/openvino)です。
