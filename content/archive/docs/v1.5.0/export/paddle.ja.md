---
title: Paddle
seo_title: LibreYOLOからPaddlePaddleへエクスポート
description: >-
  LibreYOLO検出器をX2Paddle経由でPaddlePaddle推論モデルへ変換します。固定されたツールチェーン、静的なバッチ1のFP32グラフ、CPU推論を説明します。
lead: >-
  PaddlePaddle推論モデルはmodel.pdmodelグラフとmodel.pdiparams重みファイルから構成されます。LibreYOLOは静的なopset
  15のONNXグラフをエクスポートし、X2Paddleで変換します。結果をmetadata.yamlとともにパッケージ化するため、ほかのランタイムと同じファクトリから読み込めます。
keywords:
  - yolo paddle エクスポート
  - paddlepaddle 推論
  - x2paddle 使い方
  - model.pdmodel 変換
  - model.pdiparams 読み込み
  - onnx opset 15 paddle
last_verified: 1.5.0
meta:
  - label: フラグ
    value: export(format="paddle")
    mono: true
  - label: 出力
    value: model.pdmodel、model.pdiparams、metadata.yamlを含むディレクトリ
  - label: 追加パッケージ
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: 再読み込み
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: バックエンド
    value: libreyolo.backends.paddle.PaddleBackend
    mono: true
  - label: 形状
    value: 静的、バッチ1、opset 15。3項目すべてを強制
  - label: 精度
    value: FP32のみ、CPUのみ
  - label: ツールチェーン
    value: PaddlePaddle 2.6.2、X2Paddle 1.6.0、ONNX 1.17以前を厳密に確認
verification: >-
  devブランチのlibreyolo/export/paddle.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/paddle.py、docs/paddle.md、pyproject.tomlを参照。
snippets:
  install:
    - label: インストール
      language: bash
      code: |
        # Python 3.10〜3.12。WindowsではUbuntu 22.04のWSL2経路を検証済み
        pip install "libreyolo[paddle]"
    - label: 固定バージョンを確認
      language: bash
      code: >
        python -c "from importlib.metadata import version;
        print(version('paddlepaddle'), version('x2paddle'), version('onnx'))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # ディレクトリweights/LibreYOLO9t_paddleを書き出し
        path = model.export(format="paddle")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format paddle
    - label: 引数
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # int。このファミリーの正方形キャンバス
            batch=1,          # ほかの値ではValueError
            dynamic=False,    # TrueではValueError
            simplify=True,    # FalseではValueError
            opset=15,         # ほかの値ではValueError
            output_path=None, # Noneはweights/<stem>_paddleに書き出し
        )
  run:
    - label: LibreYOLO経由
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: CLI
      language: bash
      code: |
        libreyolo predict --model weights/LibreYOLO9t_paddle \
          --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --device cpu --save
    - label: バックエンドを直接使用
      language: python
      code: |
        from libreyolo.backends.paddle import PaddleBackend

        # LibreYOLO()がPaddleディレクトリ用に構築するもの。同じResults
        # オブジェクトを返し、ファクトリの振り分けを経由しない
        backend = PaddleBackend("weights/LibreYOLO9t_paddle", device="cpu")
        result = backend.predict("parkour.jpg")
        print(result.boxes.xyxy[:3])
    - label: Paddleを直接使用
      language: python
      code: |
        import numpy as np
        import paddle.inference as paddle_infer
        import yaml

        directory = "weights/LibreYOLO9t_paddle"
        config = paddle_infer.Config(
            f"{directory}/model.pdmodel", f"{directory}/model.pdiparams"
        )
        config.disable_gpu()
        config.disable_mkldnn()
        config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(config)
        handle = predictor.get_input_handle(predictor.get_input_names()[0])
        handle.reshape([1, 3, 640, 640])
        handle.copy_from_cpu(np.zeros((1, 3, 640, 640), dtype=np.float32))
        predictor.run()
        for name in predictor.get_output_names():
            print(name, predictor.get_output_handle(name).copy_to_cpu().shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # この経路では前処理と後処理を自分で実装
  support:
    - label: エクスポート前にファミリーとタスクを確認
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cdd8bf12286e2f53
---

## インストール

<code-tabs name="install" />

追加パッケージは、同等性の検証で測定した正確なスタックに固定されます。PaddlePaddle 2.6.2、
X2Paddle 1.6.0、ONNX 1.17以前です。この固定条件はインストール時だけでなくエクスポート時にも
確認され、異なるバージョンでは期待するバージョンを示す`ImportError`が発生します。新しい
PaddleリリースはX2Paddle 1.6.0が生成する静的コードの一部を拒否します。そのため、未検証の
成果物を生成するより早期に失敗する方が適切です。

## エクスポート

<code-tabs name="export" />

4個の引数はデフォルト値ではなく固定値です。`dynamic`は`False`、`batch`は1、完全に静的な
変換グラフにするため`simplify`は`True`でなければなりません。`opset`はX2Paddle 1.6.0が
受け付ける上限の15でなければなりません。ほかの値を渡すとトレース前に例外が発生します。

中間グラフには1つの正規化処理を行います。ONNXでは省略されたMaxPool dilationを1と定義し、
PyTorchはすべて1の属性を明示的に書き込みますが、X2Paddle 1.6.0はこれを拒否します。そのため、
エクスポーターは冗長なデフォルトを削除し、指定された演算自体は変更しません。

成果物は`model.pdmodel`、`model.pdiparams`、`metadata.yaml`を含むディレクトリです。
X2Paddleが変換中に生成するPythonコードは成果物に含まれません。

## 成果物を実行

<code-tabs name="run" />

`LibreYOLO()`は`model.pdmodel`と`model.pdiparams`の両方を含むディレクトリを認識し、
`metadata.yaml`を読み込んで、チェックポイントと同じ`Results`オブジェクトを返します。
`auto`または`cpu`以外のデバイスでは例外が発生します。このバックエンドはCPU専用です。

ファクトリが構築するのは`PaddleBackend`です。これは`libreyolo`からエクスポートされ、
`libreyolo.backends.paddle.PaddleBackend`としてインポートできます。ファクトリの接尾辞による
振り分けなしでバックエンドを使う場合は直接構築してください。たとえば、自分で
`metadata.yaml`を書いていないディレクトリに`task=`を明示的に渡す場合です。その`predict()`は
同じ入力ソースを受け取り、同じ結果を返します。

ランタイムを直接使うスニペットはバックエンドの設定を再現しており、3個のオプションを無効に
しているのは意図的です。Paddle 2.6のCPU融合パイプラインは、deformable attention用に生成された
大規模なgather・scatterグラフの最適化中に異常終了する可能性があります。そのため、移植可能な
融合なしの静的グラフを同等性の測定対象にしています。この経路では前処理、デコード、NMS、
座標の再スケーリングを自分で実装します。

## 制約

動的形状、FP16、INT8、組み込みNMS、GPUランタイムには対応しません。

検証済みの組み合わせは、YOLO9物体検出、YOLO9-E2EとYOLO9-P2の物体検出、ECの物体検出・
姿勢推定・セグメンテーション、RT-DETRv4、D-FINE、DEIM、DEIMv2の物体検出、YOLO-NASの
物体検出と姿勢推定です。それぞれで変換、CPUランタイムでの再読み込み、生出力の同等性、
公開結果との一致を確認しています。

拒否対象と、組み合わせごとに記録された理由は次のとおりです。

| 組み合わせ | 理由 |
|---|---|
| RF-DETR、全タスク | ONNX opset 17とGridSampleが必要。X2Paddle 1.6.0が受け付けるのはopset 15以下で、GridSampleのマッパーもない |
| RT-DETRとRT-DETRv2の物体検出 | 学習済みグラフにはopset 16以降のGridSampleが必要 |
| D-FINEセグメンテーション | 変換と再読み込みはできるが、マスクlogitの相対RMS誤差が3.52%、対応マスクの最小IoUが0.582 |
| YOLO9セグメンテーション | LibreYOLOのYOLO9は物体検出のみ |
| RTMDet-Insセグメンテーション | 動的カーネルによるマスクのデコードにエクスポート先ランタイムの契約がない |

検証済みまたは拒否対象として記載されていないものは、ONNXからPaddleへの変換経路で未検証で
あることを示して拒否されます。

ファミリーとタスクの完全な対応表は
[エクスポート対応表](/docs/reference/export-matrix)を参照してください。1つの組み合わせを調べるには次を実行します。

<code-tabs name="support" />

