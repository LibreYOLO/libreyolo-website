---
title: TFLite
seo_title: LibreYOLOからTFLite（LiteRT）へエクスポート
description: >-
  LibreYOLOモデルをonnx2tf経由で.tflite
  FlatBufferへエクスポートします。静的形状、FP32のみ、NHWC入力、正常に変換できるファミリーについて説明します。
lead: >-
  TFLiteは、LiteRTがモバイルおよび組み込みターゲットで実行するFlatBuffer形式です。LibreYOLOは静的ONNXグラフをエクスポートし、flatbuffer-directモードのonnx2tfで変換して、モデルメタデータをJSONサイドカーとして成果物の隣に書き出します。
keywords:
  - yolo tflite エクスポート
  - litert
  - onnx2tf
  - ai-edge-litert
  - tflite flatbuffer
  - tflite nhwc 入力
  - エッジ 推論
last_verified: 1.5.0
meta:
  - label: フラグ
    value: export(format="tflite")
    mono: true
  - label: 出力
    value: 1つの.tfliteファイルと.tflite.jsonメタデータサイドカー
  - label: 追加パッケージ
    value: 'pip install "libreyolo[tflite]"'
    mono: true
  - label: 再読み込み
    value: LibreYOLO("weights/LibreYOLO9t.tflite")
    mono: true
  - label: 形状
    value: 静的のみ。dynamic=Trueは拒否されます。
  - label: 精度
    value: FP32のみ。half=Trueとint8=Trueは拒否されます。
  - label: 要件
    value: Python 3.12以降。onnx2tf 2.4.xは古いバージョン向けのwheelを公開していないためです。
verification: >-
  devブランチのlibreyolo/export/tflite.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/tflite.py、pyproject.tomlを参照しました。
snippets:
  install:
    - label: インストール
      language: bash
      code: |
        # LiteRTはGoogleによるTensorFlow Liteの現在の名称 どちらの追加パッケージも
        # 同じツールチェーンをインストールして同じ.tflite出力を生成
        pip install "libreyolo[tflite]"
    - label: 最初にPythonバージョンを確認
      language: bash
      code: |
        python -c "import sys; print(sys.version_info >= (3, 12))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t.tfliteとweights/LibreYOLO9t.tflite.jsonに書き出し
        path = model.export(format="tflite", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tflite --imgsz 640

        # litertは別名として受け付けられ同じエクスポーターとして解決
        libreyolo export --model LibreYOLO9t.pt --format litert --imgsz 640
    - label: 引数
      language: python
      code: |
        model.export(
            format="tflite",
            imgsz=640,        # 整数または(height, width)
            batch=1,
            simplify=True,    # ONNX中間表現にonnxsimを適用
            output_path=None, # Noneではweights/<stem>.tfliteに書き出し
            verbose=False,    # Trueではonnx2tfログを逐次表示
        )

        # dynamic=TrueではValueErrorが発生 コンバーターには静的形状が必要
        # half=Trueとint8=Trueはトレース前に拒否
  run:
    - label: LibreYOLO経由
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.tflite")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 素のLiteRT
      language: python
      code: >
        import json


        import numpy as np

        from ai_edge_litert.interpreter import Interpreter


        interpreter = Interpreter(model_path="weights/LibreYOLO9t.tflite")

        interpreter.allocate_tensors()

        detail = interpreter.get_input_details()[0]

        print(detail["shape"], detail["dtype"])   # NCHWではなくNHWC


        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"],
        np.float32))

        interpreter.invoke()

        for output in interpreter.get_output_details():
            print(output["name"], interpreter.get_tensor(output["index"]).shape)

        # クラス名とタスクと入力サイズはサイドカーに格納

        meta = json.load(open("weights/LibreYOLO9t.tflite.json"))

        print(meta["model_family"], meta["task"], meta["names"])


        # 前処理とNCHWからNHWCへの転置と後処理を自分で実装
  support:
    - label: エクスポート前に1つのファミリーとタスクを確認
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: fa2deaa0ef6d9978
---

## インストール

<code-tabs name="install" />

追加パッケージは、変換用の `onnx2tf` と結果の実行用の `ai-edge-litert` を導入します。どちらにもPython 3.12のマーカーがあります。古いインタープリターでは、コンバーター内で失敗する代わりに、必要なバージョンを示す `ImportError` がエクスポート時に発生します。

`libreyolo[litert]` もまったく同じものをインストールします。フォーマット文字列 `litert` は `tflite` の別名であり、どちらでも出力ファイルは `.tflite` です。

## エクスポート

<code-tabs name="export" />

何よりも先にファミリーとタスクを確認します。そのため、未対応の組み合わせでは、一般的なメッセージではなく、その組み合わせを除外した特定のコンバーターまたはランタイムエラーによって直ちに失敗します。変換自体は、静的ONNX中間表現に対して `flatbuffer_direct` モードで `onnx2tf` を呼び出すサブプロセスです。

メタデータはサイドカーです。`weights/LibreYOLO9t.tflite.json` にはファミリー、タスク、クラス名、入力サイズ、姿勢スキーマが含まれます。FlatBuffer自体にLibreYOLOのメタデータフィールドはないため、2つのファイルを一緒に扱います。

## 成果物の実行

<code-tabs name="run" />

`LibreYOLO()` は `.tflite` 接尾辞に基づいて処理を振り分け、チェックポイントと同じ `Results` オブジェクトを返します。バックエンドはサイドカーを読み取り、インタープリターがチャンネル末尾の入力を要求する場合はNCHWのblobをNHWCへ転置し、存在する場合はインタープリターの量子化スケールとゼロ点を適用し、LibreYOLOの後処理が想定する配置へ出力を転置して戻します。

2番目のスニペットは、ランタイムだけを直接使う方法です。この場合、前処理、配置の転置、デコード、NMS、座標の再スケーリングをすべて自分で実装します。特に見落としやすいのは配置の詳細です。onnx2tfはチャンネル末尾の入力を生成するため、形状 `(1, 3, 640, 640)` のblobはバインドできません。

## 制約

静的形状のみです。`dynamic=True` はトレース前に `ValueError` を発生させ、エクスポートのキャンバスは `imgsz` の解決結果に固定されます。

FP32のみです。`half=True` と `int8=True` はどちらも検証時に拒否されるため、現在はこのエクスポーターから量子化デプロイへ進めません。

ここでの対応範囲はグラフ形式より狭く、ファミリーではなく測定結果によって決まります。検証済みの組み合わせには、YOLO9、YOLOX、YOLO-NASによる検出、PIDNetによるセマンティックセグメンテーション、4つのCNN分類ファミリー、DINOv2とSigLIP2による埋め込みベクトル、SigLIP2による分類、TEEDとDexiNedによるエッジ検出、Real-ESRGANとSwinIRによる復元が含まれます。SwinIRには追加の注意点があります。元画像の寸法がエクスポート時のキャンバスに正確に一致すれば同等性が保たれますが、小さい元画像はTransformerで処理する前にキャンバスまでパディングされるため、ネイティブな可変サイズ推論とは異なる場合があります。

除外された項目には正確な失敗理由が記載されているため、回避策を試す前に読む価値があります。例をいくつか挙げます。RF-DETRの検出はネイティブの384キャンバスで変換できますが、`STRIDED_SLICE` が対応上限の5次元を超える入力を受け取るため、LiteRTは割り当てできません。PicoDetは、`RESHAPE` が19,200個の入力要素を9,600個の出力要素に対応付けるため拒否されます。D-FINEは `GatherElements` の形状処理中にコンバーターをクラッシュさせます。RTMDetは未処理の同等性を保ってエクスポートと再読み込みができますが、公開ボックスのIoUは0.911まで低下し、座標が29.9 pxずれます。

ファミリーとタスクの完全な一覧は、[エクスポート対応表](/docs/reference/export-matrix)を参照してください。除外理由の文字列を含め、1つの組み合わせを確認するには次を実行します。

<code-tabs name="support" />
