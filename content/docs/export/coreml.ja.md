---
title: Core ML
seo_title: LibreYOLOからCore MLへエクスポート
description: >-
  LibreYOLOの検出モデルをCore
  MLの.mlpackageにエクスポートします：ImageType入力の契約、FP16、コンピュートユニット、NMSの埋め込み、対応する4つのファミリー。
lead: >-
  Core
  MLはAppleのオンデバイス向けモデル形式です。LibreYOLOはファミリーごとの前処理ラッパーの背後で検出モデルをトレースし、変換後のグラフが常に統一されたRGB画像入力を受け取るようにしたうえで、モデルのメタデータを添えたML
  Program形式の.mlpackageを書き出します。
keywords:
  - yolo coreml エクスポート
  - mlpackage 変換
  - coremltools
  - ct.ImageType
  - apple neural engine
  - compute_units
  - coreml nms パイプライン
last_verified: 1.5.0
meta:
  - label: フラグ
    value: export(format="coreml")
    mono: true
  - label: 出力
    value: ML Program形式の.mlpackageバンドル（ディレクトリ）を1つ
  - label: 追加インストール
    value: 'pip install "libreyolo[coreml]"'
    mono: true
  - label: 再読み込み
    value: LibreYOLO("weights/LibreYOLO9t.mlpackage") on macOS
    mono: true
  - label: 形状
    value: 固定です。入力は形状が固定されたct.ImageTypeです。
  - label: 数値精度
    value: FP32、FP16（half=True）。INT8はありません。
  - label: ファミリー
    value: 検出タスクのみ。yolox、yolo9、rtdetr、rfdetrが対象
verification: >-
  devブランチのlibreyolo/export/coreml.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/coreml.py、pyproject.tomlを確認しました。
snippets:
  install:
    - label: インストール
      language: bash
      code: |
        pip install "libreyolo[coreml]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t.mlpackageバンドルを出力
        path = model.export(format="coreml")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml
    - label: 引数
      language: python
      code: |
        model.export(
            format="coreml",
            imgsz=640,
            batch=1,
            half=False,           # TrueにするとFLOAT16の演算精度で変換
            compute_units="all",  # all | cpu_and_gpu | cpu_and_ne | cpu_only
            output_path=None,     # Noneならweights/<stem>.mlpackageに出力
        )

        # dynamicは受け付けられるが入力は固定形状のct.ImageType
        # 埋め込みメタデータはどちらの場合もdynamic=Falseを記録
  nms:
    - label: AppleのNMSレイヤーを埋め込む
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLOXとYOLO9の検出のみ batch 1
        LibreYOLO("LibreYOLO9t.pt").export(
            format="coreml",
            nms=True,
            conf=0.25,
            iou=0.45,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml --nms \
          --conf 0.25 --iou 0.45
  run:
    - label: macOSでLibreYOLOから実行
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO(
            "weights/LibreYOLO9t.mlpackage",
            compute_units="all",   # Neural Engineに固定するならcpu_and_ne
        )
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: coremltools単体
      language: python
      code: |
        import coremltools as ct
        from PIL import Image

        mlmodel = ct.models.MLModel("weights/LibreYOLO9t.mlpackage")
        print(mlmodel.user_defined_metadata["model_family"])
        print(mlmodel.user_defined_metadata["names"])

        # 入力はエクスポート時の固定サイズで "image" という名前の画像
        image = Image.open(SAMPLE_IMAGE).convert("RGB").resize((640, 640))
        out = mlmodel.predict({"image": image})
        print({name: value.shape for name, value in out.items()})

        # この経路ではレターボックス処理と後処理は自前で行う
  support:
    - label: エクスポート前にファミリーとタスクを確認
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 09c5394e3837eca2
---

## インストール

<code-tabs name="install" />

推論にはmacOSが必要です。`LibreYOLO()`はそれ以外のプラットフォームでは`.mlpackage`を
拒否して現在のプラットフォーム名を含むメッセージを返し、対応表は、ランタイムの同等性を
確認するにはmacOSのランナーが必要だという理由から、これらの組み合わせを利用可能として
記録しています。

## エクスポート

<code-tabs name="export" />

バンドルはチェックポイントのステム名で`weights/`に出力され、`half=True`のときは`_fp16`が
付きます。`.mlpackage`はディレクトリなので、ツリー全体をコピーしてください。

どのファミリーも前処理ラッパーの背後でトレースされるため、変換後のグラフが受け取る入力は
1つに統一されます：RGB、`scale=1/255`、バイアスなし、`ct.ImageType`としての宣言。
ラッパーはファミリーごとの流儀、すなわちYOLOXでは0〜255の範囲のBGR、RF-DETRではImageNetの
平均と標準偏差、YOLO9とRT-DETRでは恒等変換を吸収します。Core MLの利用側がファミリー固有の
テンソルではなく普通の画像を渡せるのは、このためです。

変換先はML Programで、最小デプロイターゲットはiOS 15です。`compute_units`は変換後のモデルに
保存され、成果物を読み込むときに再度上書きできます。

モデルのメタデータは文字列として`user_defined_metadata`に入り、バックエンドはそこから
ファミリー、タスク、クラス名、入力サイズ、姿勢スキーマを読み取ります。

### NMSの埋め込み

<code-tabs name="nms" />

`nms=True`にすると、Appleの`NonMaximumSuppression`レイヤーで終わるCore MLパイプラインで
モデルを包みます。出力は2つあります：形状が`N`×クラス数の`confidence`と、正規化された
`xywh`で形状が`N`×4の`coordinates`です。

対象はYOLOXとYOLO9の検出のみで、バッチサイズは1である必要があります。集合予測はクエリと
クラスにわたるtop-kを取るだけでIoUの段階がなく、そのレイヤーを使えないため、DETR系の
ファミリーは名前で拒否されます。`max_det`もここでは公開されていません。検出数の上限が
問題になる場合は、代わりに[ONNXのNMS埋め込み](/docs/export/onnx)を使ってください。

## 成果物を実行する

<code-tabs name="run" />

`LibreYOLO()`は`.mlpackage`という拡張子のディレクトリを認識し、チェックポイントと同じ
`Results`オブジェクトを返します。この形式でファクトリが受け渡す引数は`compute_units`だけで、
`all`、`cpu_and_gpu`、`cpu_and_ne`、`cpu_only`を指定できます。Core MLは代わりに
コンピュートユニットを通して処理を振り分けるため、`device`引数は無視されます。

2つ目のスニペットはランタイムを直接使う経路です。この経路ではレターボックス処理、デコード、
NMS、座標のスケール戻しは自前の作業になり、クラス名は`user_defined_metadata`にあります。

## 制約

対応は4つのファミリーで、検出のみです：`yolox`、`yolo9`、`rtdetr`、`rfdetr`。ファミリーを
把握した前処理ラッパーがあってはじめて固定の画像入力という契約が正しくなり、対象外の
ファミリーでは誤った正規化で変換されてしまうため、それ以外は事前チェックで拒否されます。
エラーメッセージは代替としてONNXとTorchScriptを挙げます。

入力形状は`ct.ImageType`によって固定されるため、`dynamic=True`を指定しても何も変わらず、
メタデータには`dynamic=False`が記録されます。別の解像度が必要なら、もう1つバンドルを
エクスポートしてください。

`half=True`はFP16の演算精度で変換します。このエクスポーターにINT8の経路はありません。

ファミリーとタスクの全体表は[エクスポート対応表](/docs/reference/export-matrix)を
参照してください。Appleのより新しいオンデバイス形式については
[Core AI](/docs/export/coreai)を参照してください。1つの組み合わせを確認するには：

<code-tabs name="support" />
