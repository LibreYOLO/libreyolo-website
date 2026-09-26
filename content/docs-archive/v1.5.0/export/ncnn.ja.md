---
title: ncnn
seo_title: LibreYOLOからncnnへエクスポート
description: >-
  LibreYOLOのモデルをPNNX経由でncnnにエクスポートします：paramとbinのペア、固定のエクスポートキャンバス、YOLOXのFocus書き換え、そして変換できるファミリー。
lead: >-
  ncnnはTencentのモバイル向けCPU推論ライブラリです。LibreYOLOはPNNXを経由して変換し、model.ncnn.paramのグラフを、model.ncnn.binの重みファイルおよびファミリー、タスク、クラス名を持つmetadata.yamlと並べて書き出します。
keywords:
  - yolo ncnn エクスポート
  - pnnx 変換
  - model.ncnn.param
  - モバイル cpu 推論
  - ncnn extractor
  - focus pixel_unshuffle
last_verified: 1.5.0
meta:
  - label: フラグ
    value: export(format="ncnn")
    mono: true
  - label: 出力
    value: model.ncnn.param、model.ncnn.bin、metadata.yamlが入ったディレクトリを1つ
  - label: 追加インストール
    value: 'pip install "libreyolo[ncnn]"'
    mono: true
  - label: 再読み込み
    value: LibreYOLO("weights/LibreYOLO9t_ncnn")
    mono: true
  - label: 形状
    value: 固定です。フラグの指定にかかわらず、メタデータにはdynamic=Falseが記録されます。
  - label: 精度
    value: FP32のみ。half=Trueとint8=Trueは拒否されます。
verification: >-
  devブランチのlibreyolo/export/ncnn.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/ncnn.py、pyproject.tomlから読み取りました。
snippets:
  install:
    - label: インストール
      language: bash
      code: |
        # pnnxが変換し、ncnnがその結果を実行
        pip install "libreyolo[ncnn]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t_ncnnディレクトリを出力
        path = model.export(format="ncnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format ncnn --imgsz 640
    - label: 引数
      language: python
      code: |
        model.export(
            format="ncnn",
            imgsz=640,        # int、または(height, width)
            batch=1,
            simplify=True,    # ONNXのフォールバック経路にのみ適用
            opset=None,       # 自動。ONNXのフォールバック経路にのみ適用
            output_path=None, # Noneならweights/<stem>_ncnnに出力
        )

        # half=Trueとint8=Trueは検証時に拒否される
  run:
    - label: LibreYOLOから実行
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_ncnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ncnn単体
      language: python
      code: |
        import ncnn
        import numpy as np
        import yaml

        directory = "weights/LibreYOLO9t_ncnn"
        net = ncnn.Net()
        net.load_param(f"{directory}/model.ncnn.param")
        net.load_model(f"{directory}/model.ncnn.bin")

        # ncnnはバッチではなく単一のCHW画像を受け取る
        mat_in = ncnn.Mat(np.zeros((3, 640, 640), dtype=np.float32))
        extractor = net.create_extractor()
        extractor.input("in0", mat_in)
        ret, mat_out = extractor.extract("out0")
        print(ret, np.array(mat_out).shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # この経路では前処理と後処理は自前で行う
  support:
    - label: エクスポート前に1つのファミリーとタスクを確認
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 9a849a16a3b32334
---

## インストール

<code-tabs name="install" />

このextraはツールチェーンの両方を導入します。`pnnx`が変換を行い、`ncnn`がその結果を
実行します。主経路ではどちらもONNXを経由しません。

## エクスポート

<code-tabs name="export" />

成果物はディレクトリです。`weights/LibreYOLO9t_ncnn`には`model.ncnn.param`、
`model.ncnn.bin`、`metadata.yaml`が入っており、この3つで1つの成果物なので、常に
まとめて移動します。

変換はまずPyTorchから直接PNNXを試します。これが失敗した場合は、静的なONNXグラフを
一時ディレクトリに書き出してそれに対して`pnnx`コマンドラインツールを呼び出し、
エクスポートが例外を送出するのは両方の経路が失敗したときだけで、そのときは両方の
エラーを報告します。したがって`opset`と`simplify`が影響するのはフォールバックだけ
です。

YOLOXは、そもそも変換するために1か所の書き換えが必要です。Focus層はストライド付きの
スライスを使っており、PNNXはこれをローワリングできないため、エクスポートはこれを
`pixel_unshuffle`に置き換え、チャンネル順序の違いを埋め合わせるために後続の畳み込みの
入力チャンネルを並べ替えます。出力は数値的に同一であり、元の重みはエクスポート後に
復元されます。

## 成果物を実行する

<code-tabs name="run" />

`LibreYOLO()`は`model.ncnn.param`と`model.ncnn.bin`を含むディレクトリであれば認識し、
`metadata.yaml`を読み取り、チェックポイントと同じ`Results`オブジェクトを返します。

2つ目のスニペットはランタイムを直接使う経路で、ここでは2つの点がほかのどの形式とも
異なります。ncnnはバッチではなく単一のCHW画像を扱うため、先頭にバッチ軸がありません。
ブロブ名は`.param`ファイルから取得します。PNNXは慣例として`in0`と`out0`を書き込み
ますが、バックエンドはそれを前提にせずファイルを解析します。この経路では前処理、
デコード、NMS、座標のスケール戻しは自前の作業になります。

## 制約

固定キャンバスでFP32です。`half=True`と`int8=True`はどちらも検証時に拒否され、
エクスポートされたメタデータはフラグの指定にかかわらず`dynamic=False`を記録するため、
グラフに存在しない軸をバックエンドが前提にすることはありません。

DETR系のファミリーはすべて事前チェックで拒否されます：`detr`、`deformable_detr`、
`dinodetr`、`dfine`、`lwdetr`、`deim`、`deimv2`、`rtdetr`、`rtdetrv2`、`rtdetrv4`、
`rfdetr`、`ec`。メッセージはどれも同じで、そのモデルにはncnnにないデコーダーまたは
サンプリングの演算が必要だという内容であり、代わりにONNX、OpenVINO、TorchScript、
TensorRTを挙げます。

変換できるものは畳み込み系では広範です。YOLO9とYOLO9-E2E、YOLOX、PicoDet、
YOLO-NASの検出と姿勢推定、より古いYOLO1、YOLO3、YOLO4、YOLO7の検出器、4つのCNN分類
ファミリー、PIDNetのセマンティックセグメンテーション、96×96固定のFOMOの点検出、
ZipDepth、NAFNet、Real-ESRGANです。

ブロックされている項目には具体的な失敗理由が書かれています。トランスフォーマー系の
グラフは一般に未対応の`pnnx.Expression`ノードを残し、実行可能な入力ブロブを持たない
ネットワークになるため、DINOv2、CLIP、SigLIP2、SegFormerはこれで止まります。
BiRefNetはtorchvisionのデフォーマブル畳み込みを必要としますが、PNNXはこれを
ローワリングできません。YOLO2の変換後のグラフは、出力の取り出し中にネイティブの
整数のゼロ除算が起きて、Windowsではncnnランタイムを終了させます。

ファミリーとタスクの全体表は[エクスポート対応表](/docs/reference/export-matrix)を
参照してください。1つの組み合わせを確認するには：

<code-tabs name="support" />
