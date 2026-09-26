---
title: HRNet
families:
  - hrnet
seo_title: HRNet：LibreYOLOのトップダウン姿勢推定
description: >-
  LibreYOLOでHRNetを使い、トップダウン方式のCOCO-17姿勢推定を行います。MITライセンスのW32およびW48チェックポイントについて、インストール、推論、検証、エクスポートの方法を説明します。
lead: >-
  HRNetは、downsampling後に解像度を復元する代わりに、マルチスケール融合を繰り返して高解像度の特徴量streamを維持する畳み込みネットワークです。LibreYOLOは公式のトップダウン姿勢バリアントを推論と検証向けにラップします。
keywords:
  - HRNet
  - 人体 姿勢推定
  - トップダウン 姿勢推定
  - COCO-17 キーポイント
  - high-resolution network
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 人物ソースが未指定の場合、HRNetは軽量なLibreYOLO9t検出器と
        # 自動的に組み合わせ、その選択を一度だけログへ記録
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreHRNetw32-pose.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 人物ソース
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        # 検出を完全に省略し、画像全体を1人として扱う
        result = model(SAMPLE_IMAGE, cropped=True)

        # すでに実行済みの検出器によるボックスをHRNetへ渡す方法
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        # デフォルトのLibreYOLO9tではなく、特定のLibreYOLO検出器と
        # 組み合わせる方法
        result = model(SAMPLE_IMAGE, person_detector="rfdetr")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreHRNetw32-pose.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreHRNetw32-pose.pt format=onnx
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # エクスポート済みグラフは固定キャンバスのheatmap headだけであり
        # 切り取り・正規化済みの人物cropのバッチを受け取って
        # 未加工のheatmapを返す。人物検出、crop geometry、heatmapの
        # decode、OKS suppressionはこのグラフに含まれないため
        # LibreYOLO外で実行する場合はdecode stepを自身で再実装
        session = ort.InferenceSession("LibreHRNetw32-pose.onnx")
        name = session.get_inputs()[0].name
        heatmaps = session.run(
            None, {name: np.zeros((1, 3, 256, 192), dtype=np.float32)}
        )[0]
source_hash: 5a5540fd54ee6f23
---

## インストール

HRNetに基本パッケージ以外の追加パッケージは必要ありません。

```bash
pip install libreyolo
```

デフォルトの人物検出器である軽量なLibreYOLO9tチェックポイントは、HRNetが初めて組み合わせる際に自動的にダウンロードされます。

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

HRNetはトップダウン姿勢推定器です。姿勢ヘッドを実行する前に人物ボックスが必要なので、呼び出しのたびに1つの取得元を解決します。設定を変更しなければ、初回にLibreYOLO9t検出器と組み合わせ、その選択をログへ記録します。`cropped=True` は検出を省略して画像全体を1人として扱います。`person_boxes` はすでに実行済みの検出器によるボックスを受け付けます。`person_detector` は `"auto"`、`"rfdetr"`、任意のLibreYOLO検出モデル、通常のcallableを受け付けます。`flip_test=True` は水平方向に反転したcropでもモデルを実行して2つのヒートマップを平均します。これはHRNet独自のテスト時データ拡張であり、genericな `augment=True` はここでは定義されません。複数画像のソースは順番に実行されます。HRNetの検出器と画像ごとに異なる人物数は、stackした推論に対応しません。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

`w32` と `w48` の2サイズがあり、どちらも固定解像度の人物cropから標準COCO-17キーポイントセットを予測します。`w48` は2つのうち幅の広いバックボーンです。

アップストリームのmodel zooでは、それぞれのサイズについて独自の人物検出器、flip testing設定、公式COCO評価手順を使った姿勢精度が報告されています。LibreYOLOのデフォルトの組み合わせは異なる検出器を使うため、ここでの検証実行が測定するのはその組み合わせであり、アップストリームの組み合わせではありません。アップストリームの値を再現するには、元の評価と同じ人物ボックス、検出器スコア、flip設定が必要です。

## 検証

`val()` はCOCO形式のキーポイントOKS-APを実行し、YOLO-poseの `data.yaml` またはCOCO keypoints JSONと画像ディレクトリを受け付けます。指標のbackendはデフォルトでfaster-coco-evalであり、faster-coco-evalがインストールされていない場合は `pycocotools` が自動的に使われます。`faster_coco_eval=False` を指定すると、`pycocotools` 経路を強制します。

<code-tabs name="val" />

検証は内部でHRNet独自の `predict()` を駆動するため、モデルの構築時または呼び出し時に指定された人物検出器を使います。呼び出すたびにデフォルトを再解決させず、複数回の実行でソースを固定するには、明示的な `person_detector=` を指定してモデルを構築してください。

## エクスポート

<export-matrix />

HRNetのエクスポート仕様が対象とするのはONNX、TorchScript、OpenVINO、TensorRTだけであり、ほかの形式ではトレース開始前に例外が送出されます。すべてのエクスポートは、batch-one FP32の固定キャンバスheatmap headだけです。人物cropを受け取り、未加工のヒートマップを返します。その前段のaffine crop geometryと、後段のheatmap decode、flip restoration、OKS suppressionはPython側に残るため、画像入力からキーポイント出力までの完全なパイプラインには、出力側にもLibreYOLOが必要です。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

