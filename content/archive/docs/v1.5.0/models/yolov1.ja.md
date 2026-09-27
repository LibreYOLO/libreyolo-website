---
title: YOLOv1
families:
  - yolo1
seo_title: LibreYOLOのYOLOv1：推論、検証、エクスポート
description: >-
  LibreYOLOで初代YOLOv1検出器を実行します。凍結された推論専用の博物館ファミリーです。パブリックドメインライセンスの下で、推論、検証、エクスポートを行います。
lead: >-
  YOLOv1はYOLOファミリーに名前を与えた2016年の初代検出器です。全結合ヘッドを持つ1つの畳み込みネットワークが、アンカーボックスを使わず、1回の処理ですべてのボックスとクラススコアを予測します。LibreYOLOはこれを凍結された推論専用の展示物として提供します。
keywords:
  - YOLOv1
  - YOLO v1
  - Darknet
  - 物体検出
  - Pascal VOC
  - YOLO 歴史モデル
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO1b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO1b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO1b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO1b.pt format=onnx
        libreyolo export model=LibreYOLO1b.pt format=tensorrt half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で振り分けるためエクスポート済み成果物も
        # チェックポイントと同様に読み込まれて同じResultsオブジェクトを返す
        model = LibreYOLO("LibreYOLO1b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: a786372dba86f2f8
---

## インストール

YOLOv1には基本パッケージ以外の追加パッケージは不要です。

```bash
pip install libreyolo
```

## 推論

このファミリーは推論専用です。`train()` は `NotImplementedError` を発生させるため、このページに「学習」セクションはありません。推論、検証、エクスポートにはすべて対応します。重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される `Results` オブジェクトはすべてのファミリーで共通のため、別の検出器への置き換えは1行の変更で済みます。このファミリーには2つの固有事項があります。公開チェックポイントはCOCOではなくPascal VOC（2007+2012）で学習されているため、`box.cls` は80個のCOCOカテゴリではなく、20個のVOCカテゴリ（aeroplane、bicycle、bird、boat、bottle、bus、car、cat、chair、cow、diningtable、dog、horse、motorbike、person、pottedplant、sheep、sofa、train、tvmonitor）のインデックスです。また、全結合検出ヘッドは一度に1枚の画像だけを受け付けるため、ソース一覧は実際のバッチとして実行されず、ループで処理されます。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## 検証

`val()` は `metrics/` キーの辞書を返します。内容は適合率、再現率、mAP 50、mAP 50-95で、チェックポイントの学習に使用したものと同じVOC形式のラベル空間を持つデータセットに対して測定されます。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポート済み成果物はファイル接尾辞によって `LibreYOLO()` から再度読み込まれるため、`.onnx` または `.engine` ファイルはチェックポイントのように動作し、同じ `Results` を返します。LibreYOLOをインストールせず、素のランタイムでグラフを実行することもできますが、その場合は前処理と後処理を自分で実装します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>
