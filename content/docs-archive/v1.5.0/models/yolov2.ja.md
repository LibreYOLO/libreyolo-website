---
title: YOLOv2
families:
  - yolo2
seo_title: LibreYOLOのYOLOv2：推論、検証、エクスポート
description: >-
  LibreYOLOでYOLOv2（YOLO9000）を実行します。凍結された推論専用の博物館ファミリーです。パブリックドメインライセンスの下で、推論、検証、エクスポートを行います。
lead: >-
  YOLO9000としても公開されたYOLOv2は、YOLO系列にアンカーボックスとパススルー層を導入したDarknet-19検出器です。LibreYOLOはこれを凍結された推論専用の展示物として提供します。
keywords:
  - YOLOv2
  - YOLO9000
  - Darknet
  - Darknet-19
  - 物体検出
  - アンカーボックス
  - YOLO 歴史モデル
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO2b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO2b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO2b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO2b.pt format=onnx
        libreyolo export model=LibreYOLO2b.pt format=tensorrt half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で振り分けるためエクスポート済み成果物も
        # チェックポイントと同様に読み込まれて同じResultsオブジェクトを返す
        model = LibreYOLO("LibreYOLO2b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: ba2884a2f6e1b0da
---

## インストール

YOLOv2には基本パッケージ以外の追加パッケージは不要です。

```bash
pip install libreyolo
```

## 推論

このファミリーは推論専用です。`train()` は `NotImplementedError` を発生させるため、このページに「学習」セクションはありません。推論、検証、エクスポートにはすべて対応します。重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される `Results` オブジェクトはすべてのファミリーで共通のため、別の検出器への置き換えは1行の変更で済みます。`conf` は信頼度のしきい値、`iou` はNMSのしきい値として、`region` ヘッドのアンカーベース予測に適用されます。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## 検証

`val()` は `metrics/` キーの辞書を返します。内容は適合率、再現率、mAP 50、mAP 50-95で、検証に使用する形式の任意のデータセットに対して測定されます。

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
