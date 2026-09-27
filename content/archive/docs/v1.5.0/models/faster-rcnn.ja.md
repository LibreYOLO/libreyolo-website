---
title: Faster R-CNN
families:
  - faster_rcnn
seo_title: LibreYOLOのFaster R-CNN：推論、検証、エクスポート
description: >-
  LibreYOLOのFaster
  R-CNNで4種類のバックボーンによる物体検出を実行します。BSD-3-Clauseのtorchvision移植版について、インストール、推論、検証、エクスポートを説明します。
lead: >-
  Faster R-CNNはregion proposal networkから2段階分類器へ入力して物体を検出します。region
  proposalを独立した処理ではなく、同じ学習済みネットワークの一部にしたアーキテクチャです。LibreYOLOはtorchvisionの実装を物体検出向けに移植しています。
keywords:
  - Faster R-CNN 使い方
  - 物体検出
  - region proposal network
  - 2段階 検出器
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFasterRCNNl.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFasterRCNNl.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFasterRCNNl.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFasterRCNNl.pt format=onnx imgsz=800
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreFasterRCNNl.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 3fd82eb835399560
---

## インストール

Faster R-CNNに任意の追加パッケージは必要ありません。インポートするものはすべて基本
インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーで共通のため、別の検出器への切り替えは
1行の変更だけで済みます。`conf`と`iou`は信頼度とNMSのしきい値を設定します。Faster R-CNNは
queryベースの検出器と異なり、アップストリームのNMS処理を維持します。入力ソース、
ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

サイズは4種類で、同じ構成を拡大縮小したものではなく、それぞれ異なるtorchvision構成です。
`n`は320 px入力のMobileNetV3-Large、`s`は800 px入力の同じバックボーン、`m`はfeature pyramidを
持つResNet-50です。`l`はv2改訂版で、`m`のヘッドをより深いregion proposalヘッドと4畳み込みの
ボックスヘッドに置き換えます。`n`と`s`は精度と軽量なバックボーンのトレードオフです。

## 検証

`val()`は、学習に使った形式の任意のデータセットに対して測定した適合率、再現率、mAP 50、
mAP 50-95を含む`metrics/`キーの辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

Faster R-CNNはバッチサイズ1でONNXだけにエクスポートできます。エクスポートしたグラフは
アップストリームのリサイズ処理を内部に維持するため、LibreYOLOは渡された値にかかわらず
`dynamic=True`を強制します。これにより、正方形ではない入力ソースでもグラフが有効に保たれます。
エクスポートした`.onnx`ファイルは、ファイル接尾辞に基づいて`LibreYOLO()`から再読み込みされ、
同じ`Results`を返します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

