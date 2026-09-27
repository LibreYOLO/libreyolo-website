---
title: DETR
families:
  - detr
seo_title: DETR：Apache-2.0で推論とエクスポート
description: >-
  元祖detection
  transformerのDETRをLibreYOLOで実行します。すべてApache-2.0のResNetベース4サイズについて、インストール、推論、検証、エクスポートを説明します。
lead: >-
  DETRは元祖detection transformerで、アンカーや密なグリッドの代わりにHungarian
  matchingを行うtransformerデコーダーで固定数の物体を予測します。LibreYOLOは物体検出向けに4サイズを推論専用で提供します。
keywords:
  - DETR 使い方
  - detection transformer
  - 物体検出
  - Hungarian matching
  - transformer decoder
  - Meta AI
  - Facebook AI Research
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")

        # val()はオブジェクトではなく通常のdictを返す
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: c5549a596742d2a5
---

## インストール

DETRに任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーで共通のため、別の検出器への切り替えは
1行の変更だけで済みます。`conf`と`max_det`はqueryの選択を絞り込みます。`iou`はAPIの
一貫性のため受け付けますが、効果はありません。デコーダーがNMS処理のないset predictorである
ためです。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

LibreYOLOのDETRは推論専用です。アップストリームはHungarian matchingを使って500エポック
学習しますが、そのレシピはここでは未実装です。そのため`train()`は`NotImplementedError`を
発生させます。

## バリアント

4個のチェックポイントは、ResNet-50またはResNet-101という2種類のバックボーン深度と、任意の
dilated C5段階を組み合わせます。DC5バリアントはバックボーンの最終段階でさらにダウンサンプリング
せずフル解像度を維持するため、デコーダーは同じ入力サイズからより細かい特徴マップを読み取ります。
4種類すべてが100個の学習済みobject queryと6層のtransformer encoder-decoderを共有し、同じ
入力解像度で動作します。

## 検証

`val()`は、学習に使った形式の任意のデータセットに対して測定した適合率、再現率、mAP 50、
mAP 50-95を含む`metrics/`キーの辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポートした成果物はファイル接尾辞に基づいて`LibreYOLO()`から再読み込みされます。そのため、
`.onnx`または`.engine`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。
[エクスポート](/docs/export)には各形式が受け付ける引数の一覧があります。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

