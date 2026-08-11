---
title: DETR
families:
  - detr
seo_title: DETR：Apache-2.0の下で推論、エクスポート
description: >-
  LibreYOLOで初代Detection
  TransformerのDETRを実行します。すべてApache-2.0ライセンスのResNetベース4サイズをインストールし、推論、検証、エクスポートします。
lead: >-
  DETRは初代Detection Transformerであり、アンカーや密なグリッドの代わりに、Hungarian
  matchingを使うTransformerデコーダーで固定された物体集合を予測します。LibreYOLOは検出向けに4つのサイズを推論専用として提供します。
keywords:
  - DETR
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

        # ファクトリーはファイル接尾辞で振り分けるためエクスポート済み成果物も
        # チェックポイントと同様に読み込まれて同じResultsオブジェクトを返す
        model = LibreYOLO("LibreDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: c5549a596742d2a5
---

## インストール

DETRにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される `Results` オブジェクトはすべてのファミリーで共通のため、別の検出器への置き換えは1行の変更で済みます。`conf` と `max_det` はクエリ選択をフィルタリングします。APIの一貫性のため `iou` は受け付けますが、デコーダーがNMS処理のない集合予測器なので効果はありません。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

LibreYOLOのDETRは推論専用です。アップストリームはHungarian matchingを使って500エポック学習しますが、そのレシピはここで実装されていないため、`train()` は `NotImplementedError` を発生させます。

## バリアント

4つのチェックポイントは、ResNet-50またはResNet-101という2つのバックボーン深度と、オプションのdilated C5段階を組み合わせます。DC5バリアントは最後のバックボーン段階でさらにダウンサンプリングせず全解像度を維持するため、デコーダーは同じ入力サイズから、より細かい特徴マップを読み取ります。4つすべてが100個の学習済み物体クエリと6層のTransformerエンコーダー・デコーダーを共有し、同じ入力解像度で実行されます。

## 検証

`val()` は `metrics/` キーの辞書を返します。内容は適合率、再現率、mAP 50、mAP 50-95で、学習に使用した形式の任意のデータセットに対して測定されます。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポート済み成果物はファイル接尾辞によって `LibreYOLO()` から再度読み込まれるため、`.onnx` または `.engine` ファイルはチェックポイントのように動作し、同じ `Results` を返します。[エクスポート](/docs/export)では、すべての形式が受け付ける引数を説明しています。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>
