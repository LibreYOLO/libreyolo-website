---
title: LW-DETR
families:
  - lwdetr
seo_title: LW-DETR：Apache-2.0の下で推論、エクスポート
description: >-
  LibreYOLOのLW-DETRでリアルタイム物体検出を行います。すべてApache-2.0ライセンスのViTベース5サイズをインストールし、推論、検証、エクスポートします。
lead: >-
  BaiduがYOLO検出器に代わるリアルタイムモデルとして位置付けた、通常のViTを使うDetection
  Transformerです。LibreYOLOは検出向けに5つのサイズを推論専用として提供します。
keywords:
  - LW-DETR
  - detection transformer
  - リアルタイム 物体検出
  - plain ViT
  - DETR
  - Baidu
  - Atten4Vis
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLWDETRt.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLWDETRt.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")

        # val()はオブジェクトではなく通常のdictを返す
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLWDETRt.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreLWDETRt.pt format=onnx imgsz=640

        libreyolo export model=LibreLWDETRt.pt format=tensorrt imgsz=640
        half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で振り分けるためエクスポート済み成果物も
        # チェックポイントと同様に読み込まれて同じResultsオブジェクトを返す
        model = LibreYOLO("LibreLWDETRt.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: badd1d8255df5bbd
---

## インストール

LW-DETRにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される `Results` オブジェクトはすべてのファミリーで共通のため、別の検出器への置き換えは1行の変更で済みます。`conf` と `max_det` はクエリ選択をフィルタリングします。APIの一貫性のため `iou` は受け付けますが、デコーダーがNMS処理のない集合予測器なので効果はありません。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

LibreYOLOのLW-DETRは推論専用です。アップストリームは、複数のクエリグループをまたぐGroup-DETRの1対多教師信号と、IoUを考慮する分類損失で学習します。そのレシピはここに接続されていないため、`train()` は `NotImplementedError` を発生させます。

## バリアント

サイズは5つで、すべて通常のViTエンコーダー、マルチスケールプロジェクター、変形可能DETRデコーダーを共有し、同じ入力解像度で実行されます。最小の2つは同じエンコーダー幅を共有し、ブロック深度が異なります。次の2つは、より幅広い同じエンコーダーを共有し、デコーダーへ入力するプロジェクターレベル数が異なります。最大サイズは最も幅広いエンコーダーを使用します。

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

## 引用

<citation-block />
