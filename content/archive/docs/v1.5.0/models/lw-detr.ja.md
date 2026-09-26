---
title: LW-DETR
families:
  - lwdetr
seo_title: LW-DETR：Apache-2.0で推論とエクスポート
description: >-
  LibreYOLOのLW-DETRでリアルタイム物体検出を実行します。すべてApache-2.0のViTベース5サイズについて、インストール、推論、検証、エクスポートを説明します。
lead: >-
  BaiduがYOLO検出器のリアルタイム代替として位置付けたplain-ViT detection
  transformerです。LibreYOLOは物体検出向けに5サイズを推論専用で提供します。
keywords:
  - LW-DETR 使い方
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

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreLWDETRt.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: badd1d8255df5bbd
---

## インストール

LW-DETRに任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーで共通のため、別の検出器への切り替えは1行の変更だけで済みます。`conf`と`max_det`はqueryの選択を絞り込みます。`iou`はAPIの一貫性のため受け付けますが、デコーダーがNMS処理のないset predictorであるため効果はありません。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

LibreYOLOのLW-DETRは推論専用です。アップストリームは複数のquery groupにわたるGroup-DETRのone-to-many supervisionとIoU-aware分類損失で学習しますが、そのレシピはここに接続されていません。そのため`train()`は`NotImplementedError`を発生させます。

## バリアント

サイズは5種類で、すべてplain-ViTエンコーダー、multi-scale projector、deformable DETRデコーダーを共有し、同じ入力解像度で動作します。最小の2種類はエンコーダーの幅を共有してブロック深度が異なり、次の2種類はより幅広いエンコーダーを共有してデコーダーへ入力するprojector level数が異なります。最大サイズでは最も幅広いエンコーダーを使います。

## 検証

`val()`は、学習に使った形式の任意のデータセットに対して測定した適合率、再現率、mAP 50、mAP 50-95を含む`metrics/`キーの辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポートした成果物はファイル接尾辞に基づいて`LibreYOLO()`から再読み込みされます。そのため、`.onnx`または`.engine`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。[エクスポート](/docs/export)には各形式が受け付ける引数の一覧があります。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

