---
title: FCN
families:
  - fcn
seo_title: FCN：BSD-3-ClauseのResNet FCNを推論・エクスポート
description: >-
  LibreYOLOのFCNでセマンティックセグメンテーションを行います。torchvisionのdilated-ResNet
  FCNチェックポイントについて、インストール、推論、検証、エクスポートを説明します。
lead: >-
  検出器の全結合層を畳み込みに置き換えた密なピクセル単位の分類器で、ボックスの代わりにフル解像度のクラスマップを出力します。LibreYOLOはセマンティックセグメンテーション専用で提供します。
keywords:
  - FCN 使い方
  - fully convolutional network
  - セマンティックセグメンテーション
  - dense prediction
  - ResNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W)クラスID
        print(mask.classes)      # 画像内に存在するクラスIDをソート
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCNr50.pt format=onnx
        libreyolo export model=LibreFCNr50.pt format=tensorrt half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreFCNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 7776b0fc85a208fb
---

## インストール

FCNに任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

セマンティックセグメンテーションはボックスではなく、ピクセルごとに1個のクラスIDを返します。
`result.semantic_mask`の`.data`には`(H, W)`配列、`.classes`には画像内に存在するクラスIDの
一覧が入ります。`conf`、`iou`、`max_det`はAPIの一貫性のため受け付けますが、効果はありません。
モデルはargmaxで各ピクセルへクラスを割り当て、信頼度のしきい値処理もNMS処理も行わないためです。
入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

ResNetの深さは2種類で、どちらも固定520 px入力です。ライブラリの推論グラフはtorchvisionの
dilated-ResNet FCNで、元の論文にあるskip connection付きのVGGベースFCN-8sネットワークでは
ありません。

LibreYOLOはFCNを学習しません。このファミリーで`train()`を呼び出すと`NotImplementedError`が
発生し、上記の[サポート層](/docs/models)でも推論専用と示されています。公開されている2個の
チェックポイントは、torchvision独自のCOCO学習済み重みをLibreYOLOのローダー向けに変換したものです。

## 検証

`val()`は、学習に使った形式の任意のデータセットに対して測定した`metrics/mIoU`と
`metrics/pixel_accuracy`を返します。

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

