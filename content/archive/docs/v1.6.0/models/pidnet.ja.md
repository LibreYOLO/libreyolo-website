---
title: PIDNet
families:
  - pidnet
seo_title: PIDNet：MITでリアルタイムセグメンテーションを推論・エクスポート
description: >-
  LibreYOLOのPIDNetでリアルタイムのセマンティックセグメンテーションを行います。MITのs・m・l
  Cityscapesチェックポイントについて、インストール、推論、検証、エクスポートを説明します。
lead: >-
  proportional-integral-derivativeに着想を得た設計へ専用の境界branchを追加した、3
  branch構成のセマンティックセグメンテーションネットワークです。リアルタイム推論を目的としています。LibreYOLOはセマンティックセグメンテーション専用で提供します。
keywords:
  - PIDNet 使い方
  - リアルタイム セマンティックセグメンテーション
  - boundary-aware segmentation
  - Cityscapes
  - dense prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W)クラスID
        print(mask.classes)      # 画像内に存在するクラスIDをソート
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePIDNets-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePIDNets-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePIDNets-sem.pt format=onnx
        libreyolo export model=LibrePIDNets-sem.pt format=tensorrt half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibrePIDNets-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 489db64a39e3a61a
---

## インストール

PIDNetに任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。この
ファミリーではファイル名の`-sem`接尾辞が必要です。

<code-tabs name="predict" />

セマンティックセグメンテーションはボックスではなく、ピクセルごとに1個のクラスIDを返します。
`result.semantic_mask`の`.data`には`(H, W)`配列、`.classes`には画像内に存在するクラスIDの
一覧が入ります。`conf`、`iou`、`max_det`はAPIの一貫性のため受け付けますが、効果はありません。
モデルはargmaxで各ピクセルへクラスを割り当て、信頼度のしきい値処理もNMS処理も行わないためです。
入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

サイズは3種類で、すべて固定1024 px入力です。公開チェックポイントは公式PIDNet Cityscapesの
19クラス用重みを変換したものです。

LibreYOLOはPIDNetを学習しません。このファミリーで`train()`を呼び出すと
`NotImplementedError`が発生し、上記の[サポート層](/docs/models)でも推論専用と示されています。

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

## 引用

<citation-block />

