---
title: PIDNet
families:
  - pidnet
seo_title: PIDNet：MITの下でリアルタイムセグメンテーションを推論、エクスポート
description: >-
  LibreYOLOのPIDNetでリアルタイムのセマンティックセグメンテーションを行います。MITの下でs/m/lのCityscapesチェックポイントをインストールし、推論、検証、エクスポートします。
lead: >-
  比例・積分・微分に着想を得た設計に専用の境界分岐を追加し、リアルタイム推論を目指した3分岐のセマンティックセグメンテーションネットワークです。LibreYOLOはセマンティックセグメンテーション専用として提供します。
keywords:
  - PIDNet
  - リアルタイム セマンティックセグメンテーション
  - 境界認識 セグメンテーション
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
        print(mask.data.shape)   # (H, W)のクラスID
        print(mask.classes)      # 画像内に存在するクラスIDのソート済み一覧
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

        # ファクトリーはファイル接尾辞で振り分けるためエクスポート済み成果物も
        # チェックポイントと同様に読み込まれて同じResultsオブジェクトを返す
        model = LibreYOLO("LibrePIDNets-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 489db64a39e3a61a
---

## インストール

PIDNetにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。このファミリーでは、ファイル名の `-sem` 接尾辞が必要です。

<code-tabs name="predict" />

セマンティックセグメンテーションはボックスではなくピクセルごとに1つのクラスIDを返します。そのため、`result.semantic_mask` の `.data` は `(H, W)` 配列を保持し、`.classes` は画像内に存在するクラスIDの一覧を保持します。APIの一貫性のため `conf`、`iou`、`max_det` は受け付けますが、効果はありません。モデルはargmaxによって各ピクセルへクラスを割り当て、信頼度のしきい値やNMS処理を使用しないためです。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

サイズは3つで、すべて固定の1024 px入力です。公開チェックポイントは、19クラスの公式PIDNet Cityscapes重みを変換したものです。

LibreYOLOはPIDNetを学習しません。このファミリーで `train()` は `NotImplementedError` を発生させ、上の[サポート階層](/docs/models)でも推論専用と記載されています。

## 検証

`val()` は、学習に使用した形式の任意のデータセットに対して測定した `metrics/mIoU` と `metrics/pixel_accuracy` を返します。

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
