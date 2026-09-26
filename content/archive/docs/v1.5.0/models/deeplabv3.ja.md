---
title: DeepLabv3
families:
  - deeplabv3
seo_title: DeepLabv3：ASPPセマンティックセグメンテーションの推論とエクスポート
description: >-
  LibreYOLOのDeepLabv3でセマンティックセグメンテーションを行います。torchvisionのResNetとMobileNetV3チェックポイントのインストール、推論、検証、エクスポートを説明します。
lead: >-
  各ピクセルを分類する前に、複数のdilation rateで特徴量を並列にプーリングするセマンティックセグメンテーションネットワーク（atrous
  spatial pyramid pooling）です。LibreYOLOはセマンティックセグメンテーション専用で提供します。
keywords:
  - DeepLabv3 使い方
  - atrous spatial pyramid pooling
  - ASPP
  - セマンティックセグメンテーション
  - dense prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W)クラスID
        print(mask.classes)      # 画像内に存在するクラスIDをソート
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeepLabv3r50-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeepLabv3r50-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=onnx

        libreyolo export model=LibreDeepLabv3r50-sem.pt format=tensorrt
        half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreDeepLabv3r50-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 7abf11ebb6cece18
---

## インストール

DeepLabv3に任意の追加パッケージは必要ありません。インポートするものはすべて基本
インストールに含まれます。

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

バックボーンは、dilated ResNet-50、dilated ResNet-101、dilated MobileNetV3-Largeの
3種類です。これはDeepLabv3でありDeepLabv3+ではないため、デコーダー段階もCRF refinementも
ありません。論文独自の参照コードではなく、torchvisionの実装に準拠しています。

LibreYOLOはDeepLabv3を学習しません。このファミリーで`train()`を呼び出すと
`NotImplementedError`が発生し、上記の[サポート層](/docs/models)でも推論専用と示されています。
公開されている3個のチェックポイントは、torchvision独自のCOCO-with-VOC-label重みを
LibreYOLOのローダー向けに変換したものです。

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

