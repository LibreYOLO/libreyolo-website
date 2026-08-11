---
title: ViT
families:
  - vit
seo_title: ViT：LibreYOLOで定番のVision Transformer分類器を実行
description: >-
  LibreYOLOでViT分類器の推論、検証、エクスポートを行います。Apache-2.0のAugReg重みを使えます。ファインチューニングにはまだ対応していません。
lead: >-
  定番のVision Transformerです。学習済みclass
  tokenを持ち、畳み込みを使わず、固定サイズの画像patchへ純粋なtransformerを適用します。LibreYOLOは画像分類向けに、AugReg事前学習済みの4サイズを提供します。
keywords:
  - ViT 使い方
  - Vision Transformer
  - AugReg
  - 画像分類
  - transformer 分類器
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTti-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreViTti-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")

        # dataはtrain/とval/のクラス別フォルダーを持つルートディレクトリ
        # データセットYAMLではなくImageFolderレイアウト
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreViTti-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreViTti-cls.pt format=onnx
        libreyolo export model=LibreViTti-cls.pt format=tensorrt half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreViTti-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: f63e98454913765a
---

## インストール

ViTに任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

分類器は`result.boxes`ではなく`result.probs`を返します。`top1`と`top5`はクラス
インデックス、`top1conf`と`top5conf`はそれぞれの信頼度を示します。前処理はtimmのAugReg
評価レシピを使い、固定224 px入力へリサイズして中央クロップします。crop比率0.9のbicubic補間です。
入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

サイズはtiny〜largeの4種類です。固定224 pxのpatch-16グラフを共有し、埋め込みベクトルの幅と
transformerの深さが異なります。LibreYOLOはこのファミリーを推論専用で提供します。推論、
ImageNet形式のtop-1・top-5検証、エクスポートに対応し、AugRegファインチューニングレシピは
未実装です。

## 検証

`val()`はImageFolder形式の分割（`train/`と`val/`サブフォルダーを持ち、クラスごとに1個の
フォルダーがあるディレクトリ）に対して実行し、top-1とtop-5の精度を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポートした成果物はファイル接尾辞に基づいて`LibreYOLO()`から再読み込みされます。そのため、
`.onnx`または`.engine`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。
[エクスポート](/docs/export)には、各形式が受け付ける引数と、一部の形式で必要になる追加パッケージの一覧があります。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

