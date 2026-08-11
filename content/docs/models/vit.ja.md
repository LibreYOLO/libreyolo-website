---
title: ViT
families:
  - vit
seo_title: ViT：LibreYOLOで古典的なVision Transformer分類器を実行
description: >-
  LibreYOLOでViT分類器を使い、推論、検証、エクスポートを行います。Apache-2.0のAugReg重みを使用し、ファインチューニングにはまだ対応していません。
lead: >-
  古典的なVision
  Transformerです。学習可能なクラストークンを持ち、畳み込みを使わず、固定サイズの画像パッチに純粋なTransformerを適用します。LibreYOLOは画像分類向けにAugRegで事前学習した4つのサイズを提供します。
keywords:
  - ViT
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

        # dataはtrain/とval/にクラス別フォルダーを持つディレクトリルート
        # データセットYAMLではなくImageFolder配置
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

        # ファクトリーはファイル接尾辞で振り分けるためエクスポート済み成果物も
        # チェックポイントと同様に読み込まれて同じResultsオブジェクトを返す
        model = LibreYOLO("LibreViTti-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: f63e98454913765a
---

## インストール

ViTにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

分類器は `result.boxes` ではなく `result.probs` を返します。`top1` と `top5` はクラスインデックスを示し、`top1conf` と `top5conf` はその信頼度を示します。前処理ではtimmのAugReg評価レシピに従い、0.9のクロップ率でバイキュービック補間を使って、固定の224px入力へリサイズして中央をクロップします。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

サイズはtinyからlargeまでの4つです。固定の224px、パッチ16のグラフを共有し、埋め込み幅とTransformerの深さが異なります。LibreYOLOはこのファミリーを推論専用として提供します。推論、ImageNet形式のtop-1/top-5検証、エクスポートには対応し、AugRegのファインチューニングレシピは実装されていません。

## 検証

`val()` はImageFolder形式の分割（`train/` と `val/` サブフォルダーを持ち、クラスごとに1つのフォルダーがあるディレクトリ）に対して実行され、top-1とtop-5の精度を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポート済み成果物はファイル接尾辞によって `LibreYOLO()` から再度読み込まれるため、`.onnx` または `.engine` ファイルはチェックポイントのように動作し、同じ `Results` を返します。[エクスポート](/docs/export)では、すべての形式が受け付ける引数と、一部の形式が追加するパッケージを説明しています。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
