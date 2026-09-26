---
title: VGG
families:
  - vgg
seo_title: VGG：LibreYOLOでVGG-16・19画像分類器を実行
description: >-
  LibreYOLOでVGG分類器の推論、検証、エクスポートを行います。BSD-3-Clauseのtorchvision重みを使えます。ファインチューニングにはまだ対応していません。
lead: >-
  VGGは大きなfilterの代わりに、小さな3×3畳み込みを均一に積み重ねて構築した畳み込み画像分類器です。LibreYOLOは画像分類向けに、16層と19層のplain版とbatch
  normalization版を提供します。
keywords:
  - VGG 使い方
  - VGG-16
  - VGG-19
  - 畳み込みニューラルネットワーク
  - 画像分類
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVGG16-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreVGG16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")

        # dataはtrain/とval/のクラス別フォルダーを持つルートディレクトリ
        # データセットYAMLではなくImageFolderレイアウト
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreVGG16-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreVGG16-cls.pt format=onnx
        libreyolo export model=LibreVGG16-cls.pt format=tensorrt half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreVGG16-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 26eb6ff5811533fd
---

## インストール

VGGに任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

分類器は`result.boxes`ではなく`result.probs`を返します。`top1`と`top5`はクラス
インデックス、`top1conf`と`top5conf`はそれぞれの信頼度を示します。推論は固定224 px入力で
実行され、異なる`imgsz`を渡すと例外が発生します。入力ソース、ストリーミング、結果の処理に
ついては[推論](/docs/predict)を参照してください。

## バリアント

サイズは4種類です。16層と19層の畳み込み層に、それぞれplain版とbatch normalization版があります。
提供される重みはtorchvisionが後にImageNetでゼロから学習したもので、Oxford groupによる元の
2014年版Caffeリリースからの変換ではありません。LibreYOLOはこのファミリーを推論専用で提供します。
推論、ImageNet形式のtop-1・top-5検証、エクスポートに対応し、ファインチューニングは未実装です。

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

