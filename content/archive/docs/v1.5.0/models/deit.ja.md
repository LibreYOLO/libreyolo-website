---
title: DeiT
families:
  - deit
seo_title: DeiT画像分類器：推論、検証、エクスポート
description: >-
  LibreYOLOでDeiT画像分類器を実行します。Apache-2.0のtiny、small、baseサイズを備えた、固定済みの推論専用museumファミリーです。
lead: >-
  DeiT（Data-efficient image
  Transformer）は、追加の事前学習データを使わずImageNet-1kだけで学習した標準的なVision
  Transformer分類器です。LibreYOLOはtiny、small、baseのpatch-16サイズを固定済みの推論専用モデルとして提供します。
keywords:
  - DeiT 使い方
  - Vision Transformer
  - ViT
  - 画像分類
  - ImageNet
  - data-efficient training
  - museum model
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeiTb-cls.pt")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeiTb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeiTb-cls.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeiTb-cls.pt format=onnx
        libreyolo export model=LibreDeiTb-cls.pt format=tensorrt half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreDeiTb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 9c67c8554b2af5c6
---

## インストール

DeiTには基本パッケージ以外の追加パッケージは必要ありません。

```bash
pip install libreyolo
```

## 推論

このファミリーは推論専用です。`train()`は`NotImplementedError`を発生させるため、このページに
「学習」セクションはありません。推論、検証、エクスポートにはすべて対応します。重みは初回使用時に
Hugging Faceからダウンロードされ、ローカルにキャッシュされます。ファイル名の`-cls`接尾辞は
必須で、分類タスクを選択します。

<code-tabs name="predict" />

返される`Results`オブジェクトには`boxes`ではなく`probs`テンソルが入ります。`top1`と`top5`は
1,000個のImageNet-1kクラスのインデックスで、`top1conf`は最上位の推論結果に対するsoftmax
スコアです。各サイズの入力解像度は位置埋め込みベクトルによって固定されています。前処理はその
解像度へリサイズして中央クロップし、異なる`imgsz`を渡すと暗黙的にリサンプリングせず例外を
発生させます。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## 検証

`val()`は、一般的な`train/<class>/`と`val/<class>/`のフォルダー構造で配置された
データセットに対して測定したtop-1精度とtop-5精度の辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポートした成果物はファイル接尾辞に基づいて`LibreYOLO()`から再読み込みされます。そのため、
`.onnx`または`.engine`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。
LibreYOLOをインストールせずにランタイムからグラフを直接実行する方法にも対応しますが、その場合は
前処理と後処理を自分で実装します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

