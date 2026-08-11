---
title: Swin Transformer
families:
  - swin
seo_title: Swin Transformer：LibreYOLOのLibreSwinで画像を分類
description: >-
  LibreYOLOでSwin
  Transformer分類器を使い、推論、検証、エクスポートを行います。重みはMITで、ファインチューニングにはまだ対応していません。
lead: >-
  Swin Transformer V1は、画像全体ではなく、ずらした局所ウィンドウ内でアテンションを計算する階層型Vision
  Transformerです。LibreYOLOは画像分類向けに4つのサイズを提供します。
keywords:
  - Swin Transformer
  - 階層型 vision transformer
  - shifted window attention
  - 画像分類
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwint-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwint-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")

        # dataはtrain/とval/にクラス別フォルダーを持つディレクトリルート
        # データセットYAMLではなくImageFolder配置
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwint-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwint-cls.pt format=onnx
        libreyolo export model=LibreSwint-cls.pt format=tensorrt half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で振り分けるためエクスポート済み成果物も
        # チェックポイントと同様に読み込まれて同じResultsオブジェクトを返す
        model = LibreYOLO("LibreSwint-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: faa6bbacae62d88e
---

## インストール

Swinにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

分類器は `result.boxes` ではなく `result.probs` を返します。`top1` と `top5` はクラスインデックスを示し、`top1conf` と `top5conf` はその信頼度を示します。最終アテンション段階がその解像度向けに構築されているため、すべてのサイズが固定の224px入力を使用します。異なる `imgsz` を渡すと、推論、検証、エクスポートはすべてエラーになります。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

サイズはtinyからlargeまでの4つで、同じshifted-windowタワーから構築され、埋め込み幅と段階の深さが異なります。largeはImageNet-22kで事前学習し、ImageNet-1kでファインチューニングされています。その他の3つはImageNet-1kで直接学習されています。LibreYOLOはこのファミリーを推論専用として提供します。推論、ImageNet形式のtop-1/top-5検証、エクスポートには対応し、アップストリームのImageNet学習レシピは実装されていません。

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
