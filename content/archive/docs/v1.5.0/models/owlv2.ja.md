---
title: OWLv2
families:
  - owlv2
seo_title: LibreYOLOのOWLv2：ゼロショット物体検出
description: >-
  LibreYOLOのOWLv2を使い、テキストで記述した任意の物体を検出します。openvocab追加パッケージをインストールし、自由テキストのボキャブラリで推論します。
lead: >-
  OWLv2はGoogle
  Researchが開発したオープンボキャブラリ物体検出器で、CLIP形式のエンコーダーによるテキスト埋め込みベクトルと画像領域を照合してスコアを付けます。LibreYOLOは、オープンボキャブラリ検出器階層の推論専用ファミリーとしてこれをラップします。
keywords:
  - OWLv2
  - OWL-ViT
  - オープンボキャブラリ物体検出
  - ゼロショット物体検出
  - テキスト条件付き 検出器
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: デフォルトのボキャブラリ
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        # set_classes()を省略するとこの階層のデフォルトであるCOCO-80を維持
        model = LibreOpenVocab("owlv2-l14")
        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        print(result.names)
source_hash: 2d0ce68af0daabb7
---

## インストール

OWLv2はLibreYOLOのオープンボキャブラリ検出器階層を通じて読み込まれます。この階層には `openvocab` 追加パッケージが必要です。

```bash
pip install "libreyolo[openvocab]"
```

この追加パッケージは、この階層から呼び出すHugging Faceライブラリの `transformers` と `timm` を導入します。

## 推論

OWLv2は、LibreYOLOが `LibreYOLO()` を通じて読み込むチェックポイントではありません。関連する `LibreOpenVocab` ファクトリーを通じて読み込まれます。このファクトリーは初回使用時にHugging Faceのスナップショットをダウンロードし、`weights/` の下にキャッシュします。

<code-tabs name="predict" />

`set_classes()` は維持されるテキストボキャブラリを設定します。一覧を置き換えるには再度呼び出し、省略するとデフォルトのCOCO-80ラベルが維持されます。各ラベルは、テキストタワーへ到達する前に固定のプロンプトテンプレートで囲まれます。これは `transformers` の `Owlv2ForObjectDetection` が学習された方法と一致します。

OWLv2にはテキストトークンのしきい値がありません。検出結果をフィルタリングするのは `conf` だけで、`text_threshold` を渡すとエラーになります。API互換性のため `iou` は受け付けますが、警告を表示して何も行いません。ここではNMSを実行しないためです。`imgsz` と `augment=True` は即座に拒否されます。リサイズは `transformers` プロセッサーが管理し、テスト時拡張はこの階層の対象外です。1枚の画像に対する `predict()` は一覧ではなく1つの `Results` を返します。複数の結果を得るには、ディレクトリ、画像一覧を渡すか、動画ソースで `stream=True` を指定します。このファミリーにはCLI経路がありません。`libreyolo predict` は `LibreYOLO()` を通じて `.pt` チェックポイントだけを読み込むため、`LibreOpenVocab` ファミリーはPythonから実行します。ソースの種類とストリーミングについては、[推論](/docs/predict)を参照してください。

## バリアント

チェックポイントは `b16`（base、パッチサイズ16）と `l14`（large、パッチサイズ14）の2つです。サイズを指定しない場合、この階層のデフォルトは `b16` です。どちらも `transformers` の `Owlv2ForObjectDetection` を通じてGoogle Researchの公式リリースをミラーしています。アップストリームのファイルを保持するLibreYOLO管理のHugging Faceスナップショットへ1回だけダウンロードされます。このファミリーの精度値やレイテンシ値はまだ公開されていません。

学習、データセット検証、エクスポートはすべてこの階層の対象外です。`train()`、`val()`、`export()` はすべて無条件に `NotImplementedError` を発生させます。これは公開済みチェックポイントを扱う推論専用ラッパーです。

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
