---
title: Grounding DINO
families:
  - grounding_dino
seo_title: LibreYOLOのGrounding DINO：オープンセット検出
description: >-
  LibreYOLOのGrounding
  DINOを使い、テキストで記述した任意の物体を検出します。openvocab追加パッケージをインストールし、自由テキストのボキャブラリで推論します。
lead: >-
  Grounding DINOはIDEA
  Researchが開発したオープンセット物体検出器で、固定クラス一覧ではなく自由テキストのプロンプトと画像を照合してスコアを付けます。LibreYOLOは、オープンボキャブラリ検出器階層の推論専用ファミリーとしてこれをラップします。
keywords:
  - Grounding DINO
  - オープンボキャブラリ物体検出
  - オープンセット物体検出
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

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: テキストしきい値
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # confはボックススコアでtext_thresholdはデコードされた句の
        # トークンスコアでフィルタリング 未指定時はどちらもデフォルト0.25
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
        print(result.names)
source_hash: 06bd13b8e6a66038
---

## インストール

Grounding DINOはLibreYOLOのオープンボキャブラリ検出器階層を通じて読み込まれます。この階層には `openvocab` 追加パッケージが必要です。

```bash
pip install "libreyolo[openvocab]"
```

この追加パッケージは、この階層から呼び出すHugging Faceライブラリの `transformers` と `timm` を導入します。

## 推論

Grounding DINOは、LibreYOLOが `LibreYOLO()` を通じて読み込むチェックポイントではありません。関連する `LibreOpenVocab` ファクトリーを通じて読み込まれます。このファクトリーは初回使用時にHugging Faceのスナップショットをダウンロードし、`weights/` の下にキャッシュします。

<code-tabs name="predict" />

`set_classes()` は維持されるテキストボキャブラリを設定します。一覧を置き換えるには再度呼び出し、省略するとデフォルトのCOCO-80ラベルが維持されます。Grounding DINOは独自のテキスト出力から自由形式の句をデコードし、自分でそのボキャブラリに対応付けます。正規化後の完全一致を優先し、単語単位の一致も受け付けます。曖昧な句や一致しない句は推測せず除外されるため、`school bus` が `bus` または `school` だけに対応付けられることはありません。テキストエンコーダーのトークン上限を超える長いボキャブラリは複数のプロンプトに分割され、個別の順伝播として実行された後、`max_det` を上限とする1つの検出集合に統合されます。

API互換性のため `iou` は受け付けますが、警告を表示して何も行いません。ここではNMSを実行しないためです。`imgsz` と `augment=True` は即座に拒否されます。リサイズは `transformers` プロセッサーが管理し、テスト時拡張はこの階層の対象外です。1枚の画像に対する `predict()` は一覧ではなく1つの `Results` を返します。複数の結果を得るには、ディレクトリ、画像一覧を渡すか、動画ソースで `stream=True` を指定します。このファミリーにはCLI経路がありません。`libreyolo predict` は `LibreYOLO()` を通じて `.pt` チェックポイントだけを読み込むため、`LibreOpenVocab` ファミリーはPythonから実行します。ソースの種類とストリーミングについては、[推論](/docs/predict)を参照してください。

## バリアント

チェックポイントは `t` と `b` の2つです。サイズを指定しない場合、この階層のデフォルトは `t` です。どちらも `transformers` の `GroundingDinoForObjectDetection` を通じてIDEA Researchの公式リリースをミラーしています。アップストリームのファイルを保持するLibreYOLO管理のHugging Faceスナップショットへ1回だけダウンロードされます。このファミリーの精度値やレイテンシ値はまだ公開されていません。

学習、データセット検証、エクスポートはすべてこの階層の対象外です。`train()`、`val()`、`export()` はすべて無条件に `NotImplementedError` を発生させます。これは公開済みチェックポイントを扱う推論専用ラッパーです。

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
