---
title: Florence-2
families:
  - florence2
seo_title: LibreYOLOのFlorence-2：オープンボキャブラリ検出
description: >-
  LibreYOLOでFlorence-2を使用します。インストール、オープンボキャブラリの設定、MicrosoftによるMITライセンスの視覚モデルを使ったボックスの推論について説明します。
lead: >-
  Florence-2はMicrosoftの視覚基盤モデルであり、固定された検出ヘッドを通す代わりにタスクトークンで指示します。LibreYOLOはこれをオープンボキャブラリ物体検出器としてラップします。推論時にクラス一覧を指定してください。
keywords:
  - Florence-2
  - 視覚言語モデル
  - オープンボキャブラリ検出
  - grounding
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 動画
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])

        # ライブラリが受け付ける任意のソース ファイル フォルダー URL Webカメラ番号
        # RTSPストリームまたは.streams一覧
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: ad26d9056465d662
---

## インストール

Florence-2はLibreYOLOの検出器として使うVLM階層に属します。これは独自のファクトリーを持ち、チェックポイントベースのファミリーとは別の製品面です。`vlm` 追加パッケージが必要です。

```bash
pip install "libreyolo[vlm]"
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。LibreYOLOは元の `microsoft/Florence-2-*` リポジトリではなく、florence-communityが再アップロードしたチェックポイントをダウンロードします。その理由は「ライセンス」を参照してください。

<code-tabs name="predict" />

このファミリーは `LibreYOLO()` ではなく、`LibreVLM()` ファクトリーから読み込みます。VLMファミリーはチェックポイントローダーを宣言しないため、他のモデルページで説明しているファイル接尾辞による振り分けはここでは適用されません。`set_classes()` はFlorence-2に画像内で探させるボキャブラリを設定します。この設定は維持されるため、再設定するまで、その後のすべての `predict()` または `track()` 呼び出しに適用されます。返される `Results` は他のファミリーと同じ形状の `boxes` を持ちますが、すべての検出結果に同じ仮信頼度が設定されます。そのため、`conf` によるフィルタリングは順位付けではなく、すべてを残すかすべてを除外するかのどちらかです。また、`iou` は効果がありません。Florence-2のラッパーは解析済みタスクトークンの出力から、重複除去を行わずに検出一覧を直接構築するためです。Florence-2はチャットテンプレートではなく `<OPEN_VOCABULARY_DETECTION>` タスクトークンで動作するため、ここで `chat()` は `NotImplementedError` を発生させます。LibreYOLOのCLIはこの階層に対応しておらず、`libreyolo predict model=...` 形式はありません。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

サイズはFlorence-2-baseとFlorence-2-largeの2つで、どちらも768 pxです。`LibreVLM("florence-2-base")` または `LibreVLM("florence-2-large")` で読み込みます。LibreYOLOは両者の精度を比較するベンチマークを公開していません。

LibreYOLOはFlorence-2の学習、検証、エクスポートを行いません。この階層のすべてのファミリーで、`train()`、`val()`、`export()` は `NotImplementedError` を発生させます（上のサポート階層を参照）。組み込みのカスタムボキャブラリが必要な場合は、アップストリームでFlorence-2をファインチューニングし、得られた重みを読み込んでください。すべての検出結果が同じ仮信頼度を持つため、COCO形式の検証を行う代わりに、`predict()` の出力を目視で確認してください。

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
