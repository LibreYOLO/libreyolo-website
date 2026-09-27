---
title: Kosmos-2
families:
  - kosmos2
seo_title: LibreYOLOのKosmos-2：グラウンディング物体検出
description: >-
  LibreYOLOでKosmos-2を使用します。インストール、オープンボキャブラリの設定、MicrosoftによるMITライセンスのモデルを使ったグラウンディング済みボックスの推論について説明します。
lead: >-
  Kosmos-2はMicrosoftのグラウンディングモデルです。画像にキャプションを付け、そのキャプション内の各名詞句をボックスで特定します。LibreYOLOはこれをオープンボキャブラリ物体検出器としてラップします。推論時にクラス一覧を指定してください。
keywords:
  - Kosmos-2
  - 視覚言語モデル
  - grounding
  - オープンボキャブラリ検出
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 動画
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])

        # ライブラリが受け付ける任意のソース ファイル フォルダー URL Webカメラ番号
        # RTSPストリームまたは.streams一覧
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: 60e0796f34be6d59
---

## インストール

Kosmos-2はLibreYOLOの検出器として使うVLM階層に属します。これは独自のファクトリーを持ち、チェックポイントベースのファミリーとは別の製品面です。`vlm` 追加パッケージが必要です。

```bash
pip install "libreyolo[vlm]"
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。LibreYOLOはMicrosoft独自の `microsoft/kosmos-2-patch14-224` リポジトリを直接読み込みます。Florence-2とは異なり、コミュニティによる再アップロードは不要です。

<code-tabs name="predict" />

このファミリーは `LibreYOLO()` ではなく、`LibreVLM()` ファクトリーから読み込みます。VLMファミリーはチェックポイントローダーを宣言しないため、他のモデルページで説明しているファイル接尾辞による振り分けはここでは適用されません。`set_classes()` はKosmos-2に探させるボキャブラリを設定します。この設定は維持されるため、再設定するまで、その後のすべての `predict()` または `track()` 呼び出しに適用されます。Kosmos-2はラベルを正確に照合するのではなく名詞句をグラウンディングするため、LibreYOLOのラッパーは部分一致を受け付けます。`"boat"` というクラス名は、生成された「the boats」のような句にも一致します。すべての検出結果に同じ仮信頼度が設定されるため、`conf` によるフィルタリングは順位付けではなく、すべてを残すかすべてを除外するかのどちらかです。また、`iou` はここでは効果がありません。ラッパーはグラウンディング済みエンティティから、重複除去を行わずに検出一覧を直接構築するためです。Kosmos-2はチャットテンプレートではなく `<grounding>` プロンプトで動作するため、`chat()` は `NotImplementedError` を発生させます。LibreYOLOのCLIはこの階層に対応しておらず、`libreyolo predict model=...` 形式はありません。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

サイズは224 pxの `kosmos-2-patch14-224` 1つで、`LibreVLM("kosmos-2")` で読み込みます。これは2023年頃のモデルであり、LibreYOLO独自のラッパーでは、この階層の新しい検出器よりグラウンディングが粗いと記載されています。

LibreYOLOはKosmos-2の学習、検証、エクスポートを行いません。この階層のすべてのファミリーで、`train()`、`val()`、`export()` は `NotImplementedError` を発生させます（上のサポート階層を参照）。組み込みのカスタムボキャブラリが必要な場合は、アップストリームでKosmos-2をファインチューニングし、得られた重みを読み込んでください。すべての検出結果が同じ仮信頼度を持つため、COCO形式の検証を行う代わりに、`predict()` の出力を目視で確認してください。

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
