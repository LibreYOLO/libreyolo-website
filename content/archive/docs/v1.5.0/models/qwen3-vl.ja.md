---
title: Qwen3-VL
families:
  - qwen3vl
seo_title: LibreYOLOのQwen3-VL：オープンボキャブラリ検出
description: >-
  LibreYOLOでQwen3-VLを使用します。インストール、オープンボキャブラリの設定、AlibabaによるApache-2.0ライセンスの視覚言語モデルを使った推論またはチャットについて説明します。
lead: >-
  Qwen3-VLはネイティブの2Dグラウンディングを備えたAlibabaの視覚言語モデルです。LibreYOLOはこれをオープンボキャブラリ物体検出器としてラップし、自由形式のチャットを直接公開します。クラス一覧を指定して検出するか、質問してください。
keywords:
  - Qwen3-VL
  - 視覚言語モデル
  - オープンボキャブラリ検出
  - grounding
  - Alibaba
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-4b")
        model.set_classes(["forklift", "pallet", "safety vest"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: チャット
      language: python
      code: >
        from libreyolo import LibreVLM, SAMPLE_IMAGE


        model = LibreVLM("qwen3-vl-4b")


        # 検出用の簡易機能の下にある脱出口 任意の質問に対応し

        # バウンディングボックスの質問だけに限定されない

        answer = model.chat(SAMPLE_IMAGE, "How many people are wearing a safety
        vest?")

        print(answer)
source_hash: ee225b6221d624d9
---

## インストール

Qwen3-VLはLibreYOLOの検出器として使うVLM階層に属します。これは独自のファクトリーを持ち、チェックポイントベースのファミリーとは別の製品面です。`vlm` 追加パッケージが必要です。

```bash
pip install "libreyolo[vlm]"
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。引数なしで `LibreVLM()` を呼び出すと、デフォルトでQwen3-VL-4Bが使用されます。

<code-tabs name="predict" />

このファミリーは `LibreYOLO()` ではなく、`LibreVLM()` ファクトリーから読み込みます。VLMファミリーはチェックポイントローダーを宣言しないため、他のモデルページで説明しているファイル接尾辞による振り分けはここでは適用されません。`set_classes()` はQwen3-VLに探させるボキャブラリを設定します。この設定は維持されるため、再設定するまで、その後のすべての `predict()` または `track()` 呼び出しに適用されます。すべての検出結果に同じ仮信頼度が設定されるため、`conf` によるフィルタリングは順位付けではなく、すべてを残すかすべてを除外するかのどちらかです。このファミリーでは `iou` が効果を持ち、後から得られた同じクラスのボックスが、すでに保持したボックスとしきい値を超えて重なると除外されます。これを行わないと、反復生成器が1つの物体に対してほぼ重複したボックスを出力する場合があるためです。Florence-2やKosmos-2とは異なり、Qwen3-VLは `chat()` を通じて自由形式の質問にも回答します。これは `LibreVLM` ファクトリーで説明しているものと同じ脱出口です。LibreYOLOのCLIはこの階層に対応しておらず、`libreyolo predict model=...` 形式はありません。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

サイズはQwen3-VL-2B-Instruct、Qwen3-VL-4B-Instruct、Qwen3-VL-8B-Instructの3つで、`LibreVLM("qwen3-vl-2b")`、`LibreVLM("qwen3-vl-4b")`、`LibreVLM("qwen3-vl-8b")` で読み込みます。3つとも公称入力は1024 pxですが、ネットワークに渡す実際のキャンバスはQwenのプロセッサー独自のスマートリサイズが決定します。そのため、この値はサイト内の他のファミリーのような固定動作解像度ではありません。LibreYOLOは3つのサイズの精度を比較するベンチマークを公開していません。

LibreYOLOはQwen3-VLの学習、検証、エクスポートを行いません。この階層のすべてのファミリーで、`train()`、`val()`、`export()` は `NotImplementedError` を発生させます（上のサポート階層を参照）。組み込みのカスタムボキャブラリが必要な場合は、アップストリームでQwen3-VLをファインチューニングし、得られた重みを読み込んでください。すべての検出結果が同じ仮信頼度を持つため、COCO形式の検証を行う代わりに、`predict()` の出力を目視で確認してください。

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
