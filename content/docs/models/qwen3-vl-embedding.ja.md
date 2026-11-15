---
title: Qwen3-VL-Embedding
families: [qwen3vl_embedding]
seo_title: "PythonでQwen3-VL-Embedding：画像とテキストの検索用ベクトル"
description: "LibreYOLOでQwen3-VL-Embedding-2Bを実行します。画像とテキストを1つの2048次元空間に埋め込み、検索に使えます。Apache-2.0。"
lead: "Qwen3-VL-Embeddingは、Qwen3-VLをベースにしたQwenの埋め込みモデルです。画像とテキストを1つのベクトル空間に写像するため、テキストのクエリからそれが説明するページや写真を見つけられます。LibreYOLOは2Bサイズをembedタスクで実行し、これはライブラリで初のVLMベースの埋め込みモデルです。"
keywords: [Qwen3-VL-Embedding, Qwen, マルチモーダル エンベディング, 画像検索, テキストから画像 検索, ドキュメント検索, エンベディング python, LibreVLM]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-embedding-2b")   # 初回使用時に重みをダウンロード
        result = model.predict(SAMPLE_IMAGE)
        print(result.embeddings.shape)              # (1, 2048)、単位長
    - label: 検索
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("qwen3-vl-embedding-2b")
        pages = model.embed("pages/")               # (N, 2048)、画像ごとに1行

        # クエリにはタスク指示を付け、画像はデフォルトのまま
        q = model.embed_text(
            ["total due on the invoice"],
            prompt="Retrieve images or text relevant to the user's query.",
        )                                           # (1, 2048)
        scores = q @ pages.T                        # コサイン類似度
source_hash: 0a217649e02fb6ae
---

## インストール

Qwen3-VL-EmbeddingはLibreYOLOのVLM層に属し、`vlm` extraが必要です。

```bash
pip install "libreyolo[vlm]"
```

## 推論

重みは初回使用時に`LibreYOLO/LibreQwen3VLEmbedding2b`からダウンロードされます。これはパラメータを変更していない`Qwen/Qwen3-VL-Embedding-2B`のミラーです。アーキテクチャはtransformersのQwen3-VL実装で動作します。

<code-tabs name="predict" />

このファミリーは`LibreVLM()`、または`LibreQwen3VLEmbedding`クラスで読み込みます。CLIP、SigLIP2、PEと同じ[embed](/docs/tasks/embeddings)の契約に従います。`predict(image)`は形状`(1, 2048)`の`Results.embeddings`を返し、`embed(source)`は画像ごとに1行を積み重ね、`embed_text(texts, prompt=None)`は同じ空間のテキスト行を返します。行はL2正規化されているため、内積がそのままコサイン類似度になります。

各入力はチャット形式です。システム指示のあとに画像またはテキストが続きます。最後のトークンの隠れ状態を正規化したものがベクトルです。`prompt`がその指示です。デフォルトはアップストリームの`"Represent the user's input."`です。画像とテキストの両方で変更するには構築時に`prompt=`を指定し、呼び出しごとに変えるには`embed_text`に渡してください。検索では、クエリにタスク指示を与え、ドキュメントはデフォルトのままにしてください。

画像は32ピクセルのグリッド上で4,096〜1,843,200ピクセルにリサイズされ、アップストリームのリファレンス埋め込み実装と一致します。その実装と比べると、CPU上のfloat32でテキストベクトルの差は最大7.4e-7、画像ベクトルのコサインは0.99998以上です。

## バリアント

サイズは`2b`の1つです。8Bサイズはラップしていません。画像とテキストを組み合わせた入力、動画クリップ（動画はフレームごとに埋め込まれます）、オプションとしてのMatryoshka切り詰めには対応していません。代わりに先頭の`k`列を切り出して再正規化してください。

## エクスポート

推論のみです。`chat()`、`set_classes()`、`train()`、`val()`、`export()`は例外を発生させます。

## ライセンス

<provenance-box>

Qwen3-VL-Embedding-2Bは[AlibabaのQwenチーム](https://github.com/QwenLM/Qwen3-VL-Embedding)によるApache-2.0です。商用利用が可能です。ミラーはゲートされていません。アップストリームのHugging FaceリポジトリにはLICENSEファイルが含まれないため、GitHubリポジトリのApache-2.0の本文を同梱しています。

</provenance-box>

## 引用

<citation-block />
