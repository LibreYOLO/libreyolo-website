---
title: SigLIP2
families:
  - siglip2
seo_title: LibreYOLOのSigLIP2：ゼロショット分類と埋め込み
description: LibreYOLOでSigLIP2を使い、シグモイドによるマルチラベルスコアリングでゼロショット画像分類と画像・テキスト埋め込みを行います。学習は不要です。
lead: >-
  SigLIP2は、固定ラベルセット全体で共有するsoftmaxではなく、クラスごとに独立したsigmoidを使って画像とテキストプロンプトのスコアを算出するデュアルタワーモデルです。LibreYOLOは、学習ステップなしでゼロショット分類と画像・テキスト埋め込みに対応します。
keywords:
  - SigLIP2
  - SigLIP 2
  - ゼロショット分類
  - 画像 エンベディング
  - テキスト エンベディング
  - オープンボキャブラリ
  - 多言語 画像分類
  - sigmoid loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # set_classes()を呼び出さない場合、CLI predictはモデルがデフォルトで読み込む

        # 1,000個のImageNetクラス名を使用

        libreyolo predict model=LibreSigLIP2b16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: マルチラベルsigmoidスコアリング
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a dog", "a cat", "outdoors"], multi_label=True)
        r = model(SAMPLE_IMAGE)

        # クラスごとに独立した確率: 複数またはどのクラスも同時に高スコアになり得る
        # 一方、Softmax (デフォルト) は単一ラベル分布に正規化し
        # LibreCLIPの動作と一致
        for i, name in model.names.items():
            print(name, float(r.probs.data[i]))
    - label: 画像とテキストの埋め込み
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")
        image_embed = model(SAMPLE_IMAGE).embeddings.data
        text_embed = model.embed_text("a photo of a forklift")

        # どちらもL2正規化済みのため、通常の内積がコサイン類似度になる
        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        # dataはtrain/分割を持つImageFolderルートで、フォルダー名が
        # この実行のゼロショットクラスプロンプトになる
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSigLIP2b16-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # 現在のset_classes()ラベルと入力解像度がグラフに埋め込まれる
        # いずれかを変更した後は再エクスポートする。エクスポート時にはmulti_labelを
        # False (デフォルト) にする必要がある
    - label: CLI
      language: bash
      code: |
        # ここではset_classes()を呼び出さないため、モデルが読み込むデフォルトの
        # 1,000個のImageNetクラスが埋め込まれる
        libreyolo export model=LibreSigLIP2b16-cls.pt format=onnx
    - label: 埋め込みのエクスポート
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed"は画像タワーだけをトレースするため、クラスは不要
        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: f992655747fd8819
---

## インストール

SigLIP2には専用の追加パッケージが必要です。このパッケージにより、多言語トークナイザーが使うSentencePieceパッケージもインストールされます。

```bash
pip install "libreyolo[siglip2]"
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

`set_classes()`は、このモデルをオープンボキャブラリ分類器として機能させる基本操作です。各ラベルをすべてのプロンプトテンプレートに展開し、エンコードした結果を平均して、得られた `[K, D]` 行列を分類ヘッドとしてキャッシュするため、画像ごとに再計算されません。クラスを変更するには、いつでも再度呼び出してください。呼び出さない場合、LibreSigLIP2は1,000個のImageNet-1kクラス名が設定された状態で読み込まれます。

SigLIPは各クラスを独立してスコアリングします：`logit = scale * (image . text) + bias`。デフォルトでは、そのロジットの集合は引き続きsoftmaxに渡され、LibreCLIPの `top1`・`top5` の動作に一致する単一ラベル分布になります。`set_classes()`に `multi_label=True` を渡すか、構築時に指定すると、代わりに独立したsigmoid確率へ切り替わるため、同じ画像で複数のクラス、またはどのクラスも高スコアになり得ます。トークナイザーは多言語SentencePieceモデル（Gemma語彙）なので、英語以外のクラス名も同じように機能します。

`task="embed"` では、推論によってクラス確率ではなく、入力ごとにL2正規化された画像ベクトルが1つ返されます。また、`embed_text()`は同じベクトル空間の正規化済みテキスト行を返すため、両者の通常の内積がコサイン類似度になります。どちらのタスクでも `iou` は効果がなく、NMSステップはありません。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## 検証

`val()`はImageFolderの `train/` 分割以下にあるクラスフォルダー名を読み取り、それらを指定して `set_classes()` を呼び出した後、softmaxスコアリングでゼロショットのtop-1精度とtop-5精度を測定します。学習するものがないため、精度は重みの更新ではなく、クラス名がプロンプトとしてどう解釈されるかに左右されます。検証の対象は `task="classify"` だけです。`task="embed"` にはデータセット検証機能がありません。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポートでは、モデルの現在の状態が固定グラフに埋め込まれます。`task="classify"` では、`set_classes()` で最後に設定したラベルとエクスポート時の解像度が、学習済みのscaleおよびbiasを持つ最終線形層に埋め込まれます。そのため、エクスポートされたグラフはテキストタワーもトークナイザーも含まない通常の `[B, K]` 画像分類器になります。クラスまたはサイズを変更した後は、再度エクスポートしてください。`multi_label=True` モードでのエクスポートは実装されていません。先に `False` に戻してください。`task="embed"` のエクスポートでは、画像タワーだけをトレースします。どちらにもONNX opset 14以降が必要で、エクスポーターがデフォルトで設定します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。どちらもGoogleのApache-2.0ライセンスの `siglip2-base-patch16-256` および `siglip2-so400m-patch14-384` チェックポイントから変換されたもので、COCO学習の実行結果ではありません。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

