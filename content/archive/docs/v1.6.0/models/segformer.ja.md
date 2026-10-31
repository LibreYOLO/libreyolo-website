---
title: SegFormer
families:
  - segformer
seo_title: SegFormer：LibreYOLOのセマンティックセグメンテーション
description: >-
  LibreYOLOでSegFormerを使い、b0からb5のサイズでADE20Kセマンティックセグメンテーションを行います。インストール、推論、学習、エクスポートに対応し、学習済み重みは非商用です。
lead: >-
  SegFormerは、階層型Mix Transformer（MiT）encoderと軽量なall-MLP decode
  headを組み合わせたセマンティックセグメンテーションTransformerです。以前のセグメンテーションTransformerで必要だった大規模なdecoderと固定位置encodingを避けています。LibreYOLOは6つのサイズで、セマンティックセグメンテーションという1つのタスクに対応します。
keywords:
  - SegFormer
  - セマンティックセグメンテーション
  - Mix Transformer
  - MiT
  - transformer segmentation
  - ADE20K
  - dense prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSegformerb0-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python（ファインチューニング）
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: スクラッチ学習
      language: python
      code: |
        from libreyolo.models.segformer.model import LibreSegformer

        # model_pathを指定しないため、ランダム初期化され、何もダウンロードされない
        # 学習済みチェックポイントの非商用条件を含まない重みを得る唯一の経路
        model = LibreSegformer(size="b0", nb_classes=150)
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: マルチGPU
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512

        libreyolo export model=LibreSegformerb0-sem.pt format=tensorrt imgsz=512
        half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で経路を選ぶため、エクスポート済み成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: c236895b991beabf
---

## インストール

SegFormerに任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

`result.semantic_mask` は密なクラスマップを保持します。`.data` は元画像サイズのクラスIDを持つ `(H, W)` テンソルで、`.classes` は実際に存在するクラスIDを列挙します。インスタンスごとの検出結果がないため、`result.boxes` は `None` です。`conf` と `iou` はAPIの一貫性のために受け付けますが、出力は変更しません。モデルは絞り込みや重複排除の対象となるインスタンスごとの検出結果ではなく、pixelごとに1つのクラスを返すためです。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

b0からb5まで6つのサイズがあり、同じall-MLP decode head設計を維持しながら、段階ごとにMix Transformer encoderの幅と深さを増やします。

<checkpoint-table />

## 学習

`train()` はデフォルトで公開済みチェックポイントをファインチューニングします。代わりに `LibreSegformer(...)` へ `model_path` を渡さなければ、ランダム初期化されたencoderとヘッドを構築してスクラッチ学習します。これは、学習済みチェックポイントの非商用制限をまったく含まない重みを得る唯一の経路です（[ライセンス](#licensing)を参照）。

<code-tabs name="train" />

設定を変更しなければ、trainerはSegFormer論文のADE20K手順に従います。バックボーンの基本学習率でAdamWを使い、decode headはその10倍の学習率で学習します。LayerNormとMix-FFN positional convolution以外のすべてにweight decayを適用し、ウォームアップを伴うlinear decay scheduleを使います。b3からb5の大きなサイズについて、収束はエンドツーエンドで検証されていません。

データセット、データ拡張、マルチGPU、loggerについては[学習](/docs/train)を参照してください。

## 検証

`val()` は、学習に使った形式の任意のデータセットに対して測定したmIoUとpixel accuracyを含む `metrics/` キーの辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポート済み成果物はファイル接尾辞に基づいて `LibreYOLO()` から再度読み込めます。そのため、`.onnx` または `.engine` ファイルはチェックポイントと同様に動作し、同じ `Results` を返します。[エクスポート](/docs/export)には、すべての形式で受け付ける引数が記載されています。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box>

LibreSegformerのencoderとdecode headは、NVlabs/SegFormerではなくHugging Face TransformersのApache-2.0 SegFormer実装をPyTorchへ移植したものです。NVIDIAの元のリポジトリは読んだりコピーしたりしておらず、ここでは論文の作者への帰属表示だけを目的に記載しています。上記の学習済みチェックポイントだけにNVIDIAの非商用制限があり、アーキテクチャとLibreYOLO独自のコードは一貫してMITです。

</provenance-box>

## 引用

<citation-block />

