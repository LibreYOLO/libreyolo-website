---
title: LingBot-Vision
families:
  - lingbotvision
seo_title: LingBot-Vision：LibreYOLOのセマンティックセグメンテーション
description: >-
  LibreYOLOでApache-2.0のViTバックボーンを持つLingBot-Visionをセマンティックセグメンテーションに使います。s・b・lサイズのインストール、推論、学習、検証、エクスポートを説明します。
lead: >-
  LingBot-Visionは、密な空間認識に向けた境界中心のmasked
  modelingで自己教師あり学習された視覚Transformerバックボーンのファミリーで、Robbyantが公開しています。LibreYOLOはバックボーンを密出力ヘッドと組み合わせ、セマンティックセグメンテーションという1つのタスクでサポートします。
keywords:
  - LingBot-Vision
  - セマンティックセグメンテーション
  - vision transformer
  - 自己教師あり学習
  - 境界モデリング
  - Robbyant
  - dense prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLingBotVisions-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python（linear probe）
      language: python
      code: |
        from libreyolo import LibreYOLO

        # アップストリームの評価手順に合わせ、デフォルトでバックボーンを凍結
        # 学習するのは1x1 dense headだけ
        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(data="my-dataset.yaml", epochs=20, imgsz=512, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 imgsz=512 batch=16
    - label: 完全ファインチューニング
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(
            data="my-dataset.yaml", epochs=20, imgsz=512, batch=16,
            freeze_backbone=False,
        )
    - label: マルチGPU
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLingBotVisions-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="coreai", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLingBotVisions-sem.pt format=onnx imgsz=512
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で経路を選ぶため、エクスポート済み成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreLingBotVisions-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: c47b33fdc6fa1139
---

## インストール

LingBot-Visionに任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

`result.semantic_mask` は密なクラスマップを保持します。`.data` は元画像サイズのクラスIDを持つ `(H, W)` テンソルで、`.classes` は実際に存在するクラスIDを列挙します。インスタンスごとの検出結果がないため、`result.boxes` は `None` です。`conf` と `iou` はAPIの一貫性のために受け付けますが、出力は変更しません。モデルが絞り込み対象の検出結果ではなく、pixelごとに1つのクラスを返すためです。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

公開済みサイズはs、b、lの3つで、1.1BパラメータのViT-g/16 teacherから蒸留されています。サイズ `g` のteacher自体もLibreYOLOで読み込み、ファインチューニングできますが、LibreYOLOは独自の `g` チェックポイントをホストしません。

<checkpoint-table />

## 学習

`train()` は公開済みチェックポイントをファインチューニングします。デフォルトの手順はアップストリームの報告にあるlinear probeです。ViTバックボーンを凍結して1x1 dense headだけを学習し、上記のLibreYOLOホストの重みが生成された方法に合わせます。ネットワーク全体をファインチューニングするには `freeze_backbone=False` を渡し、それに応じて `lr0` を下げることを想定してください。

<code-tabs name="train" />

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

アップストリームのリリースでは、このViTをMeta AIが公開したDINOv2・DINOv3アーキテクチャに基づくものと説明しています。Robbyantは実装をApache-2.0で配布しており、このLibreYOLO移植はRobbyantリポジトリだけから行われ、MetaのDINOv2またはDINOv3コードからは行われていません。

</provenance-box>

## 引用

<citation-block />

