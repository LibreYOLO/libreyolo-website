---
title: DINOv2
families:
  - dinov2
seo_title: 'LibreYOLOのDINOv2: セマンティック、分類、埋め込み'
description: >-
  LibreYOLOでDINOv2-with-Registersバックボーンを使用し、セマンティックセグメンテーション、画像分類、画像全体の埋め込みベクトルを実行します。すべてApache-2.0です。
lead: >-
  DINOv2はMeta AIがラベルなしで汎用的な画像特徴量を生成できるよう自己教師あり学習したVision
  Transformerです。LibreYOLOはDINOv2-with-Registersバックボーンをラップし、セマンティックセグメンテーション、画像分類、画像全体の埋め込みベクトルという3つのタスクに対応します。
keywords:
  - DINOv2
  - DINOv2 with registers
  - 自己教師あり学習
  - Vision Transformer
  - セマンティックセグメンテーション
  - 画像埋め込み
  - 特徴抽出
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: セマンティック
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # このファミリーにはLibreYOLOがホストするチェックポイントがないため
        # MetaのHugging Face組織からApache-2.0のDINOv2-with-Registers-small
        # バックボーンをダウンロードする 密なヘッドは学習するまでランダム初期化となる
        # 下記の学習を参照
        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        result = model(SAMPLE_IMAGE)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: 分類
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # nb_classes=にはデータセットのクラス数を指定する 線形ヘッドは
        # 学習するまでランダム初期化となる
        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
    - label: 埋め込み
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # すべてのタスクヘッドを迂回する バックボーンだけで十分なので
        # 有用性を得るためのファインチューニングは不要
        model = LibreDINOv2(size="s", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (1, D) L2正規化済み
    - label: バッチを埋め込み
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # 便利なラッパー predict()を実行して各行を1つの
        # (N, D)テンソルに積み重ねる
        features = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(features.shape)
  train:
    - label: セマンティック
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: 分類
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: マルチGPU
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(
            data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4,
            device="0,1",
        )
  val:
    - label: セマンティック
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: 分類
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
  export:
    - label: セマンティック
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.export(format="onnx")
    - label: 分類
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.export(format="onnx")
    - label: 埋め込み
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        model.export(format="tflite")
    - label: エクスポートしたファイルを使用
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリはファイルの拡張子に応じて振り分けるためエクスポートした
        # アーティファクトもチェックポイントと同様に読み込まれ同じResultsオブジェクトを返す
        # エクスポート時はタスクからファイル名を付ける ここではLibreDINOv2s-sem.onnx
        model = LibreYOLO("LibreDINOv2s-sem.onnx")
        result = model(SAMPLE_IMAGE)
source_hash: 4256e0a0398e5aaf
---

## インストール

LibreDINOv2が登録されるのは`transformers`がインストールされている場合だけです。これはRF-DETRがDINOv2バックボーンに必要とするものと同じ任意依存関係なので、同じ追加パッケージが必要です。

```bash
pip install "libreyolo[rfdetr]"
```

## 推論

LibreYOLOはLibreDINOv2のチェックポイントを公開していません。ファイルを読み込む代わりにラッパーを直接構築してください。`model_path=None`（デフォルト）では、初回使用時にMetaのApache-2.0ライセンスの`facebook/dinov2-with-registers-small`バックボーンをHugging Faceからダウンロードします。`task=`で、その上で実行する処理を選択します。

<code-tabs name="predict" />

`task="semantic"`と`task="classify"`は、バックボーンの上に密なヘッドまたは線形ヘッドを追加します。このヘッドはランダムに初期化され、[学習](#train)するまで実用になりません。`task="embed"`はすべてのヘッドを省略し、バックボーンの最終的な正規化済みCLSトークンを画像全体の1行として`result.embeddings`に返すため、学習は一切不要です。3つのタスクはいずれもインスタンス単位の検出を生成しないので、`result.boxes`は常に`None`です。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

`size`で選択するのはバックボーン自体ではなく、バックボーンの上に重ねるRF-DETR形式のプロジェクター幅です。すべてのサイズで同じDINOv2-S（small）エンコーダーを共有します。セマンティックセグメンテーションはDINOv2本来の正方形パッチグリッドで実行され、画像分類と埋め込みは線形プローブの学習に使用した、より小さい分類用解像度で実行されます。

## 学習

`task="semantic"`と`task="classify"`はどちらも学習できます。`task="embed"`には適合させるクラス依存のヘッドがないため、`train()`を呼び出すと`NotImplementedError`を送出します。

<code-tabs name="train" />

ここで主に使用するキーワード引数は、多くの他ファミリーで使う`batch`と`lr0`ではなく、`batch_size`と`lr`です。`batch`と`lr0`も引き続き受け付け、対応する引数にマッピングされますが、両方を渡すと競合エラーになります。実行結果を配置する主な方法として、`project=`と`name=`の代わりに`output_dir=`（デフォルトは`"runs/train"`）を使用します。ただし、`project=`と`name=`を直接渡す方法も引き続き機能します。データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`は`metrics/`キーを持つ辞書を返します。`task="semantic"`ではmIoUとピクセル精度、`task="classify"`ではtop-1精度とtop-5精度が含まれます。`task="embed"`には評価対象となる正解データがないため、`val()`を呼び出すと`NotImplementedError`を送出します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

各タスクは上記に示す異なる形式のサブセットに対応します。エクスポートしたアーティファクトはファイルの拡張子に基づいて`LibreYOLO()`で読み込めるため、`.onnx`や`.engine`ファイルもチェックポイントと同様に動作し、同じ`Results`を返します。[エクスポート](/docs/export)には各形式で受け付ける引数が記載されています。

<code-tabs name="export" />

## ライセンス

<provenance-box>

上記の「Weights」行には適用されるライセンスとしてApache-2.0が記載されていますが、このファミリーについてLibreYOLOのHugging Face組織で実際に再公開されているものはありません。LibreYOLOは独自のLibreDINOv2チェックポイントをホストしていません。`LibreDINOv2(model_path=None)`がダウンロードするのは、変更を加えていないMeta自身の`facebook/dinov2-with-registers-small`リポジトリです。

</provenance-box>

## 引用

<citation-block />
