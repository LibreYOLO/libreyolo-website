---
title: FOMO
families:
  - fomo
seo_title: FOMO：LibreYOLOで点位置推定、学習、エクスポート
description: >-
  LibreYOLOでFOMO（Faster Objects, More
  Objects）を実行します。多数の小さな物体を数えるための小型点位置推定検出器です。インストール、推論、学習、エクスポートを説明します。
lead: >-
  FOMOはグリッドベースの点位置推定器です。低解像度グリッドの各cellをbackgroundまたは物体中心に分類し、バウンディングボックス回帰は行いません。LibreYOLOはpointタスクでFOMOに対応します。
keywords:
  - FOMO
  - Faster Objects More Objects
  - 点位置推定
  - 重心 検出
  - 小物体 検出
  - エッジ AI
  - マイコン 物体検出
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # LibreFOMOの重みは自動ダウンロードされない (以下のCheckpointsを参照)
        # すでにローカルへダウンロードしたチェックポイントを指定
        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for point in result.points:
            print(point.cls, point.conf, point.xy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=./LibreFOMOs-point.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=40, batch=32, lr0=3e-4,
        )
    - label: CLI
      language: bash
      code: >
        # imgszは必須。CLIのデフォルトは640だが

        # sチェックポイントはネイティブ解像度96だけを受け付ける

        libreyolo train model=./LibreFOMOs-point.pt data=my-dataset.yaml
        imgsz=96 epochs=40 batch=32 lr0=3e-4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/grid_F1"])
        print(metrics["metrics/grid_precision"], metrics["metrics/grid_recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=./LibreFOMOs-point.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=./LibreFOMOs-point.pt format=onnx
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で経路を選ぶため、エクスポート済み成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
source_hash: 03015f2bcd9fe99d
---

## インストール

FOMOに基本パッケージ以外の追加パッケージは必要ありません。

```bash
pip install libreyolo
```

## 推論

このサイトのほかのすべてのファミリーと異なり、LibreFOMOの重みは自動ダウンロードされません。`LibreYOLO("LibreFOMOs-point.pt")` はディスク上でそのファイルを探し、Hugging Faceから取得する代わりに、ファイル名を示す `ValueError` を送出します。まず[LibreYOLO organization](https://huggingface.co/LibreYOLO)からチェックポイントをダウンロードしてローカルパスで読み込むか、独自に学習してください（以下の「学習」を参照）。

<code-tabs name="predict" />

結果は `boxes` の代わりに `points` ペイロードを持ちます。各行は `x, y, class, confidence` で、`result.points.data`、または `.xy`、`.xyn`、`.cls`、`.conf` accessorから利用できます。抑制するボックスがないため、設定する `iou` しきい値はありません。`predict(..., nms_radius=1)` は、2つの検出結果が両方残るために必要なグリッドcell間の距離を制御します。また、loaderが認識できるように、ファイル名にはFOMOの `-point` タスク接尾辞が必要です。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

`s`、`m`、`l` の3サイズがあり、それぞれ単一の1x1分類ヘッドの前に、段階的に幅を広げたMobileNetV2形式のバックボーンを、対応して大きくなる固定入力解像度で使います。このファミリーのベンチマーク表はここにはありません。以下の表にあるチェックポイントのファイルサイズが、現在公開されているサイズごとの差をもっとも明確に示します。

## 学習

<code-tabs name="train" />

`imgsz` は自由に選べません。読み込んだチェックポイントのネイティブ解像度がデフォルトとなり、異なる値を渡すと、必要なサイズを示す `ValueError` が送出されます。`s` は96、`m` は192、`l` は224です。CLIでは `imgsz` のデフォルトが640なので、`libreyolo train` コマンドではチェックポイントに合わせて明示的に設定する必要があります。

ほかの設定を変更しなければ、trainerはAdam、`lr0=3e-4`、weight decayなしで、バッチ32の40エポックを実行します。一般的な場面ではほぼすべてのグリッドcellがbackgroundなので、cellごとのcross-entropy lossでforegroundクラスをbackgroundの100倍に重み付けします。EMAと混合精度はどちらもデフォルトで無効です。LibreYOLOのほかの場所で使われる幾何学・色のデータ拡張はどれも適用されません。mosaic、mixup、HSV jitter、flip、rotation、translation、shearはすべてゼロです。

これは公開済みLibreFOMOチェックポイントをCOCOでスクラッチ学習した経路です。

データセットとloggerについては[学習](/docs/train)を参照してください。

## 検証

`val()` はこのファミリー向けに構築されたグリッドレベルの検証機能へ処理を振り分けます。ほかのpointタスクと共有する点照合の `metrics/precision`、`metrics/recall`、`metrics/mAP@` キーに加え、信頼度しきい値と `nms_radius` 値を走査し、最良のF1の組み合わせを `metrics/grid_F1`、`metrics/grid_precision`、`metrics/grid_recall`、`metrics/grid_mean_distance` に公開します。その結果を生んだしきい値とradiusも `decode/threshold` および `decode/nms_radius` に公開します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポート済み成果物はファイル接尾辞に基づいて `LibreYOLO()` から再度読み込めます。そのため、`.onnx` または `.engine` ファイルはチェックポイントと同様に動作し、同じ `Results` を返します。LibreYOLOをインストールしていない単独のランタイムでグラフを実行することもできますが、その場合は前処理と後処理を自身で記述する必要があります。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。いずれも自動ダウンロードされません。リンク先のHugging Faceページから目的のファイルを取得し、そのローカルパスを `LibreYOLO()` へ渡してください。

<checkpoint-table />

## ライセンス

<provenance-box>

リンクできるFOMOのアップストリームコードリポジトリはありません。Edge Impulseはブログ投稿と製品ドキュメントで手法を説明していますが、FOMOの学習・推論コードは公開していません。ここでのアーキテクチャと学習は、その公開済み説明をLibreYOLOが独自に実装したものであり、公開済みLibreFOMOチェックポイントはCOCOでスクラッチ学習されています。そのため、コードとこれらの重みはどちらもLibreYOLO独自のMITです。FOMOという名前と、それが表す手法は引き続きEdge Impulseのものです。

</provenance-box>

