---
title: SAM 3
families:
  - sam3
seo_title: SAM 3：LibreYOLOでプロンプト可能なセグメンテーションとconcept segmentation
description: >-
  LibreYOLOのSAM 3で点、ボックス、テキストconceptによるセグメンテーションを行います。MetaのSAM
  Licenseでアクセス制限されたlargeチェックポイントのインストールと推論を説明します。
lead: >-
  SAM 3は通常の点とボックスにテキストconceptプロンプトを加えてSAMを拡張します。そのため「yellow school
  bus」のようなフレーズから、一致するすべてのインスタンスを返します。LibreYOLOはLibreYOLO()検出器ファクトリとは別の専用LibreSAMファクトリを通じて、その画像経路に対応します。
keywords:
  - SAM 3 使い方
  - Segment Anything
  - プロンプト セグメンテーション
  - concept segmentation
  - テキストプロンプト
  - 点プロンプト
  - ボックスプロンプト
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: 点とボックスのプロンプト
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # "sam3"が唯一のサイズ（"large"）。別名: "sam3", "sam-3", "sam3-large"
        model = LibreSAM("sam3")

        # 点プロンプト。[x, y]はピクセル座標、ラベル1は前景
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # マスクごとのポリゴン
        print(result.boxes.xyxy)    # マスクから導出した外接ボックス

        # 点の代わりにボックスプロンプト
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: テキスト（concept）プロンプト
      language: python
      code: |
        from libreyolo import LibreSAM3, SAMPLE_IMAGE

        model = LibreSAM3("large")

        # 1個の物体だけでなく、フレーズに一致するすべてのインスタンスを検出
        # text=はpoints, bboxes, labels, masksと同時使用不可
        result = model.predict(SAMPLE_IMAGE, text="a person")
        print(result.names)         # {0: "a person"}
        print(result.boxes.conf)    # インスタンスごとのPCS検出スコア
    - label: 1回エンコードして複数回プロンプト
      language: python
      code: |
        from libreyolo import LibreSAM3, SAMPLE_IMAGE

        model = LibreSAM3("large")

        # 画像エンコーダーは計算負荷が高い部分。set_image()で1回実行し
        # 以後のpredict()呼び出しはキャッシュ済み埋め込みベクトルを再利用。text=の
        # 呼び出しでは内部で再エンコード。trackerとconcept-segmentation encoderは
        # キャッシュを共有しないため
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: c4fb6d5a622f99ff
---

## インストール

SAM 3には`sam`追加パッケージが必要で、`transformers`と`timm`がインストールされます。

```bash
pip install "libreyolo[sam]"
```

重みにはアクセス制限があります。[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)を
開いてMetaのSAM Licenseに同意し、初回ダウンロード前に`hf auth login`を実行してください
（または`HF_TOKEN`を設定します）。LibreYOLOはこのファミリーを初めてダウンロードするときに
ライセンス通知を記録します。

## 推論

`LibreSAM(...)`（またはファミリー固有の`LibreSAM3(...)`）は`LibreYOLO(...)`とは別の
エントリポイントです。プロンプトなしの順伝播には意味がないため、検出器ではなくプロンプト可能な
セグメンターを返します。このファミリーに`libreyolo predict` CLIコマンドはありません。
Python APIを使ってください。対応するのは画像推論だけで、SAM 3の動画モデルは対象外です。

<code-tabs name="predict" />

点とボックスの経路はほかのSAMファミリーと同じです。点プロンプトは1個の物体に`[x, y]`、
複数の物体に`[[x, y], ...]`を受け付けます。`labels`は各点を`1`（前景）または`0`（背景）として示し、
ボックスプロンプトは`[x1, y1, x2, y2]`またはボックスのリストを受け取ります。この経路の`conf`は
物体検出の信頼度ではなく、予測マスクの品質（IoU）で絞り込みます。

`text=`経路はSAM 3で追加された機能です。concept文字列からPromptable Concept Segmentationを
通じて画像内の一致するすべてのインスタンスを返し、点、ボックス、ラベル、マスクとは同時に使えません。
この経路の`conf`はマスクIoUではなくPCS検出スコアです。デフォルトのままにするとモデル固有の0.3の
しきい値が適用され、`conf=0.0`ではすべての候補を維持します。プロンプト可能なマスクにはそれ以外の
固定クラスセットがないため、返される`names`はクラスID`0`を要求したconcept文字列へ対応付けます。
`device=`はモデルを移動し、`set_image()`セッションが有効ならキャッシュ済み埋め込みベクトルも
移動します。`train()`、`val()`、`export()`、`track()`はすべて、このファミリーでは
`NotImplementedError`を発生させます。LibreYOLOのSAM 3は推論専用で、動画追跡は対象外です。
入力ソースの種類については[推論](/docs/predict)を参照してください。

## バリアント

サイズはlargeの1種類で、固定1008 px入力です。SAM 3.1には対応しません。その実装はこのMIT
リポジトリへ同梱できない独自ライセンスを採用しており、LibreYOLOが依存するTransformersの
バージョンもチェックポイント形式をまだ読み込めないためです。

## ライセンス

<provenance-box>

LibreYOLOはSAM 3の重みの独自コピーをホストせず、再配布もしません。`LibreSAM("sam3")`は
Metaのアクセス制限付きHugging Faceリポジトリ`facebook/sam3`から直接ダウンロードします。
初回ダウンロード前にMetaのSAM Licenseへの同意と認証が必要です。

</provenance-box>

## 引用

<citation-block />

