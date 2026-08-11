---
title: SAM
families:
  - sam
seo_title: SAM（Segment Anything）：LibreYOLOでマスクを推論
description: >-
  LibreYOLOのSAMで点とボックスによるプロンプト可能なセグメンテーションを行います。Apache-2.0のbase、large、hugeチェックポイントのインストールと推論を説明します。
lead: >-
  SAM（Segment
  Anything）は点またはボックスのクリックを物体マスクに変換します。プロンプト可能なモデルには異なる呼び出し形式が必要なため、LibreYOLOはLibreYOLO()検出器ファクトリとは別の専用LibreSAMファクトリから読み込みます。
keywords:
  - SAM 使い方
  - Segment Anything
  - プロンプト セグメンテーション
  - インタラクティブ セグメンテーション
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

        # "base"は初回使用時にfacebook/sam-vit-baseを自動ダウンロード
        # ほかのサイズ: "large", "huge"（"b"/"l"/"h"も使用可能）
        model = LibreSAM("base")

        # 点プロンプト。[x, y]はピクセル座標、ラベル1は前景
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # マスクごとのポリゴン
        print(result.boxes.xyxy)    # マスクから導出した外接ボックス

        # 点の代わりにボックスプロンプト
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        # プロンプトなしでは画像全体をセグメンテーション（簡略化された自動
        # マスク生成器であり、網羅的な参照実装ではない）
        result = model.predict(SAMPLE_IMAGE)
    - label: 1回エンコードして複数回プロンプト
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # 画像エンコーダーは計算負荷が高い部分。set_image()で1回実行し
        # 以後のpredict()呼び出しはキャッシュ済み埋め込みベクトルを再利用
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: f8904d241ef8a929
---

## インストール

SAMには`sam`追加パッケージが必要で、`transformers`と`timm`がインストールされます。

```bash
pip install "libreyolo[sam]"
```

## 推論

`LibreSAM(...)`は`LibreYOLO(...)`とは別のエントリポイントです。空間プロンプトなしの
順伝播には意味がないため、検出器ではなくプロンプト可能なセグメンターを返します。この
ファミリーに`libreyolo predict` CLIコマンドはありません。Python APIを使ってください。

<code-tabs name="predict" />

点プロンプトは1個の物体に`[x, y]`、複数の物体に`[[x, y], ...]`、またはnumpy配列を
受け付けます。`labels`は各点を`1`（前景）または`0`（背景）として示し、デフォルトでは
すべて前景です。ボックスプロンプトは`[x1, y1, x2, y2]`またはボックスのリストを受け取り、
ボックスごとに1個のマスクを生成します。両方のプロンプトを省略すると、密なグリッドを
プロンプトとして画像全体をセグメンテーションし、信頼度が高く重複しないマスクを維持します。
この「すべてをセグメンテーション」モードは参照用の自動マスク生成器より簡略化されており、
混雑した場面ではセグメンテーションが不足する可能性があります。そのため、実際の点または
ボックスプロンプトを使う経路が正確です。`conf`は物体検出の信頼度ではなく、予測マスクの品質
（IoU）で絞り込みます。すべての候補を維持するには`0.0`を渡します。`multimask=True`は、
最良の1個だけでなく、プロンプトごとにSAMの全体と部分の曖昧性を表す3個のマスクをすべて返します。
`device=`はモデルを移動し、`set_image()`セッションが有効ならキャッシュ済み埋め込みベクトルも
移動します。プロンプト可能なマスクには固定クラスセットがないため、すべてのマスクのクラスIDは
`0`、名前は`"object"`です。 `train()`、`val()`、`export()`、`track()`はすべて、このファミリーでは
`NotImplementedError`を発生させます。LibreYOLOのSAMは推論専用で、動画追跡は対象外です。
入力ソースの種類については[推論](/docs/predict)を参照してください。

## バリアント

ViT画像エンコーダーのサイズはbase、large、hugeの3種類で、すべて固定1024 px入力です。
このファミリーでは精度またはレイテンシのベンチマークがまだ公開されていないため、サイズの選択は
エンコーダーの重さとマスク品質の直接的なトレードオフになります。baseはエンコードが最も速く、
hugeが最も重いモデルです。

## ライセンス

<provenance-box>

LibreYOLOはSAM-1の重みの独自コピーをホストしません。`LibreSAM("base")`、`"large"`、
`"huge"`は、Meta自身のHugging Faceリポジトリ`facebook/sam-vit-base`、
`facebook/sam-vit-large`、`facebook/sam-vit-huge`から直接ダウンロードします。それぞれが
LibreYOLOとは独立してApache-2.0と明記されています。

</provenance-box>

## 引用

<citation-block />

