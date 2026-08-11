---
title: SAM 2
families:
  - sam2
seo_title: SAM 2：LibreYOLOでプロンプト可能な画像セグメンテーション
description: >-
  LibreYOLOのSAM
  2で点とボックスによるプロンプト可能なセグメンテーションを行います。Apache-2.0のtiny、small、base-plus、largeチェックポイントのインストールと推論を説明します。
lead: >-
  SAM
  2は動画向けに構築されたstreaming-memoryアーキテクチャでSAMを拡張し、点またはボックスのクリックを物体マスクに変換します。LibreYOLOはLibreYOLO()検出器ファクトリとは別の専用LibreSAMファクトリを通じて、その画像セグメンテーション経路に対応します。
keywords:
  - SAM 2 使い方
  - Segment Anything
  - プロンプト セグメンテーション
  - インタラクティブ セグメンテーション
  - 点プロンプト
  - ボックスプロンプト
  - Meta AI
  - Hiera
last_verified: 1.5.0
snippets:
  predict:
    - label: 点とボックスのプロンプト
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # サイズの別名: "sam2-tiny", "sam2-small", "sam2-base-plus",
        # "sam2-large"（短縮形"sam2-t"/"sam2-s"/"sam2-bp"/"sam2-l"も使用可能）
        model = LibreSAM("sam2-large")

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
        from libreyolo import LibreSAM2, SAMPLE_IMAGE

        # ファミリー固有クラスには"sam2-"接頭辞なしでサイズを指定
        model = LibreSAM2("large")

        # 画像エンコーダーは計算負荷が高い部分。set_image()で1回実行し
        # 以後のpredict()呼び出しはキャッシュ済み埋め込みベクトルを再利用
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: 2a3090d7ecd533b0
---

## インストール

SAM 2には`sam`追加パッケージが必要で、`transformers`と`timm`がインストールされます。

```bash
pip install "libreyolo[sam]"
```

## 推論

`LibreSAM(...)`（またはファミリー固有の`LibreSAM2(...)`）は`LibreYOLO(...)`とは別の
エントリポイントです。空間プロンプトなしの順伝播には意味がないため、検出器ではなく
プロンプト可能なセグメンターを返します。このファミリーに`libreyolo predict` CLIコマンドは
ありません。Python APIを使ってください。対応するのは画像セグメンテーションだけで、SAM 2の
video-memory追跡はここでは対象外です。

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
`0`、名前は`"object"`です。`train()`、`val()`、`export()`、`track()`はすべて、このファミリーでは
`NotImplementedError`を発生させます。ここでLibreYOLOが対応するのは画像推論です。
入力ソースの種類については[推論](/docs/predict)を参照してください。

## バリアント

Hieraバックボーンのサイズはtiny、small、base-plus、largeの4種類で、すべて同じ入力解像度です。
このファミリーでは精度またはレイテンシのベンチマークがまだ公開されていないため、サイズの選択は
エンコーダーの重さとマスク品質の直接的なトレードオフになります。tinyはエンコードが最も速く、
largeが最も重いモデルです。

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

