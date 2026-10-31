---
title: EdgeTAM
families:
  - edgetam
seo_title: EdgeTAM：LibreYOLOのオンデバイス向けプロンプト指定可能セグメンテーション
description: >-
  LibreYOLOのEdgeTAMを使い、オンデバイス速度を重視した点とボックスのプロンプト指定可能セグメンテーションを行います。Apache-2.0のチェックポイントをインストールして推論します。
lead: >-
  EdgeTAMはSAM
  2のオンデバイス向けバリアントで、点とボックスを使う同じプロンプト指定可能ワークフローを維持しながら、モバイル推論速度を重視して構築されています。LibreYOLOは、LibreYOLO()検出器ファクトリーとは別の専用LibreSAMファクトリーを通じて、その画像セグメンテーション経路に対応します。
keywords:
  - EdgeTAM
  - SAM 2
  - プロンプト指定 セグメンテーション
  - インタラクティブ セグメンテーション
  - オンデバイス セグメンテーション
  - 点プロンプト
  - ボックスプロンプト
  - Meta Reality Labs
last_verified: 1.5.0
snippets:
  predict:
    - label: 点とボックスのプロンプト
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # EdgeTAMのサイズはedgeのみ 別名はedgetamとedge-tamとedgetam-edge
        #
        model = LibreSAM("edgetam")

        # 点プロンプト ピクセル座標[x, y]でラベル1は前景
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # マスクごとのポリゴン
        print(result.boxes.xyxy)    # マスクから得た外接ボックス

        # 点の代わりにボックスプロンプト
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        # プロンプトなしでは画像全体をセグメンテーション
        # 簡略版の自動マスク生成器で網羅的な参照実装とは異なる
        result = model.predict(SAMPLE_IMAGE)
    - label: 1回エンコードして複数回プロンプトを指定
      language: python
      code: |
        from libreyolo import LibreEdgeTAM, SAMPLE_IMAGE

        model = LibreEdgeTAM()

        # 画像エンコーダーは高コスト set_image()で1回だけ実行し
        # その後のすべてのpredict()呼び出しでキャッシュ済み埋め込みを再利用
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: e6cce8faad18e73d
---

## インストール

EdgeTAMには `sam` 追加パッケージが必要で、`transformers` と `timm` が導入されます。

```bash
pip install "libreyolo[sam]"
```

## 推論

`LibreSAM(...)`（またはファミリー固有の `LibreEdgeTAM(...)`）は `LibreYOLO(...)` とは別のエントリポイントです。空間プロンプトがなければここでの順伝播に意味がないため、検出器ではなくプロンプト指定可能なセグメンターを返します。このファミリーには `libreyolo predict` CLIコマンドがありません。Python APIを使用してください。対応するのは画像セグメンテーションだけで、EdgeTAMの動画トラッキングはここでは対象外です。

<code-tabs name="predict" />

点プロンプトは、1つの物体に `[x, y]`、複数の物体に `[[x, y], ...]`、またはnumpy配列を受け付けます。`labels` は各点を `1`（前景）または `0`（背景）として示し、デフォルトではすべて前景です。ボックスプロンプトは `[x1, y1, x2, y2]` またはボックス一覧を受け取り、ボックスごとに1つのマスクを生成します。両方のプロンプトを省略すると、密なグリッドでプロンプトを指定し、信頼度が高く重複しないマスクを保持することで画像全体をセグメンテーションします。この「すべてをセグメンテーション」モードは参照用の自動マスク生成器より簡略化されており、混雑した場面ではセグメンテーションが不足する場合があります。そのため、正確な結果には実際の点またはボックスプロンプトを使用します。`conf` は検出信頼度ではなく、予測されたマスク品質（IoU）でフィルタリングします。すべての候補を保持するには `0.0` を渡してください。`multimask=True` は、1つの最良マスクの代わりに、プロンプトごとにSAMの全体と部分の曖昧さを表す3つのマスクをすべて返します。`device=` はモデルを移動し、`set_image()` セッションが有効であれば、キャッシュ済み埋め込みベクトルも移動します。プロンプト指定可能なマスクには固定クラス集合がないため、すべてのマスクは `"object"` という名前のクラスID `0` を持ちます。このファミリーでは `train()`、`val()`、`export()`、`track()` がすべて `NotImplementedError` を発生させます。ここでLibreYOLOが対応するのは画像推論です。ソースの種類については、[推論](/docs/predict)を参照してください。

## バリアント

サイズはedgeの1つで、入力解像度は固定です。そのため、このファミリーを他のSAM階層より優先するかはサイズではなくハードウェアによって決まります。EdgeTAMは制約のあるオンデバイス推論専用として存在します。

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
