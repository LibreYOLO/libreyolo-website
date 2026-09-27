---
title: PicoSAM3
families:
  - picosam3
seo_title: PicoSAM3：LibreYOLOでのボックスプロンプト指定エッジセグメンテーション
description: >-
  LibreYOLOのPicoSAM3で、エッジセンサー上のボックスプロンプト指定領域セグメンテーションを行います。Apache-2.0のpicoチェックポイントをインストールし、推論、エクスポートします。
lead: >-
  PicoSAM3はSAM 2.1とSAM 3から蒸留された小型CNNで、Sony
  IMX500のようなセンサー上のボックスプロンプト指定ROIセグメンテーション向けに構築されています。LibreYOLOはLibreYOLO()検出器ファクトリーとは別の専用LibreSAMファクトリーを通じて、ボックスプロンプトだけに対応します。
keywords:
  - PicoSAM3
  - Segment Anything
  - エッジ セグメンテーション
  - ROI
  - ボックスプロンプト
  - センサー内 推論
  - IMX500
  - 知識蒸留
last_verified: 1.5.0
snippets:
  predict:
    - label: ボックスプロンプト
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # PicoSAM3のサイズはpicoのみで他の別名は不要
        model = LibreSAM("picosam3")

        # bboxes=は唯一対応するプロンプト [x1, y1, x2, y2]またはボックス一覧で
        # ボックスごとに1マスク 各ボックスを10%拡大して正方形にし画像内に収め
        # CNN実行前に96x96へリサイズ
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
        print(result.masks.xy)      # マスクごとのポリゴン
        print(result.boxes.xyxy)    # マスクから得た外接ボックス
    - label: 1回エンコードして複数回プロンプトを指定
      language: python
      code: |
        from libreyolo import LibrePicoSAM3, SAMPLE_IMAGE

        model = LibrePicoSAM3()

        # set_image()は元画像をキャッシュ PicoSAM3はボックスごとにCNN全体を順伝播
        # するため他のSAMファミリーのようにエンコーダー経路ではなく
        # 画像の読み込みとデコードを省略
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(bboxes=[300, 200, 900, 700])
        b = model.predict(bboxes=[100, 100, 400, 400])
        model.reset_image()
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibrePicoSAM3

        model = LibrePicoSAM3()
        model.export(format="onnx", output_path="LibrePicoSAM3pico.onnx")

        # opsetはデフォルト13でdynamicはデフォルトTrueかつバッチ軸のみ
        # このファミリーが受け付けるエクスポート引数はこの2つだけ
    - label: エクスポートしたファイルを使う
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # PicoSAM3は未処理の96x96 ROI CNNをエクスポート

        # roi_image -> mask_logits LibreYOLO側に再利用できる前処理と後処理はなく

        # 検出器チェックポイントと異なりexport()はLibreYOLO()へ戻されない

        session = ort.InferenceSession("LibrePicoSAM3pico.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 96, 96),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 5d60ff14fe61ba29
---

## インストール

PicoSAM3には `sam` 追加パッケージが必要です。推論はネイティブの非 `transformers` CNNで実行されますが、LibreYOLO独自の重みのダウンロードには引き続き `transformers` のHugging Faceツールを使用します。

```bash
pip install "libreyolo[sam]"
```

## 推論

`LibreSAM(...)`（またはファミリー固有の `LibrePicoSAM3(...)`）は `LibreYOLO(...)` とは別のエントリポイントです。プロンプトがなければここでの順伝播に意味がないため、検出器ではなくプロンプト指定可能なセグメンターを返します。このファミリーには `libreyolo predict` CLIコマンドがありません。Python APIを使用してください。

<code-tabs name="predict" />

PicoSAM3は `bboxes=` だけを受け付けます。`points=`、`labels=`、`masks=`、`text=`、`multimask=True` を渡す場合や、ボックスを省略してすべてをセグメンテーションする場合は、明確な `ValueError` が発生します。これらのモードはアップストリームモデルに存在しないためです。`conf` は検出信頼度ではなく、予測されたマスク品質（IoU）でフィルタリングし、`0.0` から `1.0` の範囲である必要があります。すべてのマスクは `"object"` という名前のクラスID `0` を持ちます。`train()`、`val()`、`track()` は `NotImplementedError` を発生させます。点、テキスト、マスク、すべてをセグメンテーションするプロンプトにはLibreSAM2またはLibreSAM3を使用してください。ソースの種類については、[推論](/docs/predict)を参照してください。

## バリアント

サイズはpicoの1つで、固定の96 px ROI入力です。PicoSAM3は画像全体を1回だけエンコードする代わりに、ボックスごとにCNN全体を1回順伝播します。

## エクスポート

<export-matrix />

PicoSAM3はSAM階層でエクスポートできる唯一のファミリーです。未処理の96x96 ROI CNNを `roi_image -> mask_logits` としてONNXへ出力し、NMSやマスクの後処理は組み込みません。その他のSAMファミリーでは、エンコーダーとデコーダーの分割に定義済みのランタイムエクスポート契約がないため、`export()` が `NotImplementedError` を発生させます。エクスポート済みPicoSAM3グラフは `LibreYOLO()` から再度読み込めません。`onnxruntime` などのランタイムで直接実行し、上で示したものと同じ10%パディング付き正方形ROI前処理を適用してください。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box>

PicoSAM3はSAM 2.1とSAM 3を教師モデルとして蒸留されています。LibreYOLOはこのファミリーでどちらの教師のコードや重みも組み込みまたは再配布しません。小型の生徒CNNと変換済みチェックポイントだけを提供します。

</provenance-box>

## 引用

<citation-block />
