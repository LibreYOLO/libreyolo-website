---
title: DexiNed
families:
  - dexined
seo_title: DexiNed：自分のチェックポイントでエッジ検出
description: LibreYOLOのDexiNedで密なエッジ確率を予測します。ライセンスを保有するチェックポイントを変換して、推論、検証、エクスポートを行います。
lead: >-
  DexiNed（Dense Extreme Inception
  Network）は、1枚のRGB画像から密なエッジ確率マップを予測する畳み込みネットワークです。LibreYOLOはそのアーキテクチャをエッジ検出専用でラップし、チェックポイントはライブラリに付属しません。
keywords:
  - DexiNed 使い方
  - Dense Extreme Inception Network
  - エッジ検出
  - BIPED
  - dense prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)        # (H, W) float32、範囲[0, 1]
        print(edges.binary(0.5).sum())  # しきい値処理後のエッジピクセル数
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=weights/LibreDexiNedb-edge.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])   # データセット全体で最適なF-measure
        print(metrics["metrics/OIS"])   # 画像ごとに最適なF-measure
    - label: CLI
      language: bash
      code: >
        libreyolo val model=weights/LibreDexiNedb-edge.pt data=my-dataset.yaml
        imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=weights/LibreDexiNedb-edge.pt format=onnx
        imgsz=352

        libreyolo export model=weights/LibreDexiNedb-edge.pt format=tensorrt
        imgsz=352 half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: 342597fde3c4ba65
---

## インストール

DexiNedに任意の追加パッケージは必要ありません。インポートするものはすべて基本
インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

LibreYOLOはDexiNedチェックポイントを提供しません。公式に公開された重みはBIPEDで学習されており、
公開済みのデータセット規約が利用を非商用目的に制限するため、LibreYOLOはミラーしません。
ライセンスを保有するチェックポイントを`weights/convert_dexined_weights.py`で変換してください。この
scriptは、LibreYOLOが直接読み込めるファイルを書き出す前に、テンソルkeyとランタイムアーキテクチャを照合します。

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

結果は`result.edges`に入ります。`[0, 1]`範囲の`(H, W)`形状のfloat32配列で、
`.binary(threshold)`はbooleanのエッジマスクを返します。ボックスはないため、`conf`、`iou`、
`max_det`に効果はありません。入力ソース、ストリーミング、結果の処理については
[推論](/docs/predict)を参照してください。

## バリアント

DexiNedはLibreYOLOで1サイズを提供します。LibreYOLOのベンチマークharnessはこのファミリーを
測定していないため、比較できる公開数値はありません。

## 検証

`val()`は、対応するエッジデータセットに対してBSDS形式のODSおよびOIS F-measureを報告します。
画像と同じstemのエッジマップを並べ、任意の有効性マスクによりpaddingされたピクセルを除外できます。
`imgsz`はネットワークのdownsample strideで割り切れる必要があり、そうでない場合はLibreYOLOが
明確なエラーを発生させます。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エッジのエクスポートは固定解像度、バッチ1のランタイム契約を使います。`dynamic`と1以外の
`batch`は拒否され、エクスポートしたグラフは1個のfused probability mapを出力します。
エクスポートした成果物はファイル接尾辞に基づいて`LibreYOLO()`から再読み込みされます。そのため、
`.onnx`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。

<code-tabs name="export" />

## ライセンス

<provenance-box>

LibreYOLOはDexiNedチェックポイントを公開しません。LibreYOLO組織の下には何もミラーされません。
代わりに、ライセンスを保有するチェックポイントを`weights/convert_dexined_weights.py`で変換してください。

</provenance-box>

## 引用

<citation-block />

