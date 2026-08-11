---
title: DexiNed
families:
  - dexined
seo_title: DexiNed：独自のチェックポイントを使うエッジ検出
description: LibreYOLOのDexiNedで密なエッジ確率を予測します。ライセンスを持つチェックポイントを変換し、推論、検証、エクスポートします。
lead: >-
  DexiNed（Dense Extreme Inception
  Network）は、1枚のRGB画像から密なエッジ確率マップを予測する畳み込みネットワークです。LibreYOLOはエッジ検出専用としてそのアーキテクチャをラップします。ライブラリにチェックポイントは付属しません。
keywords:
  - DexiNed
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
        print(edges.array.shape)        # [0, 1]の(H, W) float32
        print(edges.binary(0.5).sum())  # しきい値処理したエッジピクセル数
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

        print(metrics["metrics/ODS"])   # データセット全体の最適F値
        print(metrics["metrics/OIS"])   # 画像ごとの最適F値
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

DexiNedにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

LibreYOLOはDexiNedチェックポイントを提供しません。公式に公開された重みはBIPEDで学習されており、その公開データセットの条件は非商用目的だけに利用を制限しています。そのため、LibreYOLOは重みをミラーしません。利用許諾を持つチェックポイントを `weights/convert_dexined_weights.py` で変換してください。このスクリプトは、LibreYOLOが直接読み込めるファイルを書き出す前に、ランタイムアーキテクチャに対してテンソルキーを確認します。

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges` が結果を保持します。これは `[0, 1]` の範囲を持つ `(H, W)` のfloat32配列で、`.binary(threshold)` はブール値のエッジマスクを返します。ボックスはないため、`conf`、`iou`、`max_det` は効果を持ちません。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

LibreYOLOのDexiNedは1つのサイズを提供します。LibreYOLOのベンチマークハーネスはこのファミリーを測定していないため、比較できる公開値はありません。

## 検証

`val()` は、対応付けられたエッジデータセットに対するBSDS形式のODSおよびOIS F値を報告します。データセットでは、画像の隣に同じステム名のエッジマップを置き、オプションの有効性マスクによってパディング済みピクセルを計測対象外にできます。`imgsz` はネットワークのダウンサンプルstrideで割り切れる必要があり、そうでない場合はLibreYOLOが明確なエラーを発生させます。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エッジのエクスポートは固定解像度、バッチ1のランタイム契約を使用します。`dynamic` と1以外の `batch` は拒否され、エクスポート済みグラフは1つの統合確率マップを出力します。エクスポート済み成果物はファイル接尾辞によって `LibreYOLO()` から再度読み込まれるため、`.onnx` ファイルはチェックポイントのように動作し、同じ `Results` を返します。

<code-tabs name="export" />

## ライセンス

<provenance-box>

LibreYOLOはDexiNedチェックポイントを公開しません。LibreYOLO組織の下には何もミラーされていません。代わりに、利用許諾を持つチェックポイントを `weights/convert_dexined_weights.py` で変換してください。

</provenance-box>

## 引用

<citation-block />
