---
title: ZipDepth
families:
  - zipdepth
seo_title: ZipDepth：LibreYOLOの軽量単眼深度推定
description: >-
  LibreYOLOでZipDepthを使い、軽量な単眼深度推定を行います。MITライセンスの2つのチェックポイントについて、インストール、推論、検証、エクスポートの方法を説明します。
lead: >-
  ZipDepthは、Depth Anything V2
  Largeから蒸留された、再パラメータ化可能なコンパクトCNNであり、密な相対逆深度マップを予測します。LibreYOLOは深度タスクでZipDepthに対応し、推論とゼロショット検証を実行できますが、学習機能はありません。
keywords:
  - ZipDepth
  - 単眼深度推定
  - エッジ 深度モデル
  - 相対深度
  - depth map
  - 再パラメータ化 CNN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreZipDepthb-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: NPU・エッジ向けチェックポイント
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 同じエンコーダーと、gather/unfoldに対応しないコンパイラー向けの
        # unfold不要なアップサンプリングヘッド。出力はbチェックポイントと視覚的に同等
        model = LibreYOLO("LibreZipDepthbnpu-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreZipDepthb-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        model.export(format="onnx")
        model.export(format="ncnn")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreZipDepthb-depth.pt format=onnx
        libreyolo export model=LibreZipDepthbnpu-depth.pt format=ncnn
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で経路を選ぶため、エクスポート済み成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreZipDepthb-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: 891eaa1a42795a4c
---

## インストール

ZipDepthに任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

`result.depth_map` には密な相対逆深度マップが格納されます。値が大きいほどカメラに近いことを示し、値にメートル法の単位や画像間で共通のスケールはありません。`save=True` を指定すると、そのマップをカラーマップで可視化した画像がディスクに書き込まれます。`Results.plot()` はサーフェス法線とエッジだけを対象として定義されているため、このファミリーには対応しません。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

2つのチェックポイントがあり、どちらもエンコーダーの容量は同じで、学習済みアップサンプリングヘッドだけが異なります。`b` は凸アップサンプリングを使い、GPUまたはCPUで動作します。`bnpu` は、gather・unfoldに対応しないNPUおよびエッジ向けコンパイラー用に、unfold不要のデコーダーへ置き換えたものです。その出力は `b` と視覚的に同等であると記載されています。エクスポート先が制約のあるランタイムの場合は `bnpu` を、それ以外では `b` を選んでください。

どちらのチェックポイントもDepth Anything V2 Largeの擬似ラベルから蒸留されています。そのため、このファミリーは、より大規模なDepth Anything V2エンコーダーと並ぶ、LibreYOLO深度タスクのコンパクトなエッジ向け階層です。

このファミリーには学習機能がありません。`LibreZipDepth.train()` は無条件に `NotImplementedError` を送出します。アップストリームの手順は大規模な画像セット全体で擬似ラベルを蒸留するものであり、LibreYOLOの学習実行として再現できないためです。[fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth)でアップストリーム学習を行い、`weights/convert_zipdepth_weights.py` で結果を変換してください。

## 検証

`val()` は共通の深度検証機能を実行します。画像ごとの最小二乗法によるスケールとシフトを使って各推論結果を正解データに整合させ、標準的なゼロショット相対深度指標であるAbsRel、RMSE、3つのdeltaしきい値を報告します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポートは固定解像度の密出力仕様に従います。元画像はエクスポート先キャンバスに合うよう引き伸ばしてリサイズされ、返された深度マップはその後で元のキャンバスに合うようリサイズされます。エクスポート済み成果物はファイル接尾辞に基づいて `LibreYOLO()` から再度読み込めます。そのため、`.onnx` または `.ncnn` ファイルはチェックポイントと同様に動作し、ボックスの代わりに `depth_map` を持つ同じ `Results` を返します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

