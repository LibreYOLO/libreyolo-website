---
title: Depth Anything 3
families:
  - depth_anything3
seo_title: Depth Anything 3：LibreYOLOで単眼深度を推論
description: >-
  LibreYOLOでDepth Anything
  3を使い、単眼深度推定を行います。Apache-2.0ライセンスのDA3MONO-LARGEチェックポイントについて、インストール、推論、検証、エクスポートの方法を説明します。
lead: >-
  Depth Anything 3は、アーキテクチャを特殊化せず、1つ以上の視点から深度とカメラ形状を予測するよう学習された標準的なDINOv2
  Transformerです。LibreYOLOはDA3MONO-LARGEチェックポイントを深度タスクへ移植し、推論とゼロショット検証に対応しますが、学習機能はありません。
keywords:
  - Depth Anything 3
  - DA3
  - 単眼深度推定
  - DINOv2
  - 相対深度
  - depth map
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnything3l-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 深度マップを読み取る
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map    # DepthMap: 密な(H, W)、値が大きいほど近い
        raw = depth.data                # テンソル、メートル法の単位や画像間スケールなし
        normalized = depth.normalized() # 可視化用に[0, 1]へ再スケーリング
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnything3l-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnything3l-depth.pt format=onnx

        libreyolo export model=LibreDepthAnything3l-depth.pt format=tensorrt
        half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で経路を選ぶため、エクスポート済み成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreDepthAnything3l-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: 0ac96180165c4891
---

## インストール

Depth Anything 3に任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

`result.depth_map` には密な相対逆深度マップが格納されます。値が大きいほどカメラに近いことを示し、値にメートル法の単位や画像間で共通のスケールはありません。アップストリームのチェックポイントは正の相対深度を出力します。LibreYOLOのネットワークラッパーはこれを反転し、公式の空領域処理を再現するため、出力はLibreYOLOの共通深度仕様に従います。`save=True` を指定すると、そのマップをカラーマップで可視化した画像がディスクに書き込まれます。`Results.plot()` はサーフェス法線とエッジだけを対象として定義されているため、このファミリーには対応しません。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

固定入力解像度の `l` という1サイズがあります。アップストリームのDA3では、SmallおよびBaseの任意視点チェックポイント、メートル深度チェックポイント、NestedおよびGiantチェックポイントも公開されていますが、LibreYOLOはいずれも提供しません。メートル深度にはLibreYOLOの相対逆深度タスクとは異なる公開仕様が必要で、任意視点チェックポイントとNestedチェックポイントにはLibreYOLOが提供していない複数画像カメラAPIが必要です。LargeおよびGiantの任意視点チェックポイントはCC-BY-NC-4.0でもあり、LibreYOLOのどのダウンロード経路からも参照されません。

このファミリーには学習機能がありません。`LibreDepthAnything3.train()` は無条件に `NotImplementedError` を送出します。アップストリームで学習し、互換性のあるDA3MONO-LARGEチェックポイントを `weights/convert_depth_anything3_weights.py` で変換してください。

## 検証

`val()` は共通の深度検証機能を実行します。画像ごとの最小二乗法によるスケールとシフトを使って各推論結果を正解データに整合させ、標準的なゼロショット相対深度指標であるAbsRel、RMSE、3つのdeltaしきい値を報告します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

このファミリーのエクスポートは、ONNX、TorchScript、ExecuTorch、TensorRT、OpenVINOの5形式に制限されています。それ以外の形式を要求すると、未検証の変換を試みずに `NotImplementedError` を送出します。エクスポート済み成果物はファイル接尾辞に基づいて `LibreYOLO()` から再度読み込めます。そのため、`.onnx` または `.engine` ファイルはチェックポイントと同様に動作し、ボックスの代わりに `depth_map` を持つ同じ `Results` を返します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

