---
title: Depth Anything V2
families:
  - depth_anything
seo_title: Depth Anything V2：単眼深度を推論、検証
description: >-
  LibreYOLOのDepth Anything
  V2で単眼深度推定を行います。インストール、推論、検証に対応します。SmallはApache-2.0、BaseとLargeはCC-BY-NC-4.0です。
lead: >-
  Depth Anything
  V2はDINOv2エンコーダーとDPTデコーダーを組み合わせ、1枚の画像から密な相対逆深度マップを予測します。LibreYOLOは深度タスク向けに、学習経路なしで推論とゼロショット検証に対応します。
keywords:
  - Depth Anything V2
  - 単眼 深度推定
  - DPT
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

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnythingV2s-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 深度マップを読み取る
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map    # DepthMapは密な(H, W)で大きいほど近い
        raw = depth.data                # テンソルでメートル単位や画像間スケールはない
        normalized = depth.normalized() # 可視化用に[0, 1]へ再スケーリング
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnythingV2s-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=onnx

        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=tensorrt
        half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で振り分けるためエクスポート済み成果物も
        # チェックポイントと同様に読み込まれて同じResultsオブジェクトを返す
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e1043aba1b70b65c
---

## インストール

Depth Anything V2にオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

`result.depth_map` は密な相対逆深度マップを保持します。大きい値ほどカメラに近いことを示し、値にはメートル単位も画像をまたぐ共通スケールもありません。`save=True` はそのマップをカラーマップで可視化してディスクへ書き出します。`Results.plot()` は表面法線とエッジだけに定義されているため、このファミリーには対応しません。入力解像度は、DPTヘッドの基になるDINOv2パッチグリッドの14で割り切れる必要があります。LibreYOLOは実行前に確認し、割り切れなければエラーになります。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

エンコーダーサイズはViT-S/B/L/Gに対応するs/b/l/gの4つです。下のチェックポイント表に記載されているのはs、b、lだけで、Giantチェックポイントは公開されていません。4つすべてが同じ入力解像度を共有するため、サイズの選択によって変わるのは画像サイズではなくエンコーダー容量です。ライセンスも選択要因です。SmallチェックポイントはApache-2.0で、BaseとLargeはCC-BY-NC-4.0です。下の「ライセンス」を参照してください。

このファミリーは学習もファインチューニングも提供しません。`LibreDepthAnythingV2.train()` は無条件に `NotImplementedError` を発生させます。代わりに、`weights/convert_depth_anything_v2_weights.py` で互換性のあるアップストリームチェックポイントを変換してください。

## 検証

`val()` は共有の深度バリデーターを実行します。各推論を画像ごとの最小二乗スケールとシフトで正解データへ位置合わせし、標準のゼロショット相対深度指標であるAbsRel、RMSE、3つのdeltaしきい値を報告します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポート済み成果物はファイル接尾辞によって `LibreYOLO()` から再度読み込まれるため、`.onnx` または `.engine` ファイルはチェックポイントのように動作し、ボックスの代わりに `depth_map` を持つ同じ `Results` を返します。[エクスポート](/docs/export)では、すべての形式が受け付ける引数を説明しています。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
