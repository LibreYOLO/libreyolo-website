---
title: Real-ESRGAN
families:
  - realesrgan
seo_title: Real-ESRGAN：LibreYOLOの画像超解像
description: >-
  LibreYOLOでReal-ESRGANを使い、実用的な4倍・2倍画像超解像と高速な4倍処理を行います。インストール、推論、検証、エクスポートの方法を説明します。
lead: >-
  バイキュービック縮小だけでなく、合成した劣化を使って学習された実用的なブラインド超解像アップスケーラーです。LibreYOLOは4倍、2倍、高速4倍チェックポイントの推論と検証に対応します。
keywords:
  - Real-ESRGAN
  - RRDBNet
  - SRVGGNetCompact
  - 画像 超解像
  - 画像復元
  - ブラインド超解像
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRealESRGANx4-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 大きな画像向けのタイル分割
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # tileは順伝播を重なり合うタイルに分割し、継ぎ目を再び合成する
        # tile_padは各タイルを切り戻す前に周囲へ加えるハロー領域
        # どちらもPython専用のキーワード引数であり
        # CLIフラグではない
        result = model("large-photo.jpg", tile=512, tile_pad=10, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: >
        libreyolo val model=LibreRealESRGANx4-restore.pt
        data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # imgszを省略すると作業解像度ではなく小さな内部パッチサイズが
        # デフォルトになるため、実際にデプロイ先からモデルへ渡すサイズを指定
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRealESRGANx4-restore.pt format=onnx
        imgsz=512
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で経路を選ぶため、エクスポート済み成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreRealESRGANx4-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
source_hash: f0efb4f65d38e22d
---

## インストール

Real-ESRGANに任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

復元結果にボックスは含まれません。`result.restored` は、各次元が入力の `Results.restore_scale` 倍のキャンバスを持つ密な `(H, W, 3)` uint8 RGB画像です。`save=True` を指定すると、アノテーション付きプロットではなく、その画像を直接書き込みます。入力はRGBに変換され、alphaチャンネルは破棄されます。メモリに収まらない大きさのソースは `tile` と `tile_pad` で分割でき、出力時にタイルの継ぎ目が再び合成されます。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

拡大率にちなんで名付けられた3つのチェックポイントがあります。`x4` は、23個のResidual-in-Residual Dense Blockを持つRRDBNet（`RealESRGAN_x4plus`）で、4倍処理における品質重視のデフォルトです。`x2` は2倍処理用の同じRRDBNetアーキテクチャです。`x4t` はSRVGGNetCompact（`realesr-general-x4v3`）で、動画や低レイテンシでの4倍処理向けに構築された、より小さく高速なジェネレーターです。アップストリームの汎用モデルには、推論時に合成する一対のノイズ除去強度ネットワークも含まれます。この移植では、その強度調整機能を提供せず、基本の `x4t` ジェネレーターを実行します。

## 検証

`val()` は復元出力ときれいな目標画像との間でPSNRとSSIMを測定します。どちらも境界の切り取りやリサイズを行わず、元のキャンバス上のRGBで計算されます。SSIMはsigma 1.5の11x11 Gaussianウィンドウを使い、3つの色チャンネル全体で平均されます。

<code-tabs name="val" />

データセット引数には、劣化した入力画像のディレクトリと同じ解像度のきれいな目標画像のディレクトリを対応付けるYAMLを指定します。正確なキーについては[データセット形式](/docs/reference/dataset-formats)を参照してください。

## エクスポート

<export-matrix />

エクスポート済み成果物はファイル接尾辞に基づいて `LibreYOLO()` から再度読み込めます。そのため、`.onnx` または `.engine` ファイルはチェックポイントと同様に動作し、同じ `Results` を返します。[エクスポート](/docs/export)には、すべての形式で受け付ける引数と、一部の形式で追加される引数が記載されています。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

