---
title: BiRefNet
families:
  - birefnet
seo_title: BiRefNet：LibreYOLOで背景除去とマッティング
description: >-
  LibreYOLOのBiRefNetで背景除去とdichotomous image
  segmentationを行います。generalチェックポイントのインストール、推論、検証、エクスポートを説明します。
lead: >-
  被写体を背景から分離するソフトアルファマットを予測するbilateral-reference
  networkです。LibreYOLOはBiRefNetのマットタスク向けに推論と検証を提供します。
keywords:
  - BiRefNet 背景除去
  - 画像 背景 透過
  - dichotomous image segmentation
  - アルファマット
  - image matting
  - 画像 切り抜き
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreBiRefNetl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 切り抜き
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8。元画像のRGBとアルファチャンネルとしてのマット
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # images/と自動検出されるマットディレクトリ
        # (mattes/, matte/, gt/, masks/, mask/またはalpha/)を含むディレクトリも
        # データセットYAMLの代わりに使用可能
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreBiRefNetl-matte.pt format=onnx
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreBiRefNetl-matte.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: 1af1bd7f4f905081
---

## インストール

BiRefNetに任意の追加パッケージは必要ありません。インポートするものはすべて基本
インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

マット結果にボックスはありません。`result.matte`は`[0, 1]`範囲の密な`(H, W)`形状の
float32配列で、1は完全な前景、0は完全な背景です。ソフトマットはバイナリマスクと異なり、
髪や毛皮などのアンチエイリアスされたエッジの細部を保ちます。`result.cutout()`は元画像と
そのアルファチャンネルを合成してRGBA配列を作り、`result.save(path)`（または推論呼び出しの
`save=True`）は背景が透明なPNGへ直接書き出します。モデルはネイティブの固定1024×1024
キャンバスで動作し、ほかの解像度には対応しません。Swinバックボーンの相対位置テーブルが
この解像度に結び付いており、不一致があるとエラーを発生させずに不適切な補間をするためです。
入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

公開チェックポイントは1種類です。`l`はSwin-L層のBiRefNet-generalモデルで、アップストリームの
品質重視デフォルトです。このファミリーのコードはSwin-T lite層の`t`にも対応しますが、
LibreYOLO変換版はまだ公開されていません。

## 検証

`val()`は対応する画像とマットのフォルダーに対して2つの指標を報告します。どちらも`[0, 1]`
範囲で、解像度に依存しません。MAEは正解アルファに対する平均絶対誤差（低いほど良い）です。
S-measure（Fan et al.、ICCV 2017）は、ピクセルMAEだけでは捉えられない被写体の形状と穴の保持を
評価する構造的類似度（高いほど良い）です。検証はモデル自身の`predict`を実行するため、
このファミリー固有の正確な前処理を使います。

<code-tabs name="val" />

検証は推論専用です。ファインチューニングは提供済み機能ではなく、今後の対応として文書化されています。
将来のtrainerが継承する正確な解像度制約については「推論」を参照してください。

## エクスポート

<export-matrix />

エクスポートした成果物はファイル接尾辞に基づいて`LibreYOLO()`から再読み込みされます。そのため、
`.onnx`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。TorchScriptは
検証済みの経路です。ONNX変換は実行できますが、同じ同等性基準をまだ満たしていません。
[エクスポート](/docs/export)には、各形式が受け付ける引数と、一部の形式で必要になる追加パッケージの一覧があります。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

