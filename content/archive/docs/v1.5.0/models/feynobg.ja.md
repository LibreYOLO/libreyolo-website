---
title: FeyNobg
families:
  - feynobg
seo_title: FeyNobg：LibreYOLOでの背景除去
description: >-
  LibreYOLOのFeyNobgを使い、背景除去とアルファマッティングを行います。Feyn
  Inc.による深層化されたBiRefNetバリアントをインストールし、推論、検証します。
lead: >-
  Feyn
  Inc.がBiRefNetのアーキテクチャを深層化して再学習した背景除去モデルです。LibreYOLOはFeyNobgのマットタスクに対する推論と検証を提供します。
keywords:
  - FeyNobg
  - 背景除去
  - dichotomous image segmentation
  - アルファマット
  - image matting
  - 画像 切り抜き
  - nobg
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFeyNobgl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 切り抜き
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8で元のRGBとアルファチャンネルとしてのマット
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFeyNobgl-matte.pt")

        # images/と自動検出されるマットディレクトリを含むディレクトリも
        # データセットYAMLの代わりに使用可能 マットディレクトリ候補は
        # mattes/ matte/ gt/ masks/ mask/またはalpha/
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
source_hash: 45de3b578d7ebbf2
---

## インストール

FeyNobgにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

チェックポイントは他のファミリーと同様に、初回使用時にHugging Face上のLibreYOLO組織からダウンロードされ、ローカルにキャッシュされます。ただし、このページのチェックポイント表にはまだ掲載されていません。

<code-tabs name="predict" />

マットの結果にはボックスがありません。`result.matte` は `[0, 1]` の範囲を持つ密な `(H, W)` のfloat32配列で、1は完全な前景、0は完全な背景です。バイナリマスクとは異なり、ソフトマットは髪や毛皮など、アンチエイリアスされたエッジの細部を保持します。`result.cutout()` は元画像とそのアルファチャンネルを合成してRGBA配列を作り、`result.save(path)`（または推論呼び出しの `save=True`）は透明背景のPNGへ直接書き出します。モデルはネイティブの固定1024x1024キャンバスで実行されます。Swinバックボーンの相対位置テーブルがこのサイズに結び付いており、不一致ではエラーを発生させず不適切に補間されるため、別の解像度には対応していません。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

公開サイズはSwin-L階層のバックボーンを持つ `l` 1つです。FeyNobgはBiRefNetのアーキテクチャを採用し、再学習前に3番目のSwin段階を18ブロックから24ブロックへ深層化しています。そのため、LibreYOLOの移植版はBiRefNetの順伝播経路、前処理、単一logit出力契約を再利用します。推論、検証、チェックポイント処理は `birefnet` ファミリーと同じ動作です。

## 検証

`val()` は、対応付けられた画像とマットのフォルダーに対して2つの指標を報告します。どちらも `[0, 1]` の範囲で、解像度に依存しません。MAEは正解アルファに対する平均絶対誤差（低いほど良い）です。S-measure（Fanほか、ICCV 2017）は、ピクセル単位のMAEだけでは捉えられない、対象の形状と穴の保持を評価する構造的類似度（高いほど良い）です。検証ではモデル独自の `predict` を使用するため、ファミリー固有の正確な前処理が適用されます。

<code-tabs name="val" />

検証は推論のみです。アップストリームの `nobg` ライブラリはApache-2.0の学習コードを提供しています。現在のファインチューニングはそこで学習し、LibreYOLO独自の変換スクリプトで結果を変換する方法です。このファミリーで `train()` を呼び出すと、部分的な学習器を実行せずエラーになります。

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
