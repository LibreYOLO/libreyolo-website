---
title: 背景除去
seo_title: LibreYOLOによる背景除去
description: LibreYOLOで被写体を背景から切り抜きます。ソフトアルファマットの推論、透明PNGの書き出し、MAEとS-measureによる検証を解説します。
lead: >-
  背景除去は被写体を背後のすべてから分離します。LibreYOLOではmatteタスクとして提供され、硬い前景マスクではなく、ピクセルごとにソフトなアルファ値を返します。
keywords:
  - Python 背景除去
  - アルファマッティング モデル
  - 二分画像セグメンテーション
  - 透過PNG 切り抜き
  - ソフトアルファマット
last_verified: 1.5.0
snippets:
  predict:
    - label: マットを推論
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)   # (H, W) float32で範囲は[0, 1]
    - label: 透明PNGを書き出す
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # save()はソースとマットをアルファチャンネルとして合成する
        result.save("subject.png")

        rgba = result.cutout()   # メモリ内の同じ(H, W, 4) uint8配列
        print(rgba.shape)
    - label: 新しい背景へ合成
      language: python
      code: >
        import numpy as np

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        result = model(SAMPLE_IMAGE)


        rgba = result.cutout()

        alpha = rgba[..., 3:4].astype(np.float32) / 255.0

        backdrop = np.full_like(rgba[..., :3], 255)          # 白

        composited = (rgba[..., :3] * alpha + backdrop * (1 -
        alpha)).astype(np.uint8)

        print(composited.shape)
  val:
    - label: 検証して指標キーを確認
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # dataset YAMLの代わりにimages/とマットのディレクトリを含む
        # ディレクトリをそのまま使用できる
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])        # 小さいほど良い
        print(metrics["metrics/Smeasure"])   # fitnessで大きいほど良い
  export:
    - label: エクスポート
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="torchscript")
    - label: エクスポートしたファイルを実行
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリはファイルの拡張子に応じて振り分けるためエクスポートした
        # アーティファクトもチェックポイントと同様に読み込まれ同じResultsオブジェクトを返す
        model = LibreYOLO("LibreBiRefNetl-matte.torchscript")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: f7d88c74d9729268
---

## 定義

`matte`タスクは1枚のRGB画像からピクセルごとに1つのアルファ値を予測します。`1`は完全な前景、`0`は完全な背景です。この値は二値ではなく連続値であり、それがこのタスクの要点です。0.5でしきい値処理すれば硬いマスクを得られますが、ソフトマットには、二値マスクでは失われる髪、毛、モーションブラーのある境界での部分的な被覆も含まれます。

推論結果では`result.matte`に値が格納されます。これは元画像のキャンバス上にある`[0, 1]`範囲の`(H, W)` float32配列を保持する`Matte`ペイロードで、`.array`からNumPy配列として取得できます。`result.cutout()`はソース画像とそのアルファを合成し、`(H, W, 4)`のuint8 RGBA配列を返します。`result.save(path)`は同じものを背景が透明なPNGとして書き込みます。`result.boxes`は空のままなので、`conf`、`iou`、`max_det`は効果がありません。

## モデル

`matte`には2つのファミリーが対応し、同じ順伝播経路を共有します。

[BiRefNet](/docs/models/birefnet)はこのタスクの基盤となる双方向参照ネットワークで、ここではSwin-Lティアのチェックポイントを1つ公開しています。

[FeyNobg](/docs/models/feynobg)はFeyn Inc.による深化バリアントです。BiRefNetの第3Swinステージを18ブロックから24ブロックへ増やしたアーキテクチャを再学習しています。LibreYOLOはBiRefNetの順伝播経路、前処理、単一ロジット出力を再利用するため、推論、検証、チェックポイント処理は同じように動作します。重みとファミリー識別子はFeyNobg独自のものです。

2つの重みには異なるライセンスが適用されます。どちらもモデルページに記載されており、特定チェックポイントのHugging Faceリポジトリにあるライセンスが正式な情報源です。

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

両ファミリーは固定されたネイティブの1024×1024キャンバスで実行し、マットを元画像のサイズに戻します。Swinバックボーンの相対位置テーブルはこのサイズに結び付いており、不一致の場合は例外ではなく不適切な補間が行われるため、別の解像度には対応していません。マットの結果に対してだけ`Results.save()`が定義されており、ソース画像が必要です。画像を渡さなければ`Results.path`から再読み込みします。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

マットの検証では、各RGB画像を同じファイル名の基幹部を持つ単一チャンネルの正解アルファマットと対応付けます。0が背景、255が前景です。

```text
my-matte-dataset/
  images/
    subject.jpg
  mattes/
    subject.png
```

このルートを`data=`に渡すだけで使用できます。マットのディレクトリは`mattes/`、`matte/`、`gt/`、`masks/`、`mask/`、`alpha/`から自動検出されます。別の方法としてデータセットYAMLを使い、`path`に加えて、それを基準とする画像とマットのディレクトリを`val_images`と`val_mattes`で指定できます。

```yaml
path: my-matte-dataset
val_images: images
val_mattes: mattes
nc: 1
names: {0: matte}
```

`nc`と`names`はスキーマ上のプレースホルダーです。マットモデルは検出ではなく`Results.matte`を返します。マット値は255で割ることで`[0, 1]`範囲のアルファとして読み込まれます。予測キャンバスと形状が異なるマットは、双線形補間で同じサイズへ変更されます。完全な契約については[データセット形式](/docs/reference/dataset-formats)を参照してください。

## 学習

どちらのマットファミリーにも学習実装はありません。両方で`train()`が`NotImplementedError`を送出し、マットの対応範囲は推論、検証、エクスポートだけです。各モデルページには、学習コードを提供するアップストリームプロジェクトと、チェックポイントを取り込む変換スクリプトが記載されています。

## 検証

`val()`はモデル固有の`predict`を実行するため、検証ではそのファミリー固有の前処理を使用し、両方の指標を元画像のキャンバス上で計算します。

<code-tabs name="val" />

`metrics/MAE`は正解アルファに対する`[0, 1]`範囲の平均絶対誤差で、小さいほど良い値です。`metrics/Smeasure`はFanら（ICCV 2017）のS-measureで、被写体の形状と内部の穴を正しく捉えた場合に評価する構造的類似度です。ピクセルごとの平均だけではこの性質を捉えられません。値は大きいほど良く、最良チェックポイントの選択で使う`fitness`にもなります。どちらの指標も解像度には依存しません。

## エクスポート

エクスポートしたマットモデルはファイルの拡張子に基づいて`LibreYOLO()`で再読み込みできるため、アーティファクトはチェックポイントと同様に動作し、同じ`Results`を返します。

<code-tabs name="export" />

このタスクで検証済みの経路はTorchScriptです。ONNX変換は実行できますが、同じ一致度の基準をまだ満たしていません。残りの形式は利用できません。形式ごとの対応範囲は[BiRefNet](/docs/models/birefnet)と[FeyNobg](/docs/models/feynobg)のページ、および[完全なエクスポートマトリックス](/docs/reference/export-matrix)に記載されています。
