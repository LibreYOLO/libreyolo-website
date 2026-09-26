---
title: サーフェス法線
seo_title: LibreYOLOのサーフェス法線推定
description: LibreYOLOで1枚の画像から密なサーフェス法線fieldを推論します。カメラ座標系の規約を読み、角度誤差を検証し、モデルをエクスポートします。
lead: >-
  サーフェス法線推定は、見えている各面が向く方向を予測します。LibreYOLOはnormalタスクとして提供し、元画像のキャンバス上にunit
  vectorの密なfieldを返します。
keywords:
  - サーフェス法線 推定 python
  - 画像 normal map
  - 単眼 geometry
  - 角度誤差 指標
  - dense normal prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: 法線fieldを推論
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normals = result.normal_map
        print(normals.data.shape)      # (H, W, 3) float32 unit vector
        normals.assert_normalized()    # unit lengthでないpixelがあれば例外を送出
    - label: 1つのpixelを読み取る
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE)

        # OpenCV camera frame: +xは右、+yは下、+zはsceneの奥
        # カメラを向く面は(0, 0, -1)に近い値
        field = result.normals.data
        h, w = field.shape[:2]
        print(field[h // 2, w // 2])
    - label: 可視化を保存
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE)

        # plot()はfieldをrenderし、normalとedgeの結果向けに定義されている
        result.plot().save("normals.png")
  val:
    - label: 検証して指標キーを読み取る
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])     # degree
        print(metrics["metrics/median_angular_error"])   # degree
        print(metrics["metrics/within_11_25"])           # pixelのpercentage
        print(metrics["metrics/within_22_5"], metrics["metrics/within_30"])
  export:
    - label: エクスポート
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
    - label: エクスポートしたファイルを実行
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で経路を選ぶため、エクスポート済み成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.data.shape)
source_hash: d26d26d894b436ff
---

## 定義

`normal` タスクは1枚のRGB画像から、pixelごとに3成分のunit vectorを予測します。そのpixelにある面が向く方向です。深度と異なり、出力に任意のスケールはないため、2つの推論結果を位置合わせなしで直接比較できます。

推論によって `result.normal_map` という `NormalMap` ペイロードが埋められ、元画像キャンバス上の `(H, W, 3)` float32配列を保持します。`result.normals` からも同じものにアクセスできます。ベクトルはLibreYOLOのOpenCV camera frameを使い、`+x` は右、`+y` は下、`+z` はsceneの奥を表します。ベクトルはカメラを向くため、正面を向く面の値は `(0, 0, -1)` になります。`.assert_normalized()` はすべてのpixelが有限で、許容誤差内のunit lengthであることを確認します。`result.boxes` は空のままなので、`conf`、`iou`、`max_det` は効果がなく、`Results.plot()` はこのタスクに対応します。

## モデル

2つのファミリーが `normal` を提供します。

[MoGe-2](/docs/models/moge-2)は専用ファミリーです。3つのencoderサイズを持つ、単一forwardの単眼geometry modelです。LibreYOLOはこれらのチェックポイントを独自のorganizationへコピーしません。読み込むと、固定されたrevisionの公式リポジトリから一致するサイズをダウンロードし、記録済みSHA-256に対して検証します。

[LibreMODUS](/docs/models/libremodus)はany-to-anyモデルの1つの対象として法線を生成し、RGB画像ではなく深度マップを入力することもできます。`modus` 追加パッケージと、認証済みの自身のHugging Faceアカウントが必要です。`val()` も `export()` も提供しないため、以下の検証およびエクスポートセクションには該当しません。

## 推論

MoGe-2の重みは初回使用時にダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

`imgsz` はViT encoderのpatch sizeで割り切れる必要があり、LibreYOLOは実行開始前に確認します。画像リストの推論では、画像ごとに1回のforward passを実行します。このタスクにはstacked-batchの高速経路がありません。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

法線の検証では、各画像を同じ解像度・同じstemの3チャンネル16-bit PNG、および任意のvalidity maskと対応付けます。

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  normals/
    val/room.png
  masks/
    val/room.png
```

```yaml
path: dataset
train: images/train
val: images/val
normals_dir: normals
masks_dir: masks
nc: 1
names: {0: normal}
```

対象PNGは正確に3チャンネルの `uint16` で、チャンネルはRGBとして保存されます。decodeは `n = png / 65535 * 2 - 1` の後、各ベクトルを再正規化し、decode後のベクトルは推論結果と同じOpenCV camera frameを使います。mask pixelがnonzeroなら有効です。maskファイルがない場合は、有限でzeroでないすべてのdecode済みベクトルが有効です。無効およびpadding対象のpixelは内部で `(0, 0, 0)` として保持され、どの指標にも寄与しません。完全な仕様については[データセット形式](/docs/reference/dataset-formats)を参照してください。

## 学習

どちらのnormalファミリーにも学習実装はありません。両方で `train()` が `NotImplementedError` を送出します。MoGe-2のページには、推論、検証、エクスポート向けに固定された公式チェックポイントへのリンクがあります。

## 検証

`val()` は、データセットが有効と示すpixelについて、推論ベクトルと正解ベクトルの角度を測定します。

<code-tabs name="val" />

`metrics/mean_angular_error` と `metrics/median_angular_error` はdegree単位の角度で、小さいほど優れています。`metrics/within_11_25`、`metrics/within_22_5`、`metrics/within_30` は、角度誤差がそれぞれ11.25、22.5、30 degree以内となる有効pixelの割合で、大きいほど優れています。単位に注意してください。この3つはfractionではなくpercentageです。`fitness` は `metrics/within_11_25` を100で割った値で、best-checkpoint選択をほかのすべてのタスクと同じ `[0, 1]` スケールに合わせます。

## エクスポート

エクスポート済みnormal modelはファイル接尾辞に基づいて `LibreYOLO()` から再度読み込めるため、`.onnx` ファイルはチェックポイントと同様に動作し、同じ `Results` を返します。

<code-tabs name="export" />

normalのエクスポートは、固定解像度・batch 1のランタイム仕様を使います。`dynamic` と1以外の `batch` は拒否され、`imgsz` はencoderのpatch sizeで割り切れる必要があります。形式ごとの対応範囲は[MoGe-2ページ](/docs/models/moge-2)と[完全なエクスポートマトリックス](/docs/reference/export-matrix)にあります。[エクスポート](/docs/export)には、すべての形式で受け付ける引数が記載されています。

