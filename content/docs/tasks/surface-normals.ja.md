---
title: サーフェス法線
seo_title: LibreYOLOによるサーフェス法線推定
description: LibreYOLOで1枚の画像から密なサーフェス法線場を推論します。カメラ座標系の規約、角度誤差による検証、モデルのエクスポートを解説します。
lead: >-
  サーフェス法線推定は、可視表面の各位置が向く方向を予測します。LibreYOLOではnormalタスクとして提供され、元画像のキャンバス上に単位ベクトルの密な場を返します。
keywords:
  - Python サーフェス法線推定
  - 画像 法線マップ
  - 単眼幾何推定
  - 角度誤差 指標
  - 密な法線予測
last_verified: 1.5.0
snippets:
  predict:
    - label: 法線場を推論
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normals = result.normal_map
        print(normals.data.shape)      # (H, W, 3) float32単位ベクトル
        normals.assert_normalized()    # 単位長でないピクセルがあると例外を送出
    - label: 1ピクセルを読み取る
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE)

        # OpenCVカメラ座標系 +xは右 +yは下 +zはシーン奥
        # カメラを向く表面は(0, 0, -1)に近い値となる
        field = result.normals.data
        h, w = field.shape[:2]
        print(field[h // 2, w // 2])
    - label: 可視化を保存
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE)

        # plot()は法線場を描画する 法線とエッジの結果に定義されている
        result.plot().save("normals.png")
  val:
    - label: 検証して指標キーを確認
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])     # 度
        print(metrics["metrics/median_angular_error"])   # 度
        print(metrics["metrics/within_11_25"])           # ピクセルの割合
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

        # ファクトリはファイルの拡張子に応じて振り分けるためエクスポートした
        # アーティファクトもチェックポイントと同様に読み込まれ同じResultsオブジェクトを返す
        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.data.shape)
source_hash: d26d26d894b436ff
---

## 定義

`normal`タスクは1枚のRGB画像から、ピクセルごとに3成分の単位ベクトルを予測します。これはそのピクセルにある表面が向いている方向です。深度と異なり、出力に自由なスケールはないため、位置合わせせず2つの推論結果を直接比較できます。

推論結果では`result.normal_map`に値が格納されます。これは元画像のキャンバス上にある`(H, W, 3)` float32配列を保持する`NormalMap`ペイロードで、`result.normals`からも取得できます。ベクトルはLibreYOLOのOpenCVカメラ座標系を使用し、`+x`は右、`+y`は下、`+z`はシーンの奥です。ベクトルはカメラ側を向くため、正面向きの平面は`(0, 0, -1)`になります。`.assert_normalized()`はすべてのピクセルが有限で、許容誤差内の単位長であることを確認します。`result.boxes`は空のままなので、`conf`、`iou`、`max_det`は効果がありません。`Results.plot()`はこのタスクに対応します。

## モデル

`normal`には2つのファミリーが対応します。

[MoGe-2](/docs/models/moge-2)は専用モデルです。3種類のエンコーダーサイズを持つ、1回の順伝播で動作する単眼幾何モデルです。LibreYOLOはこれらのチェックポイントを自身の組織へコピーしません。読み込むと固定済みリビジョンの公式リポジトリから対応サイズをダウンロードし、記録済みのSHA-256と照合します。

[LibreMODUS](/docs/models/libremodus)は任意入力から任意出力へのモデルにある1つの対象として法線を生成し、RGB画像だけでなく深度マップを入力として使用できます。`modus`追加パッケージと認証済みのHugging Faceアカウントが必要で、`val()`と`export()`には対応しないため、以下の検証とエクスポートのセクションの対象ではありません。

## 推論

MoGe-2の重みは初回使用時にダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

`imgsz`はViTエンコーダーのパッチサイズで割り切れる必要があり、LibreYOLOが実行開始前に確認します。画像リストの推論は画像ごとに1回の順伝播を実行します。このタスクにはバッチを積み重ねる高速経路がありません。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

法線の検証では、各画像を同じ解像度かつ同じファイル名の基幹部を持つ3チャンネル16ビットPNGと対応付け、任意で有効性マスクも使用します。

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

対象PNGは正確に3チャンネルの`uint16`で、チャンネルはRGBとして格納します。デコード式は`n = png / 65535 * 2 - 1`で、その後に各ベクトルを再正規化します。デコード済みベクトルは予測と同じOpenCVカメラ座標系を使用します。マスクピクセルは0以外で有効と見なされます。マスクファイルがない場合は、有限かつ0でないデコード済みベクトルがすべて有効です。無効な対象ピクセルとパディングされた対象ピクセルは内部で`(0, 0, 0)`として保持され、指標に一切寄与しません。完全な契約については[データセット形式](/docs/reference/dataset-formats)を参照してください。

## 学習

どちらの法線ファミリーにも学習実装はありません。両方で`train()`が`NotImplementedError`を送出します。MoGe-2のページでは、推論、検証、エクスポートに使う固定済みの公式チェックポイントを案内しています。

## 検証

`val()`はデータセットが有効と示すピクセルに対して、予測ベクトルと正解ベクトルの角度を測定します。

<code-tabs name="val" />

`metrics/mean_angular_error`と`metrics/median_angular_error`は角度を度単位で表し、小さいほど良い指標です。`metrics/within_11_25`、`metrics/within_22_5`、`metrics/within_30`は角度誤差が11.25度、22.5度、30度以内に収まる有効ピクセルの割合で、大きいほど良い指標です。単位に注意してください。この3つは小数ではなくパーセントです。`fitness`は`metrics/within_11_25`を100で割った値で、最良チェックポイントの選択を他のタスクと同じ`[0, 1]`スケールにします。

## エクスポート

エクスポートした法線モデルはファイルの拡張子に基づいて`LibreYOLO()`で再読み込みできるため、`.onnx`ファイルもチェックポイントと同様に動作し、同じ`Results`を返します。

<code-tabs name="export" />

法線のエクスポートは固定解像度、バッチ1のランタイム契約を使用します。`dynamic`と1以外の`batch`は拒否され、`imgsz`はエンコーダーのパッチサイズで割り切れる必要があります。形式ごとの対応範囲は[MoGe-2ページ](/docs/models/moge-2)と[完全なエクスポートマトリックス](/docs/reference/export-matrix)に記載されています。[エクスポート](/docs/export)には各形式で受け付ける引数が記載されています。
