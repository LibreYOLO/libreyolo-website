---
title: MoGe-2
families:
  - moge2
seo_title: MoGe-2：サーフェス法線の推論、検証、エクスポート
description: >-
  LibreYOLOのMoGe-2で密なサーフェス法線を予測します。公式ViT-S、ViT-B、ViT-Lチェックポイントのインストール、推論、検証、エクスポートを説明します。
lead: >-
  MoGe-2は1枚のRGB画像から密なサーフェス法線場を1回の順伝播で予測する単眼geometryモデルです。LibreYOLOは公式ViT-S、ViT-B、ViT-Lチェックポイントを通じて法線推定だけに対応します。
keywords:
  - MoGe-2 使い方
  - MoGe 2
  - サーフェス法線 推定
  - 単眼 geometry
  - normal map
  - dense prediction
  - DINOv2
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normal = result.normal_map
        print(normal.array.shape)   # (H, W, 3) float32単位ベクトル
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMoGe2s-normal.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])   # 度
        print(metrics["metrics/median_angular_error"])
        print(metrics["metrics/within_11_25"])          # ピクセルの割合
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMoGe2s-normal.pt data=my-dataset.yaml imgsz=518
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
        model.export(format="tensorrt", imgsz=518, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMoGe2s-normal.pt format=onnx imgsz=518

        libreyolo export model=LibreMoGe2s-normal.pt format=tensorrt imgsz=518
        half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.array.shape)
source_hash: ddfacf6b7e9729f6
---

## インストール

MoGe-2に任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時に自動的にダウンロードされます。LibreYOLOは一致するサイズを公式チェックポイントから
直接取得し、ローカルにキャッシュします。

<code-tabs name="predict" />

MoGe-2は検出結果の集合ではなく密な場を返すため、`result.boxes`は空で、`conf`、`iou`、
`max_det`に効果はありません。結果は`result.normal_map`に入ります。OpenCVカメラ座標系における
単位ベクトルの`(H, W, 3)`配列で、`+x`は右、`+y`は下、`+z`はシーンの奥を指し、カメラへ
向いた面は`(0, 0, -1)`になります。画像リストの推論は画像ごとに1回の順伝播を実行します。
このファミリーにstacked-batchの高速経路はありません。入力ソース、ストリーミング、結果の処理に
ついては[推論](/docs/predict)を参照してください。

## バリアント

エンコーダーのサイズはViT-S、ViT-B、ViT-Lの3種類で、別々のチェックポイントとして提供され、
すべて同じ入力解像度を使います。LibreYOLOのベンチマークharnessはこのファミリーを測定していないため、
比較できる公開精度値はありません。使用できる計算資源に合わせてサイズを選んでください。

## 検証

`val()`は、対応するnormal mapデータセットに対して角度誤差を測定します。同じstemを持つ16ビットの
法線PNGを画像と並べ、任意の有効性マスクによりpaddingされたピクセルと無効なピクセルを除外できます。
平均角度誤差と中央値角度誤差を度単位で返し、さらに11.25度、22.5度、30度以内のピクセルの割合を
返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

法線のエクスポートは固定解像度、バッチ1のランタイム契約を使います。`dynamic`と1以外の`batch`は
拒否され、`imgsz`はViTエンコーダーのpatch sizeで割り切れる必要があります。LibreYOLOは実行開始前に
確認します。エクスポートした成果物はファイル接尾辞に基づいて`LibreYOLO()`から再読み込みされます。
そのため、`.onnx`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。

<code-tabs name="export" />

## ライセンス

<provenance-box>

LibreYOLOはこれらのチェックポイントを自身の組織へコピーしません。
`LibreYOLO("LibreMoGe2s-normal.pt")`は固定されたrevisionの公式Hugging Faceリポジトリから
一致するサイズを直接ダウンロードし、使用前に記録済みのSHA-256 checksumと照合します。

</provenance-box>

## 引用

<citation-block />

