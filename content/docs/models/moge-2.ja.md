---
title: MoGe-2
families:
  - moge2
seo_title: MoGe-2：表面法線を推論、検証、エクスポート
description: >-
  LibreYOLOのMoGe-2で密な表面法線を予測します。公式のViT-S、ViT-B、ViT-Lチェックポイントをインストールし、推論、検証、エクスポートします。
lead: >-
  MoGe-2は1枚のRGB画像から密な表面法線場を予測する、1回の順伝播による単眼幾何モデルです。LibreYOLOは公式のViT-S、ViT-B、ViT-Lチェックポイントを通じて、法線推定だけに対応します。
keywords:
  - MoGe-2
  - MoGe 2
  - 表面法線 推定
  - 単眼 幾何
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
        print(normal.array.shape)   # (H, W, 3) float32の単位ベクトル
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

        print(metrics["metrics/mean_angular_error"])   # 度単位
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

MoGe-2にオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時に自動的にダウンロードされます。LibreYOLOは公式チェックポイントから対応するサイズを直接取得し、ローカルにキャッシュします。

<code-tabs name="predict" />

MoGe-2は検出集合ではなく密な場を返すため、`result.boxes` は空で、`conf`、`iou`、`max_det` は効果を持ちません。`result.normal_map` が結果を保持します。OpenCVのカメラ座標系における単位ベクトルの `(H, W, 3)` 配列で、`+x` は右、`+y` は下、`+z` は場面の奥です。カメラを向く表面は `(0, 0, -1)` です。画像一覧の推論は、画像ごとに1回の順伝播を実行します。このファミリーには、スタックしたバッチを使う高速経路がありません。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

3つのエンコーダーサイズViT-S、ViT-B、ViT-Lが個別のチェックポイントとして提供され、すべて同じ入力解像度です。LibreYOLOのベンチマークハーネスはこのファミリーを測定していないため、比較できる公開精度値はありません。利用可能な計算資源に合わせてサイズを選択してください。

## 検証

`val()` は、対応付けられた法線マップデータセットに対して角度誤差を測定します。画像の隣に同じステム名の16ビット法線PNGを置き、オプションの有効性マスクによってパディング済みピクセルと無効なピクセルを計測対象外にできます。度単位の平均角度誤差と中央値角度誤差に加え、11.25度、22.5度、30度以内のピクセルの割合を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

法線のエクスポートは固定解像度、バッチ1のランタイム契約を使用します。`dynamic` と1以外の `batch` は拒否され、`imgsz` はViTエンコーダーのパッチサイズで割り切れる必要があります。LibreYOLOは実行開始前に確認します。エクスポート済み成果物はファイル接尾辞によって `LibreYOLO()` から再度読み込まれるため、`.onnx` ファイルはチェックポイントのように動作し、同じ `Results` を返します。

<code-tabs name="export" />

## ライセンス

<provenance-box>

LibreYOLOはこれらのチェックポイントを独自の組織にコピーしません。`LibreYOLO("LibreMoGe2s-normal.pt")` は固定されたリビジョンの公式Hugging Faceリポジトリから対応するサイズを直接ダウンロードし、使用前に記録済みのSHA-256チェックサムと照合します。

</provenance-box>

## 引用

<citation-block />
