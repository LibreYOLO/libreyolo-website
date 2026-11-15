---
title: RiO-DETRv2
families: [riodetrv2]
seo_title: "PythonでRiO-DETRv2：リアルタイムの回転物体検出、Apache-2.0"
description: "LibreYOLOのRiO-DETRv2で、航空画像の回転バウンディングボックス検出を行います。サイズはnからx、DOTA-v1.0とDIOR-Rのチェックポイント、コードと重みはApache-2.0。"
lead: "RiO-DETRv2は、リアルタイムの回転ボックス向けDETRです。ボックスの4辺をボックス自身の座標系における分布として推定し、角度を粗から細へと補正し、角度を考慮した品質で各ボックスをスコア付けします。LibreYOLOはこれを推論専用で、5つのサイズとDOTA-v1.0およびDIOR-Rのチェックポイントとともに移植しています。"
keywords: [RiO-DETRv2, RiO-DETR, 回転 物体検出, OBB, 回転バウンディングボックス, DOTA, DIOR-R, 航空画像 物体検出, DETR]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRiODETRv2s-obb.pt")        # DOTA-v1.0、15クラス
        result = model("aerial.jpg", save=True)

        print(result.obb.xywhr)   # (N, 5)、中心、サイズ、角度
        print(result.obb.conf, result.obb.cls)
    - label: DIOR-R
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRiODETRv2s-obb-diorr.pt")  # DIOR-R、20クラス
        result = model("aerial.jpg")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRiODETRv2s-obb.pt source=aerial.jpg save=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRiODETRv2s-obb.pt")
        model.export(format="onnx")
        model.export(format="torchscript")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRiODETRv2s-obb.pt format=onnx
source_hash: a964bd82aed1c2bc
---

## インストール

RiO-DETRv2にオプションのextraは不要です。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にLibreYOLOのHugging Faceミラーからダウンロードされます。モデルは航空画像で学習されているため、上空から撮影した写真を使ってください。

<code-tabs name="predict" />

`result.obb`は回転ボックスを保持します。`xywhr`（中心、幅、高さ、角度）、`conf`、`cls`で、クラス名はチェックポイントから取得します。ペイロードと描画については[回転物体検出](/docs/tasks/oriented-detection)のタスクページを参照してください。

## バリアント

サイズは`n`、`s`、`m`、`l`、`x`の5つで、それぞれ2つのチェックポイントがあります。

| チェックポイント | データセット | クラス数 | キャンバス |
| --- | --- | ---: | ---: |
| `LibreRiODETRv2<size>-obb.pt` | DOTA-v1.0 | 15 | 1024 px |
| `LibreRiODETRv2<size>-obb-diorr.pt` | DIOR-R | 20 | 800 px |

ネイティブのキャンバスはチェックポイントから読み取られます。生の出力とデコードした回転ボックスは、CPU上の10個すべてのチェックポイントで、アップストリームのApache-2.0のリリースと完全に一致します（最大絶対差0）。

RiO-DETRv2は推論専用です。`train()`は例外を発生させます。アップストリームの学習レシピ（DINOv3の特徴量蒸留、OBBのハンガリアンマッチング、境界と角度の分布損失）は移植していません。学習可能な回転ボックス検出には、[GTR](/docs/models/gtr)または[YOLO-NAS](/docs/models/yolo-nas)を使ってください。

## エクスポート

<export-matrix />

ONNXとTorchScriptのエクスポートは往復可能です。エクスポートしたファイルは`LibreYOLO()`で再度読み込まれ、同じ`Results`を返します。

<code-tabs name="export" />

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box>

RiO-DETRv2はRicePasteMによるものです（[RicePasteM/RiO-DETRv2](https://github.com/RicePasteM/RiO-DETRv2)）。コードと重みはApache-2.0のため、商用利用が可能です。アップストリームは最初のコミットから約33分後に、コードを変更せずにLICENSEファイルをAGPL-3.0に変更しました。LibreYOLOの移植版はその最初のApache-2.0のリリース（コミット`ab8b22d`）のみを使い、そのスナップショットを`LibreYOLO/RiO-DETRv2`に保存しています。公式のチェックポイントは、キーの名前を変更し学習済みテンソルは変更せずに、`LibreYOLO/LibreRiODETRv2{n,s,m,l,x}-obb`および`-obb-diorr`としてミラーしています。DOTA-v1.0とDIOR-Rの学習セットにはそれぞれ独自の利用条件があります。

</provenance-box>

## 引用

<citation-block />
