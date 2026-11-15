---
title: MoGe-3
families: [moge3]
seo_title: "PythonでMoGe-3：1枚の画像からメートル深度、点群マップ、カメラ"
description: "LibreYOLOのMoGe-3で、1枚のRGB画像からメートル単位の深度、カメラ座標系の点群マップ、カメラ内部パラメータ、表面法線を得ます。ViT-LとViT-g、MIT。"
lead: "MoGe-3は、Microsoftの単眼ジオメトリモデルです。1枚のRGB画像から、メートル単位の深度、カメラ座標系の点群マップ、カメラ内部パラメータ、有効マスク、表面法線を推定します。MoGe-2に、推定した深度をボクセルグリッド上で補正する疎な3D UNetを加えたものです。LibreYOLOは公式のViT-LとViT-gのチェックポイントを、推論、検証、法線のエクスポート向けに提供します。"
keywords: [MoGe-3, MoGe 3, メートル 深度推定, 単眼 ジオメトリ推定, point map, カメラ 内部パラメータ 推定, 表面法線 推定, DINOv2]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe3l-depth.pt")
        depth = model(SAMPLE_IMAGE).depth_map

        print(depth.data.shape)        # (H, W)、メートル、有効マスク外はinf
        print(depth.point_map.shape)   # (H, W, 3)、カメラ座標系の点、メートル
        print(depth.intrinsics)        # (3, 3)、OpenCVのピクセル内部パラメータ
        print(depth.valid_mask.shape)  # (H, W)、bool
    - label: 法線
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe3l-normal.pt")
        normal = model(SAMPLE_IMAGE).normal_map
        print(normal.array.shape)      # (H, W, 3)、単位ベクトル、OpenCV座標系
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreMoGe3l-depth.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg save=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # エクスポートできるのは法線のグラフのみ、深度のエクスポートは不可
        model = LibreYOLO("LibreMoGe3l-normal.pt")
        model.export(format="onnx")
        model.export(format="torchscript")
source_hash: f9d2d8df5141447a
---

## インストール

MoGe-3にオプションのextraは不要です。疎な補正器は素のPyTorchで、カメラの復元には基本インストールに含まれるSciPyを使います。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にLibreYOLOのHugging Faceミラーからダウンロードされます。

<code-tabs name="predict" />

`-depth`チェックポイントでは、`result.depth_map`はメートル単位の深度を保持します（`encoding="depth"`、`metric=True`）。点群マップ、内部パラメータ、有効マスクは`depth_map.point_map`、`depth_map.intrinsics`、`depth_map.valid_mask`として付属し、マスク外のピクセルは`inf`です。点はOpenCVの軸を使い、内部パラメータはピクセル中心を整数座標に置きます。`-normal`チェックポイントでは、`result.normal_map`はOpenCVのカメラ座標系での単位法線を保持します。

`imgsz`はViTのトークン数の上限を決め、画像のアスペクト比に合わせて`(imgsz // 14) ** 2`個のトークンを配置します。デフォルトの840はアップストリームの最高解像度レベル（3600トークン）で、490は最低レベルに近い値です。入力はネイティブ解像度を維持し、すべての出力はその解像度へ補間し直されます。推論はFP32で実行され、アップストリームのデフォルトと同じく常に3回の補正ステップを適用します。順伝播1回あたり1枚の画像を処理し、バッチをまとめる経路はありません。

CPU、CUDA、MPSのいずれも、コンパイル済みの拡張なしで動作します。補正器のメモリは入力解像度ではなく`imgsz`に応じて増えます。

## バリアント

| 重みファイル | バックボーン | タスク |
| --- | --- | --- |
| `LibreMoGe3l-depth.pt` | DINOv2 ViT-L/14 | 深度、点群マップ、内部パラメータ、マスク |
| `LibreMoGe3l-normal.pt` | DINOv2 ViT-L/14 | 法線 |
| `LibreMoGe3g-depth.pt` | DINOv2 ViT-g/14 | 深度、点群マップ、内部パラメータ、マスク |
| `LibreMoGe3g-normal.pt` | DINOv2 ViT-g/14 | 法線 |

同じサイズの`-depth`と`-normal`のファイルは同じテンソルを保持し、接尾辞でタスクが決まります。RTX 3090上のFP32でアップストリームと比べると、順伝播1回のグラフはビット単位で一致し、補正後の深度は浮動小数点の丸め誤差の範囲で一致します（相対差のp99は`l`で4.7e-6、`g`で3.7e-5）。

MoGe-3はLibreYOLOでは学習できず、`train()`は例外を発生させます。ゼロショットの`val()`は共有の深度と法線のバリデーターを実行します。

## エクスポート

<export-matrix />

エクスポートできるのは法線のグラフのみで、固定の正方形キャンバスでONNXとTorchScriptに出力します。深度のエクスポートはできません。補正器はデータに依存するボクセル集合を構築し、カメラの復元はCPUでの求解だからです。

<code-tabs name="export" />

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box>

MoGe-3はMicrosoftによるものです（[microsoft/MoGe](https://github.com/microsoft/MoGe)）。コードと重みはMITのため、商用利用が可能です。DINOv2バックボーンはApache-2.0の帰属表示を維持します。LibreYOLOは公式の`Ruicheng/moge-3-vitl`と`Ruicheng/moge-3-vitg`のチェックポイントを、変更なしで`LibreYOLO/LibreMoGe3{l,g}-{depth,normal}`としてミラーしており、ゲートされていません。

</provenance-box>

## 引用

<citation-block />
