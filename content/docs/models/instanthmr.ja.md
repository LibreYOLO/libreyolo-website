---
title: InstantHMR
families: [instanthmr]
seo_title: "PythonでInstantHMR：1枚の画像からリアルタイムで人体メッシュ"
description: "LibreYOLOのInstantHMRで、画像内のすべての人物の3D人体メッシュを復元します。サイズはn、t、s、m、さらにApache-2.0の商用向けmチェックポイント。"
lead: "InstantHMRは、NaturalpadとLAAS-CNRSによるリアルタイムの人体メッシュ復元モデルです。人物のボックスごとに、MHRの身体姿勢と体型のパラメータ、2Dと3Dの70個のキーポイント、カメラの並進、18,439頂点のメッシュを返します。LibreYOLOはこれをPyTorchでネイティブに移植し、RF-DETRで人物を検出します。"
keywords: [InstantHMR, 人体メッシュ 復元, HMR, body mesh, 3D 姿勢推定, MHR, Momentum Human Rig, リアルタイム]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreInstantHMRm-mesh.pt")   # 人物はLibreRFDETRm.ptで検出
        result = model("people.jpg")

        meshes = result.meshes            # result.boxesのボックスごとに1行
        print(meshes.body_model)          # "mhr"
        print(meshes.vertices.shape)      # (人数, 18439, 3)、メートル、カメラ座標系
        print(meshes.joints2d.shape)      # (人数, 70, 2)、ピクセル
    - label: 独自のボックス
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreInstantHMRm-mesh-commercial.pt")
        boxes = [[40, 20, 260, 600]]      # 人物ごとにx1, y1, x2, y2
        result = model("people.jpg", person_boxes=boxes)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLO-pose形式のCOCO-17データセットで学習済みチェックポイントをファインチューニング
        model = LibreYOLO("LibreInstantHMRs-mesh.pt")
        model.train(data="my-pose-dataset.yaml", epochs=20)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreInstantHMRm-mesh.pt")
        model.export(format="onnx")   # 入力はimage、cliff_cond、面はメタデータに格納
source_hash: 458443d8f7f54450
---

## インストール

InstantHMRにオプションのextraは不要です。デフォルトの人物検出器`LibreRFDETRm.pt`にはRF-DETRのextraが必要です。

```bash
pip install "libreyolo[rfdetr]"
```

## 推論

重みは初回使用時に、ライセンスの通知のあとでLibreYOLOのHugging Faceミラーからダウンロードされます。

<code-tabs name="predict" />

人物はRF-DETR検出器（デフォルトは`LibreRFDETRm.pt`）で検出されます。独自のボックスを使うには`person_boxes=`を、検出器を差し替えるには`person_detector=`を渡してください。`result.meshes`の行は`result.boxes`と対応します。すべて元画像のカメラ座標系です。`transl`はメートル単位、`vertices`と`joints3d`はメートル単位で`transl`を含み、`joints2d`は元画像上のピクセル単位です。`global_orient`、`body_pose`、`betas`はMHRのパラメータを保持し、MHRの回転はラジアン単位のオイラー角です。ペイロードの全体については[人体メッシュ](/docs/tasks/body-mesh)のタスクページを参照してください。

## バリアント

サイズは`n`、`t`、`s`、`m`の4つで、いずれも人物のクロップは288 pxです。さらに、商用利用可能な画像のみで学習した`m`である`LibreInstantHMRm-mesh-commercial.pt`があります。アップストリームはONNXのみを公開しています。LibreYOLOはPyTorchのチェックポイントを再構築しており、5つのリリースすべてでアップストリームのONNXグラフと8e-5以内で一致します。

## 学習

`train()`は、COCO-17の人物キーポイントを持つYOLO-pose形式のデータセットで、2Dキーポイント、再投影、固定事前分布の損失を使って学習済みチェックポイントをファインチューニングします。ゼロからの学習と`resume`には対応していません。`val()`は例外を発生させます。一般的な人体メッシュのベンチマーク（3DPW、EMDB）は研究用ライセンスのみのため、同梱していません。MPJPE、PA-MPJPE、PVEの指標は、手元のデータセット向けに`libreyolo.validation.mesh_metrics`にあります。

<code-tabs name="train" />

## エクスポート

<export-matrix />

`export(format="onnx")`は、`image`と`cliff_cond`を入力とし、メッシュの面をメタデータに持つグラフを書き出します。

<code-tabs name="export" />

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box>

InstantHMRはNaturalpadとLAAS-CNRSによるものです（[Naturalpad-studio/InstantHMR](https://github.com/Naturalpad-studio/InstantHMR)）。コードはApache-2.0です。重みのライセンスはチェックポイントによって異なります。

- `n`、`t`、`s`、`m`：CC BY-NC 4.0、**非商用のみ**です。学習画像の一部（SA-1B、AI Challenger）が非商用であるためです。
- `m-commercial`：Apache-2.0で、Harmony4Dと商用ライセンスのCOCO画像のみで学習しています。

すべてのチェックポイントは、MetaがSAM Licenseの下で公開したSAM 3D Bodyデータセットのアノテーションでも学習されており、その条件は重みにも引き継がれます。メッシュのデコードはクリーンルームで実装したため、SAMライセンスのsam-3d-bodyのコードは使っていません。チェックポイントは`LibreYOLO/LibreInstantHMR{n,t,s,m}-mesh`と`LibreYOLO/LibreInstantHMRm-mesh-commercial`としてミラーしており、ゲートされていません。

</provenance-box>

## 引用

<citation-block />
