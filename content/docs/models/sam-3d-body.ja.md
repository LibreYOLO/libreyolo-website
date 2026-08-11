---
title: SAM 3D Body
families:
  - sam3dbody
seo_title: SAM 3D Body：LibreYOLOで全身メッシュを復元
description: >-
  LibreYOLOのSAM 3D Bodyで人体の全身メッシュを復元します。インストールと推論を説明します。チェックポイントはMetaのSAM
  Licenseでアクセス制限され、CUDAが必要です。
lead: >-
  SAM 3D
  Bodyは、1枚の画像と人物ボックスから手足を含む全身の3Dメッシュを復元するMetaのプロンプト可能なモデルです。LibreYOLOは移植せず、アップストリームパッケージをラップします。
keywords:
  - SAM 3D Body 使い方
  - human mesh recovery
  - 人体メッシュ
  - MHR
  - Momentum Human Rig
  - 3D 姿勢推定
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # このファミリーはLibreYOLO()ファクトリに未登録のため直接構築
        # model_path=Noneでアクセス制限付きHugging Faceダウンロードを開始
        # 文字列は既存のローカルチェックポイントパスとして扱い
        # 自動取得しない。推論にはCUDAデバイスが必要でCPU経路はない
        model = LibreSAM3DBody(None, size="d3", device="cuda")
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        meshes = result.meshes
        print(meshes.vertices.shape)    # (N, V, 3)、カメラ座標系、メートル
        print(meshes.joints3d.shape)    # (N, J, 3)
    - label: 人物検出器と使用
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # ここに名前付き文字列の短縮形はない。構築済みLibreYOLO検出器
        # 通常のcallable、またはPersonDetectorインスタンスを渡す
        detector = LibreYOLO("LibreRFDETRn.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 8edc8d7872f3f875
---

## インストール

```bash
pip install libreyolo
```

これでインストールされるのはLibreYOLOのアダプターだけです。SAM 3D Body自体は同梱されません。
そのライセンスはLibreYOLO独自のコードが派生可能なものではないためです。アップストリーム
リポジトリをcloneして依存関係を自分でインストールし、そのcloneをLibreYOLOへ指定してください。

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

```python
from libreyolo.models.sam3dbody import LibreSAM3DBody

model = LibreSAM3DBody(
    None,
    size="d3",
    sam_3d_body_path="/path/to/sam-3d-body",
    device="cuda",
)
```

または、呼び出すたびに`sam_3d_body_path`を渡す代わりに、`SAM_3D_BODY_PATH`環境変数を
設定します。このファミリーを構築しなければインポートは発生せず、SAM Licenseが適用されることも
ありません。このファミリーは`LibreYOLO()`ファクトリにも`libreyolo predict` CLIコマンドにも
接続されていません。唯一のエントリポイントは`LibreSAM3DBody`です。

## 推論

<code-tabs name="predict" />

チェックポイントのダウンロードにはアクセス制限があります。初回ダウンロードを成功させるには、
Hugging FaceのモデルページでMetaのライセンスに同意し、`hf auth login`で認証する必要があります。
推論自体にも無条件でCUDAデバイスが必要です。アップストリームの推定器は確認せずにバッチをGPUへ
移動するため、CPU専用マシンではフォールバックせず例外が発生します。`result.meshes`は
`result.boxes`と行が揃った`Meshes`payloadです（検出された人物ごとに1行）。`vertices`と
`joints3d`はメートル単位で、推定されたカメラ平行移動をすでに含みます。`joints2d`は元画像上の
ピクセル単位で、回転はaxis-angleではなくEuler角を使うMHRの規則に従います。入力ソース、
ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

同じMHR body modelのバックボーンは2種類です。`d3`はDINOv3 ViT-H/16+エンコーダー、
`h`は元のViT-Hエンコーダーを使います。

## エクスポート

<export-matrix />

body meshのエクスポートは未実装です。MHRパラメータのレイアウトをPyTorch外で表現する方法を含め、
LibreYOLOはメッシュタスクのエクスポート済みグラフ契約をまだ定義していません。

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box>

チェックポイントが駆動するbody modelのMHR（Momentum Human Rig）は、Apache-2.0に基づく
Metaの別リリースです。LibreYOLOは実行時にMHR自身の公開リリースからTorchScript成果物を取得し、
ローカルにキャッシュします。そのファイルはLibreYOLOにミラーされず、SAM Licenseではなく
独自のApache-2.0条件が適用されます。

</provenance-box>

## 引用

<citation-block />

