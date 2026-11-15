---
title: TripoSplat
families: [triposplat]
seo_title: "PythonでTripoSplat：1枚の画像から3Dガウシアンスプラットへ"
description: "TripoSplatをLibreYOLOで実行します。物体の写真1枚を入力すると、物体全体が最大262,144個の3Dガウシアンとして.plyまたは.splatで出力されます。MITライセンスで、DINOv3エンコーダーにはMetaのライセンスが適用されます。"
lead: "TripoSplatはTripoAI（VAST）による画像から3Dを生成するモデルです。物体の写真1枚を受け取り、物体全体を3Dガウシアンスプラットとして返します。Speridlabs Idolのベースとなっているモデルで、LibreYOLOは同じLibreGenの呼び出しと出力でこれを実行します。"
keywords: [TripoSplat, TripoAI, VAST, 画像から3D, image to 3d, 3d gaussian splatting, ガウシアンスプラッティング, 3DGS, PLY, splat, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("triposplat")      # 初回使用時に重みをダウンロード
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.ply")           # 標準の 3DGS レイアウト
        result.save("object.splat")         # コンパクトな Web 形式
        print(len(result.gaussians))
    - label: CLI
      language: bash
      code: |
        # runs/generate/predict に物体を .ply として保存
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=triposplat seed=0
    - label: 設定
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("triposplat")
        result = model(SAMPLE_IMAGE, steps=20, gaussians=131072)
source_hash: 0e7cb2342a17effc
---

## インストール

```bash
pip install "libreyolo[gen]"
```

このextraには`diffusers`（FLUX.2 VAE）と`transformers`（DINOv3エンコーダー）が含まれます。

## 推論

アクセス制限はなく、ログインも不要です。重みは初回使用時に、固定されたリビジョンでダウンロードされます。

<code-tabs name="predict" />

0.66Bのrectified-flowトランスフォーマーが、1024 pxの入力のDINOv3トークンとFLUX.2 VAEトークンを条件として、物体の潜在表現と入力カメラをデノイズします。オクツリーデコーダーがアンカーを配置し、ガウシアンデコーダーが各アンカーを32個のガウシアンに変換し、合計で32,768〜262,144個になります。CUDAとApple MPSではfp16、CPUではfp32で動作し、コンパイル済みの経路はありません。

出力は[Speridlabs Idol](/docs/models/speridlabs-idol)と同じです。`result.gaussians`は`xyz`、`scales`、`rotations`、`opacities`、`sh`、`colors`、推定された`camera`、`seed`、`params`を持ちます。物体はz-upで、`[-0.5, 0.5]^3`の中心に配置されます。`.ply`はy-upに変換した標準の3DGSレイアウトで、`.splat`はコンパクトなWeb形式です。

シードはアップストリームと同じくモデルのデバイス上の乱数生成器を駆動するため、同じシードでもデバイスが異なれば別の物体になります。アップストリームとは異なり、オクツリーのサンプリングもシード付きの生成器から乱数を取るため、シードで物体全体を再現できます。

### 設定

| 設定 | デフォルト | 意味 |
| --- | --- | --- |
| `steps` | `20` | サンプリングステップ数。アップストリームの推奨は10〜20。 |
| `guidance` | `3.0` | classifier-free guidance。1以下で無効。 |
| `shift` | `3.0` | フローの時間シフト。 |
| `gaussians` | `262144` | 出力のガウシアン数。32,768〜262,144。 |
| `margin` | `1.2` | 物体サイズに対する切り抜きの辺の比率（アップストリームと同じ）。 |

## バリアント

サイズは`0.6b`（0.66Bパラメータ）の1つで、約1.8 GBです。これに加えて、Idolも使うDINOv3 ViT-H+エンコーダー（1.7 GB）とFLUX.2 VAEエンコーダー（0.2 GB）が必要です。

## エクスポート

<export-matrix />

LibreGenの3Dファミリーではエクスポートは実装されていません。`export()`、`train()`、`val()`は`NotImplementedError`を発生させます（ライブラリ内のADR 0031）。

## ライセンス

<provenance-box>

TripoSplatはTripoAIによるものです（[VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)）。コードと重みはMITです。LibreYOLOはデノイザー、デコーダー、BiRefNetのファイルをバイト単位で同一のままLibreYOLO/LibreTripoSplat0.6bとしてミラーしています。FLUX.2 VAEはApache-2.0のblack-forest-labs/FLUX.2-klein-4Bに由来します。

**DINOv3画像エンコーダーはMITではありません。** これはMetaのもので、DINOv3 Licenseの下にあります。貿易管理と最終用途の制限（軍事、兵器、諜報、核での利用など）を伴う独自の契約です。LibreYOLOはこれを契約とともにbfloat16でLibreYOLO/LibreDINOv3vith16plusにミラーしているため、アクセス制限はありません。このモデルで製品を出荷する前に必ず読んでください。

</provenance-box>

## 引用

<citation-block />
