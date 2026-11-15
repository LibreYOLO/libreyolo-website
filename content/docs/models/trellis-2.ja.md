---
title: TRELLIS.2
families: [trellis2]
seo_title: "PythonでTRELLIS.2：1枚の画像から最大1536^3の3Dメッシュへ"
description: "MicrosoftのTRELLIS.2 4BをLibreYOLOで実行します。物体の写真1枚を入力すると、形状が三角形メッシュとして出力されます。CUDA、MPS、CPU上の純粋なPyTorchで動作します。コードはMITで、DINOv3エンコーダーにはMetaのライセンスが適用されます。"
lead: "TRELLIS.2はMicrosoftのオープンな4Bの画像から3Dを生成するモデルです。最大1536^3のグリッド上で、物体の形状をメッシュとして返します。LibreYOLOはLibreGenファクトリーを通じて同じ重みを純粋なPyTorchで実行し、ジオメトリのみを扱います。"
keywords: [TRELLIS.2, TRELLIS 2, Microsoft, 画像から3D, image to 3d, メッシュ生成, structured latents, OBJ, DINOv3, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("trellis2")        # "trellis.2" も可
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.obj")
        m = result.meshes
        print(m.vertices[0].shape, m.faces[0].shape, m.seed)
    - label: CLI
      language: bash
      code: |
        # runs/generate/predict に物体を .obj として保存
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=trellis2 seed=0
    - label: 設定
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("trellis2")
        result = model(SAMPLE_IMAGE, resolution=512)    # 形状ステージ1つで高速
        result = model(SAMPLE_IMAGE, resolution=1536)   # 最も細かいグリッド
source_hash: 9f55834ac4ceb0cd
---

## インストール

```bash
pip install "libreyolo[gen]"
```

このextraには`transformers`が含まれ、DINOv3画像エンコーダーを読み込みます。

## 推論

アクセス制限はなく、ログインも不要です。重みは初回使用時に、固定されたリビジョンでダウンロードされます。

<code-tabs name="predict" />

構造ステージが占有されたボクセルを配置し、形状ステージがそれらの上に潜在表現を埋め（512で、続いて1024または1536でもう一度）、疎なデコーダーがそれをデュアルグリッドのメッシュに変換します。両方のフローステージはDINOv3 ViT-L/16のトークンを読み取ります。アップストリームはCUDA専用の拡張（FlexGEMM、flash-attn、o-voxel）を必要としますが、LibreYOLOは同じ重みを純粋なPyTorchで、NVIDIA GPU、Apple silicon、CPU上で実行します。ノイズはCPU上で生成されるため、シードはカーネルの丸め誤差の範囲で、どのデバイスでも同じ物体を与えます。

`result.meshes`は、z-upで`[-0.5, 0.5]^3`の範囲にある`vertices`（V, 3）と`faces`（F, 3）のメッシュを保持します。これはジオメトリのみで、アップストリームのCuMeshによるクリーンアップ前の生のデュアルグリッドです。面の向きは一貫しておらず、使われない頂点もあります。アップストリームのテクスチャステージは非商用のNVIDIAライブラリであるnvdiffrastでベイクするため、LibreYOLOには含まれていません。

ライブラリの計測では、A100で1024の物体1つに19.5秒とピーク14.3 GBのGPUメモリ、512では5.5秒と11.6 GBでした。

### 設定

| 設定 | デフォルト | 意味 |
| --- | --- | --- |
| `steps` | `12` | サンプリングステップ数。 |
| `guidance` | `7.5` | classifier-free guidance。 |
| `resolution` | `1024` | 出力グリッド：512、1024、または1536。 |
| `margin` | `1.0` | 物体サイズに対する切り抜きの辺の比率。 |
| `max_tokens` | `49152` | カスケードのトークン予算。収まるように解像度を下げることがあり、使用した解像度は`params`に記録されます。 |

## バリアント

サイズは`4b`の1つです。5つのネットワークで4Bパラメータ、約8.7 GBで、DINOv3エンコーダーにさらに1.2 GB、TRELLISのミラーから1.3 GBが必要です。

## エクスポート

<export-matrix />

LibreGenの3Dファミリーではエクスポートは実装されていません。`export()`、`train()`、`val()`は`NotImplementedError`を発生させます（ライブラリ内のADR 0031）。

## ライセンス

<provenance-box>

TRELLIS.2はMicrosoft Researchによるものです（[microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)）。コードと`microsoft/TRELLIS.2-4B`の重みはMITです。

**パイプラインの条件は画像エンコーダーが決めます。** これはMetaの`facebook/dinov3-vitl16-pretrain-lvd1689m`で、DINOv3 Licenseの下にあります。これはOSI承認ではない独自の契約で、貿易管理と最終用途の制限（軍事、兵器、諜報、核での利用など）を引き継ぎます。このモデルで製品を出荷する前に必ず読んでください。

重みは固定されたリビジョンで変更なしに再配布されており、アクセス制限のあるものはありません。ジオメトリのネットワークはLibreYOLO/LibreTRELLIS2-4bとして、TRELLISの構造デコーダーとBiRefNet（MIT）はLibreYOLO/LibreTRELLISlから、DINOv3エンコーダーは契約とともにLibreYOLO/LibreDINOv3vitl16として提供されます。nvdiffrast、nvdiffrec、RMBG-2.0（いずれも非商用）は使用しません。

</provenance-box>

## 引用

<citation-block />
