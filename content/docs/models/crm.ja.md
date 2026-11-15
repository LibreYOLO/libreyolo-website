---
title: CRM
families: [crm]
seo_title: "PythonでCRM：1枚の画像から色付き3Dメッシュへ、UVテクスチャにも対応"
description: "清華大学のCRMをLibreYOLOで実行します。物体の写真1枚を入力すると、色付きの三角形メッシュが.objとして出力され、テクスチャのベイクも選べます。純粋なPyTorchで動作し、OpenRAIL++-Mの利用制限があります。"
lead: "CRM（Convolutional Reconstruction Model）は清華大学の画像から3Dを生成するモデルです。2つのマルチビュー拡散モデルが物体の6つのビューとその座標マップを生成し、畳み込みUNetがそれらをトライプレーンに変換し、FlexiCubesが色付きメッシュを抽出します。LibreYOLOはLibreGenファクトリーを通じて、純粋なPyTorchでこれを実行します。"
keywords: [CRM, Convolutional Reconstruction Model, 清華大学, 画像から3D, image to 3d, テクスチャ付きメッシュ, FlexiCubes, マルチビュー 拡散モデル, OBJ, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("crm")             # 初回使用時に重みをダウンロード
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.obj")           # 頂点カラーは `v x y z r g b` 形式
        m = result.meshes
        print(m.vertices[0].shape, m.extras["vertex_colors"][0].shape)
    - label: CLI
      language: bash
      code: |
        # runs/generate/predict に物体を .obj として保存
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=crm seed=0
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=crm texture=true
    - label: テクスチャ
      language: python
      code: |
        # pip install xatlas   (MIT ライセンス UV 展開用)
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("crm")
        model(SAMPLE_IMAGE, texture=True).save("object.obj")
        # object.obj と object.mtl と object_texture.png を出力
source_hash: 9bc27f19244a7545
---

## インストール

```bash
pip install "libreyolo[gen]"
```

`texture=True`には、UV展開のためにMITライセンスの`xatlas`パッケージ（`pip install xatlas`）も必要です。

## 推論

アクセス制限はなく、ログインも不要です。重みは初回使用時に、固定されたリビジョンで[LibreYOLO/LibreCRM](https://huggingface.co/LibreYOLO/LibreCRM)からダウンロードされます。

<code-tabs name="predict" />

2つのマルチビュー拡散モデルが物体の6つの正射影ビューを生成し、続いてその正準座標マップを生成します。畳み込みUNetがそれらをトライプレーンに変換し、FlexiCubesがトライプレーンで色付けされたメッシュを抽出します。アップストリームはxformers、nvdiffrast（NVIDIA、非商用）、GPLのメッシュライブラリを必要とします。LibreYOLOは同じ重みを純粋なPyTorchで、NVIDIA GPU、Apple silicon、CPU上で実行し、GPUではアップストリームと同じくfp16を使います。

入力は切り抜きなしで灰色の背景に正方形へパディングされるため、写真は物体がフレームいっぱいに写っているときに最もよく機能します。シードはどのデバイスでも同じ初期ノイズを与えます。CUDAでは同じ呼び出しを繰り返すと同じビューと三角形が得られ、頂点は1e-7程度ずれることがあります。ライブラリの計測では、RTX 3090で1物体あたり約12秒、12 GBでした。

`result.meshes`は、z-upで`[-0.5, 0.5]^3`の範囲にある`vertices`（V, 3）、`faces`（F, 3）、`[0, 1]`のRGBである`extras["vertex_colors"]`（V, 3）を保持します。`texture=True`はアップストリームと同じく1024 x 1024のUVテクスチャをベイクします。

### 設定

| 設定 | デフォルト | 意味 |
| --- | --- | --- |
| `steps` | `50` | 両方の拡散ステージのサンプリングステップ数。1〜499。 |
| `guidance` | `5.0` | 両ステージのclassifier-free guidance。 |
| `texture` | `False` | UVテクスチャをベイク（.objの横に.mtlと.png）。 |

## バリアント

チェックポイントは1つで、パラメータは4.1Bです。2.0Bのマルチビュー拡散モデル2つ（それぞれUNet、OpenCLIP ViT-H/14、SD VAE）と0.12Bの再構成UNetで構成されます。約12.8 GBで、BiRefNetにさらに0.4 GBが必要です。

## エクスポート

<export-matrix />

LibreGenの3Dファミリーではエクスポートは実装されていません。`export()`、`train()`、`val()`は`NotImplementedError`を発生させます（ライブラリ内のADR 0031）。

## ライセンス

<provenance-box>

CRMは清華大学（TSAIL）によるものです（[thu-ml/CRM](https://github.com/thu-ml/CRM)）。コードはMITで、モデルカードは重みについてもMITと宣言しています。

**重みには利用制限があります。** CRMの2つの拡散モデルはImageDream、MVDream、Stable Diffusion 2.1の系譜にあるため、LibreYOLOはそれらをその系譜のCreativeML OpenRAIL++-M Licenseの下にあるものとして扱います。商用利用は許可されていますが、その利用に基づく制限（違法、有害、差別的、欺瞞的な利用の禁止など）が利用者を拘束し、モデルや派生物を共有する相手にも引き継ぐ必要があります。

`Zhengyi/CRM`の重みは変更なしでLibreYOLO/LibreCRMとして再配布されており、そのカードにはOpenRAIL++-Mの条項が記載されています。FlexiCubesはNVIDIAのApache-2.0のリリースに由来します。アップストリームが呼び出すnvdiffrastとGPL-3.0のメッシュツールは使用しません。

</provenance-box>

## 引用

<citation-block />
