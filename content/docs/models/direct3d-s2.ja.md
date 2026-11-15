---
title: Direct3D-S2
families: [direct3d_s2]
seo_title: "PythonでDirect3D-S2：1枚の画像から1024^3の詳細な3Dメッシュへ"
description: "DreamTechのDirect3D-S2をLibreYOLOで実行します。物体の写真1枚を入力すると、最大1024^3の疎な符号付き距離場から詳細なメッシュが出力されます。純粋なPyTorchで動作し、MITライセンスです。"
lead: "Direct3D-S2はDreamTechの画像から3Dを生成するモデルです。疎なボクセル上に符号付き距離場を、まず512^3で、次に1024^3で生成し、物体をメッシュとして返します。アップストリームはCUDA専用の拡張を必要としますが、LibreYOLOはLibreGenファクトリーを通じて、同じ重みを純粋なPyTorchで実行します。"
keywords: [Direct3D-S2, DreamTech, 画像から3D, image to 3d, メッシュ生成, 符号付き距離場 sdf, spatial sparse attention, OBJ, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("direct3d-s2")     # 初回使用時に重みをダウンロード
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.obj")
        m = result.meshes
        print(m.vertices[0].shape, m.faces[0].shape)
    - label: CLI
      language: bash
      code: |
        # runs/generate/predict に物体を .obj として保存
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=direct3d-s2 seed=0
    - label: 設定
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("direct3d-s2")
        result = model(SAMPLE_IMAGE, resolution=512)     # 1回目のパスで終了
        result = model(SAMPLE_IMAGE, steps=(25, 15, 8))  # ステージごとのステップ数を削減
source_hash: e90a338361a40563
---

## インストール

```bash
pip install "libreyolo[gen]"
```

このextraには`diffusers`、`transformers`（DINOv2エンコーダー）、scikit-image（マーチングキューブ）が含まれます。

## 推論

アクセス制限はなく、ログインも不要です。重みは初回使用時に、固定されたリビジョンでダウンロードされます。

<code-tabs name="predict" />

密なフロートランスフォーマーが占有された64^3のボクセルを見つけ、spatial sparse attentionを持つ疎なフロートランスフォーマーがそれらの上で潜在表現をデノイズし、疎なデコーダーがそれを512^3の符号付き距離場に変換します。2回目のパスでは同じ処理を1024^3で繰り返します。メッシュはマーチングキューブで得られる、距離場のゼロ交差です。アップストリームはCUDA上でのみ動作します（torchsparse、flash-attn、Tritonカーネル、CUDAボクセライザー）。LibreYOLOはそれらをPyTorchの演算に置き換え、パラメータ名を維持しているため、公開済みのチェックポイントはCUDA、Apple MPS、CPU上でそのまま読み込めます。

シードは同じ物体を与えます。CUDAでは、初期ノイズは`torch.manual_seed(seed)`に対するアップストリームのものと一致します。`result.meshes`は、z-upで`[-0.5, 0.5]^3`の範囲にある`vertices`（V, 3）と`faces`（F, 3）を保持します。アップストリームの省略可能なリメッシュと穴埋めは含まれないため（GPLのツールに依存するため）、メッシュはマーチングキューブが返したままの形で出力されます。

純粋なPyTorchでは低速です。ライブラリの計測では、RTX A6000で1024^3の物体1つに約18分と23 GBのGPUメモリを要し、アップストリームのCUDAカーネルでは2.5分です。`resolution=512`は1回目のパスで終了します。

### 設定

| 設定 | デフォルト | 意味 |
| --- | --- | --- |
| `resolution` | `1024` | 1024（アップストリームのデフォルト）または512。 |
| `steps` | `(50, 30, 15)` | 密、512、1024の各ステージのステップ数。整数1つで3つすべてを設定。 |
| `guidance` | `7.0` | 3つすべてのステージのclassifier-free guidance。 |

## バリアント

チェックポイントは1つで、`v11`（Direct3D-S2 v1.1）です。8つのネットワークで2.0Bパラメータ、約4.2 GBで、DINOv2 ViT-L/14にさらに1.2 GB、BiRefNetに0.4 GBが必要です。

## エクスポート

<export-matrix />

LibreGenの3Dファミリーではエクスポートは実装されていません。`export()`、`train()`、`val()`は`NotImplementedError`を発生させます（ライブラリ内のADR 0031）。

## ライセンス

<provenance-box>

Direct3D-S2はDreamTechによるものです（[DreamTechAI/Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2)）。コードと`wushuang98/Direct3D-S2`のv1.1の重みはMITであるため、表示を残せば商用利用できます。LibreYOLOは重みを変更なしでLibreYOLO/LibreDirect3DS2v11として再配布しています。BiRefNet（MIT）はLibreYOLO/LibreTRELLISlから、DINOv2エンコーダーの`facebook/dinov2-with-registers-large`（Apache-2.0）はMetaから取得します。アップストリームのGPLのpymeshfixとpython-igraphは使用しません。

</provenance-box>

## 引用

<citation-block />
