---
title: TripoSR
families: [triposr]
seo_title: "PythonでTripoSR：1枚の画像から約1秒で3Dメッシュへ"
description: "TripoSRをLibreYOLOで実行します。物体の写真1枚を入力すると、頂点カラー付きの三角形メッシュが.objとして出力されます。フィードフォワードでサンプリングは不要、コードと重みはMITです。"
lead: "TripoSRはTripo AIとStability AIによるフィードフォワード型の画像から3Dを生成するモデルです。物体の写真1枚を1回のパスでトライプレーンNeRFに変換し、物体を頂点カラー付きの三角形メッシュとして返します。LibreYOLOはLibreGenファクトリーを通じて、純粋なPyTorchでこれを実行します。"
keywords: [TripoSR, Tripo AI, Stability AI, 画像から3D, image to 3d, 1枚の画像 3d, メッシュ生成, triplane nerf, OBJ, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("triposr")        # 初回使用時に重みをダウンロード
        result = model(SAMPLE_IMAGE)

        result.save("object.obj")          # 頂点カラーは `v x y z r g b` 形式
        m = result.meshes
        print(m.vertices[0].shape, m.faces[0].shape)   # (V, 3), (F, 3)
    - label: CLI
      language: bash
      code: |
        # runs/generate/predict に物体を .obj として保存
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=triposr
    - label: 設定
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("triposr")
        result = model(SAMPLE_IMAGE, resolution=384)    # より細かいマーチングキューブのグリッド
        result = model(SAMPLE_IMAGE, threshold=25.0)    # 表面での密度
source_hash: 6936728dac15269e
---

## インストール

TripoSRはLibreGenのextraで動作し、このextraにはマーチングキューブ用のscikit-imageが含まれます。

```bash
pip install "libreyolo[gen]"
```

## 推論

重みは初回使用時に[LibreYOLO/LibreTripoSR](https://huggingface.co/LibreYOLO/LibreTripoSR)からダウンロードされ、アクセス制限もログインもありません。

<code-tabs name="predict" />

DINO ViT-B/16が512 pxの入力をエンコードし、トランスフォーマーがそこからトライプレーンNeRFを1回のパスでデコードし、メッシュはNeRFの密度のマーチングキューブ等値面になります。サンプリングがないため、`seed`は他のLibreGenファミリーと同様に受け付けますが、何も変わりません。GPUでは約1秒で、NVIDIA GPU、Apple silicon、CPU上でfp32で動作します。

ソースには、predictが受け付けるものなら何でも使えます：パス、URL、PIL画像、NumPy配列、フォルダー、リストです。実際の透明度を持つ画像はそのまま切り抜きとして使われます。それ以外は、まずLibreYOLOのBiRefNetの移植版で背景が除去されます。

`result.meshes`は物体ごとに1つのメッシュを保持します：z-upで`[-0.5, 0.5]^3`の内側にある`vertices`（V, 3）、`faces`（F, 3）、`[0, 1]`のRGBである`extras["vertex_colors"]`（V, 3）です。入力ビューは-x方向を向いているため、物体の正面は+xを向きます。`save()`は頂点カラー付きの`.obj`を出力します。

### 設定

| 設定 | デフォルト | 意味 |
| --- | --- | --- |
| `resolution` | `256` | マーチングキューブのグリッド。32〜1024。 |
| `threshold` | `25.0` | 表面での密度。 |
| `margin` | `1/0.85` | 物体サイズに対する切り抜きの辺の比率（アップストリームの前景比率0.85）。 |

マーチングキューブはアップストリームのtorchmcubesではなくscikit-imageで実行されます。等値面は同じで、三角形分割が異なります。

## バリアント

チェックポイントは1つで、0.42Bパラメータ、約1.7 GBで、BiRefNetにさらに0.4 GBが必要です。

## エクスポート

<export-matrix />

LibreGenの3Dファミリーではエクスポートは実装されていません。`export()`、`train()`、`val()`は`NotImplementedError`を発生させます。これらの名前はライブラリ内のADR 0031で予約されています。

## ライセンス

<provenance-box>

TripoSRはTripo AIとStability AIによるものです。コード（[VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR)）と`stabilityai/TripoSR`の重みはMITであるため、著作権表示を残せば商用利用できます。LibreYOLOは重みを変更なしでLibreYOLO/LibreTripoSRとして再配布しています。アップストリームのtorchmcubes（MPL-2.0）を避けてscikit-image（BSD-3-Clause）を使っています。BiRefNetの背景除去の重みはMITです。

</provenance-box>

## 引用

<citation-block />
