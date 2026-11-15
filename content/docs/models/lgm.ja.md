---
title: LGM
families: [lgm]
seo_title: "PythonでLGM：生成した4つのビューを経て1枚の画像から3Dガウシアンへ"
description: "LGMをLibreYOLOで実行します。ImageDreamが1枚の写真から物体の4つのビューをサンプリングし、LGMがそれらを3Dガウシアンに変換します。モデルはMITで、ImageDreamにはOpenRAILの利用制限があります。"
lead: "LGM（Large Multi-View Gaussian Model）は、物体の写真1枚を2つのステージで3Dガウシアンに変換します。ImageDreamが物体の周囲の4つのビューをサンプリングし、LGMがそれらを読み取ってガウシアンを返します。LibreYOLOはLibreGenファクトリーを通じて、両方のステージを純粋なPyTorchで実行します。"
keywords: [LGM, Large Multi-View Gaussian Model, ImageDream, 画像から3D, image to 3d, 3d gaussian splatting, ガウシアンスプラッティング, マルチビュー 拡散モデル, PLY, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("lgm")             # 初回使用時に重みをダウンロード
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.ply")           # 標準の 3DGS レイアウト
        print(len(result.gaussians))        # 不透明度 0.005 を超えるもの
    - label: CLI
      language: bash
      code: |
        # runs/generate/predict に物体を .ply として保存
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=lgm seed=0
    - label: 設定
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("lgm")
        result = model(SAMPLE_IMAGE, elevation=-10)     # 上から撮影した写真
        result = model(SAMPLE_IMAGE, steps=20, guidance=1.0)   # ガイダンスなし
source_hash: 90133179ce28932d
---

## インストール

```bash
pip install "libreyolo[gen]"
```

このextraには`diffusers`と`transformers`が含まれ、ImageDreamのVAE、スケジューラー、CLIPエンコーダーを読み込みます。

## 推論

アクセス制限はなく、ログインも不要です。重みは初回使用時に、固定されたリビジョンでダウンロードされます。

<code-tabs name="predict" />

Stable Diffusion 2.1からファインチューニングされたマルチビュー拡散モデルであるImageDreamが、256 pxの入力から物体の周囲の4つのビューをサンプリングします。クロスビューアテンションを持つ非対称UNetであるLGMが、4つのビューをカメラレイとともに読み取り、各ビューの出力ピクセルごとに1つのガウシアンを返します。4 x 128 x 128 = 65,536個のガウシアンのうち、アップストリームの保存方法と同じく、不透明度0.005を超えるものが残されます。

アップストリームはCUDA上のxformersを必要とし、非商用のラスタライザーでレンダリングします。LibreYOLOはアテンションをPyTorchのSDPAで実行し、レンダリングは行わないため、モデルはNVIDIA GPU、Apple silicon、CPU上で動作し、GPUではfp16、CPUではfp32を使います。CUDAではシードがアップストリームの物体を再現します。他のデバイスでは同じシードでも別のノイズが生成されます。

出力は`result.gaussians`で、[Speridlabs Idol](/docs/models/speridlabs-idol)と同じペイロードであり、`.ply`または`.splat`として保存されます。

### 設定

| 設定 | デフォルト | 意味 |
| --- | --- | --- |
| `steps` | `30` | ImageDreamのDDIMサンプリングステップ数。 |
| `guidance` | `5.0` | classifier-free guidance。1で無効。 |
| `elevation` | `0` | 写真のカメラの高さ（度）。負の値は物体の上方。 |

## バリアント

サイズは`0.4b`の1つです。0.42BのLGMネットワーク（約0.8 GB）に加え、1.3BのImageDreamマルチビューステージとその2つのCLIPエンコーダー（約4.0 GB）で構成されます。

## エクスポート

<export-matrix />

LibreGenの3Dファミリーではエクスポートは実装されていません。`export()`、`train()`、`val()`は`NotImplementedError`を発生させます（ライブラリ内のADR 0031）。

## ライセンス

<provenance-box>

LGMは3D TopiaのJiaxiang Tangらによるものです（[3DTopia/LGM](https://github.com/3DTopia/LGM)）。2つのステージには異なるライセンスが適用されます。

- LGMのガウシアンモデルとそのコードはMITです。
- **ImageDreamには利用制限があります。** アップストリームではOpenRAILと宣言されており、LibreYOLOはそのStable Diffusion 2.1の系譜のCreativeML OpenRAIL++-M Licenseを適用します。商用利用は許可されていますが、その利用に基づく制限（違法、有害、差別的、欺瞞的な利用の禁止など）が利用者を拘束し、モデルや派生物とともに引き継ぐ必要があります。

重みは固定されたリビジョンで変更なしに再配布されています。`ashawkey/LGM`はBiRefNet（MIT）とともにLibreYOLO/LibreLGM0.4bとして、`ashawkey/imagedream-ipmv-diffusers`は[LibreYOLO/LibreImageDream](https://huggingface.co/LibreYOLO/LibreImageDream)として再配布されています。どちらもアクセス制限はありません。Inriaの非商用ラスタライザーを必要とするアップストリームのレンダラーは含まれていません。

</provenance-box>

## 引用

<citation-block />
