---
title: TRELLIS
families: [trellis]
seo_title: "PythonでTRELLIS：1枚の画像から3Dガウシアンへ、CUDA拡張は不要"
description: "MicrosoftのTRELLISをLibreYOLOで実行します。物体の写真1枚を入力すると、疎なボクセル構造上の3Dガウシアンとして物体が出力されます。CUDA、MPS、CPU上の純粋なPyTorchで動作し、MITライセンスです。"
lead: "TRELLISはMicrosoftのオープンな画像から3Dを生成するモデルです。まず物体を疎なボクセル構造として構築し、次に占有された各ボクセルに32個のガウシアンを割り当てます。アップストリームはCUDA専用の拡張を必要としますが、LibreYOLOはLibreGenファクトリーを通じて、同じ重みを純粋なPyTorchで実行します。"
keywords: [TRELLIS, Microsoft, structured 3D latents, 画像から3D, image to 3d, 3d gaussian splatting, ガウシアンスプラッティング, 疎 ボクセル, PLY, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("trellis")         # 初回使用時に重みをダウンロード
        result = model(SAMPLE_IMAGE, seed=0)

        result.save("object.ply")           # 標準の 3DGS レイアウト
        result.save("object.splat")         # コンパクトな Web 形式
        print(len(result.gaussians))        # 設定値ではなく物体次第
    - label: CLI
      language: bash
      code: |
        # runs/generate/predict に物体を .ply として保存
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg model=trellis seed=0
    - label: 設定
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("trellis")
        result = model(SAMPLE_IMAGE, steps=12)          # 高速なサンプリング
        result = model(SAMPLE_IMAGE, guidance=3.0, margin=1.3)
source_hash: 1b2b957c271c561e
---

## インストール

```bash
pip install "libreyolo[gen]"
```

このextraには`transformers`が含まれ、DINOv2画像エンコーダーを読み込みます。

## 推論

アクセス制限はなく、ログインも不要です。重みは初回使用時に、固定されたリビジョンでダウンロードされます。

<code-tabs name="predict" />

密なフロートランスフォーマーが16^3で物体の占有をデノイズし、3D CNNがそれを64^3グリッドのアクティブなボクセルへデコードし、疎なフロートランスフォーマーがそれらのボクセル上で構造化された潜在表現をデノイズし、疎なトランスフォーマーが各ボクセルを32個のガウシアンに変換します。両方のフローステージは518 pxの入力のDINOv2トークンを読み取ります。そのため、ガウシアンの数は物体によって決まります（多くの場合200,000〜600,000）。

アップストリームの疎な畳み込み（spconv）とアテンション（xformersまたはflash-attn）はCUDA上でのみ動作します。LibreYOLOはそれらをPyTorchの演算に置き換え、パラメータ名を維持しているため、公開済みの重みはNVIDIA GPU、Apple silicon、CPU上でそのまま読み込めます。ノイズはアップストリームと同じくCPU上で生成されるため、シードはどのデバイスでも同じ物体を与えます。

出力は`result.gaussians`で、[Speridlabs Idol](/docs/models/speridlabs-idol)と同じペイロードです。z-upで`[-0.5, 0.5]^3`の範囲にある`xyz`、`scales`、`rotations`、`opacities`、`sh`、`colors`、`seed`、`params`を持ちます。背景除去にはLibreYOLOのBiRefNetの移植版を使います。

### 設定

| 設定 | デフォルト | 意味 |
| --- | --- | --- |
| `steps` | `25` | 両方のフローステージのサンプリングステップ数。 |
| `guidance` | `5.0` | 両方のフローステージのclassifier-free guidance。 |
| `shift` | `3.0` | フローの時間シフト。 |
| `margin` | `1.2` | 物体サイズに対する切り抜きの辺の比率（アップストリームと同じ）。 |

## バリアント

サイズは`l`（TRELLIS-image-large）の1つです。4つのネットワークで1.3Bパラメータ、約2.7 GBで、DINOv2 ViT-L/14エンコーダーにさらに1.2 GBが必要です。

## エクスポート

<export-matrix />

LibreGenの3Dファミリーではエクスポートは実装されていません。`export()`、`train()`、`val()`は`NotImplementedError`を発生させます（ライブラリ内のADR 0031）。

## ライセンス

<provenance-box>

TRELLISはMicrosoft Researchによるものです（[microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)）。コードと`microsoft/TRELLIS-image-large`の重みはMITであるため、表示を残せば商用利用できます。LibreYOLOは重みを変更なしで、BiRefNetの重み（MIT）とともにLibreYOLO/LibreTRELLISlとして再配布しています。DINOv2エンコーダーの`facebook/dinov2-with-registers-large`はApache-2.0で、Metaからダウンロードされます。InriaのGaussian Splatting非商用ライセンスの下にあるアップストリームのガウシアン保持クラスは使用しません。

</provenance-box>

## 引用

<citation-block />
