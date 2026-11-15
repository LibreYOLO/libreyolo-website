---
title: PRX
families: [prx]
seo_title: "PythonでPRXとPRX Pixel：Photoroomのオープンなテキストから画像生成モデル"
description: "LibreYOLOのLibreGenでPhotoroomのテキストから画像を生成するモデルPRXとPRX Pixelを実行します。256、512、1024の9つのPRXチェックポイントと7BのPRX Pixel。Apache-2.0。"
lead: "PRXは、Photoroomによるオープンソースのテキストから画像を生成するモデルで、256、512、1024 pxの9つのチェックポイントとして公開された1.3BのTransformerです。PRX Pixelは、オートエンコーダーを使わずにピクセルを直接ノイズ除去する7Bの兄弟モデルです。LibreYOLOは10種類すべてを、diffusersのパイプライン上でLibreGenファクトリーを通じて実行します。"
keywords: [PRX, PRX Pixel, PRXPixel, Photoroom, テキストから画像 生成, 画像生成 python, 拡散モデル, DC-AE, T5Gemma, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("prx-512-sft")
        result = model("a red fox sleeping in fresh snow", seed=0)
        result.save("fox.png")             # PNGにはプロンプト、シード、パラメータも保存
    - label: 設定
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("prx-512-sft-distilled")          # 8ステップの蒸留版
        result = model("a red fox", steps=8, guidance=1.0)

        model = LibreGen("prx")                            # prx-1024-beta
        result = model("a red fox", negative="blurry", imgsz=1024, seed=0)

        model = LibreGen("prxpixel", offload=True)         # prxpixel-7b、diffusers>=0.39が必要
        results = model(["a red fox", "a blue jay"], n=2, save=True)
    - label: CLI
      language: bash
      code: |
        libreyolo generate prompt="a red fox" model=prx-512-sft negative="blurry"
        libreyolo generate prompt="a red fox" model=prxpixel-7b seed=0 n=4
source_hash: 5eeb7c71d04d09f8
---

## インストール

```bash
pip install "libreyolo[gen]"
```

PRX Pixelには`diffusers>=0.39`が必要で、読み込み時にその旨を表示します。extraの下限は、Marigoldのextraと共存できるよう0.38になっています。

## 生成

重みは初回使用時に、Photoroomのリポジトリをバイト単位で複製したLibreYOLOのミラーからダウンロードされます。

<code-tabs name="predict" />

文字列は常にプロンプトです。1つのプロンプトで`n=1`なら`Results`を返し、プロンプトのリストまたは`n > 1`ならフラットなリストを、`stream=True`ならジェネレーターを返します。`result.generated`は画像（`array`、`image`）、`prompt`、それを再現する`seed`、適用された`params`を保持します。ファミリーの調整項目はすべて省略可能です。ファミリーにない項目を指定すると例外が発生し、対応する項目が一覧表示されます。プロンプト、シード、デバイス、オフロードについては[画像生成](/docs/tasks/image-generation)のタスクページを参照してください。

## バリアント

| モデル | パラメータ数 | 調整項目（デフォルト） | GPUメモリ（ライブラリの数値） |
| --- | --- | --- | --- |
| `prx-1024-beta`（エイリアス`prx`） | 1.3B | `steps`（28）、`guidance`（5.0）、`negative`（""） | 約21 GB |
| `prx-512-sft-distilled` | 1.3B | `steps`（8）、`guidance`（1.0）、`negative`（""） | 約16 GB |
| `prx-512-sft`、`prx-512` | 1.3B | `steps`（28）、`guidance`（5.0）、`negative`（""） | 約16 GB |
| `prx-512-dc-ae-sft-distilled` | 1.3B | `steps`（8）、`guidance`（1.0）、`negative`（""） | 約16 GB |
| `prx-512-dc-ae-sft`、`prx-512-dc-ae` | 1.3B | `steps`（28）、`guidance`（5.0）、`negative`（""） | 約16 GB |
| `prx-256-sft`、`prx-256` | 1.3B | `steps`（28）、`guidance`（5.0）、`negative`（""） | 約16 GB |
| `prxpixel-7b`（エイリアス`prxpixel`） | 7B | `steps`（28）、`guidance`（5.0）、`negative`（""） | 約18 GB |

数字はネイティブ解像度です。`-dc-ae`はFLUX.1のVAEの代わりに32xのDC-AEオートエンコーダーを使うため、辺の長さは16ではなく32の倍数に切り上げられます。`-sft`はそのまま使える教師ありファインチューニング版で、`-sft-distilled`はその8ステップの蒸留版です。`-sft`の付かないチェックポイントは、ファインチューニング用のベースモデルです。`prx-1024-beta`はPhotoroomの1024のプレビュー版です。PRX PixelはQwen3-VLのテキストエンコーダーを同梱し、オートエンコーダーを持ちません。

## エクスポート

<export-matrix />

すべてのLibreGenファミリーと同じく、`export()`、`train()`、`val()`は例外を発生させます。

## ライセンス

<provenance-box>

PRXとPRX Pixelは[Photoroom](https://github.com/Photoroom/PRX)によるものです。重みはApache-2.0のため、商用利用が可能です。例外はPRXに同梱のT5Gemma 2Bテキストエンコーダーで、Googleの**Gemma Terms of Use**とProhibited Use Policyの下にあります。商用利用は可能ですが、禁止用途は共有先にも引き継がれ、PRXのモデルカードはモデル全体についてそれらを禁止しています。オートエンコーダーはFLUX.1 [schnell]のVAE（Apache-2.0）またはDC-AE（MIT）です。PRX PixelのQwen3-VL-Embedding-2BテキストエンコーダーはApache-2.0です。

LibreYOLOのアダプターは、Apache-2.0のdiffusersの`PRXPipeline`と`PRXPixelPipeline`の上に書いた独自のMITのコードです。重みは`LibreYOLO/LibrePRX<size>`と`LibreYOLO/LibrePRXPixel7b`からダウンロードされます。これらはコミットを固定したバイト単位で同一のミラーで、ゲートされていません。

</provenance-box>

## 引用

<citation-block />
