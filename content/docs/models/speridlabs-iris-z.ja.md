---
title: Speridlabs Iris-Z
families: [speridlabs_irisz]
seo_title: "PythonでSperidlabs Iris-Z：画像と深度を1つのモデルで"
description: "LibreYOLOでSperidlabs Iris-Zを実行します。FLUX.2ベースの1組の重みで、テキストから画像、画像から深度、深度から画像、画像と深度の同時生成を行います。非商用の重み。"
lead: "Speridlabs Iris-Zは、画像とその深度を1つのシーケンスでノイズ除去するようにファインチューニングしたFLUX.2です。同じ重みで、プロンプトから画像を生成し、写真から4ステップで深度を推定し、与えた深度にプロンプトの内容を描き、画像とその深度を一緒に返します。LibreYOLOはLibreGenファクトリーを通じて2つのサイズで実行します。"
keywords: [Speridlabs Iris-Z, Iris-Z, Speridlabs, FLUX.2, 深度推定, テキストから画像 生成, depth to image, RGB-D 生成, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: 画像から深度
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("speridlabs-iris-z", task="depth")   # 9b、テキストエンコーダーは読み込まない
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map
        print(depth.encoding)          # "log_depth"、大きいほど遠い
        print(depth.data.shape)        # 写真のサイズで(H, W)
        print(depth.valid_mask.shape)  # 空はFalse
        result.save("depth.png")       # カラー化した深度
    - label: テキストから画像
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("speridlabs-iris-z")
        result = model("a lighthouse at dusk", seed=0)
        result.generated.save("lighthouse.png")
    - label: 深度から画像
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("speridlabs-iris-z")
        # 先に写真の深度を推定し、その上に新しいシーンを描く
        result = model("a sunken ruin under the sea", source=SAMPLE_IMAGE, seed=0)
        result.generated.save("ruin.png")
    - label: 画像と深度
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("speridlabs-iris-z")
        result = model("a grand old library, a spiral staircase", depth=True, seed=0)
        result.generated.save("library.png")
        print(result.depth_map.data.shape)
    - label: CLI
      language: bash
      code: |
        libreyolo generate prompt="a lighthouse at dusk" model=speridlabs-iris-z
        libreyolo generate source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg task=depth
        libreyolo generate source=photos/ task=depth ensemble=10 imgsz=1024
source_hash: c54a3e5969e7642e
---

## インストール

```bash
pip install "libreyolo[gen]"
```

## 推論

タスクはモデルの読み込み時に固定されます。画像から深度には`task="depth"`、ほかの3つのモードにはデフォルトの`generate`を使います。深度モデルはテキストエンコーダーを読み込みません。

<code-tabs name="predict" />

**画像から深度。** ソースはpredictが受け付けるものです。`result.depth_map`は`encoding="log_depth"`（相対対数深度、大きいほど遠い）の`DepthMap`でソースのサイズを持ち、`valid_mask`は空の部分で`False`です。`imgsz`は作業解像度で、512または1024です（0.5メガピクセル未満なら512、それ以外は1024がデフォルト）。調整項目は`steps`（4）と`ensemble`（1）で、後者はその数のシードにわたるピクセルごとの中央値です。デフォルトのシードは0のため、深度は再現されます。`save=True`はカラー化したマップを`runs/depth/predict`の下に書き出します。

**テキストから画像。** プロンプト、`imgsz`、`seed`、`n`は、すべての[画像生成](/docs/tasks/image-generation)モデルと同じように動作します。調整項目は`steps`（50）と`guidance`（`9b`は4.0、`32b`は3.5）です。

**深度から画像。** generateタスクの`source=`は、画像を描く土台となる深度です。写真（先にその深度を推定します）、任意の深度モデルの`Results`または`DepthMap`、あるいは`depth_format="depth"`（明るいほど遠い）または`"disparity"`（明るいほど近い）を指定したグレースケールの深度画像を指定できます。画像は`depth_format`で指定した場合にのみ深度マップとして読まれます。出力は深度のアスペクト比を維持し、`Results.depth_map`は描画の土台にした深度です。

**画像と深度。** `depth=True`は画像と深度を同時にサンプリングし、アップストリームのデフォルトと同じく、完成した画像から深度を推定し直します。結果は`result.generated`と`result.depth_map`の両方を持ちます。

CLIはテキストから画像と画像から深度に対応します。CLIでは`prompt=`と`source=`は引き続き同時に指定できません。

## バリアント

| サイズ | エイリアス | パラメータ数 | テキストエンコーダー | GPUメモリ（ライブラリの数値） |
| --- | --- | --- | --- | --- |
| `9b` | `speridlabs-iris-z`、`-9b`、`-small` | 9.1B、FLUX.2 [klein] Base 9Bから | Qwen3 8B FP8 | 約18 GB、加えてテキストエンコーダー9 GBとVAE 0.2 GB |
| `32b` | `speridlabs-iris-z-32b`、`-big` | 32.2B、FLUX.2 [dev]から | Mistral Small 3.2 24B | 約64 GB、加えてテキストエンコーダー48 GBとVAE 0.2 GB |

`offload=True`を指定すると、プロンプトの合間にテキストエンコーダーをCPUに置いたままにします。transformersが求めるFP8カーネル（`kernels`パッケージ）のないCUDA、およびMPSやCPUでは、`9b`のテキストエンコーダーは逆量子化してbfloat16で動作し、アップストリームの埋め込みベクトルに近いものの同一にはなりません。ローカルのスナップショットやバリアントのフォルダーは`LibreGen(path)`で読み込めます。

## エクスポート

<export-matrix />

すべてのLibreGenファミリーと同じく、`export()`、`train()`、`val()`は例外を発生させます。

## ライセンス

<provenance-box>

Speridlabs Iris-Zは[Speridlabs](https://speridlabs.com/research/iris-z)によるものです。重みはBlack Forest Labsの**FLUX Non-Commercial License**の下にあります。商用利用や本番利用はできず、第2条(e)は生成画像を共有する前にコンテンツフィルタリングまたは人によるレビューを求めています。LibreYOLOのアダプターはMITです。[Iris-Z SDK](https://github.com/speridlabs/iris-z-sdk)から移植したTransformer、サンプラー、プロンプト埋め込み器はApache-2.0のままです。

LibreYOLOは重みを一切再配布しません。重みは実行時に、リビジョンを固定した`speridlabs/iris-z`から変更なしでダウンロードされ、ゲートされていません。FLUX.2のVAEはApache-2.0の`black-forest-labs/FLUX.2-klein-4B`リポジトリから取得し、テキストエンコーダー（`9b`は`Qwen/Qwen3-8B-FP8`、`32b`は`mistralai/Mistral-Small-3.2-24B-Instruct-2506`）はApache-2.0です。

</provenance-box>

## 引用

<citation-block />
