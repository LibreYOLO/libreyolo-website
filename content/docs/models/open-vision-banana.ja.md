---
title: Open Vision Banana
families: [open_vision_banana]
seo_title: "PythonでOpen Vision Banana：プロンプトからセグメンテーション、深度、法線"
description: "LibreYOLOでOpen Vision Bananaを実行します。知覚タスクに画像で答えるFLUX.2 [klein]のファインチューニング版です。セグメンテーション、深度、法線、画像編集、テキストからの画像生成。非商用の重み。"
lead: "Open Vision Banana（OVB）は、FunAILabがFLUX.2 [klein] Baseを知覚タスクに画像で答えるようにファインチューニングしたモデルです。プロンプトがタスクを指定し、出力画像が答えを表します。セグメンテーションは平坦な色、深度はグレー、表面法線はRGBです。同じ重みで画像編集とテキストからの画像生成も維持しています。LibreYOLOはLibreGenファクトリーを通じて4Bと9Bで実行します。"
keywords: [Open Vision Banana, OVB, FunAILab, FLUX.2 klein, 生成モデル 知覚タスク, セマンティックセグメンテーション, 深度推定, 表面法線 推定, 画像編集 python, LibreGen]
last_verified: "1.8.0"
snippets:
  predict:
    - label: 深度
      language: python
      code: |
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("open-vision-banana")      # 4b、9bは"open-vision-banana-9b"
        DEPTH = (
            "Estimate a dense relative inverse-depth map for the input image. Produce exactly one output value "
            "for every input pixel at the same pixel location, with closer points brighter and farther points "
            "darker. Encode the result as neutral grayscale (R=G=B) using one consistent mapping over the "
            "complete image. Preserve object silhouettes, thin structures, occlusion boundaries, smooth surfaces, "
            "and small depth changes. Do not alter the scene or add texture, shading, color, text, labels, "
            "legends, borders, or objects; return only the registered depth map."
        )
        result = model(DEPTH, source=SAMPLE_IMAGE, seed=0)
        result.generated.save("depth.png")
        rgb = result.generated.array                # (H, W, 3) uint8
    - label: セマンティックセグメンテーション
      language: python
      code: |
        import json
        from libreyolo import LibreGen, SAMPLE_IMAGE

        model = LibreGen("open-vision-banana")
        legend = {"building": [255, 8, 232], "sky": [2, 204, 254], "water": [8, 4, 249]}
        prompt = (
            "[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of the input image, "
            f"using this color mapping: {json.dumps(legend, separators=(', ', ': '))} "
            "Color each pixel of a listed class with its color and leave everything else black. "
            "Output only the segmentation visualization image."
        )
        model(prompt, source=SAMPLE_IMAGE, seed=0).generated.save("segmentation.png")
    - label: テキストから画像
      language: python
      code: |
        from libreyolo import LibreGen

        model = LibreGen("open-vision-banana")
        model("an empty tram at a cobblestone stop on a rainy night", imgsz=1024, seed=0).generated.save("tram.png")
    - label: CLI
      language: bash
      code: |
        # CLIはテキストからの画像生成のみに対応
        libreyolo generate model=open-vision-banana prompt="an empty tram at a cobblestone stop on a rainy night"
source_hash: f2cae41110f376c1
---

## インストール

```bash
pip install "libreyolo[gen]"
```

## 推論

ミラーは初回使用時に一度だけダウンロードされます。LibreYOLOはdiffusersの`Flux2KleinPipeline`でOVBを読み込み、ネットワークのコードは追加しません。書くのはプロンプトと、色を読み戻すための数行のNumPyだけです。

<code-tabs name="predict" />

`source`にはパス、URL、`PIL.Image`、BGRのNumPy配列を指定できます。出力は写真のアスペクト比を維持し、上限は1メガピクセルです。読み取る前に写真のサイズへリサイズし直してください（マスクは色を正確に保つため最近傍補間を使います）。`source`を指定しなければ、通常のテキストから画像を生成するモデルになります。調整項目は論文の設定である`steps`（28）と`guidance`（3.5）です。`offload=True`を指定すると、使っていない部分をGPUから退避させます。

このモデルは各タスクについて多くの言い回しで学習されています。ライブラリの[Open Vision Bananaガイド](https://github.com/LibreYOLO/libreyolo/blob/v1.8.0/docs/open_vision_banana.md)には、セマンティック・参照・インスタンスセグメンテーション、深度、表面法線、画像編集についてのFunAILabのクイックスタートのプロンプトと読み取り方法、さらに凡例のアンサンブル（色をシャッフルした複数回の実行とピクセルごとの投票）が載っています。

想定される挙動：すべての答えが28ステップの拡散処理であるため、低速です。速度よりも1つのモデルで多くのタスクをこなすことが重要な場合に使ってください。マスクはリサイズ後に柔らかく、わずかにずれた輪郭になります。深度は未知のスケールとシフトを持つ相対逆深度で、法線は滑らかですがメートル単位ではありません。

## バリアント

| サイズ | エイリアス | ミラー | テキストエンコーダー | GPUメモリ（ライブラリの数値） |
| --- | --- | --- | --- | --- |
| `4b` | `open-vision-banana` | `LibreYOLO/LibreOpenVisionBanana4b` | Qwen3 4B | 約16 GB |
| `9b` | `open-vision-banana-9b` | `LibreYOLO/LibreOpenVisionBanana9b` | Qwen3 8B | 約34 GB |

どちらのミラーも、ベンチマークの学習分割で学習した論文の非ゼロショットのチェックポイントを保持しています。diffusersのパイプラインとして保存された別のOVBチェックポイント（ゼロショットのもの）は、`LibreGen("open-vision-banana", checkpoint_dir="path/to/folder")`で読み込めます。

## エクスポート

<export-matrix />

学習、エクスポート、`val()`には対応していません。

## ライセンス

<provenance-box>

Open Vision BananaはFunAILabによるものです（[OpenVisionBanana/OpenVisionBanana](https://github.com/OpenVisionBanana/OpenVisionBanana)）。重みはBlack Forest Labsの**FLUX Non-Commercial License**の下にあります。商用利用や本番利用はできず、第2条(e)は生成画像を共有する前にコンテンツフィルタリングまたは人によるレビューを求めています。FunAILabの学習コードはApache-2.0ですが、LibreYOLOはそのいずれも移植していません。

LibreYOLOのミラーは完全なdiffusersのパイプラインで、ゲートされていません。FunAILabのTransformerは、実行時の精度であるbfloat16へfloat32からキャストしており、BFLのFLUX.2 [klein] Baseパイプラインの残りはバイト単位で同一です。出力は、同じシードでFunAILab自身のクイックスタートとビット単位で一致します。

</provenance-box>

## 引用

<citation-block />
