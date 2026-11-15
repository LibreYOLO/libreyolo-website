---
title: DeepSeek-OCR
families: [deepseekocr]
seo_title: "PythonでDeepSeek-OCRとDeepSeek-OCR-2：ドキュメントOCRからMarkdownへ"
description: "LibreYOLOでDeepSeek-OCRとDeepSeek-OCR-2を実行します。ボックス付きのテキスト行、またはページ全体をMarkdownとして取得できます。MITとApache-2.0、リモートコードなし。"
lead: "DeepSeek-OCRは、DeepSeekによるドキュメント向けの3Bの混合エキスパート（mixture-of-experts）モデルです。LibreYOLOはDeepSeek-OCRとDeepSeek-OCR-2を、同梱のPyTorch移植版を通じてocrタスクで実行します。ボックス付きのテキスト行、レイアウトブロック、またはページ全体をMarkdownとして出力します。"
keywords: [DeepSeek-OCR, DeepSeek-OCR-2, ドキュメント OCR, OCR markdown 変換, 視覚言語モデル, mixture of experts, LibreVLM]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("deepseek-ocr")      # 初回使用時に重みをダウンロード
        result = model.predict("page.png")     # スキャンまたはレンダリングしたページ

        for text, quad in zip(result.ocr.texts, result.ocr.polygons):
            print(text, quad.tolist())         # テキスト行ごとに1領域
    - label: Markdown
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("deepseek-ocr", prompt="markdown")
        result = model.predict("page.png")

        print(result.ocr.markdown)             # ページ全体をMarkdownで
        print(result.ocr.labels)               # ブロックごとのレイアウトカテゴリ
    - label: DeepSeek-OCR-2
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("deepseek-ocr-2")     # 解像度は "gundam" または "base"
        result = model.predict("page.png")
        print(result.ocr.texts)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("deepseek-ocr")
        metrics = model.val(data="ocr_dataset/")   # 検出hmean、エンドツーエンドF1、1-NED
source_hash: 51211ee8cf74b682
---

## インストール

DeepSeek-OCRはLibreYOLOのVLM層に属し、`vlm` extraが必要です。

```bash
pip install "libreyolo[vlm]"
```

## 推論

重みは初回使用時に、リビジョンを固定したバイト単位で同一のLibreYOLOミラー`LibreYOLO/LibreDeepSeekOCR1`および`LibreYOLO/LibreDeepSeekOCR2`からダウンロードされます。アーキテクチャはリリースの推論経路をPyTorchに移植したものです。リモートコードは実行されません。

<code-tabs name="predict" />

このファミリーは`LibreVLM()`、または`LibreDeepSeekOCR`クラスで読み込みます。`prompt`の設定（コンストラクターまたは`set_prompt()`）で、`Results.ocr`に入る内容が決まります。

| `prompt` | `Results.ocr` |
| --- | --- |
| `"ocr"`（デフォルト） | テキスト行ごとに1領域、`labels`はすべて`text`、`markdown`は空 |
| `"markdown"` | レイアウトブロックごとに1領域、`labels`にレイアウトカテゴリ、`markdown`にページ |
| `"free"` | 領域なし、`markdown`にページのテキスト |
| その他の文字列 | モデルが出力するグラウンディングと`markdown` |

ボックスはモデルの0〜999のグリッドで出力され、元画像に合わせて軸に平行な4点の四角形にスケーリングされ、上から下、次に左から右の順に並べられます。生成モデルはスコアを出力しないため、`conf`と`det_conf`は定数1.0です。`model.chat("page.png", "Parse the figure.")`は任意のプロンプトに対して生成された生のテキストを返します。デコードは貪欲法で、新規トークンは最大8192、リリースと同じく`no_repeat_ngram_size=20`を使います。

## バリアント

| 名前 | モデル | 解像度 | ライセンス |
| --- | --- | --- | --- |
| `deepseek-ocr` | DeepSeek-OCR（サイズ`1`） | `tiny`（512）、`small`（640）、`base`（1024）、`large`（1280）、`gundam`（デフォルト） | MIT |
| `deepseek-ocr-2` | DeepSeek-OCR-2（サイズ`2`） | `base`（1024）、`gundam`（デフォルト） | Apache-2.0 |

`gundam`は1024 pxのグローバルビューとローカルクロップを組み合わせます。DeepSeek-OCRでは640 pxのクロップが2〜9枚、DeepSeek-OCR-2では768 pxのクロップが最大6枚です。DeepSeek-OCR-2はデコーダーとSAMステージを維持し、CLIPをQwen2エンコーダーに置き換えています。プロンプトと出力は同じです。

CUDAでは、リリースと同じく`torch.autocast`の下でbf16の重みを実行します（bf16非対応のGPUではfp16）。CPUとMPSではeager attentionでfloat32を実行し、約13 GBのメモリを使います。MPSは実機で未検証です。EXIFの向きは適用されません。RTX 3090上で、この移植版はDeepSeek-OCRの17ケースとDeepSeek-OCR-2の14ケースでリリースをトークン単位で再現します。ベンチマークの数値はリリースの論文にあり、LibreYOLOでは再現していません。

## 検証

`val()`は共有のOCRバリデーターを実行し、検出hmean、エンドツーエンドF1、1-NEDを報告します。

<code-tabs name="val" />

## エクスポート

推論のみです。`train()`と`export()`は例外を発生させます。

## ライセンス

<provenance-box>

DeepSeek-OCRはMIT、DeepSeek-OCR-2はApache-2.0で、コードと重みのいずれも[DeepSeek](https://github.com/deepseek-ai/DeepSeek-OCR)によるものです。どちらも商用利用が可能です。重みはバイト単位で同一の、ゲートされていないミラーです。

</provenance-box>

## 引用

<citation-block />
