---
title: PaddleOCR-VL
families: [paddleocrvl]
seo_title: "PythonでPaddleOCR-VL：0.9BのVLMによるドキュメント解析"
description: "LibreYOLOでPaddleOCR-VLを実行します。PP-DocLayoutV2のレイアウトと、テキスト、表（HTML）、数式（LaTeX）、グラフを読む0.9Bの認識器。Apache-2.0。"
lead: "PaddleOCR-VLは、PaddlePaddleによる2段階のドキュメントパーサーです。PP-DocLayoutV2が25種類のレイアウト領域とその読み順を検出し、PaddleOCR-VL-0.9Bが各領域をテキスト、表、数式、グラフとして読み取ります。LibreYOLOは、両ネットワークを囲むPaddleXのパイプラインをocrタスク向けに移植しています。"
keywords: [PaddleOCR-VL, PaddleOCR, PaddlePaddle, ドキュメント解析, OCR markdown 変換, 表認識, 数式認識, ERNIE, LibreVLM]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("paddleocr-vl")       # 初回使用時に重みをダウンロード
        result = model.predict("page.png")      # スキャンまたはレンダリングしたページ

        for label, text in zip(result.ocr.labels, result.ocr.texts):
            print(label, text[:60])             # ブロックごとに1領域、読み順
        print(result.ocr.markdown)              # ページ全体をMarkdownで
    - label: 単一要素
      language: python
      code: |
        from libreyolo import LibreVLM

        # レイアウト検出を省略し、切り出し済みの数式を読み取る
        model = LibreVLM("paddleocr-vl", use_layout_detection=False, prompt_label="formula")
        result = model.predict("formula.png")
        print(result.ocr.texts[0])              # LaTeX
    - label: グラフ
      language: python
      code: |
        from libreyolo import LibreVLM

        # アップストリームと同じくグラフはデフォルトでスキップ、有効にすると表として読み取る
        model = LibreVLM("paddleocr-vl", use_chart_recognition=True)
        result = model.predict("report.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("paddleocr-vl")
        metrics = model.val(data="ocr_dataset/")   # 検出hmean、エンドツーエンドF1、1-NED
source_hash: 287e66812b8d3b4a
---

## インストール

PaddleOCR-VLはLibreYOLOのVLM層に属し、`vlm` extraが必要です。両ネットワークはtransformersのネイティブクラスで読み込まれるため、transformers 5.3以降が必要です。PaddlePaddle自体は不要です。

```bash
pip install "libreyolo[vlm]"
```

## 推論

両ネットワークの重みは、初回使用時に固定された1つのミラー`LibreYOLO/LibrePaddleOCRVL0.9b`からダウンロードされます。リモートコードは実行されません。

<code-tabs name="predict" />

このファミリーは`LibreVLM()`、または`LibrePaddleOCRVL`クラスで読み込みます。`Results.ocr`は、テキスト行ごとではなく認識されたブロック（段落、タイトル、表）ごとに1領域を読み順で保持します。`texts`、`polygons`（4点の四角形で表した軸に平行なボックス）、`labels`（レイアウトクラス）、そして`markdown`のページです。表はHTML、数式はLaTeXで返されます。画像と、デフォルトではグラフと印影も、`Results.ocr`とMarkdownから除外されます。

`Results.ocr.conf`は生成トークンの確率の幾何平均で、`det_conf`はレイアウトスコアです。`conf=`は領域を絞り込みません。レイアウトはアップストリームのクラスごとのしきい値を維持し、`layout_threshold=`で上書きできます。認識は貪欲法で新規トークンは最大4096、各領域は`min_pixels` / `max_pixels`（デフォルト112896 / 1003520）の範囲でリサイズされます。

`use_layout_detection=False`と`prompt_label=`（`ocr`、`table`、`formula`、`chart`）を指定すると、切り出し済みの要素を読み取ります。

## バリアント

サイズは`0.9b`の1つで、PP-DocLayoutV2の後段にNaViT方式の視覚エンコーダーとERNIE-4.5-0.3Bのデコーダーを置いた構成です。CUDAでは認識器をbf16（bf16非対応ならfp16）、レイアウト検出器をfp32で実行し、CPUとMPSでは両方をfp32で実行します。MPSで演算子の欠落が報告された場合は、`PYTORCH_ENABLE_MPS_FALLBACK=1`を設定してください。

移植していないもの：隣接するテキストブロックの結合、表に埋め込まれた図、印影とスポッティングの認識、ポリゴン形状のレイアウト、文書の歪み補正、画像のエクスポート。MarkdownはHTMLによる中央寄せのないプレーン版です。

## 検証

`val()`は共有のOCRバリデーターを実行します。

<code-tabs name="val" />

## エクスポート

推論のみです。`train()`と`export()`は例外を発生させます。

## ライセンス

<provenance-box>

PaddleOCR-VL-0.9BとPP-DocLayoutV2は[PaddlePaddle Authors](https://github.com/PaddlePaddle/PaddleOCR)によるApache-2.0で、コードと重みの両方が対象です。商用利用が可能です。LibreYOLOがPaddleXから移植したパイプラインもApache-2.0です。重みは変更なしの、ゲートされていないミラーです。

</provenance-box>

## 引用

<citation-block />
