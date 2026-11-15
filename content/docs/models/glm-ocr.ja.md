---
title: GLM-OCR
families: [glmocr]
seo_title: "PythonでGLM-OCR：ドキュメントページをMarkdownに解析"
description: "LibreYOLOでZ.aiのGLM-OCRを実行します。PP-DocLayoutV3のレイアウトと、テキスト、表（HTML）、数式（LaTeX）を読む0.9Bの認識器。MITとApache-2.0。"
lead: "GLM-OCRは、Z.aiによる2段階のドキュメントパーサーです。PP-DocLayoutV3がレイアウト領域とその読み順を検出し、0.9BのGLM-OCR認識器が各領域をテキスト、HTMLの表、LaTeXの数式として読み取ります。LibreYOLOは、両ネットワークを囲む公式SDKのパイプラインをocrタスク向けに移植しています。"
keywords: [GLM-OCR, Z.ai, zai-org, ドキュメント解析, OCR markdown 変換, 表認識, 数式認識, PP-DocLayoutV3, LibreVLM]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("glm-ocr")            # 初回使用時に重みをダウンロード
        result = model.predict("page.png")      # スキャンまたはレンダリングしたページ

        for label, text in zip(result.ocr.labels, result.ocr.texts):
            print(label, text[:60])             # 領域ごとに1ブロック、読み順
        print(result.ocr.markdown)              # ページ全体をMarkdownで
    - label: 単一要素
      language: python
      code: |
        from libreyolo import LibreVLM

        # レイアウト検出を省略し、切り出し済みの表を読み取る
        model = LibreVLM("glm-ocr", use_layout_detection=False, prompt_label="table")
        result = model.predict("table.png")
        print(result.ocr.texts[0])              # HTMLの表
    - label: 情報抽出
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("glm-ocr")
        reply = model.chat("id_card.png", 'Extract as JSON: {"name": "", "date": ""}')
        print(reply)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("glm-ocr")
        metrics = model.val(data="ocr_dataset/")   # 検出hmean、エンドツーエンドF1、1-NED
source_hash: 49c7d7cc949d1a45
---

## インストール

GLM-OCRはLibreYOLOのVLM層に属し、`vlm` extraが必要です。両ネットワークはtransformersのネイティブクラスで読み込まれるため、transformers 5.3以降が必要です。

```bash
pip install "libreyolo[vlm]"
```

## 推論

両ネットワークの重みは、初回使用時に固定された1つのミラー`LibreYOLO/LibreGLMOCR0.9b`からダウンロードされます。リモートコードは実行されません。

<code-tabs name="predict" />

このファミリーは`LibreVLM()`、または`LibreGLMOCR`クラスで読み込みます。`Results.ocr`は読み順でレイアウト領域ごとに1ブロックを保持します。`texts`、`polygons`（4点の四角形で表した軸に平行なボックス）、`labels`（`text`、`table`、`formula`などのレイアウトクラス）、そして公式SDKと同じ整形（タイトル、`\tag{}`による数式番号、ハイフネーションと箇条書きの修正）をした`markdown`のページです。ヘッダー、フッター、ページ番号、脚注、サイドテキストはアップストリームと同じく除外され、画像とグラフは読み取られません。

`Results.ocr.conf`は生成トークンの確率の幾何平均で、`det_conf`はレイアウトスコアです。`conf=`は領域を絞り込みません。レイアウトはアップストリームのしきい値0.3を使い、`layout_threshold=`で変更できます。デコードはSDKに従い、貪欲法、繰り返しペナルティ1.1、新規トークン最大8192です。各クロップは、SDKが送るのと同じく認識の前にJPEGを経由します。`image_format=None`でこれを省略できます。

`use_layout_detection=False`と`prompt_label=`（`text`、`table`、`formula`）を指定すると、切り出し済みの要素を読み取ります。これはSDKに対応するものがないLibreYOLO独自の便利機能です。`chat()`はJSONによる情報抽出のプロンプトに対応します。

## バリアント

サイズは`0.9b`の1つで、CogViTの視覚エンコーダーとGLM-0.5Bのデコーダーで構成されます。レイアウト段は800 pxで動作します。CUDAでは認識器をbf16（bf16非対応ならfp16）、レイアウト検出器をfp32で実行し、CPUとMPSでは両方をfp32で実行します。PDF入力、ポリゴンのクロップ、画像のエクスポート、ホスト型APIのクライアントは移植していません。

## 検証

`val()`はモデル自身の`predict`を通じて共有のOCRバリデーターを実行します。

<code-tabs name="val" />

## エクスポート

推論のみです。`train()`と`export()`は例外を発生させます。

## ライセンス

<provenance-box>

GLM-OCR認識器の重みは、[Z.ai](https://github.com/zai-org/GLM-OCR)のモデルカードの宣言どおりMITです。PP-DocLayoutV3レイアウト検出器はApache-2.0（PaddlePaddle Authors）で、LibreYOLOが移植したSDKのパイプラインもApache-2.0です。商用利用が可能です。重みはゲートされていないミラーです。

</provenance-box>

## 引用

<citation-block />
