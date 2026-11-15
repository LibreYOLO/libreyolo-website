---
title: Falcon OCR
families: [falconocr]
seo_title: "PythonでFalcon OCR：0.27BのドキュメントOCRモデル、Apache-2.0"
description: "LibreYOLOでTIIのFalcon OCRを実行します。ページ全体を1回でテキスト化するか、PP-DocLayoutV3で領域ごとに読み取ります。Apache-2.0、CPUで動作。"
lead: "Falcon OCRは、Technology Innovation Institute（TII）による0.27Bの早期融合（early fusion）型ドキュメントOCRモデルです。1つのTransformerが画像パッチとテキストを一緒に読み、数式はLaTeX、表はHTMLでページをテキストとして書き出します。LibreYOLOはFalcon OCR 1.5のPyTorch移植版をocrタスクで実行します。"
keywords: [Falcon OCR, TII, ドキュメント OCR, early fusion, 軽量 OCR モデル, OCR markdown 変換, LibreVLM]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("falcon-ocr")         # 初回使用時に重みをダウンロード
        result = model.predict("page.png")      # スキャンまたはレンダリングしたページ

        print(result.ocr.markdown)              # ページのテキスト
    - label: レイアウト付き
      language: python
      code: |
        from libreyolo import LibreVLM

        # PP-DocLayoutV3が読み順で領域を検出し、各クロップを
        # カテゴリのプロンプトで読み取る
        model = LibreVLM("falcon-ocr", layout=True)
        result = model.predict("page.png")

        for label, text in zip(result.ocr.labels, result.ocr.texts):
            print(label, text[:60])
    - label: 単一要素
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("falcon-ocr", prompt="table")
        result = model.predict("table.png")     # 切り出した表
        print(result.ocr.markdown)              # HTMLの表
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("falcon-ocr")
        metrics = model.val(data="ocr_dataset/")
source_hash: 3218099fdef6cf7e
---

## インストール

Falcon OCRはLibreYOLOのVLM層に属し、`vlm` extraが必要です。

```bash
pip install "libreyolo[vlm]"
```

## 推論

重みは初回使用時に`LibreYOLO/LibreFalconOCR270m`からダウンロードされます。これはリビジョンを固定した`tiiuae/Falcon-OCR`のバイト単位で同一のミラーです。リリースはリモートコードを通じてFlexAttentionとTritonカーネルで動作しますが、LibreYOLOはそのどちらも使わない素のPyTorch移植版を実行します。

<code-tabs name="predict" />

このファミリーは`LibreVLM()`、または`LibreFalconOCR`クラスで読み込みます。アップストリームと同じく、2つのモードがあります。

- デフォルト：ページ全体を1回で読み取ります。`Results.ocr`は画像全体を覆うラベル`plain`の1領域を保持し、`Results.ocr.markdown`は同じテキストです。
- `layout=True`：LibreYOLOの[PP-DocLayoutV3](/docs/models/pp-doclayoutv3)移植版が`layout_threshold=0.3`で読み順に領域を検出し、入れ子のボックスを除外したうえで、各クロップをカテゴリのプロンプトで読み取ります。クロップごとに1領域となり、`labels`にレイアウトクラス、`det_conf`にレイアウトスコアが入り、`markdown`はテキストを空行で連結します。画像と、アップストリームのプロンプトカテゴリを持たないクラスはスキップされます。

`prompt=`にはカテゴリ（`plain`、`text`、`table`、`formula`、`caption`、`footnote`、`list-item`、`page-footer`、`page-header`、`section-header`、`title`）またはカスタム指示を指定します。ページは両辺が64〜1536 pxに収まるようにリサイズされます。デコードは貪欲法で、新規トークンの上限は8192です。生成モデルはスコアを出力しないため、`conf`は定数1.0です。

## バリアント

サイズは`270m`の1つです。CUDAでは`torch.autocast`の下でbf16の重みを実行し、CPUとMPSではfloat32で実行します。重みは約1.1 GBです。バッチサイズは1です。CPU上のfloat32で、この移植版は6つのOCRフィクスチャと合成のレポートページでリリースをトークン単位で再現します。CUDAのbf16とレイアウトモードはアップストリームと比較していません。

## 検証

`val()`は共有のOCRバリデーターを実行します。

<code-tabs name="val" />

## エクスポート

推論のみです。`train()`と`export()`は例外を発生させます。

## ライセンス

<provenance-box>

Falcon OCRは[Technology Innovation Institute](https://github.com/tiiuae/Falcon-Perception)によるApache-2.0で、コードと重みの両方が対象です。商用利用が可能です。重みはゲートされていない、バイト単位で同一のミラーです。オプションのレイアウト段は、LibreYOLOのApache-2.0のPP-DocLayoutV3移植版を使います。

</provenance-box>

## 引用

<citation-block />
