---
title: PP-DocLayoutV3
families: [ppdoclayoutv3]
seo_title: "PythonでPP-DocLayoutV3：読み順付きのドキュメントレイアウト検出"
description: "LibreYOLOのPP-DocLayoutV3で25種類のページ領域を検出し、読み順に並べます。Apache-2.0、純粋なPyTorch、ONNXとTorchScriptにエクスポート可能。"
lead: "PP-DocLayoutV3は、PaddleOCRのドキュメントレイアウト検出器です。ページの領域（テキスト、タイトル、表、図、数式、ヘッダー、フッター、印影など）を検出し、読み順に並べます。LibreYOLOはこれを、transformersやPaddlePaddleのランタイムを必要としないdetectファミリーとして移植しています。"
keywords: [PP-DocLayoutV3, PaddleOCR, ドキュメント レイアウト解析, レイアウト検出, 読み順, RT-DETR, HGNetV2, ドキュメント解析]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPDocLayoutV3l.pt")
        result = model("page.png", conf=0.5)    # アップストリームのパイプラインのしきい値

        for order, box, cls, score in zip(
            result.reading_order, result.boxes.xyxy, result.boxes.cls, result.boxes.conf
        ):
            print(int(order), result.names[int(cls)], round(float(score), 3), box.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePPDocLayoutV3l.pt source=page.png conf=0.5 save=True
    - label: 公式の重み
      language: python
      code: |
        from libreyolo import LibreYOLO

        # アップストリームのmodel.safetensorsも読み込み可能、一度だけ変換して
        # その隣にLibreYOLOのチェックポイントを書き出す
        model = LibreYOLO("model.safetensors")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPDocLayoutV3l.pt")
        model.export(format="onnx")             # 800x800固定、FP32
        model.export(format="torchscript")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePPDocLayoutV3l.pt format=onnx
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPDocLayoutV3l.onnx")
        result = model("page.png", conf=0.5)
        print(result.reading_order)
source_hash: abc633f93d73ccc5
---

## インストール

PP-DocLayoutV3にオプションのextraは不要です。純粋なPyTorchで、インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時に`LibreYOLO/LibrePPDocLayoutV3l`からダウンロードされます。

<code-tabs name="predict" />

`Results.boxes`は、元のページ上のxyxyボックス、信頼度、25のレイアウトクラスに対するクラスIDを保持します。行は読み順に並べられ、`Results.reading_order`はボックスに対応する1始まりのint64ベクトルです。`track_id`と同じく`results[i]`でスライスでき、`Results.summary()`の行には`reading_order`として現れます。順位は`conf`で残った領域の間で計算されます。`classes=`フィルターや`max_det`による打ち切りは残った順位を維持するため、値が飛ぶことがあります。

アップストリームのパイプラインは`conf=0.5`を使います。LibreYOLOのデフォルトの`conf=0.25`では、信頼度の低い領域がより多く残ります。アップストリームと同じく、スコアはクラスごとのシグモイドで、上位300の（クエリ、クラス）ペアが保持されるため、1つの領域が2つのラベルで報告されることがあります。NMSは適用されません。

クラス：`abstract`、`algorithm`、`aside_text`、`chart`、`content`、`display_formula`、`doc_title`、`figure_title`、`footer`、`footer_image`、`footnote`、`formula_number`、`header`、`header_image`、`image`、`inline_formula`、`number`、`paragraph_title`、`reference`、`reference_content`、`seal`、`table`、`text`、`vertical_text`、`vision_footnote`。

同じ移植版が、`layout=True`を指定した[Falcon OCR](/docs/models/falcon-ocr)のレイアウト段を実行します。

## バリアント

サイズは`l`の1つです。HGNetV2-Lバックボーン、マスク強化型のクエリ初期化、領域のすべてのペアをスコア付けするglobal-pointerヘッドを備えた、RT-DETR方式の集合予測検出器です。入力は800x800固定のキャンバスです（引き伸ばしリサイズ、`[0, 1]`のRGB、正規化なし）。CPU、CUDA、MPSで動作します。MPSでは位置エンコーディングをfloat32で計算するため、結果の末尾のビットが異なることがあります。

transformersのApache-2.0のPyTorchリファレンスと比べて、CUDA上のfp32で同じ入力テンソルを与えると、3つの実ページとランダムなテンソルでlogits、ボックス、読み順のlogits、マスクが完全に一致します。学習は提供していません。読み順と領域ごとのマスクの教師データが必要ですが、LibreYOLOのdetectデータセットにはそれがありません。アップストリームがマスク分岐から導出する領域ごとのポリゴンは公開していません。

## エクスポート

<export-matrix />

ONNX（opset 17）とTorchScriptで、800x800固定、FP32です。グラフは`pred_logits`、`pred_boxes`、`order_votes`を出力します。エクスポートしたバックエンドは`reading_order`を含む同じ`Results`を返します。

<code-tabs name="export" />

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box>

PP-DocLayoutV3は[PaddlePaddle Authors](https://github.com/PaddlePaddle/PaddleOCR)によるApache-2.0で、コードと重みの両方が対象です。商用利用が可能です。チェックポイントは`PaddlePaddle/PP-DocLayoutV3_safetensors`から、キーの接頭辞を取り除きメタデータを追加して変換したものです。学習済みパラメータは変更しておらず、ミラーはゲートされていません。

</provenance-box>

## 引用

<citation-block />
