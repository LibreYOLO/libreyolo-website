---
title: LibreMODUS
families:
  - libremodus
seo_title: LibreYOLOのLibreMODUS：any-to-any画像解析
description: >-
  LibreYOLOのLibreMODUSで深度、法線、エッジ、検出を実行し、any2any()で組み合わせます。推論専用で、重みはEPFL-VILABから読み込みます。
lead: >-
  LibreMODUSはMODUS
  14B-A7Bチェックポイントを統合した推論専用機能です。これは画像由来の入力を別の出力へ変換するany-to-anyモデルです。RGBから深度、深度から法線、さらにそれらのいずれかと句からボックスを生成します。LibreYOLOは標準のpredict
  APIで4つのタスクに対応し、any2any()ではさらに広い範囲に対応します。
keywords:
  - LibreMODUS
  - MODUS
  - any-to-any
  - 深度推定
  - 表面法線
  - エッジ検出
  - referring detection
  - EPFL VILAB
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(size="14b-a7b", task="normal")
        result = model.predict("room.jpg")
        normals = result.normal_map.data

        model.set_task("edge")
        result = model.predict("room.jpg")
        edges = result.edges.data

        # カスタムボキャブラリがなければdetectはチェックポイントのCOCO
        # ラベルトークンを連続したCOCO-80クラスIDへデコード
        model.set_task("detect")
        result = model.predict("street.jpg")
        print(result.boxes.xyxy)
    - label: 句のグラウンディング
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(task="detect")
        # set_classes()で検出を句のグラウンディングへ切り替え 各句を
        # 個別に実行して同じBoxes契約を通じて返す
        model.set_classes(["red bus", "cyclist"])
        result = model.predict("street.jpg", conf=0.2)
        print(result.boxes.xyxy, result.boxes.cls)
    - label: any2any()
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS()

        # 1つから3つの画像由来入力(rgb depth normal canny/edge)と
        # オプションの補助テキストを1つのターゲットに向けて構成
        result = model.any2any(
            inputs={"rgb": "room.jpg"},
            target="normal",
            steps=10,
            cfg=2.0,
            seed=0,
        )
        normals = result.normal_map.data

        # any2any()でのグラウンディングには句を指定するテキスト入力が必要
        result = model.any2any(
            {"rgb": "street.jpg", "text": "red bus"},
            target="grounding",
        )
        print(result.boxes.xyxy)
source_hash: 7386886d4c36ea9a
---

## インストール

LibreMODUSには専用の追加パッケージが必要で、このチェックポイントが必要とする大規模モデルの割り当て用に `accelerate` を導入します。

```bash
pip install "libreyolo[modus]"
```

LibreYOLOはMODUSの重みを再配布もミラーもしません。デフォルトでは、`LibreMODUS` モデルを読み込むと、固定されたHugging Faceリビジョンの `EPFL-VILAB/MODUS` から必要なファイルを直接ダウンロードします。アップストリームのホスティング制限が一時的に解除されていても、新規ダウンロードには常に利用者自身の認証済みHugging Faceアカウントが必要です。アップストリームの条件を確認して同意した後、認証してください。

```bash
hf auth login
```

```python
from libreyolo import LibreMODUS

model = LibreMODUS(token="hf_...")
```

ネットワーク要求を避けるには、すでに用意したスナップショットを指定します。

```python
model = LibreMODUS(checkpoint_path="/models/MODUS")
```

このディレクトリには `model.safetensors`、`ae.safetensors`、`llm_config.json`、`vit_config.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt` が必要です。チェックポイントの条件で許可される範囲については、下の「ライセンス」を参照してください。

## 推論

<code-tabs name="predict" />

標準のタスクAPIは4つのタスクに対応し、それぞれ1つのMODUSターゲットに対応付けられます。`depth` は相対深度（`result.depth_map`）、`normal` は表面法線（`result.normal_map`）、`edge` はCanny形式のエッジ（`result.edges`）、`detect` はCOCO-80のボックス（`result.boxes`）です。ただし、`set_classes()` を使うと句のグラウンディングへ切り替わります。同じ読み込み済みモデル上で `set_task()` によりタスクを切り替えられます。公開されたレシピは、テキストガイダンス4.0、画像ガイダンス2.0で10回のフローサンプリングを使用します。構築時に `inference_steps=`、`inference_cfg=`、`inference_image_cfg=` を指定すると上書きできます。

`any2any()` はより広い公開解析面を利用できます。1つから3つの画像由来入力（`rgb`、`depth`、`normal`、`canny` または `edge`）と、オプションの補助テキストを組み合わせ、深度、法線、エッジ、SAM由来のエッジ、COCO検出、句のグラウンディングのいずれか1つを生成します。すべての画像由来入力は同じ位置合わせ済みキャンバスを表す必要があります。LibreMODUSは幅と高さが一致しない入力を個別にリサイズせず拒否します。`chain=(...)` は中間ターゲットを生成し、チェックポイントの3条件学習の範囲内で、同じコンテキストへ入力として戻します。`verify=N`（N >= 2）はN個の候補を生成し、制約付き自己整合性チェックで最高スコアの候補を保持します。このスコアは `result.verification_score` として公開されます。

`dtype="bf16"`（デフォルト）は公開チェックポイントの精度と一致します。`dtype="fp8"` は対象となるデコーダートランクの線形重みを、出力チャンネル単位のスケールを持つE4M3として保存し、`~/.cache/libreyolo/modus/fp8` 以下のローカルキャッシュへ1回だけ変換します。その後、行列積ごとに入力のdtypeへ逆量子化します。そのため、活性化の精度ではなくメモリとの引き換えになります。

`train()`、`val()`、`export()` はすべてエラーになります。LibreMODUSは推論専用で、データセット検証は提供されず、ONNX、TensorRT、TFLiteへのエクスポート経路もありません。バッチでの `predict()` とテスト時拡張にも対応していません。各呼び出しでは1枚の画像を処理します。

## ライセンス

<provenance-box>

LibreYOLOは独自のHugging Face組織を含め、どこにもMODUSチェックポイントをホストまたはミラーしません。読み込み時には常にEPFL-VILAB/MODUSから固定リビジョンを直接取得するか、`checkpoint_path` にあるディスク上の既存スナップショットを読み取ります。

</provenance-box>

## 引用

<citation-block />
