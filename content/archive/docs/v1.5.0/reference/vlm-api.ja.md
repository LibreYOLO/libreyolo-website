---
title: 視覚言語API
seo_title: LibreVLM API：別名、set_classes、chat
description: >-
  LibreVLMファクトリー、すべてのモデル別名、保持されるset_classesの語彙、set_task、chatによる低水準操作、信頼度がプレースホルダーである理由を説明します。
lead: >-
  LibreVLMは生成型視覚言語モデルを読み込み、物体検出器として駆動します。クラスリストは固定ヘッドではなくプロンプトであり、モデルはほかのファミリーと同じResultsを返します。
keywords:
  - LibreVLM
  - 視覚言語モデル 物体検出
  - Qwen3-VL
  - LFM2-VL
  - InternVL3
  - SmolVLM2
  - Florence-2
  - libreyolo chat
last_verified: 1.5.0
verification: >-
  別名はlibreyolo/models/vlm/__init__.py、リポジトリ、サイズ、タスクリストはlibreyolo/models/vlm/以下のファミリーモジュールとlibreyolo/models/sensenova/model.py、呼び出し規則と例外はlibreyolo/models/vlm/base.pyから、すべてv1.5.0時点で確認しました。
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[vlm]'
  usage:
    - label: オープンボキャブラリを検出
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        model.set_classes(["person", "skateboard"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
    - label: 自由形式の質問をする
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        print(model.chat(SAMPLE_IMAGE, "How many people are in this image?"))
source_hash: 57ddac08bc4d4e05
---

## インストール

この階層には `vlm` 追加パッケージが必要です。

<code-tabs name="install" />

## ファクトリー

```python
LibreVLM(model: str = "qwen3-vl-4b", **kwargs) -> LibreVLMModel
```

`model` はパスではなく別名です。`**kwargs` はファミリーのconstructorに渡されます。constructorは `device`、`names`（初期語彙であり、読み込み後の `set_classes` 呼び出しと同等）、`prompt`（検出プロンプトの上書き）、`max_new_tokens` を受け取ります。不明な別名を指定すると、すべての別名を列挙する `ValueError` が送出されます。

<code-tabs name="usage" />

## 別名

| ファミリー | 別名 | サイズ | 重み |
|---|---|---|---|
| Qwen3-VL | `qwen3-vl`, `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` | `2b`, `4b`, `8b` | `Qwen/Qwen3-VL-2B-Instruct`, `-4B-`, `-8B-` |
| LFM2-VL | `lfm2-vl`, `lfm2-vl-450m`, `lfm2-vl-1.6b` | `450m`, `1.6b` | `LiquidAI/LFM2.5-VL-450M`, `-1.6B` |
| InternVL3 | `internvl3`, `internvl3-1b`, `internvl3-2b`, `internvl3-8b` | `1b`, `2b`, `8b` | `OpenGVLab/InternVL3-1B-hf`, `-2B-hf`, `-8B-hf` |
| SmolVLM2 | `smolvlm2`, `smolvlm2-2.2b`, `smolvlm2-500m` | `2.2b`, `500m` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `SmolVLM2-500M-Video-Instruct` |
| Florence-2 | `florence-2`, `florence2`, `florence-2-base`, `florence-2-large` | `base`, `large` | `florence-community/Florence-2-base`, `-large` |
| Kosmos-2 | `kosmos-2`, `kosmos2` | `224` | `microsoft/kosmos-2-patch14-224` |
| LocateAnything | `locate-anything`, `locateanything`, `locate-anything-3b`, `locateanything-3b` | `3b` | `nvidia/LocateAnything-3B` |
| SenseNova-Vision | `sensenova-vision`, `sensenova-vision-7b`, `sensenovavision` | `7b` | `LibreYOLO/SenseNovaVision7b` |
| LibreMODUS | `libremodus`, `libremodus-14b-a7b`, `modus`, `modus-14b-a7b` | `14b-a7b` | 固定されたアップストリームsnapshot |

デフォルトの別名は `qwen3-vl-4b` です。各ファミリーのデフォルト別名では、最初に記載されたサイズが使われます。`qwen3-vl` は `4b`、`lfm2-vl` は `450m`、`internvl3` は `2b`、`smolvlm2` は `2.2b`、`florence-2` は `base` に解決されます。

`LibreVLM`、`LibreLFM2VL`、`LibreQwen3VL`、`LibreSmolVLM2`、`LibreInternVL3`、`LibreFlorence2`、`LibreKosmos2`、`LibreLocateAnything`、`LibreMODUS`（`LibreModus` という綴りも可能）はパッケージレベルでエクスポートされます。

## タスク

ほとんどのファミリーは `detect` だけを提供します。2つのファミリーは、さらに多くのタスクを提供します。

| ファミリー | 対応タスク |
|---|---|
| LocateAnything | `detect`, `point` |
| SenseNova-Vision | `detect`, `segment`, `panoptic`, `pose`, `point`, `depth`, `ocr` |

タスクはチェックポイントに埋め込まれず、プロンプトで制御されるため、読み込み済みモデルで切り替えられます。

```python
model.set_task(task: str) -> LibreVLMModel
```

タスクはファミリーの対応リストに照らして検証され、その後の `predict()` と `track()` の呼び出しでも維持されます。また、呼び出しを連鎖できるようモデルが返されます。

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreVLMModel
```

オープンボキャブラリを設定します。モデルは固定ヘッドによる制約を受けず、指定した語句を使ってプロンプトされるため、どのような単語でも使用できます。リストは空にできず、大文字と小文字を区別せず比較したときに項目が一意でなければなりません。単独の文字列を渡すと、1文字ずつのクラスとして列挙されてしまうため、`TypeError` が送出されます。語彙は維持されます。読み込み後に一度設定すると、再設定するまで保持されます。

## chat

```python
model.chat(image, prompt, max_new_tokens=None, color_format="auto") -> str
```

未加工のマルチモーダル生成です。画像とプロンプトを入力し、デコードされたテキストをそのまま出力します。検出用の便利なラッパーより低水準の操作手段であり、自由形式の質問、個数の計数、検出ラッパーが対応しない出力形式に使います。`max_new_tokens` はファミリーの `MAX_NEW_TOKENS` にフォールバックし、基底クラスでは1024です。デコードは軽いrepetition penaltyを伴うgreedy方式です。

## 信頼度

生成された出力には、ボックスごとに較正された信頼度がありません。このバージョンでは `predict`、描画、`track` を機能させるために定数のプレースホルダーを割り当てます。そのため、`conf=` による絞り込みとmAPは厳密な意味を持ちません。`val()` が例外を送出する理由も同じです。プレースホルダースコアに対するCOCO mAPは誤解を招きます。

## 推論と追跡

標準の推論インターフェースが適用され、`track()` も動作するため、VLM検出器はほかのファミリーと同じパイプラインに組み込めます。クラスレベルの2つの方針が畳み込み検出器とは異なります。固定解像度のジェネレーターにマルチスケール拡張は意味がないため、テスト時データ拡張は無効です。また、生成が自己回帰的で、前処理が積み重ね可能な画像テンソルではなくテキストと画像のencodingを返すため、バッチ推論は無効です。

## 非対応

`train()`、`val()`、`export()` は `NotImplementedError` を送出します。アップストリームでファインチューニングし、得られた重みを読み込んでください。

## リモートコード

公開されるすべてのファミリーはネイティブモデルクラスから読み込まれるため、LibreYOLOはデフォルトでサードパーティーのリポジトリコードを実行しません。本当に必要なファミリーは明示的にオプトインし、snapshotのrevisionを固定する必要があります。該当するのはLocateAnythingで、commit `c32291ca5e996f5a7a485845b4f57a233936bba0` に固定されています。

LibreMODUSはチェックポイントスキーマの明示的な例外です。その別名はLibreYOLOの `.pt` ではなく、固定されたアップストリームファイルのディレクトリに解決されます。LibreYOLOはv1.0メタデータを追加せず、再公開もしません。

