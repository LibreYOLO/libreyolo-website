---
title: オープンボキャブラリAPI
seo_title: LibreOpenVocab API：別名と引数
description: >-
  LibreOpenVocabファクトリー、4つのファミリーとすべての別名、set_classes、ファミリーごとのconfデフォルト値、text_thresholdとiouの規則を説明します。
lead: >-
  LibreOpenVocabはテキスト条件付き検出器のファクトリーです。クラス一覧は固定ヘッドではなくプロンプトなので、set_classesでボキャブラリを設定し、モデルはそれに対する通常の検出Resultsを返します。
keywords:
  - LibreOpenVocab
  - オープンボキャブラリ検出
  - Grounding DINO
  - OWLv2
  - OMDet-Turbo
  - OV-DEIM
  - set_classes
last_verified: 1.5.0
verification: >-
  別名はlibreyolo/models/openvocab/__init__.pyから、リポジトリ、サイズ、しきい値はgrounding_dino.py、owlv2.py、omdet_turbo.py、ov_deim.pyから、呼び出し規則はlibreyolo/models/openvocab/base.pyから参照しました。すべてv1.5.0です。設計意図はdocs/adr/0008-open-vocab-detector-contract.mdから参照しました。
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[openvocab]'
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-tiny")
        model.set_classes(["person", "skateboard", "handrail"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
source_hash: 64e4c641c6f8cde0
---

## インストール

この階層には `openvocab` 追加パッケージが必要です。

<code-tabs name="install" />

## ファクトリー

```python
LibreOpenVocab(model: str = "grounding-dino-tiny", **kwargs) -> LibreOpenVocabDetector
```

`model` はパスではなく別名です。検索前にアンダースコアはハイフンへ変換されるため、CLIの一覧に表示される `omdet_turbo-t` や `grounding_dino-t` など、ファミリーで修飾された名前をそのまま読み込めます。不明な別名では、既知のすべての別名を列挙する `ValueError` が発生します。

コンストラクターは `size`、`nb_classes=80`、`names=None`、`device="auto"`、`task=None`、`text_threshold=None` を受け付けます。`names` を渡すことは、読み込み直後に `set_classes` を呼び出すことと同じです。対応しないファミリーへ `text_threshold` を渡すと `TypeError` が発生します。

<code-tabs name="usage" />

## ファミリーと別名

| ファミリー | 別名 | サイズ | 重み |
|---|---|---|---|
| Grounding DINO | `grounding-dino`, `groundingdino`, `grounding-dino-tiny`, `groundingdino-tiny`, `grounding-dino-t`, `groundingdino-t`, `grounding-dino-base`, `groundingdino-base`, `grounding-dino-b`, `groundingdino-b` | `t`, `b` | `LibreYOLO/LibreGroundingDINOt`, `LibreYOLO/LibreGroundingDINOb` |
| OWLv2 | `owlv2`, `owl-v2`, `owlv2-base`, `owl-v2-base`, `owlv2-b16`, `owl-v2-b16`, `owlv2-large`, `owl-v2-large`, `owlv2-l14`, `owl-v2-l14` | `b16`, `l14` | `LibreYOLO/LibreOWLv2b16`, `LibreYOLO/LibreOWLv2l14` |
| OMDet-Turbo | `omdet-turbo`, `omdet`, `omdetturbo`, `omdet-turbo-tiny`, `omdet-turbo-swin-tiny`, `omdet-turbo-t` | `t` | `LibreYOLO/LibreOMDetTurbot` |
| OV-DEIM | `ov-deim`, `ovdeim`, `ov-deim-s`, `ovdeim-s`, `ov-deim-m`, `ovdeim-m`, `ov-deim-l`, `ovdeim-l` | `s`, `m`, `l` | `LibreYOLO/LibreOVDEIMs`, `LibreYOLO/LibreOVDEIMm`, `LibreYOLO/LibreOVDEIMl` |

デフォルトの別名は `grounding-dino-tiny` です。

`LibreGroundingDINO`、`LibreOWLv2`、`LibreOMDetTurbo` はパッケージレベルでエクスポートされ、`size=` を使って直接構築できます。OV-DEIMには上記のファクトリー別名を通じて到達できます。

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreOpenVocabDetector
```

その後のすべての `predict()` 呼び出しに使うボキャブラリを設定し、呼び出しを連結できるようモデルを返します。一覧は空であってはならず、文字列だけを含む必要があります。また、大文字と小文字を区別せずに比較したとき、各項目は一意でなければなりません。空白だけのラベルは拒否されます。1つの文字列をそのまま渡すと `TypeError` が発生します。1文字ずつのクラスへ列挙されてしまうためです。

呼び出し後、`model.names` は指定した順序のラベルへ `0..N-1` を対応付け、`model.nb_classes` は `N` になります。

## 呼び出し引数

この階層は、3つの相違点を除いて標準の推論面を再利用します。

`conf` は共有値0.25ではなく、ファミリー独自の値をデフォルトにします。

| ファミリー | デフォルトのconf | 抑制 |
|---|---|---|
| Grounding DINO | 0.25 | |
| OWLv2 | 0.1 | |
| OMDet-Turbo | 0.3 | 独自の後処理、しきい値0.5、`iou=` を尊重 |
| OV-DEIM | 0.25 | top-K選択による1対1照合、抑制なし |

`iou=` は、抑制を実行するファミリーだけで意味を持ちます。OMDet-Turboはしきい値を引数として受け取り、`iou=` 未指定時のデフォルトは0.5です。その他の3つは何も抑制しないため、そこで `iou=` を渡すと警告を表示して無視します。

`text_threshold=` はGrounding DINO専用で、デフォルトは0.25です。永続的な値として構築時に渡すか、呼び出しごとに渡せます。呼び出しごとの値は `stream=True` と組み合わせられません。ストリームの結果は遅延生成されるためです。代わりにコンストラクターへ設定してください。その他のファミリーでは `TypeError` が発生します。

`imgsz=` は `ValueError` を発生させます。この階層では前処理パイプラインがリサイズを管理します。テスト時拡張はここで対象外なので、`augment=True` もエラーになります。入力サイズは参照用としてファミリーごとに記録されるだけです。Grounding DINOは800、OWLv2は960と1008、OMDet-Turboは640、OV-DEIMは640です。

## 未対応

`train()`、`val()`、`track()`、`export()` はすべて `NotImplementedError` を発生させます。アップストリームでファインチューニングして得られた重みを読み込み、トラッキングの代わりにフレームごとに `predict()` を実行してください。共有の検出バリデーターは画像テンソルでモデルを呼び出しますが、この階層はテキスト条件付き入力を必要とするため、検証には専用バリデーターが必要です。
