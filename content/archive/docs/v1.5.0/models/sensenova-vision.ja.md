---
title: SenseNova-Vision
families:
  - sensenovavision
seo_title: LibreYOLOのSenseNova-Vision：1つのチェックポイントで7つのタスク
description: >-
  LibreYOLOのSenseNova-Visionを使用し、1つのプロンプト駆動型生成チェックポイントで、検出、セグメンテーション、パノプティック、姿勢、点、深度、OCRを実行します。
lead: >-
  SenseNova-Visionは、共有デコーダー上のプロンプト駆動型生成として視覚タスクを扱う統合マルチモーダルモデルです。ボックス、点、キーポイント、OCRの単語はタグ付きテキストとして出力され、深度、マスク、パノプティックマップはデコーダーが描画する画像として出力されます。LibreYOLOはLibreVLMを通じてこれを読み込み、1つの7Bチェックポイントで7つのタスクに対応します。
keywords:
  - SenseNova-Vision
  - SenseTime
  - 統合 マルチモーダルモデル
  - Bagel
  - プロンプト 物体検出
  - dense perception
  - referring segmentation
  - パノプティックセグメンテーション
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="detect")
        model.set_classes(["bird", "boat"])
        result = model.predict("image.jpg")
        print(result.boxes.xyxy)

        # set_task()で読み込み済みの同じモデルのタスクを切り替え
        model.set_task("depth")
        result = model.predict("image.jpg")
        depth = result.depth_map.data
    - label: 参照セグメンテーションとパノプティック
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="segment")
        # セグメンテーションは参照型でありクラス一覧ではなく対象の句が必要
        model.set_classes(["the person furthest to the right"])
        result = model.predict("street.jpg")
        mask = result.masks.data[0]

        model.set_task("panoptic")
        # カスタムボキャブラリがなければパノプティックはチェックポイントの
        # チューニングに使われたCOCOパノプティックカテゴリへフォールバック
        result = model.predict("street.jpg")
        segment_map = result.panoptic.data
        for segment in result.panoptic.segments_info:
            print(segment)
    - label: 点、姿勢、OCR
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="point")
        model.set_classes(["screw"])
        result = model.predict("board.jpg")
        print(result.points.xy)

        # ボキャブラリが未設定なら姿勢はpersonへフォールバック
        model.set_task("pose")
        result = model.predict("gym.jpg")
        print(result.boxes.xyxy, result.keypoints.data.shape)

        model.set_task("ocr")
        result = model.predict("sign.jpg")
        print(result.ocr.texts)
source_hash: 8749277e1910baa4
---

## インストール

SenseNova-Visionには専用の追加パッケージが必要です。このチェックポイントに必要な大規模モデルの割り当て用に `accelerate` が導入され、macOS以外のプラットフォームでは4ビット読み込み用に `bitsandbytes` も導入されます。

```bash
pip install "libreyolo[sensenova]"
```

チェックポイントはLibreYOLO独自の組織下でHugging Faceにミラーされ、初回使用時に自動的にダウンロードされます。ライセンスはCC BY-NC 4.0で、非商用利用に限定されます。ローダーは自動ダウンロードの前に毎回この通知を表示します。下の「ライセンス」を参照してください。

## 推論

<code-tabs name="predict" />

各推論では共有Bagel-MoTバックボーン上で拡散デコードを行うため、これはリアルタイムモデルではなく能力重視のモデルです。専用の検出器やセグメンターより、画像ごとのレイテンシが明らかに高くなります。`dtype="auto"`（デフォルト）は、十分なメモリを持つGPUではbf16を読み込み、それ以外では4ビットNF4量子化へフォールバックします。後者には `bitsandbytes` が必要です。十分に大きなGPUで完全精度を強制するには `dtype="bf16"` を渡してください。構築時の `noise_seed=42` は、再現可能な密な出力のために拡散サンプラーのシードを設定します。シード設定を無効にするには `noise_seed=None` を渡してください。

7つのタスクは1つの読み込み済みチェックポイントを共有し、`set_task()` で再読み込みせずに切り替えられます。`set_classes()` は有効なボキャブラリを設定します。検出、点、姿勢、パノプティックはクラス一覧を受け付けますが、セグメンテーションは参照型であり、分離する対象を正確に表す句が必要です。各タスクは標準の `Results` オブジェクトを返し、異なるペイロードが設定されます。検出では `boxes`、点では `points`、姿勢では `boxes` と `keypoints`、OCRでは `ocr`、深度では `depth_map`、セグメンテーションでは `masks`、パノプティックでは `panoptic`（`segments_info` を含む）です。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
