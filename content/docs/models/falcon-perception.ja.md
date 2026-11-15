---
title: Falcon Perception
families: [falconperception]
seo_title: "PythonでFalcon Perception：テキストクエリによる検出とセグメンテーション"
description: "LibreYOLOでTIIのFalcon Perceptionを実行します。テキストクエリに一致するすべてのインスタンスを見つけてボックスを返し、600mではボックスとマスクを返します。Apache-2.0。"
lead: "Falcon Perceptionは、Technology Innovation Institute（TII）による早期融合（early fusion）型のTransformerです。画像と自然言語のクエリを読み、一致するすべてのインスタンスを返します。LibreYOLOはその推論ネットワークを同梱し、両サイズでのオープンボキャブラリ検出と、600mでのインスタンスセグメンテーションに対応します。"
keywords: [Falcon Perception, TII, オープンボキャブラリ 物体検出, 参照表現 セグメンテーション, テキスト プロンプト セグメンテーション, grounding, early fusion, LibreVLM]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        # 語彙の各エントリが1クエリ、インスタンスにはそのエントリのクラスIDが付く
        model = LibreVLM("falcon-perception", names=["building", "person"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.xyxy)
    - label: セグメンテーション
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("falcon-perception", names=["the titanium building"])
        result = model.set_task("segment").predict(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)          # インスタンスごとに1マスク
    - label: 300m
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        # 小さいサイズはマスクヘッドがないため検出のみ
        model = LibreVLM("falcon-perception-300m", names=["building"])
        result = model.predict(SAMPLE_IMAGE)
source_hash: 51373e45e0b278ec
---

## インストール

Falcon PerceptionはLibreYOLOのVLM層に属し、`vlm` extraが必要です。

```bash
pip install "libreyolo[vlm]"
```

## 推論

重みは初回使用時に、リビジョンを固定したTII自身のリポジトリ`tiiuae/Falcon-Perception`（600m）と`tiiuae/Falcon-Perception-300M`からダウンロードされます。ネットワークはLibreYOLOに同梱されており、リモートコードは実行されません。

<code-tabs name="predict" />

このファミリーは`LibreVLM()`、または`LibreFalconPerception`クラスで読み込みます。`names=`（または`set_classes()`）は必須です。語彙の各エントリは1つのテキストクエリと1回の生成に対応するため、コストはエントリ数に応じて増え、あるエントリで見つかったインスタンスにはすべてそのエントリのクラスIDが付きます。クエリはクラス名でも、`"the dog on the left"`のようなフレーズでもかまいません。

モデルは各インスタンスについて、ボックスの中心、ボックスの大きさ、マスククエリを出力します。`detect`は`Results.boxes`を返し、`segment`（600mのみ）はさらに`Results.masks`を返します。マスクは、マスククエリとAnyUpでアップサンプリングした画像特徴量との内積で作られます。信頼度は定数1.0で、NMSはありません。アップストリームの座標の重複除去はデコード中に実行されます（`coord_dedup_threshold`、デフォルト0.01）。生成の上限はクエリあたり新規トークン2048です（`max_new_tokens`）。

## バリアント

| 名前 | サイズ | タスク |
| --- | --- | --- |
| `falcon-perception`、`falcon-perception-600m` | 600m（デフォルト） | detect、segment |
| `falcon-perception-300m` | 300m | detect |

どちらも1024 pxで動作します。MPSしか使えない場合、このファミリーはCPUで動作します。RoPEのテーブルがcomplex64であり、MPSがそれを実装していないためです。CPU上のfp32で、この移植版はアップストリームのエンジンとビット単位で一致します。すべての順伝播のlogitsと隠れ状態、および生成トークン、ボックスの浮動小数点値、マスクのバイトが一致します。

## エクスポート

推論のみです。`train()`、`val()`、`export()`、`chat()`は例外を発生させます。

## ライセンス

<provenance-box>

Falcon Perceptionは[Technology Innovation Institute](https://github.com/tiiuae/Falcon-Perception)によるApache-2.0で、コードと重みの両方が対象です。商用利用が可能です。TIIのNOTICEは、公開する成果物でNOTICEを保持し、プロジェクトを引用するよう求めています。600mのsegmentタスクでのみ使うAnyUp特徴アップサンプラーは、AnyUpの著者によるCC BY 4.0で、帰属表示を条件に商用利用が認められています。LibreYOLOは重みを再配布しません。

</provenance-box>

## 引用

<citation-block />
