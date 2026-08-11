---
title: プロンプト可能セグメンテーションAPI
seo_title: 'LibreSAM API: プロンプト、別名、シグネチャ'
description: >-
  LibreSAMファクトリ、サイズの別名、点、ボックス、概念テキストのプロンプト形式、1回エンコードするset_imageのライフサイクル、このティアが対応しない機能を解説します。
lead: >-
  LibreSAMはプロンプト可能セグメンテーション用のファクトリです。順伝播には呼び出し時に画像ごとのプロンプトが必要なので、このティアはプロンプトなしの推論ランナーへ振り分けず、固有のpredictインターフェースを持ちます。
keywords:
  - LibreSAM
  - プロンプト可能セグメンテーション
  - SAM 点プロンプト
  - SAM ボックスプロンプト
  - set_image
  - すべてセグメンテーション
  - LibreYOLO sam 追加パッケージ
last_verified: 1.5.0
verification: >-
  ファクトリの別名、サイズ、リポジトリはlibreyolo/models/sam/model.py、sam2.py、edgetam.py、sam3.py、libreyolo/models/mobilesam/model.py、libreyolo/models/picosam3/model.pyから確認しました。プロンプト契約とデフォルト値はlibreyolo/models/sam/base.py、設計意図はdocs/adr/0007-libresam-contract.mdから確認しました。すべてv1.5.0時点です。
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[sam]'
  usage:
    - label: 点とボックスのプロンプト
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        r = model.predict(SAMPLE_IMAGE, points=[900, 370], labels=[1])
        print(r.masks.xy)
        print(r.boxes.xyxy)

        r = model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 200, 200])
        print(len(r))
    - label: 1回エンコードして何度もプロンプト
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")
        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[500, 375], labels=[1])
        b = model.predict(bboxes=[100, 100, 200, 200])
        print(len(a), len(b))

        model.reset_image()
source_hash: 18e8206c10ce17fd
---

## インストール

このティアには`sam`追加パッケージが必要です。

<code-tabs name="install" />

## ファクトリ

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model`はパスではなくサイズの別名です。`**kwargs`は`device`と`multimask`を受け取るファミリーのコンストラクターへ渡されます。未知の別名では`ValueError`を送出し、メッセージに既知の別名をすべて表示します。

<code-tabs name="usage" />

## 別名

| ファミリー | 別名 | サイズ | 重み |
|---|---|---|---|
| SAM-1 | `base`、`large`、`huge`、`b`、`l`、`h`、`sam-base`、`sam-large`、`sam-huge`、`sam_b`、`sam_l`、`sam_h` | `base`、`large`、`huge` | `facebook/sam-vit-base`、`-large`、`-huge` |
| SAM-2 | `sam2-tiny`、`sam2-small`、`sam2-base-plus`、`sam2-baseplus`、`sam2-large`と短縮形`sam2-t`、`sam2-s`、`sam2-bp`、`sam2-l`、`sam2_t`、`sam2_s`、`sam2_bp`、`sam2_l` | `tiny`、`small`、`base-plus`、`large` | `LibreYOLO/LibreSAM2tiny`、`-small`、`-base-plus`、`-large` |
| EdgeTAM | `edgetam`、`edge-tam`、`edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`、`sam-3`、`sam3-large` | `large` | `facebook/sam3` |
| MobileSAM | `mobilesam`、`mobilesam-tiny`、`mobilesam_t`、`mobile-sam`、`mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`、`picosam3-pico`、`picosam3_pico`、`pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

デフォルトは`base`です。SAM-1、SAM-2、EdgeTAM、MobileSAMは公称1024pxのキャンバス、SAM 3は1008、PicoSAM3は96で実行します。

SAM 3の重みにはアクセス制限があります。Meta独自のSAM Licenseの下で`facebook/sam3`からダウンロードされます。このライセンスはMITでもApache-2.0でもなく、LibreYOLOは再配布しません。読み込み前にリポジトリページで条件に同意し、Hugging Faceで認証してください。ローダーは最初にこの告知をログへ記録します。

ファミリークラスもエクスポートされるため、`LibreSAM1`、`LibreSAM2`、`LibreSAM3`、`LibreEdgeTAM`、`LibreMobileSAM`、`LibrePicoSAM3`を`size=`で直接構築できます。

## predict

```python
model.predict(
    source=None,
    *,
    points=None,
    bboxes=None,
    labels=None,
    masks=None,
    text=None,
    conf=None,
    multimask=None,
    max_det=300,
    device=None,
    color_format="auto",
    points_per_side=None,
) -> Results
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `source` | `None` | セグメンテーションする画像。`None`では`set_image()`がキャッシュした画像を再利用 |
| `points` | `None` | ピクセル座標の点プロンプト |
| `bboxes` | `None` | `[x1, y1, x2, y2]`のボックスプロンプト、またはボックスごとに1つのマスクを生成するリスト |
| `labels` | `None` | `points`に合う形状の点ラベル。正が`1`、負が`0`。省略時はすべて正 |
| `masks` | `None` | 予約済み。渡すと`NotImplementedError`を送出 |
| `text` | `None` | 概念プロンプト。SAM 3のみ |
| `conf` | `None` | 予測マスクIoUの下限 |
| `multimask` | `None` | プロンプトごとに曖昧性マスクをすべて返す。デフォルトは構築時の設定 |
| `max_det` | `300` | 返すマスク数の上限 |
| `device` | `None` | この呼び出しと以降の呼び出し用にモデルを移動し、キャッシュ済み埋め込みベクトルを無効化 |
| `color_format` | `"auto"` | メモリ内配列の色形式ヒント |
| `points_per_side` | `None` | すべてセグメンテーションのグリッド密度。デフォルトは32 |

戻り値は通常の`Results`で、`masks`に加えて、マスクから導出した密な`boxes`を保持します。クラス`0`の名前は`"object"`です。

## プロンプトの形状

`points`は、1物体を表す`[x, y]`、N物体を表す`[[x, y], ...]`、物体ごとに点をまとめた`[[[x, y], ...], ...]`という入れ子形式を受け付けます。リストを使用できる場所ではNumPy配列も使用できます。座標はソース画像上の通常のピクセル値です。

すべての空間プロンプトを省略すると、点のグリッドを使う自動マスク生成器で、すべてセグメンテーションを実行します。予測IoUのしきい値とボックスIoUによる重複除去を使用します。デフォルトの`points_per_side=32`では約1024回デコードし、CPUでは遅くなります。対話的な用途では小さくしてください。生成器は安定性スコアによるフィルタリング、複数クロップ、マスクIoUによる重複除去を省略するため、プロンプト経路と同一ではなく近似です。

## 信頼度

`conf`は検出の信頼度ではなく、マスク品質を表す予測マスクIoUで絞り込みます。`None`はプロンプト経路ですべてのマスクを残し、すべてセグメンテーションではファミリー固有のグリッドしきい値を適用します。`0.0`はどちらのモードでもフィルタリングを無効にします。

SAM 3のテキスト経路では、`conf`はPromptable Concept Segmentationの検出スコアです。ここで`None`は標準の0.3しきい値を意味し、`0.0`はすべての候補を残します。

## テキストプロンプト

`text=`はSAM 3専用で、空間プロンプトの各ファミリーでは`NotImplementedError`を送出します。テキストと点やボックスは同時に指定できません。返される`names`はクラス`0`を要求した概念へマッピングします。`source=None`でテキストを呼び出すと、トラッカーと概念エンコーダーはキャッシュを共有しないため、キャッシュ済み画像を再エンコードします。

キーワード`exemplars=`は将来の画像例示拡張用に予約され、未実装です。

## 1回エンコードするライフサイクル

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image`は重い画像エンコーダーを1回実行して埋め込みベクトルをキャッシュするため、以降の`source=None`を使う`predict()`は低コストです。どちらのメソッドもモデルを返すため、呼び出しを連結できます。`predict`へ`device=`を渡すとモデルを移動し、キャッシュを無効にします。

## PicoSAM3

PicoSAM3は`bboxes=`だけを受け付けます。点、テキスト、マスク、multimask、すべてセグメンテーションのプロンプトは例外を送出します。ボックスを10パーセント拡大して96pxのROIネットワークで実行します。このティアで唯一エクスポートできるファミリーで、対応形式はONNXだけです。

## 未対応の機能

このティアの全ファミリーで`train()`、`val()`、`track()`が`NotImplementedError`を送出します。プロンプト可能マスクには採点対象となる固定クラス集合がないため、ここでmAPは意味を持ちません。SAM-1、SAM-2、SAM 3、EdgeTAM、MobileSAMでは`export()`も例外を送出します。

このバージョンでは、SAM-2、SAM 3、EdgeTAMの動画とメモリの経路、SAM 3の画像例示、マスクプロンプトは対象外です。
