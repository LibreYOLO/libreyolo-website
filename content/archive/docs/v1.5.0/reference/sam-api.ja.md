---
title: プロンプト可能セグメンテーションAPI
seo_title: 'LibreSAM API: プロンプト、エイリアス、シグネチャ'
description: >-
  LibreSAMファクトリー、サイズエイリアス、点、ボックス、概念テキストのプロンプト形式、1回エンコードするset_imageのライフサイクル、このレベルで未対応の機能を説明します。
lead: >-
  LibreSAMはプロンプト可能セグメンテーション用のファクトリーです。順伝播には呼び出し時に画像ごとのプロンプトが必要なため、このレベルはプロンプト不要の推論ランナーを経由せず、独自のpredictインターフェースを持ちます。
keywords:
  - LibreSAM
  - プロンプト可能セグメンテーション
  - SAM 点プロンプト
  - SAM ボックスプロンプト
  - set_image
  - 全領域セグメンテーション
  - LibreYOLO SAM 追加パッケージ
last_verified: 1.5.0
verification: >-
  ファクトリーのエイリアス、サイズ、リポジトリはlibreyolo/models/sam/model.py、sam2.py、edgetam.py、sam3.py、libreyolo/models/mobilesam/model.py、libreyolo/models/picosam3/model.pyから確認しました。プロンプト仕様とデフォルトはlibreyolo/models/sam/base.pyから確認しました。設計意図はdocs/adr/0007-libresam-contract.mdに基づき、すべてv1.5.0です。
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
    - label: 1回エンコードして複数回プロンプトを指定
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

このレベルには`sam`追加パッケージが必要です。

<code-tabs name="install" />

## ファクトリー

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model`はパスではなくサイズエイリアスです。`**kwargs`はファミリーのコンストラクターへ渡され、`device`と`multimask`を受け取ります。未知のエイリアスでは`ValueError`が発生し、メッセージに既知の全エイリアスが表示されます。

<code-tabs name="usage" />

## エイリアス

| ファミリー | エイリアス | サイズ | 重み |
|---|---|---|---|
| SAM-1 | `base`、`large`、`huge`、`b`、`l`、`h`、`sam-base`、`sam-large`、`sam-huge`、`sam_b`、`sam_l`、`sam_h` | `base`、`large`、`huge` | `facebook/sam-vit-base`、`-large`、`-huge` |
| SAM-2 | `sam2-tiny`、`sam2-small`、`sam2-base-plus`、`sam2-baseplus`、`sam2-large`と短縮形`sam2-t`、`sam2-s`、`sam2-bp`、`sam2-l`、`sam2_t`、`sam2_s`、`sam2_bp`、`sam2_l` | `tiny`、`small`、`base-plus`、`large` | `LibreYOLO/LibreSAM2tiny`、`-small`、`-base-plus`、`-large` |
| EdgeTAM | `edgetam`、`edge-tam`、`edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`、`sam-3`、`sam3-large` | `large` | `facebook/sam3` |
| MobileSAM | `mobilesam`、`mobilesam-tiny`、`mobilesam_t`、`mobile-sam`、`mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`、`picosam3-pico`、`picosam3_pico`、`pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

デフォルトは`base`です。SAM-1、SAM-2、EdgeTAM、MobileSAMは公称1024ピクセルのキャンバス、SAM 3は1008、PicoSAM3は96で動作します。

SAM 3の重みにはアクセス制限があります。Metaの独自SAM Licenseのもとで`facebook/sam3`からダウンロードされます。このライセンスはMITでもApache-2.0でもなく、LibreYOLOは重みを再配布しません。読み込む前にリポジトリページで条項へ同意し、Hugging Faceで認証してください。ローダーは最初に通知をログへ記録します。

ファミリークラスもエクスポートされるため、`LibreSAM1`、`LibreSAM2`、`LibreSAM3`、`LibreEdgeTAM`、`LibreMobileSAM`、`LibrePicoSAM3`を`size=`付きで直接構築できます。

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
| `bboxes` | `None` | `[x1, y1, x2, y2]`形式のボックスプロンプト、またはボックスごとに1つのマスクを得るためのリスト |
| `labels` | `None` | 点ラベル。`1`は正、`0`は負で、`points`と同じ形状。省略時はすべて正 |
| `masks` | `None` | 予約済み。渡すと`NotImplementedError`が発生 |
| `text` | `None` | 概念プロンプト。SAM 3のみ |
| `conf` | `None` | 予測mask-IoUの下限 |
| `multimask` | `None` | プロンプトごとに曖昧性を表す全マスクを返す。デフォルトは構築時の設定 |
| `max_det` | `300` | 返すマスク数の上限 |
| `device` | `None` | 今回以降の呼び出し用にモデルを移動し、キャッシュ済み埋め込みを無効化 |
| `color_format` | `"auto"` | メモリ上の配列に対する色形式のヒント |
| `points_per_side` | `None` | 全領域セグメンテーションのグリッド密度。デフォルトは32 |

戻り値は通常の`Results`で、`masks`に加え、マスクから導出した密な`boxes`を持ちます。クラス`0`の名前は`"object"`です。

## プロンプトの形状

`points`は、1つの物体を表す`[x, y]`、N個の物体を表す`[[x, y], ...]`、物体ごとに点をグループ化する`[[[x, y], ...], ...]`というネスト形式を受け付けます。リストを使用できる箇所ではすべてNumPy配列も使用できます。座標はソース画像上の通常のピクセル値です。

空間プロンプトをすべて省略すると全領域セグメンテーションを実行します。グリッドを使う自動マスク生成器で、予測IoUしきい値とボックスIoUによる重複除去を行います。デフォルトの`points_per_side=32`では約1024回のデコーダー処理が必要で、CPUでは低速です。対話的に使う場合は値を下げてください。この生成器はstability score filtering、multi-crop、mask-IoUによる重複除去を省略するため、プロンプト経路と一致するものではなく近似です。

## 信頼度

`conf`は予測mask-IoUでフィルタリングします。これはマスク品質のスコアであり、物体検出の信頼度ではありません。`None`ではプロンプト経路の全マスクを保持し、全領域セグメンテーションではファミリーのグリッドしきい値を適用します。`0.0`ではどちらのモードでもフィルタリングを無効にします。

SAM 3のテキスト経路では、代わりに`conf`がPromptable Concept Segmentationの検出スコアです。この場合、`None`は標準の0.3しきい値を意味し、`0.0`はすべての候補を保持します。

## テキストプロンプト

`text=`はSAM 3専用で、空間プロンプト型の各ファミリーでは`NotImplementedError`が発生します。テキストは点およびボックスと同時に指定できません。戻り値の`names`はクラス`0`を要求した概念へマッピングします。`source=None`でテキストを呼び出す場合、trackerと概念エンコーダーはキャッシュを共有しないため、キャッシュ済み画像を再エンコードします。

キーワード`exemplars=`は将来の画像例による拡張用に予約され、未実装です。

## 1回エンコードするライフサイクル

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image`は重い画像エンコーダーを1回実行して埋め込みをキャッシュするため、その後の`source=None`による`predict()`は低コストです。どちらのメソッドもモデルを返すので、呼び出しを連結できます。`predict`へ`device=`を渡すとモデルが移動し、キャッシュが無効になります。

## PicoSAM3

PicoSAM3が受け付けるのは`bboxes=`だけです。点、テキスト、マスク、multimask、全領域セグメンテーションの各プロンプトでは例外が発生します。ボックスを10%拡大し、96ピクセルのROIネットワークで処理します。PicoSAM3はこのレベルで唯一エクスポートに対応し、形式はONNXだけです。

## 未対応の機能

このレベルの全ファミリーで`train()`、`val()`、`track()`は`NotImplementedError`を発生させます。プロンプト可能マスクには評価対象となる固定クラスセットがないため、ここではmAPに意味がありません。SAM-1、SAM-2、SAM 3、EdgeTAM、MobileSAMでは`export()`も例外を発生させます。

SAM-2、SAM 3、EdgeTAMの動画およびmemory経路、SAM 3の画像例、マスクプロンプトは、このバージョンの対象外です。
