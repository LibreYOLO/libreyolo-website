---
title: 検出器のアンサンブル
seo_title: LibreYOLOで検出器をアンサンブル
description: 1枚の画像に複数の検出器を実行し、クラス一覧が異なるモデルも含め、weighted boxes fusionまたはNMSでボックスを融合します。
lead: >-
  LibreEnsembleは2つ以上の検出器を同じデコード済み画像に対して実行し、ボックスを1つのResultsオブジェクトへ融合します。各メンバーは固有の重み、しきい値、デバイス、クラス一覧を維持します。
keywords:
  - 物体検出 モデル アンサンブル
  - weighted boxes fusion
  - wbf python
  - 2つの検出器 組み合わせ
  - バウンディングボックス 融合
  - LibreEnsemble
  - 物体検出 アンサンブル python
  - min_votes
last_verified: 1.5.0
verification: >-
  コンストラクターと呼び出しのシグネチャ、デフォルト、検証エラー、クラス空間の統合、投票数、返されるResultsはlibreyolo/ensemble/model.pyで確認しました。融合アルゴリズムと引数はlibreyolo/ops/fusion.pyで確認しました。設計意図はdocs/adr/0004-model-ensembling.mdで確認しました。使用パターンはtests/unit/test_ensemble.pyとtests/unit/test_ops_fusion.pyに照らして確認しました。
snippets:
  basic:
    - label: 2つの検出器を融合
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        # メンバーにはチェックポイントのパスまたは読み込み済みモデルを指定可能
        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        result = ensemble(SAMPLE_IMAGE)
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: 重みと投票要件
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(
            ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
            weights=[1.0, 1.3],   # 慣例として検証 mAP に比例
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,          # 両方のメンバーが検出したボックスだけを維持
        )

        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes), "agreed detections")
    - label: メンバーごとのしきい値
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # スカラーはすべてのメンバーに適用しリストはメンバーごとに読み取る
        result = ensemble(SAMPLE_IMAGE, conf=[0.3, 0.5], iou=0.5)
        print(len(result.boxes))
  external:
    - label: LibreYOLOが読み込まない検出器を追加
      language: python
      code: |
        from libreyolo import ExternalDetector, LibreEnsemble, SAMPLE_IMAGE

        def my_detector(pil_image):
            # (boxes, scores, labels) を返す xyxy は元画像のピクセル単位
            return ([[100.0, 100.0, 200.0, 300.0]], [0.9], [0])

        external = ExternalDetector(my_detector, names={0: "person"})

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", external])
        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes))
  sources:
    - label: 単一モデルと同じソース
      language: python
      code: |
        from libreyolo import LibreEnsemble

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # clip.mp4 をディスク上の動画ファイルに置き換える
        for result in ensemble("clip.mp4", stream=True, vid_stride=2):
            print(result.frame_idx, len(result.boxes))
source_hash: 6dcd2f84ec6f3f65
---

## アンサンブルとは

`LibreEnsemble`は2つ以上の検出器を受け取り、それぞれを同じ画像に対して実行し、ボックスを1つの`Results`へ融合します。推論時に使う構成です。学習するものはなく、各メンバーは個別に検証およびエクスポートできる独立したモデルのままです。

サポートするタスクは検出だけです。それ以外のタスクを持つメンバーを指定すると、構築時にメンバーのインデックスとタスクを示して`ValueError`が発生します。

どちらの名前も遅延インポートされるため、使用するまでコストはかかりません。

```python
from libreyolo import LibreEnsemble, ExternalDetector
```

## 構築

<code-tabs name="basic" />

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

`members`は2つ以上の要素を持つシーケンスです。`str`または`Path`の項目は`LibreYOLO()`で読み込まれます。それ以外は呼び出し可能で、`names`辞書を公開する必要があります。2つ未満では`ValueError`が発生し、文字を反復処理する代わりに、単独の文字列を渡すと`TypeError`が発生します。

`weights`のデフォルトは`None`で、均一な重み付けを意味します。重みを指定する場合はメンバーごとに1つずつ必要で、厳密に正でなければなりません。そのため、重みをゼロにすると暗黙にメンバーを除外せず、例外が発生します。記載されている慣例は、各メンバーの検証mAPに比例させることです。

`fusion_iou`のデフォルトは`0.55`で、異なるメンバーのボックスを同じクラスターにまとめるIoUです。呼び出しごとの`iou`とは異なるしきい値であり、後者は各メンバー固有のNMS設定です。

`min_votes`のデフォルトは`1`で、どれか1つのメンバーだけでもボックスを残せることを意味します。値を上げると、その数の異なるメンバーによって確認されたクラスターだけを残します。正の整数で、メンバー数以下でなければなりません。また、各クラスを実際に認識するメンバー数を上限とするため、1つのメンバーだけが学習したクラスが暗黙に削除されることはありません。

## 融合方法

名前で指定できる方法は3つあり、呼び出し可能オブジェクトも指定できます。

| `fusion` | 動作 |
|---|---|
| `"wbf"` | 順次処理で論文 [1] に忠実なweighted boxes fusion。デフォルト |
| `"wbf_seeded"` | 1パスのweighted boxes fusion。クラス対応NMSでクラスターのシードを選択 |
| `"nms"` | すべてのメンバーのボックスを連結し、クラス対応NMSを実行 |

[1] Roman Solovyev, Weimin Wang, Tatiana Gabruseva, ["Weighted boxes fusion:
Ensembling boxes from different object detection models"](https://arxiv.org/abs/1910.13302),
arXiv:1910.13302.

Weighted boxes fusionは、信頼度で重み付けしてクラスターの座標を平均し、どの単独メンバーも提案していないボックスを生成します。2つの重み付きバリアントは、クラスターが明確なら一致し、重なり合うクラスターの連鎖ではわずかに異なることがあります。`"nms"`は平均ではなく残すボックスを選ぶため、残ったボックスは元のスコアを維持し、重みはどのボックスが勝つかだけに影響します。クラスター化せずに選択するので投票数を数えられません。`fusion="nms"`と`1`より大きい`min_votes`を組み合わせると`ValueError`が発生します。

Weighted boxes fusionは、クラスターを支持したメンバーの重みの割合でスコアを再スケーリングします。等しい重みのメンバーが2つある場合、片方だけが見つけたボックスのスコアは半分になり、`0.9`は`0.45`になります。したがって、融合後の信頼度が各メンバーの実行時に使った`conf`を下回ることがあります。メンバーのしきい値がそのまま維持されると仮定せず、融合後のスコアでフィルタリングしてください。

## クラス一覧が異なるメンバー

メンバー間でクラス一覧を共有する必要はありません。各ラベル空間は名前によって和集合になり、各メンバーには自身のクラスIDを和集合へ再マッピングするルックアップ表が与えられます。`ensemble.names`がその和集合であり、返される`Results`にも設定されます。

ボックスは同じクラス名の中でのみ融合されます。1つのメンバーだけが認識するクラスは融合されずに通過し、そのことでペナルティを受けません。スコアの再スケーリングではクラスごとの分母を使うため、単独で認識されるクラスはスコアを維持します。

一部だけが重なる場合、すべてのメンバーで共有されないクラス名を示す警告がログに記録されます。この警告を注意深く確認してください。`class_0`のようなプレースホルダーをクラス名に持つチェックポイントでは、ほかのすべてのメンバーと交わらない和集合が作られ、メンバーをまたぐ融合が一切行われないためです。

メンバーが自身の`names`にないクラスIDを返すと`RuntimeError`が発生します。

## 外部の検出器

<code-tabs name="external" />

`ExternalDetector(fn, names)`は、PIL画像を受け取り`(boxes, scores, labels)`を返す任意の呼び出し可能オブジェクトをラップします。ボックスは元画像のピクセル単位のxyxyです。引数の個数、ボックス形状、長さの一致、各クラスIDが`names`に存在することを検証し、`conf`のしきい値を自身で適用します。

これにより、LibreYOLOが読み込まなかった検出器も融合に参加できます。

## 呼び出し

<code-tabs name="sources" />

呼び出しシグネチャは単一モデルと同様で、画像、フォルダー、リスト、動画、画面キャプチャ、Webカメラ、ネットワークストリームという同じソースを受け付けます。ライブソースには、ほかの場合と同じ理由で`stream=True`が必要です。

| 引数 | デフォルト | 注記 |
|---|---|---|
| `conf` | `0.25` | メンバーごと。スカラーは全体に適用し、またはメンバーごとに1つ指定 |
| `iou` | `0.45` | 各メンバー固有のNMSしきい値。融合しきい値ではない |
| `imgsz` | `None` | `list`はメンバーごとに読み取り、`int`またはタプルは全体に適用 |
| `device` | `None` | スカラーまたはメンバーごとに1つ。メンバーを異なるデバイスに配置可能 |
| `classes` | `None` | 和集合のクラスIDに基づいて融合後の結果をフィルタリング |
| `max_det` | `300` | 融合後の結果に適用 |

`imgsz`では`list`がメンバーごとの指定を意味するため、`imgsz=[480, 640]`は1番目のメンバーに480、2番目に640を使い、`imgsz=(480, 640)`は全メンバーに対する1つの長方形サイズになります。この違いは混同しやすい点です。

指定内容にかかわらず、メンバーは少なくとも300の`max_det`で呼び出されます。そのため、各メンバーは余裕を持って実行され、アンサンブルが最後に1回だけ絞り込みます。

画像は1回だけデコードされ、同じオブジェクトがすべてのメンバーに渡されます。`batch`は互換性のために受け付けられますが無視され、画像は順次処理されます。

## 返されるもの

単一モデルが返すものと同じ通常の`Results`で、`names`にはクラス空間の和集合が設定されます。[結果の操作](/docs/predict/results)のすべてがそのまま適用されます。

1つだけ異なるのは`result.speed`で、アンサンブルでは値が設定されます。キーは`member_0`、`member_1`のように続き、さらに`fusion`があり、単位はミリ秒です。ライブラリ内で`speed`が埋められるのはここだけです。

有限でないボックスやスコアを含む行は、融合前に削除されます。メンバーが異なるデバイスにある場合、何らかの結果を最初に返したメンバーのデバイスで融合が実行されます。

## アンサンブルでできないこと

`val()`と`export()`はどちらも`NotImplementedError`を発生させ、メンバーを示します。各メンバーを個別に検証およびエクスポートしてください。`train`メソッドは存在しないため、呼び出すと`AttributeError`が発生します。

半精度はアンサンブルレベルでは処理されません。`half=True`はほかの場所と同じ警告付きの何もしない経路に入り、各メンバーで精度を設定する必要があります。

アンサンブル用のコマンドラインインターフェイスはありません。Python APIです。

