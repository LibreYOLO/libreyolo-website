---
title: Ensemble API
seo_title: LibreEnsemble APIと融合演算
description: >-
  LibreEnsemble、ExternalDetector、およびlibreyolo.opsの3つの融合演算を説明します。weighted boxes
  fusion、そのseededバリアント、クラス対応NMS融合を扱います。
lead: >-
  LibreEnsembleは複数の検出器を同じ画像に対して実行し、検出結果を1つのResultsへ融合します。融合は各メンバー固有の後処理後に行われるため、メンバーはそれぞれの入力サイズ、正規化、抑制を維持します。
keywords:
  - LibreEnsemble
  - weighted boxes fusion
  - wbf
  - ExternalDetector
  - libreyolo.ops.fusion
  - min_votes 合意
last_verified: 1.5.0
verification: >-
  シグネチャとデフォルトはv1.5.0のlibreyolo/ensemble/model.pyとlibreyolo/ops/fusion.pyで確認しました。設計意図はdocs/adr/0004-model-ensembling.mdで確認しました。
snippets:
  usage:
    - label: 2つのメンバーとデフォルトの融合
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])

        # 単一画像ソースはリストではなく1つの Results を返す
        result = ens(SAMPLE_IMAGE, conf=0.25)

        print(result.boxes.xyxy)
        print(result.speed)
    - label: 合意とメンバーごとのしきい値
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(
            ["LibreYOLO9t.pt", "LibreYOLO9s.pt"],
            weights=[1.0, 2.0],
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,
        )
        result = ens(SAMPLE_IMAGE, conf=[0.25, 0.4])
        print(len(result))
  ops:
    - label: モデルを使わない融合演算
      language: python
      code: >
        import torch

        from libreyolo.ops import weighted_boxes_fusion


        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [12.0, 11.0, 51.0,
        49.0]])

        scores = torch.tensor([0.9, 0.8])

        labels = torch.tensor([0, 0])

        model_ids = torch.tensor([0, 1])


        fused = weighted_boxes_fusion(
            boxes, scores, labels, model_ids, num_models=2, iou_thr=0.55
        )

        print(fused)
source_hash: 3834f628efb1193d
---

## LibreEnsemble

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

| 引数 | デフォルト | 意味 |
|---|---|---|
| `members` | | 2つ以上の検出器 |
| `weights` | `None` | メンバーごとの信頼係数。省略時はすべて`1.0` |
| `fusion` | `"wbf"` | `"wbf"`、`"wbf_seeded"`、`"nms"`、または呼び出し可能オブジェクト |
| `fusion_iou` | `0.55` | 融合クラスター化のIoUしきい値 |
| `min_votes` | `1` | 少なくともこの数のメンバーが確認したボックスだけを維持 |

メンバーには、`LibreYOLO()`ファクトリーで解決する重みパス、構築済みモデル、エクスポート済みバックエンド、`ExternalDetector`を指定できます。すべてのメンバーが検出タスクのモデルでなければなりません。

<code-tabs name="usage" />

構築時には、メンバーが2つ未満、`weights`リストの長さが不正、重みが正でない、`min_votes`が正の整数でない、`min_votes`がメンバー数より大きい場合を拒否します。`fusion="nms"`と`min_votes > 1`の組み合わせでも例外が発生します。NMSはクラスターの所属情報を破棄し、投票数を数えられないためです。

`weights`は各メンバーに置く信頼度をスケーリングします。重みが大きいほど、融合後の座標とスコアがそのメンバーに近づきます。検証mAPに比例させるのが慣例です。

## クラス空間

同一の`names`を持つメンバーはそのまま通過します。それ以外では、クラス空間を名前で和集合にし、ルックアップ表を使ってメンバーのクラスIDを再マッピングし、融合後の`Results.names`をその和集合にします。融合でボックスを統合するのは同じ統合済みクラス内だけなので、1つのメンバーだけが認識するクラスは融合されずに通過します。不一致があると構築時に警告がログへ記録されます。

`min_votes`はクラスごとに、そのクラスをラベル空間に含むメンバー数を上限とします。そのため、一部だけで共有される語彙でも合意が意味を維持します。

## アンサンブルの呼び出し

```python
ens(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    output_path=None,
    color_format="auto",
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    **kwargs,
)
```

`predict`は`__call__`の別名です。返り値は通常の`Results`で、`speed`がコストをメンバーごとに分解し、`fusion`項目を追加します。単一画像ソースは1つ、リストまたはディレクトリはリスト、`stream=True`はジェネレーターを返します。

`conf`、`iou`、`device`はすべてのメンバーに適用され、メンバーごとの値も受け付けます。そのため、`conf=[0.25, 0.4]`ではメンバー0のしきい値が0.25、メンバー1が0.4になります。`imgsz`はintまたはタプルなら全体に適用され、リストの場合だけメンバーごとになります。したがって、`imgsz=(480, 640)`は全員に対する1つの長方形サイズで、`imgsz=[480, 640]`はメンバー0に480、メンバー1に640を使います。各項目はそのメンバーのファミリーに対して有効でなければなりません。

`augment`はテスト時拡張をサポートするメンバー全体に適用され、エクスポート済みバックエンドは無視します。`classes`は和集合のクラスIDを受け取り、`max_det`は融合後の結果に適用されます。そのため、メンバーは余裕を持って実行され、アンサンブルが1回だけ絞り込みます。`batch`はAPIの互換性のために受け付けられ、画像は順次処理されます。

`val()`と`export()`は`NotImplementedError`を発生させます。メンバーを個別に検証およびエクスポートしてください。

## ExternalDetector

```python
ExternalDetector(fn: Callable, names: dict[int, str])
```

任意の検出用呼び出し可能オブジェクトをメンバーへ適応させます。`fn`はPIL画像を受け取り`(boxes, scores, labels)`を返します。ボックスは元画像のピクセル単位のxyxyで、ラベルは`names`内で有効なクラスIDです。テンソル、配列、入れ子のリストがすべて動作します。LibreYOLOは外部コードから何もインポートしません。

アダプターは返り値を検証します。3要素のタプルで、ボックスの形状が`(N, 4)`、3つの配列が同じ長さ、すべてのクラスIDが`names`内に存在する必要があります。`conf`以下の検出は融合前に削除されます。

## 融合演算

融合プリミティブは`libreyolo.ops`内の独立したtorch演算です。モデルを使わず、単独でインポートできるため、アンサンブルとは別にエクスポートされています。

<code-tabs name="ops" />

3つすべてが同じ位置引数`boxes, scores, labels, model_ids`を受け取り、`(boxes, scores, labels)`を返します。

| 演算 | レジストリキー | 動作 |
|---|---|---|
| `weighted_boxes_fusion` | `wbf` | 順次処理で論文に忠実なweighted boxes fusion |
| `wbf_seeded` | `wbf_seeded` | 同じ縮約の並列1パスバリアント |
| `nms_fusion` | `nms` | すべてを連結し、クラス対応NMSを適用 |

`FUSIONS`は3つのレジストリキーを呼び出し可能オブジェクトへマッピングし、`LibreEnsemble`はそこで`fusion=`を検索します。

```python
weighted_boxes_fusion(
    boxes, scores, labels, model_ids,
    *,
    weights=None,
    num_models=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type="avg",
    min_votes=1,
    models_per_label=None,
    label_weights=None,
)
```

`wbf_seeded`は同一のシグネチャを取ります。`nms_fusion`は`conf_type`以外の同じ引数を取り、`min_votes > 1`では`ValueError`を発生させます。

`weighted_boxes_fusion`では、検出を重みでスケーリングした信頼度の降順で処理します。各検出は、`iou_thr`を上回るIoUを持つ同じラベルの既存クラスターのうち、融合途中のボックスと最もよく重なるものに加わるか、新しいクラスターを開始します。クラスターの融合ボックスはメンバー座標の信頼度加重平均で、スコアは信頼度の加重平均または最大値です。さらに、少ないモデルだけが確認したボックスのスコアが低くなるように再スケーリングされます。

`wbf_seeded`は`iou_thr`でクラス対応NMSを使ってクラスターのシードを選択し、各検出を同じラベルで最もIoUが高いシードに割り当て、その後で各クラスターを同じ方法で縮約します。クラスター形状は処理途中に変化しないため、演算全体が固定形状のテンソル演算になります。2つのバリアントはクラスターが明確なら一致し、重なり合うクラスターの連鎖ではわずかに異なることがあります。

`nms_fusion`は各重なりグループで信頼度が最も高いボックスを変更せずに維持します。モデルごとの`weights`は抑制順位付けのために信頼度をスケーリングするだけで、残ったボックスは元のスコアを維持します。

## カスタム融合

`fusion=`は上の演算と同じシグネチャを持つ呼び出し可能オブジェクトも受け付けます。名前がある場合は`ens.fusion`に記録され、ない場合は`"custom"`になります。返り値は検証され、一貫した形状を持つ`(boxes, scores, labels)`の3要素でなければなりません。

