---
title: しきい値とフィルタリング
seo_title: LibreYOLOのconf、iou、max_det
description: >-
  推論時にconf、iou、max_det、classes、agnostic_nmsが行う処理と、独自のNMSを実行しないためiouを無視するファミリーを解説します。
lead: >-
  どの予測を残すかはconf、iou、max_det、classesという4つの引数で決まります。集合予測器は固定されたクエリ集合をデコードしてNMSを実行しないため、すべてのファミリーに適用される引数はこのうち2つだけです。
keywords:
  - YOLO conf しきい値
  - IoU しきい値 NMS
  - max_det
  - Python 物体検出 クラス絞り込み
  - agnostic NMS
  - NMSなし DETR
  - 物体検出 信頼度しきい値
  - 推論 クラスフィルタ
last_verified: 1.6.0
verification: >-
  libreyolo/models/base/inference.pyのInferenceRunner.__call__からデフォルト値を引用しました。ファミリーごとのNMS動作はlibreyolo/postprocess/内の全モジュールから確認し、libreyolo/backends/base.pyの_is_nms_free_familyと照合しました。クラスのフィルタリングはInferenceRunner._apply_classes_filterと_wrap_results、agnostic_nmsはlibreyolo/utils/predict_args.pyのcheck_agnostic_nmsとpostprocess_max_det、およびInferenceRunner._wrap_resultsのクラス非依存ステップ、オープンボキャブラリーの処理はlibreyolo/models/openvocab/base.pyのNMS_THRESHOLD、検証のデフォルト値はBaseModel.valから確認しました。
snippets:
  basic:
    - label: 4つの引数
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(
            SAMPLE_IMAGE,
            conf=0.25,      # このスコア以上の予測を残す
            iou=0.45,       # NMSを実行する場合の重なりしきい値
            max_det=300,    # 画像ごとの上限
            classes=None,   # またはクラスIDのリスト
        )
        print(len(result.boxes))
    - label: confを変えて比較
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        for conf in (0.1, 0.25, 0.5, 0.75):
            result = model(SAMPLE_IMAGE, conf=conf)
            print(conf, len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt conf=0.4 iou=0.5 max_det=100 \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  classes:
    - label: 特定クラスに絞り込み
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # クラスIDはmodel.namesのインデックス COCOでは0がperson
        result = model(SAMPLE_IMAGE, classes=[0])

        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: 名前からIDを検索
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        wanted = {"person", "backpack"}
        ids = [i for i, name in result.names.items() if name in wanted]
        print(ids)

        filtered = model(SAMPLE_IMAGE, classes=ids)
        print(len(filtered.boxes))
  nmsfree:
    - label: NMSを実行しないファミリーでiouを指定
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # RF-DETRは固定クエリ集合をデコードするためここではiouを変えても効果がない
        model = LibreYOLO("LibreRFDETRs.pt")

        loose = model(SAMPLE_IMAGE, iou=0.9)
        tight = model(SAMPLE_IMAGE, iou=0.1)

        # どちらも同じ件数となる 有効な制御はconfとmax_det
        print(len(loose.boxes), len(tight.boxes))
  agnostic:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        per_class = model(SAMPLE_IMAGE)
        agnostic = model(SAMPLE_IMAGE, agnostic_nms=True)

        # 重なり合う異なるクラスのボックスは最もスコアの高い1つにまとまる
        print(len(per_class.boxes), len(agnostic.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt agnostic_nms=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: fa0e6cee02b3e336
---

## 4つの引数

| 引数 | デフォルト | 適用対象 |
|---|---|---|
| `conf` | `0.25` | すべてのファミリー |
| `iou` | `0.45` | Non-Maximum Suppressionを実行するファミリー |
| `max_det` | `300` | すべてのファミリー |
| `classes` | `None` | すべてのファミリー |

<code-tabs name="basic" />

このうち2つはすべてに適用され、2つはそうではありません。パラメーター調整を始める前に知っておくべき最も重要な点です。

検証では意図的に異なるデフォルト値を使います。`val()`は`conf=0.001`と`iou=0.6`で実行されます。平均適合率は適合率と再現率の完全な曲線に対して計算され、0.25で打ち切ると曲線が途中で切れるためです。

## conf

`conf`は、それを下回るスコアの予測を破棄するしきい値です。NMSをまったく実行しないものも含め、すべてのファミリーに適用されます。検出数が多すぎる、または少なすぎる場合に最初に調整する項目です。

デフォルトの`0.25`は画像を目視する用途に適しています。下流システムへ渡す場合は通常より高い値が必要で、精度測定にははるかに低い値が必要です。

物体検出の検証は、IoU 0.50でのF1に基づく`metrics/best_conf`とクラス別のしきい値を公開します。デプロイ時のしきい値を決めるための、測定に基づく出発点として使ってください。[検証](/docs/train/validation)を参照してください。セグメンテーションはこれらのキーを公開しません。

## iou

`iou`は、同じクラスに属する2つのボックスが重なったときに、Non-Maximum Suppressionが低スコア側を削除する重なりのしきい値です。ファミリーが抑制処理を実行する場合だけ意味があります。

集合予測器は固定数のクエリをデコードし、スコア上位のものを選びます。重複は後処理ではなくアーキテクチャ内部で学習時に抑えられるため、調整できるしきい値はありません。次のファミリーはAPI互換性のため`iou`を受け付けますが無視します。

CenterNet、DEIM、DETR、Deformable DETR、D-FINE、DINO-DETR、EdgeCrafter、Faster R-CNN、LW-DETR、Mask R-CNN、RF-DETR、RT-DETR、エンドツーエンドYOLOv9ヘッドです。これらのデコーダーを基にしたバリアントも同じ動作を継承します。

<code-tabs name="nmsfree" />

多くのファミリーでは後処理のdocstringにこの点が記載されていますが、実行時には警告が出ません。そのため、RF-DETRで`iou`を変えて測定してもエラーではなく平坦な結果になります。Faster R-CNNとMask R-CNNは少し異なり、どちらもモデル内部ですでにNMSを実行しています。その固定されたアップストリームのしきい値を`iou`で変更する方法には対応していません。

YOLOv1からYOLOv4、YOLOv7、YOLOv9、YOLOX、YOLO-NAS、RTMDet、PicoDet、EfficientDet、FCOS、RetinaNet、SSDでは`iou`が使用されます。

推論時の次の3つのオプションは、いずれもモデル処理後のボックスに作用するため、集合予測器でも`iou`が影響します。

- `tiling=True`は、重なり合うタイルを`iou`でクラス別NMSにより調整します。
- `augment=True`は、反転したビューを`iou`でクラス別NMSにより統合します。
- `agnostic_nms=True`は、[後述](#agnostic_nms)するとおり、重なり合うボックスを`iou`でクラスをまたいで抑制します。

最初の2つは[推論パフォーマンス](/docs/predict/performance)で解説しています。

オープンボキャブラリー検出器には独自の規則があります。プロセッサーがNMSを実行するファミリーは固有のデフォルトしきい値を宣言し、`iou`を適用します。OMDet-Turboがこれに該当します。抑制をまったく行わないGrounding DINO、OWLv2、OV-DEIMは、`iou`を渡すと警告します。この警告を出すのはライブラリ内でこれらだけです。

## max_det

`max_det`は1枚の画像に対して返す予測数の上限です。すべてに適用されますが、その仕組みは異なります。NMSを使うファミリーは抑制後に切り詰め、集合予測器はtop-k選択のサイズとして使用します。

アップストリームの参照設定に従い、指定値より低い上限を適用するファミリーもあります。SSDは200、FCOSは固有の画像ごとの検出上限に制限されます。それ以上に`max_det`を増やしても効果はありません。

`max_det`がファミリーごとではなく中央で適用されるのは、タイルを調整した後に統合済みリストを切り詰めるタイル推論と、フィルタリングまたは抑制の後に切り詰める`classes`または`agnostic_nms=True`付きの呼び出しの2つの場合です。

## クラスのフィルタリング

<code-tabs name="classes" />

`classes`はクラスIDのリストを受け取り、そのいずれかに属する予測だけを残します。IDは`result.names`のインデックスです。データセットの順序を仮定せず、結果の`names`を読み取る方法が最も確実です。

フィルタリングは各ファミリーの後処理後に、すべての推論経路が通る共通処理で行われます。これにより2つの重要な効果があります。NMSを使わないファミリーを含め、すべてのファミリーで機能します。また、ボックスに対応するペイロードも同時に絞り込むため、マスク、キーポイント、回転ボックスが対応しない状態で残ることもありません。

コマンドラインの`classes`には、単独の整数、リスト、カンマ区切り文字列を指定できます。

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

フィルタリングだけで精度が得られるわけではありません。後で破棄するクラスの予測にもモデルは計算量を使います。`classes`を指定すると、ファミリーは300個（`max_det`の方が大きい場合は`max_det`個）の候補ボックスを確保し、`max_det`への切り詰めはフィルタリングの後に行われるため、`max_det`が小さくても目的のクラスが押し出されることはありません。エクスポートしたバックエンドも、バッチ処理のTensorRTエンジンを含めて同じ候補数を使います。不要なクラスのスコアの高いボックスが300個を超える画像では、目的のクラスに到達する前に候補が埋まる可能性があります。その場合は`max_det`を300より大きくしてください。

## agnostic_nms

各ファミリー独自のNMSはクラス別なので、デフォルトでは、異なるクラスの2つのボックスは重なっていても、どの`iou`でも両方残ります。`agnostic_nms=True`はクラス非依存のステップを1つ追加します：`iou`を超えて重なるボックスのうち、クラスに関係なく最もスコアの高いボックスだけが残ります。

<code-tabs name="agnostic" />

このステップは、ファミリーの完成した検出結果に対して、`classes`のフィルタリングの後、`max_det`への切り詰めの前に実行されます。そのため、独自のNMSを実行しない集合予測器、エクスポートしたバックエンド、`augment=True`と`tiling=True`の場合でも同じように動作します。マスクとキーポイントは対応するボックスとともに削除され、回転ボックスは回転IoUで比較されます。対象はボックスを返すタスク、つまりdetect、segment、pose、obbです。

ファミリーはこのステップの処理対象として、300個（`max_det`の方が大きい場合は`max_det`個）の候補ボックスを確保します。`max_det`が300未満の場合、抑制で空いた枠はスコアの低い候補で埋められます。デフォルトを含む300以上の場合、その枠は空のままです。

適用されない場合は次のとおりです。

- 標準の推論経路では、ボックスを持たないタスクは`ValueError`を送出します。
- `LibreEnsemble`と[HRNet](/docs/models/hrnet)は`NotImplementedError`を送出します。
- `LibreSAM`モデルと[3D検出器](/docs/tasks/3d-object-detection)（WildDet3D、3D-MOOD、FCOS3D、DetAny3D）はこの引数を受け付けず、`TypeError`を送出します。
- [L2CS-Net](/docs/models/l2cs)、[PP-OCRv5](/docs/models/pp-ocrv5)、[SAM 3D Body](/docs/models/sam-3d-body)、[LibreFaceRec](/docs/models/librefacerec)は独自の推論経路を実行し、この引数を無視します。

`val(agnostic_nms=True)`はdetectとsegmentの検証に同じステップを適用します。[検証と指標](/docs/train/validation)を参照してください。

## predictが拒否するもの

警告ではなく例外を送出する引数が2つあります。`visualize`と`embed`はどちらも`NotImplementedError`を送出します。埋め込みには`task="embed"`を指定してモデルを読み込み、通常どおり`predict`または`embed`を呼び出してください。`agnostic_nms=True`は、上記のとおり適用されない場合に例外を送出します。

認識されない引数は、対応するオプションを示す`TypeError`を送出します。そのため、入力ミスが暗黙に無視されることはありません。

次の引数は受け付けられますが、警告して破棄されます。`boxes`、`dnn`、`half`、`line_width`、`retina_masks`、`show_conf`、`show_labels`、`verbose`です。
