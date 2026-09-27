---
title: チェックポイントスキーマ
seo_title: LibreYOLOチェックポイントメタデータスキーマv1.0
description: >-
  すべてのLibreYOLO
  .ptチェックポイントが持つメタデータを説明します。必須キー、タスクごとの追加項目、エクスポートランタイムキー、量子化マニフェスト、学習フィールドを扱います。
lead: >-
  LibreYOLO .ptファイルはtorch.saveで保存したフラットな辞書です。modelキーはstate
  dictを保持し、そのほかの最上位キーは、ファイル名の解析やstate dictの推測を行わずにチェックポイントを識別するメタデータです。
keywords:
  - libreyolo チェックポイント スキーマ
  - schema_version 1.0
  - model_family
  - libreyolo チェックポイント メタデータ
  - quant manifest
  - wrap_libreyolo_checkpoint
last_verified: 1.5.0
verification: >-
  libreyoloリポジトリv1.5.0のdocs/checkpoint_schema.mdに対応し、libreyolo/utils/serialization.pyとBaseModel.saveに照らして確認しました。
snippets:
  usage:
    - label: チェックポイントからメタデータを読み取る
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.utils.serialization import unwrap_libreyolo_checkpoint

        import torch


        # チェックポイントをダウンロードしてローカルパスができるように再保存

        LibreYOLO("LibreYOLO9t.pt").save("roundtrip.pt")


        loaded = torch.load("roundtrip.pt", map_location="cpu",
        weights_only=False)

        state_dict, metadata = unwrap_libreyolo_checkpoint(loaded)


        print(metadata["schema_version"], metadata["model_family"])

        print(metadata["size"], metadata["task"], metadata["nc"],
        metadata["imgsz"])

        print(len(state_dict), "tensors")
source_hash: ce760f1bed97bfd0
---

## スキーマv1.0

公式のLibreYOLO `.pt`チェックポイントには、次の項目がすべて含まれます。

```python
{
    "model": state_dict,
    "schema_version": "1.0",
    "libreyolo_version": "0.x.y",
    "model_family": "yolo9",
    "size": "t",
    "task": "detect",
    "nc": 80,
    "names": {0: "cat", 1: "dog"},
    "imgsz": 640,
}
```

| キー | 型 | 意味 |
|---|---|---|
| `model` | state dict | モデルの重み |
| `schema_version` | str | メタデータ規約のバージョン。v1.0では文字列`"1.0"`を使用 |
| `libreyolo_version` | str | チェックポイントを生成したバージョン |
| `model_family` | str | `yolo9`、`rfdetr`、`dfine`、`ec`などの登録済みファミリー |
| `size` | str | `t`、`s`、`r18`、`atto`など、ファミリー内のバリアント |
| `task` | str | 正規タスク名 |
| `nc` | int | 正のクラス数 |
| `names` | dict | `0..nc-1`のキーを持つ`dict[int, str]` |
| `imgsz` | int | 正の正方形入力解像度、または長方形規約向けのレガシースカラー |

`task`は`detect`、`segment`、`semantic`、`panoptic`、`pose`、`classify`、`gaze`、`obb`、`point`、`depth`、`edge`、`normal`、`restore`、`matte`、`ocr`、`embed`、`mesh`のいずれかです。

公式チェックポイントはすべての`names`キーを書き込みます。リーダーはレガシーの疎なマッピングに対し、欠けたキーを`class_i`ラベルで埋めることがありますが、範囲外のキーは無効です。

長方形チェックポイントは、レガシーリーダー向けに`max(imgsz_h, imgsz_w)`を設定したスカラー`imgsz`を維持し、実際の寸法を持つ`imgsz_h`と`imgsz_w`も書き込みます。長方形フィールドを認識するリーダーは、スカラーよりこれらを優先する必要があります。HRNet姿勢推定のように固定長方形規約を持つファミリーは、互換性のない実行時サイズを拒否します。

スキーマは意図的にフラットで、`model`も意図的にstate dictです。

<code-tabs name="usage" />

## 姿勢推定の追加項目

姿勢推定は通常、`person`を持つ単一クラスの`nc: 1`です。ただし、YOLO-NASの姿勢推定ヘッドは1つの共有キーポイントスケルトンを使うマルチクラス姿勢推定にも対応します。その場合、`nc`と`names`は検出と同様にクラスを記述します。姿勢推定のランタイムエクスポートは、形状`[batch, anchors, nc]`の`scores`を出力します。

| キー | 意味 |
|---|---|
| `num_keypoints` | 姿勢推定ヘッドが使う正のキーポイント数 |
| `keypoint_dim` | `x,y`ラベルでは`2`、`x,y,visibility`ラベルでは`3`。モデル出力は常に`x,y,visibility`を公開 |
| `oks_sigmas` | キーポイントごとの任意のOKS sigma。省略時は`num_keypoints`に対応するタスクのデフォルトを使用 |
| `num_keypoints_per_class` | キーポイントテンソルをクラス単位でパディングするGroupPose形式ヘッド向けの任意のクラス別キーポイント数。キーポイントがないクラスでは`0` |

## メッシュの追加項目

メッシュチェックポイントは`task: "mesh"`、`nc: 1`、`names: {0: "person"}`を使います。パラメータの配置は人体モデル間で異なるため、寸法を仮定せず記録します。

| キー | 意味 |
|---|---|
| `body_model` | `mhr`などのパラメータ化。必須で、以下の全フィールドの解釈に使用 |
| `num_betas` | 個体と形状の係数数。MHRでは45 |
| `num_body_pose` | 身体姿勢パラメータブロックの幅。MHRでは130。rigの関節は異なる自由度を持つため、関節ごとの3要素ではなくフラットなベクトル |
| `num_vertices` | デコーダーが出力する頂点数。MHRでは18439 |
| `num_joints` | デコーダーが出力する関節数。MHRでは127 |
| `rotation_format` | MHRの`euler_zyx`や`axis_angle`など、回転のエンコード方法。3要素ベクトルだけでは曖昧なため、テンソル形状から推測しない |

## 密なタスクのプレースホルダー

複数のタスクはクラスではなく密なマップを予測するため、クラス相当のスロットはスキーマ互換性のためだけに存在します。

| タスク | `nc` | `names` |
|---|---|---|
| `depth` | 1 | `{0: "depth"}` |
| `edge` | 1 | `{0: "edge"}` |
| `restore` | 1 | `{0: "image"}` |
| `ocr` | 1 | `{0: "text"}` |

エッジ予測は`[0, 1]`内の密なfloat32確率マップです。

復元チェックポイントは、`deblur`、`denoise`、`super-resolution`などの短い破損ラベル`degradation`、`GoPro`や`SIDD`など出所を示すラベル`dataset`、出力と入力の正の整数アップスケール係数`scale`を追加できます。たとえば、x4超解像モデルでは`4`です。省略時または`1`では、復元画像が入力解像度を維持します。ランタイムはファミリーとサイズからもスケールを導出するため、`scale`は読み込み時の要件ではなく出所メタデータです。

## OCRの追加項目

`ppocr`ファミリーはtierごとに1つの複合チェックポイントを提供し、その`model` state dictは`det.*`と`rec.*`のキーネームスペースに2つのサブモデルを保持します。

| キー | 意味 |
|---|---|
| `charset` | 出力インデックス順の完全なCTCアルファベット。インデックス0がCTC blank、その後に認識辞書、空白文字が続く。ローダーは必ずチェックポイントから読み取り、別ファイルからは読み取らない |
| `pipeline` | 変換時に組み込まれるパイプラインのデフォルト。`det_limit_side_len`、`det_db_thresh`、`det_db_box_thresh`、`det_db_unclip_ratio`、`rec_image_shape`。実行時引数で呼び出しごとに上書き可能 |
| `components` | 文書方向、歪み補正、テキスト行回転などの任意パイプライン段階用に予約。v1では空 |

## エクスポートランタイムのメタデータ

エクスポートした成果物は同じ長方形の二重書き込み規則を使います。レガシースカラー`imgsz`の横に`imgsz_h`と`imgsz_w`を書き込み、長方形フィールドを認識しないリーダーはスカラーを正方形規約として暗黙に扱ってはなりません。

長方形ランタイム対応はファミリーと形式の範囲に限定されます。YOLO9ファミリー、HRNet、NAFNet、Real-ESRGANのエクスポートは、対応形式で正方形でない`imgsz_h`と`imgsz_w`を使用できます。長方形に明示対応しないファミリーや形式は、成果物を正方形として前処理せずメタデータを拒否します。HRNetのエクスポートは固定、バッチ1、FP32の人物cropヘッドで、W32は256x192、W48は384x288を受け付け、人物検出器はグラフに組み込まれません。

NMSを組み込んだエクスポートは、次のフラットなキーを追加できます。

| キー | 意味 |
|---|---|
| `nms` | 文字列のブール値。`"true"`はグラフに組み込み後処理出力が含まれることを示す |
| `nms_conf` | 組み込み出力に固定された信頼度のしきい値 |
| `nms_iou` | 組み込み出力に固定されたIoUのしきい値 |
| `max_det` | 組み込み出力が生成するNMS後の最大検出行数 |
| `nms_raw_output` | 文字列のブール値。`"true"`はグラフが補助的な未加工検出器出力も公開することを示す |

`nms=true`を指定したONNX YOLO9検出エクスポートでは、出力`0`（名前は`output`）がエクスポート時のしきい値による単独のNMS後テンソルです。`nms_raw_output=true`の場合、出力`1`（名前は`raw`）はLibreYOLOバックエンド用に予約され、ネイティブの元キャンバスへのクリップと実行時の`predict(conf=..., iou=..., max_det=...)`セマンティクスを適用できます。第三者の利用者は最初の出力を使用してください。

姿勢推定エクスポートは`num_keypoints`を追加できます。`keypoint_dim`では、テンソルにprecisionまたはclass-logitフィールドを含むGroupPose形式の未加工エクスポートが`8`などの大きな値を使うことがあります。`num_keypoints_per_class`はJSONエンコードされたリストで、キーポイントが0個のクラススロットもスキーマを定義するため維持する必要があります。`pose_input`では、`"person_crop"`がグラフはすでに切り出された1つのcropを消費し、検出器を含まないことを意味します。HRNetランタイムエクスポートにはこの値が必要です。

分類エクスポートは`crop_pct`を追加できます。中心crop比率を示す浮動小数点数で、crop前のリサイズ先は`round(imgsz / crop_pct)`となり、省略時のデフォルトは`0.875`です。`interpolation`は`"bilinear"`または`"bicubic"`で、デフォルトは`"bilinear"`です。

ExecuTorchエクスポートは、必須の`<program>.pte.json`サイドカーにフラットなメタデータを書き込みます。v1規約はCPU、FP32、バッチ1、固定入力キャンバスで、さらに`executorch_version`、`"xnnpack"`と等しい`executorch_delegate`、正の`executorch_delegate_partitions`を必要とします。別のdelegate、動的形状、FP32以外の精度を主張するサイドカーはローダーが拒否します。

MNNエクスポートは、必須の`<model>.mnn.json`サイドカーにフラットなメタデータを書き込みます。v1規約はCPU、FP32、検出専用、固定NCHW入力形状で、さらに`mnn_version`、`"cpu"`と等しい`mnn_backend`、順序付きで空でない`mnn_input_names`と`mnn_output_names`、`[batch, channels, height, width]`順の4つの正の整数としての`mnn_input_shape`、`mnn_input_shape[0]`と等しい`mnn_batch`を必要とします。動的、FP32以外、検出以外、非対応ファミリー、または一貫しない形状のメタデータはローダーが拒否します。

`.pte`と`.mnn`はバックエンド固有の成果物であり、PyTorchチェックポイントではありません。

## 量子化チェックポイント

量子化モデルは、任意のフラットキー`quant`を1つ追加します。値は`schema`、`recipe`、`keep_high_precision`、`execution`、較正の出所、`module_count`、`state`を持つマニフェスト辞書です。FP8マニフェストは`fp8_tensorwise_weights`も持つことがあり、重みのスケールが出力チャンネルごとではなくテンソル単位である`QuantLinear`モジュール名の正確な一覧です。`quant`を見つけたローダーは、`load_state_dict`の前に量子化モジュールの構造とスケーリングポリシーを再構築します。

`state`は2つの成果物形式を区別します。

デフォルトの`"prepared"`はFP32のマスター重みと`_q_*`スケールバッファを保持し、学習可能です。量子化に対応しないリーダーは`quant`キーを無視し、マスターを浮動小数点モデルとして読み込めます。

`"finalized"`は`export(format="pt")`が書き込むデプロイ形式です。マスターは削除され、代わりに各量子化モジュールがパック済み重みを保持します。

| レシピ | パック済みテンソル | 逆量子化 |
|---|---|---|
| int8 | 元の重み形状の`weight_packed` int8、チャンネルごとの`_q_w_scale` FP32 | `weight_packed * scale` |
| fp8 | 元の形状の`weight_packed` float8_e4m3fn、出力チャンネルごとに1項目の`_q_w_scale` FP32 | `weight_packed * scale` |
| w4a16, w4a8 | `weight_packed` uint8。1バイトに2つの4ビットコード、下位nibbleから、コード`q + 8`。`_q_w_gscale` FP32 `[out, ngroups]`、in_features方向に128のグループ | グループ単位のスケール |
| int2 | 1バイトに4つの2ビットコード、コード`q + 2`、グループ64 | グループ単位のスケール |
| nvfp4 | `weight_packed` uint8 `[out, ceil(in/16)*8]`、コード`sign<<3 \| E2M1 level`。`weight_block_scale` float8_e4m3fn `[out, ceil(in/16)]`。テンソルごとの`_q_w_amax` FP32 | `block_scale * amax / (448 * 6)` |
| mxfp4 | nvfp4と同様だが32要素ブロックを使い、`weight_block_exp` int8 `[out, ceil(in/32)]`を追加 | `2 ** exponent` |

int8では活性化範囲バッファ`_q_act_lo`、`_q_act_hi`、`_q_calibrated`が維持されます。マニフェストは量子化されないテンソルについて`remainder`を記録し、`"fp16"`または`"fp32"`です。展開はシミュレーションをビット単位で再現するため、finalizedの推論はfinalizeを行ったデバイス上でpreparedの推論と完全に一致します。この配置が外部エクスポーターとランタイム向けの安定した規約です。

## 学習チェックポイント

トレーナーのチェックポイントは同じ必須メタデータの核を使い、フラットな学習フィールドと再開フィールドを追加できます。

```python
{
    "model": state_dict,
    "epoch": 42,
    "optimizer": optimizer_state_dict,
    "config": {},
    "loss": 1.23,
    "best_metric_key": "metrics/mAP50-95",
    "best_metric_value": 0.51,
    "best_epoch": 39,
    "is_ema_weights": True,
    "train_model": raw_state_dict,
    "ema": ema_state_dict,
    "ema_updates": 12345,
}
```

`is_ema_weights`は最上位の`model`がEMAで平滑化されているかを宣言します。EMAが有効な場合、`train_model`、`ema`、`ema_updates`が再開状態を維持します。公開する推論用重みは簡潔にし、学習チェックポイントとして意図的に配布する場合を除き、optimizer、epoch、config、loss、EMA再開状態を含めるべきではありません。

リリース互換性のため、リーダーはレガシーの最良メトリクス別名`best_mAP50_95`、`best_mAP50`、`best_metric`、`best_metric_name`を受け付けます。

## 外部スナップショット

このスキーマはLibreYOLOが作成した`.pt`ファイルを管理します。別のモデルtierが使う複数ファイル構成のアップストリームスナップショットを改名またはラップするものではありません。

LibreMODUSサイズ`14b-a7b`は明示的な例外です。別名は`LibreVLM(...)`を通じて固定済みアップストリームファイルのディレクトリへ解決され、LibreYOLOはv1.0メタデータを追加せず、`.pt`として再公開もしません。

## レガシーおよび外部の重み

新しい書き込み処理は厳密に検証し、v1.0メタデータを出力する必要があります。メタデータがない、または不完全な場合、LibreYOLO形式に見えるレガシーチェックポイントは警告と変換手順を伴う互換経路で読み込まれ、外部のアップストリームチェックポイントは自動変換へ振り分けられます。[アップストリームチェックポイント](/docs/reference/upstream-checkpoints)を参照してください。

## ヘルパー

スキーマヘルパーは`libreyolo.utils.serialization`にあります。

```python
wrap_libreyolo_checkpoint(
    state_dict,
    *,
    model_family,
    size,
    task,
    nc,
    names=None,
    imgsz=None,
    libreyolo_version=None,
    schema_version="1.0",
    **extra_metadata,
) -> dict

validate_checkpoint_metadata(checkpoint, *, strict=False) -> list[str]

unwrap_libreyolo_checkpoint(loaded, *, strict=False) -> tuple[dict, dict]
```

`validate_checkpoint_metadata`は値を変更せず、エラーの一覧を返します。`strict=True`では代わりに`CheckpointMetadataError`を発生させます。規約に準拠したチェックポイントを書き込むためにサポートされる方法は`model.save(path)`です。

