---
title: ONNX
seo_title: LibreYOLOからONNXへエクスポート
description: >-
  LibreYOLOモデルをONNXへエクスポートします。LibreYOLOがファミリーごとに選ぶopset、動的軸、組み込みNMS、INT8、グラフを再読み込みする方法を説明します。
lead: >-
  ONNXは移植可能なグラフ形式です。LibreYOLOはtorch.onnx.exportでモデルをトレースし、必要に応じてグラフを簡略化します。さらに、ファミリー、タスク、クラス名、入力サイズをファイル自体のメタデータに書き込むため、どのLibreYOLOバックエンドでも後処理を再構築できます。
keywords:
  - yolo onnx エクスポート
  - onnxruntime 使い方
  - torch.onnx.export yolo
  - onnx opset 選び方
  - onnx 動的軸
  - onnx nms 組み込み
  - onnx int8 qdq
  - onnx metadata_props
last_verified: 1.5.0
meta:
  - label: フラグ
    value: export(format="onnx")
    mono: true
  - label: 出力
    value: グラフにメタデータを埋め込んだ1個の.onnxファイル
  - label: 追加パッケージ
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: 再読み込み
    value: LibreYOLO("weights/LibreYOLO9t.onnx")
    mono: true
  - label: 形状
    value: Pythonではデフォルトで動的バッチ。タスクごとの例外は以下を参照
  - label: 精度
    value: FP32、FP16（half=True）、INT8（int8=True、YOLO9の物体検出）
verification: >-
  devブランチのlibreyolo/export/onnx.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/onnx.py、libreyolo/cli/commands/export.pyを参照。
snippets:
  install:
    - label: インストール
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t.onnxを書き出し
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx
    - label: 引数
      language: python
      code: |
        model.export(
            format="onnx",
            imgsz=640,        # intまたは(height, width)
            batch=1,
            dynamic=True,     # Pythonのデフォルト。CLIのデフォルトはFalse
            simplify=True,    # グラフにonnxsimを実行
            opset=None,       # Noneは13、DETR系ファミリーは17を選択
            half=False,       # FP16の重みと活性化
            int8=False,       # QDQ INT8、YOLO9の物体検出のみ
            data=None,        # キャリブレーション用data.yaml、INT8のみ
            device=None,      # トレース用デバイス。Noneはモデルのデバイスを使用
            output_path=None, # Noneはweights/<stem>.onnxに書き出し
        )
  nms:
    - label: グラフにNMSを組み込む
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLO9の物体検出のみ、バッチ1。dynamicは強制的にFalse
        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            nms=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx --nms \
          --conf 0.25 --iou 0.45 --max-det 300
  int8:
    - label: キャリブレーションデータを使うINT8
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            int8=True,
            data="coco128.yaml",   # 数百枚の代表的な画像
            fraction=1.0,
        )
  run:
    - label: LibreYOLO経由
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.onnx")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ONNX Runtimeを直接使用
      language: python
      code: >
        import numpy as np

        import onnx

        import onnxruntime as ort


        session = ort.InferenceSession(
            "weights/LibreYOLO9t.onnx",
            providers=["CPUExecutionProvider"],
        )


        # この経路では前処理と後処理を自分で実装

        batch = np.zeros((1, 3, 640, 640), dtype=np.float32)

        outputs = session.run(None, {session.get_inputs()[0].name: batch})

        print([out.shape for out in outputs])


        # グラフにファミリー、タスク、クラス名、入力サイズを格納

        meta = {p.key: p.value for p in
        onnx.load("weights/LibreYOLO9t.onnx").metadata_props}

        print(meta["model_family"], meta["task"], meta["imgsz"])
  support:
    - label: エクスポート前にファミリーとタスクを確認
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cee78250fc7189a3
---

## インストール

<code-tabs name="install" />

追加パッケージでは`onnx`、`onnxsim`、`onnxruntime`がインストールされます。ファイルの
書き出しには`onnx`だけで十分です。`onnxsim`は簡略化処理を実行し、`onnxruntime`は
成果物の実行とINT8キャリブレーションを行います。

## エクスポート

<code-tabs name="export" />

`output_path`を指定しない場合、ファイルはチェックポイントのstemを名前として
`weights/`に保存されます。その精度を要求した場合は`_fp16`または`_int8`が付加されます。

`dynamic`のデフォルトはPythonでは`True`、CLIでは`False`です。有効にするとバッチ軸が
シンボリックになり、一部のタスクではさらに多くの軸が可変になります。セマンティック
セグメンテーションではマスクの高さと幅も可変になり、Real-ESRGANの画像復元では空間軸が
可変になります。2段階検出器ではリサイズをグラフ内で行うため、入力画像の高さと幅が
可変のままになります。

`opset`を省略すると、ファミリーごとに選択されます。DETR系ファミリー（`detr`、
`deformable_detr`、`dinodetr`、`dfine`、`deim`、`deimv2`、`ec`、`lwdetr`、
`rfdetr`、`rtdetr`、`rtdetrv2`、`rtdetrv4`）に加え、`deit`、`midas`、`moge2`では、
`aten::scaled_dot_product`を変換できるopset 17が使われます。それ以外はすべて13です。
マッティングは常に19に引き上げられます。BiRefNetのデコーダーには、ONNXでopset 19から
定義される`DeformConv`演算子が必要なためです。

`simplify=True`は`onnxsim`を実行し、処理に失敗した場合は元のグラフを維持します。そのため、
簡略化エラーはエクスポート失敗ではなく警告になります。macOS arm64で`onnx` 1.22以降と
`onnxsim` 0.6.5以前を組み合わせた場合、この処理は完全にスキップされます。この組み合わせでは
Pythonプロセスが異常終了する可能性があるためです。

### 組み込みNMS

<code-tabs name="nms" />

`nms=True`はYOLO9の物体検出だけに対応し、バッチ1が必要です。`dynamic=True`とともに指定すると
警告を記録して動的軸を無効にします。この場合、グラフには2つの出力があります。`output`は
`(batch, max_det, 6)`形状で、`raw`は未デコードの検出器テンソルです。LibreYOLO自身の
バックエンドは`raw`を使うため、後処理がPyTorch経路と同一になります。

### DeepStream

`deepstream=True`はONNX専用のオプションです。NVIDIA DeepStreamのパーサーが想定する
レイアウトでグラフをエクスポートし、隣に`config_infer_primary_<stem>.txt`と
`<stem>_labels.txt`という2個のサイドカーファイルを書き出します。そのため、手作業で設定を
作らずに成果物をパイプラインへ組み込めます。

このオプションは`nms=True`と同時に使えません。両方を指定すると`ValueError`が発生します。
DeepStreamは独自のクラスタリング段階で抑制を実行するためです。ONNX以外の形式に渡した場合も
同様に例外が発生します。対応するファミリーとタスクの一覧、およびパーサーのビルド方法は
[DeepStream](/docs/export/deepstream)を参照してください。

### INT8

<code-tabs name="int8" />

`int8=True`はONNX Runtimeの静的量子化を実行し、float32の入力と出力を持つQDQグラフを
書き出します。量子化されるのは`Conv`ノードと`Gemm`ノードだけです。物体検出ヘッドの
デコードをfloat32のままにするのは意図的です。この連結処理ではピクセル単位のボックス座標と
0〜1のクラススコアが混在します。ボックスの値の大きさに支配された単一のテンソル単位の
活性化スケールを使うと、すべてのスコアが0になるためです。

このフラグは現在、YOLO9の物体検出だけに適用できます。それ以外では事前検査時に
`NotImplementedError`が発生します。`data`を省略すると警告を表示して`coco8.yaml`を
使いますが、8枚の画像は代表的なキャリブレーションセットではありません。PyTorchですでに
量子化したモデルは別の経路を使います。詳しくは[量子化](/docs/export/quantization)を参照してください。

## 成果物を実行

<code-tabs name="run" />

`LibreYOLO()`は`.onnx`拡張子に応じて処理を振り分け、`.pt`チェックポイントと同じ
`Results`オブジェクトを返します。エクスポート時にクラス名、タスク、入力サイズ、姿勢スキーマが
グラフの`metadata_props`へ書き込まれているためです。`device="auto"`では、ONNX Runtimeが
利用可能と報告した場合に`CUDAExecutionProvider`を使い、それ以外ではCPUへフォールバックします。

2番目のスニペットはLibreYOLOをインストールしていない場合に使います。この経路では前処理、
デコード、NMS、座標の再スケーリングをすべて自分で実装します。メタデータブロックは引き続き
読み取れます。

## 制約

出力テンソル名はタスクごとに固定されています。メタデータを使わないコンシューマーは、次の名前と
一致させる必要があります。

| タスク | 出力名 |
|---|---|
| 物体検出、グリッドヘッド、アンカーヘッド | `output` |
| 物体検出、DETR系 | `pred_logits`、`pred_boxes` |
| 物体検出、RF-DETR | `dets`、`labels` |
| 分類 | `output` |
| セマンティックセグメンテーション | `semantic_logits` |
| 深度 | `depth` |
| サーフェス法線 | `normal` |
| エッジ | `edges` |
| 画像復元 | `restored` |
| マッティング | `matte` |
| 視線 | `yaw_logits`、`pitch_logits` |

RF-DETRは、入力テンソル名が`images`ではなく`input`である唯一のファミリーでもあります。

このバージョンでは、いくつかのタスクに固定解像度のランタイム契約があります。深度、
サーフェス法線、エッジでは`batch != 1`を拒否して`dynamic=False`を強制します。マッティングでは
ネイティブの1024正方形を強制します。BiRefNetのSwin相対位置テーブルがその解像度に結び付いて
いるためです。画像復元ではReal-ESRGAN以外の全ファミリーで固定キャンバスを強制します。
Real-ESRGANのジェネレーターは完全畳み込み型です。

YOLO9ファミリー、HRNet、NAFNet、Real-ESRGANでは長方形の`imgsz`を使えます。固定の正方形を
必要とするファミリー（`clip`、`deformable_detr`、`detr`、`dinodetr`、`dfine`、`deim`、
`deimv2`、`ec`、`lwdetr`、`moge2`、`rtdetr`、`rtdetrv2`、`rtdetrv4`、`rfdetr`、
`siglip2`、`ssd`）では長方形を拒否します。

トレース前に拒否される組み合わせは2つあります。YOLO9はLibreYOLOで物体検出だけに対応するため、
YOLO9セグメンテーションは拒否されます。RTMDet-Insセグメンテーションは、動的カーネルによる
マスクのデコードにエクスポート先ランタイムの契約がないため拒否されます。

ファミリーとタスクの完全な対応表は
[エクスポート対応表](/docs/reference/export-matrix)を参照してください。1つの組み合わせを調べるには、
ライブラリへ直接問い合わせます。

<code-tabs name="support" />
