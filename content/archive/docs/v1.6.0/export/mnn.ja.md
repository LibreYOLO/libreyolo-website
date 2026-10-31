---
title: MNN
seo_title: LibreYOLOからMNNへエクスポート
description: >-
  LibreYOLOの検出モデルをONNXとmnnconvertを経由してMNNにエクスポートします：固定のNCHW形状、CPUでのFP32、そしてランタイム契約が要求するメタデータのサイドカー。
lead: >-
  MNNはAlibabaの軽量な推論エンジンです。LibreYOLOは静的なONNXグラフをエクスポートし、MNNパッケージに同梱されるmnnconvertツールで変換したうえで、入力名と出力名、固定された入力形状、クラス名を記録したJSONのサイドカーを書き出します。
keywords:
  - yolo mnn エクスポート
  - mnnconvert 変換
  - mnn 推論 python
  - モバイル 物体検出 推論
  - nchw 固定形状
last_verified: 1.5.0
meta:
  - label: フラグ
    value: export(format="mnn")
    mono: true
  - label: 出力
    value: .mnnファイル1つと、メタデータのサイドカーである.mnn.jsonが1つ
  - label: 追加インストール
    value: 'pip install "libreyolo[mnn]"'
    mono: true
  - label: 再読み込み
    value: LibreYOLO("weights/LibreYOLO9t.mnn")
    mono: true
  - label: 形状
    value: 固定のNCHWです。dynamic=Trueは拒否されます。
  - label: 数値精度
    value: FP32のみ、CPUのみです。
  - label: タスク
    value: このバージョンでは検出のみ
verification: >-
  devブランチのlibreyolo/export/mnn.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/mnn.py、pyproject.tomlを確認しました。
snippets:
  install:
    - label: インストール
      language: bash
      code: |
        # MNNはONNXの中間表現から変換するため、この追加インストールにはlibreyolo[onnx]が含まれる
        pip install "libreyolo[mnn]"
    - label: コンバーターがパス上にあるか確認
      language: bash
      code: |
        mnnconvert --version
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t.mnnとweights/LibreYOLO9t.mnn.jsonを出力
        path = model.export(format="mnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format mnn --imgsz 640
    - label: 引数
      language: python
      code: |
        model.export(
            format="mnn",
            imgsz=640,        # int、または (height, width)
            batch=1,          # 成果物に焼き込まれる
            simplify=True,    # ONNXの中間表現に対してonnxsimを適用
            output_path=None, # Noneならweights/<stem>.mnnに出力
            verbose=False,    # Trueにするとmnnconvertのログを逐次出力
        )

        # dynamic=TrueはValueErrorを送出。half=Trueとint8=Trueは拒否される
  run:
    - label: LibreYOLOから実行
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.mnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: MNN単体
      language: python
      code: >
        import json


        import MNN

        import numpy as np


        meta = json.load(open("weights/LibreYOLO9t.mnn.json"))

        print(meta["mnn_input_names"], meta["mnn_output_names"],
        meta["mnn_input_shape"])


        runtime = MNN.nn.create_runtime_manager(
            ({"backend": 0, "precision": 1, "numThread": 4},)
        )

        module = MNN.nn.load_module_from_file(
            "weights/LibreYOLO9t.mnn",
            meta["mnn_input_names"],
            meta["mnn_output_names"],
            runtime_manager=runtime,
            dynamic=False,
            shape_mutable=False,
        )


        blob = np.zeros(meta["mnn_input_shape"], dtype=np.float32)

        input_var = MNN.expr.const(
            blob, list(blob.shape), MNN.expr.NCHW, MNN.expr.float
        )

        outputs = module.forward([input_var])

        for out in outputs:
            print(np.array(MNN.expr.convert(out, MNN.expr.NCHW).read()).shape)

        # この経路では前処理と後処理は自前で行う
  support:
    - label: エクスポート前にファミリーとタスクを確認
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 68fad34d07aea149
---

## インストール

<code-tabs name="install" />

変換はONNXの中間表現を経由して行われるため、この追加インストールには`libreyolo[onnx]`が
含まれます。あわせて`mnnconvert`の実行ファイルも入り、エクスポーターはこれをまず実行中の
Pythonインタープリターの隣で、次に`PATH`上で探します。コンバーターが見つからない場合は、
変換の途中で失敗するのではなく、インストールコマンドを示す`ImportError`が送出されます。

## エクスポート

<code-tabs name="export" />

グラフを渡す前に、エクスポーターはONNXの入力契約を読み取り、表現できないものを拒否します：
画像入力が複数ある場合と、入力形状にシンボリックな次元がある場合です。このバージョンのMNNは
完全に固定されたNCHW形状を必要とし、`batch`は読み込み時に調整されるのではなく成果物に
焼き込まれます。

サイドカーは省略できる付随情報ではありません。`weights/LibreYOLO9t.mnn.json`には入力名と
出力名、固定された入力形状、バッチ、クラス名、使用したMNNのバージョン、成果物を構築した
対象のバックエンドが記録され、ランタイムは読み込み時にそれらのフィールドをすべて検証します。

Windowsでは、MNN 3.6.1が変換を終えたあとにプロセスの終了処理でアクセス違反やfail-fastの
ステータスとともに異常終了することがあります。エクスポーターはそれらの特定の終了コードを
認識し、出力ファイルが存在する場合は変換が成功したものとして扱います。

## 成果物を実行する

<code-tabs name="run" />

`LibreYOLO()`は`.mnn`という拡張子で振り分け、チェックポイントと同じ`Results`オブジェクトを
返します。読み込みは設計上厳密です：サイドカーは`format=mnn`、`mnn_backend=cpu`、
`dynamic=false`、`precision=fp32`、サイズ、検出タスク、記録された画像サイズと一致する固定で
正の値のNCHW形状、そして0から`nc - 1`までのすべてのインデックスを網羅するクラス名を宣言して
いる必要があります。食い違いがあれば、推測せずに例外を送出します。

成果物を構築したときと異なる`imgsz`で推論した場合も例外になり、ここではMNNのエクスポートは
CPUで動作するため`device`は警告とともに無視されます。

2つ目のスニペットはランタイムを直接使う経路です。この経路では前処理、デコード、NMS、座標の
スケール戻しは自前の作業になり、MNNのモジュールローダーが明示的に要求するため、入力名と
出力名はサイドカーから取得します。

## 制約

検出のみです。バックエンドは読み込み時にそれ以外のタスクを拒否し、エクスポート側もそれに
合わせて、記録された組み合わせ以外では事前チェックが「MNN v1 has no implemented runtime
contract for this family and task.」という例外を送出します。

FP32、CPU、固定形状です。`dynamic=True`は`ValueError`を送出し、`half=True`と`int8=True`は
検証中に拒否されます。

検証済みの検出ファミリーはYOLO9、YOLO9-E2E、YOLO9-P2、RF-DETR、EC、RT-DETR、RT-DETRv2、
RT-DETRv4、D-FINE、DEIM、YOLO-NASで、いずれも変換、成果物の新規読み込み、MNNのCPUでの実行、
メタデータの確認、PyTorchモデルに対するNMS後の検出結果の一致まで確認しています。DEIMv2は
変換、再読み込み、実行ができ、NMS後の検出結果も保たれますが、中間のONNX経路でクエリ単位の
スコアの一致が不完全なため、検証済みではなく利用可能として記録されています。

ファミリーとタスクの全体表は[エクスポート対応表](/docs/reference/export-matrix)を
参照してください。1つの組み合わせを確認するには：

<code-tabs name="support" />
