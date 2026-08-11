---
title: OpenVINO
seo_title: LibreYOLOからOpenVINO IRへエクスポート
description: >-
  LibreYOLOモデルをOpenVINO IRへ変換します。model.xmlとmodel.binの組、FP16の重み圧縮、NNCF
  INT8、CPU、GPU、NPUでの推論を説明します。
lead: >-
  OpenVINO
  IRはIntelのランタイム形式で、model.xmlグラフとmodel.binの重みblobから構成されます。LibreYOLOは中間ONNXをエクスポートし、ov.convert_modelで変換して、同じディレクトリにmetadata.yamlを書き出します。
keywords:
  - yolo openvino エクスポート
  - openvino ir 変換
  - model.xml model.bin 使い方
  - ov.convert_model yolo
  - nncf int8 量子化
  - openvino npu 推論
  - compress_to_fp16 使い方
last_verified: 1.5.0
meta:
  - label: フラグ
    value: export(format="openvino")
    mono: true
  - label: 出力
    value: model.xml、model.bin、metadata.yamlを含むディレクトリ
  - label: 追加パッケージ
    value: 'pip install "libreyolo[onnx,openvino]"'
    mono: true
  - label: 再読み込み
    value: LibreYOLO("weights/LibreYOLO9t_openvino")
    mono: true
  - label: 形状
    value: 中間ONNXに従う。dynamic=Trueの場合は動的バッチ
  - label: 精度
    value: FP32、FP16の重み圧縮（half=True）、NNCFによるINT8（int8=Trueとdata=）
verification: >-
  devブランチのlibreyolo/export/openvino.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/openvino.py、pyproject.tomlを参照。
snippets:
  install:
    - label: インストール
      language: bash
      code: |
        # IRは中間ONNXから変換するため、両方の追加パッケージが必要
        pip install "libreyolo[onnx,openvino]"
    - label: INT8ではNNCFも必要
      language: bash
      code: |
        pip install nncf
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # ディレクトリweights/LibreYOLO9t_openvinoを書き出し
        path = model.export(format="openvino")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format openvino
    - label: 引数
      language: python
      code: |
        model.export(
            format="openvino",
            imgsz=640,
            batch=1,
            dynamic=False,    # TrueはIR全体で動的バッチ軸を維持
            half=False,       # TrueはFP16の重みを保存
            int8=False,       # TrueはNNCFの学習後量子化を実行
            data=None,        # int8=Trueの場合に必須
            output_path=None, # Noneはweights/<stem>_openvinoに書き出し
        )
  int8:
    - label: キャリブレーションデータを使うINT8
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="openvino",
            int8=True,
            data="coco128.yaml",   # 必須。この形式にはデフォルトなし
            fraction=1.0,
        )
  run:
    - label: LibreYOLO経由
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_openvino")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: デバイスを選択
      language: python
      code: |
        from libreyolo import LibreYOLO

        # "auto"と"cpu"はCPU、"gpu"と"cuda"はGPUに対応
        # それ以外は大文字に変換して渡す。例: "npu" -> NPU
        model = LibreYOLO("weights/LibreYOLO9t_openvino", device="gpu")
    - label: OpenVINOを直接使用
      language: python
      code: >
        import numpy as np

        import openvino as ov

        import yaml


        core = ov.Core()

        print(core.available_devices)


        compiled = core.compile_model("weights/LibreYOLO9t_openvino/model.xml",
        "CPU")

        outputs = compiled(np.zeros((1, 3, 640, 640), dtype=np.float32))

        print([tensor.shape for tensor in outputs.values()])


        # クラス名、タスク、入力サイズはIRと並ぶmetadata.yamlに格納

        meta =
        yaml.safe_load(open("weights/LibreYOLO9t_openvino/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # この経路では前処理と後処理を自分で実装
  support:
    - label: エクスポート前にファミリーとタスクを確認
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 519816615e3aca3c
---

## インストール

<code-tabs name="install" />

変換は中間ONNXを経由するため、`onnx`追加パッケージは任意の補助ではなく要件の一部です。
NNCFは別にインストールし、`int8=True`の場合だけ必要です。

## エクスポート

<code-tabs name="export" />

成果物はファイルではなくディレクトリです。`weights/LibreYOLO9t_openvino`には
`model.xml`、`model.bin`、`metadata.yaml`が入り、`half=True`の場合は接尾辞の前に
`_fp16`が挿入されます。ディレクトリ全体を移動またはコピーしてください。3個のファイルで
1つの成果物です。

`half=True`は保存時に`compress_to_fp16`を設定します。これはIR内の重み圧縮であり、
実行時にデバイスが選ぶ推論精度の変更ではありません。

### INT8

<code-tabs name="int8" />

`int8=True`はmixedプリセットを使い、LibreYOLOのキャリブレーションローダーに対して
NNCFの学習後量子化を実行します。`data`は必須です。この形式には8枚の画像への
フォールバックがありません。NNCFがない場合は、インストールコマンドを示す`ImportError`が
発生します。

## 成果物を実行

<code-tabs name="run" />

`LibreYOLO()`は`model.xml`を含むディレクトリを認識し、チェックポイントと同じ
`Results`オブジェクトを返します。クラス名、タスク、入力サイズ、姿勢スキーマは
`metadata.yaml`から読み取ります。

デバイス文字列はそのまま渡されず、対応付けられます。`auto`と`cpu`はどちらもCPU向け、
`gpu`と`cuda`はどちらもGPU向けにコンパイルします。それ以外の値は大文字に変換して
OpenVINOへ渡され、この方法でNPUをターゲットにできます。

3番目のスニペットはLibreYOLOをインストールしていない場合に使います。この経路では前処理、
デコード、NMS、座標の再スケーリングを自分で実装します。クラス名は`metadata.yaml`にだけ
存在します。

## 制約

`metadata.yaml`がないIRも読み込めますが、その場合バックエンドは80クラスの物体検出タスクへ
フォールバックします。ほかの用途では誤った設定になるため、ディレクトリを完全な状態に保ってください。

トレース前に拒否される対象は、YOLO9セグメンテーション、RTMDet-Insセグメンテーション、
SSD、Faster R-CNNとRetinaNetの物体検出、BiRefNetまたはFeyNobgのマッティングです。
OpenVINO 2026.2は、共有マットデコーダーの標準ONNX演算`DeformConv-19`を変換できません。

検証済みでも拒否対象でもない組み合わせでは変換経路を利用できますが、プロジェクトはOpenVINO
ランタイムとの同等性を記録していません。いくつかの組み合わせは明示的な条件付きで検証済みです。
たとえば、OpenVINO 2026.2でCPUのデフォルト推論精度を使う固定520×520入力のDeepLabV3
セマンティックセグメンテーションや、固定448×448の顔クロップを使うL2CS視線推定です。
`libreyolo formats`は組み合わせごとにその条件を表示します。

ファミリーとタスクの完全な対応表は
[エクスポート対応表](/docs/reference/export-matrix)を参照してください。1つの組み合わせを調べるには次を実行します。

<code-tabs name="support" />

