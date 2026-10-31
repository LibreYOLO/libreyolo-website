---
title: ExecuTorch
seo_title: LibreYOLOからExecuTorchにエクスポート
description: >-
  LibreYOLOモデルをXNNPACKデリゲーション付きのExecuTorch
  .pteプログラムへエクスポート：固定形状、バッチ1、FP32、そして必要になるメタデータサイドカー。
lead: >-
  ExecuTorchはPyTorchのプログラムをエッジ環境で実行します。LibreYOLOはtorch.exportのstrictモードでモデルをキャプチャし、XNNPACKへローワリングして、.pteプログラムとJSONメタデータサイドカーを1つの単位としてコミットします。
keywords:
  - yolo executorch エクスポート
  - executorch pte 変換
  - xnnpack パーティショナー
  - torch.export strict モード
  - executorch ランタイム 推論
  - pytorch エッジ 推論
last_verified: 1.5.0
meta:
  - label: フラグ
    value: export(format="executorch")
    mono: true
  - label: 出力
    value: .pteプログラム1つと、.pte.jsonのメタデータサイドカー1つ
  - label: 追加インストール
    value: 'pip install "libreyolo[executorch]"'
    mono: true
  - label: 再読み込み
    value: LibreYOLO("weights/LibreYOLO9t.pte")
    mono: true
  - label: 形状
    value: 固定。dynamic=Trueとbatch != 1は拒否されます。
  - label: 精度
    value: FP32のみ。half=Trueとint8=Trueは拒否されます。
  - label: デリゲート
    value: XNNPACK、CPU。delegate='xnnpack'だけが受け付けられる値です。
verification: >-
  devブランチのlibreyolo/export/executorch.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/executorch.py、pyproject.tomlを読んで確認しました。
snippets:
  install:
    - label: インストール
      language: bash
      code: |
        # libreyolo[all]からあえて除外
        # ExecuTorchは組み合わせられるTorchのバージョンを制約するため
        pip install "libreyolo[executorch]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t.pteとweights/LibreYOLO9t.pte.jsonを出力
        path = model.export(format="executorch", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format executorch --imgsz 640
    - label: 引数
      language: python
      code: |
        model.export(
            format="executorch",
            imgsz=640,             # int、または (height, width)
            batch=1,               # これ以外の値はValueErrorを送出
            dynamic=False,         # TrueはValueErrorを送出
            delegate="xnnpack",    # 受け付けられる唯一の値
            device="cpu",          # これ以外のデバイスはValueErrorを送出
            output_path=None,      # Noneならweights/<stem>.pteに出力
        )
  run:
    - label: LibreYOLO経由
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.pte")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ExecuTorchランタイムを直接使う
      language: python
      code: >
        import json

        from pathlib import Path


        import torch

        from executorch.runtime import Runtime


        runtime = Runtime.get()

        print(runtime.backend_registry.is_available("XnnpackBackend"))


        program =
        runtime.load_program(Path("weights/LibreYOLO9t.pte").read_bytes())

        method = program.load_method("forward")


        # この経路では前処理と後処理は自分で行う

        outputs = method.execute((torch.zeros(1, 3, 640, 640),))

        print([tensor.shape for tensor in outputs])


        meta = json.load(open("weights/LibreYOLO9t.pte.json"))

        print(meta["model_family"], meta["task"], meta["executorch_delegate"])
  support:
    - label: エクスポート前にファミリーとタスクを1つ確認
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c2c354a76ee33157
---

## インストール

<code-tabs name="install" />

この追加パッケージをあえて`libreyolo[all]`の外に置いているのは、ExecuTorchが対応する
Torchのバージョンを固定してしまい、インストールすると環境全体がその組み合わせに
引きずられるからです。制約を受け入れられる環境にインストールしてください。

Windowsでは、ローワリングの段階でExecuTorchに同梱される`flatc`実行ファイルを
呼び出します。`PATH`に見つからない場合はエクスポートがその旨を伝える
`RuntimeError`を送出しますが、Visual Studio 2022のDeveloper PowerShellから
実行すれば解決します。

## エクスポート

<code-tabs name="export" />

キャプチャは`torch.export.export(..., strict=True)`で、記録したトレースではなく
ガード付きの本物のグラフキャプチャです。ホスト側のスカラー読み取りやデータ依存の
制御フローは黙って埋め込まれるのではなく拒否されるため、他の経路ではトレースに
成功するいくつかのファミリーがここで失敗します。理由は組み合わせごとにサポート
マトリクスに記録しています。

ローワリングは、XNNPACKパーティショナーを使って`to_edge_transform_and_lower`を
実行します。結果にデリゲートされたパーティションが1つも含まれない場合、ポータブル
カーネルだけのプログラムをXNNPACKと名乗らせるのではなく、エクスポートが例外を
送出します。

プログラムとサイドカーは一緒にコミットされます。どちらもステージングし、どちらも
入れ替え、失敗すれば元の状態にロールバックするため、片方だけの中途半端な組み合わせ
がディスクに残ることはありません。

## 成果物を実行する

<code-tabs name="run" />

`LibreYOLO()`は`.pte`という拡張子で振り分け、チェックポイントと同じ`Results`
オブジェクトを返します。読み込み時にサイドカーは必須です：プログラム自体はクラス名も
タスクも入力サイズも持たないため、`<program>.pte.json`がないとバックエンドは
`FileNotFoundError`を送出します。バックエンドはさらに、読み込みの前に導入済みの
ランタイムが`XnnpackBackend`を提供しているかを確認し、ファイルをマッピングするのでは
なくバイト列からプログラムを読み込むため、バックエンドが動いている間ずっとWindowsの
ファイルロックを保持せずに済みます。

2つ目のスニペットはランタイムを直接使う経路です。そこでは前処理、デコード、NMS、
座標のスケール戻しは自分で行うことになります。

## 制約

バッチ1、固定形状、FP32、CPU。`batch != 1`と`dynamic=True`はどちらもエクスポートが
何かを書き換える前に`ValueError`を送出し、`half=True`と`int8=True`は検証時に拒否
され、CPU以外のデバイスは受け付けられません。

このバージョンでは、`delegate`は`"xnnpack"`だけを受け付けます。

分類のエクスポートでは`crop_pct`と`interpolation`という2つのメタデータキーが追加され、
ランタイムがそのファミリーのリサイズとセンタークロップの方針を再現できます。

ブロックされている項目には、カテゴリではなく具体的な失敗内容が書かれています。
D-FINEの検出とセグメンテーションは、strictキャプチャのもとでdeformable attention内の
未対応な`ContextVar`読み取りに到達し、手動のgrid-sample経路を強制するとシリアライズ
まではできるものの、デリゲートされたテンソルの次元順序が不正で実行時に失敗します。
DEIMとDEIMv2はキャプチャ、ローワリング、シリアライズまで進み、実行中に失敗します。
EoMTのセマンティックセグメンテーションは、マスク経路のデータ依存のシンボリック式で
失敗します。BiRefNetのマッティングは1024×1024でキャプチャできますが、
`torchvision::deform_conv2d`のoutバリアントがありません。SwinIRの復元は再読み込み
まではできますが、次元順序の不一致により`aten::alias_copy.out`で失敗します。

ファミリーとタスクの全体表は
[エクスポートマトリクス](/docs/reference/export-matrix)を参照してください。1つの
組み合わせだけを確認する場合：

<code-tabs name="support" />
