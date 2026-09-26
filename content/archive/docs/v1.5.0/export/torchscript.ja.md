---
title: TorchScript
seo_title: LibreYOLOからTorchScriptへエクスポート
description: >-
  LibreYOLOモデルをTorchScriptへエクスポートします。LibreYOLOのメタデータを内部に持ち、Pythonまたはlibtorchから読み込めるトレース済みの.torchscriptアーカイブです。
lead: >-
  TorchScriptはPyTorch独自のシリアライズ済みグラフ形式です。LibreYOLOはtorch.jit.traceでモデルをトレースし、libreyolo_metadata.json追加ファイルとともに結果を保存します。そのため、アーカイブにはファミリー、タスク、クラス名、入力サイズが含まれます。
keywords:
  - yolo torchscript エクスポート
  - torch.jit.trace
  - torch.jit.load
  - libtorch デプロイ
  - torchscript メタデータ
  - extra_files
last_verified: 1.5.0
meta:
  - label: フラグ
    value: export(format="torchscript")
    mono: true
  - label: 出力
    value: libreyolo_metadata.json追加ファイルを含む1つの.torchscriptアーカイブ
  - label: 追加パッケージ
    value: なし。TorchScriptはPyTorchに付属します。
  - label: 再読み込み
    value: LibreYOLO("weights/LibreYOLO9t.torchscript")
    mono: true
  - label: 形状
    value: 固定。グラフは1つの入力形状でトレースされます。
  - label: 精度
    value: FP32、FP16（half=True）。INT8はありません。
verification: >-
  devブランチのlibreyolo/export/torchscript.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/torchscript.pyを参照しました。
snippets:
  install:
    - label: インストール
      language: bash
      code: |
        pip install libreyolo
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t.torchscriptに書き出し
        path = model.export(format="torchscript")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format torchscript
    - label: 引数
      language: python
      code: |
        model.export(
            format="torchscript",
            imgsz=640,        # 整数または(height, width)
            batch=1,
            half=False,       # FP16の重みと活性化
            device=None,      # この形式ではNoneによりCPUでトレース
            output_path=None, # Noneではweights/<stem>.torchscriptに書き出し
        )

        # dynamicは受け付けるがアーカイブは常に固定形状のトレース
        # どちらの場合も埋め込みメタデータにはdynamic=Falseを記録
  run:
    - label: LibreYOLO経由
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.torchscript")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 素のPyTorch
      language: python
      code: |
        import json

        import torch

        extra_files = {"libreyolo_metadata.json": ""}
        module = torch.jit.load(
            "weights/LibreYOLO9t.torchscript",
            map_location="cpu",
            _extra_files=extra_files,
        )
        module.eval()

        metadata = json.loads(extra_files["libreyolo_metadata.json"])
        print(metadata["model_family"], metadata["task"], metadata["imgsz"])

        # この方法では前処理と後処理を自分で実装
        with torch.no_grad():
            out = module(torch.zeros(1, 3, 640, 640))
        print(out.shape if torch.is_tensor(out) else [t.shape for t in out])
  support:
    - label: エクスポート前に1つのファミリーとタスクを確認
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 286a082969ccd604
---

## インストール

<code-tabs name="install" />

`torch.jit` はPyTorchに付属するため、TorchScriptには基本インストール以外に何も必要ありません。オプションの依存関係も外部コンバーターも不要な唯一のエクスポート先なので、より長いツールチェーンが失敗したときの最初の確認に役立ちます。

## エクスポート

<code-tabs name="export" />

デバイスを指定しない限りトレースはCPU上で実行されます。`output_path` を省略すると、アーカイブはチェックポイントと同じステム名で `weights/` に書き出されます。

通常は `torch.jit.trace` が実行する再トレース確認を無効にしています。一部のエクスポートラッパーは、最初の順伝播で形状依存のアンカーをキャッシュします。そのため、記録された固定形状グラフが正しくても、2回目のトレースでは異なるPython経路が観測されます。代わりに、同等性テストで保存済みモジュールを直接検証します。

メタデータはサイドカーには置かれません。`torch.jit.save` はアーカイブ内に `libreyolo_metadata.json` を保存し、`torch.jit.load` は `_extra_files` を通じてその内容を返します。

## 成果物の実行

<code-tabs name="run" />

`LibreYOLO()` は `.torchscript` 接尾辞に基づいて処理を振り分け、元のチェックポイントと同じ `Results` オブジェクトを返します。`device="auto"` の場合、利用可能であればCUDA、次にMPS、最後にCPUへモジュールを割り当てます。

2番目のスニペットは、LibreYOLOをインストールしていない場合と、同じアーカイブを `torch::jit::load` で読み込むlibtorchによるC++デプロイに使用する方法です。この場合、前処理、デコード、NMS、座標の再スケーリングを自分で実装します。追加メタデータファイルは引き続き読み取ることができ、クラス名が存在する唯一の場所です。

## 制約

グラフは1つの入力形状で取得したトレースです。インターフェースの対称性のため `dynamic=True` を受け付けますが、動作は変わりません。また、バックエンドが使用できない軸を想定しないように、埋め込みメタデータでは `dynamic=False` と報告されます。別の解像度には別のアーカイブをエクスポートしてください。

`half=True` はモデルとトレース入力をFP16にキャストします。INT8の経路はありません。検証時に `int8=True` を指定すると `NotImplementedError` が発生します。

長方形の `imgsz` はYOLO9ファミリー、HRNet、NAFNet、Real-ESRGANで使用できます。固定の正方形入力が必要なファミリーでは拒否されます。

5つの組み合わせはトレース前に拒否されます。YOLO9セグメンテーションは、LibreYOLOのYOLO9が検出専用であるためです。RTMDet-Insセグメンテーションは、動的カーネルによるマスクデコードにエクスポート済みランタイムの契約がないためです。SSD、Faster R-CNN、RetinaNetによる検出は、可変長または動的アンカーのグラフについてONNX Runtime契約を通じた同等性の証拠しかないためです。

ファミリーとタスクの完全な一覧は、[エクスポート対応表](/docs/reference/export-matrix)を参照してください。1つの組み合わせを確認するには、次を実行します。

<code-tabs name="support" />
