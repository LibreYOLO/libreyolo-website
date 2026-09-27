---
title: クイックスタート
seo_title: LibreYOLOクイックスタート
description: >-
  約10行のPythonで、画像に対して検出器を実行し、小規模データセットでファインチューニングして、TorchScriptまたはONNXへエクスポートします。すべてCPUで実行できます。
lead: >-
  LibreYOLOを使う最短手順です。1枚の画像を推論し、小規模データセットで学習して、その結果をエクスポートします。ここにあるすべてのコマンドはCPUで動作します。
keywords:
  - libreyolo クイックスタート
  - libreyolo チュートリアル
  - libreyolo 推論
  - libreyolo 学習
  - libreyolo エクスポート
  - yolo python サンプル
last_verified: 1.5.0
meta:
  - label: インストール
    value: pip install libreyolo
    mono: true
  - label: チェックポイント
    value: LibreYOLO9t.pt
    mono: true
  - label: ハードウェア
    value: このページのすべての処理にはCPUで十分
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 初回使用時にチェックポイントをダウンロードし、weights/にキャッシュ
        model = LibreYOLO("LibreYOLO9t.pt")

        # 1枚の画像は1つのResultsオブジェクトを返す
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=yolo9-t save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 動画とストリーム
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # stream=Trueはリストを構築せず、フレームごとに1つのResultsを生成
        # パスをwebcam index、RTSP URL、フォルダーに置き換え可能
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco8はライブラリに同梱される8枚の画像のデータセット
        # 初回使用時にURLからダウンロードされるため、スクリプト実行は不要
        results = model.train(
            data="coco8.yaml",
            epochs=1,
            imgsz=640,
            batch=4,
            device="cpu",
        )

        print(results["save_dir"])
        print(results["best_checkpoint"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=yolo9-t data=coco8.yaml \
          epochs=1 imgsz=640 batch=4 device=cpu
    - label: 検証
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val()はオブジェクトではなく通常のdictを返す
        metrics = model.val(data="coco8.yaml", device="cpu")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
  export:
    - label: TorchScript
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # export()は書き込んだパスを返す
        path = model.export(format="torchscript")
        print(path)

        # ファクトリーはファイル接尾辞で経路を選ぶため、成果物も
        # チェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: ONNX
      language: bash
      code: |
        pip install "libreyolo[onnx]"
        libreyolo export model=yolo9-t format=onnx imgsz=640
source_hash: c11b6bdbf0b6fdf1
---

## インストール

```bash
pip install libreyolo
```

以下の推論と学習セクションで必要なものはこれですべてです。ONNXへのエクスポートには追加パッケージが1つ必要です。全リストについては[インストール](/docs/install)を参照してください。

## 推論

<code-tabs name="predict" />

`LibreYOLO()` はファクトリーです。ファイルを読み取り、重みがどのファミリーに属するか判断し、そのファミリーのモデルを返します。そのため、別の検出器への切り替えは1行の変更で済みます。ディレクトリなしで `LibreYOLO9t.pt` を渡すと、作業ディレクトリを基準に `weights/LibreYOLO9t.pt` を探し、存在しない場合はそこへダウンロードします。ダウンロードの規則とオフラインで使う方法については[チェックポイントと重み](/docs/weights)を参照してください。

`save=True` を指定すると、アノテーション付きのコピーが `runs/detect/` 以下で実行ごとに連番となる `predict` ディレクトリに書き込まれます。返された `Results` は `boxes` を持ち、`names` がクラスインデックスをラベルに対応付けます。1枚の画像パスは1つの `Results` を返し、ディレクトリ、画像リスト、`stream=True` はリストまたはジェネレーターを返します。

## 学習

<code-tabs name="train" />

`data` はデータセットYAMLです。`coco8.yaml` はライブラリに同梱されるため、snippetをそのまま実行できます。同梱されない名前はパスとして読み取られます。データセットは `~/datasets` 以下、または `LIBREYOLO_DATASETS_DIR` 変数を設定している場合はその場所を基準に解決されます。

実行結果は `project/name` へ書き込まれ、デフォルトでは `runs/train` 以下のディレクトリとなり、その中に `weights/best.pt` と `weights/last.pt` が作成されます。`train()` は `save_dir`、`best_checkpoint`、`last_checkpoint`、エポックごとの損失、エポックごとの検証指標を含む辞書を返します。学習済みチェックポイントは、学習済みの元チェックポイントとまったく同じように `LibreYOLO()` から読み込めます。

すべてのファミリーが学習できるわけではありません。ファミリーが推論専用の場合、`train()` は `NotImplementedError` を送出してその旨を示します。各サポート階層の意味については[基本概念](/docs/concepts)で説明しています。

## エクスポート

<code-tabs name="export" />

TorchScriptには基本インストール以外のものは必要ありません。ほかの対象形式にはそれぞれ独自の追加パッケージがあり、対応範囲は一律ではなくファミリー・タスクごとに決まります。[エクスポートとデプロイ](/docs/export)を参照してください。

すべての形式で受け付ける引数には、`imgsz`（整数または高さと幅の組）、`batch`（デフォルトは1）、`half`、キャリブレーション用 `data` YAMLを伴う `int8`、`dynamic`（デフォルトはTrue）、`simplify`（デフォルトはTrue）、`opset`、`device`、`output_path` があります。`output_path` を省略すると、チェックポイントに基づく名前でファイルが `weights/` 以下に書き込まれます。

## 次に読むページ

- タスク、ファミリー、サイズ、チェックポイント名については[基本概念](/docs/concepts)。
- 自動ダウンロード、オフライン利用、読み込みの安全性については[チェックポイントと重み](/docs/weights)。
- アップストリームプロジェクトのチェックポイントをすでに持っている場合は[既存の重みをインポート](/docs/migrate)。
- 問題に合うファミリーを探すには[全モデル](/docs/models)。
- 完全なワークフローについては[学習](/docs/train)、[推論](/docs/predict)、[エクスポート](/docs/export)。

