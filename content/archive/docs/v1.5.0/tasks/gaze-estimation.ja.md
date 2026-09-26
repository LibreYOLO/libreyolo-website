---
title: 視線推定
seo_title: LibreYOLOでの視線推定
description: >-
  LibreYOLOで顔ごとの視線pitchとyawを推定します。PythonまたはCLIから推論し、ラジアン単位の角度を読み取り、視線ヘッドをONNXへエクスポートします。
lead: >-
  視線推定は、画像内の各顔について視線方向を返します。LibreYOLOでは2段階のタスクとして扱います。最初に顔検出器を実行し、視線ヘッドが返された各顔クロップからpitchとyawを読み取ります。
keywords:
  - 視線推定 python
  - eye tracking
  - 視線 pitch yaw
  - L2CS-Net
  - 視線方向
  - head pose
  - libreyolo gaze
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # face_detector未指定ではOpenCV同梱の検出器へフォールバックするため
        # チェックポイント以外は何もダウンロードされない
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        gaze = result.gaze
        print(gaze.pitch, gaze.yaw)              # ラジアンで顔ごとに1行
        print(gaze.pitch_deg, gaze.yaw_deg)      # 同じ角度を度単位で表示
        print(gaze.direction_3d)                 # (N, 3)の単位ベクトル
    - label: CLI
      language: bash
      code: >
        # Python経路と異なりCLIには自動フォールバックがないため視線モデルには

        # 明示的な顔検出器が必要 顔のボックスを返すLibreYOLO検出器を使用

        #

        libreyolo predict model=LibreL2CSr50.pt source=photo.jpg
        face_detector=face-detector.pt save=True
    - label: 顔のソースを選ぶ
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # すでに実行した検出器のボックスを視線ヘッドに渡す
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # または同梱された検出器の1つを指定
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
source_hash: 22aa3c3d87b0c730
---

## 定義

視線推定は顔ごとに2つの角度を返します。`result.gaze` は形状 `(N, 2)` の `Gaze` ペイロードで、列0がpitch、列1がyawです。単位はラジアンで、検出された顔ボックスである `result.boxes` と行単位で対応します。規約はL2CS-Netと同じで、正のyawは被写体から見て左へ視線を回転し、正のpitchは下向きへ回転します。

同じペイロードは度単位の `pitch_deg` と `yaw_deg`、およびカメラ座標系で列が `(x, y, z)` の `(N, 3)` 単位ベクトル `direction_3d` を公開します。

このタスクは2段階なので、推論は2つのモデルに依存します。検出器が見落とした顔には視線の行がなく、配置が不適切なボックスでは不適切にクロップされた顔から角度を生成します。正規のタスクキーは `gaze` で、`gaze-estimation` はこのキーへ正規化されます。

## モデル

[L2CS-Net](/docs/models/l2cs)はこのタスクを提供する唯一のファミリーです。448x448の顔クロップに対し、ResNetトランクと、pitch用とyaw用の2つの並列な角度ビン分類ヘッドを組み合わせます。アーキテクチャとして5つのバックボーン深度に対応し、そのうちResNet-50の1つに公開チェックポイントがあります。

重みにはライセンス制限があります。Gaze360で学習されており、そのライセンスは研究および非商用利用だけを許可して再配布を禁止するため、LibreYOLOはこのファミリーを何もミラーしません。ライブラリが自動取得できる唯一のチェックポイントは、ライセンス条件を表示した後、`gdown` を通じて著者独自のGoogle Drive配布から直接取得します。デプロイ前に[L2CS-Net](/docs/models/l2cs)を参照してください。

このダウンロード経路には `gaze` 追加パッケージが必要です。

```bash
pip install "libreyolo[gaze]"
```

このパッケージがない場合、ライブラリは転送を試みる代わりに手動ダウンロード手順を表示します。すでに所有するチェックポイントの推論とエクスポートには追加パッケージは不要です。

## 推論

<code-tabs name="predict" />

顔のソースは3つの方法のいずれかで選択します。`face_boxes` は計算済みのボックスを渡して検出を省略します。`face_detector` は `"auto"`、`"haar"`、`"yunet"`、LibreYOLO検出モデル、または通常の呼び出し可能オブジェクトを受け付け、コンストラクターまたは呼び出しごとに設定できます。Pythonで未指定のままにすると、推論はOpenCV同梱の検出器へフォールバックするため、追加の接続なしで通常の呼び出しが動作します。OpenCV 4では `wheel` 内に同梱されたHaar cascadeを使用し、ダウンロードは不要です。Haar APIが削除されたOpenCV 5ではYuNetを使用し、OpenCV Zooから小さなモデルファイルを1回だけ取得します。

CLIはこのフォールバックを共有しません。`libreyolo predict` は `face_detector=` のない視線モデルを拒否し、その値にはLibreYOLO検出器の名前またはチェックポイントパスを指定します。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## 学習

このタスクのファミリーはLibreYOLO内で学習できません。`LibreL2CS.train()` はエラーになります。アップストリームのL2CS-Netプロジェクトで学習し、得られた状態辞書をここで読み込んでください。

## 検証

視線の正解データを持つデータセットに対する検証は対象外で、`val()` は計算していない指標を返さずエラーになります。このタスクに `metrics/` 辞書はありません。チェックポイントの学習に使用したデータセット上で、アップストリームを使って評価してください。

## エクスポート

<code-tabs name="export" />

視線のエクスポート契約はONNX、TorchScript、ExecuTorch、TensorRT、OpenVINOに対応します。ライブラリから出力されるのはResNetトランクと2つの角度ビンヘッドだけです。グラフは前処理済み448x448の顔クロップを受け取り、未処理のyawとpitchのlogitを返します。顔検出、クロップ、softmax、ビン期待値、角度への変換はすべてPythonの `libreyolo.models.l2cs.utils` に残ります。形式とその引数については、[エクスポート](/docs/export)を参照してください。
