---
title: Quantization
seo_title: PyTorchでLibreYOLOモデルを量子化
description: >-
  LibreYOLOのPyTorch量子化APIを説明します。9種類のレシピ、学習データと分離したキャリブレーション、QATとQAD、2種類のデプロイ成果物を扱います。
lead: >-
  LibreYOLOの量子化はすべてPyTorch内で実行されます。model.quantize()はモデルのConv2dモジュールとLinearモジュールを量子化版に置き換え、キャリブレーションします。結果は通常のpredict、val、train、saveの契約を維持するため、量子化モデルも浮動小数点モデルと同じvalidatorで評価されます。
keywords:
  - libreyolo 量子化
  - int8 ptq yolo
  - 量子化 aware training
  - qat qad 違い
  - nvfp4 mxfp4 量子化
  - fp8 e4m3 pytorch
  - 量子化 キャリブレーション データセット
  - qdq onnx エクスポート
last_verified: 1.5.0
meta:
  - label: 呼び出し
    value: 'model.quantize(recipe="int8", calib="coco128.yaml")'
    mono: true
  - label: コマンド
    value: libreyolo quantize --model M.pt --recipe int8 --calib coco128.yaml
    mono: true
  - label: 追加パッケージ
    value: なし。量子化はPyTorch内で実行
  - label: ファミリー
    value: 'yolo9, rfdetr, birefnet, feynobg'
  - label: レシピ
    value: 'fp16, bf16, fp8, int8, w4a16, w4a8, nvfp4, mxfp4, int2'
    mono: true
  - label: デプロイ成果物
    value: >-
      export(format="pt") for a packed checkpoint, export(format="onnx") for a
      QDQ INT8 graph
    mono: true
verification: >-
  devブランチのlibreyolo/quant/api.py、libreyolo/models/base/model.py、libreyolo/cli/commands/quantize.py、docs/quantization.mdを参照。チェックポイントサイズはdocs/quantization.mdに記録された測定値。
snippets:
  quantize:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # 構造の置換とキャリブレーション。calibは少数のラベルなし画像セット

        # 順伝播だけで読み取り、活性化範囲とスケールを算出

        qmodel = model.quantize(recipe="int8", calib="coco128.yaml",
        samples=128)


        print(qmodel.quant_info())

        qmodel.val(data="coco8.yaml")          # 浮動小数点モデルと同じvalidator

        qmodel.save("LibreYOLO9s-int8.pt")     # チェックポイントに量子化manifestを格納
    - label: CLI
      language: bash
      code: >
        libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib
        coco128.yaml
    - label: 引数
      language: python
      code: |
        model.quantize(
            recipe="int8",
            calib="coco128.yaml",      # data.yamlのパスまたは組み込み名。Noneはキャリブレーションを省略
            samples=128,               # キャリブレーション画像の最大数
            batch=8,                   # キャリブレーションのバッチサイズ
            algorithm="auto",          # autoとminmaxは同じ。代替はpercentile
            keep_high_precision=None,  # Noneはファミリーのポリシーを使用
            verbose=True,
        )
  reload:
    - label: 量子化チェックポイントとして再読み込み
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 重みを読み込む前に量子化manifestが量子化構造とスケールを再構築
        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        print(qmodel.quant_info())
  train:
    - label: 量子化モデルで通常のtrain()を使うQAT
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # ゼロからの学習ではなくファインチューニング。ファインチューニング用学習率を使用
        qmodel.train(data="coco8.yaml", epochs=5, lr0=1e-4)
    - label: 既存の蒸留引数を追加するQAD
      language: python
      code: |
        qmodel.train(
            data="coco8.yaml",
            epochs=5,
            lr0=1e-4,
            distill_model="LibreYOLO9m.pt",
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train --model LibreYOLO9s-int8.pt --data coco8.yaml --epochs 5
        --lr0 1e-4
  export:
    - label: パック済みPyTorchチェックポイント
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # LibreYOLO9s-int8-final.ptを書き出し。低ビットの重みとスケールをパックし
        # fp32マスターを除去、量子化していない残りをfp16へキャスト
        qmodel.export(format="pt")

        # remainder="fp32"は量子化していないテンソルを正確に維持
        qmodel.export(format="pt", remainder="fp32")
    - label: QDQ INT8 ONNX
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # モデル独自のキャリブレーション済みまたはQAT学習済みスケールを持つ
        # グラフ内のQuantizeLinear/DequantizeLinearペア
        qmodel.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9s-int8.pt --format onnx
  dequantize:
    - label: QAT学習済みの重みを保って浮動小数点へ戻す
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        qmodel.dequantize()

        # 任意の浮動小数点エクスポーターを対応する精度で利用可能
        qmodel.export(format="tensorrt", half=True)
source_hash: 4ffb06b87cad017e
---

## インストール

量子化に追加パッケージは必要ありません。モジュールの置換、キャリブレーション処理、演算の
シミュレーションはすべてPyTorch内で実行されるため、必要なのは`pip install libreyolo`だけです。
デプロイ成果物には各形式固有の要件があり、ONNX経路では`libreyolo[onnx]`が必要です。

## 量子化

<code-tabs name="quantize" />

`quantize()`は読み込んだモデルをその場で変換して返します。勾配は使いません。置換処理で
量子化モジュールを導入し、キャリブレーション処理は順伝播だけを実行します。

生成されるチェックポイントは、`quant`manifestを付加した通常のLibreYOLOチェックポイントです。
そのため、構造とスケールを維持したまま再読み込みできます。

<code-tabs name="reload" />

QAT実行中に書き出されるtrainerチェックポイントにもmanifestが含まれます。したがって、その実行で
生成された`best.pt`自体が量子化チェックポイントです。

## レシピ

対応するファミリーは`yolo9`、`rfdetr`、`birefnet`、`feynobg`の4種類です。

| レシピ | 動作 | ファミリー | キャリブレーション |
|---|---|---|---|
| `fp16` | float32の入出力契約を保って半精度へキャスト。推論のみ | 4種類すべて | なし |
| `bf16` | float32の指数範囲を維持するbfloat16へキャスト。DETR系モデルでfp16がオーバーフローする場合の対策。推論のみ | 4種類すべて | なし |
| `fp8` | `Conv2d`と`Linear`でE4M3の重みと活性化を使用。チャネル単位の重みスケール、キャリブレーション済みテンソル単位の活性化スケール | 4種類すべて | 必須 |
| `int8` | `Conv2d`と`Linear`でW8A8を使用。チャネル単位の対称な重み、テンソル単位のaffine活性化 | 4種類すべて | 必須。重みだけなら`calib=None` |
| `w4a16` | `Linear`でグループ化された対称INT4重み、`in_features`方向にグループ128、浮動小数点活性化を使用 | rfdetr、birefnet、feynobg | 不要 |
| `w4a8` | `Linear`でグループ化されたINT4重みとキャリブレーション済みINT8活性化を使用 | rfdetr、birefnet、feynobg | 必須 |
| `nvfp4` | `Linear`でW4A4 NVFP4を使用。E2M1要素、16要素ブロック、FP8 E4M3ブロックスケール、FP32テンソルスケール。動的な活性化スケーリング | rfdetr、birefnet、feynobg | 不要 |
| `mxfp4` | `Linear`でOCP MXFP4を使用。E2M1要素、32要素ブロック、2の累乗のE8M0ブロックスケール。動的な活性化スケーリング | rfdetr、birefnet、feynobg | 不要 |
| `int2` | 研究用途のみ。`Linear`でグループ化された2ビット重み、グループ64、INT8活性化を使用。学習後処理だけでは実用にならないためQATまたはQADが必要 | rfdetr | 必須 |

8ビット未満のレシピは`nn.Linear`を対象とし、`yolo9`では意図的に拒否されます。現在の
ハードウェアでその高速化はGEMMだけに対応するため、畳み込みは高い精度のままです。YOLO9では
`int8`または`fp8`を使います。`birefnet`と`feynobg`では`int2`も拒否されます。これらの
ファミリーは推論専用であり、レシピが依存するQATによる精度回復を利用できないためです。

ファミリーごとのデフォルトでは最初の層とヘッドを浮動小数点のまま維持し、YOLO9のDFL畳み込みは
量子化しません。これは固定された積分期待値演算子です。必要な理由がある場合は
`keep_high_precision=("head.",)`で上書きしてください。

## キャリブレーションデータは学習データではない

`calib=`は数百枚の画像を受け取り、ラベルを読み込まず、順伝播だけを実行して活性化範囲を
推定します。`train()`と`val()`の`data=`は、勾配と指標に使うラベル付きデータセットです。
両者は目的が異なる別の引数で、`calib`のデフォルトは`coco128.yaml`です。

`algorithm="minmax"`はキャリブレーションバッチ全体で観測した絶対的な極値を維持し、
`"auto"`もこれを選びます。`"percentile"`はバッチごとの0.1パーセンタイルと99.9パーセンタイルの
平均を使います。transformerの活性化の外れ値が重要な役割を持つため、この方法ではDETRファミリーの
精度が崩れることが測定されています。小規模モデルのINT8感度を実際に改善するのは、十分な数の
バッチでキャリブレーションすることです。デフォルトの`coco128`では、YOLO9-tのスコアは浮動小数点版の
約1 mAP以内に収まります。選択したアルゴリズムはチェックポイントmanifestに記録されます。

## 精度を回復

<code-tabs name="train" />

量子化モジュールはfp32のマスター重みを保持し、straight-through estimatorでfake quantizationを
適用します。そのため、勾配はマスター重みに到達し、既存のtrainerを変更せずに使えます。EMA、AMP、
チェックポイントからの再開、蒸留引数をすべて組み合わせられます。

QATは学習済みモデルのファインチューニングです。ゼロからの学習用デフォルトではなく、
ファインチューニング用の学習率を使ってください。そうしないと、量子化の有無にかかわらず短時間の
実行で学習済みの重みが壊れます。QADの対応状況はファミリーの蒸留対応に従い、現時点では
`yolo9`と`rfdetr`です。

`fp16`および`bf16`量子化モデルは推論専用です。trainerは`amp=True`を案内してこれらを拒否します。

## エクスポート

<code-tabs name="export" />

`format="pt"`はモデルを確定します。パックした低ビットの重みとスケールがマスター重みに置き換わり、
`remainder="fp32"`を渡さない限り、量子化していない残りはfp16へキャストされます。パックの不変条件は、
確定処理を行ったデバイスで展開するとシミュレーションをビット単位で再現することです。そのため、
確定済みファイルのスコアは検証した値と正確に一致します。測定値では、YOLO9-s int8は29.5 MBから
9.6 MB、RF-DETR-n nvfp4は122 MBから26 MBになります。読み込むと推論可能なモデルになり、
`train()`を呼び出すとパック済みの重みからマスター重みが自動的に再構築されます。

`format="onnx"`は`int8`モデルに適用され、モデル独自のキャリブレーション済みまたは
QAT学習済みスケールを持つQDQグラフを生成します。ONNX RuntimeとTensorRTはこれを実際の
INT8カーネルで実行します。これは浮動小数点モデルで
[`export(format="onnx", int8=True)`](/docs/export/onnx)を使い、ONNX Runtime自身が
スケールを導出する経路とは異なります。

キャスト系レシピには量子化エクスポーターがまったく必要ありません。

<code-tabs name="dequantize" />

## 制約

量子化演算はシミュレーションで実行されます。これはAMP使用時でもfloat32領域で計算する
fake quantizationです。シミュレーションは数値的に正確なため、どのデバイスでも`val()`のスコアは
量子化演算に関する実際の結果です。ただし、速度を示すものではありません。

2つの例外はネイティブに実行されます。`fp16`と`bf16`は通常のキャストです。確定済みの`fp8`
モジュールは、Ada、Hopper、Blackwellクラスのハードウェアで`torch._scaled_mm`を介し、パック済み
E4M3重みに対してGEMMを直接実行します。活性化スケールにはシミュレーションと同じキャリブレーション済み
スケールを使います。`LIBREYOLO_KERNELS=off`を設定すると、すべての環境で正確なシミュレーション経路に戻ります。

デプロイの対応範囲はレシピ一覧より狭くなります。ここでデプロイ可能なONNX形式があるのは
`int8`だけです。`fp8`と8ビット未満のlinearレシピはPyTorchで実行し、`format="pt"`で
確定します。これらにONNXエクスポートを要求すると、その手順を示して例外が発生します。
`int8`モデルにONNX以外の形式を要求した場合も同様です。代わりにQDQグラフから後段のエンジンを
ビルドしてください。

活性化を一度もキャリブレーションしていない`int8`モデルをエクスポートすると、警告が記録され、
重みの量子化だけを含むグラフが生成されます。
