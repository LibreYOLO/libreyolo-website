---
title: カーネル
seo_title: LibreYOLOカーネルレジストリとHubカーネル
description: >-
  LibreYOLOが高速化実装を選択する仕組み：libreyolo/kernelsのカーネルレジストリ、オプションのHugging Face Hub
  MS-deform-attnカーネル、融合アテンションの切り替えについて説明します。
lead: >-
  LibreYOLOのすべての高速化演算には移植可能なデフォルト実装があり、場合によっては、その上により高速なバリアントが登録されています。実行時に述語によって選択され、オプションの依存関係がなくてもエラーではなくフォールバックとなり、エクスポートされたグラフは常に移植可能な経路を使います。
keywords:
  - libreyolo カーネル
  - LIBREYOLO_KERNELS
  - LIBREYOLO_HUB_KERNELS
  - hub-kernels インストール
  - ms_deform_attn カーネル
  - set_fused_attention
  - libreyolo triton カーネル
last_verified: 1.5.0
verification: >-
  v1.5.0のlibreyolo/kernels/__init__.pyからレジストリAPI、libreyolo/kernels/attention/__init__.pyとsdpa.pyからアテンションAPI、固定されたリビジョンと適格性の述語を含むlibreyolo/kernels/attention/ms_deform_attn.pyからHubプロバイダーを確認しました。libreyolo/kernels/からディレクトリ構成を一覧化しました。pyproject.tomlからextraの定義を確認しました。docs/kernels.mdから動作上の注意事項とベンチマーク値を確認しました。RF-DETRスロット配線コミットと1.5.0のCHANGELOG項目からv1.4.0のゲート処理履歴を確認しました。
meta:
  - label: パッケージ
    value: libreyolo.kernels
    mono: true
  - label: オプトインのextra
    value: 'libreyolo[hub-kernels]'
    mono: true
  - label: 参照実装を強制
    value: LIBREYOLO_KERNELS=off
    mono: true
snippets:
  usage:
    - label: 選択内容を確認
      language: python
      code: |
        import libreyolo.kernels as kernels

        # Opスロットから選択された実装名、または"unavailable"
        print(kernels.active())
    - label: 参照経路を強制
      language: bash
      code: |
        # offとreferenceはどちらも同じ意味で、高速化プロバイダーの
        # インポートもすべてスキップ
        LIBREYOLO_KERNELS=off python train.py
    - label: アンインストールせずにHubカーネルを無効化
      language: bash
      code: |
        LIBREYOLO_HUB_KERNELS=0 python predict.py
    - label: ファミリーを融合アテンションへ切り替え
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.kernels.attention import set_fused_attention

        model = LibreYOLO("LibreSwinIRs.pt")

        # 切り替わったアテンションモジュール数を返す
        print(set_fused_attention(model))
    - label: 独自実装を登録
      language: python
      code: |
        import libreyolo.kernels as kernels

        kernels.register(
            "fake_quant_fp8",
            my_impl,
            name="mybackend",
            predicate=my_check,
        )
source_hash: 23d504e88b7959f8
---

## レジストリ

`libreyolo/kernels/`は、差し替え可能な実装を管理する小さな実行時レジストリです。
演算スロットは、`fake_quant_fp8`や`ms_deform_attn`などの名前です。呼び出し元が
レジストリにスロットを要求すると、述語を通過した登録済み実装が返されます。
新しく登録されたものが優先され、該当するものがなければ参照実装へフォールバックします。

この構造により、オプションの依存関係が必須になることはありません。Triton、CUDA、
または`kernels`パッケージがないマシンでも同じコードが動き、同じ数値を生成しますが、
速度だけが低下します。

| 関数 | 用途 |
|---|---|
| `active()` | 演算スロットから選択された実装名、または`"unavailable"` |
| `resolve(op)` | 実行される呼び出し可能オブジェクト、または`None` |
| `register(op, impl, *, name, predicate=None)` | 実装を追加し、新しいものを優先 |
| `unregister(op, name)` | 1つを削除 |
| `clear_cache()` | メモ化された解決結果を破棄 |

<code-tabs name="usage" />

述語が例外を発生させても捕捉して警告し、伝播させません。そのため、壊れたサードパーティー
実装があっても推論は中断せず、移植可能な経路へ切り替わります。

### 構成

ツリーは最初に用途、次にバックエンドで整理されています。そのため、スロットは現在どの
ライブラリが実装しているかではなく、何を計算するかによって見つけられます。

| ディレクトリ | 内容 |
|---|---|
| `kernels/quant/simulate/` | 任意のデバイスで動作し、直通推定の逆伝播を備えた疑似量子化Tritonカーネル。QATとシミュレートされた学習後量子化の両方で使用 |
| `kernels/quant/execute/` | 確定済みモデル専用の実精度経路で、逆伝播はなし：FP8 Tensor Core GEMM、その融合Tritonプロローグとエピローグ、およびパック済み重みの展開カーネル |
| `kernels/attention/` | ファミリー間で共有されるアテンション演算：`ms_deform_attn`スロットと融合SDPAポリシー |

`simulate`と`execute`の境界は、モデルが確定済みかどうかであり、学習中かデプロイ中かでは
ありません。参照実装は数値の意味を定義する`libreyolo/quant/`に残り、`kernels/`は
処理を高速化するだけです。重みのパッキングにはバリアントが一切ありません。これは
チェックポイントの契約だからです。

GEMMとアテンションのスロットには参照実装がありません。呼び出し元は`resolve()`が何かを
返したことを確認し、独自の移植可能な経路を維持する必要があります。このため、ONNX、
TensorRT、`torch.export`のグラフには常に移植可能な演算が含まれます。

### 選択の上書き

`LIBREYOLO_KERNELS=off`または`=reference`を指定すると参照実装が強制され、高速化
プロバイダーのインポートも完全に短絡されます。それ以外の値を指定すると、その名前で
登録された実装だけに選択が制限されます。`LIBREYOLO_QUANT_KERNELS`は、レジストリが
`libreyolo/quant/`にあった時期の従来の別名として尊重され、`LIBREYOLO_KERNELS`が
未設定の場合にのみ読み取られます。どちらも他の設定とともに[設定](/docs/reference/settings)に
記載されています。

## Hubカーネル

Hugging Face Hubで公開されているコンパイル済みCUDAカーネルは、オプションの`kernels`
パッケージを通じて実行時に読み込まれます。LibreYOLOには何も同梱されません。成果物は
そのパッケージが取得してキャッシュし、各プロバイダーは監査済みコミットのリビジョンを
固定しています。そのため、固定値を更新するには、取り込む前にGPUで同等性を確認する
必要があります。

extraのインストールがオプトインになります。

```bash
pip install "libreyolo[hub-kernels]"
```

パッケージがなくても何も変わらず、ネットワーク要求も行われません。
`LIBREYOLO_HUB_KERNELS=0`を指定すると、何もアンインストールせず取得を無効にできます。
読み込みまたは実行に失敗したカーネルは、残りのプロセス中は自身を無効にし、1回だけ警告して
フォールバックします。

現在Hubが支えるスロットは1つです。`ms_deform_attn`は、Apache 2.0の下で提供される
Deformable DETR由来のコンパイル済みマルチスケール変形アテンションの順伝播と逆伝播です。
これは変形アテンション系統の全体に組み込まれています：RF-DETR、Deformable DETR、
DINO-DETR、LW-DETR、Grounding DINO、RT-DETR、RT-DETRv2、D-FINE、RT-DETRv4、DEIM、
DEIMv2、EC、OV-DEIMです。逆伝播もコンパイルされているため、推論だけでなく学習も
高速化されます。

適格性は意図的に狭く設定されています。入力はCUDAかつfloat32である必要があり、実行は
eagerモードでなければなりません。プロバイダーは`torch.jit.is_tracing()`、
`torch.compiler.is_compiling()`、`torch.compiler.is_exporting()`、
`torch.onnx.is_in_onnx_export()`の実行中は辞退します。2種類の入力レイアウトも移植可能な
経路へフォールバックします。レベルごとに異なるレベル単位の点数と、離散整数インデックス
サンプリングです。ECの姿勢バリアントは組み込まれていません。

### このカーネルが新たに到達可能になりました

既存プロジェクトにextraをインストールする前に、ここを読んでください。

v1.4.0では、空間形状のペアが存在しないことを要求する条件の背後で、ヘルパー内部から
スロットが参照されていました。RF-DETRは常にそのペアをデコーダーに渡すため、この条件は
成立せず、どのeager順伝播でもカーネルは一度も実行されませんでした。参照処理はv1.5.0で
移動され、現在はカーネルが実際に実行されます。

実際の影響として、v1.5.0へアップグレードし、CUDA環境に`libreyolo[hub-kernels]`も
インストールすると、RF-DETRとその系統の順伝播が初めてコンパイル済みバイナリを使います。
その結果、推論と指標が浮動小数点の許容範囲内で変化する場合があります。extraなしの標準
インストールは影響を受けません。アップグレードの前後で指標を比較する場合は、extraの状態を
固定するか、両方で`LIBREYOLO_HUB_KERNELS=0`を設定してください。

## 融合アテンション

融合スケールド・ドット積アテンションには、標準のPyTorch以外のオプション依存関係は
不要です。そのため、可用性ではなくポリシーによって制御されます。2つの規則が適用されます。

第1に、グラフキャプチャでは使用されません。置き換えられた各呼び出し箇所は、エクスポート
チェックの背後でプリミティブ演算による式を利用できる状態に保ちます。これには、デフォルトの
opsetにSDPAシンボリックがないONNXエクスポートと、TorchScript、CoreML、NCNNがすべて
経由する`torch.jit.trace`が含まれます。Dynamoキャプチャは意図的にゲートの対象外です。
`torch.compile`は手動の演算より適切にSDPAを低レベル化し、Core AIとExecuTorchはそれぞれ
SDPAをコアATenへ分解するためです。

第2に、デフォルトにするための同等性基準はバイト単位の完全一致です。基準を満たした
ファミリーは、デフォルトでSDPAを使用します：SegFormer、Depth AnythingとMoGe-2、BERT、
Grounding DINO、SwinIR、PP-OCRです。満たさないファミリーは手動の演算を維持し、代わりに
`fused_attn`フラグを公開します。`set_fused_attention(model)`はこのフラグを切り替えます：
Swin、DINO-DETRのSwinバックボーン、BiRefNetとFeyNobg、OWLv2、LW-DETR、SigLIP 2、
ZipDepth、MobileSAMです。ViTとDeiTも同じフラグを持ちますが、アップストリームに従って
デフォルトで有効です。そのため、同じ呼び出しに`enabled=False`を渡すと無効になります。

適用できる場合は使用する価値があります。fp16 autocastを使うRTX 5070 Tiでは、Swinの
ウィンドウアテンションが1.278 msから0.721 msになり、1.77倍高速化します。また、OWLv2の
ビジョンアテンションは6.483 msから1.735 msになり、3.74倍高速化します。

## ハードウェア

| プラットフォーム | 動作 |
|---|---|
| CPUとMPS | すべてのCUDAとTritonの述語が失敗するため、すべて参照実装で実行 |
| NVIDIA CUDA | Tritonカーネルと、適格なHubおよびGEMMカーネルが作動 |
| AMD ROCm | ROCmのwheelにはTritonのAMDバックエンドが含まれるため、Tritonは作動可能。ただし、CIで同等性を検証しているのはNVIDIAのみ |

## 実装の追加

名前と述語を指定して`register()`を呼び出します。ツリー外のコンパイル済みカーネルは、
インポート時に自身を登録する別個の`libreyolo_kernels`パッケージとして配布できます。これにより、
非公開バックエンドをLibreYOLOツリーから完全に分離できます。

ツリー内に含めるには同等性が条件です。参照実装に対する順伝播の完全一致と、テストスイートが
扱う形状の組み合わせ全体で、直通推定器に対して1e-6以内の勾配が必要です。

カーネル選択は[CUDAグラフ](/docs/reference/cuda-graphs)と相互作用します。推論同等性
マトリックスは`kernels`パッケージをインストールせずに実行されたため、コンパイル済み
カーネルが有効な状態でのキャプチャ安全性は対象外です。
