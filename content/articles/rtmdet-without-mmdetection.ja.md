---
title: mmdetectionなしでRTMDetを使う方法
description: RTMDetは高速かつ高精度な検出器の1つです。公式のインストール手順は、ここ2年間にリリースされたPyTorchでは動作しません。代替方法を紹介します。
date: 2026-06-27
author: Xuban
tags: [LibreYOLO, rtmdet, object-detection, tutorial]
faq:
  - q: "最近のPyTorchでは、なぜmmcvのインストールに失敗するのですか？"
    a: "最新のmmcv wheelである2024年4月リリースの2.2.0には、torch 2.4 / CUDA 12.1までのビルド済みバイナリしか含まれていません。それより新しいPyTorchでは、mmcvのC++/CUDA演算をソースからコンパイルする必要があり、10〜30分かかるうえ、対応するCUDAツールキット、nvcc、互換性のあるC++コンパイラーが必要です。mmcv._extモジュールがないといったエラーは、この段階で発生します。"
  - q: "mmdetectionをインストールせずにRTMDetを実行できますか？"
    a: "はい。LibreYOLOはアップストリームのOpenMMLabチェックポイントから変換したRTMDetの重みを読み込み、同じチェックポイントを使った場合、推論結果はmmdetectionとビット単位で同等です。LibreRTMDets.ptを名前で指定すると自動ダウンロードされます。TinyからXまでの5サイズがあり、すべて640 pxで動作します。"
  - q: "RTMDetは商用利用できますか？"
    a: "はい。MMDetectionとRTMDetはApache 2.0で、変換後の重みにも同じApache 2.0の条件が適用されます。LibreYOLOのコードはMITです。商用利用の制限はありません。"
  - q: "それでもmmdetectionを選ぶべきなのはどんな場合ですか？"
    a: "MMDetectionの拡張パイプライン全体を使ってRTMDetを学習またはファインチューニングする必要がある場合や、mmcvが動作するOpenMMLabエコシステムをすでに深く利用している場合です。推論とデプロイにはLibreYOLOが適しています。"
---

RTMDetはOpenMMLabのリアルタイム検出器で、高速なデプロイに適した速度を保ちながら、COCOで高い精度を実現します。アーキテクチャはシンプルです。インストール手順はそうではありません。

公式ソースからRTMDetを動かすには、OpenMMLabの4層のスタックを組み立てる必要があります。

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

バージョンの指定範囲は狭いです。mmdet 3.3.0は`mmcv < 2.2.0`を要求しますが、`mim install mmcv>=2.0.0`は問題なく2.2.0をインストールし、その後ランタイムアサーションで失敗します。手動でバージョンを固定する必要があります。

さらに大きな問題があります。最新のmmcv wheelは2.2.0で、2024年4月にリリースされて以来更新されていません。ビルド済みバイナリは**torch 2.4 / CUDA 12.1**までしかありません。現在`pip install torch`を実行すると、PyTorch 2.12がインストールされます。この差があるため、mmcvのC++/CUDA演算をソースからコンパイルしなければなりません。作業には10〜30分かかり、対応するCUDAツールキット、`nvcc`、互換性のあるC++コンパイラーが必要です。ドキュメントにも次のような失敗例が並んでいます。

* `ModuleNotFoundError: No module named 'mmcv._ext'`。コンパイル済み演算のビルドに失敗したか、バージョンが一致していません。

* RTX 30シリーズのカードで`nvcc fatal: Unsupported gpu architecture 'compute_86'`が発生します。

* `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`。このスタックではPython 3.8に戻す必要があります。

* GCCのバージョン不一致により、インポート時にセグメンテーションフォールトが発生します。

OpenMMLabのFAQでは、バージョンの問題を避けるためにmimではなくpipでmmcvをインストールするよう案内しています。一方、Get Startedページではmimを使うよう案内しています。どちらも公式ドキュメントですが、互いに矛盾しています。

スタックを動かせても、推論には学習レシピを名前に含む設定ファイルを選び、別途ダウンロードしたハッシュ付きファイル名のチェックポイントを指定し、まず使い方を覚える必要があるレジストリ経由で接続しなければなりません。

LibreYOLOなら、これらすべてが不要です。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")  # 初回実行時に自動ダウンロード
results = model("image.jpg", save=True)
```

`mim`も、4つのパッケージ間のバージョン調整も、ソースのコンパイルも、設定ファイルも、ハッシュ付きチェックポイント探しも、Python 3.8への固定も不要です。重みはアップストリームのOpenMMLabチェックポイントから変換されており、同じチェックポイントを使った場合、推論結果はmmdetectionとビット単位で同等です。

Tiny、Small、Medium、Large、Xの5サイズを利用できます。すべて640 pxで動作します。

```python
print(results[0].boxes.xyxy)  # xyxy座標
print(results[0].boxes.conf)  # 信頼度スコア
```

## 元の実装が適している場合

MMDetectionの拡張パイプライン全体を使ってRTMDetを学習またはファインチューニングする必要がある場合や、mmcvが動作するOpenMMLabエコシステムをすでに深く利用している場合は、そのまま使ってください。推論とデプロイにはLibreYOLOが適しています。

## ライセンスについて

MMDetectionとRTMDetはApache 2.0です。LibreYOLOのコードはMITです。重みにはアップストリームと同じApache 2.0の条件が適用されます。商用利用の制限はありません。

## 試す

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDetl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLOはMITライセンスで、Linux、Mac、Windowsで動作します。コードを変更せずに、GPU、Apple Silicon、通常のCPUで利用できます。1つのAPIでRTMDet、RT-DETR、RF-DETR、D-FINE、YOLOX、YOLO-NAS、セグメンテーション、姿勢推定、深度推定などを扱えます。

GitHubでスターを付けてください：[github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | ドキュメント：[libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
