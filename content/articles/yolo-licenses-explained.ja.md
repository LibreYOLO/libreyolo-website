---
title: "YOLOのライセンスを全解説：v1からYOLO26まで（2026年版）"
description: "2026年版のYOLOライセンス完全マップ。YOLOv1からYOLOv13、YOLO26、YOLO-World、YOLOEまで、オリジナルのリポジトリと寛容なライセンスでの書き直しを論文とGitHubへのリンク付きで整理し、商用製品で実際にリリースできるものを解説します。"
date: 2026-07-11
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, gpl, apache, mit-license, yolov9, yolov10, yolo11, yolov12, yolov13, yolo26]
faq:
  - q: "YOLOv9は無料で商用利用できますか？"
    a: "どのリポジトリを使うかによります。論文のリポジトリ（WongKinYiu/yolov9）はGPL-3.0です。同じ研究室は、YOLOv9とYOLOv7の別の実装をMITライセンスでMultimediaTechLab/YOLOとしても公開しています。これは商用ユーザーから寛容なライセンスの選択肢を求められたことを受けて書かれたものです。GPLのファイルのライセンスを変更したものではなく、書き直しです。コードのライセンスは明確ですが、学習済みの重みにはどちらのリポジトリにも個別のライセンス表記がないため、重みの出所は別の問題として扱ってください。"
  - q: "クローズドソースの商用利用が無料でできるYOLOのバージョンはどれですか？"
    a: "少なくとも1つの寛容なライセンスのリポジトリを通じて利用できるのは、YOLOv1からYOLOv4（オリジナルのDarknet、パブリックドメイン）、YOLOv7とYOLOv9（研究室自身によるMITの書き直し）、YOLOXとPP-YOLOE（Apache-2.0）、そしてLibreYOLOに含まれるMITの再実装です。YOLOv5、v6、v8、v10、YOLO11、v12、v13、YOLO26、YOLO-World、YOLOEには、2026年7月時点で寛容なライセンスの実装がありません。これに頼る前に、現在のLICENSEファイルを自分で確認してください。ライセンスは変わることがあり、これは一般的な情報であって法的助言ではありません。"
  - q: "YOLOv10のライセンスは何ですか？"
    a: "AGPL-3.0です。YOLOv10は清華大学による学術的なリリースですが、そのコードはUltralyticsのコードベースをもとに作られているため、AGPL-3.0を継承しています。寛容なライセンスの再実装はありません。"
  - q: "YOLOv12とYOLOv13のライセンスは何ですか？"
    a: "どちらもAGPL-3.0で、それぞれのLICENSEファイルで確認済みです。YOLOv10と同様に、これらは参照コードがUltralyticsのリポジトリをもとに作られた学術論文なので、論文の著者が誰であってもAGPLが引き継がれます。YOLOv13をApache-2.0としているサードパーティのページは誤りです。"
  - q: "UltralyticsがYOLOをAGPL-3.0に切り替えたのはいつですか？"
    a: "2023年4月14日です。広く繰り返されている2022年ではありません。ultralytics/yolov5のLICENSEファイルがこれまでに変更されたのは3回だけで、AGPLへのコミット（34cf749、PR #11359）の日付は2023-04-14です。2022年最後のリリースであるYOLOv5 v7.0は、まだGPL-3.0で出荷されていました。ultralytics/ultralyticsリポジトリも同じ日にライセンスが変更されました。したがってYOLOv8は2023年1月にGPL-3.0でリリースされています。PyPIのリリース8.0.0から8.0.76まではGPL-3.0を宣言しており、8.0.80（2023年4月16日）がAGPL-3.0を宣言した最初のリリースです。"
  - q: "YOLO26は無料で商用利用できますか？"
    a: "クローズドソースの製品ではできません。YOLO26はAGPL-3.0で配布され、代替手段として有償のEnterprise Licenseが用意されています。YOLOv8やYOLO11と同じ条件です。"
  - q: "オリジナルのDarknet YOLOは本当にパブリックドメインですか？"
    a: "はい。Joseph RedmonのDarknetのLICENSEには「Darknet is public domain. Do whatever you want with it.」（Darknetはパブリックドメインです。好きなように使ってください）と書かれています。AlexeyABのYOLOv4フォークにも同じパブリックドメインの文言があります。落とし穴は、実際に使われているPyTorchへの移植版がそれぞれ独自のライセンスを持っていることで、Apache-2.0からAGPL-3.0、さらにはライセンスなしまでさまざまです。"
  - q: "YOLO-NASの学習済み重みは商用利用できますか？"
    a: "できません。super-gradientsのコードはApache-2.0ですが、公式のYOLO-NASの重みは別のライセンスで配布されており、そこには「may not use the Software for any commercial use, including in connection with any models used in a production environment.」（本番環境で使用されるあらゆるモデルに関連する利用を含め、本ソフトウェアをいかなる商用目的にも使用してはならない）と書かれています。"
  - q: "ニューラルネットワークの重みは学習コードのライセンスを継承しますか？"
    a: "決着していません。Ultralyticsは、AGPL-3.0が「the training code and the models produced by that training code」（学習コードと、その学習コードによって生成されたモデル）を対象とすると表明しており、著作権者として自らが公開するものの条件を定めています。自分で学習させた重みが学習コードの二次的著作物であると裁判所が認めるかどうかは、一度も争われたことがありません。公開されているAGPLの重みは制約付きのものとして扱い、寛容なライセンスの再実装に読み込む際は注意してください。"
  - q: "同じYOLOでも実装によってライセンスが異なるのはなぜですか？"
    a: "著作権が保護するのは表現であり、アイデアではありません（17 U.S.C. 102(b)）。論文に記述されたアーキテクチャは独立して実装でき、そのライセンスは作者が自由に選べます。できないのは、GPLやAGPLのソースファイルをコピーしてライセンスを変更することです。なお、これは著作権に関する議論にすぎません。独立した実装は特許に対する抗弁にはなりません。"
---

YOLOは毎年増え続け、ライセンスの疑問も毎年繰り返し持ち上がります。たいていは製品の意思決定の5分前です。ややこしいのは、「YOLO」が1つのプロジェクトではないことです。異なる作者による20ほどのモデルファミリーが共有するブランド名であり、ほとんどのライセンス解説が見落としている点がここにあります：**ライセンスはモデルではなく、リポジトリに属します**。同じYOLOのバージョンが、同じ作者によるGPLのリポジトリとMITのリポジトリとして同時に存在し得ます。YOLOv9がまさにそうで、これが判断全体を変えます。

このガイドでは、2026年7月時点の状況をリポジトリごとに整理し、各論文とコードベースへのリンクを添えています。以下のライセンスはすべて、実際のリポジトリにある実際のLICENSEファイルと照合して確認しました。YOLOのライセンスについて書かれていること（検索上位の解説記事も含む）には、驚くほど誤りが多いからです。

UltralyticsのAGPLに関する疑問だけを解決したい場合は、[YOLOは無料で商用利用できる？](/articles/yolo-commercial-license)で詳しく取り上げています。この記事は全体の地図です。

**本題の前に、2つの開示事項があります**。1つ目に、私たちはLibreYOLOをメンテナンスしています。LibreYOLOは以下のプロジェクトのいくつかと競合するMITライセンスのライブラリであり、この記事の最後のセクションはLibreYOLOについてです。これは私たちの記述をうのみにせず検証すべき理由であり、だからこそここでのライセンスに関する主張はすべて一次情報源にリンクしています。2つ目に、この記事は公開されているライセンス文書と公の声明についての一般的な情報であり、上記の日付時点のものです。法的助言ではなく、各自の法域の弁護士に代わるものでもありません。ライセンスやメンテナーの立場は変わります。リリースの判断をする前に、どのリポジトリであっても現在のLICENSEファイルを読んでください。

## ライセンス一覧表

「寛容なルート」とは、アプリケーションをオープンソース化することなく、ライセンス料を払うこともなく、クローズドソースの商用製品で使える、そのモデルを実装したリポジトリを指します。

| モデル | 年 | オリジナルのコード | 寛容なルート | 論文 |
| --- | --- | --- | --- | --- |
| YOLOv1 | 2015 | [pjreddie/darknet](https://github.com/pjreddie/darknet)（パブリックドメイン） | オリジナル自体 | [arXiv](https://arxiv.org/abs/1506.02640) |
| YOLOv2 | 2016 | [pjreddie/darknet](https://github.com/pjreddie/darknet)（パブリックドメイン） | オリジナル自体 | [arXiv](https://arxiv.org/abs/1612.08242) |
| YOLOv3 | 2018 | [pjreddie/darknet](https://github.com/pjreddie/darknet)（パブリックドメイン） | オリジナル。AGPLのPyTorch移植版に注意 | [arXiv](https://arxiv.org/abs/1804.02767) |
| YOLOv4 | 2020 | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)（パブリックドメイン） | オリジナル、[pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)（Apache-2.0） | [arXiv](https://arxiv.org/abs/2004.10934) |
| YOLOv5 | 2020 | [ultralytics/yolov5](https://github.com/ultralytics/yolov5)（AGPL-3.0） | なし | 論文なし |
| YOLOv6 | 2022 | [meituan/YOLOv6](https://github.com/meituan/YOLOv6)（GPL-3.0） | なし | [arXiv](https://arxiv.org/abs/2209.02976) |
| YOLOv7 | 2022 | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)（GPL-3.0） | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)（MIT、同じ研究室） | [arXiv](https://arxiv.org/abs/2207.02696) |
| YOLOv8 | 2023 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0。リリース時はGPL-3.0、後述） | なし（Keras移植版は取り下げ、後述） | 論文なし |
| YOLOv9 | 2024 | [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)（GPL-3.0） | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)（MIT、同じ研究室） | [arXiv](https://arxiv.org/abs/2402.13616) |
| YOLOv10 | 2024 | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)（AGPL-3.0） | なし | [arXiv](https://arxiv.org/abs/2405.14458) |
| YOLO11 | 2024 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0） | なし | 論文なし |
| YOLOv12 | 2025 | [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)（AGPL-3.0） | なし | [arXiv](https://arxiv.org/abs/2502.12524) |
| YOLOv13 | 2025 | [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13)（AGPL-3.0） | なし | [arXiv](https://arxiv.org/abs/2506.17733) |
| YOLO26 | 2026 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0） | なし | [arXiv](https://arxiv.org/abs/2606.03748) |
| YOLOX | 2021 | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)（Apache-2.0） | オリジナル自体 | [arXiv](https://arxiv.org/abs/2107.08430) |
| PP-YOLOE | 2022 | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)（Apache-2.0） | オリジナル（PaddleYOLOではなくPaddleDetectionから） | [arXiv](https://arxiv.org/abs/2203.16250) |
| YOLO-NAS | 2023 | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients)（コードはApache-2.0、重みは非商用） | コードは可、公式の重みは不可 | 論文なし |
| YOLO-World | 2024 | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World)（GPL-3.0） | なし | [arXiv](https://arxiv.org/abs/2401.17270) |
| YOLOE | 2025 | [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe)（AGPL-3.0） | なし | [arXiv](https://arxiv.org/abs/2503.07465) |
| MMYOLO | 2022 | [open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo)（GPL-3.0、2024年以降停滞） | なし | 論文なし |

列ごとに見ていくと、1つのことが明らかになります：バージョン番号はライセンスについて何も教えてくれません。教えてくれるのはリポジトリです。

## 同じモデルに2つのライセンス：YOLOv9のケース

YOLOv9は、「YOLOvNのライセンスは何か」という問い自体が間違っていることを最もはっきり示す例です。その経緯は正確に伝える価値があります。YOLOの歴史の中で、コミュニティの声が実際に結果を変えた唯一の例だからです。

- **2024年2月18日**：[WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)が[論文](https://arxiv.org/abs/2402.13616)とともに公開されました。
- **2024年2月22日**：あるユーザーがライセンスを尋ねるissue #10を立てました。Chien-Yao Wang（WongKinYiu）は「I think it should be GPL3」（GPL3にすべきだと思う）と返信し、4日後にGPL-3.0のファイルを追加しました。
- **2024年2月26日**：別のユーザーが[issue #82「An Apache/MIT rewrite」](https://github.com/WongKinYiu/yolov9/issues/82)を立てました。その後の2週間半で、十数人の商用ユーザーが次々と加わりました。
- **2024年3月14日**：Wangがそのスレッドに投稿しました：「*Okay I create a new repo for mit rewrite... I think I can handle most of implementation of architectures and loss functions. And need someone to give great help about dataloader and ddp training.*」（わかりました、MITでの書き直し用に新しいリポジトリを作ります……アーキテクチャと損失関数の実装の大部分は自分で対応できると思います。データローダーとDDP学習については誰かに大いに手伝ってほしいです）

当初`yolov9mit`という名前だったそのリポジトリが、現在の[**MultimediaTechLab/YOLO**](https://github.com/MultimediaTechLab/YOLO)です。MITライセンスで、著作権表記は「Kin-Yiu, Wong and Hao-Tang, Tsui」、自らを「*the official implementation of YOLOv7 and YOLOv9, YOLO-RD.*」（YOLOv7、YOLOv9、YOLO-RDの公式実装）と説明しています。書き直しの大部分を担ったのはスレッドのボランティアではなく、Wangの研究室の同僚であるHao-Tang Tsuiで、コミットのおよそ90パーセントを書いています。

つまりYOLOv9は、どのリポジトリをクローンするかによって「出荷できない」ものにも「出荷できる」ものにもなります。同じアーキテクチャ、同じ研究室、2つのライセンスです。

**製品をこれに賭ける前に、リポジトリのトップページには書かれていない3つの注意点があります**：

1. **まだ成熟途上です**。2026年2月の[未解決のissue](https://github.com/MultimediaTechLab/YOLO/issues/231)では、配布されているチェックポイントが論文のCOCOの数値を再現できず、論文の記載よりも明らかにパラメータが多いと報告されています。セグメンテーションとキーポイントの学習は[まだ実装されていません](https://github.com/MultimediaTechLab/YOLO/issues/232)。2025年12月のv1.0タグ以降、mainには何もマージされていません。
2. **重みには個別のライセンス表記がありません**。リポジトリのMIT LICENSEは慣例的にそのリリースアセットにも及び、技術的な証拠からは、チェックポイントはGPLのリポジトリからコピーされたものではなく、このプロジェクトで学習されたものと考えられます：ファイルのサイズとパラメータ数が、GPLのリポジトリのチェックポイントとは異なっているからです。しかし、.ptファイルがMITであると明記した文書はなく、コミュニティからの[AGPLの混入に関する正式な監査の要望](https://github.com/MultimediaTechLab/YOLO/issues/51)も未解決のままです。これは書類上の空白です：「クリーンであることを検証済み」と主張した人はおらず、問題を指摘した人もいません。
3. **別のコードベースです**。GPLのリポジトリのスクリプトをそのまま置き換えられるものではありません。

だからといって使えないわけではありません。チェックボックスに印を付けて終わりにするものではなく、評価すべきプロジェクトだということです。

## 第1期：パブリックドメインの時代（YOLOv1〜YOLOv4）

Joseph Redmonは[YOLOv1](https://arxiv.org/abs/1506.02640)、[YOLOv2](https://arxiv.org/abs/1612.08242)（YOLO9000の論文として発表）、[YOLOv3](https://arxiv.org/abs/1804.02767)を自身のDarknetフレームワークの中で公開しました。その[LICENSEファイル](https://github.com/pjreddie/darknet/blob/master/LICENSE)は、わずか3行であることで有名です：

> 0. Darknet is public domain.
> 1. Do whatever you want with it.
> 2. Stop emailing me about it!

これは正真正銘のパブリックドメインへの献呈です。Alexey Bochkovskiyが[Darknetのフォーク](https://github.com/AlexeyAB/darknet)でメンテナンスしている[YOLOv4](https://arxiv.org/abs/2004.10934)にも*同じ*ファイルが含まれており、いくつかのライセンス解説が主張するUnlicenseではありません。GitHub自身の分類機能はこれを判別できず、「Other」と表示します。コンプライアンスツールがこのフィールドを読んでいる場合は知っておく価値があります：文面はこれ以上ないほど寛容であるにもかかわらず、スキャナーはこれを問題のない寛容なライセンスとしてではなく、識別できないものとしてフラグを立てます。

Darknet自体も、完全に終わったわけではありません。Redmonのリポジトリは2022年から動きがありませんが、AlexeyABのREADMEは現在、推奨される活発にメンテナンスされたC/C++の後継として[hank-ai/darknet](https://github.com/hank-ai/darknet)を案内しています。

**落とし穴は移植版です**。2026年にDarknetで学習する人はほとんどいません。多くの人はPyTorchによる再実装を使いますが、それらには独自のライセンスがあります：

- 最も人気のあるYOLOv3の移植版である[ultralytics/yolov3](https://github.com/ultralytics/yolov3)は**AGPL-3.0**です。パブリックドメインのアーキテクチャが、この分野で一般的なライセンスの中で最も厳しいもので再配布されているわけです。どのライセンスになるかは、誰の移植版をpip installしたかで完全に決まります。
- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)は**GPL-3.0**です。
- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)は**Apache-2.0**なので、v4についてはPyTorchでも寛容なルートが残っています。
- YOLOv4の共著者による[WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)には、**LICENSEファイルがまったくありません**。これはコピーレフトよりも悪い状況です：ライセンスがなければ何の権利も許諾されておらず、デフォルトではすべての権利が留保されています。

論文に書かれたバージョン番号ではなく、クローンしたリポジトリのライセンスを確認してください。

## 第2期：GPL-3.0（YOLOv6、v7、v9）

[YOLOv6](https://github.com/meituan/YOLOv6)（Meituan）、[YOLOv7](https://github.com/WongKinYiu/yolov7)、[YOLOv9](https://github.com/WongKinYiu/yolov9)はGPL-3.0で公開されました。

GPL-3.0はコピーレフトです：改変版を頒布する場合、またはGPLのコードを自分のコードと組み合わせて単一のプログラムにする場合、その結合著作物の完全な対応ソースコードをGPL-3.0のもとで提供しなければなりません。重要な境界が2つあります。**単なる集積**（mere aggregation）とは、別々の独立したプログラムを単に一緒に出荷しても、自動的に結合されたことにはならないという意味です。そして、義務の発生条件は**頒布**です：AGPLと異なり、結合著作物を一切出荷しなければ、通常のGPLは開示義務を課しません。一部の企業がGPLのモデルを厳密に自社のAPIの背後に置いているのはこのためです。

この道は見かけより狭く、あまり想定されていない3つの形で破綻します。モデルが顧客のデバイスやオンプレミス環境に置かれた瞬間、それは頒布です。コピーが自社の法人の外に出た時点で、「社内のみ」は成り立たなくなります：1つの会社の中でコピーすることは頒布にあたりませんが、ビルドを業務委託先、外部の開発チーム、別法人の関連会社に渡すことは頒布にあたります。さらに、この防御が成り立つのは、そのAPIから到達できる*すべて*がGPLかそれより寛容なライセンスである場合に限られます。提供される著作物のどこか1か所にでもAGPLのコンポーネントがあれば、GPLのファイル自体に何が書かれていても、全体が第13条の対象になるからです。

**YOLOv7とYOLOv9には、前述のMITという抜け道があります**。YOLOv6にはありません。その最も強い証拠はBaiduから来ています：PaddleDetectionはApache-2.0ですが、そのメンテナーはYOLOv5、YOLOv6、YOLOv7、YOLOv8を意図的に別のGPL-3.0リポジトリである[PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO)に隔離し、これらのモデルのコードは「will not be merged into PaddleDetection」（PaddleDetectionにはマージされない）と明言しています。Baiduほどの大企業が、YOLOv6を自社の寛容なフレームワークから締め出すために壁を築いているという事実が、このライセンスがどう評価されているかを物語っています。

## 第3期：AGPL-3.0（YOLOv5、v8、v10、11、v12、v13、26、YOLOE）

UltralyticsがAGPL-3.0に移行したのは**2023年4月14日**で、非常に広く（この疑問で検索結果の1ページ目に表示される解説記事でさえ）繰り返されている2022年ではありません。それ以降のもの（[YOLOv8](https://github.com/ultralytics/ultralytics)、YOLO11、[YOLO26](https://arxiv.org/abs/2606.03748)）はすべてAGPL-3.0で、代替手段として有償のEnterprise Licenseがあります。同社の[ライセンスページ](https://ultralytics.com/license)には、Enterpriseプランが「YOLO26, earlier YOLO versions, and any future YOLO models」（YOLO26、それ以前のYOLOバージョン、および将来のすべてのYOLOモデル）を対象とすると書かれており、この方針が意図的かつ将来を見据えたものであることがわかります。価格は公開されていません。

この日付は検証でき、検証する価値もあります。広く流布している説は、重要な点で間違っているからです：

- `ultralytics/yolov5`の`LICENSE`ファイルは、これまでにちょうど3回しか変更されていません。3回目がコミット[`34cf749`](https://github.com/ultralytics/yolov5/commit/34cf749958d2dd3ed1205f6bb07e0f20f6e2372d)「Update LICENSE to AGPL-3.0 (#11359)」で、日付は**2023-04-14**です。このコミットは101個のファイルにわたって`GNU GENERAL PUBLIC LICENSE`を`GNU AFFERO GENERAL PUBLIC LICENSE`に置き換えています。
- 2022年最後のYOLOv5リリースである**v7.0（2022年11月22日）は、まだGPL-3.0で出荷されていました**。それ以前のv6.2、v6.1、v6.0も同様です。2023年4月より前に同期を止めた何千ものフォークも同じで、どれを選んでもGitHubは今もGPL-3.0と表示します。
- 同じ日の41分後、`ultralytics/ultralytics`も[同じ変更](https://github.com/ultralytics/ultralytics/commit/2c6fc0a4443b9cf805ef17b1cfdd71a98693b4d4)を受けました（PR #2031）。

ここから、ほとんど知られていない事実が導かれます：**YOLOv8はAGPLでリリースされたのではありません**。2023年1月に**GPL-3.0**でリリースされ、3か月後にライセンスが変更されました。PyPIの記録は明白です：`ultralytics` 8.0.0（2023年1月10日）から8.0.76（2023年4月13日）までは、すべて`GPL-3.0`を宣言しています。2023年4月16日にアップロードされたバージョン8.0.80が、`AGPL-3.0`を宣言した最初のバージョンです。

これは抜け道ではありません。一度許諾されたライセンスがさかのぼって取り消されることはないので、これらの古いバージョンは公開時の条件のまま引き続き利用できます。しかし、それらは3年前のもので、メンテナンスもパッチもされていません。GPL-3.0もコピーレフトであることに変わりはなく（ネットワーク条項を頒布条項に置き換えただけで、コピーレフトから逃れたわけではありません）、そもそもUltralyticsの商用に関する立場はライセンス文面よりも広範です。役に立つ教訓は、もっと限定的で、もっと一般的なものです：**ライセンスはプロジェクトの名前ではなく、受け取ったバージョンに付随します**。依存関係のバージョンを固定し、コピーを取得した日にライセンスに何が書かれていたかを記録してください。プロジェクトは明日にもライセンスを変えられますし、コンプライアンス上の説明は、実際にどのバージョンを入手したかにかかっているからです。

AGPL-3.0は、GPLにネットワーク条項である第13条を加えたものです：*プログラムを改変した場合*（if you modify the Program）、ネットワーク経由で改変版とやり取りするユーザーには、そのソースを提供しなければなりません。これにより、通常のGPLでは開いていたSaaSという道が閉ざされます。

この限定条件に注意してください。ほとんどの記事が省いているからです。AGPLの第0条は、「modify」（改変）を、*著作権上の許諾を必要とする方法で*（in a fashion requiring copyright permission）著作物をコピーまたは翻案することと定義しています。改変していない学習コードを自分のデータセットに対して実行することは、プログラムを実行しているのであって、そのソースを翻案しているのではありません。改変していないGPLのコンパイラーで自分のコードをコンパイルしても、コンパイラーを改変したことにならないのと同じです。したがって、学習だけで自動的に義務が発生するわけではなく、そうだと言う人は定義を読み飛ばしています。

ライセンスの範囲に入ることになるのは、パッケージを自分のコードベースに結合または組み込み、両者が1つのプログラムとして出荷または提供される場合です。FSF自身の解釈では、GPLやAGPLの著作物をほかのモジュールとリンクすると結合著作物になります。それがPythonのimportにまで及ぶと裁判所が判断するかどうかは、一度も争われたことがありません。また、Ultralytics自身の[商用条件](https://ultralytics.com/license)は、裏側でYOLOを使うSaaSプラットフォーム、API、クラウドシステムにEnterprise Licenseを求めており、ベンダーの立場はライセンス文面そのものよりも広範である点にも注意してください。リスクとして扱うべきは、提供または出荷する製品への統合です。単独での学習がリスクだと想定しないでください。

YOLO26については、日付がいたるところで混同されているので整理しておきます：YOLO Visionで**2025年9月にプレビュー公開**され、**実際のリリースは2026年1月**（`ultralytics` 8.4.0のリリースで、そのリリースノートには「Ultralytics YOLO26 has arrived」とあります）、そして**論文は2026年6月に公開**されました。YOLO26はメインの`ultralytics/ultralytics`リポジトリにあり、`ultralytics/yolo26`はランディングページにすぎません。

**意外に思われる点**：ここ2年の学術系YOLOもAGPLです。Ultralyticsが書いたからではありません。

- [YOLOv10](https://arxiv.org/abs/2405.14458)は清華大学によるものです。そのREADMEには「*The code base is built with ultralytics.*」と書かれています。
- [YOLOv12](https://arxiv.org/abs/2502.12524)：「*The code is based on ultralytics.*」
- [YOLOv13](https://arxiv.org/abs/2506.17733)：「*The code is based on Ultralytics.*」
- YOLOv10のグループによるオープンボキャブラリの「see anything」モデルである[YOLOE](https://arxiv.org/abs/2503.07465)も、同様にAGPL-3.0で、Ultralyticsをもとに作られています。

コピーレフトは継承されます：AGPLのコードの派生物はAGPLのもとで提供しなければならないため、これらのフォークのいずれかが出荷または提供された瞬間に、ライセンスもついてきます。（改変しても厳密に自分の手元だけにとどめておけば義務は発生しません。研究用途が影響を受けないのはこのためです。）研究は独立していても、ライセンスは独立していません。2024年以降、「次のYOLO」をUltralyticsの派生物として公開することが標準的なやり方になっており、その結果、AGPLはバージョン番号を追うように自動的に広がっていきます。

YOLOv13をApache-2.0だとするページを見かけたら、それは誤りです。LICENSEファイル、GitHub API、モデルカードのすべてがAGPL-3.0となっています。

### もう存在しないYOLOv8の寛容なルート

今もオンラインにある多くの解説（この記事の初期の草稿も含む）は、AGPLのYOLOv8から抜け出すクリーンな手段として、KerasCVのApache-2.0版YOLOv8を挙げています。**現在はこれに頼ることはできません**。公開されている記録は次のとおりです：

- KerasCVのディスカッション[Clarifications regarding YOLOv8 licensing](https://github.com/keras-team/keras-cv/discussions/2032)で、あるコントリビューターが「many parts are derived form ultralytics」（多くの部分がultralyticsから派生している）と書きました。メンテナーは誰もこのスレッドに回答しておらず、実装がどの程度独立していたのかという問題は、公の場ではどちらとも決着していません。
- KerasHubのissue [add YOLOV8](https://github.com/keras-team/keras-hub/issues/1760)は、2025年7月にKerasのメンテナーによって次のコメントとともにクローズされました：「Hey all!! Because of license issues, we have decided to drop this.」（皆さん！！ライセンスの問題により、これは取りやめることにしました）。その後2人が、KerasCV版を使っている既存の商用ユーザーにとってそれが何を意味するのかを尋ねました。どちらにも返信はありませんでした。
- KerasCVは現在アーカイブされており、メンテナンスされている後継のKerasHubは**YOLOv8を提供していません**。

この2つの出来事が関係しているかどうかはわかりませんし、何か不適切なことがあったと主張しているわけでもありません。これは製品を築く土台にはなりません。

## 寛容なライセンスの例外：YOLOX、PP-YOLOE、YOLO-NAS

すべてのYOLOがコピーレフトの流れに乗ったわけではありません。

- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)**（Megvii、2021年）は、コードも重みも例外なくApache-2.0です。商用利用においては今でも最もクリーンな古典的YOLOですが、リポジトリには2025年半ば以降コミットがなく、[学習済みの重みを商用アプリに同梱することについての質問](https://github.com/Megvii-BaseDetection/YOLOX/issues/1865)は2026年3月から返信のないまま残っています。ライセンスの文面は明確です。ただ、現在それについての質問に答えられる人がいないだけです。[LibreYOLOでYOLOXを動かす方法](/articles/yolox-with-libreyolo)についても記事を書いています。
- **[PP-YOLOE](https://arxiv.org/abs/2203.16250)**（Baidu）は、**PaddleDetectionの中では**Apache-2.0です。PaddleYOLOからではなく、PaddleDetectionから取得してください。PaddleYOLOにはほぼ同一の設定ディレクトリが含まれていますが、GPL-3.0です。同じモデル、同じ会社、2つのリポジトリ、2つのライセンス。YOLOv9の状況を逆にしたものです。
- **[YOLO-NAS](https://github.com/Deci-AI/super-gradients)**（Deci、2023年）は、多くの人がつまずくケースです。コードはApache-2.0ですが、公式の重みはそうではありません。その[別ライセンス](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md)には、「*may not use the Software for any commercial use, including in connection with any models used in a production environment.*」（本番環境で使用されるあらゆるモデルに関連する利用を含め、本ソフトウェアをいかなる商用目的にも使用してはならない）と定められています。2024年にNVIDIAがDeciを買収して以降、リポジトリに新機能の開発は見当たらず、元の重みのURLはもう解決できません（チェックポイントは現在、別のホストにあります）。アーキテクチャをゼロから学習すればApache-2.0の条件だけで済むという回避策がときどき提案されますが、もっともらしいものの、DeciやNVIDIAによって確認されてはおらず、「any components comprising the model」（モデルを構成するあらゆるコンポーネント）にまで及ぶライセンス文面とはうまく整合しません。YOLO-NASについての詳細は[こちら](/articles/yolo-nas-with-libreyolo)。
- [**MMYOLO**](https://github.com/open-mmlab/mmyolo)は、古い解説ではApache-2.0と記載されていることがあります。実際は違います：MMYOLOはGPL-3.0で（Apacheライセンスなのは兄弟プロジェクトのMMDetectionのほうです）、2024年7月以降コミットがありません。
- **[YOLO-World](https://github.com/AILab-CVC/YOLO-World)**（Tencent）は**GPL-3.0**で、AGPLでも寛容なライセンスでもありません。オープンボキャブラリのYOLOは、借用元であるCLIP系のエコシステムからの連想で寛容なライセンスだと思われがちなので、明記しておきます。

これらのケースは、この分野で最もよくある3つのライセンス上の誤りを網羅しています：重みがコードのライセンスに従うと思い込むこと、1つの組織から出たものはすべて同じライセンスだと思い込むこと、そしてモデルのライセンスを名前から推測することです。

## ライセンスが対象とするのはコードであり、アイデアではない（ただし特許には注意）

上記のすべての寛容なルートの法的根拠となっているのは、1つの原則です：**著作権が保護するのは表現であり、アイデアではありません**。これはスローガンではなく、大西洋の両側で確立された法原則です。

米国では[17 U.S.C. 102(b)](https://www.law.cornell.edu/uscode/text/17/102)がそれにあたり、「idea, procedure, process, system, method of operation, concept, principle, or discovery」（アイデア、手順、プロセス、システム、操作方法、概念、原理、発見）を著作権保護の対象から除外しています。この考え方は*Baker v. Selden*（1879年）にまでさかのぼります。EU（私たちに適用される法であり、おそらく多くの読者にとってもそうです）では、[ソフトウェア指令（2009/24/EC）](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32009L0024)の第1条第2項が、「ideas and principles which underlie any element of a computer program」（コンピュータープログラムのあらゆる要素の基礎となるアイデアおよび原理）は保護されないと定めています。さらに欧州司法裁判所は*SAS Institute v World Programming*（[C-406/10](https://curia.europa.eu/juris/liste.jsf?num=C-406/10)、2012年）において、プログラムの機能、プログラミング言語、データ形式はそれ自体としては保護される表現ではないことを確認しました。保護されるのはコードだけです。

arXivの論文に記述されたアーキテクチャは、公開されたアイデアです。誰でも実装できます。できないのは、他人のGPL/AGPLのソースファイルをコピーまたは翻案し、その結果のライセンスを変更することです。ソースではなく論文をもとに作業した人々が書いた独立した実装は新しい著作物であり、そのライセンスは作者が選びます。

多くのベンダーが省略する、正直な注意点が2つあります：

- **「独立」が本当でなければなりません**。これを正当に主張できる形はクリーンルーム方式です：コードを書く人は、コピーレフトのソースではなく、論文と仕様をもとに作業します。GPLのリポジトリをよく読んでから「書き直す」のは同じことではなく、誰かが調べることになればその違いが重要になります。
- **これは著作権の議論であり、特許の議論ではありません**。独自に発明したことは、特許に対する抗弁には*なりません*。クリーンルームでの再実装は著作権の問題を解決しますが、特許の問題はまったく手つかずのまま残ります。現在、YOLOのアーキテクチャに対して広く主張されている特許はありませんが、「まだ誰も訴えていない」ことは「訴える根拠が何もない」ことと同じではありません。

## 重みには別のライセンスがある

この分野全体で最も高くつく間違いは、コードを監査してチェックポイントを忘れることです。ライセンスが付随する層は3つあり、それぞれ異なる可能性があります：

**1. コード**。ここまで述べたことすべてです。

**2. 重み**。YOLO-NASのように明示されている場合があります。主張されている場合もあります：Ultralyticsの[ライセンスページ](https://ultralytics.com/license)は、AGPL-3.0が「covers the training code and the models produced by that training code」（学習コードと、その学習コードによって生成されたモデルを対象とする）と述べ、その範囲を同社のコードを使って自分で学習させたモデルにまで広げています。自らが公開するものの著作権者として、同社は自社のチェックポイントの条件を定めることができます。*自分で*生成した重みが法的に学習コードの二次的著作物にあたるかどうかは、本当に未解決の問題です：判例はなく、プログラムの出力は自動的にはそのプログラムの著作権の対象にならないというFSF自身の一般原則とも相反します。ただし、未解決であることは安全であることと同じではありません。実務上は、公開されているAGPLの重みは制約付きのものとして扱ってください。

この記事のアドバイスに従う人が特に注意すべき点：**GPLのYOLOv7とYOLOv9のリポジトリは、重みについて何も述べていません**。それらのチェックポイントはGPL-3.0リポジトリのリリースアセットであり、個別の許諾はありません。その.ptファイルをMITの再実装に読み込むと、コードはクリーンでも、チェックポイントの位置付けは誰も明らかにしていない状態になります。ライセンスが乗り換えた理由なのであれば、寛容なライセンスのプロジェクトが自ら学習させた重みを使うか、自分で学習させてください。

**3. 学習データ**。COCOの*アノテーション*はCC BY 4.0なので、ラベルは問題ありません。*画像*はCOCOがライセンスできるものではありません：個別の条件がばらばらに付いたFlickrの写真であり、COCOコンソーシアムはそれらを所有していないと明言しています。業界の大半はこれを低リスクとみなして先に進みますが、「COCOなら大丈夫」というのはアノテーションだけについての話です。COCO以外では一気に厳しくなります：航空写真、医療、運転のデータセットの多くは非商用であり、モデルのコードが寛容なライセンスでも、それらの制約が消えるわけではありません。

毎回、3つの層すべてを監査してください。

## MITとApache-2.0の比較について

私たちはMITライセンスのライブラリを提供しているので、MITこそが最良のライセンスだと言えれば好都合です。しかし正直に比較すると、もっと興味深い話になります。

MITも義務がゼロではありません：頒布するコピーには著作権表示とライセンス文を残さなければなりません。これは実際の条件ですが、負担の小さいものです。そしてApache-2.0には、MITにはないものがあります：すべてのコントリビューターからの**明示的な特許許諾**と、その著作物をめぐって訴訟を起こした者の許諾を終了させる報復条項です。MITは特許について何も述べていません。特許リスクを最も懸念する企業にとっては、2つのうちApache-2.0のほうが*より安全*だという見方もできます。寛容なライセンスの物体検出エコシステムの多く（YOLOX、RT-DETR、D-FINE、DEIM）がApache-2.0なのは、まさにそのためです。

MITで得られるのは、シンプルさと互換性です。どちらのライセンスも、ソースの公開を強制することは決してありません。そしてこの記事が本当に扱っているのは、まさにその軸です。MITかApacheかが重要な判断だと言う人がいたら、その人は何かを売り込もうとしています。重要な判断は、寛容なライセンスかコピーレフトかです。

## 結局、どのYOLOを使えるのか

- **クローズドソースの商用製品**：研究室の[MITリポジトリ](https://github.com/MultimediaTechLab/YOLO)経由のYOLOv7またはYOLOv9、オリジナルの中ではYOLOXまたは（PaddleDetectionの）PP-YOLOE、あるいは自分で組み合わせて監査したくなければLibreYOLOのようなメンテナンスされているMITフレームワーク。アプリケーション全体を公開するか、Enterprise Licenseを購入するつもりがない限り、GPLやAGPLのものは避けてください。
- **オープンソースプロジェクト**：ライセンスに互換性がある限り、何でも使えます。プロジェクトがAGPLなら、Ultralytics系は自然な選択肢であり、最新の研究（YOLOv13、YOLO26、YOLOE）もまずそこに登場します。
- **研究とベンチマーク**：ライセンスによる制約はほとんどありません。比較対象の論文が使ったものを使ってください。
- **「後で対処する」という計画**：うまくいきません。製品ができあがった後でAGPLへの依存を解消するには、重みを含むすべてを再学習、再検証、再エクスポートする必要があります。最初にリポジトリを、つまりライセンスを選んでください。

ライセンスではなくフレームワークそのものの比較については、[2026年版 Ultralyticsの代替ライブラリ比較](/articles/best-ultralytics-alternatives)を参照してください。

## MITという選択肢：LibreYOLOが提供するもの

開示：LibreYOLOは私たちのプロジェクトであり、そのライセンスこそがこの記事が存在する理由です。

**[LibreYOLO](https://github.com/LibreYOLO/libreyolo)は、MITライセンスの単一のコードベースです**。以下の検出器を1つのAPIで扱え、学習とONNX、TensorRT、OpenVINO、NCNNへのエクスポートも組み込まれています。コピーレフトも、ネットワーク条項も、エンタープライズプランもありません。

以下は、物体検出モデルの全ラインナップと、各ファミリーのアップストリームおよびそのライセンスです。それぞれの寛容なルートがどこから来ているのかを正確に確認できます：

| LibreYOLOでの名前 | モデル | アップストリーム | アップストリームのライセンス |
| --- | --- | --- | --- |
| `LibreYOLO2`、`LibreYOLO3`、`LibreYOLO4` | Darknet時代のYOLOをPyTorchで実装 | [pjreddie/darknet](https://github.com/pjreddie/darknet)、[AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) | パブリックドメイン |
| `LibreYOLO7` | YOLOv7 | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT（GPLのリポジトリではなく、研究室自身による書き直し） |
| `LibreYOLO9`、`LibreYOLO9E2E` | YOLOv9と、NMSフリーのエンドツーエンドのバリアント | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT（GPLのリポジトリではなく、研究室自身による書き直し） |
| `LibreYOLO9P2` | 小さな物体向けにストライド4のヘッドを備えたYOLOv9 | LibreYOLOオリジナル | MIT |
| `LibreYOLOX` | YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 |
| `LibreYOLONAS` | YOLO-NAS（検出と姿勢推定） | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) | コードはApache-2.0。制限付きのアップストリームの重みは再配布**していません** |
| `LibreRTDETR`、`LibreRTDETRv2` | RT-DETRとv2 | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 |
| `LibreRTDETRv4` | RT-DETRv4 | [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | Apache-2.0 |
| `LibreRFDETR` | RF-DETR（検出、セグメンテーション、姿勢推定、OBB） | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | N/S/M/LはApache-2.0。XL/2XLはRoboflowのPlatform Model Licenseのため、Apacheのサイズのみを提供しています |
| `LibreDFINE` | D-FINE | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 |
| `LibreDEIM` | DEIM | [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | Apache-2.0 |
| `LibreDEIMv2` | DEIMv2 | [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | コードはApache-2.0。大きいサイズはMeta独自の非OSIライセンスのもとでMetaのDINOv3バックボーンを使用しており、再配布は認められていますが、**軍事、核、スパイ活動、兵器用途での使用は禁止されています**。これらのサイズはApacheではなく`other`として公開しています。デュアルユースにあたるものを出荷する前に、このライセンスを読んでください |
| `LibreRTMDet` | RTMDet（[MMDetectionなし](/articles/rtmdet-without-mmdetection)） | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | Apache-2.0（GPL-3.0のmmyoloではなく、Apacheライセンスの兄弟プロジェクト） |
| `LibrePICODET` | PP-PicoDet | アーキテクチャは[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)から、チェックポイントは[Picodet_Pytorch](https://github.com/Bo396543018/Picodet_Pytorch)による再移植版から | Apache-2.0（両方） |
| `LibreEC` | EdgeCrafter（検出、姿勢推定、セグメンテーション） | [Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) | Apache-2.0 |
| `LibreFOMO` | マイコンクラスのハードウェア向けの重心検出器 | LibreYOLOオリジナル | MIT |
| `LibreGroundingDINO` | Grounding DINO、オープンボキャブラリ（推論） | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 |
| `LibreOWLv2` | OWLv2、オープンボキャブラリ（推論） | Google（`transformers`経由） | Apache-2.0 |

この列が意図的に正直にしている点が2つあります。

**GPLやAGPLのソースは、コードベースに一切コピーされていません**。上記のすべてのファミリーは、パブリックドメイン、MIT、またはApache-2.0のアップストリームから作られており、これがフレームワークをMITに保つルールです。あるアーキテクチャに寛容なライセンスの実装とコピーレフトの実装の両方がある場合は、寛容なほうをもとにします：私たちのYOLOv9とYOLOv7はGPLのオリジナルではなく研究室のMITリポジトリに、RTMDetはGPLライセンスのMMYOLOではなくApacheライセンスのMMDetectionに由来します。YOLO-Worldは定期的に要望がありますが、GPL-3.0であり、寛容なライセンスで取得できる場所がないため、リストに含まれていません。

**寛容なライセンスは無制限と同じではなく、監査で気付かれるよりも先にそう伝えておきたいと考えています**。DEIMv2の大きいサイズは、軍事、核、スパイ活動、兵器用途の禁止を含むMetaのDINOv3の条件を継承しており、デュアルユースに少しでも関わる仕事をしているなら現実の制約になります。いくつかの研究プレビュー版のチェックポイントは非商用のデータセット（DOTA、VisDrone）で学習されており、無料であるかのように黙って提供するのではなく、Hugging Faceで`cc-by-nc`とラベル付けしています。そして、ほかの人の重みに適用するのと同じ注意は、私たちの重みにも当てはまります：私たちのYOLO9のチェックポイントはMITの研究室リポジトリ自身のチェックポイントから変換したものなので、それらについて言えることをそのまま継承します。前述のとおり、それは「主張されているが、正式には監査されていない」という状態です。MITライセンスが対象とするのは、私たちが書いたコードです。各チェックポイントには、何を使って、何から学習したかに応じた条件が付いており、重みごとに公開しています。

検出以外にも、同じAPIでインスタンスセグメンテーションとセマンティックセグメンテーション、姿勢推定、回転バウンディングボックス、分類（MobileNetV4、ConvNeXt、EfficientNetV2、ResNet、CLIP）、単眼深度推定、画像復元、視線推定を扱えます。

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
```

同じYOLOのワークフローで、ライセンスの下調べは不要です。

GitHubでスターを付けてください：[github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | ドキュメント：[libreyolo.com/docs](https://libreyolo.com/docs)

---

*商標と提携関係：YOLO、YOLOv8、YOLO11、YOLO26、Ultralyticsは、Ultralytics Inc.を含むそれぞれの所有者の商標です。LibreYOLOは独立したプロジェクトであり、Ultralyticsおよびこの記事で言及しているその他のいかなる組織とも提携しておらず、それらから後援や推奨を受けてもいません。製品名およびリポジトリ名は、取り上げたソフトウェアを識別し比較する目的でのみ使用しています。*

*この記事は一般的な情報であって法的助言ではなく、2026年7月11日時点のものであり、比較対象のベンダーの1つによって書かれています。出荷する前に、現在のライセンスを確認してください。*
