---
title: 引用
seo_title: LibreYOLOとアップストリームの著者を引用する方法
description: 論文でLibreYOLOを引用する方法と、使用したモデルファミリーの著者を引用する方法を説明します。どちらも同じ手法セクションに記載します。
lead: LibreYOLOを完全に引用するには、ライブラリと、結果を生成したモデルファミリーの基になった公開研究の2つが必要です。
keywords:
  - libreyolo 引用
  - libreyolo bibtex
  - libreyolo citation cff
  - モデル 引用
  - コンピュータビジョン 引用
last_verified: 1.5.0
source_hash: 0f3f23e4e85e38be
---

## LibreYOLOの引用

リポジトリは引用メタデータをBibTeXブロックではなく、[`CITATION.cff`](https://github.com/LibreYOLO/libreyolo/blob/release/CITATION.cff)として公開しています。GitHubはこのファイルを読み取り、リポジトリページに「Cite this repository」ボタンを表示します。このボタンからAPA形式とBibTeXを生成できます。自分で入力せず、そこからエントリを取得してください。

ファイルの全文は次のとおりです。

```yaml
cff-version: 1.2.0
message: "If you use LibreYOLO in your research or software, please cite it as below."
title: "LibreYOLO"
type: software
authors:
  - family-names: Ceccon
    given-names: Xuban
  - name: "The LibreYOLO contributors"
license: MIT
url: "https://github.com/LibreYOLO/libreyolo"
repository-code: "https://github.com/LibreYOLO/libreyolo"
```

意図的にバージョンとリリース日は含まれていません。[`RELEASING.md`](https://github.com/LibreYOLO/libreyolo/blob/release/RELEASING.md)では、リリース時に `CITATION.cff` や `.zenodo.json` のバージョン、日付、タイトルを変更しないようメンテナーに指示しています。これにより、引用がバージョンごとに分散せず、すべて1つの記録に集約されます。実行したバージョンは自分の本文で報告し、引用情報は変更しないでください。

## モデルファミリーの引用

LibreYOLOは移植版です。`LibreRFDETRm.pt` を実行することはRF-DETRを実行することであり、査読者がクレジット表記を期待する相手はRF-DETRの著者です。ライブラリだけを引用すると、その研究成果を誤ったプロジェクトに帰属させることになります。

必要な情報はすべてファミリーのページにあります。ヘッダーの「Upstream」行には元の研究とその組織が記載され、論文とソースリポジトリへのリンクがあります。ページ下部の「Citation」セクションにはBibTeXがあります。

このBibTeXは、通常はアップストリームREADMEの「Citation」セクションまたは `CITATION.cff` にある、著者自身の引用ブロックからそのままコピーされています。表示時には出典ブロックへのリンクも付くため、ソースと照合できます。論文のメタデータから組み立てることはありません。エントリを手作業で再構築すると、共著者の欠落、誤った発表会場、誤ったエントリ種別、プレプリントの年など、気付きにくく重大な誤りが生じます。プレプリントが後に採択されることもあるため、読んだ版がarXivにあっても、エントリが `@inproceedings` の場合があります。

ブロックをそのままコピーしてください。参考文献のスタイルで別のエントリ種別が必要な場合は、再入力せずエントリを変換し、著者一覧の元の順序を維持してください。

## 手法セクションに必要な情報

LibreYOLOの結果を再現可能にし、正しく帰属させるには、次の3点が必要です。

- `CITATION.cff` から引用したライブラリと、実行したバージョン。`libreyolo version` を実行すると、バージョンに加えて、使用中のPython、torch、CUDAのバージョンが表示されます。
- ファミリーのページにある「Citation」セクションから引用したアップストリームの研究。
- `LibreRFDETRm.pt` など、正確なチェックポイントのファイル名。ファミリー内でもサイズによって動作は異なります。また、複数のファミリーが同じ接頭辞の下で、異なるデータセットで学習したチェックポイントを公開しているため、ファミリー名だけでは実行したものを特定できません。

LibreYOLOが公開する多くの成果物では、帰属表示もライセンス条件です。Apache-2.0とCC BYファミリーはどちらも、再配布する重みに通知を添付することを求めています。これは論文の引用とは別の義務です。各チェックポイントに適用される条件は、[ライセンス](/docs/licensing)を参照してください。
