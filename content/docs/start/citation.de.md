---
title: Zitieren
seo_title: LibreYOLO und die Upstream-Autoren zitieren
description: >-
  So zitierst du LibreYOLO in einer wissenschaftlichen Arbeit und die Autoren
  der verwendeten Modellfamilie. Beide Quellen gehören in denselben
  Methodenabschnitt.
lead: >-
  Eine vollständige Quellenangabe für LibreYOLO besteht aus zwei Teilen: der
  Bibliothek und der veröffentlichten Arbeit hinter der Modellfamilie, die das
  Ergebnis erzeugt hat.
keywords:
  - libreyolo zitieren
  - libreyolo bibtex
  - libreyolo citation cff
  - modell zitieren
  - computer vision quellenangabe
last_verified: 1.5.0
source_hash: 0f3f23e4e85e38be
---

## Zitieren von LibreYOLO

Das Repository veröffentlicht seine Zitiermetadaten als
[`CITATION.cff`](https://github.com/LibreYOLO/libreyolo/blob/release/CITATION.cff)
und nicht als BibTeX-Block. GitHub liest diese Datei und bietet auf der
Repository-Seite die Schaltfläche „Cite this repository“ an, die daraus APA-
und BibTeX-Angaben erzeugt. Übernimm den Eintrag von dort, anstatt ihn selbst
einzutippen.

Die vollständige Datei:

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

Die Datei enthält absichtlich weder eine Version noch ein Veröffentlichungsdatum.
[`RELEASING.md`](https://github.com/LibreYOLO/libreyolo/blob/release/RELEASING.md)
weist Maintainer an, `CITATION.cff` oder `.zenodo.json` bei einem Release nie
hochzusetzen, zu datieren oder umzubenennen. Dadurch verweisen alle Zitate auf
einen Datensatz, statt sich über mehrere Versionen zu verteilen. Nenne die
verwendete Version in deinem eigenen Text und lasse die Quellenangabe
unverändert.

## Zitieren der Modellfamilie

LibreYOLO ist eine Portierung. Wenn du `LibreRFDETRm.pt` ausführst, verwendest
du RF-DETR. Reviewer erwarten daher, dass die Entwickler von RF-DETR genannt
werden. Wenn du ausschließlich die Bibliothek zitierst, ordnest du ihre Arbeit
dem falschen Projekt zu.

Alle erforderlichen Angaben stehen auf der Seite der Familie. Die
Upstream-Zeile im Kopfbereich nennt die ursprüngliche Arbeit und die
dahinterstehende Organisation und verlinkt Paper sowie Quell-Repository. Der
weiter unten stehende Abschnitt zum Zitieren enthält den BibTeX-Eintrag.

Dieser BibTeX-Eintrag wird unverändert aus dem eigenen Zitierblock der Autoren
übernommen, normalerweise aus dem Zitierabschnitt der Upstream-README oder
einer `CITATION.cff`. Er wird zusammen mit einem Link zum ursprünglichen Block
angezeigt, damit du ihn mit der Quelle abgleichen kannst. Der Eintrag wird nie
aus Paper-Metadaten zusammengesetzt. Ein von Hand rekonstruierter Eintrag kann
unbemerkt folgenschwere Fehler enthalten: einen fehlenden Co-Autor, den
falschen Veranstaltungsort, einen falschen Eintragstyp oder das Jahr des
Preprints. Auch Preprints können später angenommen werden. Ein Eintrag kann
daher `@inproceedings` sein, obwohl die von dir gelesene Version auf arXiv lag.

Kopiere den Block unverändert. Wenn dein Bibliografiestil einen anderen
Eintragstyp verlangt, konvertiere den Eintrag, statt ihn neu einzutippen, und
behalte die ursprüngliche Reihenfolge der Autoren bei.

## Anforderungen an den Methodenabschnitt

Drei Angaben machen ein LibreYOLO-Ergebnis reproduzierbar und weisen die
Urheberschaft korrekt zu:

- Die anhand von `CITATION.cff` zitierte Bibliothek zusammen mit der verwendeten
  Version. `libreyolo version` gibt sie sowie die verwendeten Versionen von
  Python, torch und CUDA aus.
- Die Upstream-Arbeit, zitiert anhand des Zitierabschnitts auf der Seite der
  Familie.
- Der genaue Checkpoint-Dateiname, zum Beispiel `LibreRFDETRm.pt`. Größen
  innerhalb einer Familie verhalten sich unterschiedlich. Mehrere Familien
  veröffentlichen außerdem unter demselben Präfix Checkpoints, die auf
  verschiedenen Datensätzen trainiert wurden. Der Familienname allein
  identifiziert daher nicht das ausgeführte Modell.

Bei vielen von LibreYOLO veröffentlichten Inhalten ist die Namensnennung auch
eine Lizenzbedingung. Apache-2.0 und die CC-BY-Lizenzfamilie verlangen beide,
dass der Hinweis mit weitergegebenen Gewichten erhalten bleibt. Dies ist eine
von der Zitierung eines Papers getrennte Pflicht. Unter
[Lizenzierung](/docs/licensing) erfährst du, welche Bedingungen für welchen
Checkpoint gelten.

