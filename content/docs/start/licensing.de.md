---
title: Lizenzierung
seo_title: 'LibreYOLO-Lizenzierung: Code und Gewichte'
description: >-
  Der eigene Code von LibreYOLO steht unter MIT. Eingebundener Upstream-Code und
  veröffentlichte Checkpoints besitzen eigene Lizenzen, von denen mehrere die
  kommerzielle Nutzung ausschließen.
lead: >-
  LibreYOLO enthält drei getrennt lizenzierte Bestandteile: eigenen Code, in
  eine Modellfamilie eingebundenen Upstream-Code und vortrainierte Checkpoints.
  Häufig gelten dafür unterschiedliche Lizenzen.
keywords:
  - libreyolo lizenz
  - mit computer vision bibliothek
  - nichtkommerzielle modellgewichte
  - modell-checkpoint lizenz
  - apache-2.0 objekterkennung
last_verified: 1.5.0
source_hash: 83536fea4dc4eaec
---

## Eigener Code von LibreYOLO

Die Bibliothek steht unter MIT. Dies umfasst die Python-API, die CLI, Trainer,
Validatoren und Exporter, die Datensatz-Loader sowie die Konvertierungsskripte
unter `weights/`. Du darfst sie in einem kommerziellen oder
Closed-Source-Produkt verwenden. Bewahre bei jeder weitergegebenen Kopie die
Copyright-Zeile und den Lizenztext auf. Weitere Pflichten bestehen nicht.

Die Genehmigung endet beim Code. Die Datei
[`LICENSE`](https://github.com/LibreYOLO/libreyolo/blob/release/LICENSE)
formuliert dies deutlich:

> Diese Lizenzen unterscheiden sich und sind nicht alle freizügig: Einige
> veröffentlichte Gewichte sind nicht kommerziell nutzbar oder anderweitig
> eingeschränkt. Diese MIT-Lizenz gilt nicht für sie. Mit der Wahl eines
> Modells wählst du auch seine Lizenz.

## Upstream-Code pro Familie

Die meisten Familien sind Portierungen veröffentlichter Forschung. Mehrere
binden Upstream-Quellcode direkt ein. Eine eingebundene Datei behält ihren
ursprünglichen Copyright-Header und ihre ursprüngliche Lizenz. MIT überschreibt
sie nicht, und LibreYOLO lizenziert die Arbeit anderer nicht neu. Apache-2.0
und BSD-3-Clause treten am häufigsten auf.

Apache-2.0 gilt für die DETR-Linie und einen großen Teil der
Transformer-Arbeit: DETR von Meta AI (FAIR), Deformable DETR von SenseTime,
LW-DETR von Baidu, OV-DEIM von Leilei Wang und Mitautoren, die von LibreYOLO
portierte SegFormer-Implementierung aus Hugging Face Transformers, PP-OCRv5
von den PaddlePaddle-Autoren, SwinIR vom Computer Vision Lab der ETH Zürich und
Depth Anything 3 von ByteDance Seed. Die Lizenz gilt außerdem für die von timm
von Ross Wightman und den timm-Mitwirkenden abgeleiteten Klassifikatoren,
darunter ResNet, DeiT, EfficientNetV2, MobileNetV4 und Swin. Deren Modulnamen
entsprechen timm, damit dessen ImageNet-Tensoren unverändert geladen werden
können.

BSD-3-Clause gilt für alle aus torchvision abgeleiteten Familien: Faster
R-CNN, Mask R-CNN, FCOS, RetinaNet, SSD300, AlexNet, VGG, FCN und DeepLabv3.

MIT gilt für eine kleinere Gruppe, darunter NAFNet von Megvii, CenterNet von
Xingyi Zhou und YOLOv7 in der Neuveröffentlichung seiner Autoren Kin-Yiu Wong
und Hao-Tang Tsui am MultimediaTechLab. Die Familien YOLOv1 bis YOLOv4 bilden
Architekturen des Darknet-Projekts von Joseph Redmon und bei YOLOv4 von Alexey
Bochkovskiy nach. Darknet ist gemeinfrei, weshalb daraus keine Pflichten
entstehen.

Ein gebündelter Teilbaum steht nicht unter einer Open-Source-Lizenz. Die
Familie DEIMv2 enthält DINOv3-Backbone-Code von Meta Platforms unter dem
DINOv3 License Agreement, einer benutzerdefinierten Nicht-OSI-Lizenz. Bei der
Weitergabe dieses Codes muss eine Kopie der Vereinbarung enthalten sein. Die
Vereinbarung verbietet die Nutzung für ITAR-pflichtige Aktivitäten, militärische
oder kriegerische Zwecke, die Nuklearindustrie, Spionage und
Waffenentwicklung. Diese Bedingungen gelten ausschließlich für diesen Teilbaum.

Zwei Dateien im Repository enthalten die vollständige Übersicht.
[`NOTICE`](https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE) führt
jeden gebündelten Drittanbieter-Teilbaum mit Pfad, Lizenzdatei und
Upstream-Quelle auf.
[`THIRD_PARTY_NOTICES.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/THIRD_PARTY_NOTICES.txt)
listet die Upstream-Projekte, von denen LibreYOLO abgeleitet ist, und gibt
jeden Lizenztext vollständig wieder.

## Gewichte pro Checkpoint

Das Paket selbst enthält keine vortrainierte Gewichtsdatei. Veröffentlichte
Checkpoints liegen auf Hugging Face in der
[LibreYOLO-Organisation](https://huggingface.co/LibreYOLO). Jedes Repository
enthält eine eigene `LICENSE` und Namensnennung entsprechend dem Projekt, aus
dem die Gewichte stammen.

Dieses Repository ist die maßgebliche Quelle für die Bedingungen. Nicht diese
Seite, nicht die Modellseite und nicht die Zusammenfassung im Quellbaum. Unter
[Checkpoints und Gewichte](/docs/weights) erfährst du, wie Dateien benannt sind
und von wo sie heruntergeladen werden.

Lizenzen unterscheiden sich zwischen Familien und auch zwischen Dateien
innerhalb einer Familie. Zwei Beispiele für den zweiten Fall:

- Die COCO-Checkpoints von YOLO9 stehen unter MIT.
  `LibreYOLO9P2s-visdrone.pt`, trainiert auf VisDrone2019-DET, steht unter CC
  BY-NC-SA 3.0 und darf nicht kommerziell genutzt werden.
- Die Erkennungs-Checkpoints von RF-DETR stehen unter Apache-2.0. Die
  Checkpoints für orientierte Boxen stehen unter CC BY 4.0, weil sie mit einem
  unter CC BY 4.0 veröffentlichten Datensatz von Roboflow Universe
  nachtrainiert wurden und die Gewichte dessen Namensnennungspflicht übernehmen.

Zwischen Familien ist die Spanne noch größer. Mehrere veröffentlichte
Checkpoints dürfen nicht in einem kommerziellen Produkt eingesetzt werden:

- SegFormer zeigt die Trennung der beiden Ebenen besonders deutlich. Die
  Implementierung ist eine Apache-2.0-Portierung des Codes von Hugging Face
  Transformers. Die veröffentlichten ADE20K-Checkpoints wurden aus der
  Veröffentlichung von NVIDIA unter der NVIDIA Source Code License
  konvertiert. Diese erlaubt die Weitergabe, beschränkt die Nutzung jedoch auf
  nichtkommerzielle Forschung oder Evaluierung und überträgt diese
  Einschränkung auf abgeleitete Werke. Diese Checkpoints fallen nicht unter die
  freizügigen Bedingungen von LibreYOLO.
- OV-DEIM-Checkpoints stehen nach Bestätigung des Upstream-Autors unter CC
  BY-NC 4.0. Jede Vorhersage lädt außerdem Apples MobileCLIP-B(LT)-Text-Tower,
  dessen Lizenz die Nutzung auf Forschung beschränkt. Diese Bedingung ist
  strenger als die eigene Lizenz des Checkpoints.
- Der Code von SenseNova-Vision steht unter Apache-2.0, seine Gewichte unter CC
  BY-NC 4.0. Der Loader zeigt vor jedem automatischen Download den Hinweis auf
  die nichtkommerzielle Nutzung an.

Einige Familien besitzen überhaupt keinen von LibreYOLO gehosteten Checkpoint.
Ihre Seiten weisen in der Zeile zu den Gewichten darauf hin. SAM 3 ist auf
Hugging Face unter Metas benutzerdefinierter SAM License zugangsbeschränkt und
wird direkt von Meta heruntergeladen. Release-Artefakte von MiDaS werden von
den offiziellen URLs abgerufen und per Hash überprüft, statt erneut gehostet zu
werden. Dome-DETR wird direkt im Upstream-Projekt verlinkt, weil seine
Modellkarte in den Metadaten keine Lizenz angibt, im Text zugleich Apache-2.0
beansprucht und die Nutzung auf akademische Forschung beschränkt. Diese Angaben
widersprechen sich. Die Architekturen TEED und DexiNed stehen unter MIT, aber
die veröffentlichten Checkpoints der Autoren wurden auf BIPED trainiert, dessen
Datensatzbedingungen die kommerzielle Nutzung ausschließen. LibreYOLO bündelt
sie daher weder noch lädt es sie automatisch herunter.

Mehrere torchvision-Checkpoints enthalten keine eigene Lizenzdatei. LibreYOLO
spiegelt sie unter der Lizenz des veröffentlichenden Projekts. Jede Modellkarte
weist darauf hin, dass diese Grundlage angenommen und nicht pro Checkpoint
ausdrücklich erteilt wird. Außerdem wiederholt sie die Warnung von torchvision,
dass Bedingungen für vortrainierte Modelle aus den Trainingsdaten entstehen
können.

## Bedingungen für ein bestimmtes Modell finden

Die Modellseite enthält im Kopfbereich eine Zeile **Lizenzen** in der Form
`Code X, Gewichte Y`, die auf den Lizenzierungsabschnitt der Seite verweist.
Dieser Abschnitt führt die ursprüngliche Arbeit und ihre Autoren, die
Upstream-Lizenz, die Upstream-Quelle, die LibreYOLO-Code-Lizenz, die Gewichte
und eine Interpretation der erlaubten Nutzungen auf. Die Checkpoint-Tabelle
derselben Seite besitzt eine Spalte **Gewichtslizenz** mit einer Zeile pro
veröffentlichter Datei. Gemischte Bedingungen innerhalb einer Familie werden
daher dateiweise dargestellt.

Alle Angaben werden aus denselben Daten gerendert, gegen die die Bibliothek
geprüft wird. Deshalb wiederholt diese Seite sie nicht als Tabelle. Eine von
Hand gepflegte Lizenzmatrix wäre bereits innerhalb eines Releases falsch, und
Fehler wären hier kostspielig.

Im Quellbaum entsprechen dem `NOTICE` für gebündelten Code,
`THIRD_PARTY_NOTICES.txt` für Upstream-Projekte und ihre Lizenztexte sowie
[`weights/LICENSE_NOTICE.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/weights/LICENSE_NOTICE.txt)
für eine familienweise Zusammenfassung der veröffentlichten Checkpoints.

Prüfe anschließend das Hugging-Face-Repository der konkreten Datei, die du
herunterladen möchtest. Es ist maßgeblich und kann sich ändern, ohne dass sich
eine Dokumentationsseite ändert.

## Kommerzielle Nutzung

Der Code ist selten das Problem. MIT, Apache-2.0 und BSD-3-Clause erlauben
kommerzielle und Closed-Source-Nutzung. Alle verlangen, dass Lizenztext und
Namensnennungshinweise mit weitergegebenen Kopien erhalten bleiben. Apache-2.0
gewährt zusätzlich eine Patentlizenz. Keine davon stellt Bedingungen an deinen
eigenen Anwendungscode.

Checkpoints sind häufig das Hindernis für Produkte. Ein nichtkommerzieller
Checkpoint bleibt unabhängig von der freizügigen Lizenz des umgebenden Codes
nichtkommerziell. Die Konvertierung der Datei ändert ihre anwendbaren
Bedingungen nicht, wie `weights/LICENSE_NOTICE.txt` ausdrücklich festhält. Ein
aus einem eingeschränkten Checkpoint erzeugtes ONNX- oder TensorRT-Artefakt
übernimmt die Einschränkung.

Wenn eine Lizenz ihre Einschränkung auf abgeleitete Werke überträgt, wie die
NVIDIA Source Code License, umgeht auch Fine-Tuning sie nicht. Du kannst
dagegen dieselbe Architektur von Grund auf neu mit Daten trainieren, zu deren
Nutzung du berechtigt bist. Der Code ist freizügig lizenziert. Ein selbst
trainiertes Modell gehört daher dir, und die Bedingungen des vortrainierten
Checkpoints gelten nie dafür. Die SegFormer-Seite erläutert dies für ihre
Gewichte. Lies für jede Familie, die du ausliefern möchtest, die Zeile zur
Interpretation.

Kläre die Lizenzfrage bei der Modellauswahl und nicht erst bei der Auslieferung.
Lies die Bedingungen der tatsächlich heruntergeladenen Datei, weil neben einem
freizügigen Checkpoint derselben Familie ein eingeschränkter liegen kann.

## Keine Rechtsberatung

Diese Seite beschreibt die beteiligten Lizenzen. Sie ist eine Beschreibung und
keine Rechtsberatung und begründet keine Gewährleistung. Wenn die Antwort
kommerziell relevant ist, lies die Lizenzen selbst und hole eigenen Rechtsrat
ein.

