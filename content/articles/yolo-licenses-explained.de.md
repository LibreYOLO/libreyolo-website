---
title: "YOLO-Lizenzen erklärt: alle Versionen von v1 bis YOLO26 (Leitfaden 2026)"
description: "Die vollständige Übersicht der YOLO-Lizenzen für 2026: YOLOv1 bis YOLOv13, YOLO26, YOLO-World und YOLOE, Original-Repos und permissive Neuimplementierungen, mit Links zu Paper und GitHub, und was du tatsächlich in einem kommerziellen Produkt ausliefern kannst."
date: 2026-07-11
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, gpl, apache, mit-license, yolov9, yolov10, yolo11, yolov12, yolov13, yolo26]
faq:
  - q: "Ist YOLOv9 kostenlos für die kommerzielle Nutzung?"
    a: "Das hängt davon ab, welches Repository du verwendest. Das Paper-Repository (WongKinYiu/yolov9) steht unter GPL-3.0. Dasselbe Labor veröffentlicht außerdem eine separate, MIT-lizenzierte Implementierung von YOLOv9 und YOLOv7 unter MultimediaTechLab/YOLO, geschrieben, nachdem kommerzielle Nutzer nach einer permissiven Option gefragt hatten. Sie ist eine Neuimplementierung, keine Umlizenzierung der GPL-Dateien. Die Lizenz des Codes ist eindeutig; für die vortrainierten Gewichte gibt es in keinem der beiden Repos eine eigene Lizenzangabe, behandle die Herkunft der Gewichte also als eigene Frage."
  - q: "Welche YOLO-Versionen sind kostenlos für die kommerzielle Closed-Source-Nutzung?"
    a: "Über mindestens ein permissives Repository: YOLOv1 bis YOLOv4 (Original-Darknet, gemeinfrei), YOLOv7 und YOLOv9 (die eigene MIT-Neuimplementierung des Labors), YOLOX und PP-YOLOE (Apache-2.0) sowie die MIT-Neuimplementierungen in LibreYOLO. YOLOv5, v6, v8, v10, YOLO11, v12, v13, YOLO26, YOLO-World und YOLOE haben Stand Juli 2026 keine permissive Implementierung. Prüf die aktuelle LICENSE-Datei selbst, bevor du dich darauf verlässt: Lizenzen ändern sich, und dies sind allgemeine Informationen, keine Rechtsberatung."
  - q: "Unter welcher Lizenz steht YOLOv10?"
    a: "AGPL-3.0. YOLOv10 ist ein akademisches Release der Tsinghua University, aber sein Code baut auf der Ultralytics-Codebasis auf und erbt daher AGPL-3.0. Eine permissive Neuimplementierung gibt es nicht."
  - q: "Unter welcher Lizenz stehen YOLOv12 und YOLOv13?"
    a: "Beide stehen unter AGPL-3.0, bestätigt durch ihre LICENSE-Dateien. Wie YOLOv10 sind es akademische Paper, deren Referenzcode auf dem Ultralytics-Repository aufbaut, also überträgt sich die AGPL, egal wer das Paper geschrieben hat. Seiten Dritter, die YOLOv13 als Apache-2.0 führen, liegen falsch."
  - q: "Wann hat Ultralytics YOLO auf AGPL-3.0 umgestellt?"
    a: "Am 14. April 2023, nicht 2022, wie vielfach wiederholt wird. Die LICENSE-Datei in ultralytics/yolov5 wurde überhaupt nur dreimal geändert, und der AGPL-Commit (34cf749, PR #11359) ist auf den 2023-04-14 datiert. Das letzte Release von 2022, YOLOv5 v7.0, wurde noch unter GPL-3.0 ausgeliefert. Das Repo ultralytics/ultralytics wurde am selben Tag umlizenziert. YOLOv8 ist daher im Januar 2023 unter GPL-3.0 gestartet: Die PyPI-Releases 8.0.0 bis 8.0.76 geben GPL-3.0 an, und 8.0.80 (16. April 2023) ist das erste, das AGPL-3.0 angibt."
  - q: "Ist YOLO26 kostenlos für die kommerzielle Nutzung?"
    a: "Nicht für Closed-Source-Produkte. YOLO26 steht unter AGPL-3.0, mit einer kostenpflichtigen Enterprise License als Alternative, zu denselben Bedingungen wie YOLOv8 und YOLO11."
  - q: "Ist das originale Darknet-YOLO wirklich gemeinfrei?"
    a: "Ja. In Joseph Redmons Darknet-LICENSE steht „Darknet is public domain. Do whatever you want with it.“ Der YOLOv4-Fork von AlexeyAB enthält denselben Public-Domain-Text. Der Haken: Die PyTorch-Portierungen, die tatsächlich genutzt werden, haben ihre eigenen Lizenzen, von Apache-2.0 über AGPL-3.0 bis hin zu gar keiner Lizenz."
  - q: "Darf ich die vortrainierten YOLO-NAS-Gewichte kommerziell nutzen?"
    a: "Nein. Der Code von super-gradients steht unter Apache-2.0, aber die offiziellen YOLO-NAS-Gewichte werden unter einer separaten Lizenz ausgeliefert. Dort heißt es, du „may not use the Software for any commercial use, including in connection with any models used in a production environment“ (darfst die Software für keinerlei kommerzielle Nutzung verwenden, auch nicht im Zusammenhang mit Modellen, die in einer Produktionsumgebung eingesetzt werden)."
  - q: "Erben die Gewichte eines neuronalen Netzes die Lizenz des Trainingscodes?"
    a: "Das ist ungeklärt. Ultralytics erklärt, dass AGPL-3.0 „the training code and the models produced by that training code“ abdeckt (den Trainingscode und die damit erzeugten Modelle), und als Rechteinhaber legen sie die Bedingungen für das fest, was sie veröffentlichen. Ob ein Gericht zustimmen würde, dass Gewichte, die du selbst trainierst, ein abgeleitetes Werk des Trainingscodes sind, wurde nie gerichtlich geprüft. Behandle veröffentlichte AGPL-Gewichte als belastet und sei vorsichtig damit, sie in eine permissive Neuimplementierung zu laden."
  - q: "Warum können verschiedene Implementierungen desselben YOLO unterschiedliche Lizenzen haben?"
    a: "Das Urheberrecht schützt den Ausdruck, nicht Ideen (17 U.S.C. 102(b)). Eine Architektur, die in einem Paper beschrieben ist, kann unabhängig implementiert und so lizenziert werden, wie ihr Autor es wählt. Was du nicht tun darfst: GPL- oder AGPL-Quelldateien kopieren und neu lizenzieren. Beachte, dass dies nur ein urheberrechtliches Argument ist: Eine unabhängige Implementierung ist keine Verteidigung gegen Patente."
---

Jedes Jahr gibt es mehr YOLOs, und jedes Jahr wird die Lizenzfrage neu gestellt, meist fünf Minuten vor einer Produktentscheidung. Verwirrend ist, dass „YOLO“ kein einzelnes Projekt ist. Es ist ein Markenname, den sich gut zwanzig Modellfamilien verschiedener Autoren teilen, und hier ist das Detail, das den meisten Lizenzratgebern entgeht: **Die Lizenz gehört zu einem Repository, nicht zu einem Modell.** Dieselbe YOLO-Version kann gleichzeitig als GPL-Repo und als MIT-Repo existieren, von denselben Autoren. Bei YOLOv9 ist das so, und es verändert die ganze Entscheidung.

Dieser Leitfaden kartiert die Landschaft Repository für Repository, mit Links zu jedem Paper und jeder Codebasis, Stand Juli 2026. Jede Lizenz unten wurde anhand der tatsächlichen LICENSE-Datei im tatsächlichen Repository geprüft, denn erstaunlich viel von dem, was über YOLO-Lizenzen geschrieben wird (auch in den Ratgebern, die dafür am höchsten ranken), ist falsch.

Wenn du nur die Frage zur AGPL von Ultralytics beantwortet haben willst: Die behandeln wir ausführlich in [YOLO kostenlos kommerziell nutzen?](/articles/yolo-commercial-license). Dieser Artikel ist die vollständige Karte.

**Zwei Offenlegungen vorab.** Erstens pflegen wir LibreYOLO, eine MIT-lizenzierte Bibliothek, die mit mehreren der unten genannten Projekte konkurriert, und im letzten Abschnitt dieses Artikels geht es um sie. Das ist ein Grund, unsere Arbeit zu überprüfen, nicht, sie ungeprüft zu glauben, und deshalb verlinkt jede Lizenzaussage hier auf die Primärquelle. Zweitens enthält dieser Artikel allgemeine Informationen über veröffentlichte Lizenztexte und öffentliche Aussagen, Stand des oben genannten Datums. Er ist keine Rechtsberatung und kein Ersatz für einen Anwalt in deiner Rechtsordnung. Lizenzen und die Positionen der Maintainer ändern sich. Lies die aktuelle LICENSE-Datei jedes Repositorys, bevor du eine Entscheidung über die Auslieferung triffst.

## Die Lizenztabelle

„Permissiver Weg“ bedeutet: ein Repository, das dieses Modell implementiert und das du in einem kommerziellen Closed-Source-Produkt nutzen kannst, ohne deine Anwendung als Open Source zu veröffentlichen und ohne für eine Lizenz zu zahlen.

| Modell | Jahr | Originalcode | Permissiver Weg | Paper |
| --- | --- | --- | --- | --- |
| YOLOv1 | 2015 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (gemeinfrei) | Das Original selbst | [arXiv](https://arxiv.org/abs/1506.02640) |
| YOLOv2 | 2016 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (gemeinfrei) | Das Original selbst | [arXiv](https://arxiv.org/abs/1612.08242) |
| YOLOv3 | 2018 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (gemeinfrei) | Das Original; Vorsicht bei der AGPL-PyTorch-Portierung | [arXiv](https://arxiv.org/abs/1804.02767) |
| YOLOv4 | 2020 | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) (gemeinfrei) | Das Original; [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) (Apache-2.0) | [arXiv](https://arxiv.org/abs/2004.10934) |
| YOLOv5 | 2020 | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) (AGPL-3.0) | Keiner | kein Paper |
| YOLOv6 | 2022 | [meituan/YOLOv6](https://github.com/meituan/YOLOv6) (GPL-3.0) | Keiner | [arXiv](https://arxiv.org/abs/2209.02976) |
| YOLOv7 | 2022 | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, dasselbe Labor) | [arXiv](https://arxiv.org/abs/2207.02696) |
| YOLOv8 | 2023 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0; beim Start GPL-3.0, siehe unten) | Keiner (die Keras-Portierung wurde eingestellt, siehe unten) | kein Paper |
| YOLOv9 | 2024 | [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, dasselbe Labor) | [arXiv](https://arxiv.org/abs/2402.13616) |
| YOLOv10 | 2024 | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) (AGPL-3.0) | Keiner | [arXiv](https://arxiv.org/abs/2405.14458) |
| YOLO11 | 2024 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Keiner | kein Paper |
| YOLOv12 | 2025 | [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12) (AGPL-3.0) | Keiner | [arXiv](https://arxiv.org/abs/2502.12524) |
| YOLOv13 | 2025 | [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13) (AGPL-3.0) | Keiner | [arXiv](https://arxiv.org/abs/2506.17733) |
| YOLO26 | 2026 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Keiner | [arXiv](https://arxiv.org/abs/2606.03748) |
| YOLOX | 2021 | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) | Das Original selbst | [arXiv](https://arxiv.org/abs/2107.08430) |
| PP-YOLOE | 2022 | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) (Apache-2.0) | Das Original, aus PaddleDetection (nicht PaddleYOLO) | [arXiv](https://arxiv.org/abs/2203.16250) |
| YOLO-NAS | 2023 | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) (Code unter Apache-2.0, nicht kommerzielle Gewichte) | Code ja, offizielle Gewichte nein | kein Paper |
| YOLO-World | 2024 | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) (GPL-3.0) | Keiner | [arXiv](https://arxiv.org/abs/2401.17270) |
| YOLOE | 2025 | [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe) (AGPL-3.0) | Keiner | [arXiv](https://arxiv.org/abs/2503.07465) |
| MMYOLO | 2022 | [open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo) (GPL-3.0, seit 2024 zum Stillstand gekommen) | Keiner | kein Paper |

Lies die Tabelle Spalte für Spalte, und eines wird offensichtlich: Die Versionsnummer sagt dir nichts über die Lizenz. Das Repository schon.

## Gleiches Modell, zwei Lizenzen: der Fall YOLOv9

YOLOv9 ist der klarste Beweis dafür, dass „Unter welcher Lizenz steht YOLOvN?“ die falsche Frage ist, und die Chronologie lohnt eine genaue Darstellung, gerade weil es das einzige Mal in der Geschichte von YOLO ist, dass Druck aus der Community tatsächlich ein Ergebnis verändert hat.

- **18. Februar 2024:** [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) geht zusammen mit dem [Paper](https://arxiv.org/abs/2402.13616) online.
- **22. Februar 2024:** Ein Nutzer eröffnet Issue #10 und fragt, wie die Lizenz lautet. Chien-Yao Wang (WongKinYiu) antwortet „I think it should be GPL3“ und fügt vier Tage später die GPL-3.0-Datei hinzu.
- **26. Februar 2024:** Ein zweiter Nutzer eröffnet [Issue #82, „An Apache/MIT rewrite“](https://github.com/WongKinYiu/yolov9/issues/82). In den folgenden zweieinhalb Wochen schließt sich ein Dutzend kommerzieller Nutzer an.
- **14. März 2024:** Wang schreibt in diesem Thread: *„Okay I create a new repo for mit rewrite... I think I can handle most of implementation of architectures and loss functions. And need someone to give great help about dataloader and ddp training.“*

Dieses Repo, zunächst `yolov9mit` genannt, ist heute **[MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)**: MIT-lizenziert, Copyright „Kin-Yiu, Wong and Hao-Tang, Tsui“, und es beschreibt sich selbst als *„the official implementation of YOLOv7 and YOLOv9, YOLO-RD.“* Den Großteil der Neuimplementierung haben nicht die Freiwilligen aus dem Thread geleistet, sondern Hao-Tang Tsui, Wangs Laborkollege, der rund 90 Prozent der Commits geschrieben hat.

YOLOv9 ist also gleichzeitig „nicht auslieferbar“ und „auslieferbar“, je nachdem, welches Repo du klonst. Gleiche Architektur, gleiches Labor, zwei Lizenzen.

**Bevor du ein Produkt darauf setzt: drei Einschränkungen, die dir das Repo nicht auf der Startseite verrät.**

1. **Es reift noch.** Ein [offenes Issue](https://github.com/MultimediaTechLab/YOLO/issues/231) vom Februar 2026 berichtet, dass die ausgelieferten Checkpoints die COCO-Werte des Papers nicht reproduzieren und merklich mehr Parameter haben, als das Paper angibt. Das Training für Segmentierung und Keypoints ist [noch nicht implementiert](https://github.com/MultimediaTechLab/YOLO/issues/232). Seit dem v1.0-Tag im Dezember 2025 wurde nichts mehr in main gemergt.
2. **Die Gewichte haben keine eigene Lizenzangabe.** Die MIT-LICENSE des Repos deckt nach gängiger Praxis auch seine Release-Assets ab, und die technischen Indizien sprechen für Checkpoints, die dieses Projekt selbst trainiert und nicht aus dem GPL-Repo kopiert hat: Die Dateien unterscheiden sich in Größe und Parameterzahl von den Checkpoints des GPL-Repos. Aber kein Dokument erklärt, dass die .pt-Dateien unter MIT stehen, und eine [Anfrage aus der Community nach einem formalen Audit auf AGPL-Kontamination](https://github.com/MultimediaTechLab/YOLO/issues/51) ist weiterhin offen. Das ist eine Lücke in den Unterlagen: Niemand hat „nachweislich sauber“ behauptet, und niemand hat ein Problem behauptet.
3. **Es ist eine andere Codebasis**, kein Drop-in-Ersatz für die Skripte des GPL-Repos.

Nichts davon macht es unbrauchbar. Es macht es zu einem Projekt, das du evaluieren musst, nicht zu einem Häkchen, das du einfach setzt.

## Ära 1: die gemeinfreien Jahre (YOLOv1 bis YOLOv4)

Joseph Redmon veröffentlichte [YOLOv1](https://arxiv.org/abs/1506.02640), [YOLOv2](https://arxiv.org/abs/1612.08242) (erschienen als YOLO9000-Paper) und [YOLOv3](https://arxiv.org/abs/1804.02767) in seinem Darknet-Framework, und die [LICENSE-Datei](https://github.com/pjreddie/darknet/blob/master/LICENSE) ist berühmt dafür, dass sie drei Zeilen lang ist:

> 0. Darknet is public domain.
> 1. Do whatever you want with it.
> 2. Stop emailing me about it!

Das ist eine echte Public-Domain-Widmung. [YOLOv4](https://arxiv.org/abs/2004.10934), gepflegt von Alexey Bochkovskiy in einem [Darknet-Fork](https://github.com/AlexeyAB/darknet), enthält *dieselbe* Datei, nicht die Unlicense, die ihm mehrere Lizenzratgeber zuschreiben. GitHubs eigener Klassifikator kapituliert davor und meldet „Other“. Das solltest du wissen, wenn dein Compliance-Tooling dieses Feld ausliest: Ein Scanner markiert die Lizenz als nicht identifiziert statt als saubere permissive Lizenz, obwohl der Text kaum permissiver sein könnte.

Auch Darknet selbst ist nicht ganz tot. Redmons Repo ruht seit 2022, aber die README von AlexeyAB verweist inzwischen auf [hank-ai/darknet](https://github.com/hank-ai/darknet) als empfohlenen, aktiv gepflegten C/C++-Nachfolger.

**Die Falle sind die Portierungen.** Kaum jemand trainiert 2026 noch in Darknet; die Leute greifen zu einer PyTorch-Neuimplementierung, und die hat ihre eigene Lizenz:

- [ultralytics/yolov3](https://github.com/ultralytics/yolov3), die populärste YOLOv3-Portierung, steht unter **AGPL-3.0**. Eine gemeinfreie Architektur, weiterverbreitet unter der strengsten gängigen Lizenz in diesem Bereich. Welche Lizenz du bekommst, hängt ganz davon ab, wessen Portierung du per pip installiert hast.
- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) steht unter **GPL-3.0**.
- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) steht unter **Apache-2.0**, ein permissiver Weg für v4 bleibt also auch in PyTorch erhalten.
- [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4), vom Co-Autor von YOLOv4, hat **überhaupt keine LICENSE-Datei**. Das ist schlimmer als Copyleft: Ohne Lizenz werden keine Rechte eingeräumt, und standardmäßig gilt „alle Rechte vorbehalten“.

Prüf die Lizenz des Repositorys, das du geklont hast, nicht die Versionsnummer auf dem Paper.

## Ära 2: GPL-3.0 (YOLOv6, v7, v9)

[YOLOv6](https://github.com/meituan/YOLOv6) (Meituan), [YOLOv7](https://github.com/WongKinYiu/yolov7) und [YOLOv9](https://github.com/WongKinYiu/yolov9) erschienen unter GPL-3.0.

GPL-3.0 ist Copyleft: Wenn du eine modifizierte Version verbreitest oder GPL-Code mit deinem Code zu einem einzigen Programm kombinierst, musst du den vollständigen korrespondierenden Quellcode dieses kombinierten Werks unter GPL-3.0 weitergeben. Zwei Grenzen sind wichtig. **Bloße Aggregation** bedeutet, dass separate, unabhängige Programme, die lediglich zusammen ausgeliefert werden, nicht automatisch kombiniert sind. Und der Auslöser ist die **Verbreitung**: Anders als die AGPL erlegt die einfache GPL keine Offenlegungspflicht auf, wenn du das kombinierte Werk nie auslieferst. Deshalb halten manche Unternehmen ein GPL-Modell strikt hinter ihrer eigenen API.

Dieser Weg ist schmaler, als er klingt, und er scheitert auf drei Arten, mit denen selten jemand plant. Sobald das Modell auf einem Kundengerät oder in einer On-Premises-Installation landet, verbreitest du es. „Nur intern“ stimmt nicht mehr, sobald eine Kopie deine juristische Person verlässt: Einen Build an einen Auftragnehmer, ein ausgelagertes Entwicklungsteam oder ein rechtlich eigenständiges verbundenes Unternehmen zu übergeben, ist Verbreitung, auch wenn das Kopieren innerhalb eines Unternehmens es nicht ist. Und der Schutz hält nur, wenn *alles*, was über diese API erreichbar ist, unter GPL oder einer permissiveren Lizenz steht, denn eine einzige AGPL-Komponente irgendwo im bereitgestellten Werk zieht das Ganze in den Geltungsbereich von Abschnitt 13, unabhängig davon, was die GPL-Datei selbst sagt.

**YOLOv7 und YOLOv9 haben den oben beschriebenen MIT-Notausgang.** YOLOv6 hat ihn nicht, und der stärkste Beleg dafür kommt von Baidu: PaddleDetection steht unter Apache-2.0, und seine Maintainer haben YOLOv5, YOLOv6, YOLOv7 und YOLOv8 bewusst in ein separates GPL-3.0-Repo abgeschottet, [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO), mit dem Hinweis, dass der Code dieser Modelle „will not be merged into PaddleDetection“ (nicht in PaddleDetection übernommen wird). Wenn ein Unternehmen von der Größe Baidus eine Mauer baut, um YOLOv6 aus seinem permissiven Framework herauszuhalten, sagt dir das, was die Lizenz wert ist.

## Ära 3: AGPL-3.0 (YOLOv5, v8, v10, 11, v12, v13, 26, YOLOE)

Ultralytics ist am **14. April 2023** auf AGPL-3.0 umgestiegen, nicht 2022, wie sehr häufig wiederholt wird (auch von Ratgebern, die für diese Frage auf der ersten Seite ranken). Alles seitdem ([YOLOv8](https://github.com/ultralytics/ultralytics), YOLO11 und [YOLO26](https://arxiv.org/abs/2606.03748)) steht unter AGPL-3.0, mit einer kostenpflichtigen Enterprise License als Alternative. Laut der [Lizenzseite](https://ultralytics.com/license) deckt die Enterprise-Stufe „YOLO26, earlier YOLO versions, and any future YOLO models“ ab (YOLO26, frühere YOLO-Versionen und alle künftigen YOLO-Modelle), die Politik ist also bewusst gewählt und auf die Zukunft ausgerichtet. Preise sind nicht öffentlich.

Dieses Datum lässt sich überprüfen, und das lohnt sich, denn die populäre Version dieser Geschichte ist auf eine Art falsch, die eine Rolle spielt:

- Die `LICENSE`-Datei in `ultralytics/yolov5` wurde in ihrer Geschichte genau dreimal angefasst. Das dritte Mal ist Commit [`34cf749`](https://github.com/ultralytics/yolov5/commit/34cf749958d2dd3ed1205f6bb07e0f20f6e2372d), „Update LICENSE to AGPL-3.0 (#11359)“, datiert auf den **2023-04-14**, der in 101 Dateien `GNU GENERAL PUBLIC LICENSE` durch `GNU AFFERO GENERAL PUBLIC LICENSE` ersetzt.
- Das letzte YOLOv5-Release von 2022, **v7.0 (22. November 2022), wurde noch unter GPL-3.0 ausgeliefert**, ebenso wie davor v6.2, v6.1 und v6.0. Dasselbe gilt für die Tausenden Forks, die vor April 2023 aufgehört haben zu synchronisieren: Nimm irgendeinen davon, und GitHub meldet bis heute GPL-3.0.
- Am selben Tag, 41 Minuten später, bekam `ultralytics/ultralytics` [dieselbe Behandlung](https://github.com/ultralytics/ultralytics/commit/2c6fc0a4443b9cf805ef17b1cfdd71a98693b4d4) (PR #2031).

Daraus ergibt sich die Tatsache, die fast niemand kennt: **YOLOv8 ist nicht unter AGPL gestartet.** Es erschien im Januar 2023 unter **GPL-3.0** und wurde drei Monate später umlizenziert. Der PyPI-Eintrag ist eindeutig: `ultralytics` 8.0.0 (10. Januar 2023) bis 8.0.76 (13. April 2023) geben alle `GPL-3.0` an. Version 8.0.80, hochgeladen am 16. April 2023, ist die erste, die `AGPL-3.0` angibt.

Das ist kein Schlupfloch. Eine erteilte Lizenz wird nicht rückwirkend widerrufen, daher bleiben diese alten Versionen zu den Bedingungen verfügbar, unter denen sie veröffentlicht wurden. Aber sie sind drei Jahre veraltet, ungepflegt und ungepatcht, GPL-3.0 ist weiterhin Copyleft (du hast eine Netzwerkklausel gegen eine Verbreitungsklausel getauscht, nicht das Copyleft hinter dir gelassen), und die kommerzielle Position von Ultralytics reicht ohnehin weiter als der Lizenztext. Die nützliche Erkenntnis ist enger und zugleich allgemeiner: **Eine Lizenz hängt an der Version, die du erhalten hast, nicht am Namen des Projekts.** Pinne deine Abhängigkeiten und halte fest, was die Lizenz an dem Tag besagte, an dem du die Kopie bezogen hast, denn das Projekt kann sie morgen ändern, und deine Compliance hängt davon ab, welche Version du tatsächlich bekommen hast.

AGPL-3.0 ist GPL plus Abschnitt 13, die Netzwerkklausel: *Wenn du das Programm modifizierst*, muss Nutzern, die über ein Netzwerk mit deiner modifizierten Version interagieren, deren Quellcode angeboten werden. Das schließt den SaaS-Weg, den die einfache GPL offenlässt.

Beachte die Einschränkung, denn die meisten Artikel lassen sie weg. Abschnitt 0 der AGPL definiert „modify“ als das Kopieren aus dem Werk oder dessen Anpassung *in einer Weise, die eine urheberrechtliche Erlaubnis erfordert*. Unmodifizierten Trainingscode auf deinem eigenen Datensatz laufen zu lassen, heißt, das Programm auszuführen, nicht seinen Quellcode anzupassen, genauso wie das Kompilieren deines Codes mit einem unmodifizierten GPL-Compiler den Compiler nicht modifiziert. Training allein ist also nicht automatisch der Auslöser, und wer dir das Gegenteil erzählt, hat die Definition übersprungen.

Was dich dagegen in den Geltungsbereich der Lizenz bringt, ist das Kombinieren oder Einbetten des Pakets in deine eigene Codebasis, sodass beide als ein Programm ausgeliefert oder bereitgestellt werden. Die FSF selbst liest die Lizenz so, dass das Linken eines GPL- oder AGPL-Werks mit anderen Modulen ein kombiniertes Werk ergibt; ob ein Gericht das bis auf einen Python-Import ausdehnen würde, wurde nie gerichtlich geprüft. Beachte außerdem, dass die eigenen [kommerziellen Bedingungen](https://ultralytics.com/license) von Ultralytics eine Enterprise License für SaaS-Plattformen, APIs und Cloud-Systeme verlangen, die YOLO im Hintergrund nutzen, die Position des Anbieters reicht also weiter als der reine Lizenztext. Betrachte die Integration in ein Produkt, das du bereitstellst oder auslieferst, als das Risiko; geh nicht davon aus, dass Training für sich allein das Risiko ist.

Speziell bei YOLO26 geraten die Daten überall durcheinander, deshalb: Es wurde auf der YOLO Vision **im September 2025 als Vorschau gezeigt**, **im Januar 2026 tatsächlich veröffentlicht** (das Release `ultralytics` 8.4.0, in dessen Release Notes „Ultralytics YOLO26 has arrived“ steht), und **das Paper erschien im Juni 2026**. Es liegt im Haupt-Repo `ultralytics/ultralytics`; `ultralytics/yolo26` ist nur eine Landingpage.

**Der Teil, der Leute überrascht:** Auch die akademischen YOLOs der letzten zwei Jahre stehen unter AGPL, und zwar nicht, weil Ultralytics sie geschrieben hätte.

- [YOLOv10](https://arxiv.org/abs/2405.14458) stammt von der Tsinghua University. In der README heißt es *„The code base is built with ultralytics.“*
- [YOLOv12](https://arxiv.org/abs/2502.12524): *„The code is based on ultralytics.“*
- [YOLOv13](https://arxiv.org/abs/2506.17733): *„The code is based on Ultralytics.“*
- [YOLOE](https://arxiv.org/abs/2503.07465), das Open-Vocabulary-Modell „see anything“ aus der YOLOv10-Gruppe, steht ebenfalls unter AGPL-3.0 und baut auf Ultralytics auf.

Copyleft wird vererbt: Ein Derivat von AGPL-Code muss unter AGPL weitergegeben werden, sobald also einer dieser Forks ausgeliefert oder bereitgestellt wird, kommt die Lizenz mit. (Modifizierst du ihn und behältst ihn strikt für dich, entsteht keine Pflicht, weshalb die Nutzung in der Forschung nicht betroffen ist.) Die Forschung ist unabhängig; die Lizenz ist es nicht. Seit 2024 ist es zum Standard-Workflow geworden, „das nächste YOLO“ als Ultralytics-Derivat zu veröffentlichen, was bedeutet, dass sich die AGPL jetzt automatisch die Versionsnummern hinauf fortpflanzt.

Wenn du eine Seite siehst, die behauptet, YOLOv13 stehe unter Apache-2.0, liegt sie falsch. Die LICENSE-Datei, die GitHub-API und die Model Card sagen alle AGPL-3.0.

### Der permissive YOLOv8-Weg, den es nicht mehr gibt

Viele noch online verfügbare Ratgeber, einschließlich eines früheren Entwurfs dieses Artikels, verweisen auf das YOLOv8 von KerasCV unter Apache-2.0 als sauberen Ausweg aus dem AGPL-YOLOv8. **Darauf kannst du dich heute nicht verlassen.** Das ist die öffentliche Aktenlage:

- In der KerasCV-Diskussion [Clarifications regarding YOLOv8 licensing](https://github.com/keras-team/keras-cv/discussions/2032) schrieb ein Contributor, dass „many parts are derived form ultralytics“. Kein Maintainer hat im Thread geantwortet, daher wurde die Frage, wie unabhängig die Implementierung war, öffentlich nie in die eine oder andere Richtung geklärt.
- Das KerasHub-Issue [add YOLOV8](https://github.com/keras-team/keras-hub/issues/1760) wurde im Juli 2025 von einem Keras-Maintainer geschlossen, mit den Worten: „Hey all!! Because of license issues, we have decided to drop this.“ Zwei Personen fragten daraufhin, was das für bestehende kommerzielle Nutzer der KerasCV-Version bedeute. Keine von beiden bekam eine Antwort.
- KerasCV ist inzwischen archiviert, und KerasHub, sein gepflegter Nachfolger, **liefert YOLOv8 nicht aus**.

Wir wissen nicht, ob diese beiden Dinge zusammenhängen, und wir behaupten nicht, dass irgendetwas Unrechtmäßiges passiert ist. Eine Grundlage, auf der man ein Produkt aufbaut, ist das nicht.

## Die permissiven Ausreißer: YOLOX, PP-YOLOE, YOLO-NAS

Nicht jedes YOLO ist auf den Copyleft-Zug aufgesprungen.

- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** (Megvii, 2021) steht ohne Ausnahmen unter Apache-2.0, Code und Gewichte. Es ist nach wie vor das sauberste klassische YOLO für die kommerzielle Nutzung, auch wenn das Repository seit Mitte 2025 keine Commits mehr hatte und eine [Frage zum Bündeln der vortrainierten Gewichte in einer kommerziellen App](https://github.com/Megvii-BaseDetection/YOLOX/issues/1865) seit März 2026 ohne Antwort offen ist. Der Lizenztext ist eindeutig. Es ist nur gerade niemand da, der Fragen dazu beantwortet. Wir haben darüber geschrieben, wie du [YOLOX mit LibreYOLO ausführst](/articles/yolox-with-libreyolo).
- **[PP-YOLOE](https://arxiv.org/abs/2203.16250)** (Baidu) steht **innerhalb von PaddleDetection** unter Apache-2.0. Bezieh es von dort, nicht aus PaddleYOLO, das ein nahezu identisches Konfigurationsverzeichnis enthält und unter GPL-3.0 steht. Gleiches Modell, gleiches Unternehmen, zwei Repos, zwei Lizenzen. Es ist die YOLOv9-Situation mit umgekehrten Vorzeichen.
- **[YOLO-NAS](https://github.com/Deci-AI/super-gradients)** (Deci, 2023) ist das Modell, bei dem Leute in die Falle tappen. Der Code steht unter Apache-2.0; die offiziellen Gewichte nicht. Ihre [separate Lizenz](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md) sagt, du *„may not use the Software for any commercial use, including in connection with any models used in a production environment.“* (darfst die Software für keinerlei kommerzielle Nutzung verwenden, auch nicht im Zusammenhang mit Modellen, die in einer Produktionsumgebung eingesetzt werden). Seit NVIDIA Deci 2024 übernommen hat, haben wir keine neue Feature-Arbeit im Repository gefunden, und die ursprünglichen URLs der Gewichte funktionieren nicht mehr (die Checkpoints liegen jetzt bei einem anderen Host). Ein manchmal vorgeschlagener Workaround, nämlich dass du Apache-sauber bleibst, wenn du die Architektur von Grund auf neu trainierst, ist plausibel, wurde aber weder von Deci noch von NVIDIA bestätigt, und er passt schlecht zu einem Lizenztext, der „any components comprising the model“ (alle Komponenten, aus denen das Modell besteht) erfasst. Mehr zu YOLO-NAS [hier](/articles/yolo-nas-with-libreyolo).
- **[MMYOLO](https://github.com/open-mmlab/mmyolo)** wird in älteren Ratgebern manchmal als Apache-2.0 geführt. Das stimmt nicht: MMYOLO steht unter GPL-3.0 (sein Geschwisterprojekt MMDetection ist das Apache-Projekt), und es hatte seit Juli 2024 keine Commits mehr.
- **[YOLO-World](https://github.com/AILab-CVC/YOLO-World)** (Tencent) steht unter **GPL-3.0**, nicht AGPL und nicht permissiv. Das ist erwähnenswert, weil man bei Open-Vocabulary-YOLOs oft annimmt, sie seien permissiv, allein durch ihre Nähe zum CLIP-artigen Ökosystem, aus dem sie Anleihen nehmen.

Diese Fälle decken die drei häufigsten Lizenzfehler in diesem Bereich ab: anzunehmen, dass die Gewichte der Lizenz des Codes folgen, anzunehmen, dass alles von einer Organisation dieselbe Lizenz hat, und die Lizenz eines Modells aus seinem Namen abzuleiten.

## Lizenzen decken Code ab, keine Ideen (aber Vorsicht bei Patenten)

Ein Prinzip ist die rechtliche Grundlage für jeden permissiven Weg oben: **Das Urheberrecht schützt den Ausdruck, nicht Ideen.** Das ist kein Slogan, sondern gefestigtes Recht auf beiden Seiten des Atlantiks.

In den USA ist das [17 U.S.C. 102(b)](https://www.law.cornell.edu/uscode/text/17/102), das jede „idea, procedure, process, system, method of operation, concept, principle, or discovery“ (Idee, Verfahren, Prozess, System, Funktionsweise, Konzept, Prinzip oder Entdeckung) vom urheberrechtlichen Schutz ausnimmt, zurückgehend auf *Baker v. Selden* (1879). In der EU, deren Recht für uns gilt und vermutlich für einen guten Teil der Leser, sagt Artikel 1 Absatz 2 der [Software-Richtlinie (2009/24/EG)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32009L0024), dass „ideas and principles which underlie any element of a computer program“ (Ideen und Grundsätze, die irgendeinem Element eines Computerprogramms zugrunde liegen) nicht geschützt sind, und der Gerichtshof der EU hat in *SAS Institute v World Programming* ([C-406/10](https://curia.europa.eu/juris/liste.jsf?num=C-406/10), 2012) bestätigt, dass die Funktionalität eines Programms, seine Programmiersprache und seine Datenformate selbst kein geschützter Ausdruck sind. Geschützt ist nur der Code.

Eine Architektur, die in einem arXiv-Paper beschrieben ist, ist eine veröffentlichte Idee. Jeder darf sie implementieren. Was du nicht tun darfst: die GPL/AGPL-Quelldateien eines anderen kopieren oder anpassen und das Ergebnis neu lizenzieren. Eine unabhängige Implementierung, geschrieben von Leuten, die mit dem Paper statt mit dem Quellcode arbeiten, ist ein neues Werk, dessen Autor die Lizenz wählt.

Zwei ehrliche Einschränkungen, die die meisten Anbieter weglassen:

- **„Unabhängig“ muss auch stimmen.** Die belastbare Variante davon ist ein Clean-Room-Prozess: Die Leute, die den Code schreiben, arbeiten mit dem Paper und einer Spezifikation, nicht mit dem Copyleft-Quellcode. Das GPL-Repo genau zu lesen und es dann „neu zu schreiben“, ist nicht dasselbe, und der Unterschied zählt, falls jemals jemand genauer hinsieht.
- **Das ist ein urheberrechtliches Argument, kein patentrechtliches.** Unabhängige Erfindung ist *keine* Verteidigung gegen ein Patent. Eine Clean-Room-Neuimplementierung klärt die Urheberrechtsfrage und lässt die Patentfrage genau dort, wo sie war. Derzeit schwebt kein breit geltend gemachtes Patent über den YOLO-Architekturen, aber „bisher hat niemand geklagt“ ist nicht dasselbe wie „es gibt nichts, worüber man klagen könnte“.

## Die Gewichte sind eine zweite Lizenz

Der teuerste Fehler in diesem ganzen Bereich ist, den Code zu prüfen und den Checkpoint zu vergessen. Drei Ebenen tragen Lizenzen, und sie können sich alle unterscheiden:

**1. Der Code.** Alles oben Gesagte.

**2. Die Gewichte.** Manchmal explizit, wie bei YOLO-NAS. Manchmal behauptet: Die [Lizenzseite](https://ultralytics.com/license) von Ultralytics erklärt, dass AGPL-3.0 „covers the training code and the models produced by that training code“ (den Trainingscode und die damit erzeugten Modelle abdeckt), und dehnt das auf Modelle aus, die du selbst mit ihrem Code trainierst. Als Rechteinhaber dessen, was sie veröffentlichen, können sie die Bedingungen für ihre eigenen Checkpoints festlegen. Ob Gewichte, die *du* erzeugst, rechtlich ein abgeleitetes Werk des Trainingscodes sind, ist eine wirklich offene Frage: Es gibt keine Rechtsprechung dazu, und die Behauptung steht im Spannungsverhältnis zur eigenen allgemeinen Regel der FSF, dass die Ausgabe eines Programms nicht automatisch unter das Urheberrecht des Programms fällt. Ungeklärt ist aber nicht dasselbe wie sicher. Behandle veröffentlichte AGPL-Gewichte in der Praxis als belastet.

Die heikle Stelle für alle, die dem Rat dieses Artikels folgen: **Die GPL-Repos von YOLOv7 und YOLOv9 sagen überhaupt nichts über ihre Gewichte.** Ihre Checkpoints sind Release-Assets in einem GPL-3.0-Repository ohne eigene Rechteeinräumung. Wenn du diese .pt-Dateien in eine MIT-Neuimplementierung lädst, bekommst du sauberen Code und einen Checkpoint, dessen Status niemand geklärt hat. Wenn die Lizenz der Grund für deinen Wechsel ist, verwende Gewichte, die das permissive Projekt selbst trainiert hat, oder trainiere deine eigenen.

**3. Die Trainingsdaten.** Die *Annotationen* von COCO stehen unter CC BY 4.0, die Labels sind also in Ordnung. Die *Bilder* kann COCO gar nicht lizenzieren: Es sind Flickr-Fotos unter einem Flickenteppich individueller Bedingungen, die das COCO-Konsortium ausdrücklich nicht besitzt. Der Großteil der Branche stuft das als geringes Risiko ein und geht weiter, aber „COCO ist in Ordnung“ ist nur eine Aussage über die Annotationen. Jenseits von COCO wird es schnell strenger: Viele Luftbild-, Medizin- und Fahrdatensätze sind nicht kommerziell, und die permissive Codelizenz eines Modells wäscht sie nicht rein.

Prüf alle drei Ebenen, jedes Mal.

## Eine Anmerkung zu MIT gegenüber Apache-2.0

Wir liefern eine MIT-lizenzierte Bibliothek aus, es wäre also bequem, dir zu sagen, MIT sei einfach die beste Lizenz. Der ehrliche Vergleich ist interessanter.

MIT ist nicht pflichtenfrei: Du musst den Copyright-Hinweis und den Lizenztext in Kopien beibehalten, die du verbreitest. Das ist eine echte Bedingung, nur eine günstige. Und Apache-2.0 hat etwas, das MIT nicht hat: eine **ausdrückliche Patentlizenz** jedes Contributors, dazu eine Vergeltungsklausel, die die Lizenz von jedem beendet, der wegen des Werks klagt. MIT schweigt zu Patenten. Für ein Unternehmen, dessen größte Sorge das Patentrisiko ist, ist Apache-2.0 wohl die *sicherere* der beiden, und ein Großteil des permissiven Ökosystems für Objekterkennung (YOLOX, RT-DETR, D-FINE, DEIM) steht genau aus diesem Grund unter Apache-2.0.

Was dir MIT bringt, ist Einfachheit und Kompatibilität. Keine der beiden Lizenzen wird dich jemals zwingen, deinen Quellcode zu veröffentlichen, und genau um diese Achse geht es in diesem Artikel eigentlich. Wenn dir jemand erzählt, MIT oder Apache sei die wichtige Entscheidung, will er dir etwas verkaufen. Die wichtige Entscheidung ist permissiv oder Copyleft.

## Welches YOLO kannst du also tatsächlich verwenden?

- **Kommerzielles Closed-Source-Produkt:** YOLOv7 oder YOLOv9 über das [MIT-Repo](https://github.com/MultimediaTechLab/YOLO) des Labors, YOLOX oder PP-YOLOE (aus PaddleDetection) unter den Originalen, oder ein gepflegtes MIT-Framework wie LibreYOLO, wenn du das nicht selbst zusammenstellen und prüfen willst. Meide alles unter GPL oder AGPL, es sei denn, du legst deine gesamte Anwendung offen oder kaufst die Enterprise License.
- **Open-Source-Projekt:** alles, solange deine Lizenz kompatibel ist. Wenn dein Projekt unter AGPL steht, passt die Ultralytics-Linie natürlich, und die neueste Forschung (YOLOv13, YOLO26, YOLOE) landet dort zuerst.
- **Forschung und Benchmarking:** Lizenzen schränken dich kaum ein. Nimm das, was das Paper verwendet hat, mit dem du dich vergleichst.
- **Der Plan „kümmern wir uns später drum“:** funktioniert nicht. Eine AGPL-Abhängigkeit aufzulösen, nachdem dein Produkt gebaut ist, heißt, alles neu zu trainieren, neu zu validieren und neu zu exportieren, einschließlich der Gewichte. Wähle zuerst das Repository und damit die Lizenz.

Einen Vergleich der Frameworks selbst statt ihrer Lizenzen findest du in [Die besten Ultralytics-Alternativen 2026](/articles/best-ultralytics-alternatives).

## Die MIT-Option: was LibreYOLO ausliefert

Offenlegung: LibreYOLO ist unser Projekt, und seine Lizenz ist der Grund, warum es diesen Artikel gibt.

**[LibreYOLO](https://github.com/LibreYOLO/libreyolo) ist eine einzige MIT-lizenzierte Codebasis**, die die unten aufgeführten Detektoren hinter einer einzigen API abdeckt, mit eingebautem Training und Export nach ONNX, TensorRT, OpenVINO und NCNN. Kein Copyleft, keine Netzwerkklausel, keine Enterprise-Stufe.

Hier ist das vollständige Aufgebot für die Objekterkennung, mit dem Upstream jeder Familie und dessen Lizenz, damit du genau siehst, woher jeder permissive Weg kommt:

| In LibreYOLO | Modell | Upstream | Upstream-Lizenz |
| --- | --- | --- | --- |
| `LibreYOLO2`, `LibreYOLO3`, `LibreYOLO4` | Die YOLOs der Darknet-Ära, in PyTorch | [pjreddie/darknet](https://github.com/pjreddie/darknet), [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) | Gemeinfrei |
| `LibreYOLO7` | YOLOv7 | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (die eigene Neuimplementierung des Labors, nicht das GPL-Repo) |
| `LibreYOLO9`, `LibreYOLO9E2E` | YOLOv9, dazu eine NMS-freie End-to-End-Variante | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (die eigene Neuimplementierung des Labors, nicht das GPL-Repo) |
| `LibreYOLO9P2` | YOLOv9 mit einem Stride-4-Head für kleine Objekte | LibreYOLO-Eigenentwicklung | MIT |
| `LibreYOLOX` | YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 |
| `LibreYOLONAS` | YOLO-NAS, Detektion und Pose | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) | Code unter Apache-2.0. Die eingeschränkten Upstream-Gewichte werden **nicht** weiterverbreitet |
| `LibreRTDETR`, `LibreRTDETRv2` | RT-DETR und v2 | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 |
| `LibreRTDETRv4` | RT-DETRv4 | [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | Apache-2.0 |
| `LibreRFDETR` | RF-DETR, Detektion, Segmentierung, Pose, OBB | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | Apache-2.0 für N/S/M/L. XL/2XL stehen unter der Platform Model License von Roboflow, daher liefern wir nur die Apache-Größen aus |
| `LibreDFINE` | D-FINE | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 |
| `LibreDEIM` | DEIM | [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | Apache-2.0 |
| `LibreDEIMv2` | DEIMv2 | [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | Code unter Apache-2.0. Die größeren Varianten nutzen Metas DINOv3-Backbone unter Metas eigener Nicht-OSI-Lizenz, die die Weiterverbreitung erlaubt, aber **militärische, nukleare, Spionage- und Waffennutzung untersagt**. Diese Größen werden als `other` veröffentlicht, nicht als Apache. Lies die Lizenz, bevor du irgendetwas mit Dual-Use-Bezug auslieferst |
| `LibreRTMDet` | RTMDet ([ohne MMDetection](/articles/rtmdet-without-mmdetection)) | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | Apache-2.0 (das Apache-Geschwisterprojekt, nicht das GPL-3.0-lizenzierte mmyolo) |
| `LibrePICODET` | PP-PicoDet | Architektur aus [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), Checkpoints über die Neuportierung [Picodet_Pytorch](https://github.com/Bo396543018/Picodet_Pytorch) | Apache-2.0 (beide) |
| `LibreEC` | EdgeCrafter, Detektion, Pose, Segmentierung | [Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) | Apache-2.0 |
| `LibreFOMO` | Zentroid-Detektor für Hardware der Mikrocontroller-Klasse | LibreYOLO-Eigenentwicklung | MIT |
| `LibreGroundingDINO` | Grounding DINO, Open-Vocabulary (Inferenz) | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 |
| `LibreOWLv2` | OWLv2, Open-Vocabulary (Inferenz) | Google, über `transformers` | Apache-2.0 |

Zwei Dinge, bei denen diese Spalte bewusst ehrlich ist.

**Kein GPL- oder AGPL-Quellcode wird in die Codebasis kopiert.** Jede Familie oben baut auf einem gemeinfreien, MIT- oder Apache-2.0-Upstream auf, und diese Regel hält das Framework MIT. Wo eine Architektur sowohl ein permissives als auch ein Copyleft-Zuhause hat, bauen wir auf dem permissiven auf: Unser YOLOv9 und YOLOv7 gehen auf das MIT-Repo des Labors zurück statt auf die GPL-Originale, und unser RTMDet auf das Apache-lizenzierte MMDetection statt auf das GPL-lizenzierte MMYOLO. YOLO-World fehlt in der Liste, obwohl regelmäßig danach gefragt wird, weil es unter GPL-3.0 steht und es keine permissive Quelle gibt, von der wir es übernehmen könnten.

**Permissiv ist nicht dasselbe wie uneingeschränkt, und wir sagen das lieber selbst, als dich es in einem Audit entdecken zu lassen.** Die größeren DEIMv2-Varianten erben Metas DINOv3-Bedingungen, einschließlich eines Verbots militärischer, nuklearer, Spionage- und Waffennutzung, was eine echte Einschränkung ist, wenn du auch nur in der Nähe von Dual-Use arbeitest. Eine Handvoll Research-Preview-Checkpoints ist auf nicht kommerziellen Datensätzen (DOTA, VisDrone) trainiert und auf Hugging Face als `cc-by-nc` gekennzeichnet, statt stillschweigend so ausgeliefert zu werden, als wären sie frei. Und dieselbe Vorsicht, die wir bei den Gewichten aller anderen anwenden, gilt auch für unsere: Unsere YOLO9-Checkpoints sind aus den eigenen Checkpoints des MIT-Repos des Labors konvertiert, sie erben also, was auch immer für diese gilt, und das ist, wie oben erwähnt, „behauptet, nicht formal geprüft“. Die MIT-Lizenz deckt den Code ab, den wir geschrieben haben. Jeder Checkpoint trägt die Bedingungen dessen, worauf und woraus er trainiert wurde, veröffentlicht pro Gewichtsdatei.

Über die Detektion hinaus deckt dieselbe API Instanz- und semantische Segmentierung, Pose, orientierte Boxen, Klassifikation (MobileNetV4, ConvNeXt, EfficientNetV2, ResNet, CLIP), monokulare Tiefenschätzung, Bildrestaurierung und Blickrichtungsschätzung ab.

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
```

Derselbe YOLO-Workflow, ohne die Lizenz-Hausaufgaben.

Gib ihm einen Stern auf GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Doku: [libreyolo.com/docs](https://libreyolo.com/docs)

---

*Marken und Zugehörigkeit: YOLO, YOLOv8, YOLO11, YOLO26 und Ultralytics sind Marken ihrer jeweiligen Inhaber, darunter Ultralytics Inc. LibreYOLO ist ein unabhängiges Projekt und ist weder mit Ultralytics noch mit einer anderen in diesem Artikel genannten Organisation verbunden, noch wird es von diesen gesponsert oder unterstützt. Produkt- und Repository-Namen werden nur verwendet, um die besprochene Software zu identifizieren und zu vergleichen.*

*Dieser Artikel enthält allgemeine Informationen, keine Rechtsberatung, Stand 11. Juli 2026, und wurde von einem der Anbieter im Vergleich geschrieben. Prüf die aktuelle Lizenz, bevor du auslieferst.*
