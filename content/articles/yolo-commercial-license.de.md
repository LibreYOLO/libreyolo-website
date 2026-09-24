---
title: "YOLO kostenlos kommerziell nutzen? Die Lizenzen von YOLOv8, YOLO11 und YOLO26"
description: "YOLOv5, YOLOv8, YOLO11 und YOLO26 stehen unter AGPL-3.0 und sind daher nicht kostenlos für die kommerzielle Closed-Source-Nutzung. Hier steht, was die Lizenz tatsächlich verlangt, Version für Version, und welche MIT-lizenzierte Alternative es gibt."
date: 2026-07-04
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, commercial-use, yolov8, yolo11, yolo26, mit-license]
faq:
  - q: "Ist YOLOv8 kostenlos für die kommerzielle Nutzung?"
    a: "Nicht für Closed-Source-Produkte. YOLOv8 steht unter AGPL-3.0. Du musst also entweder den Quellcode deiner gesamten Anwendung unter AGPL-3.0 veröffentlichen oder die kostenpflichtige Enterprise License des Anbieters kaufen. Für uneingeschränkte, kostenlose kommerzielle Nutzung verwendest du eine MIT-lizenzierte Bibliothek wie LibreYOLO."
  - q: "Unter welcher Lizenz steht YOLO?"
    a: "Die gängigen Releases YOLOv5, YOLOv8, YOLO11 und YOLO26 stehen standardmäßig unter AGPL-3.0. AGPL-3.0 ist eine Lizenz mit starkem Copyleft: Sie ist Open Source, aber nicht kostenlos in einem proprietären Produkt nutzbar."
  - q: "Kann ich YOLO in einem kommerziellen Closed-Source-Produkt verwenden?"
    a: "Unter AGPL-3.0 nicht, ohne die Lizenz einzuhalten, und das heißt: dein gesamtes Produkt unter AGPL-3.0 als Open Source veröffentlichen. Um deinen Code privat zu halten, kaufst du entweder die Enterprise License des Anbieters oder wechselst zu einem permissiv lizenzierten Framework wie LibreYOLO (MIT)."
  - q: "Gibt es ein YOLO ohne AGPL?"
    a: "Ja. LibreYOLO steht unter der MIT-Lizenz, es gibt also kein AGPL-Copyleft und keine Netzwerkklausel. Du kannst es frei und ohne Gebühr in kommerzieller und Closed-Source-Software einsetzen."
  - q: "Was kostet eine kommerzielle YOLO-Lizenz?"
    a: "Die Enterprise License des Anbieters ist eine kostenpflichtige Vereinbarung pro Unternehmen mit nicht öffentlichen Preisen. LibreYOLO kostet nichts: Die MIT-Lizenz erlaubt die kommerzielle Nutzung kostenlos."
  - q: "Umgehe ich die AGPL, wenn ich YOLO nur auf meinen eigenen Servern betreibe?"
    a: "Nein. Abschnitt 13 der AGPL-3.0 erfasst die Interaktion über ein Netzwerk. Wenn Nutzer deine modifizierte Version über ein Netzwerk erreichen, musst du ihnen den korrespondierenden Quellcode anbieten. Der Betrieb ausschließlich auf deinen eigenen Servern ist keine Ausnahme."
---

**Nein. Die populären YOLO-Modelle YOLOv5, YOLOv8, YOLO11 und das neue YOLO26 sind nicht kostenlos für die kommerzielle Closed-Source-Nutzung.** Sie stehen unter **AGPL-3.0**, einer Lizenz mit starkem Copyleft. Wenn du ein Produkt auf einem davon aufbaust, musst du deine *gesamte* Anwendung unter AGPL-3.0 als Open Source veröffentlichen oder die kostenpflichtige Enterprise License des Anbieters kaufen. Wenn beides für dich nicht infrage kommt, gibt es eine permissiv lizenzierte Alternative: **[LibreYOLO](/) steht unter der MIT-Lizenz und ist kostenlos für jede kommerzielle Nutzung, ohne Wenn und Aber.**

Wenn du nach „ist YOLOv8 kostenlos kommerziell nutzbar“, „YOLO Lizenz“ oder „YOLO ohne AGPL“ gesucht hast, beantwortet diese Seite die Frage präzise, Version für Version.

## YOLO-Lizenzen im Überblick

| Modell | Standardlizenz | Kostenlos für kommerzielle **Closed-Source**-Nutzung? | Was du tun musst |
| --- | --- | --- | --- |
| YOLOv5 | AGPL-3.0 | Nein | Deine ganze App als Open Source veröffentlichen oder eine Enterprise License kaufen |
| YOLOv8 | AGPL-3.0 | Nein | Deine ganze App als Open Source veröffentlichen oder eine Enterprise License kaufen |
| YOLO11 | AGPL-3.0 | Nein | Deine ganze App als Open Source veröffentlichen oder eine Enterprise License kaufen |
| YOLO26 | AGPL-3.0 | Nein | Deine ganze App als Open Source veröffentlichen oder eine Enterprise License kaufen |
| **LibreYOLO** | **MIT** | **Ja** | **Nichts. Kostenlos in proprietären Produkten ausliefern.** |

Kurz gesagt: Die gängigen YOLO-Modelle sind Open Source, aber unter **AGPL-3.0**, einer Copyleft-Lizenz, die die meisten Unternehmen in einem proprietären Produkt nicht einhalten können. LibreYOLO bietet dir denselben YOLO-Workflow unter der **MIT-Lizenz**, die dir keine solche Pflicht auferlegt.

## Unter welcher Lizenz steht YOLO?

Die weit verbreiteten YOLO-Modelle, **YOLOv5, YOLOv8, YOLO11 und das neueste YOLO26**, werden von ihrem Anbieter unter der **GNU Affero General Public License v3.0 (AGPL-3.0)** vertrieben.

AGPL-3.0 ist eine Lizenz mit *starkem Copyleft*. Sie gehört nicht zur permissiven „mach, was du willst“-Sorte von Open Source wie MIT oder Apache-2.0. Sie bringt eine bestimmte, weitreichende Pflicht mit sich, und genau über diese Pflicht stolpern kommerzielle Nutzer.

## Ist YOLOv8 kostenlos für die kommerzielle Nutzung?

Nicht in dem Sinn, den die meisten meinen. Du *kannst* YOLOv8 herunterladen und kommerziell nutzen, aber nur, wenn du die AGPL-3.0 einhältst. In der Praxis heißt das:

- Wenn du ein Produkt, das YOLOv8 enthält, verbreitest, auslieferst oder hostest, musst du **den vollständigen Quellcode dieses Produkts** unter AGPL-3.0 veröffentlichen: deine App, deinen Integrationscode, deine Skripte und deine Konfiguration.
- Ausgelöst wird diese Pflicht nicht nur, wenn du Software an Nutzer auslieferst, sondern auch, wenn du Nutzer **über ein Netzwerk** damit interagieren lässt, die prägende Klausel in Abschnitt 13 der AGPL. Ein SaaS- oder API-Backend entkommt ihr nicht.

Für ein Hobbyprojekt oder akademische Forschung ist das meist kein Problem. Für ein proprietäres kommerzielles Produkt kommt es fast nie infrage, die gesamte Codebasis zu veröffentlichen. Alles hier gilt genauso für **YOLO11** und **YOLO26**: dieselbe Lizenz, dieselbe Pflicht.

## Und YOLOv5, YOLO11 und YOLO26?

Gleiches Bild. Alle stehen standardmäßig unter **AGPL-3.0**. Die Enterprise License des Anbieters umfasst ausdrücklich „YOLO26, earlier YOLO versions, and any future YOLO models“ (YOLO26, frühere YOLO-Versionen und alle künftigen YOLO-Modelle), was zeigt, dass die Lizenzierung über die gesamte Familie bewusst und einheitlich ist. Eine neuere Version bringt dir keine freundlichere Lizenz.

Der Vollständigkeit halber: Nicht *jedes* Modell mit „YOLO“ im Namen steht unter AGPL. Einige Community-Varianten stehen unter GPL-3.0 oder Apache-2.0, und die Lizenzen unterscheiden sich je nach Autor. Aber die Releases, die Leute meinen, wenn sie „YOLOv8“ oder „YOLO11“ eintippen, stehen unter AGPL-3.0.

## Was AGPL-3.0 tatsächlich verlangt, in einfachen Worten

AGPL-3.0 erweitert die GPL um einen entscheidenden Zusatz: die **Netzwerkklausel**. Die praktische Checkliste:

1. **Den Quellcode bereitstellen.** Wenn du Software weitergibst, die AGPL-Code enthält, musst du den *vollständigen korrespondierenden Quellcode* des gesamten kombinierten Werks unter AGPL-3.0 zugänglich machen.
2. **Netzwerknutzung zählt.** Wenn Nutzer über ein Netzwerk mit deiner modifizierten Version interagieren, etwa über eine Web-App, eine API oder ein SaaS, musst du ihnen denselben Quellcode anbieten. Ein Schlupfloch nach dem Motto „wir betreiben es nur auf unseren Servern“ gibt es nicht.
3. **Copyleft ist viral.** Die Pflicht erfasst das *gesamte* abgeleitete Werk, nicht nur die YOLO-Dateien. Deine Geschäftslogik gerät in den Geltungsbereich der AGPL.

Das sind allgemeine Informationen, keine Rechtsberatung. Sprich für deinen konkreten Fall mit einem Anwalt. Die Kernaussage ist aber einfach: **AGPL-3.0 und kommerzielle Closed-Source-Software passen nicht zusammen.**

## Die Option Enterprise License und ihr Haken

Der Anbieter bietet eine kostenpflichtige **Enterprise License** an, die die AGPL-Pflichten aufhebt. Damit kannst du YOLO in ein proprietäres Produkt einbetten, ohne irgendetwas als Open Source zu veröffentlichen. Das ist ein legitimer Weg, und für manche Unternehmen ist es der richtige.

Die Haken:

- **Sie kostet Geld**: eine laufende kommerzielle Gebühr, pro Unternehmen verhandelt, mit nicht öffentlichen Preisen.
- **Sie macht dich von den Bedingungen eines einzelnen Anbieters abhängig**, die sich ändern können und nur den Code dieses Anbieters abdecken.
- **Du zahlst für die Erlaubnis**, eine Modellarchitektur zu nutzen, die im Kern veröffentlichte Forschung ist.

Wenn eine wiederkehrende Gebühr und Vendor-Lock-in für dich akzeptabel sind, funktioniert die Enterprise License. Wenn du lieber nicht zahlen oder die Pflicht gar nicht erst tragen willst, lies weiter.

## Die MIT-Alternative: LibreYOLO

Offenlegung: LibreYOLO ist unser Projekt. Es ist auch der Grund, warum diese Seite mit einer klaren Antwort enden kann statt mit einem Schulterzucken.

**LibreYOLO ist ein YOLO-Framework, das unter der permissiven MIT-Lizenz veröffentlicht ist.** Diese eine Tatsache ändert bei der kommerziellen Nutzung alles:

- **Kostenlos für die kommerzielle Nutzung**, auch in proprietären Closed-Source-Produkten.
- **Kein Copyleft, keine Netzwerkklausel.** Du musst deine App nie als Open Source veröffentlichen.
- **Keine Gebühr pro Arbeitsplatz und keine Enterprise-Gebühr.** MIT heißt MIT.
- **Derselbe Workflow, den du schon kennst**, daher ist die Migration unkompliziert.

LibreYOLO ist ein echtes, gepflegtes Framework, kein Wrapper: Objekterkennung, Segmentierung, Pose, Klassifikation, Tiefe und mehr unter einer vertrauten API, mit eingebautem Training und Export nach ONNX/TensorRT/OpenVINO/NCNN. Der Unterschied liegt nicht im Workflow. Der Unterschied ist die Lizenz.

Wenn du von der allgemeineren Frage „weg vom AGPL-YOLO“ hierhergekommen bist: Einen ausführlicheren Vergleich des Ökosystems haben wir in [Die besten Ultralytics-Alternativen 2026](/articles/best-ultralytics-alternatives) geschrieben.

Die Lizenz jedes anderen YOLO, von den gemeinfreien Darknet-Originalen über YOLOv9, YOLOv10, YOLOv12 und YOLOv13 bis YOLO26, findest du in [YOLO-Lizenzen erklärt](/articles/yolo-licenses-explained).

### Warum MIT für ein Unternehmen wichtig ist

Die MIT-Lizenz erlaubt dir, Software auf Basis von LibreYOLO zu nutzen, zu verändern, einzubetten und zu verkaufen, **ohne Pflicht, deinen Quellcode offenzulegen**, und **ohne Gebühr**. Sie steckt gerade deshalb hinter einem großen Teil des modernen Software-Stacks, weil sie für den kommerziellen Einsatz sicher ist. Dein Produkt gehört dir; du schuldest niemandem etwas.

## Ausprobieren

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

Du behältst die YOLO-Entwicklungserfahrung: auf deinen Daten trainieren, vorhersagen und für die Auslieferung exportieren. Du lässt die AGPL-Pflicht und die Enterprise-Rechnung hinter dir.

LibreYOLO steht unter der MIT-Lizenz, läuft unter Linux, Mac und Windows und funktioniert ohne Codeänderung auf GPU, Apple Silicon und reiner CPU. Eine API umfasst YOLO9, RF-DETR, RTMDet, YOLOX, D-FINE, die RT-DETR-Reihe, Segmentierung, Pose, Tiefe und mehr, alles permissiv lizenziert.

Gib ihm einen Stern auf GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Doku: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
