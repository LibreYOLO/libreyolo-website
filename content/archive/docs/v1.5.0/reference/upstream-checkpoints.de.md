---
title: Upstream-Checkpoints
seo_title: Upstream-Checkpoints in LibreYOLO laden
description: >-
  So wandelt die automatische Konvertierung einen veröffentlichten
  Upstream-Checkpoint in einen LibreYOLO-v1.0-Checkpoint um: entpackte Layouts,
  Familienerkennung und Grenzen.
lead: >-
  Die Modellfamilien von LibreYOLO sind Portierungen aus Upstream-Projekten.
  Deren veröffentlichte Checkpoints sind fast direkt ladbar, enthalten aber
  keine LibreYOLO-Metadaten. Die automatische Konvertierung erkennt diese
  Dateien, verpackt sie in Schema v1.0 und schreibt das Ergebnis neben die
  Quelle.
keywords:
  - libreyolo autoconvert
  - upstream checkpoint laden
  - convert_upstream_state_dict
  - upstream gewichte libreyolo
  - checkpoint konvertierung
last_verified: 1.5.0
verification: >-
  Verhalten aus libreyolo/models/autoconvert.py und
  BaseModel.convert_upstream_state_dict; Familienerkenner durch Lesen jeder
  Überschreibung von convert_upstream_state_dict geprüft, jeweils für v1.5.0.
  RF-DETR-COCO-Regeln aus docs/checkpoint_schema.md.
snippets:
  usage:
    - label: Datei direkt an die Factory übergeben
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Eine erkannte Upstream-Datei wird beim Laden konvertiert und der
        # konvertierte Checkpoint daneben geschrieben.
        # model = LibreYOLO("yolov9-t-converted.pt")

        # Jeder LibreYOLO-Checkpoint wird unverändert geladen.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.family, model.size, model.task, model.nb_classes)
source_hash: c6022771a2a207a1
---

## Vorgang beim Laden

Wenn `LibreYOLO()` eine `.pt`-Datei erhält, die noch kein vollständiger
v1.0-Checkpoint ist, ruft es den automatischen Konverter auf. Dieser:

1. entpackt das Tensor-Dictionary aus üblichen Upstream-Layouts;
2. fragt jede registrierte Familie, ob sie das Layout erkennt, und ordnet
   Schlüssel neu zu, wenn die Upstream-Benennung von der nativen Portierung
   abweicht;
3. verpackt den Treffer in einen strikten Metadaten-Checkpoint nach v1.0 und
   liest Größe, Aufgabe und Klassenzahl aus den Tensoren, damit nachtrainierte
   Checkpoints richtig konvertiert werden;
4. schreibt ihn neben die Quelle als `<source>-<Prefix><size>[-task].pt` und
   gibt diesen Pfad zurück, damit die Factory ihn normal lädt.

Der aufrufende Code muss nichts weiter tun. Wenn keine Familie eine Datei
beansprucht, gibt der Konverter nichts zurück und die Factory meldet, dass sie
die Datei nicht laden konnte.

<code-tabs name="usage" />

## Entpackte Layouts

Das Tensor-Dictionary wird in der folgenden Prioritätsreihenfolge gesucht,
beginnend mit EMA. Jeder Kandidat wird geprüft, bis einer tatsächlich Tensoren
enthält. Ein leerer oder nur aus Metadaten bestehender EMA-Block verdeckt daher
keine gültigen darunterliegenden Gewichte.

| Schlüssel | Hinweis |
|---|---|
| `ema.module` | Üblicher EMA-Wrapper |
| `ema` | Veraltete flache EMA-Wrapper, die Tensoren direkt speichern |
| `ema_state_dict` | Das Präfix `module.` wird von Einträgen entfernt |
| `params_ema` | |
| `params` | |
| `ema_net` | |
| `net` | |
| `model` | |
| `state_dict` | |
| Die Datei selbst | Ein einfaches State Dict |

Anschließend wird jeder Kandidat auf seine tensorwertigen Einträge beschränkt
und normalisiert. Ein vorangestelltes Präfix `module.` oder `_orig_mod.` wird
entfernt. Wenn alle Schlüssel eines Dictionaries mit `model.model.` beginnen,
wird auch dieses Präfix entfernt.

## Erkennungsregeln der Familien

Die Erkennung ist eine Klassenmethode pro Familie. Die Standardimplementierung
beansprucht ein Layout, dessen Schlüssel bereits zur nativen Portierung passen.
Eine Familie mit abweichender Upstream-Benennung überschreibt sie mit einer
Zuordnung und gibt für unbekannte Layouts nichts zurück.

Familien mit einem Erkenner zur Neuzuordnung: `centernet`, `deeplabv3`,
`deformable_detr`, `dexined`, `moge2`, `picodet`, `rtdetr`, `rtdetrv2`,
`rtdetrv4`, `rtmdet`, `segformer`, `swin`, `teed`, `yolo7`, `yolo9`,
`yolo9_e2e`, `yolo9_p2`.

Familien, die die automatische Konvertierung vollständig ablehnen:
`efficientdet`, `eomt` und `pidnet` geben im Erkenner nichts zurück. Ihre
Upstream-Dateien müssen stattdessen ein Konvertierungsskript durchlaufen.
`l2cs` ist vom generischen Erkenner ausgeschlossen, da es nur Inferenz
unterstützt und seine Gewichte Weitergabebeschränkungen unterliegen.

RF-DETR besitzt einen eigenen Erkenner, weil es zur Größenbestimmung den ganzen
Checkpoint statt nur des Tensor-Dictionaries und zur Neuzuordnung der
COCO-Klassen weitere Informationen benötigt. Er wird nur registriert, wenn
seine optionalen Abhängigkeiten installiert sind.

Jede andere registrierte Familie verwendet den Standard. Sie beansprucht die
Datei, wenn ihr eigener Loader die Schlüssel bereits erkennt.

## Auswahl der Familie

Mehrere Familien können dieselbe Datei beanspruchen. Die Auflösung entspricht
den Dispatch-Regeln der Factory.

Der Anspruch einer Unterklasse hat Vorrang vor dem der Basisklasse. Die
Registrierungsreihenfolge folgt der Klassenerstellung. Eine abgeleitete Familie
wird daher nach der von ihr verfeinerten Basis registriert. Ihre positiven
Merkmale dürfen nicht gegen den breiteren Passthrough der Basis verlieren.

Anschließend entscheidet die Registry-Reihenfolge, weil sie die Spezifität
codiert. Der früheste Anspruch ist die spezifischste Übereinstimmung.

Nur die Mehrdeutigkeit zwischen DEIM und D-FINE lässt sich nicht durch die
Registry-Reihenfolge auflösen, da ihre Architekturschlüssel identisch sind. In
diesem und ausschließlich diesem Fall entscheidet der Dateiname. Eine Datei
ohne Hinweis im Namen wird abgelehnt und nicht erraten. Der Dateiname wird
bewusst an keiner anderen Stelle berücksichtigt. Eine breite fälschlich
positive Übereinstimmung kann daher nie allein durch den Dateinamen gegenüber
einer spezifischeren bevorzugt werden.

## Sicheres Laden

Upstream-Dateien werden mit dem Weights-Only-Unpickler geladen. Einige
Upstream-Trainings-Checkpoints enthalten Bibliotheksobjekte, die dieser
Unpickler ablehnt. Diese Objekte sind Trainingsmetadaten und keine Gewichte.
Deshalb wird jeder blockierte globale Name mit einer inerten Platzhalterklasse
erneut versucht, die den Unpickler zufriedenstellt, ohne etwas auszuführen. Der
erfasste Name wird nur als Stringlabel verwendet und nie importiert, evaluiert
oder aufgerufen.

Sensible Modulnamen werden vollständig abgelehnt und nie durch Platzhalter
ersetzt: `builtins`, `os`, `sys`, `posix`, `nt` und `subprocess`. Die
Wiederholungsschleife ist auf 32 Versuche begrenzt. Eine Datei, die eine
unbegrenzte Reihe unterschiedlicher globaler Namen einführt, schlägt daher
sicher fehl, statt endlos zu laufen. Nur Tensoren gelangen in den konvertierten
Checkpoint.

## Speicherort der konvertierten Datei

Die Ausgabe wird neben der Quelle unter dem Namen
`<source>-<Prefix><size>[-task].pt` geschrieben. Sie wird stets neu geschrieben
und nicht wiederverwendet. Wiederholtes Laden derselben Quelle bleibt dadurch
aktuell. Zugleich werden Kollisionen mit offiziellen Gewichten oder einem
anderen Fine-Tuning derselben Familie, Größe und Aufgabe im selben Verzeichnis
vermieden.

Wenn das Quellverzeichnis schreibgeschützt ist, verwendet die Konvertierung ein
neues privates temporäres Verzeichnis pro Aufruf. Die Protokollzeile nennt den
verwendeten Pfad. Nur wenn auch dies fehlschlägt, wird die Konvertierung mit
einer Warnung verworfen.

## Vorhandene LibreYOLO-Checkpoints

Eine Datei mit einem LibreYOLO-spezifischen Marker, `libreyolo_version` oder
`model_family`, gehört zum normalen Ladepfad und wird nicht erneut konvertiert.
Das Überspringen gilt nur für einen Passthrough-Anspruch, bei dem die
Schlüsselmenge unverändert blieb. Wenn eine Konvertierung die Schlüsselmenge
verändert hat, belegt dies ein fremdes Upstream-Layout. Ein solcher Anspruch
wird auch bei einer markierten Datei akzeptiert.

`schema_version` gilt bewusst nicht als Marker, da andere Trainings- und
Exportwerkzeuge diesen generischen Namen verwenden. Auch `names`, `nc`,
`size`, `task` und `imgsz` sind keine Marker, weil ein Upstream-Fine-Tuning sie
ebenfalls enthalten kann. Ein fremdes Fine-Tuning mit lediglich einem
generischen `names`-Schlüssel ist daher nicht markiert. Sein Anspruch mit
nativen Schlüsseln wird normal konvertiert und leitet die Klassenzahl aus dem
Tensor-Head ab, statt fälschlich als 80-Klassen-Modell geladen zu werden.

## Vom Konverter gelesene Metadaten

Klassennamen werden aus einem Schlüssel `names` auf oberster Ebene oder aus
`class_names` in einem Block `args` beziehungsweise `hyper_parameters`
übernommen. Eine Namenszuordnung, deren Schlüssel Labels statt Klassenindizes
sind, ist unbrauchbar und wird durch generierte Standardwerte ersetzt. Eine
Namensliste, die länger als die erkannte Klassenzahl ist, wird gekürzt.
Außerhalb des Bereichs liegende Indizes würden sonst beim strikten Validator
scheitern und die Konvertierung unbemerkt abbrechen.

Upstream-`args` werden als einfache Metadaten übernommen. Jeder Wert, der kein
String, keine Zahl, kein boolescher Wert, keine Liste und kein Dictionary ist,
wird verworfen. So gelangt nichts Unsicheres in die gespeicherte Datei.

## COCO-Normalisierung von RF-DETR

Upstream-Checkpoints von RF-DETR besitzen einen Klassifikations-Head mit 91
Ausgaben, also die 90 COCO-Klassen plus Hintergrund. Die automatische
Konvertierung normalisiert einen COCO-RF-DETR auf die COCO-80-Konvention. Die
Neuzuordnung wird in der Nachverarbeitung angewendet.

Ein Checkpoint gilt als COCO, wenn er genau 80 Namen enthält, eine Klassenzahl
von 80 deklariert, einen `coco`-Datensatzhinweis besitzt oder überhaupt keine
Klassen- oder Datensatzmetadaten enthält. Der letzte Fall ist wichtig: Ein
einfaches Upstream-State-Dict ist der kanonische COCO-vortrainierte Checkpoint
und der einzige verteilte RF-DETR mit 91 Ausgaben ohne Metadaten.

Ein echtes benutzerdefiniertes RF-DETR mit 90 Klassen bleibt als solches
erhalten. Es wird anhand einer Namensliste, einer expliziten von 80 abweichenden
Klassenzahl oder eines Nicht-COCO-Datensatzhinweises erkannt. Der Fallback für
einen einfachen Checkpoint greift daher nicht. Leere Platzhalter werden bei der
Entscheidung, ob ein Datensatzhinweis vorhanden ist, ignoriert.

## Grenzen

Die automatische Konvertierung erkennt veröffentlichte Upstream-Layouts. Sie
schreibt keine Architektur um und macht ein nicht portiertes Modell nicht
ladbar. Wenn keine Familie eine Datei beansprucht, benötigst du ein
Konvertierungsskript statt eines Factory-Arguments. Das Repository enthält
`weights/convert_*.py` für Familien, die ein solches Skript benötigen,
darunter EoMT, PIDNet und EfficientDet.

Die Konvertierung erfindet auch keine Metadaten, die sie nicht lesen kann.
Größe, Aufgabe und Klassenzahl stammen aus den Tensoren. Namen stammen aus der
Datei, sofern vorhanden, und werden andernfalls als `class_i` generiert.

